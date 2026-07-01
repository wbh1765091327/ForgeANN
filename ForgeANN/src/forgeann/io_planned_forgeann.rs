use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::json;

use super::io_runtime::VectorWindowCacheStats;
use super::point_pipeline::PointPipelineStats;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StorageState {
    RawIds,
    VectorRun,
    Resident,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NodeAction {
    PrefetchStreaming,
    ExternalStreaming,
    HotVectorRun,
    ExactFallback,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct IoPlanConfig {
    pub(crate) enabled: bool,
    pub(crate) dry_run: bool,
    pub(crate) allow_full_build: bool,
    pub(crate) min_depth: usize,
    pub(crate) vector_run_enable: bool,
    pub(crate) resident_enable: bool,
    pub(crate) prefetch_streaming_enable: bool,
    pub(crate) small_run_wave_enable: bool,
    pub(crate) child_run_batching_enable: bool,
    pub(crate) temp_budget_bytes: usize,
    pub(crate) window_cache_bytes: usize,
    pub(crate) min_resident_points: usize,
    pub(crate) max_resident_points: usize,
    pub(crate) max_avg_read_size_bytes: usize,
    pub(crate) min_consumer_wait_ratio: f64,
    pub(crate) min_saved_wait_ms: f64,
    pub(crate) min_saved_wait_per_gib_ms: f64,
    pub(crate) verify: bool,
}

impl IoPlanConfig {
    pub(crate) fn disabled() -> Self {
        Self {
            enabled: false,
            dry_run: true,
            allow_full_build: false,
            min_depth: 2,
            vector_run_enable: false,
            resident_enable: false,
            prefetch_streaming_enable: false,
            small_run_wave_enable: false,
            child_run_batching_enable: false,
            temp_budget_bytes: 0,
            window_cache_bytes: 0,
            min_resident_points: 32_768,
            max_resident_points: 0,
            max_avg_read_size_bytes: 16 * 1024,
            min_consumer_wait_ratio: 0.5,
            min_saved_wait_ms: 50.0,
            min_saved_wait_per_gib_ms: 1_000.0,
            verify: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_default() -> Self {
        Self {
            enabled: true,
            dry_run: true,
            allow_full_build: false,
            min_depth: 2,
            vector_run_enable: true,
            resident_enable: true,
            prefetch_streaming_enable: true,
            small_run_wave_enable: true,
            child_run_batching_enable: true,
            temp_budget_bytes: 1 << 30,
            window_cache_bytes: 0,
            min_resident_points: 1,
            max_resident_points: 0,
            max_avg_read_size_bytes: 16 * 1024,
            min_consumer_wait_ratio: 0.5,
            min_saved_wait_ms: 50.0,
            min_saved_wait_per_gib_ms: 1_000.0,
            verify: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct IoPainSample {
    pub(crate) raw_read_calls: u64,
    pub(crate) logical_bytes: u64,
    pub(crate) physical_bytes: u64,
    pub(crate) consumer_wait_ms: f64,
    pub(crate) producer_wait_ms: f64,
    pub(crate) io_wall_ms: f64,
    pub(crate) compute_wall_ms: f64,
    pub(crate) observed_direct_read_latency_ms: f64,
    pub(crate) blocked_worker_wait_ms: f64,
}

impl IoPainSample {
    pub(crate) fn estimated(point_count: usize, row_bytes: usize) -> Self {
        let logical_bytes = point_count.saturating_mul(row_bytes) as u64;
        Self {
            raw_read_calls: point_count as u64,
            logical_bytes,
            physical_bytes: logical_bytes,
            consumer_wait_ms: point_count as f64 * 0.002,
            producer_wait_ms: 0.0,
            io_wall_ms: point_count as f64 * 0.0025,
            compute_wall_ms: 0.0,
            observed_direct_read_latency_ms: 0.002,
            blocked_worker_wait_ms: 0.0,
        }
    }
}

/// Global atomic budget state for io-planned vector-run temp bytes.
///
/// **Why this exists:** `plan_node` is called per-node from parallel workers.
/// A simple read-modify-write of a counter would race and over-subscribe.
/// This struct uses atomic CAS to reserve temp bytes before execution,
/// preventing the 143 GB blowup observed on wiki35m when `temp_used_bytes`
/// was hardcoded to 0.
#[derive(Debug)]
pub(crate) struct IoPlanBudgetState {
    temp_budget_bytes: u64,
    temp_reserved_bytes: AtomicU64,
    /// Tracks actual bytes written to disk (recorded after materialization).
    temp_written_bytes: AtomicU64,
}

impl IoPlanBudgetState {
    pub(crate) fn new(temp_budget_bytes: u64) -> Self {
        Self {
            temp_budget_bytes,
            temp_reserved_bytes: AtomicU64::new(0),
            temp_written_bytes: AtomicU64::new(0),
        }
    }

    /// Atomically reserve `bytes` of temp budget.
    /// Returns `true` if reservation succeeded (CAS loop), `false` if budget exceeded.
    pub(crate) fn try_reserve_temp(&self, bytes: u64) -> bool {
        if bytes == 0 {
            return true;
        }
        let mut current = self.temp_reserved_bytes.load(Ordering::Acquire);
        loop {
            let next = match current.checked_add(bytes) {
                Some(n) => n,
                None => return false,
            };
            if next > self.temp_budget_bytes {
                return false;
            }
            match self.temp_reserved_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    /// Release previously reserved temp bytes (e.g. on fallback/cleanup).
    pub(crate) fn release_temp(&self, bytes: u64) {
        self.temp_reserved_bytes.fetch_sub(bytes, Ordering::AcqRel);
    }

    /// Record actual bytes written after materialization.
    pub(crate) fn record_temp_written(&self, bytes: u64) {
        self.temp_written_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Current total reserved bytes.
    pub(crate) fn temp_reserved(&self) -> u64 {
        self.temp_reserved_bytes.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NodeCost {
    pub(crate) action: NodeAction,
    pub(crate) allowed: bool,
    pub(crate) estimated_raw_read_calls: u64,
    pub(crate) estimated_raw_read_bytes: u64,
    pub(crate) temp_bytes: usize,
    pub(crate) resident_bytes: usize,
    pub(crate) estimated_benefit: f64,
    pub(crate) gate: String,
}

impl NodeCost {
    fn external(point_count: usize, row_bytes: usize) -> Self {
        Self {
            action: NodeAction::ExternalStreaming,
            allowed: true,
            estimated_raw_read_calls: point_count as u64,
            estimated_raw_read_bytes: point_count.saturating_mul(row_bytes) as u64,
            temp_bytes: 0,
            resident_bytes: 0,
            estimated_benefit: 0.0,
            gate: "allowed".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NodeExecutionPlan {
    pub(crate) storage_state: StorageState,
    pub(crate) selected: NodeAction,
    pub(crate) external_streaming: NodeCost,
    pub(crate) prefetch_streaming: NodeCost,
    pub(crate) hot_vector_run: NodeCost,
    pub(crate) exact_fallback: NodeCost,
    pub(crate) depth: usize,
    pub(crate) point_count: usize,
    pub(crate) fanout: usize,
    pub(crate) pain_observed: bool,
    pub(crate) estimated_speedup: f64,
    pub(crate) benefit_per_temp_byte: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct IoCostModel {
    pub(crate) vector_run_read_amortization: f64,
}

impl IoCostModel {
    const MIN_CALL_SAVINGS_RATIO: f64 = 0.5;

    pub(crate) fn plan_node(
        &self,
        config: IoPlanConfig,
        state: StorageState,
        depth: usize,
        point_count: usize,
        fanout: usize,
        row_bytes: usize,
        resident_used_bytes: usize,
        temp_used_bytes: usize,
    ) -> NodeExecutionPlan {
        self.plan_node_with_pain_observed(
            config,
            state,
            depth,
            point_count,
            fanout,
            row_bytes,
            resident_used_bytes,
            temp_used_bytes,
            IoPainSample::estimated(point_count, row_bytes),
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_node_with_pain(
        &self,
        config: IoPlanConfig,
        state: StorageState,
        depth: usize,
        point_count: usize,
        fanout: usize,
        row_bytes: usize,
        resident_used_bytes: usize,
        temp_used_bytes: usize,
        pain: IoPainSample,
    ) -> NodeExecutionPlan {
        self.plan_node_with_pain_observed(
            config,
            state,
            depth,
            point_count,
            fanout,
            row_bytes,
            resident_used_bytes,
            temp_used_bytes,
            pain,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_node_with_pain_observed(
        &self,
        config: IoPlanConfig,
        state: StorageState,
        depth: usize,
        point_count: usize,
        fanout: usize,
        row_bytes: usize,
        _resident_used_bytes: usize,
        temp_used_bytes: usize,
        pain: IoPainSample,
        _pain_observed: bool,
    ) -> NodeExecutionPlan {
        let fanout = fanout.max(1);
        let external = NodeCost::external(point_count, row_bytes);
        let before_calls = external.estimated_raw_read_calls.max(1);
        let before_bytes = external.estimated_raw_read_bytes.max(1);

        let prefetch_calls = estimate_prefetch_read_calls(point_count, row_bytes, &config, pain);
        let prefetch_bytes = estimate_prefetch_physical_bytes(point_count, row_bytes, pain);
        let prefetch_benefit = estimate_saved_wait_ms(before_calls, prefetch_calls, pain) * 0.50;
        let prefetch_streaming = NodeCost {
            action: NodeAction::PrefetchStreaming,
            allowed: config.enabled
                && config.prefetch_streaming_enable
                && state != StorageState::Resident,
            estimated_raw_read_calls: prefetch_calls,
            estimated_raw_read_bytes: prefetch_bytes,
            temp_bytes: 0,
            resident_bytes: 0,
            estimated_benefit: prefetch_benefit,
            gate: String::new(),
        }
        .with_gate();

        let temp_bytes = point_count.saturating_mul(row_bytes);
        let vector_after_calls = fanout as u64;
        let vector_after_bytes = temp_bytes as u64;
        let vector_benefit = estimate_saved_wait_ms(before_calls, vector_after_calls, pain)
            * self.vector_run_factor();

        let vector_run_depth_allowed = depth >= config.min_depth || fanout == 1;
        let vector_run_density_ok = if temp_bytes > 0 && point_count > 0 {
            let calls_saved = before_calls.saturating_sub(vector_after_calls) as f64;
            calls_saved / point_count as f64 >= Self::MIN_CALL_SAVINGS_RATIO
        } else {
            false
        };

        let hot_vector_run = NodeCost {
            action: NodeAction::HotVectorRun,
            allowed: config.enabled
                && config.vector_run_enable
                && state == StorageState::RawIds
                && vector_run_depth_allowed
                && temp_bytes > 0
                && temp_used_bytes.saturating_add(temp_bytes) <= config.temp_budget_bytes
                && vector_run_density_ok,
            estimated_raw_read_calls: vector_after_calls,
            estimated_raw_read_bytes: vector_after_bytes,
            temp_bytes,
            resident_bytes: 0,
            estimated_benefit: vector_benefit,
            gate: String::new(),
        }
        .with_gate();

        let exact_fallback = NodeCost {
            action: NodeAction::ExactFallback,
            allowed: false,
            estimated_raw_read_calls: before_calls,
            estimated_raw_read_bytes: before_bytes,
            temp_bytes: 0,
            resident_bytes: 0,
            estimated_benefit: 0.0,
            gate: "blocked".to_string(),
        };

        let selected = [
            hot_vector_run.clone(),
            prefetch_streaming.clone(),
            exact_fallback.clone(),
            external.clone(),
        ]
        .into_iter()
        .filter(|cost| cost.allowed)
        .max_by(|left, right| {
            left.estimated_benefit
                .partial_cmp(&right.estimated_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|cost| cost.estimated_benefit > 0.0)
        .map(|cost| cost.action)
        .unwrap_or(NodeAction::ExternalStreaming);

        let selected_cost = match selected {
            NodeAction::ExternalStreaming => &external,
            NodeAction::PrefetchStreaming => &prefetch_streaming,
            NodeAction::HotVectorRun => &hot_vector_run,
            NodeAction::ExactFallback => &exact_fallback,
        };
        let after_calls = selected_cost.estimated_raw_read_calls.max(1);
        NodeExecutionPlan {
            storage_state: state,
            selected,
            external_streaming: external,
            prefetch_streaming,
            hot_vector_run,
            exact_fallback,
            depth,
            point_count,
            fanout,
            pain_observed: _pain_observed,
            estimated_speedup: before_calls as f64 / after_calls as f64,
            benefit_per_temp_byte: if temp_bytes == 0 {
                0.0
            } else {
                vector_benefit / temp_bytes as f64
            },
        }
    }

    fn vector_run_factor(self) -> f64 {
        if self.vector_run_read_amortization > 0.0 {
            self.vector_run_read_amortization
        } else {
            1.0
        }
    }
}

impl NodeCost {
    fn with_gate(mut self) -> Self {
        if self.gate.is_empty() {
            self.gate = if self.allowed { "allowed" } else { "blocked" }.to_string();
        }
        self
    }
}

fn estimate_prefetch_read_calls(
    point_count: usize,
    row_bytes: usize,
    config: &IoPlanConfig,
    pain: IoPainSample,
) -> u64 {
    if point_count == 0 {
        return 0;
    }
    let target_window_bytes = if config.max_avg_read_size_bytes == 0 {
        64 * 1024
    } else {
        config
            .max_avg_read_size_bytes
            .saturating_mul(4)
            .max(64 * 1024)
    };
    let rows_per_window = (target_window_bytes / row_bytes.max(1)).max(1);
    let physical_rows =
        (pain.physical_bytes as usize).saturating_add(row_bytes.max(1) - 1) / row_bytes.max(1);
    physical_rows.div_ceil(rows_per_window).max(1) as u64
}

fn estimate_prefetch_physical_bytes(
    point_count: usize,
    row_bytes: usize,
    pain: IoPainSample,
) -> u64 {
    let logical = point_count.saturating_mul(row_bytes) as u64;
    pain.physical_bytes.max(logical)
}

fn estimate_saved_wait_ms(before_calls: u64, after_calls: u64, pain: IoPainSample) -> f64 {
    let calls_saved = before_calls.saturating_sub(after_calls) as f64;
    let latency_ms = if pain.observed_direct_read_latency_ms.is_finite()
        && pain.observed_direct_read_latency_ms > 0.0
    {
        pain.observed_direct_read_latency_ms
    } else {
        0.002
    };
    let call_wait = calls_saved * latency_ms;
    let consumer_share = pain.consumer_wait_ms.max(0.0) * 0.75;
    let blocked = pain.blocked_worker_wait_ms.max(0.0);
    call_wait.max(consumer_share).saturating_add(blocked)
}

trait SaturatingAddF64 {
    fn saturating_add(self, rhs: Self) -> Self;
}

impl SaturatingAddF64 for f64 {
    fn saturating_add(self, rhs: Self) -> Self {
        let sum = self + rhs;
        if sum.is_finite() { sum } else { f64::MAX }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct DepthIoPainStats {
    pub(crate) depth: usize,
    pub(crate) point_gather_windows: usize,
    pub(crate) point_gather_batches: usize,
    pub(crate) logical_bytes: u64,
    pub(crate) physical_bytes: u64,
    pub(crate) io_ms: f64,
    pub(crate) consumer_wait_ms: f64,
    pub(crate) producer_wait_ms: f64,
    pub(crate) prefetch_used_peak_bytes: usize,
}

impl DepthIoPainStats {
    fn add_io_pain(
        &mut self,
        point_gather_windows: usize,
        point_gather_batches: usize,
        logical_bytes: u64,
        physical_bytes: u64,
        io_ms: f64,
        consumer_wait_ms: f64,
        producer_wait_ms: f64,
        prefetch_used_peak_bytes: usize,
    ) {
        self.point_gather_windows += point_gather_windows;
        self.point_gather_batches += point_gather_batches;
        self.logical_bytes = self.logical_bytes.saturating_add(logical_bytes);
        self.physical_bytes = self.physical_bytes.saturating_add(physical_bytes);
        self.io_ms += io_ms;
        self.consumer_wait_ms += consumer_wait_ms;
        self.producer_wait_ms += producer_wait_ms;
        self.prefetch_used_peak_bytes = self.prefetch_used_peak_bytes.max(prefetch_used_peak_bytes);
    }

    fn avg_read_size(&self) -> f64 {
        if self.point_gather_windows == 0 {
            0.0
        } else {
            self.physical_bytes as f64 / self.point_gather_windows as f64
        }
    }

    fn read_amplification(&self) -> f64 {
        if self.logical_bytes == 0 {
            0.0
        } else {
            self.physical_bytes as f64 / self.logical_bytes as f64
        }
    }

    fn consumer_wait_over_io(&self) -> f64 {
        if self.io_ms <= f64::EPSILON {
            0.0
        } else {
            self.consumer_wait_ms / self.io_ms
        }
    }

    fn to_pain_sample(&self) -> Option<IoPainSample> {
        if self.point_gather_windows == 0
            || self.logical_bytes == 0
            || self.physical_bytes == 0
            || self.io_ms <= f64::EPSILON
        {
            return None;
        }
        Some(IoPainSample {
            raw_read_calls: self.point_gather_windows as u64,
            logical_bytes: self.logical_bytes,
            physical_bytes: self.physical_bytes,
            consumer_wait_ms: self.consumer_wait_ms,
            producer_wait_ms: self.producer_wait_ms,
            io_wall_ms: self.io_ms,
            compute_wall_ms: 0.0,
            observed_direct_read_latency_ms: self.io_ms / self.point_gather_windows.max(1) as f64,
            blocked_worker_wait_ms: 0.0,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct IoPlannerStats {
    pub(crate) enabled: bool,
    pub(crate) dry_run: bool,
    pub(crate) verify: bool,
    pub(crate) planned_nodes: usize,
    pub(crate) selected_external: usize,
    pub(crate) selected_prefetch_streaming: usize,
    pub(crate) selected_vector_run: usize,
    pub(crate) selected_exact_fallback: usize,
    pub(crate) executed_external: usize,
    pub(crate) executed_prefetch_streaming: usize,
    pub(crate) executed_vector_run: usize,
    pub(crate) executed_exact_fallback: usize,
    pub(crate) vector_to_external_fallbacks: usize,
    pub(crate) estimated_before_raw_read_calls: u64,
    pub(crate) estimated_after_raw_read_calls: u64,
    pub(crate) estimated_before_raw_read_bytes: u64,
    pub(crate) estimated_after_raw_read_bytes: u64,
    pub(crate) planned_temp_bytes: usize,
    pub(crate) materialized_temp_bytes: usize,
    pub(crate) best_benefit_per_temp_byte: f64,
    pub(crate) vector_run_materialize_wall: Duration,
    pub(crate) child_extent_logical_extents: usize,
    pub(crate) child_extent_coalesced_reads: usize,
    pub(crate) child_extent_logical_bytes: u64,
    pub(crate) child_extent_physical_bytes: u64,
    pub(crate) child_extent_header_read_savings: usize,
    pub(crate) small_run_waves: usize,
    pub(crate) small_run_wave_runs: usize,
    pub(crate) small_run_wave_points: usize,
    pub(crate) window_cache_budget_bytes: usize,
    pub(crate) window_cache: VectorWindowCacheStats,
    pub(crate) io_pain_by_depth: BTreeMap<usize, DepthIoPainStats>,
}

impl IoPlannerStats {
    pub(crate) fn record_config(&mut self, config: IoPlanConfig) {
        self.enabled = true;
        self.dry_run = config.dry_run;
        self.verify = config.verify;
        self.window_cache_budget_bytes = self
            .window_cache_budget_bytes
            .max(config.window_cache_bytes);
    }

    pub(crate) fn record_plan(&mut self, plan: &NodeExecutionPlan, config: IoPlanConfig) {
        self.record_config(config);
        self.planned_nodes += 1;
        self.io_pain_by_depth
            .entry(plan.depth)
            .or_insert_with(|| DepthIoPainStats {
                depth: plan.depth,
                ..DepthIoPainStats::default()
            });
        match plan.selected {
            NodeAction::ExternalStreaming => self.selected_external += 1,
            NodeAction::PrefetchStreaming => self.selected_prefetch_streaming += 1,
            NodeAction::HotVectorRun => self.selected_vector_run += 1,
            NodeAction::ExactFallback => self.selected_exact_fallback += 1,
        }
        self.estimated_before_raw_read_calls += plan.external_streaming.estimated_raw_read_calls;
        let selected_cost = match plan.selected {
            NodeAction::ExternalStreaming => &plan.external_streaming,
            NodeAction::PrefetchStreaming => &plan.prefetch_streaming,
            NodeAction::HotVectorRun => &plan.hot_vector_run,
            NodeAction::ExactFallback => &plan.exact_fallback,
        };
        self.estimated_after_raw_read_calls += selected_cost.estimated_raw_read_calls;
        self.estimated_before_raw_read_bytes += plan.external_streaming.estimated_raw_read_bytes;
        self.estimated_after_raw_read_bytes += selected_cost.estimated_raw_read_bytes;
        self.planned_temp_bytes = self
            .planned_temp_bytes
            .saturating_add(selected_cost.temp_bytes);
        self.best_benefit_per_temp_byte = self
            .best_benefit_per_temp_byte
            .max(plan.benefit_per_temp_byte);
    }

    pub(crate) fn record_external_execution(&mut self) {
        self.enabled = true;
        self.executed_external += 1;
    }

    pub(crate) fn record_prefetch_streaming_execution(&mut self) {
        self.enabled = true;
        self.executed_prefetch_streaming += 1;
    }

    pub(crate) fn record_vector_run_execution(&mut self, temp_bytes: usize, wall: Duration) {
        self.enabled = true;
        self.executed_vector_run += 1;
        self.materialized_temp_bytes = self.materialized_temp_bytes.saturating_add(temp_bytes);
        self.vector_run_materialize_wall += wall;
    }

    pub(crate) fn record_exact_fallback_execution(&mut self) {
        self.enabled = true;
        self.executed_exact_fallback += 1;
    }

    pub(crate) fn record_child_extent_batching(
        &mut self,
        logical_extents: usize,
        coalesced_reads: usize,
        logical_bytes: u64,
        physical_bytes: u64,
        header_read_savings: usize,
    ) {
        self.enabled = true;
        self.child_extent_logical_extents += logical_extents;
        self.child_extent_coalesced_reads += coalesced_reads;
        self.child_extent_logical_bytes = self
            .child_extent_logical_bytes
            .saturating_add(logical_bytes);
        self.child_extent_physical_bytes = self
            .child_extent_physical_bytes
            .saturating_add(physical_bytes);
        self.child_extent_header_read_savings += header_read_savings;
    }

    pub(crate) fn record_small_run_wave(&mut self, runs: usize, points: usize) {
        self.enabled = true;
        self.small_run_waves += 1;
        self.small_run_wave_runs += runs;
        self.small_run_wave_points += points;
    }

    pub(crate) fn record_vector_to_external_fallback(&mut self) {
        self.enabled = true;
        self.vector_to_external_fallbacks += 1;
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.enabled |= other.enabled;
        self.dry_run |= other.dry_run;
        self.verify |= other.verify;
        self.planned_nodes += other.planned_nodes;
        self.selected_external += other.selected_external;
        self.selected_prefetch_streaming += other.selected_prefetch_streaming;
        self.selected_vector_run += other.selected_vector_run;
        self.selected_exact_fallback += other.selected_exact_fallback;
        self.executed_external += other.executed_external;
        self.executed_prefetch_streaming += other.executed_prefetch_streaming;
        self.executed_vector_run += other.executed_vector_run;
        self.executed_exact_fallback += other.executed_exact_fallback;
        self.vector_to_external_fallbacks += other.vector_to_external_fallbacks;
        self.estimated_before_raw_read_calls += other.estimated_before_raw_read_calls;
        self.estimated_after_raw_read_calls += other.estimated_after_raw_read_calls;
        self.estimated_before_raw_read_bytes += other.estimated_before_raw_read_bytes;
        self.estimated_after_raw_read_bytes += other.estimated_after_raw_read_bytes;
        self.planned_temp_bytes = self
            .planned_temp_bytes
            .saturating_add(other.planned_temp_bytes);
        self.materialized_temp_bytes = self
            .materialized_temp_bytes
            .saturating_add(other.materialized_temp_bytes);
        self.best_benefit_per_temp_byte = self
            .best_benefit_per_temp_byte
            .max(other.best_benefit_per_temp_byte);
        self.vector_run_materialize_wall += other.vector_run_materialize_wall;
        self.child_extent_logical_extents += other.child_extent_logical_extents;
        self.child_extent_coalesced_reads += other.child_extent_coalesced_reads;
        self.child_extent_logical_bytes = self
            .child_extent_logical_bytes
            .saturating_add(other.child_extent_logical_bytes);
        self.child_extent_physical_bytes = self
            .child_extent_physical_bytes
            .saturating_add(other.child_extent_physical_bytes);
        self.child_extent_header_read_savings += other.child_extent_header_read_savings;
        self.small_run_waves += other.small_run_waves;
        self.small_run_wave_runs += other.small_run_wave_runs;
        self.small_run_wave_points += other.small_run_wave_points;
        self.window_cache_budget_bytes = self
            .window_cache_budget_bytes
            .max(other.window_cache_budget_bytes);
        self.window_cache.merge(&other.window_cache);
        for (depth, other_depth) in other.io_pain_by_depth {
            let entry = self
                .io_pain_by_depth
                .entry(depth)
                .or_insert_with(|| DepthIoPainStats {
                    depth,
                    ..DepthIoPainStats::default()
                });
            entry.point_gather_windows += other_depth.point_gather_windows;
            entry.point_gather_batches += other_depth.point_gather_batches;
            entry.logical_bytes = entry
                .logical_bytes
                .saturating_add(other_depth.logical_bytes);
            entry.physical_bytes = entry
                .physical_bytes
                .saturating_add(other_depth.physical_bytes);
            entry.io_ms += other_depth.io_ms;
            entry.consumer_wait_ms += other_depth.consumer_wait_ms;
            entry.producer_wait_ms += other_depth.producer_wait_ms;
            entry.prefetch_used_peak_bytes = entry
                .prefetch_used_peak_bytes
                .max(other_depth.prefetch_used_peak_bytes);
        }
    }

    pub(crate) fn record_depth_point_pipeline(&mut self, depth: usize, stats: &PointPipelineStats) {
        if stats.batches == 0 && stats.planned_windows == 0 {
            return;
        }
        self.enabled = true;
        self.record_depth_io_pain(
            depth,
            stats.planned_windows,
            stats.batches,
            stats.logical_bytes,
            stats.physical_bytes,
            stats.read_wall.as_secs_f64() * 1000.0,
            stats.consumer_wait.as_secs_f64() * 1000.0,
            stats.producer_wait.as_secs_f64() * 1000.0,
            stats.permit_peak_bytes,
        );
        self.window_cache.merge(&stats.window_cache);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_depth_io_pain(
        &mut self,
        depth: usize,
        point_gather_windows: usize,
        point_gather_batches: usize,
        logical_bytes: u64,
        physical_bytes: u64,
        io_ms: f64,
        consumer_wait_ms: f64,
        producer_wait_ms: f64,
        prefetch_used_peak_bytes: usize,
    ) {
        if point_gather_windows == 0 && point_gather_batches == 0 && logical_bytes == 0 {
            return;
        }
        self.enabled = true;
        let entry = self
            .io_pain_by_depth
            .entry(depth)
            .or_insert_with(|| DepthIoPainStats {
                depth,
                ..DepthIoPainStats::default()
            });
        entry.add_io_pain(
            point_gather_windows,
            point_gather_batches,
            logical_bytes,
            physical_bytes,
            io_ms,
            consumer_wait_ms,
            producer_wait_ms,
            prefetch_used_peak_bytes,
        );
    }

    pub(crate) fn pain_sample_for_depth(&self, depth: usize) -> Option<IoPainSample> {
        self.io_pain_by_depth
            .range(..=depth)
            .rev()
            .find_map(|(_, stats)| stats.to_pain_sample())
    }

    pub(crate) fn merge_depth_io_pain_from(&mut self, other: &Self) {
        for (depth, other_depth) in &other.io_pain_by_depth {
            let entry = self
                .io_pain_by_depth
                .entry(*depth)
                .or_insert_with(|| DepthIoPainStats {
                    depth: *depth,
                    ..DepthIoPainStats::default()
                });
            // The shared assignment context records the same stage samples that
            // child stats may also carry. Treat it as a profile snapshot and use
            // the larger aggregate per depth instead of adding it twice.
            entry.point_gather_windows = entry
                .point_gather_windows
                .max(other_depth.point_gather_windows);
            entry.point_gather_batches = entry
                .point_gather_batches
                .max(other_depth.point_gather_batches);
            entry.logical_bytes = entry.logical_bytes.max(other_depth.logical_bytes);
            entry.physical_bytes = entry.physical_bytes.max(other_depth.physical_bytes);
            entry.io_ms = entry.io_ms.max(other_depth.io_ms);
            entry.consumer_wait_ms = entry.consumer_wait_ms.max(other_depth.consumer_wait_ms);
            entry.producer_wait_ms = entry.producer_wait_ms.max(other_depth.producer_wait_ms);
            entry.prefetch_used_peak_bytes = entry
                .prefetch_used_peak_bytes
                .max(other_depth.prefetch_used_peak_bytes);
        }
    }

    pub(crate) fn profile_json(&self) -> String {
        if !self.enabled {
            return "null".to_string();
        }
        let speedup = if self.estimated_after_raw_read_calls == 0 {
            1.0
        } else {
            self.estimated_before_raw_read_calls as f64
                / self.estimated_after_raw_read_calls.max(1) as f64
        };
        let io_pain_by_depth = self
            .io_pain_by_depth
            .values()
            .map(|entry| {
                json!({
                    "depth": entry.depth,
                    "point_gather_windows": entry.point_gather_windows,
                    "point_gather_batches": entry.point_gather_batches,
                    "logical_bytes": entry.logical_bytes,
                    "physical_bytes": entry.physical_bytes,
                    "avg_read_size": entry.avg_read_size(),
                    "read_amplification": entry.read_amplification(),
                    "io_ms": entry.io_ms,
                    "consumer_wait_ms": entry.consumer_wait_ms,
                    "producer_wait_ms": entry.producer_wait_ms,
                    "consumer_wait_over_io": entry.consumer_wait_over_io(),
                    "prefetch_used_peak_bytes": entry.prefetch_used_peak_bytes,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "verify": self.verify,
            "planned_nodes": self.planned_nodes,
            "action_counts": {
                "selected": {
                    "prefetch_streaming": self.selected_prefetch_streaming,
                    "external_streaming": self.selected_external,
                    "hot_vector_run": self.selected_vector_run,
                    "exact_fallback": self.selected_exact_fallback,
                },
                "executed": {
                    "prefetch_streaming": self.executed_prefetch_streaming,
                    "external_streaming": self.executed_external,
                    "hot_vector_run": self.executed_vector_run,
                    "exact_fallback": self.executed_exact_fallback,
                }
            },
            "selected": {
                "external_streaming": self.selected_external,
                "prefetch_streaming": self.selected_prefetch_streaming,
                "hot_vector_run": self.selected_vector_run,
                "exact_fallback": self.selected_exact_fallback,
            },
            "executed": {
                "external_streaming": self.executed_external,
                "prefetch_streaming": self.executed_prefetch_streaming,
                "hot_vector_run": self.executed_vector_run,
                "exact_fallback": self.executed_exact_fallback,
            },
            "fallbacks": {
                "hot_vector_run_to_external": self.vector_to_external_fallbacks,
            },
            "estimated_before_raw_read_calls": self.estimated_before_raw_read_calls,
            "estimated_after_raw_read_calls": self.estimated_after_raw_read_calls,
            "estimated_before_raw_read_bytes": self.estimated_before_raw_read_bytes,
            "estimated_after_raw_read_bytes": self.estimated_after_raw_read_bytes,
            "estimated_speedup": speedup,
            "temp_bytes": {
                "planned": self.planned_temp_bytes,
                "materialized": self.materialized_temp_bytes,
            },
            "benefit_density": {
                "temp": self.best_benefit_per_temp_byte,
            },
            "vector_run_materialize_wall_ms": self.vector_run_materialize_wall.as_millis(),
            "global_point_runtime": {
                "executed_prefetch_streaming": self.executed_prefetch_streaming,
            },
            "estimated_before_raw_read_bytes": self.estimated_before_raw_read_bytes,
            "estimated_after_raw_read_bytes": self.estimated_after_raw_read_bytes,
            "estimated_speedup": speedup,
            "temp_bytes": {
                "planned": self.planned_temp_bytes,
                "materialized": self.materialized_temp_bytes,
            },
            "benefit_density": {
                "temp": self.best_benefit_per_temp_byte,
            },
            "vector_run_materialize_wall_ms": self.vector_run_materialize_wall.as_millis(),
            "global_point_runtime": {
                "executed_prefetch_streaming": self.executed_prefetch_streaming,
            },
            "vector_window_cache": {
                "budget_bytes": self.window_cache_budget_bytes,
                "hits": self.window_cache.hits,
                "misses": self.window_cache.misses,
                "inserts": self.window_cache.inserts,
                "evictions": self.window_cache.evictions,
                "used_peak_bytes": self.window_cache.used_peak_bytes,
                "saved_direct_read_calls": self.window_cache.saved_direct_read_calls,
                "logical_bytes": self.window_cache.logical_bytes,
                "physical_bytes": self.window_cache.physical_bytes,
            },
            "small_run_wave": {
                "enabled": self.small_run_waves > 0,
                "waves": self.small_run_waves,
                "runs": self.small_run_wave_runs,
                "points": self.small_run_wave_points,
            },
            "child_extent_batching": {
                "enabled": self.child_extent_logical_extents > 0,
                "logical_extents": self.child_extent_logical_extents,
                "coalesced_reads": self.child_extent_coalesced_reads,
                "logical_bytes": self.child_extent_logical_bytes,
                "physical_bytes": self.child_extent_physical_bytes,
                "header_read_savings": self.child_extent_header_read_savings,
            },
            "io_pain_by_depth": io_pain_by_depth,
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{IoCostModel, IoPainSample, IoPlanConfig, NodeAction, StorageState};
    use crate::forgeann::point_pipeline::PointPipelineStats;

    #[test]
    fn io_cost_model_selects_external_and_vector_run_cases() {
        let model = IoCostModel::default();
        let external = model.plan_node(
            IoPlanConfig {
                vector_run_enable: true,
                resident_enable: false,
                prefetch_streaming_enable: false,
                min_depth: 2,
                ..IoPlanConfig::test_default()
            },
            StorageState::RawIds,
            1,
            20_000,
            8,
            512,
            0,
            0,
        );
        assert_eq!(external.selected, NodeAction::ExternalStreaming);

        let vector_run = model.plan_node(
            IoPlanConfig {
                vector_run_enable: true,
                resident_enable: false,
                min_depth: 1,
                temp_budget_bytes: 2 << 30,
                ..IoPlanConfig::test_default()
            },
            StorageState::RawIds,
            2,
            200_000,
            1,
            512,
            0,
            0,
        );
        assert_eq!(vector_run.selected, NodeAction::HotVectorRun);
    }

    #[test]
    fn depth_pain_sample_uses_nearest_prior_depth_when_exact_depth_is_missing() {
        let mut stats = super::IoPlannerStats::default();
        stats.record_depth_io_pain(1, 128, 4, 128 * 3_072, 128 * 3_072, 64.0, 60.0, 0.0, 0);

        assert!(stats.pain_sample_for_depth(2).is_some());
        assert!(stats.pain_sample_for_depth(0).is_none());
    }

    #[test]
    fn planner_profile_emits_depth_pain_and_window_cache() {
        let mut stats = super::IoPlannerStats::default();
        let pain = IoPainSample {
            raw_read_calls: 8,
            logical_bytes: 8 * 128,
            physical_bytes: 8 * 128,
            consumer_wait_ms: 20.0,
            producer_wait_ms: 2.0,
            io_wall_ms: 30.0,
            compute_wall_ms: 40.0,
            observed_direct_read_latency_ms: 0.2,
            blocked_worker_wait_ms: 1.0,
        };
        let plan = IoCostModel::default().plan_node_with_pain(
            IoPlanConfig {
                min_depth: 3,
                prefetch_streaming_enable: true,
                window_cache_bytes: 256 * 1024 * 1024,
                ..IoPlanConfig::test_default()
            },
            StorageState::RawIds,
            2,
            8,
            2,
            128,
            0,
            0,
            pain,
        );
        let point_pipeline = PointPipelineStats {
            batches: 2,
            points: 8,
            requested_rows: 8,
            physical_rows: 8,
            planned_windows: 4,
            logical_bytes: 8 * 128,
            physical_bytes: 8 * 128,
            ..PointPipelineStats::default()
        };

        stats.record_plan(
            &plan,
            IoPlanConfig {
                window_cache_bytes: 256 * 1024 * 1024,
                ..IoPlanConfig::test_default()
            },
        );
        stats.record_depth_point_pipeline(2, &point_pipeline);

        let json: serde_json::Value = serde_json::from_str(&stats.profile_json()).unwrap();
        let pain_by_depth = json["io_pain_by_depth"].as_array().unwrap();
        assert_eq!(pain_by_depth.len(), 1);
        assert_eq!(pain_by_depth[0]["depth"].as_u64(), Some(2));
        assert_eq!(pain_by_depth[0]["point_gather_windows"].as_u64(), Some(4));
        assert_eq!(pain_by_depth[0]["point_gather_batches"].as_u64(), Some(2));
        assert_eq!(
            json["vector_window_cache"]["budget_bytes"].as_u64(),
            Some(256 * 1024 * 1024)
        );
    }
}
