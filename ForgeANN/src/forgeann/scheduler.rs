use std::cell::RefCell;
use std::cmp;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crossbeam_queue::ArrayQueue;
use indicatif::ProgressBar;
use parking_lot::Mutex;
use rayon::ScopeFifo;
use rayon::prelude::*;

use super::adsampling::LeafAdsOperatorRuntime;
use super::hash_prune::SketchAccessor;
use super::leaf_build::{
    LeafProfile, LeafScratch, PendingEdge, PendingEdgeSink, PendingLuneWitness,
    process_leaf_parallel_large_profiled_with_sink_with_ads_runtime,
    process_leaf_profiled_with_sink_with_ads_runtime,
};
use super::params::ForgeANNParams;
use super::point_pipeline::{PointPipelineConfig, hydrate_resident_subset_with_stats};
use super::point_store::PointStore;
use super::spine_overlay_prune::{SpineOverlayEdgeRecorder, SpineOverlayTaggedEdgeSink};
use super::view_lune_prune::{ViewLuneEdgeRecorder, ViewLuneTaggedEdgeSink};
use crate::common::{AnnError, AnnResult, Metric};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SchedulerBudget {
    pub worker_count: usize,
    pub leaf_backlog_capacity: usize,
    pub producer_leaf_backlog_soft_limit: usize,
    pub large_leaf_min_size: usize,
    pub leaf_block_rows: usize,
}

impl SchedulerBudget {
    pub(super) const DEFAULT_LEAF_BLOCK_ROWS: usize = 1024;
    const MAX_LEAF_BACKLOG_CAPACITY: usize = 1_048_576;
    const MIN_LEAF_BACKLOG_CAPACITY: usize = 1;
    const TARGET_LEAF_BACKLOG_BYTES: usize = 12 * 1024 * 1024 * 1024;

    #[inline]
    pub(super) fn for_worker_count(worker_count: usize, kernel_safe_leaf_size: usize) -> Self {
        Self::for_memory_budget(worker_count, kernel_safe_leaf_size, usize::MAX)
    }

    #[inline]
    pub(super) fn for_memory_budget(
        worker_count: usize,
        kernel_safe_leaf_size: usize,
        memory_budget_bytes: usize,
    ) -> Self {
        let worker_count = worker_count.max(1);
        let estimated_leaf_bytes = kernel_safe_leaf_size
            .max(1)
            .saturating_mul(std::mem::size_of::<u32>())
            .max(1);
        let budget_backlog_capacity = if memory_budget_bytes == usize::MAX {
            Self::MAX_LEAF_BACKLOG_CAPACITY
        } else {
            (memory_budget_bytes.min(Self::TARGET_LEAF_BACKLOG_BYTES) / estimated_leaf_bytes).clamp(
                Self::MIN_LEAF_BACKLOG_CAPACITY,
                Self::MAX_LEAF_BACKLOG_CAPACITY,
            )
        };
        // D1 materialization can emit hundreds of thousands of small leaves before child
        // processing. Keep enough backlog headroom for that burst so the producer can
        // continue into later pipeline stages instead of repeatedly bouncing into inline
        // drain/help.
        let worker_floor = worker_count.saturating_mul(2048).clamp(
            Self::MIN_LEAF_BACKLOG_CAPACITY,
            Self::MAX_LEAF_BACKLOG_CAPACITY,
        );
        let leaf_backlog_capacity = if memory_budget_bytes == usize::MAX {
            worker_floor.min(budget_backlog_capacity)
        } else {
            budget_backlog_capacity
        };
        let large_leaf_min_size = cmp::max(4096, kernel_safe_leaf_size.saturating_mul(2));

        Self {
            worker_count,
            leaf_backlog_capacity,
            producer_leaf_backlog_soft_limit: 0,
            large_leaf_min_size,
            leaf_block_rows: Self::DEFAULT_LEAF_BLOCK_ROWS,
        }
    }
}

pub(crate) trait LeafEmitter: Sync {
    fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()>;

    fn emit_leaf_deferred(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.emit_leaf(leaf)
    }

    fn emit_leaf_from_dataset(
        &self,
        _dataset: &dyn PointStore,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        Ok(Some(leaf))
    }

    fn emit_leaf_inline_from_dataset(
        &self,
        dataset: &dyn PointStore,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        self.emit_leaf_from_dataset(dataset, leaf)
    }

    fn emit_leaf_inline_from_dataset_with_sketches(
        &self,
        dataset: &dyn PointStore,
        _sketches: &dyn SketchAccessor,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        self.emit_leaf_inline_from_dataset(dataset, leaf)
    }

    fn emit_leaf_batch_inline_from_dataset(
        &self,
        dataset: &dyn PointStore,
        leaves: Vec<Vec<u32>>,
    ) -> AnnResult<()> {
        for leaf in leaves {
            if let Some(leaf) = self.emit_leaf_inline_from_dataset(dataset, leaf)? {
                self.emit_leaf(leaf)?;
            }
        }
        Ok(())
    }

    fn wait_for_leaf_backlog_below(&self, _target_backlog: usize) -> AnnResult<bool> {
        Ok(false)
    }

    fn scheduler_signals(&self) -> Option<&dyn SchedulerSignals> {
        None
    }

    fn scheduler_telemetry(&self) -> Option<SchedulerTelemetry> {
        None
    }

    fn leaf_profile_snapshot(&self) -> Option<LeafProfile> {
        None
    }

    fn sketch_accessor(&self) -> Option<&dyn SketchAccessor> {
        None
    }
}

impl LeafEmitter for crossbeam::channel::Sender<Vec<u32>> {
    fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.send(leaf).map_err(|err| {
            AnnError::log_index_error(format!("Failed to emit ForgeANN leaf: {err}"))
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SchedulerTelemetry {
    pub peak_inflight_leaf_tasks: usize,
    pub peak_leaf_backlog: usize,
    pub leaf_backlog: usize,
    pub producer_leaf_backlog_soft_limit: usize,
    pub outstanding_leaf_tasks: usize,
    pub inflight_leaf_tasks: usize,
    pub active_leaf_drainers: usize,
    pub leaf_drainer_limit: usize,
    pub leaf_drainer_backpressure_limit: usize,
    pub inline_leaf_fallbacks: usize,
    pub producer_help_drains: usize,
    pub producer_help_drain_ms: u64,
    pub backlog_full_help_drains: usize,
    pub backlog_full_help_ms: u64,
    pub producer_help_yields: usize,
    pub large_leaf_count: usize,
    pub large_leaf_block_tasks: usize,
    pub active_large_assignment_peak: usize,
    pub large_assignment_guard_count: usize,
    pub leaf_cap_large_assignment_hits: usize,
    pub leaf_cap_producer_hits: usize,
    pub leaf_cap_done_hits: usize,
    pub batched_leaf_drains: usize,
    pub batched_leaf_leaves: usize,
    pub batched_leaf_points: usize,
    pub batched_leaf_unique_points: usize,
    pub batched_leaf_load_ms: u64,
    pub leaf_batch_hydration_limit: usize,
    pub leaf_batch_post_producer_scale: usize,
    pub peak_leaf_batch_leaves: usize,
    pub peak_leaf_batch_points: usize,
    pub peak_leaf_batch_unique_points: usize,
    pub active_leaf_batch_hydrations: usize,
    pub peak_leaf_batch_hydrations: usize,
    pub leaf_batch_hydration_wait_ms: u64,
    pub work_graph_enabled: bool,
    pub queued_work_ms_peak: u64,
    pub queued_mem_bytes_peak: usize,
    pub replay_admitted_runs: usize,
    pub replay_paused_ms: u64,
    pub worker_busy_ms: u64,
    pub worker_idle_ms: u64,
}

fn duration_nanos_u64(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn nanos_to_millis_u64(nanos: u64) -> u64 {
    nanos / 1_000_000
}

fn update_atomic_max_u64(max_value: &AtomicU64, value: u64) {
    let mut current = max_value.load(Ordering::Relaxed);
    while value > current {
        match max_value.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn update_atomic_max_usize(max_value: &AtomicUsize, value: usize) {
    let mut current = max_value.load(Ordering::Relaxed);
    while value > current {
        match max_value.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn stable_leaf_route_family(leaf: &[u32]) -> u32 {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    let first = leaf.first().copied().unwrap_or_default() as u64;
    let mid = leaf.get(leaf.len() / 2).copied().unwrap_or_default() as u64;
    let last = leaf.last().copied().unwrap_or_default() as u64;
    let seed = (leaf.len() as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
        ^ first.rotate_left(7)
        ^ mid.rotate_left(17)
        ^ last.rotate_left(29);
    ((mix(seed) & u32::MAX as u64) as u32).max(1)
}

pub(crate) trait SchedulerSignals: Sync {
    fn begin_large_assignment(&self);
    fn finish_large_assignment(&self);
    fn begin_leaf_drainer_limit(&self, limit: usize, backpressure_limit: usize);
    fn finish_leaf_drainer_limit(&self, limit: usize, backpressure_limit: usize);
}

pub(crate) struct LargeAssignmentGuard<'a> {
    signals: &'a dyn SchedulerSignals,
}

impl<'a> LargeAssignmentGuard<'a> {
    pub(crate) fn enter(signals: &'a dyn SchedulerSignals) -> Self {
        signals.begin_large_assignment();
        Self { signals }
    }
}

impl Drop for LargeAssignmentGuard<'_> {
    fn drop(&mut self) {
        self.signals.finish_large_assignment();
    }
}

pub(crate) struct LeafDrainerLimitGuard<'a> {
    signals: &'a dyn SchedulerSignals,
    limit: usize,
    backpressure_limit: usize,
}

impl<'a> LeafDrainerLimitGuard<'a> {
    #[cfg(test)]
    pub fn enter(signals: &'a dyn SchedulerSignals, limit: usize) -> Self {
        Self::enter_with_backpressure_limit(signals, limit, limit)
    }

    pub(crate) fn enter_with_backpressure_limit(
        signals: &'a dyn SchedulerSignals,
        limit: usize,
        backpressure_limit: usize,
    ) -> Self {
        let limit = limit.max(1);
        let backpressure_limit = backpressure_limit.max(1);
        signals.begin_leaf_drainer_limit(limit, backpressure_limit);
        Self {
            signals,
            limit,
            backpressure_limit,
        }
    }
}

impl Drop for LeafDrainerLimitGuard<'_> {
    fn drop(&mut self) {
        self.signals
            .finish_leaf_drainer_limit(self.limit, self.backpressure_limit);
    }
}

#[cfg(test)]
pub trait SchedulerTestObserver: Sync {
    fn on_leaf_start(&self, leaf: &[u32]);
}

#[derive(Clone, Copy)]
pub(super) struct LeafTaskContext<'a> {
    pub dataset: &'a dyn PointStore,
    pub metric: Metric,
    pub sketches: &'a dyn SketchAccessor,
    pub params: &'a ForgeANNParams,
    pub edge_sink: &'a dyn PendingEdgeSink,
    pub spine_overlay_recorder: Option<&'a dyn SpineOverlayEdgeRecorder>,
    pub view_lune_recorder: Option<&'a dyn ViewLuneEdgeRecorder>,
    pub point_pipeline_config: Option<&'a PointPipelineConfig>,
    pub detailed_profiling: bool,
    #[cfg(test)]
    pub observer: Option<&'a dyn SchedulerTestObserver>,
}

pub(super) struct ForgeannScheduler<'a> {
    budget: SchedulerBudget,
    context: LeafTaskContext<'a>,
    leaf_ads_runtime: LeafAdsOperatorRuntime,
    leaf_batch_drain_policy: LeafBatchDrainPolicy,
    progress: &'a ProgressBar,
    leaf_backlog: ArrayQueue<QueuedLeaf>,
    outstanding_leaf_tasks: AtomicUsize,
    inflight_leaf_tasks: AtomicUsize,
    peak_inflight_leaf_tasks: AtomicUsize,
    peak_leaf_backlog: AtomicUsize,
    active_leaf_drainers: AtomicUsize,
    retiring_leaf_drainers: AtomicUsize,
    leaf_drainer_limits: Mutex<Vec<LeafDrainerLimitEntry>>,
    leaf_drainer_limit: AtomicUsize,
    leaf_drainer_backpressure_limit: AtomicUsize,
    active_large_assignment_count: AtomicUsize,
    active_large_assignment_peak: AtomicUsize,
    producer_active: AtomicBool,
    inline_leaf_fallbacks: AtomicUsize,
    producer_help_drains: AtomicUsize,
    producer_help_drain_ns: AtomicU64,
    backlog_full_help_drains: AtomicUsize,
    backlog_full_help_ns: AtomicU64,
    producer_help_yields: AtomicUsize,
    large_leaf_count: AtomicUsize,
    large_leaf_block_tasks: AtomicUsize,
    large_assignment_guard_count: AtomicUsize,
    leaf_cap_large_assignment_hits: AtomicUsize,
    leaf_cap_producer_hits: AtomicUsize,
    leaf_cap_done_hits: AtomicUsize,
    batched_leaf_drains: AtomicUsize,
    batched_leaf_leaves: AtomicUsize,
    batched_leaf_points: AtomicUsize,
    batched_leaf_unique_points: AtomicUsize,
    batched_leaf_load_ns: AtomicU64,
    peak_leaf_batch_leaves: AtomicUsize,
    peak_leaf_batch_points: AtomicUsize,
    peak_leaf_batch_unique_points: AtomicUsize,
    active_leaf_batch_hydrations: AtomicUsize,
    peak_leaf_batch_hydrations: AtomicUsize,
    leaf_batch_hydration_wait_ns: AtomicU64,
    queued_work_ms: AtomicU64,
    queued_work_ms_peak: AtomicU64,
    queued_mem_bytes: AtomicUsize,
    queued_mem_bytes_peak: AtomicUsize,
    replay_admitted_runs: AtomicUsize,
    replay_paused_ns: AtomicU64,
    worker_busy_ns: AtomicU64,
    worker_idle_ns: AtomicU64,
    first_error: Mutex<Option<String>>,
    leaf_profile: Mutex<LeafProfile>,
}

struct QueuedLeaf {
    leaf: Vec<u32>,
    dataset: Option<Arc<dyn PointStore>>,
    estimated_work_ms: u64,
    estimated_mem_bytes: usize,
}

impl std::fmt::Debug for QueuedLeaf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueuedLeaf")
            .field("leaf_len", &self.leaf.len())
            .field("has_dataset", &self.dataset.is_some())
            .field("estimated_work_ms", &self.estimated_work_ms)
            .field("estimated_mem_bytes", &self.estimated_mem_bytes)
            .finish()
    }
}

struct BatchEdgeCollector {
    edges: Mutex<Vec<PendingEdge>>,
    lune_witnesses: Mutex<Vec<PendingLuneWitness>>,
}

impl BatchEdgeCollector {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            edges: Mutex::new(Vec::with_capacity(capacity)),
            lune_witnesses: Mutex::new(Vec::new()),
        }
    }

    fn flush_into(self, edge_sink: &dyn PendingEdgeSink) -> AnnResult<()> {
        let mut edges = self.edges.into_inner();
        edge_sink.flush_pending_edges(&mut edges)?;
        let mut witnesses = self.lune_witnesses.into_inner();
        edge_sink.flush_lune_witnesses(&mut witnesses)
    }
}

impl PendingEdgeSink for BatchEdgeCollector {
    fn flush_pending_edges(&self, edges: &mut Vec<PendingEdge>) -> AnnResult<()> {
        self.edges.lock().append(edges);
        Ok(())
    }

    fn flush_lune_witnesses(&self, witnesses: &mut Vec<PendingLuneWitness>) -> AnnResult<()> {
        self.lune_witnesses.lock().append(witnesses);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LeafDrainerLimitEntry {
    limit: usize,
    backpressure_limit: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LeafBatchDrainPolicy {
    enabled: bool,
    max_leaves: usize,
    max_points: usize,
    min_backlog: usize,
    max_active_hydrations: usize,
    post_producer_scale: usize,
    post_producer_max_leaves: usize,
    post_producer_max_points: usize,
}

struct LeafBatchHydrationGuard<'s, 'a> {
    scheduler: &'s ForgeannScheduler<'a>,
}

impl Drop for LeafBatchHydrationGuard<'_, '_> {
    fn drop(&mut self) {
        self.scheduler
            .active_leaf_batch_hydrations
            .fetch_sub(1, Ordering::AcqRel);
    }
}

pub(super) struct ScopedLeafEmitter<'borrow, 'scope, 'a> {
    scheduler: Arc<ForgeannScheduler<'a>>,
    scope: &'borrow ScopeFifo<'scope>,
}

thread_local! {
    static TLS_LEAF_SCRATCH: RefCell<Option<LeafScratch>> = const { RefCell::new(None) };
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn default_leaf_batch_hydration_limit(worker_count: usize) -> usize {
    let _ = worker_count;
    0
}

fn default_leaf_batch_post_producer_scale(worker_count: usize) -> usize {
    if worker_count >= 8 { 2 } else { 1 }
}

fn leaf_batch_drain_policy_with_override(
    params: &ForgeANNParams,
    enabled_override: Option<usize>,
    max_leaves_override: Option<usize>,
    max_points_override: Option<usize>,
    min_backlog_override: Option<usize>,
    max_active_hydrations_override: Option<usize>,
    post_producer_scale_override: Option<usize>,
    point_pipeline_config: Option<&PointPipelineConfig>,
    dim: usize,
    worker_count: usize,
) -> LeafBatchDrainPolicy {
    let enabled = enabled_override
        .map(|value| value != 0)
        .unwrap_or(params.leaf_batch_drain_enable);
    if !enabled || point_pipeline_config.is_none() || dim == 0 {
        return LeafBatchDrainPolicy::default();
    }

    let row_bytes = dim.saturating_mul(size_of::<f32>()).max(1);
    let config_budget_points = point_pipeline_config
        .map(|config| config.budget_bytes.max(row_bytes) / row_bytes)
        .unwrap_or(1)
        .max(1);
    let max_leaves = max_leaves_override
        .unwrap_or(params.leaf_batch_drain_max_leaves)
        .max(1);
    let max_points = max_points_override
        .unwrap_or(params.leaf_batch_drain_max_points)
        .max(1)
        .min(config_budget_points);
    let min_backlog = min_backlog_override
        .unwrap_or(params.leaf_batch_drain_min_backlog)
        .max(1);
    let max_active_hydrations = max_active_hydrations_override
        .unwrap_or_else(|| default_leaf_batch_hydration_limit(worker_count))
        .min(worker_count.max(1));
    let post_producer_scale = post_producer_scale_override
        .unwrap_or_else(|| default_leaf_batch_post_producer_scale(worker_count))
        .max(1)
        .min(worker_count.max(1));
    let post_producer_max_leaves = max_leaves
        .saturating_mul(post_producer_scale)
        .max(max_leaves);
    let post_producer_max_points = max_points
        .saturating_mul(post_producer_scale)
        .max(max_points)
        .min(config_budget_points);

    LeafBatchDrainPolicy {
        enabled: true,
        max_leaves,
        max_points,
        min_backlog,
        max_active_hydrations,
        post_producer_scale,
        post_producer_max_leaves,
        post_producer_max_points,
    }
}

fn leaf_batch_drain_policy(
    params: &ForgeANNParams,
    point_pipeline_config: Option<&PointPipelineConfig>,
    dim: usize,
    worker_count: usize,
) -> LeafBatchDrainPolicy {
    leaf_batch_drain_policy_with_override(
        params,
        env_usize("FORGEANN_LEAF_BATCH_DRAIN"),
        env_usize("FORGEANN_LEAF_BATCH_DRAIN_MAX_LEAVES"),
        env_usize("FORGEANN_LEAF_BATCH_DRAIN_MAX_POINTS"),
        env_usize("FORGEANN_LEAF_BATCH_DRAIN_MIN_BACKLOG"),
        env_usize("FORGEANN_LEAF_BATCH_DRAIN_MAX_ACTIVE_HYDRATIONS"),
        env_usize("FORGEANN_LEAF_BATCH_DRAIN_POST_PRODUCER_SCALE"),
        point_pipeline_config,
        dim,
        worker_count,
    )
}

impl<'a> ForgeannScheduler<'a> {
    pub(super) fn new(
        budget: SchedulerBudget,
        context: LeafTaskContext<'a>,
        progress: &'a ProgressBar,
    ) -> Arc<Self> {
        let leaf_batch_drain_policy = leaf_batch_drain_policy(
            context.params,
            context.point_pipeline_config,
            context.dataset.dim(),
            budget.worker_count,
        );
        let leaf_ads_runtime =
            LeafAdsOperatorRuntime::from_params(context.params, budget.worker_count);
        Arc::new(Self {
            budget,
            context,
            leaf_ads_runtime,
            leaf_batch_drain_policy,
            progress,
            leaf_backlog: ArrayQueue::new(budget.leaf_backlog_capacity.max(1)),
            outstanding_leaf_tasks: AtomicUsize::new(0),
            inflight_leaf_tasks: AtomicUsize::new(0),
            peak_inflight_leaf_tasks: AtomicUsize::new(0),
            peak_leaf_backlog: AtomicUsize::new(0),
            active_leaf_drainers: AtomicUsize::new(0),
            retiring_leaf_drainers: AtomicUsize::new(0),
            leaf_drainer_limits: Mutex::new(Vec::new()),
            leaf_drainer_limit: AtomicUsize::new(0),
            leaf_drainer_backpressure_limit: AtomicUsize::new(0),
            active_large_assignment_count: AtomicUsize::new(0),
            active_large_assignment_peak: AtomicUsize::new(0),
            producer_active: AtomicBool::new(true),
            inline_leaf_fallbacks: AtomicUsize::new(0),
            producer_help_drains: AtomicUsize::new(0),
            producer_help_drain_ns: AtomicU64::new(0),
            backlog_full_help_drains: AtomicUsize::new(0),
            backlog_full_help_ns: AtomicU64::new(0),
            producer_help_yields: AtomicUsize::new(0),
            large_leaf_count: AtomicUsize::new(0),
            large_leaf_block_tasks: AtomicUsize::new(0),
            large_assignment_guard_count: AtomicUsize::new(0),
            leaf_cap_large_assignment_hits: AtomicUsize::new(0),
            leaf_cap_producer_hits: AtomicUsize::new(0),
            leaf_cap_done_hits: AtomicUsize::new(0),
            batched_leaf_drains: AtomicUsize::new(0),
            batched_leaf_leaves: AtomicUsize::new(0),
            batched_leaf_points: AtomicUsize::new(0),
            batched_leaf_unique_points: AtomicUsize::new(0),
            batched_leaf_load_ns: AtomicU64::new(0),
            peak_leaf_batch_leaves: AtomicUsize::new(0),
            peak_leaf_batch_points: AtomicUsize::new(0),
            peak_leaf_batch_unique_points: AtomicUsize::new(0),
            active_leaf_batch_hydrations: AtomicUsize::new(0),
            peak_leaf_batch_hydrations: AtomicUsize::new(0),
            leaf_batch_hydration_wait_ns: AtomicU64::new(0),
            queued_work_ms: AtomicU64::new(0),
            queued_work_ms_peak: AtomicU64::new(0),
            queued_mem_bytes: AtomicUsize::new(0),
            queued_mem_bytes_peak: AtomicUsize::new(0),
            replay_admitted_runs: AtomicUsize::new(0),
            replay_paused_ns: AtomicU64::new(0),
            worker_busy_ns: AtomicU64::new(0),
            worker_idle_ns: AtomicU64::new(0),
            first_error: Mutex::new(None),
            leaf_profile: Mutex::new(LeafProfile::default()),
        })
    }

    pub(super) fn leaf_profile(&self) -> LeafProfile {
        self.leaf_profile.lock().clone()
    }

    pub(super) fn telemetry(&self) -> SchedulerTelemetry {
        SchedulerTelemetry {
            peak_inflight_leaf_tasks: self.peak_inflight_leaf_tasks.load(Ordering::Relaxed),
            peak_leaf_backlog: self.peak_leaf_backlog.load(Ordering::Relaxed),
            leaf_backlog: self.leaf_backlog.len(),
            producer_leaf_backlog_soft_limit: self.budget.producer_leaf_backlog_soft_limit,
            outstanding_leaf_tasks: self.outstanding_leaf_tasks.load(Ordering::Relaxed),
            inflight_leaf_tasks: self.inflight_leaf_tasks.load(Ordering::Relaxed),
            active_leaf_drainers: self.active_leaf_drainers.load(Ordering::Relaxed),
            leaf_drainer_limit: self.leaf_drainer_limit.load(Ordering::Relaxed),
            leaf_drainer_backpressure_limit: self
                .leaf_drainer_backpressure_limit
                .load(Ordering::Relaxed),
            inline_leaf_fallbacks: self.inline_leaf_fallbacks.load(Ordering::Relaxed),
            producer_help_drains: self.producer_help_drains.load(Ordering::Relaxed),
            producer_help_drain_ms: nanos_to_millis_u64(
                self.producer_help_drain_ns.load(Ordering::Relaxed),
            ),
            backlog_full_help_drains: self.backlog_full_help_drains.load(Ordering::Relaxed),
            backlog_full_help_ms: nanos_to_millis_u64(
                self.backlog_full_help_ns.load(Ordering::Relaxed),
            ),
            producer_help_yields: self.producer_help_yields.load(Ordering::Relaxed),
            large_leaf_count: self.large_leaf_count.load(Ordering::Relaxed),
            large_leaf_block_tasks: self.large_leaf_block_tasks.load(Ordering::Relaxed),
            active_large_assignment_peak: self.active_large_assignment_peak.load(Ordering::Relaxed),
            large_assignment_guard_count: self.large_assignment_guard_count.load(Ordering::Relaxed),
            leaf_cap_large_assignment_hits: self
                .leaf_cap_large_assignment_hits
                .load(Ordering::Relaxed),
            leaf_cap_producer_hits: self.leaf_cap_producer_hits.load(Ordering::Relaxed),
            leaf_cap_done_hits: self.leaf_cap_done_hits.load(Ordering::Relaxed),
            batched_leaf_drains: self.batched_leaf_drains.load(Ordering::Relaxed),
            batched_leaf_leaves: self.batched_leaf_leaves.load(Ordering::Relaxed),
            batched_leaf_points: self.batched_leaf_points.load(Ordering::Relaxed),
            batched_leaf_unique_points: self.batched_leaf_unique_points.load(Ordering::Relaxed),
            batched_leaf_load_ms: nanos_to_millis_u64(
                self.batched_leaf_load_ns.load(Ordering::Relaxed),
            ),
            leaf_batch_hydration_limit: self.leaf_batch_drain_policy.max_active_hydrations,
            leaf_batch_post_producer_scale: self.leaf_batch_drain_policy.post_producer_scale,
            peak_leaf_batch_leaves: self.peak_leaf_batch_leaves.load(Ordering::Relaxed),
            peak_leaf_batch_points: self.peak_leaf_batch_points.load(Ordering::Relaxed),
            peak_leaf_batch_unique_points: self
                .peak_leaf_batch_unique_points
                .load(Ordering::Relaxed),
            active_leaf_batch_hydrations: self.active_leaf_batch_hydrations.load(Ordering::Relaxed),
            peak_leaf_batch_hydrations: self.peak_leaf_batch_hydrations.load(Ordering::Relaxed),
            leaf_batch_hydration_wait_ms: nanos_to_millis_u64(
                self.leaf_batch_hydration_wait_ns.load(Ordering::Relaxed),
            ),
            work_graph_enabled: self.work_graph_enabled(),
            queued_work_ms_peak: self.queued_work_ms_peak.load(Ordering::Relaxed),
            queued_mem_bytes_peak: self.queued_mem_bytes_peak.load(Ordering::Relaxed),
            replay_admitted_runs: self.replay_admitted_runs.load(Ordering::Relaxed),
            replay_paused_ms: nanos_to_millis_u64(self.replay_paused_ns.load(Ordering::Relaxed)),
            worker_busy_ms: nanos_to_millis_u64(self.worker_busy_ns.load(Ordering::Relaxed)),
            worker_idle_ms: nanos_to_millis_u64(self.worker_idle_ns.load(Ordering::Relaxed)),
        }
    }

    #[cfg(test)]
    pub(crate) fn enter_large_assignment(&self) -> LargeAssignmentGuard<'_> {
        LargeAssignmentGuard::enter(self)
    }

    #[cfg(test)]
    pub(crate) fn enter_leaf_drainer_limit(&self, limit: usize) -> LeafDrainerLimitGuard<'_> {
        LeafDrainerLimitGuard::enter(self, limit)
    }

    #[cfg(test)]
    pub(crate) fn enter_leaf_drainer_limit_with_backpressure_limit(
        &self,
        limit: usize,
        backpressure_limit: usize,
    ) -> LeafDrainerLimitGuard<'_> {
        LeafDrainerLimitGuard::enter_with_backpressure_limit(self, limit, backpressure_limit)
    }

    pub(super) fn check_for_error(&self) -> AnnResult<()> {
        if let Some(message) = self.first_error.lock().clone() {
            Err(AnnError::log_index_error(message))
        } else {
            Ok(())
        }
    }

    pub(super) fn wait_until_idle(&self) -> AnnResult<()> {
        while self.outstanding_leaf_tasks.load(Ordering::Acquire) != 0 {
            self.check_for_error()?;
            if !self.try_help_from_backlog(false)? && rayon::yield_now().is_none() {
                let idle_start = Instant::now();
                self.producer_help_yields.fetch_add(1, Ordering::Relaxed);
                std::thread::yield_now();
                self.worker_idle_ns
                    .fetch_add(duration_nanos_u64(idle_start.elapsed()), Ordering::Relaxed);
            }
        }
        self.check_for_error()
    }

    fn wait_for_leaf_backlog_below<'scope>(
        self: &Arc<Self>,
        scope: &ScopeFifo<'scope>,
        target_backlog: usize,
    ) -> AnnResult<bool>
    where
        'a: 'scope,
    {
        let mut waited = false;
        while self.leaf_backlog.len() > target_backlog {
            waited = true;
            self.check_for_error()?;
            self.maybe_spawn_leaf_drainer(scope);
            if self.try_help_from_backlog(true)? {
                continue;
            }
            if rayon::yield_now().is_none() {
                let idle_start = Instant::now();
                self.producer_help_yields.fetch_add(1, Ordering::Relaxed);
                std::thread::yield_now();
                self.worker_idle_ns
                    .fetch_add(duration_nanos_u64(idle_start.elapsed()), Ordering::Relaxed);
            }
        }
        self.check_for_error()?;
        Ok(waited)
    }

    fn mark_producer_finished(&self) {
        self.producer_active.store(false, Ordering::Release);
    }

    fn discard_queued_leaf_backlog(&self) {
        while let Some(queued) = self.leaf_backlog.pop() {
            self.remove_queued_leaf_credit(&queued);
            self.outstanding_leaf_tasks.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn producer_is_active(&self) -> bool {
        self.producer_active.load(Ordering::Acquire)
    }

    fn max_leaf_worker_capacity(&self) -> usize {
        self.budget.worker_count.saturating_sub(1).max(1)
    }

    fn effective_leaf_batch_limits(
        &self,
        policy: LeafBatchDrainPolicy,
        available_with_first: usize,
    ) -> (usize, usize) {
        if self.producer_is_active() || policy.post_producer_scale <= 1 {
            return (policy.max_leaves, policy.max_points);
        }

        let scale_threshold = policy
            .max_leaves
            .saturating_mul(self.max_leaf_worker_capacity())
            .max(policy.min_backlog);
        if available_with_first < scale_threshold {
            return (policy.max_leaves, policy.max_points);
        }

        (
            policy.post_producer_max_leaves,
            policy.post_producer_max_points,
        )
    }

    fn active_producer_leaf_worker_cap(&self) -> usize {
        if self.budget.worker_count <= 1 {
            return 1;
        }

        // While RBC is still producing recursive work, leaf drainers should leave room for the
        // producer and the rest of the pool, but the previous three-quarter cap still left the
        // backlog saturated on wiki35m. Keep the full leaf worker budget available except for
        // one worker reserved for the producer-side path itself.
        self.max_leaf_worker_capacity()
    }

    fn leaf_worker_limit_for_backlog(&self, backlog: usize, count_hits: bool) -> usize {
        if backlog == 0 {
            return 0;
        }

        let cap = if self.active_large_assignment_count.load(Ordering::Acquire) > 0 {
            if count_hits {
                self.leaf_cap_large_assignment_hits
                    .fetch_add(1, Ordering::Relaxed);
            }
            self.max_leaf_worker_capacity().min(2).max(1)
        } else if self.producer_is_active() {
            if count_hits {
                self.leaf_cap_producer_hits.fetch_add(1, Ordering::Relaxed);
            }
            self.active_producer_leaf_worker_cap()
        } else {
            if count_hits {
                self.leaf_cap_done_hits.fetch_add(1, Ordering::Relaxed);
            }
            self.max_leaf_worker_capacity()
        };

        let limit = self.effective_leaf_drainer_limit_for_backlog(backlog);
        let cap = if limit == 0 { cap } else { cap.min(limit) };
        backlog.min(cap)
    }

    fn effective_leaf_drainer_limit_for_backlog(&self, backlog: usize) -> usize {
        let limit = self.leaf_drainer_limit.load(Ordering::Acquire);
        if limit == 0 || backlog == 0 {
            return limit;
        }

        let backlog_capacity = self.budget.leaf_backlog_capacity.max(1);
        if backlog.saturating_mul(4) < backlog_capacity.saturating_mul(3) {
            return limit;
        }

        let worker_capacity = self.max_leaf_worker_capacity();
        let backpressure_limit = self.leaf_drainer_backpressure_limit.load(Ordering::Acquire);
        if backpressure_limit == 0 {
            return limit;
        }
        backpressure_limit.min(worker_capacity).max(1)
    }

    fn spawned_leaf_worker_limit_for_backlog(&self, backlog: usize) -> usize {
        self.leaf_worker_limit_for_backlog(backlog, true)
    }

    fn try_reserve_leaf_drainer_retirement(&self, backlog: usize) -> bool {
        if backlog == 0 {
            return false;
        }
        let target = self.leaf_worker_limit_for_backlog(backlog, false);
        loop {
            let active = self.active_leaf_drainers.load(Ordering::Acquire);
            let retiring = self.retiring_leaf_drainers.load(Ordering::Acquire);
            if active.saturating_sub(retiring) <= target {
                return false;
            }
            match self.retiring_leaf_drainers.compare_exchange_weak(
                retiring,
                retiring + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(_) => continue,
            }
        }
    }

    #[cfg(test)]
    fn try_enqueue_leaf(&self, leaf: Vec<u32>) -> Result<(), QueuedLeaf> {
        let queued = self.make_queued_leaf(leaf, None);
        self.try_enqueue_queued_leaf(queued)
    }

    fn try_enqueue_queued_leaf(&self, queued: QueuedLeaf) -> Result<(), QueuedLeaf> {
        let estimated_work_ms = queued.estimated_work_ms;
        let estimated_mem_bytes = queued.estimated_mem_bytes;
        self.outstanding_leaf_tasks.fetch_add(1, Ordering::AcqRel);
        match self.leaf_backlog.push(queued) {
            Ok(()) => {
                self.add_queued_leaf_credit(estimated_work_ms, estimated_mem_bytes);
                self.update_peak_backlog(self.leaf_backlog.len());
                Ok(())
            }
            Err(queued) => {
                self.outstanding_leaf_tasks.fetch_sub(1, Ordering::AcqRel);
                Err(queued)
            }
        }
    }

    fn make_queued_leaf(&self, leaf: Vec<u32>, dataset: Option<Arc<dyn PointStore>>) -> QueuedLeaf {
        let estimated_work_ms = self.estimate_leaf_work_ms(leaf.len());
        let estimated_mem_bytes = leaf
            .len()
            .saturating_mul(size_of::<u32>())
            .saturating_add(estimated_work_ms as usize);
        QueuedLeaf {
            leaf,
            dataset,
            estimated_work_ms,
            estimated_mem_bytes,
        }
    }

    fn add_queued_leaf_credit(&self, estimated_work_ms: u64, estimated_mem_bytes: usize) {
        if !self.work_graph_enabled() {
            return;
        }
        self.replay_admitted_runs.fetch_add(1, Ordering::Relaxed);
        let new_work = self
            .queued_work_ms
            .fetch_add(estimated_work_ms, Ordering::AcqRel)
            + estimated_work_ms;
        update_atomic_max_u64(&self.queued_work_ms_peak, new_work);
        let new_mem = self
            .queued_mem_bytes
            .fetch_add(estimated_mem_bytes, Ordering::AcqRel)
            + estimated_mem_bytes;
        update_atomic_max_usize(&self.queued_mem_bytes_peak, new_mem);
    }

    fn remove_queued_leaf_credit(&self, queued: &QueuedLeaf) {
        if !self.work_graph_enabled() {
            return;
        }
        self.queued_work_ms
            .fetch_sub(queued.estimated_work_ms, Ordering::AcqRel);
        self.queued_mem_bytes
            .fetch_sub(queued.estimated_mem_bytes, Ordering::AcqRel);
    }

    fn work_graph_enabled(&self) -> bool {
        self.context.params.leaf_ads_work_graph_enable
    }

    fn target_queued_work_ms(&self) -> u64 {
        (self.budget.worker_count.max(1) as u64)
            .saturating_mul(
                self.context
                    .params
                    .leaf_ads_work_graph_target_queue_ms
                    .max(1),
            )
            .max(1)
    }

    fn estimate_leaf_work_ms(&self, leaf_len: usize) -> u64 {
        if !self.work_graph_enabled() {
            return 0;
        }
        let quantum = self.context.params.leaf_ads_work_graph_quantum_ms.max(1);
        let tile_rows = self.context.params.leaf_ads_work_graph_min_tile_rows.max(1);
        let tiles = leaf_len.div_ceil(tile_rows).max(1) as u64;
        tiles.saturating_mul(quantum).max(1)
    }

    fn update_peak_inflight(&self, inflight: usize) {
        let mut peak = self.peak_inflight_leaf_tasks.load(Ordering::Acquire);
        while inflight > peak {
            match self.peak_inflight_leaf_tasks.compare_exchange_weak(
                peak,
                inflight,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(next) => peak = next,
            }
        }
    }

    fn update_peak_backlog(&self, backlog: usize) {
        let mut peak = self.peak_leaf_backlog.load(Ordering::Acquire);
        while backlog > peak {
            match self.peak_leaf_backlog.compare_exchange_weak(
                peak,
                backlog,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(next) => peak = next,
            }
        }
    }

    fn record_error(&self, err: AnnError) {
        let mut first_error = self.first_error.lock();
        if first_error.is_none() {
            *first_error = Some(err.to_string());
        }
    }

    fn start_leaf_execution(&self) {
        let inflight = self.inflight_leaf_tasks.fetch_add(1, Ordering::AcqRel) + 1;
        self.update_peak_inflight(inflight);
    }

    fn finish_leaf_execution(&self) {
        self.inflight_leaf_tasks.fetch_sub(1, Ordering::AcqRel);
        self.outstanding_leaf_tasks.fetch_sub(1, Ordering::AcqRel);
    }

    fn finish_leaf_executions(&self, count: usize) {
        for _ in 0..count {
            self.finish_leaf_execution();
        }
    }

    fn try_take_queued_leaf(&self) -> Option<QueuedLeaf> {
        let queued = self.leaf_backlog.pop()?;
        self.remove_queued_leaf_credit(&queued);
        self.start_leaf_execution();
        Some(queued)
    }

    fn try_take_queued_leaf_batch(&self) -> Option<Vec<QueuedLeaf>> {
        let first = self.try_take_queued_leaf()?;
        let policy = self.leaf_batch_drain_policy;
        let available_with_first = self.leaf_backlog.len().saturating_add(1);
        if !policy.enabled || available_with_first < policy.min_backlog {
            return Some(vec![first]);
        }

        let (max_leaves, max_points) =
            self.effective_leaf_batch_limits(policy, available_with_first);
        let mut leaves = Vec::with_capacity(max_leaves.max(1));
        let mut points = first.leaf.len();
        leaves.push(first);
        while leaves.len() < max_leaves && points < max_points {
            let Some(leaf) = self.try_take_queued_leaf() else {
                break;
            };
            points = points.saturating_add(leaf.leaf.len());
            leaves.push(leaf);
        }
        Some(leaves)
    }

    #[cfg(test)]
    fn process_taken_leaf(&self, queued: QueuedLeaf) -> AnnResult<()> {
        struct FinishOnDrop<'s, 'a> {
            scheduler: &'s ForgeannScheduler<'a>,
        }

        impl Drop for FinishOnDrop<'_, '_> {
            fn drop(&mut self) {
                self.scheduler.finish_leaf_execution();
            }
        }

        let _finish = FinishOnDrop { scheduler: self };
        let busy_start = Instant::now();
        let dataset = queued.dataset.as_deref().unwrap_or(self.context.dataset);
        let result = self.process_leaf_task_with_dataset(queued.leaf, dataset);
        self.worker_busy_ns
            .fetch_add(duration_nanos_u64(busy_start.elapsed()), Ordering::Relaxed);
        result
    }

    fn process_taken_leaf_batch(&self, queued_leaves: Vec<QueuedLeaf>) -> AnnResult<()> {
        struct FinishBatchOnDrop<'s, 'a> {
            scheduler: &'s ForgeannScheduler<'a>,
            count: usize,
        }

        impl Drop for FinishBatchOnDrop<'_, '_> {
            fn drop(&mut self) {
                self.scheduler.finish_leaf_executions(self.count);
            }
        }

        let count = queued_leaves.len();
        let _finish = FinishBatchOnDrop {
            scheduler: self,
            count,
        };
        let busy_start = Instant::now();
        let result = self.process_queued_leaf_batch(queued_leaves);
        self.worker_busy_ns
            .fetch_add(duration_nanos_u64(busy_start.elapsed()), Ordering::Relaxed);
        result
    }

    fn same_queued_dataset(
        left: &Option<Arc<dyn PointStore>>,
        right: &Option<Arc<dyn PointStore>>,
    ) -> bool {
        match (left, right) {
            (None, None) => true,
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    fn flush_queued_leaf_batch(
        &self,
        dataset: Option<Arc<dyn PointStore>>,
        leaves: &mut Vec<Vec<u32>>,
    ) -> AnnResult<()> {
        if leaves.is_empty() {
            return Ok(());
        }
        let batch = std::mem::take(leaves);
        if let Some(dataset) = dataset.as_deref() {
            self.process_leaf_batch_with_dataset(batch, dataset)
        } else {
            self.process_leaf_batch(batch)
        }
    }

    fn process_queued_leaf_batch(&self, queued_leaves: Vec<QueuedLeaf>) -> AnnResult<()> {
        let mut current_dataset: Option<Arc<dyn PointStore>> = None;
        let mut current_leaves = Vec::new();
        for queued in queued_leaves {
            if current_leaves.is_empty() {
                current_dataset = queued.dataset.clone();
            } else if !Self::same_queued_dataset(&current_dataset, &queued.dataset) {
                self.flush_queued_leaf_batch(current_dataset.take(), &mut current_leaves)?;
                current_dataset = queued.dataset.clone();
            }
            current_leaves.push(queued.leaf);
        }
        self.flush_queued_leaf_batch(current_dataset, &mut current_leaves)
    }

    fn try_help_from_backlog(&self, count_drain: bool) -> AnnResult<bool> {
        self.try_help_from_backlog_recorded(count_drain, false)
    }

    fn try_help_from_backlog_recorded(
        &self,
        count_drain: bool,
        backlog_full: bool,
    ) -> AnnResult<bool> {
        let Some(leaves) = self.try_take_queued_leaf_batch() else {
            return Ok(false);
        };
        if count_drain {
            self.producer_help_drains.fetch_add(1, Ordering::Relaxed);
        }
        if backlog_full {
            self.backlog_full_help_drains
                .fetch_add(1, Ordering::Relaxed);
        }
        let start = Instant::now();
        let result = self.process_taken_leaf_batch(leaves);
        let elapsed_ns = duration_nanos_u64(start.elapsed());
        if count_drain {
            self.producer_help_drain_ns
                .fetch_add(elapsed_ns, Ordering::Relaxed);
        }
        if backlog_full {
            self.backlog_full_help_ns
                .fetch_add(elapsed_ns, Ordering::Relaxed);
        }
        result?;
        Ok(true)
    }

    fn finish_leaf_drainer<'scope>(self: &Arc<Self>, scope: &ScopeFifo<'scope>, retiring: bool)
    where
        'a: 'scope,
    {
        if retiring {
            self.retiring_leaf_drainers.fetch_sub(1, Ordering::AcqRel);
        }
        self.active_leaf_drainers.fetch_sub(1, Ordering::AcqRel);
        self.maybe_spawn_leaf_drainer(scope);
    }

    fn run_leaf_drainer<'scope>(self: Arc<Self>, scope: &ScopeFifo<'scope>)
    where
        'a: 'scope,
    {
        let mut retiring = false;
        loop {
            if self.check_for_error().is_err() {
                break;
            }

            let backlog = self.leaf_backlog.len();
            if self.try_reserve_leaf_drainer_retirement(backlog) {
                retiring = true;
                break;
            }

            let Some(leaves) = self.try_take_queued_leaf_batch() else {
                break;
            };

            if let Err(err) = self.process_taken_leaf_batch(leaves) {
                self.record_error(err);
                break;
            }
        }

        self.finish_leaf_drainer(scope, retiring);
    }

    fn maybe_spawn_leaf_drainer<'scope>(self: &Arc<Self>, scope: &ScopeFifo<'scope>)
    where
        'a: 'scope,
    {
        loop {
            let backlog = self.leaf_backlog.len();
            if backlog == 0 {
                return;
            }

            let active = self.active_leaf_drainers.load(Ordering::Acquire);
            let target = self.spawned_leaf_worker_limit_for_backlog(backlog);
            if active >= target {
                return;
            }

            match self.active_leaf_drainers.compare_exchange_weak(
                active,
                active + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let scheduler = Arc::clone(self);
                    scope.spawn_fifo(move |scope| scheduler.run_leaf_drainer(scope));
                }
                Err(_) => continue,
            }
        }
    }

    fn process_leaf_batch(&self, leaves: Vec<Vec<u32>>) -> AnnResult<()> {
        self.process_leaf_batch_with_dataset(leaves, self.context.dataset)
    }

    fn process_leaf_batch_with_dataset(
        &self,
        leaves: Vec<Vec<u32>>,
        dataset: &dyn PointStore,
    ) -> AnnResult<()> {
        if leaves.len() <= 1 || !self.leaf_batch_drain_policy.enabled {
            for leaf in leaves {
                self.process_leaf_task_with_dataset(leaf, dataset)?;
            }
            return Ok(());
        }

        let total_points = leaves.iter().map(Vec::len).sum::<usize>();
        let mut unique_ids = leaves
            .iter()
            .flat_map(|leaf| leaf.iter().copied())
            .collect::<Vec<_>>();
        unique_ids.sort_unstable();
        unique_ids.dedup();

        if unique_ids.is_empty() {
            return Ok(());
        }

        update_atomic_max_usize(&self.peak_leaf_batch_leaves, leaves.len());
        update_atomic_max_usize(&self.peak_leaf_batch_points, total_points);
        update_atomic_max_usize(&self.peak_leaf_batch_unique_points, unique_ids.len());

        if dataset.is_resident_subset() {
            leaves
                .par_iter()
                .try_for_each(|leaf| self.process_leaf_task_with_dataset(leaf.clone(), dataset))?;
            self.batched_leaf_drains.fetch_add(1, Ordering::Relaxed);
            self.batched_leaf_leaves
                .fetch_add(leaves.len(), Ordering::Relaxed);
            self.batched_leaf_points
                .fetch_add(total_points, Ordering::Relaxed);
            self.batched_leaf_unique_points
                .fetch_add(unique_ids.len(), Ordering::Relaxed);
            return Ok(());
        }

        let Some(config) = self.context.point_pipeline_config else {
            for leaf in leaves {
                self.process_leaf_task_with_dataset(leaf, dataset)?;
            }
            return Ok(());
        };

        let hydrated = if self.leaf_batch_drain_policy.max_active_hydrations == 0 {
            hydrate_resident_subset_with_stats(dataset, &unique_ids, config)?
        } else {
            let _hydration_guard = self.acquire_leaf_batch_hydration_slot()?;
            hydrate_resident_subset_with_stats(dataset, &unique_ids, config)?
        };
        let mut batch_profile = LeafProfile::default();
        let edge_capacity = total_points
            .saturating_mul(self.context.params.leaf_knn.max(1))
            .saturating_mul(2);
        let batch_edge_sink = BatchEdgeCollector::with_capacity(edge_capacity);
        for leaf in &leaves {
            let profile = self.build_leaf_profile_with_dataset_and_sink(
                leaf,
                &hydrated.store,
                &batch_edge_sink,
            )?;
            batch_profile.merge(profile);
        }
        let flush_start = Instant::now();
        batch_edge_sink.flush_into(self.context.edge_sink)?;
        batch_profile.flush += flush_start.elapsed();

        if self.context.detailed_profiling {
            batch_profile.merge(LeafProfile {
                load: hydrated.load,
                io_stats: hydrated.io_stats,
                ..LeafProfile::default()
            });
        }
        self.record_completed_leaf_profile(batch_profile, leaves.len());

        self.batched_leaf_drains.fetch_add(1, Ordering::Relaxed);
        self.batched_leaf_leaves
            .fetch_add(leaves.len(), Ordering::Relaxed);
        self.batched_leaf_points
            .fetch_add(total_points, Ordering::Relaxed);
        self.batched_leaf_unique_points
            .fetch_add(unique_ids.len(), Ordering::Relaxed);
        self.batched_leaf_load_ns
            .fetch_add(duration_nanos_u64(hydrated.load), Ordering::Relaxed);
        Ok(())
    }

    fn acquire_leaf_batch_hydration_slot(&self) -> AnnResult<LeafBatchHydrationGuard<'_, 'a>> {
        let limit = self.leaf_batch_drain_policy.max_active_hydrations.max(1);
        let wait_start = Instant::now();
        let mut waited = false;
        loop {
            self.check_for_error()?;
            let active = self.active_leaf_batch_hydrations.load(Ordering::Acquire);
            if active < limit {
                match self.active_leaf_batch_hydrations.compare_exchange_weak(
                    active,
                    active + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        update_atomic_max_usize(&self.peak_leaf_batch_hydrations, active + 1);
                        if waited {
                            self.leaf_batch_hydration_wait_ns.fetch_add(
                                duration_nanos_u64(wait_start.elapsed()),
                                Ordering::Relaxed,
                            );
                        }
                        return Ok(LeafBatchHydrationGuard { scheduler: self });
                    }
                    Err(_) => continue,
                }
            }

            waited = true;
            if rayon::yield_now().is_none() {
                let idle_start = Instant::now();
                self.producer_help_yields.fetch_add(1, Ordering::Relaxed);
                std::thread::yield_now();
                self.worker_idle_ns
                    .fetch_add(duration_nanos_u64(idle_start.elapsed()), Ordering::Relaxed);
            }
        }
    }

    fn maybe_wait_for_producer_leaf_backlog_soft_limit<'scope>(
        self: &Arc<Self>,
        scope: &ScopeFifo<'scope>,
    ) -> AnnResult<()>
    where
        'a: 'scope,
    {
        if self.budget.producer_leaf_backlog_soft_limit == 0 {
            return Ok(());
        }
        self.wait_for_leaf_backlog_below(scope, self.budget.producer_leaf_backlog_soft_limit)?;
        Ok(())
    }

    fn maybe_wait_for_work_graph_credit<'scope>(
        self: &Arc<Self>,
        scope: &ScopeFifo<'scope>,
    ) -> AnnResult<()>
    where
        'a: 'scope,
    {
        if !self.work_graph_enabled() {
            return Ok(());
        }
        let target = self.target_queued_work_ms();
        let mut waited = false;
        let wait_start = Instant::now();
        while self.queued_work_ms.load(Ordering::Acquire) >= target
            && self.outstanding_leaf_tasks.load(Ordering::Acquire) != 0
        {
            waited = true;
            self.check_for_error()?;
            self.maybe_spawn_leaf_drainer(scope);
            if self.try_help_from_backlog_recorded(true, false)? {
                continue;
            }
            if rayon::yield_now().is_none() {
                let idle_start = Instant::now();
                self.producer_help_yields.fetch_add(1, Ordering::Relaxed);
                std::thread::yield_now();
                self.worker_idle_ns
                    .fetch_add(duration_nanos_u64(idle_start.elapsed()), Ordering::Relaxed);
            }
        }
        if waited {
            self.replay_paused_ns
                .fetch_add(duration_nanos_u64(wait_start.elapsed()), Ordering::Relaxed);
        }
        self.check_for_error()
    }

    fn process_leaf_task_with_dataset(
        &self,
        leaf: Vec<u32>,
        dataset: &dyn PointStore,
    ) -> AnnResult<()> {
        let profile = self.build_leaf_profile_with_dataset_and_sketches(
            &leaf,
            dataset,
            self.context.sketches,
        )?;
        self.record_completed_leaf_profile(profile, 1);
        Ok(())
    }

    fn process_leaf_task_with_dataset_and_sketches(
        &self,
        leaf: Vec<u32>,
        dataset: &dyn PointStore,
        sketches: &dyn SketchAccessor,
    ) -> AnnResult<()> {
        let profile =
            self.build_leaf_profile_with_dataset_and_sketches(&leaf, dataset, sketches)?;
        self.record_completed_leaf_profile(profile, 1);
        Ok(())
    }

    fn build_leaf_profile_with_dataset_and_sketches(
        &self,
        leaf: &[u32],
        dataset: &dyn PointStore,
        sketches: &dyn SketchAccessor,
    ) -> AnnResult<LeafProfile> {
        self.build_leaf_profile_with_dataset_sketches_and_sink(
            leaf,
            dataset,
            sketches,
            self.context.edge_sink,
        )
    }

    fn build_leaf_profile_with_dataset_and_sink(
        &self,
        leaf: &[u32],
        dataset: &dyn PointStore,
        edge_sink: &dyn PendingEdgeSink,
    ) -> AnnResult<LeafProfile> {
        self.build_leaf_profile_with_dataset_sketches_and_sink(
            leaf,
            dataset,
            self.context.sketches,
            edge_sink,
        )
    }

    fn build_leaf_profile_with_dataset_sketches_and_sink(
        &self,
        leaf: &[u32],
        dataset: &dyn PointStore,
        sketches: &dyn SketchAccessor,
        edge_sink: &dyn PendingEdgeSink,
    ) -> AnnResult<LeafProfile> {
        self.check_for_error()?;
        #[cfg(test)]
        if let Some(observer) = self.context.observer {
            observer.on_leaf_start(leaf);
        }

        let hydrated = self
            .context
            .point_pipeline_config
            .filter(|config| {
                config.enabled
                    && leaf.len() >= config.min_leaf_points
                    && !dataset.is_resident_subset()
            })
            .map(|config| hydrate_resident_subset_with_stats(dataset, leaf, config))
            .transpose()?;
        let leaf_dataset: &dyn PointStore = hydrated
            .as_ref()
            .map(|hydrated| &hydrated.store as &dyn PointStore)
            .unwrap_or(dataset);
        let spine_overlay_sink;
        let view_lune_sink;
        let active_edge_sink: &dyn PendingEdgeSink = if let Some(recorder) =
            self.context.view_lune_recorder
        {
            view_lune_sink =
                ViewLuneTaggedEdgeSink::new(recorder, stable_leaf_route_family(leaf), edge_sink);
            &view_lune_sink
        } else if let Some(recorder) = self.context.spine_overlay_recorder {
            spine_overlay_sink = SpineOverlayTaggedEdgeSink::new(
                recorder,
                stable_leaf_route_family(leaf),
                edge_sink,
            );
            &spine_overlay_sink
        } else {
            edge_sink
        };

        let mut profile = if leaf.len() >= self.budget.large_leaf_min_size {
            self.large_leaf_count.fetch_add(1, Ordering::Relaxed);
            self.large_leaf_block_tasks.fetch_add(
                leaf.len().div_ceil(self.budget.leaf_block_rows),
                Ordering::Relaxed,
            );
            process_leaf_parallel_large_profiled_with_sink_with_ads_runtime(
                leaf_dataset,
                self.context.metric,
                sketches,
                leaf,
                self.context.params,
                active_edge_sink,
                self.budget.leaf_block_rows,
                Some(&self.leaf_ads_runtime),
            )?
        } else {
            with_thread_local_leaf_scratch(
                self.context.params.kernel_safe_leaf_size(),
                leaf_dataset.dim(),
                self.context.params.leaf_knn,
                |scratch| {
                    process_leaf_profiled_with_sink_with_ads_runtime(
                        leaf_dataset,
                        self.context.metric,
                        sketches,
                        leaf,
                        self.context.params,
                        active_edge_sink,
                        scratch,
                        Some(&self.leaf_ads_runtime),
                    )
                },
            )?
        };

        if let Some(hydrated) = hydrated {
            profile.load += hydrated.load;
            profile.io_stats.point_calls += hydrated.io_stats.point_calls;
            profile.io_stats.range_calls += hydrated.io_stats.range_calls;
            profile.io_stats.range_rows_read += hydrated.io_stats.range_rows_read;
            profile.io_stats.bytes_read += hydrated.io_stats.bytes_read;
            let _ = hydrated.pipeline_stats;
        }

        if !self.context.detailed_profiling {
            profile.total_wall = Duration::ZERO;
            profile.load = Duration::ZERO;
            profile.distance = Duration::ZERO;
            profile.topk = Duration::ZERO;
            profile.hash = Duration::ZERO;
            profile.flush = Duration::ZERO;
        }

        Ok(profile)
    }

    fn record_completed_leaf_profile(&self, profile: LeafProfile, leaf_count: usize) {
        self.leaf_profile.lock().merge(profile);
        let previous = self.progress.position();
        self.progress.inc(leaf_count as u64);
        let current = self.progress.position();
        if current / 1000 != previous / 1000 {
            self.progress.set_message(format!("Processed {}", current));
        }
    }
}

impl ForgeannScheduler<'_> {
    fn refresh_leaf_drainer_limits(&self, limits: &[LeafDrainerLimitEntry]) {
        let current = limits.iter().map(|entry| entry.limit).min().unwrap_or(0);
        let current_backpressure = limits
            .iter()
            .map(|entry| entry.backpressure_limit)
            .min()
            .unwrap_or(0);
        self.leaf_drainer_limit.store(current, Ordering::Release);
        self.leaf_drainer_backpressure_limit
            .store(current_backpressure, Ordering::Release);
    }
}

impl SchedulerSignals for ForgeannScheduler<'_> {
    fn begin_large_assignment(&self) {
        let active = self
            .active_large_assignment_count
            .fetch_add(1, Ordering::AcqRel)
            + 1;
        self.large_assignment_guard_count
            .fetch_add(1, Ordering::Relaxed);
        let mut peak = self.active_large_assignment_peak.load(Ordering::Acquire);
        while active > peak {
            match self.active_large_assignment_peak.compare_exchange_weak(
                peak,
                active,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(next) => peak = next,
            }
        }
    }

    fn finish_large_assignment(&self) {
        self.active_large_assignment_count
            .fetch_sub(1, Ordering::AcqRel);
    }

    fn begin_leaf_drainer_limit(&self, limit: usize, backpressure_limit: usize) {
        let limit = limit.max(1).min(self.max_leaf_worker_capacity());
        let backpressure_limit = backpressure_limit
            .max(1)
            .min(self.max_leaf_worker_capacity());
        let mut limits = self.leaf_drainer_limits.lock();
        limits.push(LeafDrainerLimitEntry {
            limit,
            backpressure_limit,
        });
        self.refresh_leaf_drainer_limits(&limits);
    }

    fn finish_leaf_drainer_limit(&self, limit: usize, backpressure_limit: usize) {
        let limit = limit.max(1).min(self.max_leaf_worker_capacity());
        let backpressure_limit = backpressure_limit
            .max(1)
            .min(self.max_leaf_worker_capacity());
        let mut limits = self.leaf_drainer_limits.lock();
        if let Some(position) = limits.iter().rposition(|&active| {
            active
                == LeafDrainerLimitEntry {
                    limit,
                    backpressure_limit,
                }
        }) {
            limits.remove(position);
        } else {
            limits.pop();
        }
        self.refresh_leaf_drainer_limits(&limits);
    }
}

pub(super) fn run_scheduler_scope<'a, F>(
    pool: &rayon::ThreadPool,
    budget: SchedulerBudget,
    context: LeafTaskContext<'a>,
    progress: &'a ProgressBar,
    producer: F,
) -> AnnResult<(LeafProfile, SchedulerTelemetry)>
where
    F: FnOnce(&dyn LeafEmitter) -> AnnResult<()> + Send,
{
    let scheduler = ForgeannScheduler::new(budget, context, progress);
    let scheduler_for_scope = Arc::clone(&scheduler);
    pool.scope_fifo(move |scope| {
        let emitter = ScopedLeafEmitter {
            scheduler: Arc::clone(&scheduler_for_scope),
            scope,
        };
        let result = producer(&emitter);
        scheduler_for_scope.mark_producer_finished();
        if result.is_err() {
            scheduler_for_scope.discard_queued_leaf_backlog();
        } else {
            scheduler_for_scope.maybe_spawn_leaf_drainer(scope);
        }
        let post_producer_leaf_backlog = scheduler_for_scope.leaf_backlog.len();
        let post_producer_drain_start = Instant::now();
        let idle_result = scheduler_for_scope.wait_until_idle();
        let post_producer_drain_wall = post_producer_drain_start.elapsed();
        let telemetry = scheduler_for_scope.telemetry();
        tracing::info!(
            "[forgeann/scheduler-post-producer-leaf-drain] leaf_backlog_start={} drain_ms={} active_leaf_drainers_end={} outstanding_leaf_tasks_end={} batched_leaf_drains={} batched_leaf_leaves={} batched_leaf_unique_points={} batched_leaf_load_ms={} leaf_batch_post_producer_scale={} peak_leaf_batch_leaves={} peak_leaf_batch_points={} peak_leaf_batch_unique_points={} leaf_batch_hydration_limit={} peak_leaf_batch_hydrations={} leaf_batch_hydration_wait_ms={}",
            post_producer_leaf_backlog,
            post_producer_drain_wall.as_millis(),
            telemetry.active_leaf_drainers,
            telemetry.outstanding_leaf_tasks,
            telemetry.batched_leaf_drains,
            telemetry.batched_leaf_leaves,
            telemetry.batched_leaf_unique_points,
            telemetry.batched_leaf_load_ms,
            telemetry.leaf_batch_post_producer_scale,
            telemetry.peak_leaf_batch_leaves,
            telemetry.peak_leaf_batch_points,
            telemetry.peak_leaf_batch_unique_points,
            telemetry.leaf_batch_hydration_limit,
            telemetry.peak_leaf_batch_hydrations,
            telemetry.leaf_batch_hydration_wait_ms,
        );
        result.and(idle_result)
    })?;

    Ok((scheduler.leaf_profile(), scheduler.telemetry()))
}

impl<'borrow, 'scope, 'a: 'scope> LeafEmitter for ScopedLeafEmitter<'borrow, 'scope, 'a> {
    fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.emit_leaf_with_optional_producer_wait(leaf, true)
    }

    fn emit_leaf_deferred(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.emit_leaf_with_optional_producer_wait(leaf, false)
    }

    fn emit_leaf_from_dataset(
        &self,
        dataset: &dyn PointStore,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        if !dataset.is_resident_subset() {
            return Ok(Some(leaf));
        }

        self.emit_leaf_inline_from_dataset(dataset, leaf)
    }

    fn emit_leaf_inline_from_dataset(
        &self,
        dataset: &dyn PointStore,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        self.scheduler
            .process_leaf_task_with_dataset(leaf, dataset)?;
        Ok(None)
    }

    fn emit_leaf_inline_from_dataset_with_sketches(
        &self,
        dataset: &dyn PointStore,
        sketches: &dyn SketchAccessor,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        self.scheduler
            .process_leaf_task_with_dataset_and_sketches(leaf, dataset, sketches)?;
        Ok(None)
    }

    fn emit_leaf_batch_inline_from_dataset(
        &self,
        dataset: &dyn PointStore,
        leaves: Vec<Vec<u32>>,
    ) -> AnnResult<()> {
        self.scheduler
            .process_leaf_batch_with_dataset(leaves, dataset)
    }

    fn wait_for_leaf_backlog_below(&self, target_backlog: usize) -> AnnResult<bool> {
        self.scheduler
            .wait_for_leaf_backlog_below(self.scope, target_backlog)
    }

    fn scheduler_signals(&self) -> Option<&dyn SchedulerSignals> {
        Some(self.scheduler.as_ref())
    }

    fn scheduler_telemetry(&self) -> Option<SchedulerTelemetry> {
        Some(self.scheduler.telemetry())
    }

    fn leaf_profile_snapshot(&self) -> Option<LeafProfile> {
        Some(self.scheduler.leaf_profile())
    }

    fn sketch_accessor(&self) -> Option<&dyn SketchAccessor> {
        Some(self.scheduler.context.sketches)
    }
}

impl<'borrow, 'scope, 'a: 'scope> ScopedLeafEmitter<'borrow, 'scope, 'a> {
    fn emit_leaf_with_optional_producer_wait(
        &self,
        leaf: Vec<u32>,
        wait_for_producer_backpressure: bool,
    ) -> AnnResult<()> {
        self.emit_leaf_with_optional_dataset_wait(leaf, None, wait_for_producer_backpressure)
    }

    fn emit_leaf_with_optional_dataset_wait(
        &self,
        leaf: Vec<u32>,
        dataset: Option<Arc<dyn PointStore>>,
        wait_for_producer_backpressure: bool,
    ) -> AnnResult<()> {
        self.scheduler.check_for_error()?;

        let mut queued = self.scheduler.make_queued_leaf(leaf, dataset);
        loop {
            match self.scheduler.try_enqueue_queued_leaf(queued) {
                Ok(()) => {
                    self.scheduler.maybe_spawn_leaf_drainer(self.scope);
                    if wait_for_producer_backpressure {
                        self.scheduler
                            .maybe_wait_for_producer_leaf_backlog_soft_limit(self.scope)?;
                        self.scheduler
                            .maybe_wait_for_work_graph_credit(self.scope)?;
                    }
                    return Ok(());
                }
                Err(next_queued) => {
                    queued = next_queued;
                    self.scheduler.maybe_spawn_leaf_drainer(self.scope);
                    // When the bounded backlog is full, free one slot inline before yielding.
                    // Using rayon::yield_now() as the primary backpressure path can execute more
                    // nested Rayon jobs on the current worker stack, which is exactly the
                    // pattern that triggered the observed RBC stack overflow.
                    if self.scheduler.try_help_from_backlog_recorded(true, true)? {
                        continue;
                    }

                    if rayon::yield_now().is_none() {
                        let idle_start = Instant::now();
                        self.scheduler
                            .producer_help_yields
                            .fetch_add(1, Ordering::Relaxed);
                        std::thread::yield_now();
                        self.scheduler
                            .worker_idle_ns
                            .fetch_add(duration_nanos_u64(idle_start.elapsed()), Ordering::Relaxed);
                    }
                }
            }
        }
    }
}

fn with_thread_local_leaf_scratch<R, F>(
    c_max: usize,
    dim: usize,
    leaf_knn: usize,
    f: F,
) -> AnnResult<R>
where
    F: FnOnce(&mut LeafScratch) -> AnnResult<R>,
{
    TLS_LEAF_SCRATCH.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            let mut scratch = LeafScratch::new(c_max, dim, leaf_knn);
            return f(&mut scratch);
        };

        if slot
            .as_ref()
            .is_none_or(|scratch| scratch.x.ncols() != dim || scratch.x.nrows() < c_max)
        {
            *slot = Some(LeafScratch::new(c_max, dim, leaf_knn));
        }
        f(slot.as_mut().expect("thread-local scratch initialized"))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use indicatif::ProgressBar;
    use parking_lot::Mutex;
    use rayon::prelude::*;

    use super::{
        ForgeannScheduler, LeafBatchDrainPolicy, LeafTaskContext, SchedulerBudget,
        SchedulerTelemetry, SchedulerTestObserver, leaf_batch_drain_policy_with_override,
        run_scheduler_scope,
    };
    use crate::common::Metric;
    use crate::forgeann::ForgeANNParams;
    use crate::forgeann::hash_prune::{HashPruneReservoir, SketchAccessor, SketchStore};
    use crate::forgeann::leaf_build::ReservoirEdgeSink;
    use crate::forgeann::point_pipeline::PointPipelineConfig;
    use crate::forgeann::point_store::{
        InmemDatasetPointStore, PointStore, ResidentSubsetPointStore,
    };
    use crate::model::InmemDataset;

    #[derive(Default)]
    struct RecordingObserver {
        starts: parking_lot::Mutex<Vec<u32>>,
    }

    impl RecordingObserver {
        fn started(&self) -> Vec<u32> {
            self.starts.lock().clone()
        }
    }

    impl SchedulerTestObserver for RecordingObserver {
        fn on_leaf_start(&self, leaf: &[u32]) {
            self.starts.lock().push(leaf[0]);
        }
    }

    struct PanicObserver;

    impl SchedulerTestObserver for PanicObserver {
        fn on_leaf_start(&self, _leaf: &[u32]) {
            panic!("forced scheduler test panic");
        }
    }

    fn build_test_context<'a>(
        dataset: &'a InmemDatasetPointStore<'a>,
        sketches: &'a dyn SketchAccessor,
        params: &'a ForgeANNParams,
        edge_sink: &'a dyn super::PendingEdgeSink,
        observer: Option<&'a dyn SchedulerTestObserver>,
    ) -> LeafTaskContext<'a> {
        LeafTaskContext {
            dataset,
            metric: Metric::L2,
            sketches,
            params,
            edge_sink,
            spine_overlay_recorder: None,
            view_lune_recorder: None,
            point_pipeline_config: None,
            detailed_profiling: false,
            observer,
        }
    }

    fn build_test_scheduler_inputs() -> (
        InmemDataset<f32>,
        SketchStore,
        ForgeANNParams,
        Vec<Mutex<HashPruneReservoir>>,
    ) {
        let dataset = InmemDataset::new(4, 1.0, 1).expect("dataset");
        let params = ForgeANNParams::default();
        let sketches =
            SketchStore::from_test_data(vec![0.0; 4 * params.m_hash_bits], params.m_hash_bits);
        let reservoirs = (0..4)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        (dataset, sketches, params, reservoirs)
    }

    fn single_thread_budget() -> SchedulerBudget {
        SchedulerBudget {
            worker_count: 1,
            leaf_backlog_capacity: 1,
            producer_leaf_backlog_soft_limit: 0,
            large_leaf_min_size: usize::MAX,
            leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
        }
    }

    fn producer_active_budget() -> SchedulerBudget {
        SchedulerBudget {
            worker_count: 2,
            leaf_backlog_capacity: 1,
            producer_leaf_backlog_soft_limit: 0,
            large_leaf_min_size: usize::MAX,
            leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
        }
    }

    #[test]
    fn backlog_full_drains_queued_leaf_before_current_leaf() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = RecordingObserver::default();
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let (profile, telemetry) = run_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, Some(&observer)),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![1])?;
                emitter.emit_leaf(vec![2])?;
                Ok(())
            },
        )
        .expect("scheduler should complete");

        assert_eq!(observer.started(), vec![1, 2]);
        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 2);
        assert_eq!(telemetry.peak_leaf_backlog, 1);
        assert_eq!(telemetry.inline_leaf_fallbacks, 0);
        assert_eq!(telemetry.producer_help_drains, 1);
        assert_eq!(
            telemetry.backlog_full_help_drains, 1,
            "backlog-full inline drain must be visible separately from other producer help drains"
        );
        assert_eq!(telemetry.producer_help_yields, 0);
    }

    #[test]
    fn scheduler_wait_until_idle_completes_backlog_without_deadlock() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = Arc::new(RecordingObserver::default());
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let (_profile, telemetry) = run_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context(
                &store,
                &sketches,
                &params,
                &edge_sink,
                Some(observer.as_ref()),
            ),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![1])?;
                emitter.emit_leaf(vec![2])?;
                emitter.emit_leaf(vec![3])?;
                Ok(())
            },
        )
        .expect("scheduler should drain backlog");

        assert_eq!(observer.started(), vec![1, 2, 3]);
        assert!(telemetry.peak_leaf_backlog >= 1);
    }

    #[test]
    fn work_graph_credit_backpressure_makes_producer_help_drain() {
        let (dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        params.leaf_ads_work_graph_enable = true;
        params.leaf_ads_work_graph_quantum_ms = 10;
        params.leaf_ads_work_graph_target_queue_ms = 1;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = Arc::new(RecordingObserver::default());
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let (_profile, telemetry) = run_scheduler_scope(
            &pool,
            SchedulerBudget {
                worker_count: 1,
                leaf_backlog_capacity: 8,
                producer_leaf_backlog_soft_limit: 0,
                large_leaf_min_size: usize::MAX,
                leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
            },
            build_test_context(
                &store,
                &sketches,
                &params,
                &edge_sink,
                Some(observer.as_ref()),
            ),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![1])?;
                emitter.emit_leaf(vec![2])?;
                emitter.emit_leaf(vec![3])?;
                Ok(())
            },
        )
        .expect("scheduler should drain with work graph credits");

        assert!(telemetry.work_graph_enabled);
        assert!(telemetry.queued_work_ms_peak >= params.leaf_ads_work_graph_quantum_ms);
        assert_eq!(telemetry.replay_admitted_runs, 3);
        assert!(
            telemetry.producer_help_drains > 0,
            "producer should help drain when queued work exceeds the work-graph credit target"
        );
        assert_eq!(telemetry.leaf_backlog, 0);
        assert_eq!(observer.started(), vec![1, 2, 3]);
    }

    #[test]
    fn producer_error_does_not_drain_queued_leaf_backlog() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = RecordingObserver::default();
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let result = run_scheduler_scope(
            &pool,
            SchedulerBudget {
                worker_count: 1,
                leaf_backlog_capacity: 8,
                producer_leaf_backlog_soft_limit: 0,
                large_leaf_min_size: usize::MAX,
                leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
            },
            build_test_context(&store, &sketches, &params, &edge_sink, Some(&observer)),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![1])?;
                emitter.emit_leaf(vec![2])?;
                emitter.emit_leaf(vec![3])?;
                Err(crate::common::AnnError::log_index_error(
                    "diagnostic stop".to_string(),
                ))
            },
        );

        assert!(result.is_err());
        assert_eq!(
            observer.started(),
            Vec::<u32>::new(),
            "diagnostic stops should not spend time draining leaf backlog that will be discarded"
        );
    }

    #[test]
    fn thread_local_leaf_scratch_falls_back_when_reentered() {
        let result = super::with_thread_local_leaf_scratch(4, 1, 1, |_| {
            super::with_thread_local_leaf_scratch(4, 1, 1, |_| Ok(()))
        });

        assert!(
            result.is_ok(),
            "nested leaf processing must not panic on TLS scratch reuse"
        );
    }

    #[test]
    fn panicking_leaf_task_releases_scheduler_counters() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = PanicObserver;
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, Some(&observer)),
            &progress,
        );

        scheduler.try_enqueue_leaf(vec![1]).unwrap();
        let leaf = scheduler.try_take_queued_leaf().unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scheduler.process_taken_leaf(leaf)
        }));

        assert!(result.is_err());
        assert_eq!(scheduler.outstanding_leaf_tasks.load(Ordering::Acquire), 0);
        assert_eq!(scheduler.inflight_leaf_tasks.load(Ordering::Acquire), 0);
    }

    #[test]
    fn producer_active_backlog_full_still_drains_inline() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = RecordingObserver::default();
        let progress = ProgressBar::hidden();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let (_profile, telemetry) = run_scheduler_scope(
            &pool,
            producer_active_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, Some(&observer)),
            &progress,
            |emitter| {
                emitter.emit_leaf(vec![1])?;
                emitter.emit_leaf(vec![2])?;
                Ok(())
            },
        )
        .expect("scheduler should complete");

        assert_eq!(observer.started(), vec![1, 2]);
        assert_eq!(
            telemetry.producer_help_drains, 1,
            "backlog-full producer path must free one slot inline instead of relying on Rayon yield"
        );
        assert_eq!(
            telemetry.backlog_full_help_drains, 1,
            "D2 diagnostics need to distinguish queue-full backpressure from other help drains"
        );
    }

    #[test]
    fn resident_subset_leaf_is_consumed_inline_without_global_backlog() {
        let (dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        params.m_hash_bits = 1;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let resident_store =
            ResidentSubsetPointStore::new(vec![0, 1, 2], store.dim(), vec![0.0, 1.0, 2.0])
                .expect("resident store");

        let (profile, telemetry) = run_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &ProgressBar::hidden(),
            |emitter| {
                assert!(
                    emitter
                        .emit_leaf_from_dataset(&resident_store, vec![0, 1, 2])?
                        .is_none(),
                    "resident D2 leaves should be consumed before their resident vectors are dropped"
                );
                Ok(())
            },
        )
        .expect("scheduler should complete");

        assert_eq!(profile.leaves, 1);
        assert_eq!(profile.points, 3);
        assert_eq!(telemetry.peak_leaf_backlog, 0);
        assert_eq!(telemetry.producer_help_drains, 0);
        assert_eq!(telemetry.backlog_full_help_drains, 0);
    }

    #[test]
    fn explicit_inline_leaf_from_nonresident_dataset_avoids_global_backlog() {
        let (dataset, sketches, mut params, reservoirs) = build_test_scheduler_inputs();
        params.m_hash_bits = 1;
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");

        let (profile, telemetry) = run_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &ProgressBar::hidden(),
            |emitter| {
                assert!(
                    emitter
                        .emit_leaf_inline_from_dataset(&store, vec![0, 1, 2])?
                        .is_none(),
                    "D1 materialize should be able to synchronously consume leaves even when the source dataset is not a resident subset"
                );
                Ok(())
            },
        )
        .expect("scheduler should complete");

        assert_eq!(profile.leaves, 1);
        assert_eq!(profile.points, 3);
        assert_eq!(telemetry.peak_leaf_backlog, 0);
        assert_eq!(telemetry.producer_help_drains, 0);
        assert_eq!(telemetry.backlog_full_help_drains, 0);
    }

    #[test]
    fn telemetry_defaults_remain_zero_without_work() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        assert_eq!(
            scheduler.telemetry(),
            SchedulerTelemetry {
                peak_inflight_leaf_tasks: 0,
                peak_leaf_backlog: 0,
                leaf_backlog: 0,
                producer_leaf_backlog_soft_limit: 0,
                outstanding_leaf_tasks: 0,
                inflight_leaf_tasks: 0,
                active_leaf_drainers: 0,
                leaf_drainer_limit: 0,
                leaf_drainer_backpressure_limit: 0,
                inline_leaf_fallbacks: 0,
                producer_help_drains: 0,
                producer_help_drain_ms: 0,
                backlog_full_help_drains: 0,
                backlog_full_help_ms: 0,
                producer_help_yields: 0,
                large_leaf_count: 0,
                large_leaf_block_tasks: 0,
                active_large_assignment_peak: 0,
                large_assignment_guard_count: 0,
                leaf_cap_large_assignment_hits: 0,
                leaf_cap_producer_hits: 0,
                leaf_cap_done_hits: 0,
                batched_leaf_drains: 0,
                batched_leaf_leaves: 0,
                batched_leaf_points: 0,
                batched_leaf_unique_points: 0,
                batched_leaf_load_ms: 0,
                leaf_batch_hydration_limit: 0,
                leaf_batch_post_producer_scale: 0,
                peak_leaf_batch_leaves: 0,
                peak_leaf_batch_points: 0,
                peak_leaf_batch_unique_points: 0,
                active_leaf_batch_hydrations: 0,
                peak_leaf_batch_hydrations: 0,
                leaf_batch_hydration_wait_ms: 0,
                work_graph_enabled: false,
                queued_work_ms_peak: 0,
                queued_mem_bytes_peak: 0,
                replay_admitted_runs: 0,
                replay_paused_ms: 0,
                worker_busy_ms: 0,
                worker_idle_ms: 0,
            }
        );
    }

    #[test]
    fn leaf_batch_drain_policy_defaults_to_batched_hydration() {
        let config = PointPipelineConfig {
            enabled: true,
            budget_bytes: 1024 * 1024,
            ..PointPipelineConfig::default()
        };
        let mut params = ForgeANNParams::production_sota_oom();

        assert_eq!(
            leaf_batch_drain_policy_with_override(
                &params,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&config),
                16,
                8,
            ),
            LeafBatchDrainPolicy {
                enabled: true,
                max_leaves: ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MAX_LEAVES,
                max_points: 16_384,
                min_backlog: ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MIN_BACKLOG,
                max_active_hydrations: 0,
                post_producer_scale: 2,
                post_producer_max_leaves: 32,
                post_producer_max_points: 16_384,
            }
        );

        params.leaf_batch_drain_enable = true;
        params.leaf_batch_drain_max_leaves = 8;
        params.leaf_batch_drain_max_points = 64;
        params.leaf_batch_drain_min_backlog = 3;
        assert_eq!(
            leaf_batch_drain_policy_with_override(
                &params,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&config),
                16,
                8,
            ),
            LeafBatchDrainPolicy {
                enabled: true,
                max_leaves: 8,
                max_points: 64,
                min_backlog: 3,
                max_active_hydrations: 0,
                post_producer_scale: 2,
                post_producer_max_leaves: 16,
                post_producer_max_points: 128,
            }
        );

        assert_eq!(
            leaf_batch_drain_policy_with_override(
                &params,
                Some(1),
                Some(4),
                Some(32),
                Some(2),
                Some(3),
                Some(4),
                Some(&config),
                16,
                8,
            ),
            LeafBatchDrainPolicy {
                enabled: true,
                max_leaves: 4,
                max_points: 32,
                min_backlog: 2,
                max_active_hydrations: 3,
                post_producer_scale: 4,
                post_producer_max_leaves: 16,
                post_producer_max_points: 128,
            }
        );
    }

    #[test]
    fn batched_leaf_drain_hydrates_union_and_processes_each_leaf() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let pipeline_config = PointPipelineConfig {
            enabled: true,
            budget_bytes: 1024 * 1024,
            min_leaf_points: 1,
            ..PointPipelineConfig::default()
        };
        let mut scheduler = ForgeannScheduler::new(
            SchedulerBudget {
                worker_count: 2,
                leaf_backlog_capacity: 8,
                producer_leaf_backlog_soft_limit: 0,
                large_leaf_min_size: usize::MAX,
                leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
            },
            LeafTaskContext {
                dataset: &store,
                metric: Metric::L2,
                sketches: &sketches,
                params: &params,
                edge_sink: &edge_sink,
                spine_overlay_recorder: None,
                view_lune_recorder: None,
                point_pipeline_config: Some(&pipeline_config),
                detailed_profiling: true,
                observer: None,
            },
            &progress,
        );
        Arc::get_mut(&mut scheduler)
            .expect("scheduler has no clones")
            .leaf_batch_drain_policy = LeafBatchDrainPolicy {
            enabled: true,
            max_leaves: 4,
            max_points: 16,
            min_backlog: 1,
            max_active_hydrations: 1,
            post_producer_scale: 1,
            post_producer_max_leaves: 4,
            post_producer_max_points: 16,
        };

        scheduler
            .process_leaf_batch(vec![vec![0, 1], vec![1, 2, 3]])
            .expect("batched leaf drain");

        let telemetry = scheduler.telemetry();
        assert_eq!(telemetry.batched_leaf_drains, 1);
        assert_eq!(telemetry.batched_leaf_leaves, 2);
        assert_eq!(telemetry.batched_leaf_points, 5);
        assert_eq!(telemetry.batched_leaf_unique_points, 4);
        assert_eq!(telemetry.peak_leaf_batch_leaves, 2);
        assert_eq!(telemetry.peak_leaf_batch_points, 5);
        assert_eq!(telemetry.peak_leaf_batch_unique_points, 4);
        assert_eq!(telemetry.peak_leaf_batch_hydrations, 1);
        let profile = scheduler.leaf_profile();
        assert_eq!(profile.leaves, 2);
        assert_eq!(profile.points, 5);
        assert!(profile.io_stats.bytes_read > 0);
    }

    #[test]
    fn producer_active_leaf_batch_uses_base_morsel_size() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let mut scheduler = ForgeannScheduler::new(
            SchedulerBudget {
                worker_count: 4,
                leaf_backlog_capacity: 16,
                producer_leaf_backlog_soft_limit: 0,
                large_leaf_min_size: usize::MAX,
                leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
            },
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );
        Arc::get_mut(&mut scheduler)
            .expect("scheduler has no clones")
            .leaf_batch_drain_policy = LeafBatchDrainPolicy {
            enabled: true,
            max_leaves: 2,
            max_points: 32,
            min_backlog: 1,
            max_active_hydrations: 0,
            post_producer_scale: 4,
            post_producer_max_leaves: 8,
            post_producer_max_points: 128,
        };

        for _ in 0..10 {
            scheduler.try_enqueue_leaf(vec![0]).unwrap();
        }

        let batch = scheduler.try_take_queued_leaf_batch().unwrap();
        assert_eq!(batch.len(), 2);
        scheduler.finish_leaf_executions(batch.len());
    }

    #[test]
    fn producer_finished_leaf_batch_uses_scaled_morsel_size() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let mut scheduler = ForgeannScheduler::new(
            SchedulerBudget {
                worker_count: 4,
                leaf_backlog_capacity: 16,
                producer_leaf_backlog_soft_limit: 0,
                large_leaf_min_size: usize::MAX,
                leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
            },
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );
        Arc::get_mut(&mut scheduler)
            .expect("scheduler has no clones")
            .leaf_batch_drain_policy = LeafBatchDrainPolicy {
            enabled: true,
            max_leaves: 2,
            max_points: 32,
            min_backlog: 1,
            max_active_hydrations: 0,
            post_producer_scale: 4,
            post_producer_max_leaves: 8,
            post_producer_max_points: 128,
        };

        for _ in 0..10 {
            scheduler.try_enqueue_leaf(vec![0]).unwrap();
        }
        scheduler.mark_producer_finished();

        let batch = scheduler.try_take_queued_leaf_batch().unwrap();
        assert_eq!(batch.len(), 8);
        scheduler.finish_leaf_executions(batch.len());
    }

    #[test]
    fn producer_help_from_backlog_uses_leaf_batch_morsel() {
        let dataset = InmemDataset::new(4, 1.0, 1).expect("dataset");
        let mut params = ForgeANNParams::default();
        params.leaf_batch_drain_enable = true;
        let sketches =
            SketchStore::from_test_data(vec![0.0; 4 * params.m_hash_bits], params.m_hash_bits);
        let reservoirs = (0..4)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect::<Vec<_>>();
        let store = InmemDatasetPointStore::new(&dataset, 1);
        let resident_store: Arc<dyn PointStore> = Arc::new(
            ResidentSubsetPointStore::new(vec![0, 1, 2, 3], 1, vec![0.0, 1.0, 2.0, 3.0])
                .expect("resident store"),
        );
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let observer = RecordingObserver::default();
        let progress = ProgressBar::hidden();
        let mut scheduler = ForgeannScheduler::new(
            SchedulerBudget {
                worker_count: 4,
                leaf_backlog_capacity: 8,
                producer_leaf_backlog_soft_limit: 0,
                large_leaf_min_size: usize::MAX,
                leaf_block_rows: SchedulerBudget::DEFAULT_LEAF_BLOCK_ROWS,
            },
            build_test_context(&store, &sketches, &params, &edge_sink, Some(&observer)),
            &progress,
        );
        Arc::get_mut(&mut scheduler)
            .expect("scheduler has no clones")
            .leaf_batch_drain_policy = LeafBatchDrainPolicy {
            enabled: true,
            max_leaves: 4,
            max_points: 32,
            min_backlog: 1,
            max_active_hydrations: 0,
            post_producer_scale: 1,
            post_producer_max_leaves: 4,
            post_producer_max_points: 32,
        };

        for point_id in 0..4 {
            let queued =
                scheduler.make_queued_leaf(vec![point_id], Some(Arc::clone(&resident_store)));
            scheduler.try_enqueue_queued_leaf(queued).unwrap();
        }

        assert!(
            scheduler.try_help_from_backlog(true).unwrap(),
            "producer help should consume one queued leaf morsel"
        );

        let mut started = observer.started();
        started.sort_unstable();
        assert_eq!(started, vec![0, 1, 2, 3]);
        let telemetry = scheduler.telemetry();
        assert_eq!(telemetry.producer_help_drains, 1);
        assert_eq!(telemetry.batched_leaf_drains, 1);
        assert_eq!(telemetry.batched_leaf_leaves, 4);
        assert_eq!(telemetry.outstanding_leaf_tasks, 0);
        assert_eq!(telemetry.inflight_leaf_tasks, 0);
    }

    #[test]
    fn telemetry_reports_live_backlog_and_inflight_leaf_counts() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        scheduler.try_enqueue_leaf(vec![1]).unwrap();
        let queued = scheduler.telemetry();
        assert_eq!(queued.leaf_backlog, 1);
        assert_eq!(queued.outstanding_leaf_tasks, 1);
        assert_eq!(queued.inflight_leaf_tasks, 0);

        let leaf = scheduler.try_take_queued_leaf().unwrap();
        let running = scheduler.telemetry();
        assert_eq!(running.leaf_backlog, 0);
        assert_eq!(running.outstanding_leaf_tasks, 1);
        assert_eq!(running.inflight_leaf_tasks, 1);

        scheduler.process_taken_leaf(leaf).unwrap();
        let drained = scheduler.telemetry();
        assert_eq!(drained.leaf_backlog, 0);
        assert_eq!(drained.outstanding_leaf_tasks, 0);
        assert_eq!(drained.inflight_leaf_tasks, 0);
    }

    #[test]
    fn producer_active_without_backlog_spawns_no_leaf_drainers() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        assert_eq!(scheduler.spawned_leaf_worker_limit_for_backlog(0), 0);
    }

    #[test]
    fn producer_active_with_backlog_keeps_workers_reserved_for_rbc() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let target = scheduler.spawned_leaf_worker_limit_for_backlog(64);
        assert!(target > 0, "expected some leaf workers once backlog exists");
        assert!(
            target < 42,
            "expected producer-active mode to reserve at least one worker for RBC, got {target}"
        );
    }

    #[test]
    fn producer_active_with_backlog_keeps_one_worker_reserved_for_rbc() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let target = scheduler.spawned_leaf_worker_limit_for_backlog(64);
        assert_eq!(
            target, 41,
            "expected producer-active mode to keep only one worker reserved for RBC/ADS wave"
        );
    }

    #[test]
    fn large_assignment_active_caps_leaf_drainers_to_two() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let _guard = scheduler.enter_large_assignment();

        assert_eq!(scheduler.spawned_leaf_worker_limit_for_backlog(64), 2);
        let telemetry = scheduler.telemetry();
        assert_eq!(telemetry.active_large_assignment_peak, 1);
        assert_eq!(telemetry.large_assignment_guard_count, 1);
        assert_eq!(telemetry.leaf_cap_large_assignment_hits, 1);
    }

    #[test]
    fn large_assignment_guard_drop_restores_producer_active_cap() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        {
            let _guard = scheduler.enter_large_assignment();
            assert_eq!(scheduler.spawned_leaf_worker_limit_for_backlog(64), 2);
        }

        assert_eq!(scheduler.spawned_leaf_worker_limit_for_backlog(64), 41);
    }

    #[test]
    fn leaf_drainer_limit_guard_caps_producer_active_drainers_and_restores_default_cap() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        {
            let _guard = scheduler.enter_leaf_drainer_limit(10);
            assert_eq!(
                scheduler.spawned_leaf_worker_limit_for_backlog(8_000),
                10,
                "child processing should leave most workers available while still draining leaves"
            );
            assert_eq!(scheduler.telemetry().leaf_drainer_limit, 10);
        }

        assert_eq!(scheduler.spawned_leaf_worker_limit_for_backlog(80_000), 41);
        assert_eq!(scheduler.telemetry().leaf_drainer_limit, 0);
    }

    #[test]
    fn leaf_drainer_limit_guard_uses_explicit_backpressure_limit_when_backlog_is_nearly_full() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let budget = SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size());
        let scheduler = ForgeannScheduler::new(
            budget,
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let _guard = scheduler.enter_leaf_drainer_limit_with_backpressure_limit(10, 24);
        let low_backlog = budget.leaf_backlog_capacity / 4;
        let high_backlog = budget.leaf_backlog_capacity * 9 / 10;

        assert_eq!(
            scheduler.spawned_leaf_worker_limit_for_backlog(low_backlog),
            10
        );
        assert_eq!(
            scheduler.spawned_leaf_worker_limit_for_backlog(high_backlog),
            24,
            "child-processing leaf guard should only relax to the explicit diagnostic backpressure limit"
        );
    }

    #[test]
    fn leaf_drainer_limit_guard_keeps_child_processing_cap_under_near_full_backlog_by_default() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let budget = SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size());
        let scheduler = ForgeannScheduler::new(
            budget,
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let _guard = scheduler.enter_leaf_drainer_limit(10);
        let high_backlog = budget.leaf_backlog_capacity * 9 / 10;

        assert_eq!(
            scheduler.spawned_leaf_worker_limit_for_backlog(high_backlog),
            10,
            "child processing should not automatically hand most workers back to leaf drainers when the backlog is near full"
        );
    }

    #[test]
    fn leaf_drainer_limit_guard_retires_excess_active_drainers() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let _guard = scheduler.enter_leaf_drainer_limit(10);
        scheduler.active_leaf_drainers.store(41, Ordering::Release);
        assert!(
            scheduler.try_reserve_leaf_drainer_retirement(80_000),
            "existing D1 leaf drainers should retire once child processing applies a lower cap"
        );

        scheduler.active_leaf_drainers.store(10, Ordering::Release);
        scheduler.retiring_leaf_drainers.store(0, Ordering::Release);
        assert!(!scheduler.try_reserve_leaf_drainer_retirement(80_000));
    }

    #[test]
    fn leaf_drainer_limit_guard_retires_only_excess_drainers() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let _guard = scheduler.enter_leaf_drainer_limit(10);
        scheduler.active_leaf_drainers.store(41, Ordering::Release);

        let retirements = (0..41)
            .filter(|_| scheduler.try_reserve_leaf_drainer_retirement(80_000))
            .count();

        assert_eq!(
            retirements, 31,
            "under near-full backlog pressure, active drainers above the strict D2 leaf-drainer cap should retire"
        );
        assert_eq!(scheduler.retiring_leaf_drainers.load(Ordering::Acquire), 31);
    }

    #[test]
    fn large_assignment_guard_drop_restores_after_panic() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = scheduler.enter_large_assignment();
            panic!("forced guard unwind");
        }));

        assert!(result.is_err());
        assert_eq!(scheduler.spawned_leaf_worker_limit_for_backlog(64), 41);
    }

    #[test]
    fn producer_finished_can_use_full_leaf_worker_capacity() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let scheduler = ForgeannScheduler::new(
            SchedulerBudget::for_worker_count(42, params.kernel_safe_leaf_size()),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
        );

        scheduler.mark_producer_finished();

        assert_eq!(
            scheduler.spawned_leaf_worker_limit_for_backlog(64),
            scheduler.budget.worker_count.saturating_sub(1)
        );
    }

    #[test]
    fn scheduler_budget_shrinks_backlog_under_tight_memory_budget() {
        let tight = SchedulerBudget::for_memory_budget(42, 3500, 512 * 1024);
        let relaxed = SchedulerBudget::for_memory_budget(42, 3500, usize::MAX);

        assert!(tight.leaf_backlog_capacity < relaxed.leaf_backlog_capacity);
        assert!(tight.leaf_backlog_capacity >= 1);
    }

    #[test]
    fn scheduler_budget_uses_more_backlog_headroom_for_many_workers() {
        let relaxed = SchedulerBudget::for_worker_count(42, 3500);

        assert!(
            relaxed.leaf_backlog_capacity >= 80_000,
            "expected the backlog to absorb a substantial D1 leaf burst"
        );
    }

    #[test]
    fn scheduler_scope_runs_producer_inside_provided_rayon_pool() {
        let (dataset, sketches, params, reservoirs) = build_test_scheduler_inputs();
        let store = InmemDatasetPointStore::new(&dataset, 4);
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let progress = ProgressBar::hidden();
        let seen_threads = Arc::new(Mutex::new(HashSet::new()));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .thread_name(|idx| format!("scheduler-test-{idx}"))
            .build()
            .expect("pool");

        let seen_threads_for_producer = Arc::clone(&seen_threads);
        run_scheduler_scope(
            &pool,
            single_thread_budget(),
            build_test_context(&store, &sketches, &params, &edge_sink, None),
            &progress,
            move |_emitter| {
                (0..128usize).into_par_iter().for_each(|_| {
                    let name = std::thread::current()
                        .name()
                        .unwrap_or("<unnamed>")
                        .to_string();
                    seen_threads_for_producer.lock().insert(name);
                });
                Ok(())
            },
        )
        .expect("scheduler should complete");

        let seen_threads = seen_threads.lock();
        assert!(
            !seen_threads.is_empty(),
            "expected producer to execute parallel work"
        );
        assert!(
            seen_threads
                .iter()
                .all(|name| name.starts_with("scheduler-test-")),
            "expected nested parallel work to stay inside the provided pool, saw {seen_threads:?}"
        );
    }
}
