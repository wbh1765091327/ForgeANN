use std::collections::BTreeMap;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crossbeam::channel;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::params::ForgeANNParams;
use super::point_pipeline::PointPipelineStats;
use super::point_store::{PointBatchStats, PointStore, WindowedGatherOptions};
use crate::common::{AnnError, AnnResult};

const ADS_LEADER_BLOCK: usize = 64;
const ADS_QUERY_CHUNK: usize = 4096;
const ADS_QUERY_TILE: usize = 8;
const ADS_LEAF_ROW_CHUNK: usize = 32;
const ADS_LEAF_PARALLEL_MIN_ROWS: usize = 512;
const ADS_ASSIGN_MIN_CHUNK_ROWS: usize = 256;
const ADS_ASSIGN_MAX_CHUNK_ROWS: usize = ADS_QUERY_CHUNK * 2;
pub(crate) const LEAF_ADS_EWMA_BUCKETS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LeafAdsOperatorConfig {
    pub(crate) enabled: bool,
    pub(crate) work_graph: bool,
    pub(crate) cpu_budget: usize,
    pub(crate) target_tile_ms: u64,
    pub(crate) min_tile_rows: usize,
    pub(crate) max_tile_rows: usize,
    pub(crate) split_threshold: usize,
}

impl LeafAdsOperatorConfig {
    pub(crate) fn from_params(params: &ForgeANNParams, worker_count: usize) -> Self {
        let worker_count = worker_count.max(1);
        let auto_reserved = (worker_count / 8)
            .max(2)
            .min(worker_count.saturating_sub(1));
        let auto_budget = worker_count.saturating_sub(auto_reserved).max(1);
        let cpu_budget = if params.leaf_ads_cpu_budget == 0 {
            auto_budget
        } else {
            params.leaf_ads_cpu_budget.min(worker_count).max(1)
        };
        let work_graph = params.leaf_ads_work_graph_enable;
        Self {
            enabled: params.leaf_ads_tiling_enable || work_graph,
            work_graph,
            cpu_budget,
            target_tile_ms: if work_graph {
                params.leaf_ads_work_graph_quantum_ms.max(1)
            } else {
                params.leaf_ads_target_tile_ms.max(1)
            },
            min_tile_rows: align_leaf_ads_tile_rows(if work_graph {
                params.leaf_ads_work_graph_min_tile_rows.max(1)
            } else {
                params.leaf_ads_min_tile_rows.max(1)
            }),
            max_tile_rows: align_leaf_ads_tile_rows(if work_graph {
                params.leaf_ads_work_graph_max_tile_rows.max(1)
            } else {
                params.leaf_ads_max_tile_rows.max(1)
            }),
            split_threshold: if work_graph {
                params.leaf_ads_work_graph_split_threshold
            } else {
                params.leaf_ads_split_threshold
            }
            .max(ForgeANNParams::LEAF_ADSAMPLING_MIN_SIZE),
        }
        .normalized()
    }

    fn normalized(mut self) -> Self {
        if self.min_tile_rows > self.max_tile_rows {
            std::mem::swap(&mut self.min_tile_rows, &mut self.max_tile_rows);
        }
        self
    }
}

#[derive(Debug)]
pub(crate) struct LeafAdsOperatorRuntime {
    config: LeafAdsOperatorConfig,
    active_contexts: AtomicUsize,
    active_context_peak: AtomicUsize,
    active_workers: AtomicUsize,
    active_workers_peak: AtomicUsize,
    ewma_ns_per_row_by_bucket: [AtomicU64; LEAF_ADS_EWMA_BUCKETS],
}

impl LeafAdsOperatorRuntime {
    pub(crate) fn from_params(params: &ForgeANNParams, worker_count: usize) -> Self {
        Self::new(LeafAdsOperatorConfig::from_params(params, worker_count))
    }

    pub(crate) fn new(config: LeafAdsOperatorConfig) -> Self {
        Self {
            config,
            active_contexts: AtomicUsize::new(0),
            active_context_peak: AtomicUsize::new(0),
            active_workers: AtomicUsize::new(0),
            active_workers_peak: AtomicUsize::new(0),
            ewma_ns_per_row_by_bucket: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    #[inline]
    pub(crate) fn enabled_for_leaf(&self, leaf_size: usize) -> bool {
        self.config.enabled && leaf_size >= self.config.split_threshold
    }

    #[inline]
    pub(crate) fn cpu_budget(&self) -> usize {
        self.config.cpu_budget
    }

    #[inline]
    pub(crate) fn work_graph_enabled(&self) -> bool {
        self.config.enabled && self.config.work_graph
    }

    #[inline]
    pub(crate) fn target_tile_ms(&self) -> u64 {
        self.config.target_tile_ms
    }

    pub(crate) fn ewma_snapshot(&self) -> [u64; LEAF_ADS_EWMA_BUCKETS] {
        std::array::from_fn(|idx| self.ewma_ns_per_row_by_bucket[idx].load(Ordering::Relaxed))
    }

    fn begin_context(&self) -> LeafAdsContextGuard<'_> {
        let active = self.active_contexts.fetch_add(1, Ordering::AcqRel) + 1;
        update_atomic_max(&self.active_context_peak, active);
        LeafAdsContextGuard { runtime: self }
    }

    fn reserve_workers(&self, requested: usize) -> (LeafAdsWorkerReservation<'_>, Duration) {
        let requested = requested.max(1).min(self.config.cpu_budget.max(1));
        let wait_start = Instant::now();
        let mut spins = 0usize;
        loop {
            let active = self.active_workers.load(Ordering::Acquire);
            if active < self.config.cpu_budget {
                let grant = requested.min(self.config.cpu_budget - active).max(1);
                match self.active_workers.compare_exchange_weak(
                    active,
                    active + grant,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        update_atomic_max(&self.active_workers_peak, active + grant);
                        return (
                            LeafAdsWorkerReservation {
                                runtime: self,
                                count: grant,
                            },
                            wait_start.elapsed(),
                        );
                    }
                    Err(_) => continue,
                }
            }
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else if spins < 256 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    fn worker_cap_for_current_contexts(&self) -> usize {
        let active_contexts = self.active_contexts.load(Ordering::Acquire).max(1);
        if active_contexts == 1 {
            self.config.cpu_budget.max(1)
        } else {
            (self.config.cpu_budget / active_contexts)
                .max(4)
                .min(self.config.cpu_budget.max(1))
        }
    }

    fn tile_rows_for_bucket(&self, bucket: usize) -> usize {
        let config = self.config;
        let ewma = self.ewma_ns_per_row_by_bucket[bucket].load(Ordering::Relaxed);
        let rows = if ewma == 0 {
            256
        } else {
            let target_ns = config.target_tile_ms.saturating_mul(1_000_000);
            (target_ns / ewma.max(1)) as usize
        };
        align_leaf_ads_tile_rows(rows.clamp(config.min_tile_rows, config.max_tile_rows))
    }

    fn record_tile(&self, bucket: usize, rows: usize, elapsed: Duration) {
        if rows == 0 {
            return;
        }
        let sample = (elapsed.as_nanos() / rows as u128)
            .min(u128::from(u64::MAX))
            .max(1) as u64;
        let slot = &self.ewma_ns_per_row_by_bucket[bucket];
        let mut old = slot.load(Ordering::Acquire);
        loop {
            let next = if old == 0 {
                sample
            } else {
                old.saturating_mul(7).saturating_add(sample) / 8
            };
            match slot.compare_exchange_weak(old, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return,
                Err(current) => old = current,
            }
        }
    }

    fn active_context_peak(&self) -> usize {
        self.active_context_peak.load(Ordering::Relaxed)
    }

    fn active_workers_peak(&self) -> usize {
        self.active_workers_peak.load(Ordering::Relaxed)
    }
}

struct LeafAdsContextGuard<'a> {
    runtime: &'a LeafAdsOperatorRuntime,
}

impl Drop for LeafAdsContextGuard<'_> {
    fn drop(&mut self) {
        self.runtime.active_contexts.fetch_sub(1, Ordering::AcqRel);
    }
}

struct LeafAdsWorkerReservation<'a> {
    runtime: &'a LeafAdsOperatorRuntime,
    count: usize,
}

impl Drop for LeafAdsWorkerReservation<'_> {
    fn drop(&mut self) {
        self.runtime
            .active_workers
            .fetch_sub(self.count, Ordering::AcqRel);
    }
}

#[inline]
pub(crate) fn leaf_ads_size_bucket(leaf_size: usize) -> usize {
    if leaf_size < 1024 {
        0
    } else if leaf_size < 2048 {
        1
    } else if leaf_size < 4096 {
        2
    } else {
        3
    }
}

#[inline]
fn align_leaf_ads_tile_rows(rows: usize) -> usize {
    rows.max(1).div_ceil(ADS_QUERY_TILE) * ADS_QUERY_TILE
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AdSamplingConfig {
    pub(crate) epsilon: f32,
    pub(crate) group_dims: usize,
    pub(crate) seed_exact: usize,
    pub(crate) sparse_full64_threshold: u32,
    pub(crate) sparse_group16_threshold: u32,
    pub(crate) validate_sources: usize,
}

impl AdSamplingConfig {
    pub(crate) fn root_from_params(params: &ForgeANNParams, fanout: usize) -> Self {
        Self {
            epsilon: env_f32("FORGEANN_D0_ADS_EPSILON")
                .or_else(|| env_f32("FORGEANN_ADS_EPSILON"))
                .unwrap_or(params.adsampling_epsilon),
            group_dims: env_usize("FORGEANN_D0_ADS_GROUP_DIMS")
                .or_else(|| env_usize("FORGEANN_ADS_GROUP_DIMS"))
                .unwrap_or(params.adsampling_group_dims)
                .max(1),
            seed_exact: env_usize("FORGEANN_D0_ADS_SEED_EXACT")
                .or_else(|| env_usize("FORGEANN_ADS_SEED_EXACT"))
                .unwrap_or(ForgeANNParams::ADS_ROOT_SEED_EXACT_M)
                .max(fanout)
                .max(1),
            sparse_full64_threshold: ForgeANNParams::ADS_SPARSE_FULL64_THRESHOLD,
            sparse_group16_threshold: ForgeANNParams::ADS_SPARSE_GROUP16_THRESHOLD,
            validate_sources: env_usize("FORGEANN_D0_ADS_VALIDATE_SOURCES")
                .or_else(|| env_usize("FORGEANN_ADS_VALIDATE_SOURCES"))
                .unwrap_or(0),
        }
    }

    pub(crate) fn depth_from_params(params: &ForgeANNParams, fanout: usize) -> Self {
        Self {
            epsilon: env_f32("FORGEANN_D1_ADS_EPSILON")
                .or_else(|| env_f32("FORGEANN_ADS_EPSILON"))
                .unwrap_or(params.adsampling_epsilon),
            group_dims: env_usize("FORGEANN_D1_ADS_GROUP_DIMS")
                .or_else(|| env_usize("FORGEANN_ADS_GROUP_DIMS"))
                .unwrap_or(params.adsampling_group_dims)
                .max(1),
            seed_exact: ForgeANNParams::ADS_DEPTH_SEED_EXACT_M.max(fanout).max(1),
            sparse_full64_threshold: ForgeANNParams::ADS_SPARSE_FULL64_THRESHOLD,
            sparse_group16_threshold: ForgeANNParams::ADS_SPARSE_GROUP16_THRESHOLD,
            validate_sources: 0,
        }
    }

    pub(crate) fn leaf_from_params(params: &ForgeANNParams, k: usize) -> Self {
        Self {
            epsilon: env_f32("FORGEANN_LEAF_ADS_EPSILON").unwrap_or(params.adsampling_epsilon),
            group_dims: env_usize("FORGEANN_LEAF_ADS_GROUP_DIMS")
                .unwrap_or(params.adsampling_group_dims)
                .max(1),
            seed_exact: ForgeANNParams::LEAF_ADSAMPLING_SEED_EXACT_M.max(k).max(1),
            sparse_full64_threshold: ForgeANNParams::ADS_SPARSE_FULL64_THRESHOLD,
            sparse_group16_threshold: ForgeANNParams::ADS_SPARSE_GROUP16_THRESHOLD,
            validate_sources: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AdSamplingRatios {
    dim: usize,
    group_dims: usize,
    ratios: Vec<f32>,
}

impl AdSamplingRatios {
    pub(crate) fn new(dim: usize, epsilon: f32, group_dims: usize) -> AnnResult<Self> {
        if dim == 0 || group_dims == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(AnnError::log_index_config_error(
                "adsampling".to_string(),
                format!(
                    "invalid ADSampling config: dim={dim} epsilon={epsilon} group_dims={group_dims}"
                ),
            ));
        }
        let ratios = (0..=dim)
            .map(|visited| pdx_adsampling_ratio(dim, epsilon, visited))
            .collect();
        Ok(Self {
            dim,
            group_dims,
            ratios,
        })
    }

    #[inline]
    pub(crate) fn ratio_after_visited(&self, visited: usize) -> f32 {
        self.ratios[visited.min(self.dim)]
    }

    #[inline]
    pub(crate) fn group_dims(&self) -> usize {
        self.group_dims
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
}

fn d0_validation_stride() -> usize {
    static STRIDE: OnceLock<usize> = OnceLock::new();
    *STRIDE.get_or_init(|| {
        env_usize("FORGEANN_D0_ADS_VALIDATE_STRIDE")
            .or_else(|| env_usize("FORGEANN_ADS_VALIDATE_STRIDE"))
            .unwrap_or(1)
            .max(1)
    })
}

fn d0_validation_offset() -> usize {
    static OFFSET: OnceLock<usize> = OnceLock::new();
    *OFFSET.get_or_init(|| {
        env_usize("FORGEANN_D0_ADS_VALIDATE_OFFSET")
            .or_else(|| env_usize("FORGEANN_ADS_VALIDATE_OFFSET"))
            .unwrap_or(0)
    })
}

#[inline]
fn should_validate_source_with(
    sample_limit: usize,
    sample_stride: usize,
    sample_offset: usize,
    source_id: usize,
) -> bool {
    if sample_limit == 0 || sample_stride == 0 || source_id < sample_offset {
        return false;
    }
    let delta = source_id - sample_offset;
    delta % sample_stride == 0 && delta / sample_stride < sample_limit
}

#[inline]
fn should_validate_d0_source(sample_limit: usize, source_id: usize) -> bool {
    should_validate_source_with(
        sample_limit,
        d0_validation_stride(),
        d0_validation_offset(),
        source_id,
    )
}

fn adsampling_windowed_options(dataset: &dyn PointStore, rows: usize) -> WindowedGatherOptions {
    let row_bytes = dataset.dim().saturating_mul(size_of::<f32>());
    let mut max_gap_rows = if row_bytes <= 4096 { 1 } else { 0 };
    let mut max_window_bytes = row_bytes
        .saturating_mul(rows.max(1))
        .min(4 * 1024 * 1024)
        .max(256 * 1024);
    if dataset.prefers_coalesced_window_reads() {
        max_window_bytes = env_usize("FORGEANN_ADSAMPLING_MAX_WINDOW_BYTES")
            .or_else(|| env_usize("FORGEANN_STRICT_GATHER_MAX_WINDOW_BYTES"))
            .unwrap_or(max_window_bytes)
            .max(max_window_bytes);
        let window_rows = (max_window_bytes / row_bytes.max(1)).max(1);
        let default_gap_rows = window_rows.saturating_sub(1).min(u32::MAX as usize);
        max_gap_rows = env_usize("FORGEANN_ADSAMPLING_MAX_GAP_ROWS")
            .or_else(|| env_usize("FORGEANN_STRICT_GATHER_MAX_GAP_ROWS"))
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(default_gap_rows as u32);
    }
    WindowedGatherOptions {
        max_gap_rows,
        max_window_bytes,
        alignment_bytes: 4096,
        sort_ids: true,
    }
}

#[inline]
fn pdx_adsampling_ratio(dim: usize, epsilon: f32, visited: usize) -> f32 {
    if visited == 0 || visited >= dim {
        return 1.0;
    }
    let visited = visited as f32;
    visited / dim as f32 * (1.0 + epsilon / visited.sqrt()).powi(2)
}

impl AdSamplingLeaderLayout {
    pub(crate) fn build(dataset: &dyn PointStore, leaders: &[u32]) -> AnnResult<Self> {
        let dim = dataset.dim();
        let mut row_major = vec![0.0f32; leaders.len().saturating_mul(dim)];
        let read_options = adsampling_windowed_options(dataset, leaders.len());
        let mut io_stats = PointBatchStats::default();
        dataset.read_points_windowed_into_batch_stats(
            leaders,
            &mut row_major,
            &read_options,
            &mut io_stats,
        )?;
        Ok(Self::from_row_major(row_major, leaders.len(), dim))
    }

    fn from_row_major(row_major: Vec<f32>, leaders: usize, dim: usize) -> Self {
        debug_assert_eq!(row_major.len(), leaders.saturating_mul(dim));
        let mut blocks = Vec::with_capacity(leaders.div_ceil(ADS_LEADER_BLOCK));
        for start in (0..leaders).step_by(ADS_LEADER_BLOCK) {
            let len = (leaders - start).min(ADS_LEADER_BLOCK);
            blocks.push(AdSamplingLeaderBlock::build(&row_major, dim, start, len));
        }
        Self {
            row_major,
            blocks,
            dim,
            leaders,
        }
    }

    #[cfg(test)]
    fn from_row_major_for_test(row_major: Vec<f32>, leaders: usize, dim: usize) -> Self {
        Self::from_row_major(row_major, leaders, dim)
    }

    #[inline]
    fn leader(&self, leader_idx: usize) -> &[f32] {
        let start = leader_idx * self.dim;
        &self.row_major[start..start + self.dim]
    }
}

impl AdSamplingLeaderBlock {
    fn build(row_major: &[f32], dim: usize, start: usize, len: usize) -> Self {
        let stride = ADS_LEADER_BLOCK;
        let mut data = vec![0.0f32; stride.saturating_mul(dim)];
        for local in 0..len {
            let row_start = (start + local) * dim;
            for dim_idx in 0..dim {
                data[dim_idx * stride + local] = row_major[row_start + dim_idx];
            }
        }
        Self {
            data,
            start,
            len,
            stride,
        }
    }
}

#[derive(Clone, Debug)]
struct AdSamplingLeafLayout<'a> {
    row_major: &'a [f32],
    blocks: Vec<AdSamplingLeaderBlock>,
    dim: usize,
    rows: usize,
}

impl<'a> AdSamplingLeafLayout<'a> {
    fn build(row_major: &'a [f32], rows: usize, dim: usize) -> Self {
        debug_assert_eq!(row_major.len(), rows.saturating_mul(dim));
        let mut blocks = Vec::with_capacity(rows.div_ceil(ADS_LEADER_BLOCK));
        for start in (0..rows).step_by(ADS_LEADER_BLOCK) {
            let len = (rows - start).min(ADS_LEADER_BLOCK);
            blocks.push(AdSamplingLeaderBlock::build(row_major, dim, start, len));
        }
        Self {
            row_major,
            blocks,
            dim,
            rows,
        }
    }

    #[inline]
    fn row(&self, row_idx: usize) -> &'a [f32] {
        let start = row_idx * self.dim;
        &self.row_major[start..start + self.dim]
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct AdSamplingProfile {
    pub(crate) depth: usize,
    pub(crate) points: usize,
    pub(crate) leaders: usize,
    pub(crate) fanout: usize,
    pub(crate) epsilon: f32,
    pub(crate) group_dims: usize,
    pub(crate) seed_exact_m: usize,
    pub(crate) total_ms: f64,
    pub(crate) layout_ms: f64,
    pub(crate) seed_ms: f64,
    pub(crate) seed_accumulated_ms: f64,
    pub(crate) seed_wall_ms: f64,
    pub(crate) scan_ms: f64,
    pub(crate) scan_accumulated_ms: f64,
    pub(crate) chunks: usize,
    pub(crate) called_inside_rayon_worker: bool,
    pub(crate) chunk_wall_accumulated_ms: f64,
    pub(crate) chunk_wall_max_ms: f64,
    pub(crate) chunk_send_block_ms: f64,
    pub(crate) chunk_active_max: usize,
    pub(crate) chunk_active_start_avg: f64,
    pub(crate) recv_wait_ms: f64,
    pub(crate) recv_empty_polls: u64,
    pub(crate) ordered_pending_max: usize,
    pub(crate) read_ms: f64,
    pub(crate) compute_ms: f64,
    pub(crate) apply_ms: f64,
    pub(crate) loaded_queue_depth_max: usize,
    pub(crate) computed_queue_depth_max: usize,
    pub(crate) visit_chunk_ms: f64,
    pub(crate) scheduler_mode: String,
    pub(crate) effective_parallelism: f64,
    pub(crate) fallback_reason: Option<String>,
    pub(crate) full_evals: u64,
    pub(crate) pruned_evals: u64,
    pub(crate) group_evals: u64,
    pub(crate) simd_group_calls: u64,
    pub(crate) simd_active_lane_evals: u64,
    pub(crate) scalar_group_evals: u64,
    pub(crate) validation_sources: usize,
    pub(crate) validation_mismatches: usize,
    pub(crate) validation_recall_hits: u64,
    pub(crate) validation_recall_total: u64,
    pub(crate) validation_recall_at_fanout: f64,
    #[serde(skip)]
    pub(crate) point_pipeline: PointPipelineStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AdSamplingTaskClass {
    pub(crate) large_ads: bool,
    pub(crate) huge_ads: bool,
}

#[derive(Debug)]
pub(crate) struct AdSamplingPointAssignmentChunk {
    pub(crate) source_start: usize,
    pub(crate) leaders_by_point: Vec<Vec<usize>>,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct AdSamplingOutcome {
    pub(crate) leaders: Vec<usize>,
    pub(crate) full_evals: u64,
    pub(crate) pruned_evals: u64,
    pub(crate) group_evals: u64,
    pub(crate) simd_group_calls: u64,
    pub(crate) simd_active_lane_evals: u64,
    pub(crate) scalar_group_evals: u64,
    pub(crate) validation_mismatch: bool,
    pub(crate) validation_recall_hits: u64,
    pub(crate) validation_recall_total: u64,
    pub(crate) validation_recall_at_fanout: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct AdSamplingLeaderLayout {
    row_major: Vec<f32>,
    blocks: Vec<AdSamplingLeaderBlock>,
    dim: usize,
    leaders: usize,
}

#[derive(Clone, Debug)]
struct AdSamplingLeaderBlock {
    data: Vec<f32>,
    start: usize,
    len: usize,
    stride: usize,
}

#[derive(Debug)]
pub(crate) struct AdSamplingChunkResult {
    pub(crate) chunk: AdSamplingPointAssignmentChunk,
    pub(crate) seed: Duration,
    pub(crate) scan: Duration,
    pub(crate) full_evals: u64,
    pub(crate) pruned_evals: u64,
    pub(crate) group_evals: u64,
    pub(crate) simd_group_calls: u64,
    pub(crate) simd_active_lane_evals: u64,
    pub(crate) scalar_group_evals: u64,
    pub(crate) validation_sources: usize,
    pub(crate) validation_mismatches: usize,
    pub(crate) validation_recall_hits: u64,
    pub(crate) validation_recall_total: u64,
}

#[derive(Debug)]
struct AdSamplingChunkMessage {
    chunk_idx: usize,
    chunk: AnnResult<AdSamplingChunkResult>,
    chunk_wall: Duration,
}

#[derive(Debug)]
struct AdSamplingTileResult {
    leaders_by_point: Vec<Vec<usize>>,
    seed: Duration,
    scan: Duration,
    full_evals: u64,
    pruned_evals: u64,
    group_evals: u64,
    simd_group_calls: u64,
    simd_active_lane_evals: u64,
    scalar_group_evals: u64,
    validation_sources: usize,
    validation_mismatches: usize,
    validation_recall_hits: u64,
    validation_recall_total: u64,
}

impl AdSamplingTileResult {
    fn new(points: usize) -> Self {
        Self {
            leaders_by_point: Vec::with_capacity(points),
            seed: Duration::ZERO,
            scan: Duration::ZERO,
            full_evals: 0,
            pruned_evals: 0,
            group_evals: 0,
            simd_group_calls: 0,
            simd_active_lane_evals: 0,
            scalar_group_evals: 0,
            validation_sources: 0,
            validation_mismatches: 0,
            validation_recall_hits: 0,
            validation_recall_total: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TinyTopK {
    items: [(f32, u32); 32],
    len: usize,
    k: usize,
}

impl TinyTopK {
    fn new(k: usize) -> Self {
        Self {
            items: [(f32::MAX, 0); 32],
            len: 0,
            k: k.clamp(1, 32),
        }
    }

    #[inline]
    fn push(&mut self, dist: f32, idx: usize) {
        debug_assert!(u32::try_from(idx).is_ok());
        let idx = idx as u32;
        if self.items[..self.len]
            .iter()
            .any(|&(_, existing)| existing == idx)
        {
            return;
        }
        if self.len < self.k {
            self.items[self.len] = (dist, idx);
            self.len += 1;
            self.sift_up(self.len - 1);
        } else if dist < self.threshold() {
            self.items[self.k - 1] = (dist, idx);
            self.sift_up(self.k - 1);
        }
    }

    #[inline]
    fn push_unique(&mut self, dist: f32, idx: usize) {
        debug_assert!(u32::try_from(idx).is_ok());
        let idx = idx as u32;
        debug_assert!(
            !self.items[..self.len]
                .iter()
                .any(|&(_, existing)| existing == idx)
        );
        if self.len < self.k {
            self.items[self.len] = (dist, idx);
            self.len += 1;
            self.sift_up(self.len - 1);
        } else if dist < self.threshold() {
            self.items[self.k - 1] = (dist, idx);
            self.sift_up(self.k - 1);
        }
    }

    #[inline]
    fn threshold(&self) -> f32 {
        if self.len < self.k {
            f32::MAX
        } else {
            self.items[self.k - 1].0
        }
    }

    #[inline]
    fn leaders(&self) -> Vec<usize> {
        self.items[..self.len]
            .iter()
            .map(|&(_, idx)| idx as usize)
            .collect()
    }

    #[inline]
    fn entries(&self) -> Vec<(usize, f32)> {
        self.items[..self.len]
            .iter()
            .map(|&(dist, idx)| (idx as usize, dist))
            .collect()
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 && self.items[idx].0 < self.items[idx - 1].0 {
            self.items.swap(idx, idx - 1);
            idx -= 1;
        }
    }
}

#[inline]
fn l2_sq(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if left.len() == right.len() && adsampling_avx512_available() {
            return unsafe { l2_sq_avx512(left, right) };
        }
    }
    l2_sq_scalar(left, right)
}

#[inline]
fn l2_sq_scalar(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| {
            let diff = left - right;
            diff * diff
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_sq_avx512(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(left.len(), right.len());

    let mut acc = _mm512_setzero_ps();
    let simd_len = left.len() / 16 * 16;
    for offset in (0..simd_len).step_by(16) {
        let left = unsafe { _mm512_loadu_ps(left.as_ptr().add(offset)) };
        let right = unsafe { _mm512_loadu_ps(right.as_ptr().add(offset)) };
        let diff = _mm512_sub_ps(left, right);
        acc = _mm512_fmadd_ps(diff, diff, acc);
    }

    let mut sum = _mm512_reduce_add_ps(acc);
    for offset in simd_len..left.len() {
        let diff = left[offset] - right[offset];
        sum += diff * diff;
    }
    sum
}

#[cfg(test)]
pub(crate) fn exact_topk(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    fanout: usize,
) -> Vec<usize> {
    let mut top = TinyTopK::new(fanout);
    for (idx, candidate) in candidates.chunks_exact(dim).enumerate() {
        top.push(l2_sq(query, candidate), idx);
    }
    top.leaders()
}

fn exact_topk_layout(query: &[f32], layout: &AdSamplingLeaderLayout, fanout: usize) -> Vec<usize> {
    let mut top = TinyTopK::new(fanout);
    for idx in 0..layout.leaders {
        top.push(l2_sq(query, layout.leader(idx)), idx);
    }
    top.leaders()
}

#[inline]
fn topk_overlap_count(approx: &[usize], exact: &[usize]) -> usize {
    approx
        .iter()
        .filter(|candidate| exact.contains(candidate))
        .count()
}

#[inline]
fn recall_at_fanout(hits: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

#[cfg(test)]
pub(crate) fn assign_one_adsampling(
    query: &[f32],
    candidates: &[f32],
    dim: usize,
    fanout: usize,
    config: AdSamplingConfig,
) -> AnnResult<AdSamplingOutcome> {
    if query.len() != dim || candidates.len() % dim != 0 {
        return Err(AnnError::log_index_error(format!(
            "Invalid ADSampling dimensions: query={} candidates={} dim={dim}",
            query.len(),
            candidates.len()
        )));
    }
    let ratios = AdSamplingRatios::new(dim, config.epsilon, config.group_dims)?;
    let candidate_count = candidates.len() / dim;
    let seed_indices = adsampling_seed_indices(candidate_count, config.seed_exact, fanout);
    let seeded = adsampling_seed_mask(candidate_count, &seed_indices);
    let mut top = TinyTopK::new(fanout.min(candidate_count.max(1)));
    let mut outcome = AdSamplingOutcome::default();

    for &idx in &seed_indices {
        let candidate = &candidates[idx * dim..(idx + 1) * dim];
        top.push(l2_sq(query, candidate), idx);
        outcome.full_evals += 1;
    }

    for idx in 0..candidate_count {
        if seeded[idx] {
            continue;
        }
        let candidate = &candidates[idx * dim..(idx + 1) * dim];
        let mut partial = 0.0f32;
        let mut pruned = false;
        for start in (0..dim).step_by(ratios.group_dims()) {
            let end = (start + ratios.group_dims()).min(dim);
            for axis in start..end {
                let diff = query[axis] - candidate[axis];
                partial += diff * diff;
            }
            outcome.group_evals += 1;
            let threshold = top.threshold() * ratios.ratio_after_visited(end);
            if partial >= threshold {
                outcome.pruned_evals += 1;
                pruned = true;
                break;
            }
        }
        if !pruned {
            top.push(partial, idx);
            outcome.full_evals += 1;
        }
    }

    outcome.leaders = top.leaders();
    if config.validate_sources > 0 {
        let exact = exact_topk(query, candidates, dim, fanout);
        outcome.validation_recall_hits = topk_overlap_count(&outcome.leaders, &exact) as u64;
        outcome.validation_recall_total = exact.len() as u64;
        outcome.validation_recall_at_fanout = recall_at_fanout(
            outcome.validation_recall_hits,
            outcome.validation_recall_total,
        );
        if outcome.leaders != exact {
            outcome.validation_mismatch = true;
        }
    }
    Ok(outcome)
}

#[cfg(test)]
pub(crate) fn assign_one_adsampling_layout(
    query: &[f32],
    layout: &AdSamplingLeaderLayout,
    fanout: usize,
    config: AdSamplingConfig,
) -> AnnResult<AdSamplingOutcome> {
    if query.len() != layout.dim {
        return Err(AnnError::log_index_error(format!(
            "Invalid ADSampling query dim: query={} layout_dim={}",
            query.len(),
            layout.dim
        )));
    }
    let ratios = AdSamplingRatios::new(layout.dim, config.epsilon, config.group_dims)?;
    let seed_indices = adsampling_seed_indices(layout.leaders, config.seed_exact, fanout);
    let seeded = adsampling_seed_mask(layout.leaders, &seed_indices);
    let mut top = TinyTopK::new(fanout.min(layout.leaders.max(1)));
    let mut outcome = AdSamplingOutcome::default();

    let seed_start = Instant::now();
    for &idx in &seed_indices {
        top.push(l2_sq(query, layout.leader(idx)), idx);
        outcome.full_evals += 1;
    }
    let _seed_elapsed = seed_start.elapsed();

    let mut distances = [0.0f32; ADS_LEADER_BLOCK];
    for block in &layout.blocks {
        distances.fill(0.0);
        let mut active_mask =
            block_len_mask(block.len) & !seeded_block_mask(&seeded, block.start, block.len);
        if active_mask == 0 {
            continue;
        }
        for start_dim in (0..layout.dim).step_by(ratios.group_dims()) {
            if active_mask == 0 {
                break;
            }
            let end_dim = (start_dim + ratios.group_dims()).min(layout.dim);
            outcome.group_evals += active_mask.count_ones() as u64;
            if accumulate_ads_soa_group_mask(
                query,
                block,
                start_dim,
                end_dim,
                active_mask,
                &mut distances,
                config.sparse_full64_threshold,
                config.sparse_group16_threshold,
            ) {
                outcome.simd_group_calls += simd_group_call_count(active_mask);
                outcome.simd_active_lane_evals += active_mask.count_ones() as u64;
            } else {
                outcome.scalar_group_evals += active_mask.count_ones() as u64;
            }
            let before = active_mask.count_ones() as u64;
            active_mask = prune_ads_mask(
                active_mask,
                &distances,
                top.threshold() * ratios.ratio_after_visited(end_dim),
            );
            outcome.pruned_evals += before - active_mask.count_ones() as u64;
        }
        let survivors = active_mask.count_ones() as u64;
        push_active_distances(&mut top, block, active_mask, &distances);
        outcome.full_evals += survivors;
    }

    outcome.leaders = top.leaders();
    if config.validate_sources > 0 {
        let exact = exact_topk_layout(query, layout, fanout);
        outcome.validation_recall_hits = topk_overlap_count(&outcome.leaders, &exact) as u64;
        outcome.validation_recall_total = exact.len() as u64;
        outcome.validation_recall_at_fanout = recall_at_fanout(
            outcome.validation_recall_hits,
            outcome.validation_recall_total,
        );
        if outcome.leaders != exact {
            outcome.validation_mismatch = true;
        }
    }
    Ok(outcome)
}

#[inline]
fn block_len_mask(len: usize) -> u64 {
    if len >= ADS_LEADER_BLOCK {
        u64::MAX
    } else {
        (1_u64 << len) - 1
    }
}

#[inline]
fn clear_block_mask_range(
    mask: &mut u64,
    block_start: usize,
    block_end: usize,
    range_start: usize,
    range_end: usize,
) {
    let start = block_start.max(range_start);
    let end = block_end.min(range_end);
    if start < end {
        let local_start = start - block_start;
        *mask &= !(block_len_mask(end - start) << local_start);
    }
}

pub(crate) fn adsampling_seed_indices(
    candidate_count: usize,
    seed_exact: usize,
    fanout: usize,
) -> Vec<usize> {
    let seed_count = candidate_count.min(seed_exact.max(fanout).max(1));
    if seed_count == 0 {
        return Vec::new();
    }
    if seed_count >= candidate_count {
        return (0..candidate_count).collect();
    }
    (0..seed_count).collect()
}

pub(crate) fn adsampling_seed_mask(candidate_count: usize, seed_indices: &[usize]) -> Vec<bool> {
    let mut seeded = vec![false; candidate_count];
    for &idx in seed_indices {
        if let Some(slot) = seeded.get_mut(idx) {
            *slot = true;
        }
    }
    seeded
}

#[inline]
fn seeded_block_mask(seeded: &[bool], block_start: usize, block_len: usize) -> u64 {
    let mut mask = 0_u64;
    for local in 0..block_len {
        if seeded.get(block_start + local).copied().unwrap_or(false) {
            mask |= 1_u64 << local;
        }
    }
    mask
}

#[inline]
#[cfg(test)]
fn simd_group_call_count(active_mask: u64) -> u64 {
    (0..ADS_LEADER_BLOCK)
        .step_by(16)
        .filter(|&local| ((active_mask >> local) & 0xffff) != 0)
        .count() as u64
}

#[inline]
fn simd_batch8_group_call_count(active_masks: [u64; ADS_QUERY_TILE]) -> u64 {
    let active = active_masks[0]
        | active_masks[1]
        | active_masks[2]
        | active_masks[3]
        | active_masks[4]
        | active_masks[5]
        | active_masks[6]
        | active_masks[7];
    u64::from((active & 0x0000_0000_0000_ffff) != 0)
        + u64::from((active & 0x0000_0000_ffff_0000) != 0)
        + u64::from((active & 0x0000_ffff_0000_0000) != 0)
        + u64::from((active & 0xffff_0000_0000_0000) != 0)
}

#[inline]
fn push_active_distances(
    top: &mut TinyTopK,
    block: &AdSamplingLeaderBlock,
    mut active_mask: u64,
    distances: &[f32; ADS_LEADER_BLOCK],
) -> u64 {
    let mut pushed = 0_u64;
    while active_mask != 0 {
        let local = active_mask.trailing_zeros() as usize;
        top.push(distances[local], block.start + local);
        active_mask &= active_mask - 1;
        pushed += 1;
    }
    pushed
}

#[inline]
fn push_unique_active_distances(
    top: &mut TinyTopK,
    block: &AdSamplingLeaderBlock,
    mut active_mask: u64,
    distances: &[f32; ADS_LEADER_BLOCK],
) -> u64 {
    let mut pushed = 0_u64;
    while active_mask != 0 {
        let local = active_mask.trailing_zeros() as usize;
        top.push_unique(distances[local], block.start + local);
        active_mask &= active_mask - 1;
        pushed += 1;
    }
    pushed
}

fn assign_point_tile_adsampling_layout(
    tile_data: &[f32],
    source_start: usize,
    layout: &AdSamplingLeaderLayout,
    fanout: usize,
    config: AdSamplingConfig,
    seed_indices: &[usize],
    seeded: &[bool],
) -> AnnResult<AdSamplingTileResult> {
    if tile_data.len() % layout.dim != 0 {
        return Err(AnnError::log_index_error(format!(
            "Invalid ADSampling tile dimensions: tile_len={} dim={}",
            tile_data.len(),
            layout.dim
        )));
    }
    let query_count = tile_data.len() / layout.dim;
    debug_assert!(query_count > 0 && query_count <= ADS_QUERY_TILE);
    let ratios = AdSamplingRatios::new(layout.dim, config.epsilon, config.group_dims)?;
    let mut top = vec![TinyTopK::new(fanout.min(layout.leaders.max(1))); query_count];
    let mut result = AdSamplingTileResult::new(query_count);

    let seed_start = Instant::now();
    for &leader_idx in seed_indices {
        let leader = layout.leader(leader_idx);
        for (query_idx, query) in tile_data.chunks_exact(layout.dim).enumerate() {
            top[query_idx].push(l2_sq(query, leader), leader_idx);
            result.full_evals += 1;
        }
    }
    result.seed += seed_start.elapsed();

    let first_query = &tile_data[..layout.dim];
    let mut queries = [first_query; ADS_QUERY_TILE];
    for (query_idx, query) in tile_data.chunks_exact(layout.dim).enumerate() {
        queries[query_idx] = query;
    }

    let scan_start = Instant::now();
    let mut distances = [[0.0f32; ADS_LEADER_BLOCK]; ADS_QUERY_TILE];
    for block in &layout.blocks {
        for row in distances.iter_mut().take(query_count) {
            row.fill(0.0);
        }
        let base_mask =
            block_len_mask(block.len) & !seeded_block_mask(&seeded, block.start, block.len);
        if base_mask == 0 {
            continue;
        }
        let mut active_masks = [0_u64; ADS_QUERY_TILE];
        active_masks[..query_count].fill(base_mask);

        for start_dim in (0..layout.dim).step_by(ratios.group_dims()) {
            if active_masks[..query_count].iter().all(|&mask| mask == 0) {
                break;
            }
            let end_dim = (start_dim + ratios.group_dims()).min(layout.dim);
            let active_before = active_masks;
            let active_count = active_before[..query_count]
                .iter()
                .map(|mask| mask.count_ones() as u64)
                .sum::<u64>();
            result.group_evals += active_count;
            if accumulate_ads_soa_group_batch8(
                queries,
                query_count,
                block,
                start_dim,
                end_dim,
                active_before,
                &mut distances,
                config.sparse_full64_threshold,
                config.sparse_group16_threshold,
            ) {
                result.simd_group_calls += simd_batch8_group_call_count(active_before);
                result.simd_active_lane_evals += active_count;
            } else {
                result.scalar_group_evals += active_before[..query_count]
                    .iter()
                    .map(|mask| mask.count_ones() as u64)
                    .sum::<u64>();
            }

            let ratio = ratios.ratio_after_visited(end_dim);
            for query_idx in 0..query_count {
                let before = active_masks[query_idx].count_ones() as u64;
                active_masks[query_idx] = prune_ads_mask(
                    active_masks[query_idx],
                    &distances[query_idx],
                    top[query_idx].threshold() * ratio,
                );
                result.pruned_evals += before - active_masks[query_idx].count_ones() as u64;
            }
        }

        for query_idx in 0..query_count {
            let survivors = active_masks[query_idx].count_ones() as u64;
            push_active_distances(
                &mut top[query_idx],
                block,
                active_masks[query_idx],
                &distances[query_idx],
            );
            result.full_evals += survivors;
        }
    }
    result.scan += scan_start.elapsed();

    for (query_idx, query) in tile_data.chunks_exact(layout.dim).enumerate() {
        let leaders = top[query_idx].leaders();
        if should_validate_d0_source(config.validate_sources, source_start + query_idx) {
            result.validation_sources += 1;
            let exact = exact_topk_layout(query, layout, fanout);
            result.validation_recall_hits += topk_overlap_count(&leaders, &exact) as u64;
            result.validation_recall_total += exact.len() as u64;
            if leaders != exact {
                result.validation_mismatches += 1;
            }
        }
        result.leaders_by_point.push(leaders);
    }
    Ok(result)
}

#[inline]
fn prune_ads_mask(active_mask: u64, distances: &[f32; ADS_LEADER_BLOCK], threshold: f32) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if adsampling_avx512_available() {
            return unsafe { prune_ads_mask_avx512(active_mask, distances, threshold) };
        }
    }
    prune_ads_mask_scalar(active_mask, distances, threshold)
}

#[inline]
fn prune_ads_mask_scalar(
    mut active_mask: u64,
    distances: &[f32; ADS_LEADER_BLOCK],
    threshold: f32,
) -> u64 {
    let mut keep_mask = 0_u64;
    while active_mask != 0 {
        let local = active_mask.trailing_zeros() as usize;
        if distances[local] < threshold {
            keep_mask |= 1_u64 << local;
        }
        active_mask &= active_mask - 1;
    }
    keep_mask
}

#[inline]
#[cfg(test)]
fn accumulate_ads_soa_group_mask(
    point: &[f32],
    block: &AdSamplingLeaderBlock,
    start_dim: usize,
    end_dim: usize,
    active_mask: u64,
    distances: &mut [f32; ADS_LEADER_BLOCK],
    sparse_full64_threshold: u32,
    sparse_group16_threshold: u32,
) -> bool {
    if should_use_sparse_ads_fallback(
        &[active_mask],
        active_mask.count_ones(),
        sparse_full64_threshold,
        sparse_group16_threshold,
    ) {
        accumulate_ads_soa_group_mask_scalar(
            point,
            block,
            start_dim,
            end_dim,
            active_mask,
            distances,
        );
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if adsampling_avx512_available() {
            unsafe {
                accumulate_ads_soa_group_mask_avx512(
                    point,
                    block,
                    start_dim,
                    end_dim,
                    active_mask,
                    distances,
                );
            }
            return true;
        }
    }
    accumulate_ads_soa_group_mask_scalar(point, block, start_dim, end_dim, active_mask, distances);
    false
}

#[inline]
fn accumulate_ads_soa_group_batch8(
    points: [&[f32]; ADS_QUERY_TILE],
    query_count: usize,
    block: &AdSamplingLeaderBlock,
    start_dim: usize,
    end_dim: usize,
    active_masks: [u64; ADS_QUERY_TILE],
    distances: &mut [[f32; ADS_LEADER_BLOCK]; ADS_QUERY_TILE],
    sparse_full64_threshold: u32,
    sparse_group16_threshold: u32,
) -> bool {
    let active_count = active_masks[..query_count]
        .iter()
        .map(|mask| mask.count_ones())
        .sum::<u32>();
    accumulate_ads_soa_group_batch8_with_active_count(
        points,
        query_count,
        block,
        start_dim,
        end_dim,
        active_masks,
        active_count,
        distances,
        sparse_full64_threshold,
        sparse_group16_threshold,
    )
}

fn accumulate_ads_soa_group_batch8_with_active_count(
    points: [&[f32]; ADS_QUERY_TILE],
    query_count: usize,
    block: &AdSamplingLeaderBlock,
    start_dim: usize,
    end_dim: usize,
    active_masks: [u64; ADS_QUERY_TILE],
    active_count: u32,
    distances: &mut [[f32; ADS_LEADER_BLOCK]; ADS_QUERY_TILE],
    sparse_full64_threshold: u32,
    sparse_group16_threshold: u32,
) -> bool {
    if should_use_sparse_ads_fallback(
        &active_masks[..query_count],
        active_count,
        sparse_full64_threshold,
        sparse_group16_threshold,
    ) {
        for query_idx in 0..query_count {
            accumulate_ads_soa_group_mask_scalar(
                points[query_idx],
                block,
                start_dim,
                end_dim,
                active_masks[query_idx],
                &mut distances[query_idx],
            );
        }
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if adsampling_avx512_available() {
            unsafe {
                accumulate_ads_soa_group_batch8_avx512(
                    points,
                    block,
                    start_dim,
                    end_dim,
                    active_masks,
                    distances,
                );
            }
            return true;
        }
    }
    for query_idx in 0..query_count {
        accumulate_ads_soa_group_mask_scalar(
            points[query_idx],
            block,
            start_dim,
            end_dim,
            active_masks[query_idx],
            &mut distances[query_idx],
        );
    }
    false
}

fn accumulate_ads_soa_group_mask_scalar(
    point: &[f32],
    block: &AdSamplingLeaderBlock,
    start_dim: usize,
    end_dim: usize,
    mut active_mask: u64,
    distances: &mut [f32; ADS_LEADER_BLOCK],
) {
    while active_mask != 0 {
        let local = active_mask.trailing_zeros() as usize;
        let mut acc = distances[local];
        for dim_idx in start_dim..end_dim {
            let diff = point[dim_idx] - block.data[dim_idx * block.stride + local];
            acc += diff * diff;
        }
        distances[local] = acc;
        active_mask &= active_mask - 1;
    }
}

fn should_use_sparse_ads_fallback(
    active_masks: &[u64],
    total_active: u32,
    sparse_full64_threshold: u32,
    sparse_group16_threshold: u32,
) -> bool {
    if total_active == 0 {
        return false;
    }
    if total_active < sparse_full64_threshold {
        return true;
    }
    let mut saw_group = false;
    let mut all_groups_sparse = true;
    for local in (0..ADS_LEADER_BLOCK).step_by(16) {
        let active_in_group = active_masks
            .iter()
            .map(|mask| ((mask >> local) & 0xffff).count_ones())
            .sum::<u32>();
        if active_in_group == 0 {
            continue;
        }
        saw_group = true;
        if active_in_group > sparse_group16_threshold {
            all_groups_sparse = false;
            break;
        }
    }
    saw_group && all_groups_sparse
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_ads_soa_group_batch8_avx512(
    points: [&[f32]; ADS_QUERY_TILE],
    block: &AdSamplingLeaderBlock,
    start_dim: usize,
    end_dim: usize,
    active_masks: [u64; ADS_QUERY_TILE],
    distances: &mut [[f32; ADS_LEADER_BLOCK]; ADS_QUERY_TILE],
) {
    use std::arch::x86_64::*;

    let point_ptrs = [
        points[0].as_ptr(),
        points[1].as_ptr(),
        points[2].as_ptr(),
        points[3].as_ptr(),
        points[4].as_ptr(),
        points[5].as_ptr(),
        points[6].as_ptr(),
        points[7].as_ptr(),
    ];

    for local in (0..ADS_LEADER_BLOCK).step_by(16) {
        let lane_masks = [
            ((active_masks[0] >> local) & 0xffff) as __mmask16,
            ((active_masks[1] >> local) & 0xffff) as __mmask16,
            ((active_masks[2] >> local) & 0xffff) as __mmask16,
            ((active_masks[3] >> local) & 0xffff) as __mmask16,
            ((active_masks[4] >> local) & 0xffff) as __mmask16,
            ((active_masks[5] >> local) & 0xffff) as __mmask16,
            ((active_masks[6] >> local) & 0xffff) as __mmask16,
            ((active_masks[7] >> local) & 0xffff) as __mmask16,
        ];
        if lane_masks.iter().all(|&mask| mask == 0) {
            continue;
        }

        let mut acc0 = unsafe { _mm512_loadu_ps(distances[0].as_ptr().add(local)) };
        let mut acc1 = unsafe { _mm512_loadu_ps(distances[1].as_ptr().add(local)) };
        let mut acc2 = unsafe { _mm512_loadu_ps(distances[2].as_ptr().add(local)) };
        let mut acc3 = unsafe { _mm512_loadu_ps(distances[3].as_ptr().add(local)) };
        let mut acc4 = unsafe { _mm512_loadu_ps(distances[4].as_ptr().add(local)) };
        let mut acc5 = unsafe { _mm512_loadu_ps(distances[5].as_ptr().add(local)) };
        let mut acc6 = unsafe { _mm512_loadu_ps(distances[6].as_ptr().add(local)) };
        let mut acc7 = unsafe { _mm512_loadu_ps(distances[7].as_ptr().add(local)) };
        for dim_idx in start_dim..end_dim {
            let values =
                unsafe { _mm512_loadu_ps(block.data.as_ptr().add(dim_idx * block.stride + local)) };
            if lane_masks[0] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[0].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc0 = _mm512_fmadd_ps(diff, diff, acc0);
            }
            if lane_masks[1] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[1].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc1 = _mm512_fmadd_ps(diff, diff, acc1);
            }
            if lane_masks[2] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[2].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc2 = _mm512_fmadd_ps(diff, diff, acc2);
            }
            if lane_masks[3] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[3].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc3 = _mm512_fmadd_ps(diff, diff, acc3);
            }
            if lane_masks[4] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[4].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc4 = _mm512_fmadd_ps(diff, diff, acc4);
            }
            if lane_masks[5] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[5].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc5 = _mm512_fmadd_ps(diff, diff, acc5);
            }
            if lane_masks[6] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[6].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc6 = _mm512_fmadd_ps(diff, diff, acc6);
            }
            if lane_masks[7] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[7].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc7 = _mm512_fmadd_ps(diff, diff, acc7);
            }
        }
        if lane_masks[0] != 0 {
            unsafe { _mm512_storeu_ps(distances[0].as_mut_ptr().add(local), acc0) };
        }
        if lane_masks[1] != 0 {
            unsafe { _mm512_storeu_ps(distances[1].as_mut_ptr().add(local), acc1) };
        }
        if lane_masks[2] != 0 {
            unsafe { _mm512_storeu_ps(distances[2].as_mut_ptr().add(local), acc2) };
        }
        if lane_masks[3] != 0 {
            unsafe { _mm512_storeu_ps(distances[3].as_mut_ptr().add(local), acc3) };
        }
        if lane_masks[4] != 0 {
            unsafe { _mm512_storeu_ps(distances[4].as_mut_ptr().add(local), acc4) };
        }
        if lane_masks[5] != 0 {
            unsafe { _mm512_storeu_ps(distances[5].as_mut_ptr().add(local), acc5) };
        }
        if lane_masks[6] != 0 {
            unsafe { _mm512_storeu_ps(distances[6].as_mut_ptr().add(local), acc6) };
        }
        if lane_masks[7] != 0 {
            unsafe { _mm512_storeu_ps(distances[7].as_mut_ptr().add(local), acc7) };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn adsampling_avx512_available() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn adsampling_avx512_available() -> bool {
    false
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_ads_soa_group_mask_avx512(
    point: &[f32],
    block: &AdSamplingLeaderBlock,
    start_dim: usize,
    end_dim: usize,
    active_mask: u64,
    distances: &mut [f32; ADS_LEADER_BLOCK],
) {
    use std::arch::x86_64::*;

    for local in (0..ADS_LEADER_BLOCK).step_by(16) {
        let lane_mask = ((active_mask >> local) & 0xffff) as __mmask16;
        if lane_mask == 0 {
            continue;
        }
        let mut acc = unsafe { _mm512_loadu_ps(distances.as_ptr().add(local)) };
        for dim_idx in start_dim..end_dim {
            let q = _mm512_set1_ps(point[dim_idx]);
            let values =
                unsafe { _mm512_loadu_ps(block.data.as_ptr().add(dim_idx * block.stride + local)) };
            let diff = _mm512_sub_ps(q, values);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }
        unsafe { _mm512_mask_storeu_ps(distances.as_mut_ptr().add(local), lane_mask, acc) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn prune_ads_mask_avx512(
    active_mask: u64,
    distances: &[f32; ADS_LEADER_BLOCK],
    threshold: f32,
) -> u64 {
    use std::arch::x86_64::*;

    let threshold = _mm512_set1_ps(threshold);
    let mut keep_mask = 0_u64;
    for local in (0..ADS_LEADER_BLOCK).step_by(16) {
        let lane_active = ((active_mask >> local) & 0xffff) as __mmask16;
        if lane_active == 0 {
            continue;
        }
        let values = unsafe { _mm512_loadu_ps(distances.as_ptr().add(local)) };
        let keep = _mm512_cmp_ps_mask(values, threshold, _CMP_LT_OQ) & lane_active;
        keep_mask |= (keep as u64) << local;
    }
    keep_mask
}

pub(crate) fn should_use_adsampling_root(
    params: &ForgeANNParams,
    points: usize,
    leaders: usize,
    fanout: usize,
) -> bool {
    params.root_adsampling_enabled()
        && points > 0
        && leaders > fanout.max(1)
        && params.adsampling_rotation_available()
}

pub(crate) fn adsampling_depth_fallback_reason(
    params: &ForgeANNParams,
    depth: usize,
    points: usize,
    leaders: usize,
    fanout: usize,
) -> Option<&'static str> {
    if depth == 0 {
        return Some("root-depth");
    }
    if depth > ForgeANNParams::ADS_DEPTH_MAX_DEPTH {
        return Some("max-depth");
    }
    if points == 0 {
        return Some("empty-points");
    }
    if fanout < 2 {
        return Some("fanout");
    }
    if leaders <= fanout.max(1) {
        return Some("leaders-le-fanout");
    }
    if points < 1 {
        return Some("min-points");
    }
    if leaders < 1 {
        return Some("min-leaders");
    }
    if (points as u128).saturating_mul(leaders as u128) < 1 {
        return Some("min-work");
    }
    if params.adsampling_group_dims == 0 {
        return Some("group-dims");
    }
    None
}

pub(crate) fn classify_adsampling_task(
    _params: &ForgeANNParams,
    points: usize,
    leaders: usize,
) -> AdSamplingTaskClass {
    let work = (points as u128).saturating_mul(leaders as u128);
    let large_ads = work >= ForgeANNParams::ADS_LARGE_MIN_WORK as u128
        || (points >= ForgeANNParams::ADS_LARGE_MIN_POINTS
            && leaders >= ForgeANNParams::ADS_LARGE_MIN_LEADERS);
    let huge_ads =
        work >= ForgeANNParams::ADS_HUGE_MIN_WORK as u128 || points >= 1_000_000 || leaders >= 4096;
    AdSamplingTaskClass {
        large_ads,
        huge_ads,
    }
}

pub(crate) fn should_use_adsampling_depth(
    params: &ForgeANNParams,
    depth: usize,
    points: usize,
    leaders: usize,
    fanout: usize,
) -> bool {
    adsampling_depth_fallback_reason(params, depth, points, leaders, fanout).is_none()
}

pub(crate) fn should_use_adsampling_assignment(
    params: &ForgeANNParams,
    depth: usize,
    points: usize,
    leaders: usize,
    fanout: usize,
) -> bool {
    if depth == 0 {
        should_use_adsampling_root(params, points, leaders, fanout)
    } else {
        should_use_adsampling_depth(params, depth, points, leaders, fanout)
    }
}

fn choose_adsampling_assignment_chunk_rows_for_threads(points: usize, threads: usize) -> usize {
    if points == 0 {
        return ADS_ASSIGN_MIN_CHUNK_ROWS;
    }
    let max_blocks_by_min = points.div_ceil(ADS_ASSIGN_MIN_CHUNK_ROWS).max(1);
    let target_blocks = threads.max(1).min(max_blocks_by_min);
    points
        .div_ceil(target_blocks)
        .clamp(ADS_ASSIGN_MIN_CHUNK_ROWS, ADS_ASSIGN_MAX_CHUNK_ROWS)
}

fn choose_adsampling_assignment_chunk_rows(points: usize) -> usize {
    choose_adsampling_assignment_chunk_rows_for_threads(points, rayon::current_num_threads().max(1))
}

pub(crate) fn assign_point_leaders_adsampling_streaming(
    dataset: &dyn PointStore,
    cur: &[u32],
    leaders: &[u32],
    local_fanout: usize,
    params: &ForgeANNParams,
    depth: usize,
    mut visit_chunk: impl FnMut(&AdSamplingPointAssignmentChunk) -> AnnResult<()> + Send,
) -> AnnResult<AdSamplingProfile> {
    if leaders.is_empty() {
        return Err(AnnError::log_index_error(
            "ADSampling assignment requires at least one leader".to_string(),
        ));
    }
    let total_start = Instant::now();
    let dim = dataset.dim();
    let fanout = local_fanout.min(leaders.len()).clamp(1, 32);
    let config = if depth == 0 {
        AdSamplingConfig::root_from_params(params, fanout)
    } else {
        AdSamplingConfig::depth_from_params(params, fanout)
    };
    let layout_start = Instant::now();
    let leader_layout = AdSamplingLeaderLayout::build(dataset, leaders)?;
    let layout_elapsed = layout_start.elapsed();
    let seed_indices = adsampling_seed_indices(leaders.len(), config.seed_exact, fanout);
    let seeded = adsampling_seed_mask(leaders.len(), &seed_indices);
    let seed_count = seed_indices.len();
    let chunk_rows = choose_adsampling_assignment_chunk_rows(cur.len());
    let mut profile = AdSamplingProfile {
        depth,
        points: cur.len(),
        leaders: leaders.len(),
        fanout,
        epsilon: config.epsilon,
        group_dims: config.group_dims,
        seed_exact_m: seed_count,
        layout_ms: duration_ms(layout_elapsed),
        chunks: cur.len().div_ceil(chunk_rows),
        called_inside_rayon_worker: rayon::current_thread_index().is_some(),
        scheduler_mode: "depth-wave".to_string(),
        ..AdSamplingProfile::default()
    };

    let (tx, rx) = channel::bounded(rayon::current_num_threads().max(1) * 2);
    let active_chunks = Arc::new(AtomicUsize::new(0));
    let active_chunk_max = Arc::new(AtomicUsize::new(0));
    let active_chunk_start_sum = Arc::new(AtomicU64::new(0));
    let send_block_nanos = Arc::new(AtomicU64::new(0));
    rayon::scope(|scope| -> AnnResult<()> {
        for (chunk_idx, point_ids) in cur.chunks(chunk_rows).enumerate() {
            let tx = tx.clone();
            let leader_layout = &leader_layout;
            let seed_indices = &seed_indices;
            let seeded = &seeded;
            let active_chunks = Arc::clone(&active_chunks);
            let active_chunk_max = Arc::clone(&active_chunk_max);
            let active_chunk_start_sum = Arc::clone(&active_chunk_start_sum);
            let send_block_nanos = Arc::clone(&send_block_nanos);
            scope.spawn(move |_| {
                let active = active_chunks.fetch_add(1, Ordering::AcqRel) + 1;
                update_atomic_max(&active_chunk_max, active);
                active_chunk_start_sum.fetch_add(active as u64, Ordering::Relaxed);
                let chunk_start = Instant::now();
                let chunk = assign_point_chunk_adsampling_layout(
                    dataset,
                    point_ids,
                    chunk_idx * chunk_rows,
                    dim,
                    leader_layout,
                    fanout,
                    config,
                    seed_indices,
                    seeded,
                );
                let chunk_wall = chunk_start.elapsed();
                active_chunks.fetch_sub(1, Ordering::AcqRel);
                let message = AdSamplingChunkMessage {
                    chunk_idx,
                    chunk,
                    chunk_wall,
                };
                let send_start = Instant::now();
                let _ = tx.send(message);
                send_block_nanos
                    .fetch_add(duration_nanos_u64(send_start.elapsed()), Ordering::Relaxed);
            });
        }
        drop(tx);

        let mut next_chunk = 0usize;
        let mut pending = BTreeMap::new();
        let mut first_error = None;
        let mut wait_start = Instant::now();
        loop {
            match rx.try_recv() {
                Ok(message) => {
                    profile.recv_wait_ms += duration_ms(wait_start.elapsed());
                    profile.chunk_wall_accumulated_ms += duration_ms(message.chunk_wall);
                    profile.chunk_wall_max_ms = profile
                        .chunk_wall_max_ms
                        .max(duration_ms(message.chunk_wall));
                    pending.insert(message.chunk_idx, message.chunk);
                    profile.ordered_pending_max = profile.ordered_pending_max.max(pending.len());
                }
                Err(channel::TryRecvError::Empty) => {
                    profile.recv_empty_polls += 1;
                    if rayon::yield_now().is_none() {
                        std::thread::yield_now();
                    }
                    continue;
                }
                Err(channel::TryRecvError::Disconnected) => {
                    profile.recv_wait_ms += duration_ms(wait_start.elapsed());
                    break;
                }
            }

            while let Some(chunk) = pending.remove(&next_chunk) {
                if first_error.is_none() {
                    match chunk {
                        Ok(chunk) => {
                            profile.seed_ms += duration_ms(chunk.seed);
                            profile.seed_accumulated_ms += duration_ms(chunk.seed);
                            profile.seed_wall_ms =
                                profile.seed_wall_ms.max(duration_ms(chunk.seed));
                            profile.scan_ms += duration_ms(chunk.scan);
                            profile.scan_accumulated_ms += duration_ms(chunk.scan);
                            profile.full_evals += chunk.full_evals;
                            profile.pruned_evals += chunk.pruned_evals;
                            profile.group_evals += chunk.group_evals;
                            profile.simd_group_calls += chunk.simd_group_calls;
                            profile.simd_active_lane_evals += chunk.simd_active_lane_evals;
                            profile.scalar_group_evals += chunk.scalar_group_evals;
                            profile.validation_sources += chunk.validation_sources;
                            profile.validation_mismatches += chunk.validation_mismatches;
                            profile.validation_recall_hits += chunk.validation_recall_hits;
                            profile.validation_recall_total += chunk.validation_recall_total;
                            let visit_start = Instant::now();
                            if let Err(err) = visit_chunk(&chunk.chunk) {
                                first_error = Some(err);
                            }
                            profile.visit_chunk_ms += duration_ms(visit_start.elapsed());
                        }
                        Err(err) => {
                            first_error = Some(err);
                        }
                    }
                }
                next_chunk += 1;
            }
            wait_start = Instant::now();
        }
        if let Some(err) = first_error {
            return Err(err);
        }
        Ok(())
    })?;
    profile.chunk_send_block_ms = duration_ms(Duration::from_nanos(
        send_block_nanos.load(Ordering::Relaxed),
    ));
    profile.chunk_active_max = active_chunk_max.load(Ordering::Relaxed);
    let active_start_sum = active_chunk_start_sum.load(Ordering::Relaxed);
    if profile.chunks > 0 {
        profile.chunk_active_start_avg = active_start_sum as f64 / profile.chunks as f64;
    }

    profile.validation_recall_at_fanout = recall_at_fanout(
        profile.validation_recall_hits,
        profile.validation_recall_total,
    );
    profile.total_ms = duration_ms(total_start.elapsed());
    if profile.total_ms > 0.0 {
        profile.effective_parallelism = profile.chunk_wall_accumulated_ms / profile.total_ms;
    }
    Ok(profile)
}

#[allow(clippy::too_many_arguments)]
fn assign_point_chunk_adsampling_layout(
    dataset: &dyn PointStore,
    point_ids: &[u32],
    source_start: usize,
    dim: usize,
    leader_layout: &AdSamplingLeaderLayout,
    fanout: usize,
    config: AdSamplingConfig,
    seed_indices: &[usize],
    seeded: &[bool],
) -> AnnResult<AdSamplingChunkResult> {
    let mut point_data = vec![0.0f32; point_ids.len() * dim];
    let read_options = adsampling_windowed_options(dataset, point_ids.len());
    let mut io_stats = PointBatchStats::default();
    dataset.read_points_windowed_into_batch_stats(
        point_ids,
        &mut point_data,
        &read_options,
        &mut io_stats,
    )?;
    assign_loaded_point_chunk_adsampling_layout(
        point_ids,
        &point_data,
        source_start,
        dim,
        leader_layout,
        fanout,
        config,
        seed_indices,
        seeded,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assign_loaded_point_chunk_adsampling_layout(
    point_ids: &[u32],
    point_data: &[f32],
    source_start: usize,
    dim: usize,
    leader_layout: &AdSamplingLeaderLayout,
    fanout: usize,
    config: AdSamplingConfig,
    seed_indices: &[usize],
    seeded: &[bool],
) -> AnnResult<AdSamplingChunkResult> {
    let mut result = AdSamplingChunkResult {
        chunk: AdSamplingPointAssignmentChunk {
            source_start,
            leaders_by_point: Vec::with_capacity(point_ids.len()),
        },
        seed: Duration::ZERO,
        scan: Duration::ZERO,
        full_evals: 0,
        pruned_evals: 0,
        group_evals: 0,
        simd_group_calls: 0,
        simd_active_lane_evals: 0,
        scalar_group_evals: 0,
        validation_sources: 0,
        validation_mismatches: 0,
        validation_recall_hits: 0,
        validation_recall_total: 0,
    };

    if point_data.len() != point_ids.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "ADSampling loaded chunk size mismatch: got {} expected {}",
            point_data.len(),
            point_ids.len().saturating_mul(dim)
        )));
    }

    for tile_start in (0..point_ids.len()).step_by(ADS_QUERY_TILE) {
        let tile_end = (tile_start + ADS_QUERY_TILE).min(point_ids.len());
        let tile = &point_data[tile_start * dim..tile_end * dim];
        let tile_result = assign_point_tile_adsampling_layout(
            tile,
            source_start + tile_start,
            leader_layout,
            fanout,
            config,
            seed_indices,
            seeded,
        )?;
        result.seed += tile_result.seed;
        result.scan += tile_result.scan;
        result.full_evals += tile_result.full_evals;
        result.pruned_evals += tile_result.pruned_evals;
        result.group_evals += tile_result.group_evals;
        result.simd_group_calls += tile_result.simd_group_calls;
        result.simd_active_lane_evals += tile_result.simd_active_lane_evals;
        result.scalar_group_evals += tile_result.scalar_group_evals;
        result.validation_sources += tile_result.validation_sources;
        result.validation_mismatches += tile_result.validation_mismatches;
        result.validation_recall_hits += tile_result.validation_recall_hits;
        result.validation_recall_total += tile_result.validation_recall_total;
        result
            .chunk
            .leaders_by_point
            .extend(tile_result.leaders_by_point);
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assign_loaded_point_indexed_adsampling_layout(
    point_ids: &[u32],
    source_data: &[f32],
    source_offsets: &[usize],
    source_start: usize,
    dim: usize,
    leader_layout: &AdSamplingLeaderLayout,
    fanout: usize,
    config: AdSamplingConfig,
    seed_indices: &[usize],
    seeded: &[bool],
) -> AnnResult<AdSamplingChunkResult> {
    if point_ids.len() != source_offsets.len() {
        return Err(AnnError::log_index_error(format!(
            "ADSampling indexed chunk size mismatch: point_ids={} source_offsets={}",
            point_ids.len(),
            source_offsets.len()
        )));
    }
    if dim == 0 || source_data.len() % dim != 0 {
        return Err(AnnError::log_index_error(format!(
            "ADSampling indexed source dimensions mismatch: source_len={} dim={dim}",
            source_data.len()
        )));
    }
    let source_rows = source_data.len() / dim;
    let mut result = AdSamplingChunkResult {
        chunk: AdSamplingPointAssignmentChunk {
            source_start,
            leaders_by_point: Vec::with_capacity(point_ids.len()),
        },
        seed: Duration::ZERO,
        scan: Duration::ZERO,
        full_evals: 0,
        pruned_evals: 0,
        group_evals: 0,
        simd_group_calls: 0,
        simd_active_lane_evals: 0,
        scalar_group_evals: 0,
        validation_sources: 0,
        validation_mismatches: 0,
        validation_recall_hits: 0,
        validation_recall_total: 0,
    };

    let mut tile_data = Vec::with_capacity(ADS_QUERY_TILE.saturating_mul(dim));
    for tile_start in (0..point_ids.len()).step_by(ADS_QUERY_TILE) {
        let tile_end = (tile_start + ADS_QUERY_TILE).min(point_ids.len());
        tile_data.clear();
        for &source_offset in &source_offsets[tile_start..tile_end] {
            if source_offset >= source_rows {
                return Err(AnnError::log_index_error(format!(
                    "ADSampling indexed source offset {source_offset} exceeds source rows {source_rows}"
                )));
            }
            let row_start = source_offset * dim;
            let row_end = row_start + dim;
            tile_data.extend_from_slice(&source_data[row_start..row_end]);
        }
        let tile_result = assign_point_tile_adsampling_layout(
            &tile_data,
            source_start + tile_start,
            leader_layout,
            fanout,
            config,
            seed_indices,
            seeded,
        )?;
        result.seed += tile_result.seed;
        result.scan += tile_result.scan;
        result.full_evals += tile_result.full_evals;
        result.pruned_evals += tile_result.pruned_evals;
        result.group_evals += tile_result.group_evals;
        result.simd_group_calls += tile_result.simd_group_calls;
        result.simd_active_lane_evals += tile_result.simd_active_lane_evals;
        result.scalar_group_evals += tile_result.scalar_group_evals;
        result.validation_sources += tile_result.validation_sources;
        result.validation_mismatches += tile_result.validation_mismatches;
        result.validation_recall_hits += tile_result.validation_recall_hits;
        result.validation_recall_total += tile_result.validation_recall_total;
        result
            .chunk
            .leaders_by_point
            .extend(tile_result.leaders_by_point);
    }
    Ok(result)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LeafAdSamplingTelemetry {
    pub(crate) layout: Duration,
    pub(crate) seed: Duration,
    pub(crate) scan: Duration,
    pub(crate) seed_evals: u64,
    pub(crate) full_evals: u64,
    pub(crate) pruned_evals: u64,
    pub(crate) group_evals: u64,
    pub(crate) simd_group_calls: u64,
    pub(crate) simd_active_lane_evals: u64,
    pub(crate) scalar_group_evals: u64,
    pub(crate) wavefront_pairmask: bool,
    pub(crate) tiled: bool,
    pub(crate) work_graph: bool,
    pub(crate) tile_count: usize,
    pub(crate) tile_rows_total: usize,
    pub(crate) tile_rows_min: usize,
    pub(crate) tile_rows_max: usize,
    pub(crate) tile_wall: Duration,
    pub(crate) tile_wait: Duration,
    pub(crate) tile_handle_requeues: usize,
    pub(crate) cpu_budget: usize,
    pub(crate) active_context_peak: usize,
    pub(crate) active_workers_peak: usize,
    pub(crate) ewma_ns_per_row_by_bucket: [u64; LEAF_ADS_EWMA_BUCKETS],
}

impl LeafAdSamplingTelemetry {
    #[inline]
    pub(crate) fn compute_time(&self) -> Duration {
        self.layout + self.seed + self.scan
    }

    fn merge(&mut self, other: Self) {
        self.layout += other.layout;
        self.seed += other.seed;
        self.scan += other.scan;
        self.seed_evals += other.seed_evals;
        self.full_evals += other.full_evals;
        self.pruned_evals += other.pruned_evals;
        self.group_evals += other.group_evals;
        self.simd_group_calls += other.simd_group_calls;
        self.simd_active_lane_evals += other.simd_active_lane_evals;
        self.scalar_group_evals += other.scalar_group_evals;
        self.wavefront_pairmask |= other.wavefront_pairmask;
        self.tiled |= other.tiled;
        self.work_graph |= other.work_graph;
        self.tile_count += other.tile_count;
        self.tile_rows_total += other.tile_rows_total;
        self.tile_rows_min = match (self.tile_rows_min, other.tile_rows_min) {
            (0, value) => value,
            (value, 0) => value,
            (left, right) => left.min(right),
        };
        self.tile_rows_max = self.tile_rows_max.max(other.tile_rows_max);
        self.tile_wall += other.tile_wall;
        self.tile_wait += other.tile_wait;
        self.tile_handle_requeues += other.tile_handle_requeues;
        self.cpu_budget = self.cpu_budget.max(other.cpu_budget);
        self.active_context_peak = self.active_context_peak.max(other.active_context_peak);
        self.active_workers_peak = self.active_workers_peak.max(other.active_workers_peak);
        for (dst, src) in self
            .ewma_ns_per_row_by_bucket
            .iter_mut()
            .zip(other.ewma_ns_per_row_by_bucket)
        {
            *dst = (*dst).max(src);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LeafAdSamplingResult {
    pub(crate) row_topk: Vec<Vec<(usize, f32)>>,
    pub(crate) telemetry: LeafAdSamplingTelemetry,
}

#[cfg(test)]
pub(crate) fn compute_leaf_adsampling_topk_l2(
    vectors: &[f32],
    leaf_size: usize,
    dim: usize,
    k: usize,
    config: AdSamplingConfig,
) -> AnnResult<LeafAdSamplingResult> {
    compute_leaf_adsampling_topk_l2_with_runtime(vectors, leaf_size, dim, k, config, None)
}

pub(crate) fn compute_leaf_adsampling_topk_l2_with_runtime(
    vectors: &[f32],
    leaf_size: usize,
    dim: usize,
    k: usize,
    config: AdSamplingConfig,
    runtime: Option<&LeafAdsOperatorRuntime>,
) -> AnnResult<LeafAdSamplingResult> {
    if vectors.len() != leaf_size.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Invalid leaf ADSampling matrix: len={} leaf_size={leaf_size} dim={dim}",
            vectors.len()
        )));
    }
    let (context, layout_duration) = build_leaf_ads_context(vectors, leaf_size, dim, config)?;
    if let Some(runtime) = runtime
        && runtime.enabled_for_leaf(leaf_size)
    {
        return Ok(compute_leaf_adsampling_topk_l2_tiled(
            &context,
            leaf_size,
            k,
            layout_duration,
            runtime,
        ));
    }

    Ok(compute_leaf_adsampling_topk_l2_from_context(
        &context,
        leaf_size,
        k,
        layout_duration,
    ))
}

pub(crate) fn compute_leaf_adsampling_topk_l2_wavefront_pairmask(
    vectors: &[f32],
    leaf_size: usize,
    dim: usize,
    k: usize,
    config: AdSamplingConfig,
) -> AnnResult<LeafAdSamplingResult> {
    if vectors.len() != leaf_size.saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Invalid wavefront pair-mask leaf ADSampling matrix: len={} leaf_size={leaf_size} dim={dim}",
            vectors.len()
        )));
    }
    let (context, layout_duration) = build_leaf_ads_context(vectors, leaf_size, dim, config)?;
    Ok(
        compute_leaf_adsampling_topk_l2_wavefront_pairmask_from_context(
            &context,
            leaf_size,
            k,
            layout_duration,
        ),
    )
}

#[derive(Debug)]
struct LeafAdsContext<'a> {
    layout: AdSamplingLeafLayout<'a>,
    ratios: AdSamplingRatios,
    config: AdSamplingConfig,
}

fn build_leaf_ads_context<'a>(
    vectors: &'a [f32],
    leaf_size: usize,
    dim: usize,
    config: AdSamplingConfig,
) -> AnnResult<(LeafAdsContext<'a>, Duration)> {
    let ratios = AdSamplingRatios::new(dim, config.epsilon, config.group_dims)?;
    let layout_start = Instant::now();
    let layout = AdSamplingLeafLayout::build(vectors, leaf_size, dim);
    Ok((
        LeafAdsContext {
            layout,
            ratios,
            config,
        },
        layout_start.elapsed(),
    ))
}

fn compute_leaf_adsampling_topk_l2_from_context(
    context: &LeafAdsContext<'_>,
    leaf_size: usize,
    k: usize,
    layout_duration: Duration,
) -> LeafAdSamplingResult {
    let mut telemetry = LeafAdSamplingTelemetry {
        layout: layout_duration,
        ..LeafAdSamplingTelemetry::default()
    };

    let row_ranges = (0..leaf_size)
        .step_by(ADS_LEAF_ROW_CHUNK)
        .map(|start| (start, (start + ADS_LEAF_ROW_CHUNK).min(leaf_size)))
        .collect::<Vec<_>>();
    let chunks = if should_parallelize_leaf_adsampling_rows(leaf_size) {
        row_ranges
            .into_par_iter()
            .map(|(start, end)| {
                compute_leaf_adsampling_row_range(
                    &context.layout,
                    start,
                    end,
                    k,
                    context.config,
                    &context.ratios,
                )
            })
            .collect::<Vec<_>>()
    } else {
        row_ranges
            .into_iter()
            .map(|(start, end)| {
                compute_leaf_adsampling_row_range(
                    &context.layout,
                    start,
                    end,
                    k,
                    context.config,
                    &context.ratios,
                )
            })
            .collect::<Vec<_>>()
    };

    let mut row_topk = Vec::with_capacity(leaf_size);
    for chunk in chunks {
        telemetry.merge(chunk.telemetry);
        row_topk.extend(chunk.row_topk);
    }

    LeafAdSamplingResult {
        row_topk,
        telemetry,
    }
}

struct LeafWavefrontPairMaskCounters {
    group_evals: u64,
    full_evals: u64,
    pruned_evals: u64,
    simd_group_calls: u64,
    simd_active_lane_evals: u64,
    scalar_group_evals: u64,
}

impl LeafWavefrontPairMaskCounters {
    fn new() -> Self {
        Self {
            group_evals: 0,
            full_evals: 0,
            pruned_evals: 0,
            simd_group_calls: 0,
            simd_active_lane_evals: 0,
            scalar_group_evals: 0,
        }
    }

    fn add_group_work(
        &mut self,
        active_count: u64,
        active_masks: [u64; ADS_QUERY_TILE],
        simd: bool,
    ) {
        self.group_evals += active_count;
        if simd {
            self.simd_group_calls += simd_batch8_group_call_count(active_masks);
            self.simd_active_lane_evals += active_count;
        } else {
            self.scalar_group_evals += active_count;
        }
    }
}

struct LeafWavefrontPairMaskScratch {
    distances: [[f32; ADS_LEADER_BLOCK]; ADS_LEADER_BLOCK],
    survivor_masks: [u64; ADS_LEADER_BLOCK],
    left_thresholds: [f32; ADS_LEADER_BLOCK],
    right_thresholds: [f32; ADS_LEADER_BLOCK],
}

impl LeafWavefrontPairMaskScratch {
    fn new() -> Self {
        Self {
            distances: [[0.0; ADS_LEADER_BLOCK]; ADS_LEADER_BLOCK],
            survivor_masks: [0; ADS_LEADER_BLOCK],
            left_thresholds: [f32::INFINITY; ADS_LEADER_BLOCK],
            right_thresholds: [f32::INFINITY; ADS_LEADER_BLOCK],
        }
    }
}

fn compute_leaf_adsampling_topk_l2_wavefront_pairmask_from_context(
    context: &LeafAdsContext<'_>,
    leaf_size: usize,
    k: usize,
    layout_duration: Duration,
) -> LeafAdSamplingResult {
    let rows = context.layout.rows;
    debug_assert_eq!(rows, leaf_size);
    let k = k.min(rows.saturating_sub(1)).max(1);
    let seed_exact = context.config.seed_exact.min(rows.saturating_sub(1));
    let mut telemetry = LeafAdSamplingTelemetry {
        layout: layout_duration,
        wavefront_pairmask: true,
        ..LeafAdSamplingTelemetry::default()
    };

    let seed_start = Instant::now();
    let (mut top, seeded, seed_evals) =
        initialize_leaf_wavefront_mutual_seed(&context.layout, k, seed_exact);
    telemetry.seed += seed_start.elapsed();
    telemetry.seed_evals = seed_evals;

    let scan_start = Instant::now();
    let mut counters = LeafWavefrontPairMaskCounters::new();
    let mut scratch = LeafWavefrontPairMaskScratch::new();
    for wave in 0..context.layout.blocks.len() {
        for left_block_idx in 0..wave {
            run_leaf_wavefront_offdiag_pair(
                context,
                left_block_idx,
                wave,
                &seeded,
                &mut top,
                &mut scratch,
                &mut counters,
            );
        }
        run_leaf_wavefront_diagonal_block(
            context,
            wave,
            &seeded,
            &mut top,
            &mut scratch,
            &mut counters,
        );
    }
    telemetry.scan += scan_start.elapsed();
    telemetry.group_evals = counters.group_evals;
    telemetry.full_evals = counters.full_evals;
    telemetry.pruned_evals = counters.pruned_evals;
    telemetry.simd_group_calls = counters.simd_group_calls;
    telemetry.simd_active_lane_evals = counters.simd_active_lane_evals;
    telemetry.scalar_group_evals = counters.scalar_group_evals;

    LeafAdSamplingResult {
        row_topk: top.into_iter().map(|top| top.entries()).collect(),
        telemetry,
    }
}

fn initialize_leaf_wavefront_mutual_seed(
    layout: &AdSamplingLeafLayout<'_>,
    k: usize,
    seed_exact: usize,
) -> (Vec<TinyTopK>, Vec<u8>, u64) {
    let rows = layout.rows;
    let mut top = vec![TinyTopK::new(k); rows];
    let mut seeded = vec![0_u8; rows.saturating_mul(rows)];
    let mut seed_evals = 0_u64;
    for row in 0..rows {
        for offset in 1..=seed_exact {
            let candidate = (row + offset) % rows;
            let left = row.min(candidate);
            let right = row.max(candidate);
            if left == right || seeded[left * rows + right] != 0 {
                continue;
            }
            seeded[left * rows + right] = 1;
            seeded[right * rows + left] = 1;
            let dist = l2_sq(layout.row(left), layout.row(right));
            top[left].push(dist, right);
            top[right].push(dist, left);
            seed_evals += 1;
        }
    }
    (top, seeded, seed_evals)
}

fn run_leaf_wavefront_offdiag_pair(
    context: &LeafAdsContext<'_>,
    left_block_idx: usize,
    right_block_idx: usize,
    seeded: &[u8],
    top: &mut [TinyTopK],
    scratch: &mut LeafWavefrontPairMaskScratch,
    counters: &mut LeafWavefrontPairMaskCounters,
) {
    debug_assert!(left_block_idx < right_block_idx);
    let layout = &context.layout;
    let left_block = &layout.blocks[left_block_idx];
    let right_block = &layout.blocks[right_block_idx];

    for row in scratch.distances.iter_mut().take(left_block.len) {
        row.fill(0.0);
    }
    scratch.survivor_masks.fill(0);
    scratch.left_thresholds.fill(f32::INFINITY);
    scratch.right_thresholds.fill(f32::INFINITY);

    for left_local in 0..left_block.len {
        let left = left_block.start + left_local;
        let mut active = 0_u64;
        for right_local in 0..right_block.len {
            let right = right_block.start + right_local;
            if !leaf_wavefront_seeded_pair(seeded, layout.rows, left, right) {
                active |= 1_u64 << right_local;
            }
        }
        scratch.survivor_masks[left_local] = active;
        scratch.left_thresholds[left_local] = top[left].threshold();
    }
    for right_local in 0..right_block.len {
        scratch.right_thresholds[right_local] = top[right_block.start + right_local].threshold();
    }

    for start_dim in (0..layout.dim).step_by(context.ratios.group_dims()) {
        if scratch.survivor_masks[..left_block.len]
            .iter()
            .all(|&mask| mask == 0)
        {
            break;
        }
        let end_dim = (start_dim + context.ratios.group_dims()).min(layout.dim);
        for left_tile_start in (0..left_block.len).step_by(ADS_QUERY_TILE) {
            let query_count = (left_block.len - left_tile_start).min(ADS_QUERY_TILE);
            let active_masks =
                leaf_wavefront_tile_masks(&scratch.survivor_masks, left_tile_start, query_count);
            let active_count = active_masks[..query_count]
                .iter()
                .map(|mask| mask.count_ones() as u64)
                .sum::<u64>();
            if active_count == 0 {
                continue;
            }

            let first_query = layout.row(left_block.start + left_tile_start);
            let mut queries = [first_query; ADS_QUERY_TILE];
            for query_idx in 0..query_count {
                queries[query_idx] = layout.row(left_block.start + left_tile_start + query_idx);
            }
            let simd = accumulate_leaf_wavefront_pair_batch8(
                queries,
                query_count,
                right_block,
                start_dim,
                end_dim,
                active_masks,
                active_count as u32,
                &mut scratch.distances,
                left_tile_start,
                context.config.sparse_full64_threshold,
                context.config.sparse_group16_threshold,
            );
            counters.add_group_work(active_count, active_masks, simd);

            let ratio = context.ratios.ratio_after_visited(end_dim);
            for query_idx in 0..query_count {
                let left_local = left_tile_start + query_idx;
                let before = scratch.survivor_masks[left_local];
                let after = prune_leaf_wavefront_pair_mask(
                    before,
                    &scratch.distances[left_local],
                    scratch.left_thresholds[left_local],
                    &scratch.right_thresholds,
                    ratio,
                );
                let pruned_pairs = (before & !after).count_ones() as u64;
                scratch.survivor_masks[left_local] = after;
                counters.pruned_evals += pruned_pairs.saturating_mul(2);
            }
        }
    }

    for left_local in 0..left_block.len {
        let left = left_block.start + left_local;
        let mut survivors = scratch.survivor_masks[left_local];
        while survivors != 0 {
            let right_local = survivors.trailing_zeros() as usize;
            let right = right_block.start + right_local;
            let dist = scratch.distances[left_local][right_local];
            top[left].push(dist, right);
            top[right].push(dist, left);
            counters.full_evals += 1;
            survivors &= survivors - 1;
        }
    }
}

fn run_leaf_wavefront_diagonal_block(
    context: &LeafAdsContext<'_>,
    block_idx: usize,
    seeded: &[u8],
    top: &mut [TinyTopK],
    scratch: &mut LeafWavefrontPairMaskScratch,
    counters: &mut LeafWavefrontPairMaskCounters,
) {
    let layout = &context.layout;
    let block = &layout.blocks[block_idx];
    for row_tile_start in (0..block.len).step_by(ADS_QUERY_TILE) {
        let query_count = (block.len - row_tile_start).min(ADS_QUERY_TILE);
        for row in scratch.distances.iter_mut().take(query_count) {
            row.fill(0.0);
        }
        scratch.survivor_masks.fill(0);

        let first_query = layout.row(block.start + row_tile_start);
        let mut queries = [first_query; ADS_QUERY_TILE];
        for query_idx in 0..query_count {
            let row = block.start + row_tile_start + query_idx;
            queries[query_idx] = layout.row(row);
            let mut active = block_len_mask(block.len);
            active &= !(1_u64 << (row - block.start));
            for local in 0..block.len {
                let candidate = block.start + local;
                if leaf_wavefront_seeded_pair(seeded, layout.rows, row, candidate) {
                    active &= !(1_u64 << local);
                }
            }
            scratch.survivor_masks[query_idx] = active;
        }

        for start_dim in (0..layout.dim).step_by(context.ratios.group_dims()) {
            if scratch.survivor_masks[..query_count]
                .iter()
                .all(|&mask| mask == 0)
            {
                break;
            }
            let end_dim = (start_dim + context.ratios.group_dims()).min(layout.dim);
            let active_masks = leaf_wavefront_tile_masks(&scratch.survivor_masks, 0, query_count);
            let active_count = active_masks[..query_count]
                .iter()
                .map(|mask| mask.count_ones() as u64)
                .sum::<u64>();
            if active_count == 0 {
                continue;
            }
            let simd = accumulate_leaf_wavefront_pair_batch8(
                queries,
                query_count,
                block,
                start_dim,
                end_dim,
                active_masks,
                active_count as u32,
                &mut scratch.distances,
                0,
                context.config.sparse_full64_threshold,
                context.config.sparse_group16_threshold,
            );
            counters.add_group_work(active_count, active_masks, simd);

            let ratio = context.ratios.ratio_after_visited(end_dim);
            for query_idx in 0..query_count {
                let row = block.start + row_tile_start + query_idx;
                let before = scratch.survivor_masks[query_idx];
                let after = prune_ads_mask(
                    before,
                    &scratch.distances[query_idx],
                    top[row].threshold() * ratio,
                );
                counters.pruned_evals += (before & !after).count_ones() as u64;
                scratch.survivor_masks[query_idx] = after;
            }
        }

        for query_idx in 0..query_count {
            let row = block.start + row_tile_start + query_idx;
            counters.full_evals += push_active_distances(
                &mut top[row],
                block,
                scratch.survivor_masks[query_idx],
                &scratch.distances[query_idx],
            );
        }
    }
}

fn leaf_wavefront_tile_masks(
    masks: &[u64; ADS_LEADER_BLOCK],
    start: usize,
    query_count: usize,
) -> [u64; ADS_QUERY_TILE] {
    let mut active = [0_u64; ADS_QUERY_TILE];
    active[..query_count].copy_from_slice(&masks[start..start + query_count]);
    active
}

#[allow(clippy::too_many_arguments)]
fn accumulate_leaf_wavefront_pair_batch8(
    points: [&[f32]; ADS_QUERY_TILE],
    query_count: usize,
    block: &AdSamplingLeaderBlock,
    start_dim: usize,
    end_dim: usize,
    active_masks: [u64; ADS_QUERY_TILE],
    active_count: u32,
    distances: &mut [[f32; ADS_LEADER_BLOCK]; ADS_LEADER_BLOCK],
    row_offset: usize,
    sparse_full64_threshold: u32,
    sparse_group16_threshold: u32,
) -> bool {
    if should_use_sparse_ads_fallback(
        &active_masks[..query_count],
        active_count,
        sparse_full64_threshold,
        sparse_group16_threshold,
    ) {
        for query_idx in 0..query_count {
            accumulate_ads_soa_group_mask_scalar(
                points[query_idx],
                block,
                start_dim,
                end_dim,
                active_masks[query_idx],
                &mut distances[row_offset + query_idx],
            );
        }
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if adsampling_avx512_available() {
            unsafe {
                accumulate_leaf_wavefront_pair_batch8_avx512(
                    points,
                    block,
                    start_dim,
                    end_dim,
                    active_masks,
                    distances,
                    row_offset,
                );
            }
            return true;
        }
    }
    for query_idx in 0..query_count {
        accumulate_ads_soa_group_mask_scalar(
            points[query_idx],
            block,
            start_dim,
            end_dim,
            active_masks[query_idx],
            &mut distances[row_offset + query_idx],
        );
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_leaf_wavefront_pair_batch8_avx512(
    points: [&[f32]; ADS_QUERY_TILE],
    block: &AdSamplingLeaderBlock,
    start_dim: usize,
    end_dim: usize,
    active_masks: [u64; ADS_QUERY_TILE],
    distances: &mut [[f32; ADS_LEADER_BLOCK]; ADS_LEADER_BLOCK],
    row_offset: usize,
) {
    use std::arch::x86_64::*;

    let point_ptrs = [
        points[0].as_ptr(),
        points[1].as_ptr(),
        points[2].as_ptr(),
        points[3].as_ptr(),
        points[4].as_ptr(),
        points[5].as_ptr(),
        points[6].as_ptr(),
        points[7].as_ptr(),
    ];

    for local in (0..ADS_LEADER_BLOCK).step_by(16) {
        let lane_masks = [
            ((active_masks[0] >> local) & 0xffff) as __mmask16,
            ((active_masks[1] >> local) & 0xffff) as __mmask16,
            ((active_masks[2] >> local) & 0xffff) as __mmask16,
            ((active_masks[3] >> local) & 0xffff) as __mmask16,
            ((active_masks[4] >> local) & 0xffff) as __mmask16,
            ((active_masks[5] >> local) & 0xffff) as __mmask16,
            ((active_masks[6] >> local) & 0xffff) as __mmask16,
            ((active_masks[7] >> local) & 0xffff) as __mmask16,
        ];
        if lane_masks.iter().all(|&mask| mask == 0) {
            continue;
        }

        let mut acc0 = unsafe { _mm512_loadu_ps(distances[row_offset].as_ptr().add(local)) };
        let mut acc1 = unsafe { _mm512_loadu_ps(distances[row_offset + 1].as_ptr().add(local)) };
        let mut acc2 = unsafe { _mm512_loadu_ps(distances[row_offset + 2].as_ptr().add(local)) };
        let mut acc3 = unsafe { _mm512_loadu_ps(distances[row_offset + 3].as_ptr().add(local)) };
        let mut acc4 = unsafe { _mm512_loadu_ps(distances[row_offset + 4].as_ptr().add(local)) };
        let mut acc5 = unsafe { _mm512_loadu_ps(distances[row_offset + 5].as_ptr().add(local)) };
        let mut acc6 = unsafe { _mm512_loadu_ps(distances[row_offset + 6].as_ptr().add(local)) };
        let mut acc7 = unsafe { _mm512_loadu_ps(distances[row_offset + 7].as_ptr().add(local)) };
        for dim_idx in start_dim..end_dim {
            let values =
                unsafe { _mm512_loadu_ps(block.data.as_ptr().add(dim_idx * block.stride + local)) };
            if lane_masks[0] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[0].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc0 = _mm512_fmadd_ps(diff, diff, acc0);
            }
            if lane_masks[1] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[1].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc1 = _mm512_fmadd_ps(diff, diff, acc1);
            }
            if lane_masks[2] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[2].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc2 = _mm512_fmadd_ps(diff, diff, acc2);
            }
            if lane_masks[3] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[3].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc3 = _mm512_fmadd_ps(diff, diff, acc3);
            }
            if lane_masks[4] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[4].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc4 = _mm512_fmadd_ps(diff, diff, acc4);
            }
            if lane_masks[5] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[5].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc5 = _mm512_fmadd_ps(diff, diff, acc5);
            }
            if lane_masks[6] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[6].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc6 = _mm512_fmadd_ps(diff, diff, acc6);
            }
            if lane_masks[7] != 0 {
                let q = _mm512_set1_ps(unsafe { *point_ptrs[7].add(dim_idx) });
                let diff = _mm512_sub_ps(q, values);
                acc7 = _mm512_fmadd_ps(diff, diff, acc7);
            }
        }
        if lane_masks[0] != 0 {
            unsafe { _mm512_storeu_ps(distances[row_offset].as_mut_ptr().add(local), acc0) };
        }
        if lane_masks[1] != 0 {
            unsafe { _mm512_storeu_ps(distances[row_offset + 1].as_mut_ptr().add(local), acc1) };
        }
        if lane_masks[2] != 0 {
            unsafe { _mm512_storeu_ps(distances[row_offset + 2].as_mut_ptr().add(local), acc2) };
        }
        if lane_masks[3] != 0 {
            unsafe { _mm512_storeu_ps(distances[row_offset + 3].as_mut_ptr().add(local), acc3) };
        }
        if lane_masks[4] != 0 {
            unsafe { _mm512_storeu_ps(distances[row_offset + 4].as_mut_ptr().add(local), acc4) };
        }
        if lane_masks[5] != 0 {
            unsafe { _mm512_storeu_ps(distances[row_offset + 5].as_mut_ptr().add(local), acc5) };
        }
        if lane_masks[6] != 0 {
            unsafe { _mm512_storeu_ps(distances[row_offset + 6].as_mut_ptr().add(local), acc6) };
        }
        if lane_masks[7] != 0 {
            unsafe { _mm512_storeu_ps(distances[row_offset + 7].as_mut_ptr().add(local), acc7) };
        }
    }
}

#[inline]
fn prune_leaf_wavefront_pair_mask(
    active_mask: u64,
    distances: &[f32; ADS_LEADER_BLOCK],
    left_threshold: f32,
    right_thresholds: &[f32; ADS_LEADER_BLOCK],
    ratio: f32,
) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if adsampling_avx512_available() {
            return unsafe {
                prune_leaf_wavefront_pair_mask_avx512(
                    active_mask,
                    distances,
                    left_threshold,
                    right_thresholds,
                    ratio,
                )
            };
        }
    }
    prune_leaf_wavefront_pair_mask_scalar(
        active_mask,
        distances,
        left_threshold,
        right_thresholds,
        ratio,
    )
}

#[inline]
fn prune_leaf_wavefront_pair_mask_scalar(
    mut active_mask: u64,
    distances: &[f32; ADS_LEADER_BLOCK],
    left_threshold: f32,
    right_thresholds: &[f32; ADS_LEADER_BLOCK],
    ratio: f32,
) -> u64 {
    let mut keep = 0_u64;
    while active_mask != 0 {
        let local = active_mask.trailing_zeros() as usize;
        if distances[local] < ratio * left_threshold.max(right_thresholds[local]) {
            keep |= 1_u64 << local;
        }
        active_mask &= active_mask - 1;
    }
    keep
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn prune_leaf_wavefront_pair_mask_avx512(
    active_mask: u64,
    distances: &[f32; ADS_LEADER_BLOCK],
    left_threshold: f32,
    right_thresholds: &[f32; ADS_LEADER_BLOCK],
    ratio: f32,
) -> u64 {
    use std::arch::x86_64::*;

    let left = _mm512_set1_ps(left_threshold);
    let ratio = _mm512_set1_ps(ratio);
    let mut keep_mask = 0_u64;
    for local in (0..ADS_LEADER_BLOCK).step_by(16) {
        let lane_active = ((active_mask >> local) & 0xffff) as __mmask16;
        if lane_active == 0 {
            continue;
        }
        let dist = unsafe { _mm512_loadu_ps(distances.as_ptr().add(local)) };
        let right = unsafe { _mm512_loadu_ps(right_thresholds.as_ptr().add(local)) };
        let threshold = _mm512_mul_ps(_mm512_max_ps(left, right), ratio);
        let keep = _mm512_cmp_ps_mask(dist, threshold, _CMP_LT_OQ) & lane_active;
        keep_mask |= (keep as u64) << local;
    }
    keep_mask
}

#[inline]
fn leaf_wavefront_seeded_pair(seeded: &[u8], rows: usize, row: usize, candidate: usize) -> bool {
    seeded
        .get(row.saturating_mul(rows).saturating_add(candidate))
        .copied()
        .unwrap_or(0)
        != 0
}

#[inline]
fn should_parallelize_leaf_adsampling_rows(leaf_size: usize) -> bool {
    leaf_size >= ADS_LEAF_PARALLEL_MIN_ROWS && rayon::current_thread_index().is_none()
}

#[derive(Debug)]
struct LeafAdSamplingChunk {
    row_start: usize,
    row_end: usize,
    row_topk: Vec<Vec<(usize, f32)>>,
    telemetry: LeafAdSamplingTelemetry,
}

fn compute_leaf_adsampling_row_range(
    layout: &AdSamplingLeafLayout<'_>,
    row_start: usize,
    row_end: usize,
    k: usize,
    config: AdSamplingConfig,
    ratios: &AdSamplingRatios,
) -> LeafAdSamplingChunk {
    let mut chunk = LeafAdSamplingChunk {
        row_start,
        row_end,
        row_topk: Vec::with_capacity(row_end - row_start),
        telemetry: LeafAdSamplingTelemetry::default(),
    };
    for tile_start in (row_start..row_end).step_by(ADS_QUERY_TILE) {
        let tile_end = (tile_start + ADS_QUERY_TILE).min(row_end);
        let tile =
            compute_leaf_adsampling_row_tile(layout, tile_start, tile_end, k, config, ratios);
        chunk.telemetry.merge(tile.telemetry);
        chunk.row_topk.extend(tile.row_topk);
    }
    chunk
}

fn compute_leaf_adsampling_topk_l2_tiled(
    context: &LeafAdsContext<'_>,
    leaf_size: usize,
    k: usize,
    layout_duration: Duration,
    runtime: &LeafAdsOperatorRuntime,
) -> LeafAdSamplingResult {
    if runtime.work_graph_enabled() {
        return compute_leaf_adsampling_topk_l2_work_graph(
            context,
            leaf_size,
            k,
            layout_duration,
            runtime,
        );
    }

    let _context_guard = runtime.begin_context();
    let bucket = leaf_ads_size_bucket(leaf_size);
    let initial_tile_rows = runtime.tile_rows_for_bucket(bucket).max(ADS_QUERY_TILE);
    let max_tiles = leaf_size.div_ceil(initial_tile_rows).max(1);
    let worker_cap = runtime
        .worker_cap_for_current_contexts()
        .min(max_tiles)
        .max(1);
    let (reservation, worker_wait) = runtime.reserve_workers(worker_cap);
    let worker_count = reservation.count.max(1);
    let cursor = AtomicUsize::new(0);
    let chunks = Mutex::new(Vec::<LeafAdSamplingChunk>::new());

    rayon::scope(|scope| {
        for _ in 1..worker_count {
            scope.spawn(|_| {
                run_leaf_ads_tile_worker(context, k, runtime, bucket, &cursor, &chunks);
            });
        }
        run_leaf_ads_tile_worker(context, k, runtime, bucket, &cursor, &chunks);
    });

    let chunks = chunks
        .into_inner()
        .expect("leaf ADS tile worker mutex should not be poisoned");
    finalize_leaf_ads_tiles(chunks, layout_duration, worker_wait, runtime)
}

fn compute_leaf_adsampling_topk_l2_work_graph(
    context: &LeafAdsContext<'_>,
    leaf_size: usize,
    k: usize,
    layout_duration: Duration,
    runtime: &LeafAdsOperatorRuntime,
) -> LeafAdSamplingResult {
    let _context_guard = runtime.begin_context();
    update_atomic_max(&runtime.active_workers_peak, 1);
    let bucket = leaf_ads_size_bucket(leaf_size);
    let quantum = Duration::from_millis(runtime.target_tile_ms().max(1));
    let cursor = AtomicUsize::new(0);
    let mut chunks = Vec::with_capacity(leaf_size.div_ceil(runtime.tile_rows_for_bucket(bucket)));
    let mut handle_requeues = 0usize;

    loop {
        let handle_start = Instant::now();
        loop {
            let tile_rows = runtime.tile_rows_for_bucket(bucket).max(ADS_QUERY_TILE);
            let row_start = cursor.fetch_add(tile_rows, Ordering::Relaxed);
            if row_start >= context.layout.rows {
                let mut result =
                    finalize_leaf_ads_tiles(chunks, layout_duration, Duration::ZERO, runtime);
                result.telemetry.work_graph = true;
                result.telemetry.tile_handle_requeues = handle_requeues;
                return result;
            }
            let row_end = (row_start + tile_rows).min(context.layout.rows);
            let wall_start = Instant::now();
            let mut chunk = compute_leaf_adsampling_row_range(
                &context.layout,
                row_start,
                row_end,
                k,
                context.config,
                &context.ratios,
            );
            let wall = wall_start.elapsed();
            chunk.telemetry.tile_count = 1;
            chunk.telemetry.tile_rows_total = row_end - row_start;
            chunk.telemetry.tile_rows_min = row_end - row_start;
            chunk.telemetry.tile_rows_max = row_end - row_start;
            chunk.telemetry.tile_wall = wall;
            chunk.telemetry.work_graph = true;
            runtime.record_tile(bucket, row_end - row_start, wall);
            chunks.push(chunk);
            if handle_start.elapsed() >= quantum {
                break;
            }
        }

        if cursor.load(Ordering::Relaxed) >= context.layout.rows {
            let mut result =
                finalize_leaf_ads_tiles(chunks, layout_duration, Duration::ZERO, runtime);
            result.telemetry.work_graph = true;
            result.telemetry.tile_handle_requeues = handle_requeues;
            return result;
        }
        handle_requeues += 1;
        if rayon::yield_now().is_none() {
            std::thread::yield_now();
        }
    }
}

fn run_leaf_ads_tile_worker(
    context: &LeafAdsContext<'_>,
    k: usize,
    runtime: &LeafAdsOperatorRuntime,
    bucket: usize,
    cursor: &AtomicUsize,
    chunks: &Mutex<Vec<LeafAdSamplingChunk>>,
) {
    let mut local_chunks = Vec::new();
    loop {
        let tile_rows = runtime.tile_rows_for_bucket(bucket).max(ADS_QUERY_TILE);
        let row_start = cursor.fetch_add(tile_rows, Ordering::Relaxed);
        if row_start >= context.layout.rows {
            break;
        }
        let row_end = (row_start + tile_rows).min(context.layout.rows);
        let wall_start = Instant::now();
        let mut chunk = compute_leaf_adsampling_row_range(
            &context.layout,
            row_start,
            row_end,
            k,
            context.config,
            &context.ratios,
        );
        let wall = wall_start.elapsed();
        chunk.telemetry.tile_count = 1;
        chunk.telemetry.tile_rows_total = row_end - row_start;
        chunk.telemetry.tile_rows_min = row_end - row_start;
        chunk.telemetry.tile_rows_max = row_end - row_start;
        chunk.telemetry.tile_wall = wall;
        runtime.record_tile(bucket, row_end - row_start, wall);
        local_chunks.push(chunk);
    }
    if !local_chunks.is_empty() {
        chunks
            .lock()
            .expect("leaf ADS tile worker mutex should not be poisoned")
            .extend(local_chunks);
    }
}

fn finalize_leaf_ads_tiles(
    mut chunks: Vec<LeafAdSamplingChunk>,
    layout_duration: Duration,
    tile_wait: Duration,
    runtime: &LeafAdsOperatorRuntime,
) -> LeafAdSamplingResult {
    chunks.sort_unstable_by_key(|chunk| chunk.row_start);
    let row_count = chunks.iter().map(|chunk| chunk.row_topk.len()).sum();
    let mut row_topk = Vec::with_capacity(row_count);
    let mut telemetry = LeafAdSamplingTelemetry {
        layout: layout_duration,
        tiled: true,
        tile_wait,
        cpu_budget: runtime.cpu_budget(),
        active_context_peak: runtime.active_context_peak(),
        active_workers_peak: runtime.active_workers_peak(),
        ewma_ns_per_row_by_bucket: runtime.ewma_snapshot(),
        ..LeafAdSamplingTelemetry::default()
    };

    let mut expected_start = 0usize;
    for chunk in chunks {
        debug_assert_eq!(chunk.row_start, expected_start);
        debug_assert_eq!(chunk.row_end - chunk.row_start, chunk.row_topk.len());
        expected_start = chunk.row_end;
        telemetry.merge(chunk.telemetry);
        row_topk.extend(chunk.row_topk);
    }
    LeafAdSamplingResult {
        row_topk,
        telemetry,
    }
}

#[derive(Debug)]
struct LeafAdSamplingTile {
    row_topk: Vec<Vec<(usize, f32)>>,
    telemetry: LeafAdSamplingTelemetry,
}

fn compute_leaf_adsampling_row_tile(
    layout: &AdSamplingLeafLayout<'_>,
    row_start: usize,
    row_end: usize,
    k: usize,
    config: AdSamplingConfig,
    ratios: &AdSamplingRatios,
) -> LeafAdSamplingTile {
    debug_assert!(row_end > row_start);
    debug_assert!(row_end - row_start <= ADS_QUERY_TILE);
    let query_count = row_end - row_start;
    let k = k.min(layout.rows.saturating_sub(1)).max(1);
    let seed_exact = config.seed_exact.min(layout.rows.saturating_sub(1));
    let mut telemetry = LeafAdSamplingTelemetry::default();
    let mut top: [TinyTopK; ADS_QUERY_TILE] = std::array::from_fn(|_| TinyTopK::new(k));

    let seed_start = Instant::now();
    for (query_idx, row) in (row_start..row_end).enumerate() {
        let query = layout.row(row);
        for offset in 1..=seed_exact {
            let candidate = (row + offset) % layout.rows;
            top[query_idx].push(l2_sq(query, layout.row(candidate)), candidate);
            telemetry.seed_evals += 1;
        }
    }
    telemetry.seed += seed_start.elapsed();

    let first_query = layout.row(row_start);
    let mut queries = [first_query; ADS_QUERY_TILE];
    for (query_idx, row) in (row_start..row_end).enumerate() {
        queries[query_idx] = layout.row(row);
    }

    let scan_start = Instant::now();
    let mut distances = [[0.0f32; ADS_LEADER_BLOCK]; ADS_QUERY_TILE];
    for block in &layout.blocks {
        for row in distances.iter_mut().take(query_count) {
            row.fill(0.0);
        }
        let mut active_masks = [0_u64; ADS_QUERY_TILE];
        let mut active_total = 0_u64;
        for (query_idx, row) in (row_start..row_end).enumerate() {
            let mask =
                leaf_adsampling_active_mask(row, block.start, block.len, layout.rows, seed_exact);
            active_masks[query_idx] = mask;
            active_total += mask.count_ones() as u64;
        }
        if active_total == 0 {
            continue;
        }

        for start_dim in (0..layout.dim).step_by(ratios.group_dims()) {
            if active_total == 0 {
                break;
            }
            let end_dim = (start_dim + ratios.group_dims()).min(layout.dim);
            let active_before = active_masks;
            let active_count = active_total;
            telemetry.group_evals += active_count;
            if accumulate_ads_soa_group_batch8_with_active_count(
                queries,
                query_count,
                block,
                start_dim,
                end_dim,
                active_before,
                active_count as u32,
                &mut distances,
                config.sparse_full64_threshold,
                config.sparse_group16_threshold,
            ) {
                telemetry.simd_group_calls += simd_batch8_group_call_count(active_before);
                telemetry.simd_active_lane_evals += active_count;
            } else {
                telemetry.scalar_group_evals += active_count;
            }

            let ratio = ratios.ratio_after_visited(end_dim);
            let mut pruned_total = 0_u64;
            for query_idx in 0..query_count {
                let before_mask = active_masks[query_idx];
                let after_mask = prune_ads_mask(
                    before_mask,
                    &distances[query_idx],
                    top[query_idx].threshold() * ratio,
                );
                let pruned = (before_mask & !after_mask).count_ones() as u64;
                active_masks[query_idx] = after_mask;
                telemetry.pruned_evals += pruned;
                pruned_total += pruned;
            }
            debug_assert!(pruned_total <= active_total);
            active_total -= pruned_total;
        }

        for query_idx in 0..query_count {
            telemetry.full_evals += push_unique_active_distances(
                &mut top[query_idx],
                block,
                active_masks[query_idx],
                &distances[query_idx],
            );
        }
    }
    telemetry.scan += scan_start.elapsed();

    let row_topk = top
        .into_iter()
        .take(query_count)
        .map(|top| top.entries())
        .collect();
    LeafAdSamplingTile {
        row_topk,
        telemetry,
    }
}

#[inline]
fn leaf_adsampling_active_mask(
    row: usize,
    block_start: usize,
    block_len: usize,
    leaf_size: usize,
    seed_exact: usize,
) -> u64 {
    let mut mask = block_len_mask(block_len);
    let block_end = block_start + block_len;
    if row >= block_start && row < block_end {
        mask &= !(1_u64 << (row - block_start));
    }
    if leaf_size == 0 {
        return mask;
    }

    let seed_exact = seed_exact.min(leaf_size.saturating_sub(1));
    if seed_exact == 0 {
        return mask;
    }

    let seed_start = row + 1;
    let seed_end = seed_start + seed_exact;
    if seed_end <= leaf_size {
        clear_block_mask_range(&mut mask, block_start, block_end, seed_start, seed_end);
    } else {
        clear_block_mask_range(&mut mask, block_start, block_end, seed_start, leaf_size);
        clear_block_mask_range(&mut mask, block_start, block_end, 0, seed_end - leaf_size);
    }
    mask
}

#[inline]
pub(crate) fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[inline]
fn duration_nanos_u64(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn update_atomic_max(max_value: &AtomicUsize, value: usize) {
    let mut current = max_value.load(Ordering::Relaxed);
    while value > current {
        match max_value.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Read, Write};
    use std::os::unix::fs::FileExt;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::super::point_store::{PointBatchStats, PointStore, WindowedGatherOptions};
    use super::*;

    #[test]
    fn adsampling_ratio_matches_pdx_formula() {
        let ratios = AdSamplingRatios::new(128, 1.75, 32).unwrap();

        let visited = 32usize;
        let expected = visited as f32 / 128.0 * (1.0 + 1.75 / (visited as f32).sqrt()).powi(2);

        assert!((ratios.ratio_after_visited(visited) - expected).abs() < 1.0e-6);
        assert_eq!(ratios.ratio_after_visited(0), 1.0);
        assert_eq!(ratios.ratio_after_visited(128), 1.0);
    }

    #[test]
    fn tiny_topk_uses_compact_index_storage() {
        assert!(std::mem::size_of::<TinyTopK>() <= 288);

        let mut top = TinyTopK::new(2);
        top.push(2.0, 3);
        top.push_unique(1.0, 7);

        assert_eq!(top.entries(), vec![(7, 1.0), (3, 2.0)]);
    }

    #[test]
    fn leaf_adsampling_active_mask_matches_scalar_seed_exclusion() {
        for leaf_size in [1_usize, 2, 7, 63, 64, 65, 129] {
            for seed_exact in [0_usize, 1, 2, 8, 63, 128] {
                for row in 0..leaf_size {
                    for block_start in (0..leaf_size).step_by(13) {
                        let block_len = (leaf_size - block_start).min(64);
                        let actual = leaf_adsampling_active_mask(
                            row,
                            block_start,
                            block_len,
                            leaf_size,
                            seed_exact,
                        );

                        let mut expected = block_len_mask(block_len);
                        if row >= block_start && row < block_start + block_len {
                            expected &= !(1_u64 << (row - block_start));
                        }
                        let seed_exact = seed_exact.min(leaf_size.saturating_sub(1));
                        if leaf_size != 0 {
                            for offset in 1..=seed_exact {
                                let candidate = (row + offset) % leaf_size;
                                if candidate >= block_start && candidate < block_start + block_len {
                                    expected &= !(1_u64 << (candidate - block_start));
                                }
                            }
                        }

                        assert_eq!(
                            actual, expected,
                            "leaf_size={leaf_size} seed_exact={seed_exact} row={row} block_start={block_start} block_len={block_len}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn adsampling_windowed_options_coalesce_direct_io_gaps() {
        struct DirectLikePointStore {
            coalesced: bool,
        }

        impl PointStore for DirectLikePointStore {
            fn len(&self) -> usize {
                0
            }

            fn dim(&self) -> usize {
                768
            }

            fn read_point_into(&self, _pid: u32, _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_range_into(
                &self,
                _start_pid: u32,
                _count: usize,
                _out: &mut [f32],
            ) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_points_into(&self, _ids: &[u32], _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn prefers_coalesced_window_reads(&self) -> bool {
                self.coalesced
            }
        }

        let buffered = DirectLikePointStore { coalesced: false };
        let strict_direct = DirectLikePointStore { coalesced: true };

        let buffered_options = adsampling_windowed_options(&buffered, 16_384);
        let strict_options = adsampling_windowed_options(&strict_direct, 16_384);

        assert_eq!(buffered_options.max_gap_rows, 1);
        assert!(
            strict_options.max_gap_rows >= 1024,
            "direct ADSampling sidecar reads should spend RSS on gap rows to avoid tiny direct reads"
        );
        assert!(strict_options.max_window_bytes >= buffered_options.max_window_bytes);
    }

    #[test]
    fn root_adsampling_matches_exact_with_conservative_epsilon() {
        let dim = 8usize;
        let leaders = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
        ];
        let query = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = AdSamplingConfig {
            epsilon: 100.0,
            group_dims: 2,
            seed_exact: 1,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 0,
        };

        let result = assign_one_adsampling(&query, &leaders, dim, 2, config).unwrap();
        let exact = exact_topk(&query, &leaders, dim, 2);

        assert_eq!(result.leaders, exact);
        assert_eq!(
            result.full_evals + result.pruned_evals,
            (leaders.len() / dim) as u64
        );
    }

    #[test]
    fn root_adsampling_counts_mismatch_when_pruning_is_too_aggressive() {
        let dim = 4usize;
        let leaders = [0.0, 0.0, 10.0, 10.0, 8.0, 0.0, 0.0, 0.0];
        let query = [0.0, 0.0, 0.0, 0.0];
        let config = AdSamplingConfig {
            epsilon: 0.01,
            group_dims: 1,
            seed_exact: 1,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 1,
        };

        let outcome = assign_one_adsampling(&query, &leaders, dim, 1, config).unwrap();
        assert!(outcome.validation_mismatch);
    }

    #[test]
    fn root_adsampling_validation_records_topk_recall() {
        let dim = 4usize;
        let leaders = [
            0.0, 0.0, 10.0, 10.0, // seeded, not exact top-2
            0.1, 0.0, 0.0, 0.0, // seeded, exact top-1
            8.0, 0.0, 0.0, 0.0, // exact top-2, pruned aggressively
        ];
        let query = [0.0, 0.0, 0.0, 0.0];
        let config = AdSamplingConfig {
            epsilon: 0.01,
            group_dims: 1,
            seed_exact: 2,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 1,
        };

        let outcome = assign_one_adsampling(&query, &leaders, dim, 2, config).unwrap();

        assert!(outcome.validation_mismatch);
        assert_eq!(outcome.validation_recall_hits, 1);
        assert_eq!(outcome.validation_recall_total, 2);
        assert_eq!(outcome.validation_recall_at_fanout, 0.5);
    }

    #[test]
    fn root_adsampling_validation_sampling_respects_limit_stride_and_offset() {
        assert!(should_validate_source_with(3, 1, 0, 0));
        assert!(should_validate_source_with(3, 1, 0, 2));
        assert!(!should_validate_source_with(3, 1, 0, 3));

        assert!(!should_validate_source_with(3, 10, 5, 4));
        assert!(should_validate_source_with(3, 10, 5, 5));
        assert!(should_validate_source_with(3, 10, 5, 15));
        assert!(should_validate_source_with(3, 10, 5, 25));
        assert!(!should_validate_source_with(3, 10, 5, 35));
        assert!(!should_validate_source_with(3, 10, 5, 16));
    }

    #[test]
    fn depth_adsampling_gate_uses_production_guards() {
        let params = ForgeANNParams::default();

        assert!(!should_use_adsampling_depth(&params, 0, 8, 4, 2));
        assert!(should_use_adsampling_depth(&params, 1, 8, 4, 2));
        assert!(should_use_adsampling_depth(&params, 2, 8, 4, 2));
        assert!(!should_use_adsampling_depth(&params, 3, 8, 4, 2));
        assert!(!should_use_adsampling_depth(&params, 1, 0, 4, 2));
        assert!(!should_use_adsampling_depth(&params, 1, 8, 2, 2));
        assert!(!should_use_adsampling_depth(&params, 1, 8, 4, 1));
    }

    #[test]
    fn depth_adsampling_gate_reports_fallback_reason() {
        let params = ForgeANNParams::default();

        assert_eq!(adsampling_depth_fallback_reason(&params, 1, 8, 4, 2), None);
        assert_eq!(
            adsampling_depth_fallback_reason(&params, 3, 8, 4, 2),
            Some("max-depth")
        );
        assert_eq!(
            adsampling_depth_fallback_reason(&params, 1, 0, 4, 2),
            Some("empty-points")
        );
        assert_eq!(
            adsampling_depth_fallback_reason(&params, 1, 8, 2, 2),
            Some("leaders-le-fanout")
        );
        assert_eq!(
            adsampling_depth_fallback_reason(&params, 1, 8, 4, 1),
            Some("fanout")
        );
        assert_eq!(adsampling_depth_fallback_reason(&params, 1, 8, 4, 2), None);
    }

    #[test]
    fn root_adsampling_layout_matches_scalar_and_records_vector_work() {
        let dim = 64usize;
        let leader_count = 80usize;
        let leaders = (0..leader_count)
            .flat_map(|row| (0..dim).map(move |col| ((row * 17 + col * 13) % 97) as f32 * 0.01))
            .collect::<Vec<_>>();
        let query = (0..dim)
            .map(|col| ((col * 19 + 7) % 89) as f32 * 0.01)
            .collect::<Vec<_>>();
        let config = AdSamplingConfig {
            epsilon: 100.0,
            group_dims: 16,
            seed_exact: 8,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 1,
        };

        let layout =
            AdSamplingLeaderLayout::from_row_major_for_test(leaders.clone(), leader_count, dim);
        let layout_result = assign_one_adsampling_layout(&query, &layout, 4, config).unwrap();
        let scalar_result = assign_one_adsampling(&query, &leaders, dim, 4, config).unwrap();

        assert_eq!(layout_result.leaders, scalar_result.leaders);
        assert!(!layout_result.validation_mismatch);
        if adsampling_avx512_available() {
            assert!(layout_result.simd_group_calls > 0);
        }
    }

    #[test]
    fn leaf_adsampling_records_vector_work() {
        let dim = 64usize;
        let leaf_size = 96usize;
        let vectors = (0..leaf_size)
            .flat_map(|row| (0..dim).map(move |col| ((row * 31 + col * 7) % 113) as f32 * 0.01))
            .collect::<Vec<_>>();
        let config = AdSamplingConfig {
            epsilon: 100.0,
            group_dims: 16,
            seed_exact: 8,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 0,
        };

        let result = compute_leaf_adsampling_topk_l2(&vectors, leaf_size, dim, 3, config).unwrap();
        let expected = exact_leaf_topk(&vectors, leaf_size, dim, 3);

        assert_leaf_topk_close(&result.row_topk, &expected);
        if adsampling_avx512_available() {
            assert!(result.telemetry.simd_group_calls > 0);
            assert_eq!(result.telemetry.scalar_group_evals, 0);
        }
    }

    #[test]
    fn tiled_leaf_adsampling_matches_default_row_order() {
        let dim = 64usize;
        let leaf_size = 640usize;
        let vectors = (0..leaf_size)
            .flat_map(|row| (0..dim).map(move |col| ((row * 17 + col * 29) % 131) as f32 * 0.01))
            .collect::<Vec<_>>();
        let config = AdSamplingConfig {
            epsilon: 100.0,
            group_dims: 16,
            seed_exact: 8,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 0,
        };
        let runtime = LeafAdsOperatorRuntime::new(LeafAdsOperatorConfig {
            enabled: true,
            work_graph: false,
            cpu_budget: 4,
            target_tile_ms: 1,
            min_tile_rows: 64,
            max_tile_rows: 64,
            split_threshold: 64,
        });

        let default =
            compute_leaf_adsampling_topk_l2_with_runtime(&vectors, leaf_size, dim, 3, config, None)
                .unwrap();
        let tiled = compute_leaf_adsampling_topk_l2_with_runtime(
            &vectors,
            leaf_size,
            dim,
            3,
            config,
            Some(&runtime),
        )
        .unwrap();

        assert_leaf_topk_close(&tiled.row_topk, &default.row_topk);
        assert!(tiled.telemetry.tiled);
        assert!(tiled.telemetry.tile_count > 1);
        assert_eq!(tiled.telemetry.tile_rows_min, 64);
        assert_eq!(tiled.telemetry.tile_rows_max, 64);
        assert_eq!(tiled.telemetry.tile_rows_total, leaf_size);
        assert!(tiled.telemetry.ewma_ns_per_row_by_bucket[0] > 0);
    }

    #[test]
    fn work_graph_leaf_adsampling_matches_default_without_nested_workers() {
        let dim = 64usize;
        let leaf_size = 768usize;
        let vectors = (0..leaf_size)
            .flat_map(|row| (0..dim).map(move |col| ((row * 19 + col * 23) % 149) as f32 * 0.01))
            .collect::<Vec<_>>();
        let config = AdSamplingConfig {
            epsilon: 100.0,
            group_dims: 16,
            seed_exact: 8,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 0,
        };
        let runtime = LeafAdsOperatorRuntime::new(LeafAdsOperatorConfig {
            enabled: true,
            work_graph: true,
            cpu_budget: 4,
            target_tile_ms: 1,
            min_tile_rows: 256,
            max_tile_rows: 256,
            split_threshold: 64,
        });

        let default =
            compute_leaf_adsampling_topk_l2_with_runtime(&vectors, leaf_size, dim, 3, config, None)
                .unwrap();
        let work_graph = compute_leaf_adsampling_topk_l2_with_runtime(
            &vectors,
            leaf_size,
            dim,
            3,
            config,
            Some(&runtime),
        )
        .unwrap();

        assert_leaf_topk_close(&work_graph.row_topk, &default.row_topk);
        assert!(work_graph.telemetry.tiled);
        assert!(work_graph.telemetry.work_graph);
        assert_eq!(work_graph.telemetry.tile_rows_total, leaf_size);
        assert_eq!(work_graph.telemetry.tile_rows_min, 256);
        assert_eq!(work_graph.telemetry.tile_rows_max, 256);
        assert_eq!(work_graph.telemetry.active_workers_peak, 1);
    }

    #[test]
    fn leaf_adsampling_avoids_nested_parallelism_inside_rayon_worker() {
        assert!(
            ADS_LEAF_PARALLEL_MIN_ROWS <= 512,
            "leaf ADS should parallelize 512-1024 row leaves used by the io-planned full build"
        );
        assert!(
            ADS_LEAF_ROW_CHUNK <= 64,
            "row chunks must be small enough to keep workers busy when only a few leaves remain"
        );
        assert!(should_parallelize_leaf_adsampling_rows(
            ADS_LEAF_PARALLEL_MIN_ROWS
        ));

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        assert!(
            !pool.install(|| should_parallelize_leaf_adsampling_rows(ADS_LEAF_PARALLEL_MIN_ROWS))
        );
    }

    #[test]
    fn tiled_leaf_adsampling_can_run_inside_rayon_worker() {
        let dim = 32usize;
        let leaf_size = ADS_LEAF_PARALLEL_MIN_ROWS;
        let vectors = (0..leaf_size)
            .flat_map(|row| (0..dim).map(move |col| ((row * 11 + col * 5) % 97) as f32 * 0.01))
            .collect::<Vec<_>>();
        let config = AdSamplingConfig {
            epsilon: 100.0,
            group_dims: 16,
            seed_exact: 8,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 0,
        };
        let runtime = LeafAdsOperatorRuntime::new(LeafAdsOperatorConfig {
            enabled: true,
            work_graph: false,
            cpu_budget: 2,
            target_tile_ms: 1,
            min_tile_rows: 64,
            max_tile_rows: 64,
            split_threshold: ADS_LEAF_PARALLEL_MIN_ROWS,
        });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let result = pool
            .install(|| {
                compute_leaf_adsampling_topk_l2_with_runtime(
                    &vectors,
                    leaf_size,
                    dim,
                    3,
                    config,
                    Some(&runtime),
                )
            })
            .unwrap();

        assert!(result.telemetry.tiled);
        assert!(result.telemetry.tile_count > 1);
        assert!(result.telemetry.active_workers_peak >= 1);
    }

    #[test]
    fn root_adsampling_chunk_loads_queries_with_windowed_batch_reads() {
        struct CountingPointStore {
            rows: Vec<f32>,
            dim: usize,
            plain_calls: AtomicUsize,
            windowed_batch_calls: AtomicUsize,
        }

        impl PointStore for CountingPointStore {
            fn len(&self) -> usize {
                self.rows.len() / self.dim
            }

            fn dim(&self) -> usize {
                self.dim
            }

            fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
                let start = pid as usize * self.dim;
                out.copy_from_slice(&self.rows[start..start + self.dim]);
                Ok(())
            }

            fn read_range_into(
                &self,
                start_pid: u32,
                count: usize,
                out: &mut [f32],
            ) -> AnnResult<()> {
                let start = start_pid as usize * self.dim;
                let end = start + count * self.dim;
                out.copy_from_slice(&self.rows[start..end]);
                Ok(())
            }

            fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
                self.plain_calls.fetch_add(1, Ordering::Relaxed);
                for (row, &pid) in ids.iter().enumerate() {
                    let src = pid as usize * self.dim;
                    let dst = row * self.dim;
                    out[dst..dst + self.dim].copy_from_slice(&self.rows[src..src + self.dim]);
                }
                Ok(())
            }

            fn read_points_windowed_into_batch_stats(
                &self,
                ids: &[u32],
                out: &mut [f32],
                _options: &WindowedGatherOptions,
                _stats: &mut PointBatchStats,
            ) -> AnnResult<()> {
                self.windowed_batch_calls.fetch_add(1, Ordering::Relaxed);
                for (row, &pid) in ids.iter().enumerate() {
                    let src = pid as usize * self.dim;
                    let dst = row * self.dim;
                    out[dst..dst + self.dim].copy_from_slice(&self.rows[src..src + self.dim]);
                }
                Ok(())
            }
        }

        let dim = 16usize;
        let store = CountingPointStore {
            rows: (0..64)
                .flat_map(|row| (0..dim).map(move |col| ((row * 13 + col * 7) % 101) as f32))
                .collect(),
            dim,
            plain_calls: AtomicUsize::new(0),
            windowed_batch_calls: AtomicUsize::new(0),
        };
        let leaders = (0..ADS_LEADER_BLOCK)
            .flat_map(|leader| (0..dim).map(move |col| ((leader * 11 + col * 3) % 97) as f32))
            .collect::<Vec<_>>();
        let layout =
            AdSamplingLeaderLayout::from_row_major_for_test(leaders, ADS_LEADER_BLOCK, dim);
        let config = AdSamplingConfig {
            epsilon: 100.0,
            group_dims: 8,
            seed_exact: 4,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 0,
        };
        let seed_indices = adsampling_seed_indices(ADS_LEADER_BLOCK, config.seed_exact, 2);
        let seeded = adsampling_seed_mask(ADS_LEADER_BLOCK, &seed_indices);

        let result = assign_point_chunk_adsampling_layout(
            &store,
            &[3, 17, 5, 23],
            0,
            dim,
            &layout,
            2,
            config,
            &seed_indices,
            &seeded,
        )
        .unwrap();

        assert_eq!(result.chunk.leaders_by_point.len(), 4);
        assert_eq!(store.plain_calls.load(Ordering::Relaxed), 0);
        assert!(
            store.windowed_batch_calls.load(Ordering::Relaxed) > 0,
            "ADSampling query chunk loading should use the windowed batch read path"
        );
    }

    #[test]
    fn indexed_loaded_adsampling_matches_contiguous_loaded_adsampling() {
        let dim = 8;
        let source_data = (0..6)
            .flat_map(|row| (0..dim).map(move |col| ((row * 17 + col * 5) % 101) as f32))
            .collect::<Vec<_>>();
        let point_ids = vec![11_u32, 12, 13, 14];
        let source_offsets = vec![4_usize, 1, 5, 2];
        let mut contiguous = Vec::new();
        for &offset in &source_offsets {
            contiguous.extend_from_slice(&source_data[offset * dim..(offset + 1) * dim]);
        }
        let leaders = (0..ADS_LEADER_BLOCK)
            .flat_map(|leader| (0..dim).map(move |col| ((leader * 7 + col * 13) % 89) as f32))
            .collect::<Vec<_>>();
        let layout =
            AdSamplingLeaderLayout::from_row_major_for_test(leaders, ADS_LEADER_BLOCK, dim);
        let config = AdSamplingConfig {
            epsilon: 100.0,
            group_dims: 4,
            seed_exact: 4,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 0,
        };
        let seed_indices = adsampling_seed_indices(ADS_LEADER_BLOCK, config.seed_exact, 2);
        let seeded = adsampling_seed_mask(ADS_LEADER_BLOCK, &seed_indices);

        let contiguous = assign_loaded_point_chunk_adsampling_layout(
            &point_ids,
            &contiguous,
            0,
            dim,
            &layout,
            2,
            config,
            &seed_indices,
            &seeded,
        )
        .unwrap();
        let indexed = assign_loaded_point_indexed_adsampling_layout(
            &point_ids,
            &source_data,
            &source_offsets,
            0,
            dim,
            &layout,
            2,
            config,
            &seed_indices,
            &seeded,
        )
        .unwrap();

        assert_eq!(
            indexed.chunk.leaders_by_point,
            contiguous.chunk.leaders_by_point
        );
        assert_eq!(indexed.full_evals, contiguous.full_evals);
        assert_eq!(indexed.group_evals, contiguous.group_evals);
    }

    #[test]
    fn streaming_adsampling_profile_records_scheduler_counters() {
        struct CountingPointStore {
            rows: Vec<f32>,
            dim: usize,
        }

        impl PointStore for CountingPointStore {
            fn len(&self) -> usize {
                self.rows.len() / self.dim
            }

            fn dim(&self) -> usize {
                self.dim
            }

            fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
                let start = pid as usize * self.dim;
                out.copy_from_slice(&self.rows[start..start + self.dim]);
                Ok(())
            }

            fn read_range_into(
                &self,
                start_pid: u32,
                count: usize,
                out: &mut [f32],
            ) -> AnnResult<()> {
                let start = start_pid as usize * self.dim;
                let end = start + count * self.dim;
                out.copy_from_slice(&self.rows[start..end]);
                Ok(())
            }

            fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
                for (row, &pid) in ids.iter().enumerate() {
                    let src = pid as usize * self.dim;
                    let dst = row * self.dim;
                    out[dst..dst + self.dim].copy_from_slice(&self.rows[src..src + self.dim]);
                }
                Ok(())
            }

            fn read_points_windowed_into_batch_stats(
                &self,
                ids: &[u32],
                out: &mut [f32],
                _options: &WindowedGatherOptions,
                _stats: &mut PointBatchStats,
            ) -> AnnResult<()> {
                self.read_points_into(ids, out)
            }
        }

        let dim = 8usize;
        let rows = ADS_QUERY_CHUNK + 48;
        let store = CountingPointStore {
            rows: (0..rows)
                .flat_map(|row| (0..dim).map(move |col| ((row * 17 + col * 5) % 131) as f32))
                .collect(),
            dim,
        };
        let cur = (0..rows as u32).collect::<Vec<_>>();
        let leaders = (0..32u32).collect::<Vec<_>>();
        let mut params = ForgeANNParams::default();
        params.adsampling_group_dims = 8;

        let mut visited_chunks = 0usize;
        let profile = assign_point_leaders_adsampling_streaming(
            &store,
            &cur,
            &leaders,
            2,
            &params,
            1,
            |chunk| {
                visited_chunks += 1;
                assert!(!chunk.leaders_by_point.is_empty());
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(visited_chunks, profile.chunks);
        assert_eq!(
            profile.chunks,
            cur.len()
                .div_ceil(choose_adsampling_assignment_chunk_rows(cur.len()))
        );
        assert!(profile.chunks > 2);
        assert!(!profile.called_inside_rayon_worker);
        assert!(profile.chunk_wall_max_ms > 0.0);
        assert!(profile.chunk_wall_accumulated_ms >= profile.chunk_wall_max_ms);
        assert!(profile.chunk_active_max >= 1);
        assert!(profile.chunk_active_start_avg >= 1.0);
        assert!(profile.visit_chunk_ms > 0.0);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let nested_profile = pool
            .install(|| {
                assign_point_leaders_adsampling_streaming(
                    &store,
                    &cur[..64],
                    &leaders,
                    2,
                    &params,
                    1,
                    |_| Ok(()),
                )
            })
            .unwrap();

        assert!(nested_profile.called_inside_rayon_worker);
    }

    #[test]
    fn adsampling_scheduler_classifier_marks_large_and_huge_tasks() {
        let params = ForgeANNParams::default();

        let large = classify_adsampling_task(&params, 350_000, 1_500);
        assert!(large.large_ads);
        assert!(!large.huge_ads);

        let huge = classify_adsampling_task(&params, 1_000_000, 64);
        assert!(!huge.large_ads);
        assert!(huge.huge_ads);

        let medium = classify_adsampling_task(&params, 65_000, 525);
        assert!(!medium.large_ads);
        assert!(!medium.huge_ads);
    }

    #[test]
    fn adsampling_assignment_chunks_match_gemm_style_tiling() {
        assert_eq!(
            choose_adsampling_assignment_chunk_rows_for_threads(2_048, 42),
            256
        );
        assert_eq!(
            choose_adsampling_assignment_chunk_rows_for_threads(65_000, 42),
            1_548
        );
        assert_eq!(
            choose_adsampling_assignment_chunk_rows_for_threads(3_135_080, 42),
            ADS_ASSIGN_MAX_CHUNK_ROWS
        );
    }

    #[test]
    fn wavefront_pairmask_preserves_candidate_block_order() {
        for rows in [1_usize, 17, 64, 65, 129, 257] {
            let baseline = baseline_candidate_block_sequences_for_test(rows);
            let wavefront = wavefront_candidate_block_sequences_for_test(rows);

            assert_eq!(
                wavefront, baseline,
                "wavefront candidate-block order diverged for rows={rows}"
            );
        }
    }

    #[test]
    fn wavefront_pairmask_mutual_seed_matches_pairmask_semantics() {
        let dim = 8usize;
        let rows = 19usize;
        let seed_exact = 5usize;
        let vectors = deterministic_leaf_vectors(rows, dim);
        let layout = AdSamplingLeafLayout::build(&vectors, rows, dim);
        let (top, seeded, seed_evals) =
            initialize_wavefront_mutual_seed_for_test(&layout, 2, seed_exact);
        let mut expected_unique_pairs = 0_u64;

        for row in 0..rows {
            for candidate in 0..rows {
                let expected = (1..=seed_exact).any(|offset| {
                    (row + offset) % rows == candidate || (candidate + offset) % rows == row
                });
                assert_eq!(
                    wavefront_seeded_pair(&seeded, rows, row, candidate),
                    expected,
                    "seed coverage mismatch row={row} candidate={candidate}"
                );
                if row < candidate && expected {
                    expected_unique_pairs += 1;
                }
            }
        }

        assert_eq!(seed_evals, expected_unique_pairs);
        for row in 0..rows {
            for &(candidate, _) in &top[row].entries() {
                assert!(wavefront_seeded_pair(&seeded, rows, row, candidate));
            }
        }
    }

    #[test]
    fn wavefront_pairmask_diagonal_fallback_matches_baseline_self_block_scan() {
        let dim = 32usize;
        let leaf_size = 57usize;
        let vectors = deterministic_leaf_vectors(leaf_size, dim);
        let config = AdSamplingConfig {
            epsilon: 100.0,
            group_dims: 8,
            seed_exact: 0,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 0,
        };

        let baseline =
            compute_leaf_adsampling_topk_l2(&vectors, leaf_size, dim, 4, config).unwrap();
        let wavefront = compute_leaf_adsampling_topk_l2_wavefront_pairmask_for_test(
            &vectors, leaf_size, dim, 4, config,
        )
        .unwrap();

        assert_leaf_topk_close(&wavefront.row_topk, &baseline.row_topk);
        assert_eq!(wavefront.candidate_block_order_violations, 0);
        assert_eq!(wavefront.self_edge_violations, 0);
        assert_eq!(wavefront.duplicate_topk_violations, 0);
    }

    #[test]
    fn wavefront_pairmask_block_pair_commit_has_no_self_or_duplicates() {
        let dim = 64usize;
        let leaf_size = 153usize;
        let vectors = deterministic_leaf_vectors(leaf_size, dim);
        let config = AdSamplingConfig {
            epsilon: 2.0,
            group_dims: 16,
            seed_exact: 8,
            sparse_full64_threshold: 8,
            sparse_group16_threshold: 4,
            validate_sources: 0,
        };

        let run = compute_leaf_adsampling_topk_l2_wavefront_pairmask_for_test(
            &vectors, leaf_size, dim, 3, config,
        )
        .unwrap();

        assert_eq!(run.candidate_block_order_violations, 0);
        assert_eq!(run.self_edge_violations, 0);
        assert_eq!(run.duplicate_topk_violations, 0);
        assert!(run.group_evals > 0);
        assert!(run.full_evals > 0);
    }

    #[test]
    #[ignore = "requires real Wiki leaf dump; set FORGEANN_ADS_DATA_PATH, FORGEANN_ADS_LEAF_MANIFEST, FORGEANN_LEAF_ADS_WAVEFRONT_OUT"]
    fn leaf_ads_wavefront_pairmask_real_dump() {
        let data_path = PathBuf::from(
            std::env::var("FORGEANN_ADS_DATA_PATH")
                .expect("FORGEANN_ADS_DATA_PATH must point at wiki_base.fbin"),
        );
        let manifest_path = PathBuf::from(
            std::env::var("FORGEANN_ADS_LEAF_MANIFEST")
                .expect("FORGEANN_ADS_LEAF_MANIFEST must point at partition_d01_leaf_manifest.bin"),
        );
        let out_path = PathBuf::from(
            std::env::var("FORGEANN_LEAF_ADS_WAVEFRONT_OUT")
                .expect("FORGEANN_LEAF_ADS_WAVEFRONT_OUT must point at report.json"),
        );
        let samples_per_bucket = std::env::var("FORGEANN_ADS_SAMPLES_PER_BUCKET")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        let reps = std::env::var("FORGEANN_ADS_BENCH_REPS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);

        let records = read_real_leaf_records(&manifest_path, samples_per_bucket)
            .expect("real leaf manifest should be readable");
        assert!(
            !records.is_empty(),
            "real leaf manifest did not contain any >=512 row samples"
        );
        let dataset = RealFbinDataset::open(&data_path).expect("wiki fbin should open");
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread wavefront pair-mask profiler pool should build");
        let config = AdSamplingConfig {
            epsilon: 1.0,
            group_dims: 64,
            seed_exact: ForgeANNParams::LEAF_ADSAMPLING_SEED_EXACT_M,
            sparse_full64_threshold: ForgeANNParams::ADS_SPARSE_FULL64_THRESHOLD,
            sparse_group16_threshold: ForgeANNParams::ADS_SPARSE_GROUP16_THRESHOLD,
            validate_sources: 0,
        };

        let mut grouped: BTreeMap<String, Vec<WavefrontPairMaskGroupedSample>> = BTreeMap::new();
        let mut samples = Vec::new();
        for record in records {
            let point_ids = read_real_leaf_points(&manifest_path, &record)
                .expect("real leaf points should be readable");
            assert_eq!(point_ids.len(), record.len);
            let vectors = dataset
                .read_points(&point_ids)
                .expect("real leaf vectors should be readable");

            let baseline = serial_pool
                .install(|| {
                    compute_leaf_adsampling_topk_l2(&vectors, record.len, dataset.dim, 2, config)
                })
                .expect("baseline leaf ADS should run");
            let candidate = serial_pool
                .install(|| {
                    compute_leaf_adsampling_topk_l2_wavefront_pairmask_for_test(
                        &vectors,
                        record.len,
                        dataset.dim,
                        2,
                        config,
                    )
                })
                .expect("wavefront pair-mask ADS should run");
            let exact = exact_leaf_topk(&vectors, record.len, dataset.dim, 2);
            let baseline_exact_recall = leaf_topk_overlap_ratio(&baseline.row_topk, &exact);
            let candidate_exact_recall = leaf_topk_overlap_ratio(&candidate.row_topk, &exact);
            let candidate_exact_recall_delta = candidate_exact_recall - baseline_exact_recall;
            let output_overlap = leaf_topk_overlap_ratio(&candidate.row_topk, &baseline.row_topk);

            let mut baseline_timings = Vec::with_capacity(reps);
            let mut candidate_runs = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                let result = serial_pool
                    .install(|| {
                        compute_leaf_adsampling_topk_l2(
                            &vectors,
                            record.len,
                            dataset.dim,
                            2,
                            config,
                        )
                    })
                    .expect("baseline leaf ADS should run");
                assert_eq!(result.row_topk, baseline.row_topk);
                baseline_timings.push(duration_nanos_u64(start.elapsed()));

                let result = serial_pool
                    .install(|| {
                        compute_leaf_adsampling_topk_l2_wavefront_pairmask_for_test(
                            &vectors,
                            record.len,
                            dataset.dim,
                            2,
                            config,
                        )
                    })
                    .expect("wavefront pair-mask ADS should run");
                assert_eq!(result.row_topk, candidate.row_topk);
                candidate_runs.push(result);
            }

            let baseline_ns = median_u64(&mut baseline_timings);
            let candidate_run = take_median_wavefront_pairmask_run(candidate_runs);
            let wall_ratio = candidate_run.elapsed_ns as f64 / baseline_ns.max(1) as f64;
            let group_ratio =
                candidate_run.group_evals as f64 / baseline.telemetry.group_evals.max(1) as f64;
            let full_ratio =
                candidate_run.full_evals as f64 / baseline.telemetry.full_evals.max(1) as f64;
            let pair_survivor_ratio = candidate_run.pair_survivor_ratio();
            grouped.entry(record.bucket.clone()).or_default().push(
                WavefrontPairMaskGroupedSample {
                    wall_ratio,
                    group_ratio,
                    full_ratio,
                    pair_survivor_ratio,
                    output_overlap,
                    baseline_exact_recall,
                    candidate_exact_recall,
                    candidate_exact_recall_delta,
                    candidate_block_order_violations: candidate_run
                        .candidate_block_order_violations,
                    self_edge_violations: candidate_run.self_edge_violations,
                    duplicate_topk_violations: candidate_run.duplicate_topk_violations,
                },
            );
            samples.push(WavefrontPairMaskSample {
                child_index: record.child_index,
                bucket: record.bucket,
                leaf_size: record.len,
                dim: dataset.dim,
                k: 2,
                reps,
                baseline_ns,
                candidate_ns: candidate_run.elapsed_ns,
                wall_ratio,
                baseline_group_evals: baseline.telemetry.group_evals,
                candidate_group_evals: candidate_run.group_evals,
                group_ratio,
                baseline_full_evals: baseline.telemetry.full_evals,
                candidate_full_evals: candidate_run.full_evals,
                full_ratio,
                seed_evals: candidate_run.seed_evals,
                pair_pruned_both: candidate_run.pair_pruned_both,
                pair_full: candidate_run.pair_full,
                pair_survivor_ratio,
                side_pruned: candidate_run.side_pruned,
                side_full_updates: candidate_run.side_full_updates,
                commit_ms: nanos_to_ms(candidate_run.commit_ns),
                prune_ms: nanos_to_ms(candidate_run.prune_ns),
                scratch_clear_ms: nanos_to_ms(candidate_run.scratch_clear_ns),
                output_overlap,
                baseline_exact_recall,
                candidate_exact_recall,
                candidate_exact_recall_delta,
                candidate_block_order_violations: candidate_run.candidate_block_order_violations,
                self_edge_violations: candidate_run.self_edge_violations,
                duplicate_topk_violations: candidate_run.duplicate_topk_violations,
                baseline_output_hash: hash_leaf_topk(&baseline.row_topk),
                candidate_output_hash: hash_leaf_topk(&candidate_run.row_topk),
            });
        }

        let mut summaries = Vec::new();
        for (bucket, values) in grouped {
            let mut wall = values
                .iter()
                .map(|value| value.wall_ratio)
                .collect::<Vec<_>>();
            let mut group = values
                .iter()
                .map(|value| value.group_ratio)
                .collect::<Vec<_>>();
            let mut full = values
                .iter()
                .map(|value| value.full_ratio)
                .collect::<Vec<_>>();
            let mut survivor = values
                .iter()
                .map(|value| value.pair_survivor_ratio)
                .collect::<Vec<_>>();
            let mut overlap = values
                .iter()
                .map(|value| value.output_overlap)
                .collect::<Vec<_>>();
            let mut baseline_recall = values
                .iter()
                .map(|value| value.baseline_exact_recall)
                .collect::<Vec<_>>();
            let mut candidate_recall = values
                .iter()
                .map(|value| value.candidate_exact_recall)
                .collect::<Vec<_>>();
            let mut delta = values
                .iter()
                .map(|value| value.candidate_exact_recall_delta)
                .collect::<Vec<_>>();
            let max_order_violations = values
                .iter()
                .map(|value| value.candidate_block_order_violations)
                .max()
                .unwrap_or(0);
            let max_self_edge_violations = values
                .iter()
                .map(|value| value.self_edge_violations)
                .max()
                .unwrap_or(0);
            let max_duplicate_topk_violations = values
                .iter()
                .map(|value| value.duplicate_topk_violations)
                .max()
                .unwrap_or(0);
            let median_wall_ratio = median_f64(&mut wall);
            let median_group_ratio = median_f64(&mut group);
            let median_output_overlap = median_f64(&mut overlap);
            let median_candidate_exact_recall_delta = median_f64(&mut delta);
            let acceptance = wavefront_pairmask_acceptance(
                &bucket,
                median_wall_ratio,
                median_group_ratio,
                median_output_overlap,
                median_candidate_exact_recall_delta,
                max_order_violations,
                max_self_edge_violations,
                max_duplicate_topk_violations,
            );
            summaries.push(WavefrontPairMaskSummary {
                bucket,
                samples: values.len(),
                median_wall_ratio,
                median_group_ratio,
                median_full_ratio: median_f64(&mut full),
                median_pair_survivor_ratio: median_f64(&mut survivor),
                median_output_overlap,
                median_baseline_exact_recall: median_f64(&mut baseline_recall),
                median_candidate_exact_recall: median_f64(&mut candidate_recall),
                median_candidate_exact_recall_delta,
                max_candidate_block_order_violations: max_order_violations,
                max_self_edge_violations,
                max_duplicate_topk_violations,
                acceptance,
            });
        }

        let report = WavefrontPairMaskReport {
            schema: "leaf_ads_wavefront_pairmask_real_dump_v1",
            data_path: data_path.display().to_string(),
            manifest_path: manifest_path.display().to_string(),
            samples_per_bucket,
            reps,
            avx512: adsampling_avx512_available(),
            epsilon: config.epsilon,
            group_dims: config.group_dims,
            seed_exact: config.seed_exact,
            summaries,
            samples,
        };
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .expect("wavefront pair-mask output parent should be creatable");
        }
        let mut out = File::create(&out_path).expect("wavefront pair-mask output should create");
        let text = serde_json::to_string_pretty(&report)
            .expect("wavefront pair-mask report should serialize");
        out.write_all(text.as_bytes())
            .expect("wavefront pair-mask report should write");
        eprintln!("{text}");
    }

    #[test]
    #[ignore = "requires real Wiki leaf dump; set FORGEANN_ADS_DATA_PATH, FORGEANN_ADS_LEAF_MANIFEST, FORGEANN_ADS_BENCH_OUT"]
    fn ads_kernel_real_dump_bench() {
        let data_path = PathBuf::from(
            std::env::var("FORGEANN_ADS_DATA_PATH")
                .expect("FORGEANN_ADS_DATA_PATH must point at wiki_base.fbin"),
        );
        let manifest_path = PathBuf::from(
            std::env::var("FORGEANN_ADS_LEAF_MANIFEST")
                .expect("FORGEANN_ADS_LEAF_MANIFEST must point at partition_d01_leaf_manifest.bin"),
        );
        let out_path = PathBuf::from(
            std::env::var("FORGEANN_ADS_BENCH_OUT")
                .expect("FORGEANN_ADS_BENCH_OUT must point at bench.json"),
        );
        let samples_per_bucket = std::env::var("FORGEANN_ADS_SAMPLES_PER_BUCKET")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8)
            .max(1);
        let reps = std::env::var("FORGEANN_ADS_BENCH_REPS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);
        let allow_divergence = parse_bool_env("FORGEANN_ADS_BENCH_ALLOW_DIVERGENCE");

        let records = read_real_leaf_records(&manifest_path, samples_per_bucket)
            .expect("real leaf manifest should be readable");
        assert!(
            !records.is_empty(),
            "real leaf manifest did not contain any >=512 row samples"
        );
        let dataset = RealFbinDataset::open(&data_path).expect("wiki fbin should open");
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread ADS bench pool should build");
        let baseline_config = AdSamplingConfig {
            epsilon: 1.0,
            group_dims: 64,
            seed_exact: ForgeANNParams::LEAF_ADSAMPLING_SEED_EXACT_M,
            sparse_full64_threshold: ForgeANNParams::ADS_SPARSE_FULL64_THRESHOLD,
            sparse_group16_threshold: ForgeANNParams::ADS_SPARSE_GROUP16_THRESHOLD,
            validate_sources: 0,
        };
        let variant_config = AdSamplingConfig {
            epsilon: parse_f32_env("FORGEANN_ADS_BENCH_VARIANT_EPSILON")
                .unwrap_or(baseline_config.epsilon),
            group_dims: parse_usize_env("FORGEANN_ADS_BENCH_VARIANT_GROUP_DIMS")
                .unwrap_or(baseline_config.group_dims)
                .max(1),
            seed_exact: baseline_config.seed_exact,
            sparse_full64_threshold: parse_u32_env("FORGEANN_ADS_BENCH_VARIANT_SPARSE_FULL64")
                .unwrap_or(baseline_config.sparse_full64_threshold),
            sparse_group16_threshold: parse_u32_env("FORGEANN_ADS_BENCH_VARIANT_SPARSE_GROUP16")
                .unwrap_or(baseline_config.sparse_group16_threshold),
            validate_sources: 0,
        };

        let mut sample_results = Vec::new();
        let mut grouped: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut grouped_overlap: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for record in records {
            let point_ids = read_real_leaf_points(&manifest_path, &record)
                .expect("real leaf points should be readable");
            assert_eq!(point_ids.len(), record.len);
            let vectors = dataset
                .read_points(&point_ids)
                .expect("real leaf vectors should be readable");

            let legacy = serial_pool
                .install(|| {
                    compute_leaf_adsampling_topk_l2_legacy_for_test(
                        &vectors,
                        record.len,
                        dataset.dim,
                        2,
                        baseline_config,
                    )
                })
                .expect("legacy ADS should run");
            let current = serial_pool
                .install(|| {
                    compute_leaf_adsampling_topk_l2(
                        &vectors,
                        record.len,
                        dataset.dim,
                        2,
                        variant_config,
                    )
                })
                .expect("current ADS should run");
            let output_overlap = leaf_topk_overlap_ratio(&current.row_topk, &legacy.row_topk);
            if !allow_divergence {
                assert_eq!(
                    current.row_topk, legacy.row_topk,
                    "current ADS output diverged from legacy oracle for record {} bucket {}",
                    record.child_index, record.bucket
                );
            }

            let mut legacy_timings = Vec::with_capacity(reps);
            let mut current_timings = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                let result = serial_pool
                    .install(|| {
                        compute_leaf_adsampling_topk_l2_legacy_for_test(
                            &vectors,
                            record.len,
                            dataset.dim,
                            2,
                            baseline_config,
                        )
                    })
                    .expect("legacy ADS should run");
                assert_eq!(result.row_topk, legacy.row_topk);
                legacy_timings.push(duration_nanos_u64(start.elapsed()));

                let start = Instant::now();
                let result = serial_pool
                    .install(|| {
                        compute_leaf_adsampling_topk_l2(
                            &vectors,
                            record.len,
                            dataset.dim,
                            2,
                            variant_config,
                        )
                    })
                    .expect("current ADS should run");
                if !allow_divergence {
                    assert_eq!(result.row_topk, legacy.row_topk);
                }
                current_timings.push(duration_nanos_u64(start.elapsed()));
            }

            let current_median = median_u64(&mut current_timings);
            let legacy_median = median_u64(&mut legacy_timings);
            let current_ns_per_row = current_median as f64 / record.len.max(1) as f64;
            let legacy_ns_per_row = legacy_median as f64 / record.len.max(1) as f64;
            grouped
                .entry(record.bucket.clone())
                .or_default()
                .push(current_ns_per_row / legacy_ns_per_row.max(1.0));
            grouped_overlap
                .entry(record.bucket.clone())
                .or_default()
                .push(output_overlap);
            sample_results.push(AdsKernelBenchSample {
                child_index: record.child_index,
                bucket: record.bucket,
                leaf_size: record.len,
                dim: dataset.dim,
                k: 2,
                legacy_median_ns: legacy_median,
                current_median_ns: current_median,
                legacy_ns_per_row,
                current_ns_per_row,
                current_over_legacy: current_ns_per_row / legacy_ns_per_row.max(1.0),
                legacy_layout_ms: duration_ms(legacy.telemetry.layout),
                legacy_seed_ms: duration_ms(legacy.telemetry.seed),
                legacy_scan_ms: duration_ms(legacy.telemetry.scan),
                legacy_group_evals: legacy.telemetry.group_evals,
                legacy_simd_group_calls: legacy.telemetry.simd_group_calls,
                legacy_simd_active_lane_evals: legacy.telemetry.simd_active_lane_evals,
                legacy_scalar_group_evals: legacy.telemetry.scalar_group_evals,
                legacy_full_evals: legacy.telemetry.full_evals,
                legacy_pruned_evals: legacy.telemetry.pruned_evals,
                layout_ms: duration_ms(current.telemetry.layout),
                seed_ms: duration_ms(current.telemetry.seed),
                scan_ms: duration_ms(current.telemetry.scan),
                group_evals: current.telemetry.group_evals,
                simd_group_calls: current.telemetry.simd_group_calls,
                simd_active_lane_evals: current.telemetry.simd_active_lane_evals,
                scalar_group_evals: current.telemetry.scalar_group_evals,
                full_evals: current.telemetry.full_evals,
                pruned_evals: current.telemetry.pruned_evals,
                output_overlap,
                legacy_output_hash: hash_leaf_topk(&legacy.row_topk),
                output_hash: hash_leaf_topk(&current.row_topk),
            });
        }

        let mut summaries = Vec::new();
        for (bucket, mut ratios) in grouped {
            let mut overlaps = grouped_overlap.remove(&bucket).unwrap_or_default();
            summaries.push(AdsKernelBenchBucketSummary {
                bucket,
                samples: ratios.len(),
                median_current_over_legacy: median_f64(&mut ratios),
                median_output_overlap: median_f64(&mut overlaps),
            });
        }
        let report = AdsKernelBenchReport {
            schema: "ads_kernel_real_dump_bench_v2",
            data_path: data_path.display().to_string(),
            manifest_path: manifest_path.display().to_string(),
            samples_per_bucket,
            reps,
            avx512: adsampling_avx512_available(),
            allow_divergence,
            baseline_epsilon: baseline_config.epsilon,
            baseline_group_dims: baseline_config.group_dims,
            baseline_sparse_full64_threshold: baseline_config.sparse_full64_threshold,
            baseline_sparse_group16_threshold: baseline_config.sparse_group16_threshold,
            variant_epsilon: variant_config.epsilon,
            variant_group_dims: variant_config.group_dims,
            variant_sparse_full64_threshold: variant_config.sparse_full64_threshold,
            variant_sparse_group16_threshold: variant_config.sparse_group16_threshold,
            summaries,
            samples: sample_results,
        };
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).expect("bench output parent should be creatable");
        }
        let mut out = File::create(&out_path).expect("bench output should be creatable");
        let text = serde_json::to_string_pretty(&report).expect("bench report should serialize");
        out.write_all(text.as_bytes())
            .expect("bench report should write");
        eprintln!("{text}");
    }

    #[derive(Clone, Debug)]
    struct RealLeafRecord {
        child_index: usize,
        len: usize,
        bucket: String,
        extents: Vec<RealRunExtent>,
    }

    #[derive(Clone, Copy, Debug)]
    struct RealRunExtent {
        byte_offset: u64,
        len: usize,
    }

    struct RealFbinDataset {
        file: File,
        rows: usize,
        dim: usize,
        row_bytes: usize,
    }

    impl RealFbinDataset {
        fn open(path: &Path) -> AnnResult<Self> {
            let mut file = File::open(path)?;
            let mut header = [0_u8; 8];
            file.read_exact(&mut header)?;
            let rows = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let dim = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            Ok(Self {
                file,
                rows,
                dim,
                row_bytes: dim * size_of::<f32>(),
            })
        }

        fn read_points(&self, ids: &[u32]) -> AnnResult<Vec<f32>> {
            let mut vectors = vec![0.0_f32; ids.len() * self.dim];
            let mut row_bytes = vec![0_u8; self.row_bytes];
            for (row, &pid) in ids.iter().enumerate() {
                let pid = pid as usize;
                if pid >= self.rows {
                    return Err(AnnError::log_index_error(format!(
                        "point id {pid} exceeds fbin rows {}",
                        self.rows
                    )));
                }
                let offset = 8_u64 + (pid * self.row_bytes) as u64;
                self.file.read_exact_at(&mut row_bytes, offset)?;
                let dst = row * self.dim;
                for (col, chunk) in row_bytes.chunks_exact(size_of::<f32>()).enumerate() {
                    vectors[dst + col] = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            Ok(vectors)
        }
    }

    #[derive(Serialize)]
    struct AdsKernelBenchReport {
        schema: &'static str,
        data_path: String,
        manifest_path: String,
        samples_per_bucket: usize,
        reps: usize,
        avx512: bool,
        allow_divergence: bool,
        baseline_epsilon: f32,
        baseline_group_dims: usize,
        baseline_sparse_full64_threshold: u32,
        baseline_sparse_group16_threshold: u32,
        variant_epsilon: f32,
        variant_group_dims: usize,
        variant_sparse_full64_threshold: u32,
        variant_sparse_group16_threshold: u32,
        summaries: Vec<AdsKernelBenchBucketSummary>,
        samples: Vec<AdsKernelBenchSample>,
    }

    #[derive(Serialize)]
    struct AdsKernelBenchBucketSummary {
        bucket: String,
        samples: usize,
        median_current_over_legacy: f64,
        median_output_overlap: f64,
    }

    #[derive(Serialize)]
    struct AdsKernelBenchSample {
        child_index: usize,
        bucket: String,
        leaf_size: usize,
        dim: usize,
        k: usize,
        legacy_median_ns: u64,
        current_median_ns: u64,
        legacy_ns_per_row: f64,
        current_ns_per_row: f64,
        current_over_legacy: f64,
        legacy_layout_ms: f64,
        legacy_seed_ms: f64,
        legacy_scan_ms: f64,
        legacy_group_evals: u64,
        legacy_simd_group_calls: u64,
        legacy_simd_active_lane_evals: u64,
        legacy_scalar_group_evals: u64,
        legacy_full_evals: u64,
        legacy_pruned_evals: u64,
        layout_ms: f64,
        seed_ms: f64,
        scan_ms: f64,
        group_evals: u64,
        simd_group_calls: u64,
        simd_active_lane_evals: u64,
        scalar_group_evals: u64,
        full_evals: u64,
        pruned_evals: u64,
        output_overlap: f64,
        legacy_output_hash: String,
        output_hash: String,
    }

    #[derive(Clone, Debug)]
    struct WavefrontPairMaskRun {
        row_topk: Vec<Vec<(usize, f32)>>,
        elapsed_ns: u64,
        seed_evals: u64,
        group_evals: u64,
        full_evals: u64,
        pair_pruned_both: u64,
        pair_full: u64,
        side_pruned: u64,
        side_full_updates: u64,
        commit_ns: u64,
        prune_ns: u64,
        scratch_clear_ns: u64,
        candidate_block_order_violations: u64,
        self_edge_violations: u64,
        duplicate_topk_violations: u64,
    }

    impl WavefrontPairMaskRun {
        fn pair_survivor_ratio(&self) -> f64 {
            finite_ratio_u64(
                self.pair_full,
                self.pair_full.saturating_add(self.pair_pruned_both),
            )
        }
    }

    #[derive(Default)]
    struct WavefrontPairMaskCounters {
        group_evals: u64,
        full_evals: u64,
        pair_pruned_both: u64,
        pair_full: u64,
        side_pruned: u64,
        side_full_updates: u64,
        commit_ns: u64,
        prune_ns: u64,
        scratch_clear_ns: u64,
        candidate_block_order_violations: u64,
    }

    struct WavefrontPairScratch {
        distances: [[f32; ADS_LEADER_BLOCK]; ADS_LEADER_BLOCK],
        survivor_masks: [u64; ADS_LEADER_BLOCK],
        left_thresholds: [f32; ADS_LEADER_BLOCK],
        right_thresholds: [f32; ADS_LEADER_BLOCK],
    }

    impl WavefrontPairScratch {
        fn new() -> Self {
            Self {
                distances: [[0.0; ADS_LEADER_BLOCK]; ADS_LEADER_BLOCK],
                survivor_masks: [0; ADS_LEADER_BLOCK],
                left_thresholds: [f32::INFINITY; ADS_LEADER_BLOCK],
                right_thresholds: [f32::INFINITY; ADS_LEADER_BLOCK],
            }
        }
    }

    #[derive(Serialize)]
    struct WavefrontPairMaskReport {
        schema: &'static str,
        data_path: String,
        manifest_path: String,
        samples_per_bucket: usize,
        reps: usize,
        avx512: bool,
        epsilon: f32,
        group_dims: usize,
        seed_exact: usize,
        summaries: Vec<WavefrontPairMaskSummary>,
        samples: Vec<WavefrontPairMaskSample>,
    }

    #[derive(Serialize)]
    struct WavefrontPairMaskSummary {
        bucket: String,
        samples: usize,
        median_wall_ratio: f64,
        median_group_ratio: f64,
        median_full_ratio: f64,
        median_pair_survivor_ratio: f64,
        median_output_overlap: f64,
        median_baseline_exact_recall: f64,
        median_candidate_exact_recall: f64,
        median_candidate_exact_recall_delta: f64,
        max_candidate_block_order_violations: u64,
        max_self_edge_violations: u64,
        max_duplicate_topk_violations: u64,
        acceptance: &'static str,
    }

    #[derive(Serialize)]
    struct WavefrontPairMaskSample {
        child_index: usize,
        bucket: String,
        leaf_size: usize,
        dim: usize,
        k: usize,
        reps: usize,
        baseline_ns: u64,
        candidate_ns: u64,
        wall_ratio: f64,
        baseline_group_evals: u64,
        candidate_group_evals: u64,
        group_ratio: f64,
        baseline_full_evals: u64,
        candidate_full_evals: u64,
        full_ratio: f64,
        seed_evals: u64,
        pair_pruned_both: u64,
        pair_full: u64,
        pair_survivor_ratio: f64,
        side_pruned: u64,
        side_full_updates: u64,
        commit_ms: f64,
        prune_ms: f64,
        scratch_clear_ms: f64,
        output_overlap: f64,
        baseline_exact_recall: f64,
        candidate_exact_recall: f64,
        candidate_exact_recall_delta: f64,
        candidate_block_order_violations: u64,
        self_edge_violations: u64,
        duplicate_topk_violations: u64,
        baseline_output_hash: String,
        candidate_output_hash: String,
    }

    struct WavefrontPairMaskGroupedSample {
        wall_ratio: f64,
        group_ratio: f64,
        full_ratio: f64,
        pair_survivor_ratio: f64,
        output_overlap: f64,
        baseline_exact_recall: f64,
        candidate_exact_recall: f64,
        candidate_exact_recall_delta: f64,
        candidate_block_order_violations: u64,
        self_edge_violations: u64,
        duplicate_topk_violations: u64,
    }

    fn compute_leaf_adsampling_topk_l2_wavefront_pairmask_for_test(
        vectors: &[f32],
        leaf_size: usize,
        dim: usize,
        k: usize,
        config: AdSamplingConfig,
    ) -> AnnResult<WavefrontPairMaskRun> {
        if vectors.len() != leaf_size.saturating_mul(dim) {
            return Err(AnnError::log_index_error(format!(
                "Invalid wavefront pair-mask leaf matrix: len={} leaf_size={leaf_size} dim={dim}",
                vectors.len()
            )));
        }
        let total_start = Instant::now();
        let (context, _) = build_leaf_ads_context(vectors, leaf_size, dim, config)?;
        let rows = context.layout.rows;
        let k = k.min(rows.saturating_sub(1)).max(1);
        let seed_exact = config.seed_exact.min(rows.saturating_sub(1));
        let (mut top, seeded, seed_evals) =
            initialize_wavefront_mutual_seed_for_test(&context.layout, k, seed_exact);
        let mut scratch = WavefrontPairScratch::new();
        let mut counters = WavefrontPairMaskCounters::default();
        let block_count = context.layout.blocks.len();
        let mut next_candidate_block_by_row = vec![0_usize; rows];

        for wave in 0..block_count {
            for left_block_idx in 0..wave {
                run_wavefront_offdiag_pair_for_test(
                    &context,
                    left_block_idx,
                    wave,
                    &seeded,
                    &mut top,
                    &mut scratch,
                    &mut counters,
                );
                record_wavefront_block_commit_for_rows(
                    &context.layout,
                    left_block_idx,
                    wave,
                    &mut next_candidate_block_by_row,
                    &mut counters,
                );
                record_wavefront_block_commit_for_rows(
                    &context.layout,
                    wave,
                    left_block_idx,
                    &mut next_candidate_block_by_row,
                    &mut counters,
                );
            }
            run_wavefront_diagonal_block_directed_for_test(
                &context,
                wave,
                &seeded,
                &mut top,
                &mut scratch,
                &mut counters,
            );
            record_wavefront_block_commit_for_rows(
                &context.layout,
                wave,
                wave,
                &mut next_candidate_block_by_row,
                &mut counters,
            );
        }
        for next in next_candidate_block_by_row {
            if next != block_count {
                counters.candidate_block_order_violations += 1;
            }
        }

        let row_topk = top.into_iter().map(|top| top.entries()).collect::<Vec<_>>();
        let (self_edge_violations, duplicate_topk_violations) =
            leaf_topk_integrity_violations(&row_topk);
        Ok(WavefrontPairMaskRun {
            row_topk,
            elapsed_ns: duration_nanos_u64(total_start.elapsed()),
            seed_evals,
            group_evals: counters.group_evals,
            full_evals: counters.full_evals,
            pair_pruned_both: counters.pair_pruned_both,
            pair_full: counters.pair_full,
            side_pruned: counters.side_pruned,
            side_full_updates: counters.side_full_updates,
            commit_ns: counters.commit_ns,
            prune_ns: counters.prune_ns,
            scratch_clear_ns: counters.scratch_clear_ns,
            candidate_block_order_violations: counters.candidate_block_order_violations,
            self_edge_violations,
            duplicate_topk_violations,
        })
    }

    fn initialize_wavefront_mutual_seed_for_test(
        layout: &AdSamplingLeafLayout<'_>,
        k: usize,
        seed_exact: usize,
    ) -> (Vec<TinyTopK>, Vec<u8>, u64) {
        let rows = layout.rows;
        let mut top = vec![TinyTopK::new(k); rows];
        let mut seeded = vec![0_u8; rows.saturating_mul(rows)];
        let mut seed_evals = 0_u64;
        for row in 0..rows {
            for offset in 1..=seed_exact {
                let candidate = (row + offset) % rows;
                let left = row.min(candidate);
                let right = row.max(candidate);
                if left == right || seeded[left * rows + right] != 0 {
                    continue;
                }
                seeded[left * rows + right] = 1;
                seeded[right * rows + left] = 1;
                let dist = l2_sq(layout.row(left), layout.row(right));
                top[left].push(dist, right);
                top[right].push(dist, left);
                seed_evals += 1;
            }
        }
        (top, seeded, seed_evals)
    }

    fn run_wavefront_offdiag_pair_for_test(
        context: &LeafAdsContext<'_>,
        left_block_idx: usize,
        right_block_idx: usize,
        seeded: &[u8],
        top: &mut [TinyTopK],
        scratch: &mut WavefrontPairScratch,
        counters: &mut WavefrontPairMaskCounters,
    ) {
        debug_assert!(left_block_idx < right_block_idx);
        let layout = &context.layout;
        let left_block = &layout.blocks[left_block_idx];
        let right_block = &layout.blocks[right_block_idx];

        let clear_start = Instant::now();
        for row in scratch.distances.iter_mut().take(left_block.len) {
            row.fill(0.0);
        }
        scratch.survivor_masks.fill(0);
        scratch.left_thresholds.fill(f32::INFINITY);
        scratch.right_thresholds.fill(f32::INFINITY);
        counters.scratch_clear_ns += duration_nanos_u64(clear_start.elapsed());

        for left_local in 0..left_block.len {
            let left = left_block.start + left_local;
            let mut active = 0_u64;
            for right_local in 0..right_block.len {
                let right = right_block.start + right_local;
                if !wavefront_seeded_pair(seeded, layout.rows, left, right) {
                    active |= 1_u64 << right_local;
                }
            }
            scratch.survivor_masks[left_local] = active;
            scratch.left_thresholds[left_local] = top[left].threshold();
        }
        for right_local in 0..right_block.len {
            scratch.right_thresholds[right_local] =
                top[right_block.start + right_local].threshold();
        }

        for start_dim in (0..layout.dim).step_by(context.ratios.group_dims()) {
            if scratch.survivor_masks[..left_block.len]
                .iter()
                .all(|&mask| mask == 0)
            {
                break;
            }
            let end_dim = (start_dim + context.ratios.group_dims()).min(layout.dim);
            for left_tile_start in (0..left_block.len).step_by(ADS_QUERY_TILE) {
                let query_count = (left_block.len - left_tile_start).min(ADS_QUERY_TILE);
                let active_masks = wavefront_active_tile_masks(
                    &scratch.survivor_masks,
                    left_tile_start,
                    query_count,
                );
                let active_count = active_masks[..query_count]
                    .iter()
                    .map(|mask| mask.count_ones() as u64)
                    .sum::<u64>();
                if active_count == 0 {
                    continue;
                }
                let first_query = layout.row(left_block.start + left_tile_start);
                let mut queries = [first_query; ADS_QUERY_TILE];
                for query_idx in 0..query_count {
                    queries[query_idx] = layout.row(left_block.start + left_tile_start + query_idx);
                }
                counters.group_evals += active_count;
                accumulate_wavefront_pair_batch8_for_test(
                    queries,
                    query_count,
                    right_block,
                    start_dim,
                    end_dim,
                    active_masks,
                    active_count as u32,
                    &mut scratch.distances,
                    left_tile_start,
                    context.config.sparse_full64_threshold,
                    context.config.sparse_group16_threshold,
                );

                let prune_start = Instant::now();
                let ratio = context.ratios.ratio_after_visited(end_dim);
                for query_idx in 0..query_count {
                    let left_local = left_tile_start + query_idx;
                    let before = scratch.survivor_masks[left_local];
                    let after = prune_wavefront_pair_mask_for_test(
                        before,
                        &scratch.distances[left_local],
                        scratch.left_thresholds[left_local],
                        &scratch.right_thresholds,
                        ratio,
                    );
                    let pruned = (before & !after).count_ones() as u64;
                    scratch.survivor_masks[left_local] = after;
                    counters.pair_pruned_both += pruned;
                    counters.side_pruned += pruned.saturating_mul(2);
                }
                counters.prune_ns += duration_nanos_u64(prune_start.elapsed());
            }
        }

        let commit_start = Instant::now();
        for left_local in 0..left_block.len {
            let left = left_block.start + left_local;
            let mut survivors = scratch.survivor_masks[left_local];
            while survivors != 0 {
                let right_local = survivors.trailing_zeros() as usize;
                let right = right_block.start + right_local;
                let dist = scratch.distances[left_local][right_local];
                top[left].push(dist, right);
                top[right].push(dist, left);
                counters.pair_full += 1;
                counters.full_evals += 1;
                counters.side_full_updates += 2;
                survivors &= survivors - 1;
            }
        }
        counters.commit_ns += duration_nanos_u64(commit_start.elapsed());
    }

    fn run_wavefront_diagonal_block_directed_for_test(
        context: &LeafAdsContext<'_>,
        block_idx: usize,
        seeded: &[u8],
        top: &mut [TinyTopK],
        scratch: &mut WavefrontPairScratch,
        counters: &mut WavefrontPairMaskCounters,
    ) {
        let layout = &context.layout;
        let block = &layout.blocks[block_idx];
        for row_tile_start in (0..block.len).step_by(ADS_QUERY_TILE) {
            let query_count = (block.len - row_tile_start).min(ADS_QUERY_TILE);
            let clear_start = Instant::now();
            for row in scratch.distances.iter_mut().take(query_count) {
                row.fill(0.0);
            }
            scratch.survivor_masks.fill(0);
            counters.scratch_clear_ns += duration_nanos_u64(clear_start.elapsed());

            let first_query = layout.row(block.start + row_tile_start);
            let mut queries = [first_query; ADS_QUERY_TILE];
            for query_idx in 0..query_count {
                let row = block.start + row_tile_start + query_idx;
                queries[query_idx] = layout.row(row);
                let mut active = block_len_mask(block.len);
                active &= !(1_u64 << (row - block.start));
                for local in 0..block.len {
                    let candidate = block.start + local;
                    if wavefront_seeded_pair(seeded, layout.rows, row, candidate) {
                        active &= !(1_u64 << local);
                    }
                }
                scratch.survivor_masks[query_idx] = active;
            }

            for start_dim in (0..layout.dim).step_by(context.ratios.group_dims()) {
                if scratch.survivor_masks[..query_count]
                    .iter()
                    .all(|&mask| mask == 0)
                {
                    break;
                }
                let end_dim = (start_dim + context.ratios.group_dims()).min(layout.dim);
                let active_masks =
                    wavefront_active_tile_masks(&scratch.survivor_masks, 0, query_count);
                let active_count = active_masks[..query_count]
                    .iter()
                    .map(|mask| mask.count_ones() as u64)
                    .sum::<u64>();
                if active_count == 0 {
                    continue;
                }
                counters.group_evals += active_count;
                accumulate_wavefront_pair_batch8_for_test(
                    queries,
                    query_count,
                    block,
                    start_dim,
                    end_dim,
                    active_masks,
                    active_count as u32,
                    &mut scratch.distances,
                    0,
                    context.config.sparse_full64_threshold,
                    context.config.sparse_group16_threshold,
                );

                let prune_start = Instant::now();
                let ratio = context.ratios.ratio_after_visited(end_dim);
                for query_idx in 0..query_count {
                    let row = block.start + row_tile_start + query_idx;
                    let before = scratch.survivor_masks[query_idx];
                    let after = prune_ads_mask(
                        before,
                        &scratch.distances[query_idx],
                        top[row].threshold() * ratio,
                    );
                    let pruned = (before & !after).count_ones() as u64;
                    scratch.survivor_masks[query_idx] = after;
                    counters.side_pruned += pruned;
                }
                counters.prune_ns += duration_nanos_u64(prune_start.elapsed());
            }

            let commit_start = Instant::now();
            for query_idx in 0..query_count {
                let row = block.start + row_tile_start + query_idx;
                let survivors = scratch.survivor_masks[query_idx];
                counters.full_evals += push_active_distances(
                    &mut top[row],
                    block,
                    survivors,
                    &scratch.distances[query_idx],
                );
            }
            counters.commit_ns += duration_nanos_u64(commit_start.elapsed());
        }
    }

    fn record_wavefront_block_commit_for_rows(
        layout: &AdSamplingLeafLayout<'_>,
        row_block_idx: usize,
        candidate_block_idx: usize,
        next_candidate_block_by_row: &mut [usize],
        counters: &mut WavefrontPairMaskCounters,
    ) {
        let row_block = &layout.blocks[row_block_idx];
        for row in row_block.start..row_block.start + row_block.len {
            if next_candidate_block_by_row[row] != candidate_block_idx {
                counters.candidate_block_order_violations += 1;
            }
            next_candidate_block_by_row[row] = candidate_block_idx.saturating_add(1);
        }
    }

    fn wavefront_active_tile_masks(
        masks: &[u64; ADS_LEADER_BLOCK],
        start: usize,
        query_count: usize,
    ) -> [u64; ADS_QUERY_TILE] {
        let mut active = [0_u64; ADS_QUERY_TILE];
        active[..query_count].copy_from_slice(&masks[start..start + query_count]);
        active
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_wavefront_pair_batch8_for_test(
        points: [&[f32]; ADS_QUERY_TILE],
        query_count: usize,
        block: &AdSamplingLeaderBlock,
        start_dim: usize,
        end_dim: usize,
        active_masks: [u64; ADS_QUERY_TILE],
        active_count: u32,
        distances: &mut [[f32; ADS_LEADER_BLOCK]; ADS_LEADER_BLOCK],
        row_offset: usize,
        sparse_full64_threshold: u32,
        sparse_group16_threshold: u32,
    ) -> bool {
        if should_use_sparse_ads_fallback(
            &active_masks[..query_count],
            active_count,
            sparse_full64_threshold,
            sparse_group16_threshold,
        ) {
            for query_idx in 0..query_count {
                accumulate_ads_soa_group_mask_scalar(
                    points[query_idx],
                    block,
                    start_dim,
                    end_dim,
                    active_masks[query_idx],
                    &mut distances[row_offset + query_idx],
                );
            }
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if adsampling_avx512_available() {
                unsafe {
                    accumulate_wavefront_pair_batch8_avx512_for_test(
                        points,
                        block,
                        start_dim,
                        end_dim,
                        active_masks,
                        distances,
                        row_offset,
                    );
                }
                return true;
            }
        }
        for query_idx in 0..query_count {
            accumulate_ads_soa_group_mask_scalar(
                points[query_idx],
                block,
                start_dim,
                end_dim,
                active_masks[query_idx],
                &mut distances[row_offset + query_idx],
            );
        }
        false
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn accumulate_wavefront_pair_batch8_avx512_for_test(
        points: [&[f32]; ADS_QUERY_TILE],
        block: &AdSamplingLeaderBlock,
        start_dim: usize,
        end_dim: usize,
        active_masks: [u64; ADS_QUERY_TILE],
        distances: &mut [[f32; ADS_LEADER_BLOCK]; ADS_LEADER_BLOCK],
        row_offset: usize,
    ) {
        use std::arch::x86_64::*;

        let point_ptrs = [
            points[0].as_ptr(),
            points[1].as_ptr(),
            points[2].as_ptr(),
            points[3].as_ptr(),
            points[4].as_ptr(),
            points[5].as_ptr(),
            points[6].as_ptr(),
            points[7].as_ptr(),
        ];

        for local in (0..ADS_LEADER_BLOCK).step_by(16) {
            let lane_masks = [
                ((active_masks[0] >> local) & 0xffff) as __mmask16,
                ((active_masks[1] >> local) & 0xffff) as __mmask16,
                ((active_masks[2] >> local) & 0xffff) as __mmask16,
                ((active_masks[3] >> local) & 0xffff) as __mmask16,
                ((active_masks[4] >> local) & 0xffff) as __mmask16,
                ((active_masks[5] >> local) & 0xffff) as __mmask16,
                ((active_masks[6] >> local) & 0xffff) as __mmask16,
                ((active_masks[7] >> local) & 0xffff) as __mmask16,
            ];
            if lane_masks.iter().all(|&mask| mask == 0) {
                continue;
            }

            let mut acc0 = unsafe { _mm512_loadu_ps(distances[row_offset].as_ptr().add(local)) };
            let mut acc1 =
                unsafe { _mm512_loadu_ps(distances[row_offset + 1].as_ptr().add(local)) };
            let mut acc2 =
                unsafe { _mm512_loadu_ps(distances[row_offset + 2].as_ptr().add(local)) };
            let mut acc3 =
                unsafe { _mm512_loadu_ps(distances[row_offset + 3].as_ptr().add(local)) };
            let mut acc4 =
                unsafe { _mm512_loadu_ps(distances[row_offset + 4].as_ptr().add(local)) };
            let mut acc5 =
                unsafe { _mm512_loadu_ps(distances[row_offset + 5].as_ptr().add(local)) };
            let mut acc6 =
                unsafe { _mm512_loadu_ps(distances[row_offset + 6].as_ptr().add(local)) };
            let mut acc7 =
                unsafe { _mm512_loadu_ps(distances[row_offset + 7].as_ptr().add(local)) };
            for dim_idx in start_dim..end_dim {
                let values = unsafe {
                    _mm512_loadu_ps(block.data.as_ptr().add(dim_idx * block.stride + local))
                };
                if lane_masks[0] != 0 {
                    let q = _mm512_set1_ps(unsafe { *point_ptrs[0].add(dim_idx) });
                    let diff = _mm512_sub_ps(q, values);
                    acc0 = _mm512_fmadd_ps(diff, diff, acc0);
                }
                if lane_masks[1] != 0 {
                    let q = _mm512_set1_ps(unsafe { *point_ptrs[1].add(dim_idx) });
                    let diff = _mm512_sub_ps(q, values);
                    acc1 = _mm512_fmadd_ps(diff, diff, acc1);
                }
                if lane_masks[2] != 0 {
                    let q = _mm512_set1_ps(unsafe { *point_ptrs[2].add(dim_idx) });
                    let diff = _mm512_sub_ps(q, values);
                    acc2 = _mm512_fmadd_ps(diff, diff, acc2);
                }
                if lane_masks[3] != 0 {
                    let q = _mm512_set1_ps(unsafe { *point_ptrs[3].add(dim_idx) });
                    let diff = _mm512_sub_ps(q, values);
                    acc3 = _mm512_fmadd_ps(diff, diff, acc3);
                }
                if lane_masks[4] != 0 {
                    let q = _mm512_set1_ps(unsafe { *point_ptrs[4].add(dim_idx) });
                    let diff = _mm512_sub_ps(q, values);
                    acc4 = _mm512_fmadd_ps(diff, diff, acc4);
                }
                if lane_masks[5] != 0 {
                    let q = _mm512_set1_ps(unsafe { *point_ptrs[5].add(dim_idx) });
                    let diff = _mm512_sub_ps(q, values);
                    acc5 = _mm512_fmadd_ps(diff, diff, acc5);
                }
                if lane_masks[6] != 0 {
                    let q = _mm512_set1_ps(unsafe { *point_ptrs[6].add(dim_idx) });
                    let diff = _mm512_sub_ps(q, values);
                    acc6 = _mm512_fmadd_ps(diff, diff, acc6);
                }
                if lane_masks[7] != 0 {
                    let q = _mm512_set1_ps(unsafe { *point_ptrs[7].add(dim_idx) });
                    let diff = _mm512_sub_ps(q, values);
                    acc7 = _mm512_fmadd_ps(diff, diff, acc7);
                }
            }
            if lane_masks[0] != 0 {
                unsafe { _mm512_storeu_ps(distances[row_offset].as_mut_ptr().add(local), acc0) };
            }
            if lane_masks[1] != 0 {
                unsafe {
                    _mm512_storeu_ps(distances[row_offset + 1].as_mut_ptr().add(local), acc1)
                };
            }
            if lane_masks[2] != 0 {
                unsafe {
                    _mm512_storeu_ps(distances[row_offset + 2].as_mut_ptr().add(local), acc2)
                };
            }
            if lane_masks[3] != 0 {
                unsafe {
                    _mm512_storeu_ps(distances[row_offset + 3].as_mut_ptr().add(local), acc3)
                };
            }
            if lane_masks[4] != 0 {
                unsafe {
                    _mm512_storeu_ps(distances[row_offset + 4].as_mut_ptr().add(local), acc4)
                };
            }
            if lane_masks[5] != 0 {
                unsafe {
                    _mm512_storeu_ps(distances[row_offset + 5].as_mut_ptr().add(local), acc5)
                };
            }
            if lane_masks[6] != 0 {
                unsafe {
                    _mm512_storeu_ps(distances[row_offset + 6].as_mut_ptr().add(local), acc6)
                };
            }
            if lane_masks[7] != 0 {
                unsafe {
                    _mm512_storeu_ps(distances[row_offset + 7].as_mut_ptr().add(local), acc7)
                };
            }
        }
    }

    #[inline]
    fn prune_wavefront_pair_mask_for_test(
        active_mask: u64,
        distances: &[f32; ADS_LEADER_BLOCK],
        left_threshold: f32,
        right_thresholds: &[f32; ADS_LEADER_BLOCK],
        ratio: f32,
    ) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if adsampling_avx512_available() {
                return unsafe {
                    prune_wavefront_pair_mask_avx512_for_test(
                        active_mask,
                        distances,
                        left_threshold,
                        right_thresholds,
                        ratio,
                    )
                };
            }
        }
        prune_wavefront_pair_mask_scalar_for_test(
            active_mask,
            distances,
            left_threshold,
            right_thresholds,
            ratio,
        )
    }

    #[inline]
    fn prune_wavefront_pair_mask_scalar_for_test(
        mut active_mask: u64,
        distances: &[f32; ADS_LEADER_BLOCK],
        left_threshold: f32,
        right_thresholds: &[f32; ADS_LEADER_BLOCK],
        ratio: f32,
    ) -> u64 {
        let mut keep = 0_u64;
        while active_mask != 0 {
            let local = active_mask.trailing_zeros() as usize;
            if distances[local] < ratio * left_threshold.max(right_thresholds[local]) {
                keep |= 1_u64 << local;
            }
            active_mask &= active_mask - 1;
        }
        keep
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn prune_wavefront_pair_mask_avx512_for_test(
        active_mask: u64,
        distances: &[f32; ADS_LEADER_BLOCK],
        left_threshold: f32,
        right_thresholds: &[f32; ADS_LEADER_BLOCK],
        ratio: f32,
    ) -> u64 {
        use std::arch::x86_64::*;

        let left = _mm512_set1_ps(left_threshold);
        let ratio = _mm512_set1_ps(ratio);
        let mut keep_mask = 0_u64;
        for local in (0..ADS_LEADER_BLOCK).step_by(16) {
            let lane_active = ((active_mask >> local) & 0xffff) as __mmask16;
            if lane_active == 0 {
                continue;
            }
            let dist = unsafe { _mm512_loadu_ps(distances.as_ptr().add(local)) };
            let right = unsafe { _mm512_loadu_ps(right_thresholds.as_ptr().add(local)) };
            let threshold = _mm512_mul_ps(_mm512_max_ps(left, right), ratio);
            let keep = _mm512_cmp_ps_mask(dist, threshold, _CMP_LT_OQ) & lane_active;
            keep_mask |= (keep as u64) << local;
        }
        keep_mask
    }

    #[inline]
    fn wavefront_seeded_pair(seeded: &[u8], rows: usize, row: usize, candidate: usize) -> bool {
        seeded
            .get(row.saturating_mul(rows).saturating_add(candidate))
            .copied()
            .unwrap_or(0)
            != 0
    }

    fn baseline_candidate_block_sequences_for_test(rows: usize) -> Vec<Vec<usize>> {
        let block_count = rows.div_ceil(ADS_LEADER_BLOCK);
        vec![(0..block_count).collect::<Vec<_>>(); rows]
    }

    fn wavefront_candidate_block_sequences_for_test(rows: usize) -> Vec<Vec<usize>> {
        let block_count = rows.div_ceil(ADS_LEADER_BLOCK);
        let mut sequences = vec![Vec::with_capacity(block_count); rows];
        for wave in 0..block_count {
            for left_block_idx in 0..wave {
                let left = block_row_range_for_test(rows, left_block_idx);
                let right = block_row_range_for_test(rows, wave);
                for row in left {
                    sequences[row].push(wave);
                }
                for row in right {
                    sequences[row].push(left_block_idx);
                }
            }
            for row in block_row_range_for_test(rows, wave) {
                sequences[row].push(wave);
            }
        }
        sequences
    }

    fn block_row_range_for_test(rows: usize, block_idx: usize) -> std::ops::Range<usize> {
        let start = block_idx * ADS_LEADER_BLOCK;
        start..(start + ADS_LEADER_BLOCK).min(rows)
    }

    fn deterministic_leaf_vectors(rows: usize, dim: usize) -> Vec<f32> {
        (0..rows)
            .flat_map(|row| {
                (0..dim).map(move |col| {
                    let value = (row * 37 + col * 17 + row * col * 3) % 251;
                    value as f32 * 0.00390625
                })
            })
            .collect()
    }

    fn take_median_wavefront_pairmask_run(
        mut runs: Vec<WavefrontPairMaskRun>,
    ) -> WavefrontPairMaskRun {
        runs.sort_by_key(|run| run.elapsed_ns);
        runs.swap_remove(runs.len() / 2)
    }

    fn leaf_topk_integrity_violations(rows: &[Vec<(usize, f32)>]) -> (u64, u64) {
        let mut self_edges = 0_u64;
        let mut duplicates = 0_u64;
        for (row, topk) in rows.iter().enumerate() {
            for (pos, &(candidate, _)) in topk.iter().enumerate() {
                if candidate == row {
                    self_edges += 1;
                }
                if topk[..pos]
                    .iter()
                    .any(|&(existing, _)| existing == candidate)
                {
                    duplicates += 1;
                }
            }
        }
        (self_edges, duplicates)
    }

    fn wavefront_pairmask_acceptance(
        bucket: &str,
        median_wall_ratio: f64,
        median_group_ratio: f64,
        median_output_overlap: f64,
        median_candidate_exact_recall_delta: f64,
        max_order_violations: u64,
        max_self_edge_violations: u64,
        max_duplicate_topk_violations: u64,
    ) -> &'static str {
        let quality_ok = median_group_ratio <= 0.70
            && median_output_overlap >= 0.97
            && median_candidate_exact_recall_delta >= -0.002
            && max_order_violations == 0
            && max_self_edge_violations == 0
            && max_duplicate_topk_violations == 0;
        match bucket {
            "[1024,2048)" => {
                if quality_ok && median_wall_ratio <= 0.85 {
                    "pass"
                } else {
                    "fail"
                }
            }
            "[512,1024)" => {
                if quality_ok && median_wall_ratio <= 0.90 {
                    "pass"
                } else if quality_ok {
                    "disabled-small-bucket"
                } else {
                    "fail"
                }
            }
            _ => {
                if quality_ok {
                    "not-gated"
                } else {
                    "fail"
                }
            }
        }
    }

    fn finite_ratio_u64(numerator: u64, denominator: u64) -> f64 {
        if denominator == 0 {
            0.0
        } else {
            numerator as f64 / denominator as f64
        }
    }

    fn nanos_to_ms(nanos: u64) -> f64 {
        Duration::from_nanos(nanos).as_secs_f64() * 1000.0
    }

    fn read_real_leaf_records(
        manifest_path: &Path,
        samples_per_bucket: usize,
    ) -> AnnResult<Vec<RealLeafRecord>> {
        let mut reader = BufReader::new(File::open(manifest_path)?);
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"ADSL01M1" {
            return Err(AnnError::log_index_error(format!(
                "unsupported leaf manifest magic {:?} at {}",
                magic,
                manifest_path.display()
            )));
        }
        let record_count = read_u64(&mut reader)? as usize;
        let mut by_bucket: BTreeMap<String, Vec<RealLeafRecord>> = BTreeMap::new();
        for child_index in 0..record_count {
            let len = read_u64(&mut reader)? as usize;
            let _seed = read_u64(&mut reader)?;
            let extent_count = read_u64(&mut reader)? as usize;
            let mut extents = Vec::with_capacity(extent_count);
            for _ in 0..extent_count {
                let byte_offset = read_u64(&mut reader)?;
                let len = read_u64(&mut reader)? as usize;
                extents.push(RealRunExtent { byte_offset, len });
            }
            let Some(bucket) = real_leaf_bucket(len) else {
                continue;
            };
            let entries = by_bucket.entry(bucket.to_string()).or_default();
            if entries.len() < samples_per_bucket {
                entries.push(RealLeafRecord {
                    child_index,
                    len,
                    bucket: bucket.to_string(),
                    extents,
                });
            }
            if by_bucket.len() >= 4
                && by_bucket
                    .values()
                    .all(|records| records.len() >= samples_per_bucket)
            {
                break;
            }
        }
        Ok(by_bucket.into_values().flatten().collect())
    }

    fn read_real_leaf_points(manifest_path: &Path, record: &RealLeafRecord) -> AnnResult<Vec<u32>> {
        let depth = manifest_depth_from_path(manifest_path).unwrap_or(1);
        let run_path = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("partition_d{depth:02}_leaf_runs.bin"));
        let run_file = File::open(&run_path)?;
        let mut points = Vec::with_capacity(record.len);
        for extent in &record.extents {
            let mut header = [0_u8; 8];
            run_file.read_exact_at(&mut header, extent.byte_offset)?;
            let len = u64::from_le_bytes(header) as usize;
            if len != extent.len {
                return Err(AnnError::log_index_error(format!(
                    "extent length mismatch in {} at offset {}: manifest={} file={}",
                    run_path.display(),
                    extent.byte_offset,
                    extent.len,
                    len
                )));
            }
            let mut bytes = vec![0_u8; len * size_of::<u32>()];
            run_file.read_exact_at(&mut bytes, extent.byte_offset + 8)?;
            for chunk in bytes.chunks_exact(size_of::<u32>()) {
                points.push(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        Ok(points)
    }

    fn read_u64(reader: &mut BufReader<File>) -> AnnResult<u64> {
        let mut buf = [0_u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn manifest_depth_from_path(path: &Path) -> Option<usize> {
        let file_name = path.file_name()?.to_str()?;
        let marker = file_name.strip_prefix("partition_d")?;
        marker.get(0..2)?.parse().ok()
    }

    fn real_leaf_bucket(len: usize) -> Option<&'static str> {
        if (512..1024).contains(&len) {
            Some("[512,1024)")
        } else if (1024..2048).contains(&len) {
            Some("[1024,2048)")
        } else if (2048..4096).contains(&len) {
            Some("[2048,4096)")
        } else if len >= 4096 {
            Some(">=4096")
        } else {
            None
        }
    }

    fn compute_leaf_adsampling_topk_l2_legacy_for_test(
        vectors: &[f32],
        leaf_size: usize,
        dim: usize,
        k: usize,
        config: AdSamplingConfig,
    ) -> AnnResult<LeafAdSamplingResult> {
        if vectors.len() != leaf_size.saturating_mul(dim) {
            return Err(AnnError::log_index_error(format!(
                "Invalid legacy leaf ADSampling matrix: len={} leaf_size={leaf_size} dim={dim}",
                vectors.len()
            )));
        }
        let (context, layout_duration) = build_leaf_ads_context(vectors, leaf_size, dim, config)?;
        let mut telemetry = LeafAdSamplingTelemetry {
            layout: layout_duration,
            ..LeafAdSamplingTelemetry::default()
        };
        let mut row_topk = Vec::with_capacity(leaf_size);
        for row_start in (0..leaf_size).step_by(ADS_LEAF_ROW_CHUNK) {
            let row_end = (row_start + ADS_LEAF_ROW_CHUNK).min(leaf_size);
            for tile_start in (row_start..row_end).step_by(ADS_QUERY_TILE) {
                let tile_end = (tile_start + ADS_QUERY_TILE).min(row_end);
                let tile = compute_leaf_adsampling_row_tile_legacy_for_test(
                    &context.layout,
                    tile_start,
                    tile_end,
                    k,
                    context.config,
                    &context.ratios,
                );
                telemetry.merge(tile.telemetry);
                row_topk.extend(tile.row_topk);
            }
        }
        Ok(LeafAdSamplingResult {
            row_topk,
            telemetry,
        })
    }

    fn compute_leaf_adsampling_row_tile_legacy_for_test(
        layout: &AdSamplingLeafLayout<'_>,
        row_start: usize,
        row_end: usize,
        k: usize,
        config: AdSamplingConfig,
        ratios: &AdSamplingRatios,
    ) -> LeafAdSamplingTile {
        let query_count = row_end - row_start;
        let k = k.min(layout.rows.saturating_sub(1)).max(1);
        let seed_exact = config.seed_exact.min(layout.rows.saturating_sub(1));
        let mut telemetry = LeafAdSamplingTelemetry::default();
        let mut top = vec![TinyTopK::new(k); query_count];

        let seed_start = Instant::now();
        for (query_idx, row) in (row_start..row_end).enumerate() {
            let query = layout.row(row);
            for offset in 1..=seed_exact {
                let candidate = (row + offset) % layout.rows;
                top[query_idx].push(l2_sq(query, layout.row(candidate)), candidate);
                telemetry.seed_evals += 1;
            }
        }
        telemetry.seed += seed_start.elapsed();

        let first_query = layout.row(row_start);
        let mut queries = [first_query; ADS_QUERY_TILE];
        for (query_idx, row) in (row_start..row_end).enumerate() {
            queries[query_idx] = layout.row(row);
        }

        let scan_start = Instant::now();
        let mut distances = [[0.0_f32; ADS_LEADER_BLOCK]; ADS_QUERY_TILE];
        for block in &layout.blocks {
            for row in distances.iter_mut().take(query_count) {
                row.fill(0.0);
            }
            let mut active_masks = [0_u64; ADS_QUERY_TILE];
            for (query_idx, row) in (row_start..row_end).enumerate() {
                active_masks[query_idx] = leaf_adsampling_active_mask(
                    row,
                    block.start,
                    block.len,
                    layout.rows,
                    seed_exact,
                );
            }
            if active_masks[..query_count].iter().all(|&mask| mask == 0) {
                continue;
            }

            for start_dim in (0..layout.dim).step_by(ratios.group_dims()) {
                if active_masks[..query_count].iter().all(|&mask| mask == 0) {
                    break;
                }
                let end_dim = (start_dim + ratios.group_dims()).min(layout.dim);
                let active_before = active_masks;
                let active_count = active_before[..query_count]
                    .iter()
                    .map(|mask| mask.count_ones() as u64)
                    .sum::<u64>();
                telemetry.group_evals += active_count;
                if legacy_accumulate_ads_soa_group_batch8_for_test(
                    queries,
                    query_count,
                    block,
                    start_dim,
                    end_dim,
                    active_before,
                    &mut distances,
                    config.sparse_full64_threshold,
                    config.sparse_group16_threshold,
                ) {
                    telemetry.simd_group_calls += simd_batch8_group_call_count(active_before);
                    telemetry.simd_active_lane_evals += active_count;
                } else {
                    telemetry.scalar_group_evals += active_count;
                }

                let ratio = ratios.ratio_after_visited(end_dim);
                for query_idx in 0..query_count {
                    let before = active_masks[query_idx].count_ones() as u64;
                    active_masks[query_idx] = prune_ads_mask(
                        active_masks[query_idx],
                        &distances[query_idx],
                        top[query_idx].threshold() * ratio,
                    );
                    telemetry.pruned_evals += before - active_masks[query_idx].count_ones() as u64;
                }
            }

            for query_idx in 0..query_count {
                let survivors = active_masks[query_idx].count_ones() as u64;
                push_active_distances(
                    &mut top[query_idx],
                    block,
                    active_masks[query_idx],
                    &distances[query_idx],
                );
                telemetry.full_evals += survivors;
            }
        }
        telemetry.scan += scan_start.elapsed();

        LeafAdSamplingTile {
            row_topk: top.into_iter().map(|top| top.entries()).collect(),
            telemetry,
        }
    }

    #[inline]
    fn legacy_accumulate_ads_soa_group_batch8_for_test(
        points: [&[f32]; ADS_QUERY_TILE],
        query_count: usize,
        block: &AdSamplingLeaderBlock,
        start_dim: usize,
        end_dim: usize,
        active_masks: [u64; ADS_QUERY_TILE],
        distances: &mut [[f32; ADS_LEADER_BLOCK]; ADS_QUERY_TILE],
        sparse_full64_threshold: u32,
        sparse_group16_threshold: u32,
    ) -> bool {
        let active_count = active_masks[..query_count]
            .iter()
            .map(|mask| mask.count_ones())
            .sum::<u32>();
        if should_use_sparse_ads_fallback(
            &active_masks[..query_count],
            active_count,
            sparse_full64_threshold,
            sparse_group16_threshold,
        ) {
            for query_idx in 0..query_count {
                accumulate_ads_soa_group_mask_scalar(
                    points[query_idx],
                    block,
                    start_dim,
                    end_dim,
                    active_masks[query_idx],
                    &mut distances[query_idx],
                );
            }
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if adsampling_avx512_available() {
                unsafe {
                    legacy_accumulate_ads_soa_group_batch8_avx512_for_test(
                        points,
                        block,
                        start_dim,
                        end_dim,
                        active_masks,
                        distances,
                    );
                }
                return true;
            }
        }
        for query_idx in 0..query_count {
            accumulate_ads_soa_group_mask_scalar(
                points[query_idx],
                block,
                start_dim,
                end_dim,
                active_masks[query_idx],
                &mut distances[query_idx],
            );
        }
        false
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn legacy_accumulate_ads_soa_group_batch8_avx512_for_test(
        points: [&[f32]; ADS_QUERY_TILE],
        block: &AdSamplingLeaderBlock,
        start_dim: usize,
        end_dim: usize,
        active_masks: [u64; ADS_QUERY_TILE],
        distances: &mut [[f32; ADS_LEADER_BLOCK]; ADS_QUERY_TILE],
    ) {
        use std::arch::x86_64::*;

        for local in (0..ADS_LEADER_BLOCK).step_by(16) {
            let lane_masks = [
                ((active_masks[0] >> local) & 0xffff) as __mmask16,
                ((active_masks[1] >> local) & 0xffff) as __mmask16,
                ((active_masks[2] >> local) & 0xffff) as __mmask16,
                ((active_masks[3] >> local) & 0xffff) as __mmask16,
                ((active_masks[4] >> local) & 0xffff) as __mmask16,
                ((active_masks[5] >> local) & 0xffff) as __mmask16,
                ((active_masks[6] >> local) & 0xffff) as __mmask16,
                ((active_masks[7] >> local) & 0xffff) as __mmask16,
            ];
            if lane_masks.iter().all(|&mask| mask == 0) {
                continue;
            }

            let mut acc0 = unsafe { _mm512_loadu_ps(distances[0].as_ptr().add(local)) };
            let mut acc1 = unsafe { _mm512_loadu_ps(distances[1].as_ptr().add(local)) };
            let mut acc2 = unsafe { _mm512_loadu_ps(distances[2].as_ptr().add(local)) };
            let mut acc3 = unsafe { _mm512_loadu_ps(distances[3].as_ptr().add(local)) };
            let mut acc4 = unsafe { _mm512_loadu_ps(distances[4].as_ptr().add(local)) };
            let mut acc5 = unsafe { _mm512_loadu_ps(distances[5].as_ptr().add(local)) };
            let mut acc6 = unsafe { _mm512_loadu_ps(distances[6].as_ptr().add(local)) };
            let mut acc7 = unsafe { _mm512_loadu_ps(distances[7].as_ptr().add(local)) };
            for dim_idx in start_dim..end_dim {
                let values = unsafe {
                    _mm512_loadu_ps(block.data.as_ptr().add(dim_idx * block.stride + local))
                };
                if lane_masks[0] != 0 {
                    let q = _mm512_set1_ps(points[0][dim_idx]);
                    let diff = _mm512_sub_ps(q, values);
                    acc0 = _mm512_fmadd_ps(diff, diff, acc0);
                }
                if lane_masks[1] != 0 {
                    let q = _mm512_set1_ps(points[1][dim_idx]);
                    let diff = _mm512_sub_ps(q, values);
                    acc1 = _mm512_fmadd_ps(diff, diff, acc1);
                }
                if lane_masks[2] != 0 {
                    let q = _mm512_set1_ps(points[2][dim_idx]);
                    let diff = _mm512_sub_ps(q, values);
                    acc2 = _mm512_fmadd_ps(diff, diff, acc2);
                }
                if lane_masks[3] != 0 {
                    let q = _mm512_set1_ps(points[3][dim_idx]);
                    let diff = _mm512_sub_ps(q, values);
                    acc3 = _mm512_fmadd_ps(diff, diff, acc3);
                }
                if lane_masks[4] != 0 {
                    let q = _mm512_set1_ps(points[4][dim_idx]);
                    let diff = _mm512_sub_ps(q, values);
                    acc4 = _mm512_fmadd_ps(diff, diff, acc4);
                }
                if lane_masks[5] != 0 {
                    let q = _mm512_set1_ps(points[5][dim_idx]);
                    let diff = _mm512_sub_ps(q, values);
                    acc5 = _mm512_fmadd_ps(diff, diff, acc5);
                }
                if lane_masks[6] != 0 {
                    let q = _mm512_set1_ps(points[6][dim_idx]);
                    let diff = _mm512_sub_ps(q, values);
                    acc6 = _mm512_fmadd_ps(diff, diff, acc6);
                }
                if lane_masks[7] != 0 {
                    let q = _mm512_set1_ps(points[7][dim_idx]);
                    let diff = _mm512_sub_ps(q, values);
                    acc7 = _mm512_fmadd_ps(diff, diff, acc7);
                }
            }
            if lane_masks[0] != 0 {
                unsafe { _mm512_storeu_ps(distances[0].as_mut_ptr().add(local), acc0) };
            }
            if lane_masks[1] != 0 {
                unsafe { _mm512_storeu_ps(distances[1].as_mut_ptr().add(local), acc1) };
            }
            if lane_masks[2] != 0 {
                unsafe { _mm512_storeu_ps(distances[2].as_mut_ptr().add(local), acc2) };
            }
            if lane_masks[3] != 0 {
                unsafe { _mm512_storeu_ps(distances[3].as_mut_ptr().add(local), acc3) };
            }
            if lane_masks[4] != 0 {
                unsafe { _mm512_storeu_ps(distances[4].as_mut_ptr().add(local), acc4) };
            }
            if lane_masks[5] != 0 {
                unsafe { _mm512_storeu_ps(distances[5].as_mut_ptr().add(local), acc5) };
            }
            if lane_masks[6] != 0 {
                unsafe { _mm512_storeu_ps(distances[6].as_mut_ptr().add(local), acc6) };
            }
            if lane_masks[7] != 0 {
                unsafe { _mm512_storeu_ps(distances[7].as_mut_ptr().add(local), acc7) };
            }
        }
    }

    fn median_u64(values: &mut [u64]) -> u64 {
        values.sort_unstable();
        values[values.len() / 2]
    }

    fn median_f64(values: &mut [f64]) -> f64 {
        values.sort_by(|left, right| left.total_cmp(right));
        values[values.len() / 2]
    }

    fn parse_bool_env(name: &str) -> bool {
        std::env::var(name)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    }

    fn parse_f32_env(name: &str) -> Option<f32> {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
    }

    fn parse_usize_env(name: &str) -> Option<usize> {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
    }

    fn parse_u32_env(name: &str) -> Option<u32> {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
    }

    fn hash_leaf_topk(rows: &[Vec<(usize, f32)>]) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        for row in rows {
            hasher.update((row.len() as u64).to_le_bytes());
            for &(idx, dist) in row {
                hasher.update((idx as u64).to_le_bytes());
                hasher.update(dist.to_bits().to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    fn leaf_topk_overlap_ratio(
        current: &[Vec<(usize, f32)>],
        baseline: &[Vec<(usize, f32)>],
    ) -> f64 {
        assert_eq!(current.len(), baseline.len());
        let mut hits = 0usize;
        let mut total = 0usize;
        for (current_row, baseline_row) in current.iter().zip(baseline.iter()) {
            total += baseline_row.len();
            for &(idx, _) in current_row {
                if baseline_row
                    .iter()
                    .any(|&(baseline_idx, _)| baseline_idx == idx)
                {
                    hits += 1;
                }
            }
        }
        if total == 0 {
            1.0
        } else {
            hits as f64 / total as f64
        }
    }

    fn assert_leaf_topk_close(actual: &[Vec<(usize, f32)>], expected: &[Vec<(usize, f32)>]) {
        assert_eq!(actual.len(), expected.len());
        for (row, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                actual_row.len(),
                expected_row.len(),
                "row {row} length mismatch"
            );
            let mut actual_indices = actual_row.iter().map(|&(idx, _)| idx).collect::<Vec<_>>();
            let mut expected_indices = expected_row.iter().map(|&(idx, _)| idx).collect::<Vec<_>>();
            actual_indices.sort_unstable();
            expected_indices.sort_unstable();
            assert_eq!(
                actual_indices, expected_indices,
                "row {row} index set mismatch"
            );

            for &(actual_idx, actual_dist) in actual_row {
                let expected_dist = expected_row
                    .iter()
                    .find_map(|&(idx, dist)| (idx == actual_idx).then_some(dist))
                    .expect("index set was checked above");
                assert!(
                    (actual_dist - expected_dist).abs() <= 1.0e-5,
                    "row {row} idx {actual_idx} distance mismatch: actual={actual_dist} expected={expected_dist}"
                );
            }
        }
    }

    fn exact_leaf_topk(
        vectors: &[f32],
        leaf_size: usize,
        dim: usize,
        k: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut rows = Vec::with_capacity(leaf_size);
        for row in 0..leaf_size {
            let query = &vectors[row * dim..(row + 1) * dim];
            let mut top = TinyTopK::new(k);
            for candidate in 0..leaf_size {
                if candidate == row {
                    continue;
                }
                top.push(
                    l2_sq(query, &vectors[candidate * dim..(candidate + 1) * dim]),
                    candidate,
                );
            }
            rows.push(top.entries());
        }
        rows
    }
}
