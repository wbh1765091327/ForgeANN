use std::cmp;
use std::mem::size_of;
use std::time::{Duration, Instant};

use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};
use parking_lot::Mutex;
use rayon::prelude::*;

use super::adsampling::{
    AdSamplingConfig, LEAF_ADS_EWMA_BUCKETS, LeafAdSamplingTelemetry, LeafAdsOperatorRuntime,
    compute_leaf_adsampling_topk_l2_wavefront_pairmask,
    compute_leaf_adsampling_topk_l2_with_runtime,
};
use super::hash_prune::{HashPruneReservoir, SketchAccessor, compute_hash};
use super::params::ForgeANNParams;
use super::point_store::{PointBatchStats, PointStore, WindowedGatherOptions};
use crate::common::{AnnResult, Metric};

/// Prefetched leaf-local sketch data for fast in-memory hash computation.
pub(crate) struct LeafSketchCache {
    data: Vec<f32>,
    width: usize,
}

impl LeafSketchCache {
    pub(crate) fn prefetch(sketches: &dyn SketchAccessor, leaf: &[u32]) -> AnnResult<Self> {
        let width = sketches.width();
        let mut data = vec![0.0f32; leaf.len() * width];
        let row_bytes = width.saturating_mul(size_of::<f32>());
        let options = WindowedGatherOptions {
            max_gap_rows: if row_bytes <= 4096 { 1 } else { 0 },
            max_window_bytes: row_bytes
                .saturating_mul(leaf.len().max(1))
                .min(2 * 1024 * 1024)
                .max(128 * 1024),
            alignment_bytes: 4096,
            sort_ids: true,
        };
        let mut stats = super::point_store::WindowedGatherStats::default();
        sketches.read_rows_windowed_into_stats(leaf, &mut data, &options, &mut stats)?;
        Ok(Self { data, width })
    }

    #[inline]
    pub(crate) fn row(&self, local_idx: usize) -> &[f32] {
        let start = local_idx * self.width;
        &self.data[start..start + self.width]
    }
}

/// 暂存待插入的边，用于减少锁竞争
#[derive(Clone, Debug)]
pub struct PendingEdge {
    pub p: usize,
    pub c: u32,
    pub hash: u16,
    pub dist: f32,
    pub mandatory: bool,
    pub local_rank: u8,
    pub flags: u8,
}

pub const PENDING_EDGE_DIRECT: u8 = 1;
pub const PENDING_EDGE_MIRROR: u8 = 1 << 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingLuneWitness {
    pub src: u32,
    pub pivot: u32,
    pub victim: u32,
    pub margin_q16: u16,
    pub pivot_rank: u8,
    pub victim_rank: u8,
}

pub(crate) trait PendingEdgeSink: Sync {
    fn flush_pending_edges(&self, edges: &mut Vec<PendingEdge>) -> AnnResult<()>;

    fn wants_lune_witnesses(&self) -> bool {
        false
    }

    fn flush_lune_witnesses(&self, witnesses: &mut Vec<PendingLuneWitness>) -> AnnResult<()> {
        witnesses.clear();
        Ok(())
    }
}

pub(crate) struct ReservoirEdgeSink<'a> {
    reservoirs: &'a [Mutex<HashPruneReservoir>],
}

impl<'a> ReservoirEdgeSink<'a> {
    pub(crate) fn new(reservoirs: &'a [Mutex<HashPruneReservoir>]) -> Self {
        Self { reservoirs }
    }
}

impl PendingEdgeSink for ReservoirEdgeSink<'_> {
    fn flush_pending_edges(&self, edges: &mut Vec<PendingEdge>) -> AnnResult<()> {
        flush_pending_edges_to_reservoirs(edges, self.reservoirs);
        Ok(())
    }
}

pub struct LeafScratch {
    pub dmat: Vec<f32>,
    pub block_dmat: Vec<f32>,
    pub x: Array2<f32>,
    pub edges: Vec<PendingEdge>,
    row_topk: Vec<RowTopK>,
}

pub(crate) struct ParallelBlockResult {
    pub(crate) block_start: usize,
    pub(crate) profile: LeafProfile,
    pub(crate) edges: Vec<PendingEdge>,
    pub(crate) lune_witnesses: Vec<PendingLuneWitness>,
}

pub const LEAF_SIZE_BUCKETS: usize = 5;

#[derive(Clone, Copy, Debug, Default)]
pub struct LeafSizeBucketProfile {
    pub leaves: usize,
    pub rows: usize,
    pub exact_leaves: usize,
    pub exact_rows: usize,
    pub exact_wall: Duration,
    pub exact_distance: Duration,
    pub exact_topk: Duration,
    pub adsampling_leaves: usize,
    pub adsampling_rows: usize,
    pub adsampling_wall: Duration,
    pub adsampling_layout: Duration,
    pub adsampling_seed: Duration,
    pub adsampling_scan: Duration,
    pub adsampling_seed_evals: u64,
    pub adsampling_full_evals: u64,
    pub adsampling_pruned_evals: u64,
    pub adsampling_group_evals: u64,
    pub adsampling_simd_group_calls: u64,
    pub adsampling_simd_active_lane_evals: u64,
    pub adsampling_scalar_group_evals: u64,
}

impl LeafSizeBucketProfile {
    fn merge(&mut self, other: Self) {
        self.leaves += other.leaves;
        self.rows += other.rows;
        self.exact_leaves += other.exact_leaves;
        self.exact_rows += other.exact_rows;
        self.exact_wall += other.exact_wall;
        self.exact_distance += other.exact_distance;
        self.exact_topk += other.exact_topk;
        self.adsampling_leaves += other.adsampling_leaves;
        self.adsampling_rows += other.adsampling_rows;
        self.adsampling_wall += other.adsampling_wall;
        self.adsampling_layout += other.adsampling_layout;
        self.adsampling_seed += other.adsampling_seed;
        self.adsampling_scan += other.adsampling_scan;
        self.adsampling_seed_evals += other.adsampling_seed_evals;
        self.adsampling_full_evals += other.adsampling_full_evals;
        self.adsampling_pruned_evals += other.adsampling_pruned_evals;
        self.adsampling_group_evals += other.adsampling_group_evals;
        self.adsampling_simd_group_calls += other.adsampling_simd_group_calls;
        self.adsampling_simd_active_lane_evals += other.adsampling_simd_active_lane_evals;
        self.adsampling_scalar_group_evals += other.adsampling_scalar_group_evals;
    }
}

pub(crate) fn leaf_size_bucket_index(leaf_size: usize) -> Option<usize> {
    match leaf_size {
        256..=511 => Some(0),
        512..=1023 => Some(1),
        1024..=2047 => Some(2),
        2048..=4095 => Some(3),
        4096.. => Some(4),
        _ => None,
    }
}

pub(crate) fn leaf_size_bucket_label(bucket: usize) -> &'static str {
    match bucket {
        0 => "[256,512)",
        1 => "[512,1024)",
        2 => "[1024,2048)",
        3 => "[2048,4096)",
        4 => ">=4096",
        _ => "unknown",
    }
}

#[derive(Clone, Debug, Default)]
pub struct LeafProfile {
    pub total_wall: Duration,
    pub load: Duration,
    pub distance: Duration,
    pub topk: Duration,
    pub hash: Duration,
    pub flush: Duration,
    pub leaves: usize,
    pub points: usize,
    pub blockwise_leaves: usize,
    pub io_stats: PointBatchStats,
    pub sketch_prefetch: Duration,
    pub sketch_rows_prefetched: usize,
    pub lune_witness_emit: Duration,
    pub lune_witness_records: usize,
    pub leaf_adsampling_leaves: usize,
    pub leaf_adsampling_rows: usize,
    pub leaf_adsampling_layout: Duration,
    pub leaf_adsampling_seed: Duration,
    pub leaf_adsampling_scan: Duration,
    pub leaf_adsampling_seed_evals: u64,
    pub leaf_adsampling_full_evals: u64,
    pub leaf_adsampling_pruned_evals: u64,
    pub leaf_adsampling_group_evals: u64,
    pub leaf_adsampling_simd_group_calls: u64,
    pub leaf_adsampling_simd_active_lane_evals: u64,
    pub leaf_adsampling_scalar_group_evals: u64,
    pub leaf_ads_tiling_enabled: bool,
    pub leaf_ads_wavefront_pairmask_enabled: bool,
    pub leaf_ads_work_graph_enabled: bool,
    pub leaf_ads_tiled_leaves: usize,
    pub leaf_ads_tiled_rows: usize,
    pub leaf_ads_tiles: usize,
    pub leaf_ads_tile_rows_total: usize,
    pub leaf_ads_tile_rows_min: usize,
    pub leaf_ads_tile_rows_max: usize,
    pub leaf_ads_tile_wall: Duration,
    pub leaf_ads_tile_wait: Duration,
    pub leaf_ads_handle_requeues: usize,
    pub leaf_ads_cpu_budget: usize,
    pub leaf_ads_active_context_peak: usize,
    pub leaf_ads_active_workers_peak: usize,
    pub leaf_ads_ewma_ns_per_row_by_bucket: [u64; LEAF_ADS_EWMA_BUCKETS],
    pub leaf_size_buckets: [LeafSizeBucketProfile; LEAF_SIZE_BUCKETS],
}

impl LeafProfile {
    pub fn merge(&mut self, other: LeafProfile) {
        self.total_wall += other.total_wall;
        self.load += other.load;
        self.distance += other.distance;
        self.topk += other.topk;
        self.hash += other.hash;
        self.flush += other.flush;
        self.leaves += other.leaves;
        self.points += other.points;
        self.blockwise_leaves += other.blockwise_leaves;
        self.io_stats.point_calls += other.io_stats.point_calls;
        self.io_stats.range_calls += other.io_stats.range_calls;
        self.io_stats.range_rows_read += other.io_stats.range_rows_read;
        self.io_stats.bytes_read += other.io_stats.bytes_read;
        self.sketch_prefetch += other.sketch_prefetch;
        self.sketch_rows_prefetched += other.sketch_rows_prefetched;
        self.lune_witness_emit += other.lune_witness_emit;
        self.lune_witness_records += other.lune_witness_records;
        self.leaf_adsampling_leaves += other.leaf_adsampling_leaves;
        self.leaf_adsampling_rows += other.leaf_adsampling_rows;
        self.leaf_adsampling_layout += other.leaf_adsampling_layout;
        self.leaf_adsampling_seed += other.leaf_adsampling_seed;
        self.leaf_adsampling_scan += other.leaf_adsampling_scan;
        self.leaf_adsampling_seed_evals += other.leaf_adsampling_seed_evals;
        self.leaf_adsampling_full_evals += other.leaf_adsampling_full_evals;
        self.leaf_adsampling_pruned_evals += other.leaf_adsampling_pruned_evals;
        self.leaf_adsampling_group_evals += other.leaf_adsampling_group_evals;
        self.leaf_adsampling_simd_group_calls += other.leaf_adsampling_simd_group_calls;
        self.leaf_adsampling_simd_active_lane_evals += other.leaf_adsampling_simd_active_lane_evals;
        self.leaf_adsampling_scalar_group_evals += other.leaf_adsampling_scalar_group_evals;
        self.leaf_ads_tiling_enabled |= other.leaf_ads_tiling_enabled;
        self.leaf_ads_wavefront_pairmask_enabled |= other.leaf_ads_wavefront_pairmask_enabled;
        self.leaf_ads_work_graph_enabled |= other.leaf_ads_work_graph_enabled;
        self.leaf_ads_tiled_leaves += other.leaf_ads_tiled_leaves;
        self.leaf_ads_tiled_rows += other.leaf_ads_tiled_rows;
        self.leaf_ads_tiles += other.leaf_ads_tiles;
        self.leaf_ads_tile_rows_total += other.leaf_ads_tile_rows_total;
        self.leaf_ads_tile_rows_min =
            match (self.leaf_ads_tile_rows_min, other.leaf_ads_tile_rows_min) {
                (0, value) => value,
                (value, 0) => value,
                (left, right) => left.min(right),
            };
        self.leaf_ads_tile_rows_max = self
            .leaf_ads_tile_rows_max
            .max(other.leaf_ads_tile_rows_max);
        self.leaf_ads_tile_wall += other.leaf_ads_tile_wall;
        self.leaf_ads_tile_wait += other.leaf_ads_tile_wait;
        self.leaf_ads_handle_requeues += other.leaf_ads_handle_requeues;
        self.leaf_ads_cpu_budget = self.leaf_ads_cpu_budget.max(other.leaf_ads_cpu_budget);
        self.leaf_ads_active_context_peak = self
            .leaf_ads_active_context_peak
            .max(other.leaf_ads_active_context_peak);
        self.leaf_ads_active_workers_peak = self
            .leaf_ads_active_workers_peak
            .max(other.leaf_ads_active_workers_peak);
        for (dst, src) in self
            .leaf_ads_ewma_ns_per_row_by_bucket
            .iter_mut()
            .zip(other.leaf_ads_ewma_ns_per_row_by_bucket)
        {
            *dst = (*dst).max(src);
        }
        for (dst, src) in self
            .leaf_size_buckets
            .iter_mut()
            .zip(other.leaf_size_buckets)
        {
            dst.merge(src);
        }
    }

    pub(crate) fn record_leaf_adsampling(
        &mut self,
        leaf_size: usize,
        telemetry: &LeafAdSamplingTelemetry,
    ) {
        self.leaf_ads_tiling_enabled |= telemetry.tiled;
        self.leaf_ads_wavefront_pairmask_enabled |= telemetry.wavefront_pairmask;
        self.leaf_ads_work_graph_enabled |= telemetry.work_graph;
        self.leaf_adsampling_leaves += 1;
        self.leaf_adsampling_rows += leaf_size;
        self.leaf_adsampling_layout += telemetry.layout;
        self.leaf_adsampling_seed += telemetry.seed;
        self.leaf_adsampling_scan += telemetry.scan;
        self.leaf_adsampling_seed_evals += telemetry.seed_evals;
        self.leaf_adsampling_full_evals += telemetry.full_evals;
        self.leaf_adsampling_pruned_evals += telemetry.pruned_evals;
        self.leaf_adsampling_group_evals += telemetry.group_evals;
        self.leaf_adsampling_simd_group_calls += telemetry.simd_group_calls;
        self.leaf_adsampling_simd_active_lane_evals += telemetry.simd_active_lane_evals;
        self.leaf_adsampling_scalar_group_evals += telemetry.scalar_group_evals;
        if telemetry.tiled {
            self.leaf_ads_tiled_leaves += 1;
            self.leaf_ads_tiled_rows += leaf_size;
        }
        self.leaf_ads_tiles += telemetry.tile_count;
        self.leaf_ads_tile_rows_total += telemetry.tile_rows_total;
        self.leaf_ads_tile_rows_min = match (self.leaf_ads_tile_rows_min, telemetry.tile_rows_min) {
            (0, value) => value,
            (value, 0) => value,
            (left, right) => left.min(right),
        };
        self.leaf_ads_tile_rows_max = self.leaf_ads_tile_rows_max.max(telemetry.tile_rows_max);
        self.leaf_ads_tile_wall += telemetry.tile_wall;
        self.leaf_ads_tile_wait += telemetry.tile_wait;
        self.leaf_ads_handle_requeues += telemetry.tile_handle_requeues;
        self.leaf_ads_cpu_budget = self.leaf_ads_cpu_budget.max(telemetry.cpu_budget);
        self.leaf_ads_active_context_peak = self
            .leaf_ads_active_context_peak
            .max(telemetry.active_context_peak);
        self.leaf_ads_active_workers_peak = self
            .leaf_ads_active_workers_peak
            .max(telemetry.active_workers_peak);
        for (dst, src) in self
            .leaf_ads_ewma_ns_per_row_by_bucket
            .iter_mut()
            .zip(telemetry.ewma_ns_per_row_by_bucket)
        {
            *dst = (*dst).max(src);
        }
    }

    pub(crate) fn record_leaf_size_bucket(
        &mut self,
        leaf_size: usize,
        adsampling: Option<&LeafAdSamplingTelemetry>,
    ) {
        let Some(bucket_idx) = leaf_size_bucket_index(leaf_size) else {
            return;
        };
        let bucket = &mut self.leaf_size_buckets[bucket_idx];
        bucket.leaves += 1;
        bucket.rows += leaf_size;
        if let Some(telemetry) = adsampling {
            bucket.adsampling_leaves += 1;
            bucket.adsampling_rows += leaf_size;
            bucket.adsampling_wall += self.total_wall;
            bucket.adsampling_layout += telemetry.layout;
            bucket.adsampling_seed += telemetry.seed;
            bucket.adsampling_scan += telemetry.scan;
            bucket.adsampling_seed_evals += telemetry.seed_evals;
            bucket.adsampling_full_evals += telemetry.full_evals;
            bucket.adsampling_pruned_evals += telemetry.pruned_evals;
            bucket.adsampling_group_evals += telemetry.group_evals;
            bucket.adsampling_simd_group_calls += telemetry.simd_group_calls;
            bucket.adsampling_simd_active_lane_evals += telemetry.simd_active_lane_evals;
            bucket.adsampling_scalar_group_evals += telemetry.scalar_group_evals;
        } else {
            bucket.exact_leaves += 1;
            bucket.exact_rows += leaf_size;
            bucket.exact_wall += self.total_wall;
            bucket.exact_distance += self.distance;
            bucket.exact_topk += self.topk;
        }
    }
}

fn default_leaf_point_read_options(dataset: &dyn PointStore, rows: usize) -> WindowedGatherOptions {
    let row_bytes = dataset.dim().saturating_mul(size_of::<f32>());
    WindowedGatherOptions {
        max_gap_rows: if row_bytes <= 4096 { 1 } else { 0 },
        max_window_bytes: row_bytes
            .saturating_mul(rows.max(1))
            .min(4 * 1024 * 1024)
            .max(256 * 1024),
        alignment_bytes: 4096,
        sort_ids: true,
    }
}

fn read_leaf_points_into_matrix(
    dataset: &dyn PointStore,
    leaf: &[u32],
    mut x_view: ArrayViewMut2<'_, f32>,
    read_options: &WindowedGatherOptions,
    stats: &mut PointBatchStats,
) -> AnnResult<()> {
    if let Some(dst) = x_view.as_slice_memory_order_mut() {
        dataset.read_points_windowed_into_batch_stats(leaf, dst, read_options, stats)?;
        return Ok(());
    }

    let dim = dataset.dim();
    let mut flat = vec![0.0f32; leaf.len().saturating_mul(dim)];
    dataset.read_points_windowed_into_batch_stats(leaf, &mut flat, read_options, stats)?;
    for (i_local, chunk) in flat.chunks(dim).enumerate() {
        x_view
            .row_mut(i_local)
            .assign(&ndarray::ArrayView1::from(chunk));
    }
    Ok(())
}

#[inline]
pub(crate) fn should_use_leaf_adsampling_executor(
    params: &ForgeANNParams,
    metric: Metric,
    leaf_size: usize,
) -> bool {
    metric == Metric::L2
        && leaf_size >= ForgeANNParams::LEAF_ADSAMPLING_MIN_SIZE
        && params.adsampling_group_dims > 0
        && leaf_size > 1
}

fn compute_leaf_adsampling_for_params(
    vectors: &[f32],
    leaf_size: usize,
    dim: usize,
    k: usize,
    params: &ForgeANNParams,
    runtime: Option<&LeafAdsOperatorRuntime>,
) -> AnnResult<super::adsampling::LeafAdSamplingResult> {
    if effective_leaf_ads_wavefront_pairmask(params) {
        return compute_leaf_adsampling_topk_l2_wavefront_pairmask(
            vectors,
            leaf_size,
            dim,
            k,
            AdSamplingConfig::leaf_from_params(params, k),
        );
    }
    compute_leaf_adsampling_topk_l2_with_runtime(
        vectors,
        leaf_size,
        dim,
        k,
        AdSamplingConfig::leaf_from_params(params, k),
        runtime,
    )
}

#[inline]
fn effective_leaf_ads_wavefront_pairmask(params: &ForgeANNParams) -> bool {
    params.leaf_ads_wavefront_pairmask_enable
        && !params.leaf_ads_tiling_enable
        && !params.leaf_ads_work_graph_enable
}

fn emit_row_topk_edges(
    row_topk: &[Vec<(usize, f32)>],
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    edge_sink: &dyn PendingEdgeSink,
    profile: &mut LeafProfile,
) -> AnnResult<()> {
    let sketch_cache = {
        let pf_start = Instant::now();
        let cache = LeafSketchCache::prefetch(sketches, leaf)?;
        profile.sketch_prefetch += pf_start.elapsed();
        profile.sketch_rows_prefetched += leaf.len();
        cache
    };
    let hash_start = Instant::now();
    let mut edges = Vec::with_capacity(
        leaf.len()
            .saturating_mul(params.leaf_knn.max(1))
            .saturating_mul(2),
    );
    for (i_local, &i_global) in leaf.iter().enumerate() {
        let sketch_i = sketch_cache.row(i_local);
        for (rank, &(j_local, d)) in row_topk[i_local].iter().enumerate() {
            let j_global = leaf[j_local] as usize;
            let sketch_j = sketch_cache.row(j_local);
            let hash_ij = compute_hash(sketch_i, sketch_j, params.m_hash_bits);
            let hash_ji = compute_hash(sketch_j, sketch_i, params.m_hash_bits);
            let local_rank = rank.saturating_add(1).min(u8::MAX as usize) as u8;

            edges.push(PendingEdge {
                p: i_global as usize,
                c: j_global as u32,
                hash: hash_ij,
                dist: d,
                mandatory: false,
                local_rank,
                flags: PENDING_EDGE_DIRECT,
            });
            edges.push(PendingEdge {
                p: j_global,
                c: i_global,
                hash: hash_ji,
                dist: d,
                mandatory: false,
                local_rank,
                flags: PENDING_EDGE_MIRROR,
            });

            if edges.len() >= 16384 {
                let flush_start = Instant::now();
                flush_pending_edges(&mut edges, edge_sink)?;
                profile.flush += flush_start.elapsed();
            }
        }
    }
    profile.hash += hash_start.elapsed();

    let flush_start = Instant::now();
    flush_pending_edges(&mut edges, edge_sink)?;
    profile.flush += flush_start.elapsed();
    Ok(())
}

const VIEW_LUNE_ALPHA_IMPL: f32 = 1.2;

#[inline]
fn quantize_lune_margin_q16(threshold: f32, witness_dist2: f32) -> u16 {
    if !threshold.is_finite() || threshold <= 0.0 || witness_dist2 >= threshold {
        return 0;
    }
    let normalized = ((threshold - witness_dist2) / threshold).clamp(0.0, 1.0);
    (normalized * u16::MAX as f32).round() as u16
}

#[inline]
fn l2_dist2_from_flat(vectors: &[f32], dim: usize, left: usize, right: usize) -> f32 {
    let left = &vectors[left * dim..(left + 1) * dim];
    let right = &vectors[right * dim..(right + 1) * dim];
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = *a - *b;
            delta * delta
        })
        .sum()
}

fn emit_lune_witnesses_from_vec_topk_l2(
    row_topk: &[Vec<(usize, f32)>],
    vectors: &[f32],
    dim: usize,
    leaf: &[u32],
    edge_sink: &dyn PendingEdgeSink,
    profile: &mut LeafProfile,
) -> AnnResult<()> {
    if !edge_sink.wants_lune_witnesses() {
        return Ok(());
    }
    let emit_start = Instant::now();
    let mut witnesses = Vec::new();
    for (src_local, src_topk) in row_topk.iter().enumerate() {
        for victim_pos in 1..src_topk.len() {
            let (victim_local, source_victim_dist2) = src_topk[victim_pos];
            let threshold = source_victim_dist2 / VIEW_LUNE_ALPHA_IMPL;
            for pivot_pos in 0..victim_pos {
                let (pivot_local, _) = src_topk[pivot_pos];
                let pivot_victim_dist2 =
                    l2_dist2_from_flat(vectors, dim, pivot_local, victim_local);
                if pivot_victim_dist2 < threshold {
                    witnesses.push(PendingLuneWitness {
                        src: leaf[src_local],
                        pivot: leaf[pivot_local],
                        victim: leaf[victim_local],
                        margin_q16: quantize_lune_margin_q16(threshold, pivot_victim_dist2),
                        pivot_rank: pivot_pos.saturating_add(1).min(u8::MAX as usize) as u8,
                        victim_rank: victim_pos.saturating_add(1).min(u8::MAX as usize) as u8,
                    });
                }
            }
        }
        if witnesses.len() >= 16384 {
            profile.lune_witness_records += witnesses.len();
            edge_sink.flush_lune_witnesses(&mut witnesses)?;
        }
    }
    profile.lune_witness_records += witnesses.len();
    edge_sink.flush_lune_witnesses(&mut witnesses)?;
    profile.lune_witness_emit += emit_start.elapsed();
    Ok(())
}

fn emit_lune_witnesses_from_row_topk_matrix(
    row_topk: &[RowTopK],
    dmat: &[f32],
    leaf_size: usize,
    leaf: &[u32],
    edge_sink: &dyn PendingEdgeSink,
    profile: &mut LeafProfile,
) -> AnnResult<()> {
    if !edge_sink.wants_lune_witnesses() {
        return Ok(());
    }
    let emit_start = Instant::now();
    let mut witnesses = Vec::new();
    for (src_local, src_topk) in row_topk.iter().take(leaf_size).enumerate() {
        let topk = src_topk.as_slice();
        for victim_pos in 1..topk.len() {
            let (victim_local, source_victim_dist2) = topk[victim_pos];
            let threshold = source_victim_dist2 / VIEW_LUNE_ALPHA_IMPL;
            for pivot_pos in 0..victim_pos {
                let (pivot_local, _) = topk[pivot_pos];
                let pivot_victim_dist2 = dmat[pivot_local * leaf_size + victim_local];
                if pivot_victim_dist2 < threshold {
                    witnesses.push(PendingLuneWitness {
                        src: leaf[src_local],
                        pivot: leaf[pivot_local],
                        victim: leaf[victim_local],
                        margin_q16: quantize_lune_margin_q16(threshold, pivot_victim_dist2),
                        pivot_rank: pivot_pos.saturating_add(1).min(u8::MAX as usize) as u8,
                        victim_rank: victim_pos.saturating_add(1).min(u8::MAX as usize) as u8,
                    });
                }
            }
        }
        if witnesses.len() >= 16384 {
            profile.lune_witness_records += witnesses.len();
            edge_sink.flush_lune_witnesses(&mut witnesses)?;
        }
    }
    profile.lune_witness_records += witnesses.len();
    edge_sink.flush_lune_witnesses(&mut witnesses)?;
    profile.lune_witness_emit += emit_start.elapsed();
    Ok(())
}

fn emit_lune_witnesses_from_row_topk_vectors(
    row_topk: &[RowTopK],
    vectors: &[f32],
    dim: usize,
    leaf: &[u32],
    edge_sink: &dyn PendingEdgeSink,
    profile: &mut LeafProfile,
) -> AnnResult<()> {
    if !edge_sink.wants_lune_witnesses() {
        return Ok(());
    }
    let emit_start = Instant::now();
    let mut witnesses = Vec::new();
    for (src_local, src_topk) in row_topk.iter().take(leaf.len()).enumerate() {
        let topk = src_topk.as_slice();
        for victim_pos in 1..topk.len() {
            let (victim_local, source_victim_dist2) = topk[victim_pos];
            let threshold = source_victim_dist2 / VIEW_LUNE_ALPHA_IMPL;
            for pivot_pos in 0..victim_pos {
                let (pivot_local, _) = topk[pivot_pos];
                let pivot_victim_dist2 =
                    l2_dist2_from_flat(vectors, dim, pivot_local, victim_local);
                if pivot_victim_dist2 < threshold {
                    witnesses.push(PendingLuneWitness {
                        src: leaf[src_local],
                        pivot: leaf[pivot_local],
                        victim: leaf[victim_local],
                        margin_q16: quantize_lune_margin_q16(threshold, pivot_victim_dist2),
                        pivot_rank: pivot_pos.saturating_add(1).min(u8::MAX as usize) as u8,
                        victim_rank: victim_pos.saturating_add(1).min(u8::MAX as usize) as u8,
                    });
                }
            }
        }
        if witnesses.len() >= 16384 {
            profile.lune_witness_records += witnesses.len();
            edge_sink.flush_lune_witnesses(&mut witnesses)?;
        }
    }
    profile.lune_witness_records += witnesses.len();
    edge_sink.flush_lune_witnesses(&mut witnesses)?;
    profile.lune_witness_emit += emit_start.elapsed();
    Ok(())
}

impl LeafScratch {
    pub fn new(c_max: usize, dim: usize, leaf_knn: usize) -> Self {
        let edge_capacity = c_max.saturating_mul(leaf_knn.max(1)).saturating_mul(2);
        Self {
            dmat: Vec::with_capacity(c_max * c_max),
            block_dmat: Vec::new(),
            x: Array2::zeros((c_max, dim)),
            edges: Vec::with_capacity(edge_capacity),
            row_topk: Vec::new(),
        }
    }

    /// 将暂存的边写入全局 reservoirs。
    /// 为了减少锁竞争，我们可以对边进行排序，使得对同一个 reservoir 的操作集中在一起，
    /// 或者仅仅是利用批量写入来平摊锁开销。
    pub(crate) fn flush_edges(&mut self, edge_sink: &dyn PendingEdgeSink) -> AnnResult<()> {
        flush_pending_edges(&mut self.edges, edge_sink)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RowTopK {
    buf: [(usize, f32); 32],
    len: usize,
}

impl Default for RowTopK {
    fn default() -> Self {
        Self {
            buf: [(0, f32::INFINITY); 32],
            len: 0,
        }
    }
}

impl RowTopK {
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    pub(crate) fn push(&mut self, k: usize, idx: usize, dist: f32) {
        debug_assert!(k <= 32, "RowTopK: k={k} exceeds fixed buffer size 32");
        if k == 0 {
            return;
        }

        if self.len < k {
            // Buffer not full yet: insertion sort into buf[0..len+1]
            let mut i = self.len;
            while i > 0 && self.buf[i - 1].1 > dist {
                self.buf[i] = self.buf[i - 1];
                i -= 1;
            }
            self.buf[i] = (idx, dist);
            self.len += 1;
        } else {
            // Buffer full: only insert if closer than the farthest (buf[k-1])
            if dist >= self.buf[k - 1].1 {
                return;
            }
            // Insertion sort: shift right from the end, stop when we find the right spot
            let mut i = k - 1;
            while i > 0 && self.buf[i - 1].1 > dist {
                self.buf[i] = self.buf[i - 1];
                i -= 1;
            }
            self.buf[i] = (idx, dist);
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &(usize, f32)> {
        self.buf[..self.len].iter()
    }

    pub(crate) fn as_slice(&self) -> &[(usize, f32)] {
        &self.buf[..self.len]
    }
}

/// 对单个叶子执行：
///  - 计算叶内所有点对距离（O(|b|^2)）
///  - 为每个点选取 leaf_knn 个最近邻
///  - 对每一条候选边 (p, c) 分别插入 p 和 c 的 HashPrune 水库
///
/// 当度量为 L2 时，使用 ndarray+BLAS 的 GEMM 计算点积矩阵 X·X^T，
/// 再通过行范数组装距离平方矩阵 dist2 = ||x||^2 + ||y||^2 - 2·(X·X^T)；
/// 非 L2 度量则回退到逐对距离计算，以兼容 Cosine/IP 等扩展。
pub fn process_leaf(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    reservoirs: &[Mutex<HashPruneReservoir>],
    scratch: &mut LeafScratch,
) -> AnnResult<()> {
    process_leaf_profiled(dataset, metric, sketches, leaf, params, reservoirs, scratch).map(|_| ())
}

pub fn process_leaf_profiled(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    reservoirs: &[Mutex<HashPruneReservoir>],
    scratch: &mut LeafScratch,
) -> AnnResult<LeafProfile> {
    let edge_sink = ReservoirEdgeSink::new(reservoirs);
    process_leaf_profiled_with_sink(dataset, metric, sketches, leaf, params, &edge_sink, scratch)
}

pub(crate) fn process_leaf_profiled_with_sink(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    edge_sink: &dyn PendingEdgeSink,
    scratch: &mut LeafScratch,
) -> AnnResult<LeafProfile> {
    process_leaf_profiled_with_sink_with_ads_runtime(
        dataset, metric, sketches, leaf, params, edge_sink, scratch, None,
    )
}

pub(crate) fn process_leaf_profiled_with_sink_with_ads_runtime(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    edge_sink: &dyn PendingEdgeSink,
    scratch: &mut LeafScratch,
    ads_runtime: Option<&LeafAdsOperatorRuntime>,
) -> AnnResult<LeafProfile> {
    let total_start = Instant::now();
    let mut profile = LeafProfile {
        leaves: usize::from(!leaf.is_empty()),
        points: leaf.len(),
        leaf_ads_tiling_enabled: params.leaf_ads_tiling_enable,
        leaf_ads_wavefront_pairmask_enabled: effective_leaf_ads_wavefront_pairmask(params),
        leaf_ads_work_graph_enabled: params.leaf_ads_work_graph_enable,
        ..LeafProfile::default()
    };
    let leaf_size = leaf.len();
    if leaf_size <= 1 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }

    let max_leaf_size = params.kernel_safe_leaf_size();
    let k = params.leaf_knn.min(leaf_size.saturating_sub(1));
    if k == 0 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }

    if leaf_size > max_leaf_size {
        profile.blockwise_leaves = 1;
        process_leaf_blockwise(
            dataset,
            metric,
            sketches,
            leaf,
            params,
            edge_sink,
            scratch,
            ads_runtime,
            &mut profile,
        )?;
        profile.total_wall = total_start.elapsed();
        if profile.leaf_adsampling_leaves > 0 {
            let telemetry = LeafAdSamplingTelemetry {
                layout: profile.leaf_adsampling_layout,
                seed: profile.leaf_adsampling_seed,
                scan: profile.leaf_adsampling_scan,
                seed_evals: profile.leaf_adsampling_seed_evals,
                full_evals: profile.leaf_adsampling_full_evals,
                pruned_evals: profile.leaf_adsampling_pruned_evals,
                group_evals: profile.leaf_adsampling_group_evals,
                simd_group_calls: profile.leaf_adsampling_simd_group_calls,
                simd_active_lane_evals: profile.leaf_adsampling_simd_active_lane_evals,
                scalar_group_evals: profile.leaf_adsampling_scalar_group_evals,
                wavefront_pairmask: profile.leaf_ads_wavefront_pairmask_enabled,
                ..LeafAdSamplingTelemetry::default()
            };
            profile.record_leaf_size_bucket(leaf_size, Some(&telemetry));
        } else {
            profile.record_leaf_size_bucket(leaf_size, None);
        }
        return Ok(profile);
    }

    // 重置并调整 scratch 大小
    scratch.dmat.clear();
    scratch.dmat.resize(leaf_size * leaf_size, 0.0f32);

    if metric == Metric::L2 {
        let dim = dataset.dim();
        // 确保 scratch.x 足够大，如果不匹配则重新分配
        if scratch.x.nrows() < leaf_size || scratch.x.ncols() < dim {
            scratch.x = Array2::<f32>::zeros((leaf_size.max(scratch.x.nrows()), dim));
        }

        // 局部视图，避免分配
        let mut x_view = scratch.x.slice_mut(ndarray::s![0..leaf_size, 0..dim]);

        let load_start = Instant::now();
        let read_options = default_leaf_point_read_options(dataset, leaf_size);
        read_leaf_points_into_matrix(
            dataset,
            leaf,
            x_view.view_mut(),
            &read_options,
            &mut profile.io_stats,
        )?;
        profile.load += load_start.elapsed();

        if should_use_leaf_adsampling_executor(params, metric, leaf_size) {
            let vectors = x_view
                .as_slice_memory_order()
                .expect("leaf matrix is contiguous");
            let result = compute_leaf_adsampling_for_params(
                vectors,
                leaf_size,
                dim,
                k,
                params,
                ads_runtime,
            )?;
            profile.distance += result.telemetry.compute_time();
            profile.record_leaf_adsampling(leaf_size, &result.telemetry);
            emit_row_topk_edges(
                &result.row_topk,
                sketches,
                leaf,
                params,
                edge_sink,
                &mut profile,
            )?;
            emit_lune_witnesses_from_vec_topk_l2(
                &result.row_topk,
                vectors,
                dim,
                leaf,
                edge_sink,
                &mut profile,
            )?;
            profile.total_wall = total_start.elapsed();
            profile.record_leaf_size_bucket(leaf_size, Some(&result.telemetry));
            return Ok(profile);
        }

        // 行范数 ||x_i||^2
        let distance_start = Instant::now();
        let norms: Vec<f32> = x_view.map_axis(Axis(1), |row| row.dot(&row)).to_vec();

        // 使用 dmat 作为 Gram 矩阵的存储空间
        let mut dmat_view =
            ArrayViewMut2::from_shape((leaf_size, leaf_size), &mut scratch.dmat).unwrap();
        ndarray::linalg::general_mat_mul(1.0, &x_view, &x_view.t(), 0.0, &mut dmat_view);

        // 现在 scratch.dmat 中存储的是 gram[i, j] = x_i · x_j
        // 转换为 dist2 = norms[i] + norms[j] - 2.0 * gram[i, j]
        for i in 0..leaf_size {
            let offset = i * leaf_size;
            let norm_i = norms[i];
            for j in 0..leaf_size {
                let gram_ij = scratch.dmat[offset + j];
                let mut dist2 = norm_i + norms[j] - 2.0 * gram_ij;
                if dist2 < 0.0 {
                    dist2 = 0.0;
                }
                scratch.dmat[offset + j] = dist2;
            }
        }
        profile.distance += distance_start.elapsed();
    } else {
        let distance_start = Instant::now();
        for (i_local, &i_global) in leaf.iter().enumerate() {
            for (j_local, &j_global) in leaf.iter().enumerate().skip(i_local + 1) {
                let d = dataset.get_distance(i_global, j_global, metric)?;
                scratch.dmat[i_local * leaf_size + j_local] = d;
                scratch.dmat[j_local * leaf_size + i_local] = d;
            }
        }
        profile.distance += distance_start.elapsed();
    }

    // 对每个局部点做 kNN 并存入 scratch.edges
    if scratch.row_topk.len() < leaf_size {
        scratch.row_topk.resize_with(leaf_size, RowTopK::default);
    }

    let topk_start = Instant::now();
    for (i_local, &_i_global) in leaf.iter().enumerate() {
        let row = &scratch.dmat[i_local * leaf_size..(i_local + 1) * leaf_size];
        let topk = &mut scratch.row_topk[i_local];
        topk.clear();
        for (j_local, &d) in row.iter().enumerate() {
            if j_local != i_local {
                topk.push(k, j_local, d);
            }
        }
    }
    profile.topk += topk_start.elapsed();

    let hash_start = Instant::now();
    let sketch_cache = {
        let pf_start = Instant::now();
        let cache = LeafSketchCache::prefetch(sketches, leaf)?;
        profile.sketch_prefetch += pf_start.elapsed();
        profile.sketch_rows_prefetched += leaf.len();
        cache
    };
    for (i_local, &i_global) in leaf.iter().enumerate() {
        let topk = &scratch.row_topk[i_local];
        let i_idx = i_global as usize;
        let sketch_i = sketch_cache.row(i_local);

        for (rank, &(j_local, d)) in topk.iter().enumerate() {
            let j_global = leaf[j_local] as usize;
            let sketch_j = sketch_cache.row(j_local);

            let hash_ij = compute_hash(sketch_i, sketch_j, params.m_hash_bits);
            let hash_ji = compute_hash(sketch_j, sketch_i, params.m_hash_bits);
            let local_rank = rank.saturating_add(1).min(u8::MAX as usize) as u8;

            scratch.edges.push(PendingEdge {
                p: i_idx,
                c: j_global as u32,
                hash: hash_ij,
                dist: d,
                mandatory: false,
                local_rank,
                flags: PENDING_EDGE_DIRECT,
            });
            scratch.edges.push(PendingEdge {
                p: j_global,
                c: i_global,
                hash: hash_ji,
                dist: d,
                mandatory: false,
                local_rank,
                flags: PENDING_EDGE_MIRROR,
            });
        }

        if scratch.edges.len() >= 16384 {
            let flush_start = Instant::now();
            scratch.flush_edges(edge_sink)?;
            profile.flush += flush_start.elapsed();
        }
    }
    profile.hash += hash_start.elapsed();

    // 确保处理完当前叶子后，所有剩余的边都写入 reservoirs
    let flush_start = Instant::now();
    scratch.flush_edges(edge_sink)?;
    profile.flush += flush_start.elapsed();

    if metric == Metric::L2 {
        emit_lune_witnesses_from_row_topk_matrix(
            &scratch.row_topk,
            &scratch.dmat,
            leaf_size,
            leaf,
            edge_sink,
            &mut profile,
        )?;
    }

    profile.total_wall = total_start.elapsed();
    profile.record_leaf_size_bucket(leaf_size, None);
    Ok(profile)
}

pub fn process_leaf_parallel_large_profiled(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    reservoirs: &[Mutex<HashPruneReservoir>],
    block_rows: usize,
) -> AnnResult<LeafProfile> {
    process_leaf_parallel_large_profiled_with_sink(
        dataset,
        metric,
        sketches,
        leaf,
        params,
        &ReservoirEdgeSink::new(reservoirs),
        block_rows,
    )
}

pub(crate) fn process_leaf_parallel_large_profiled_with_sink(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    edge_sink: &dyn PendingEdgeSink,
    block_rows: usize,
) -> AnnResult<LeafProfile> {
    process_leaf_parallel_large_profiled_with_sink_with_ads_runtime(
        dataset, metric, sketches, leaf, params, edge_sink, block_rows, None,
    )
}

pub(crate) fn process_leaf_parallel_large_profiled_with_sink_with_ads_runtime(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    edge_sink: &dyn PendingEdgeSink,
    block_rows: usize,
    ads_runtime: Option<&LeafAdsOperatorRuntime>,
) -> AnnResult<LeafProfile> {
    if metric == Metric::L2 {
        return process_leaf_parallel_large_profiled_cpu(
            dataset,
            sketches,
            leaf,
            params,
            edge_sink,
            block_rows,
            ads_runtime,
        );
    }

    process_leaf_parallel_large_profiled_impl(
        dataset,
        metric,
        sketches,
        leaf,
        params,
        edge_sink,
        block_rows,
        ads_runtime,
    )
}

pub(super) fn process_leaf_parallel_large_profiled_cpu(
    dataset: &dyn PointStore,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    edge_sink: &dyn PendingEdgeSink,
    block_rows: usize,
    ads_runtime: Option<&LeafAdsOperatorRuntime>,
) -> AnnResult<LeafProfile> {
    process_leaf_parallel_large_profiled_impl(
        dataset,
        Metric::L2,
        sketches,
        leaf,
        params,
        edge_sink,
        block_rows,
        ads_runtime,
    )
}

fn process_leaf_parallel_large_profiled_impl(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    edge_sink: &dyn PendingEdgeSink,
    block_rows: usize,
    ads_runtime: Option<&LeafAdsOperatorRuntime>,
) -> AnnResult<LeafProfile> {
    let total_start = Instant::now();
    let leaf_size = leaf.len();
    let mut profile = LeafProfile {
        leaves: usize::from(!leaf.is_empty()),
        points: leaf_size,
        blockwise_leaves: usize::from(!leaf.is_empty()),
        leaf_ads_tiling_enabled: params.leaf_ads_tiling_enable,
        leaf_ads_wavefront_pairmask_enabled: effective_leaf_ads_wavefront_pairmask(params),
        leaf_ads_work_graph_enabled: params.leaf_ads_work_graph_enable,
        ..LeafProfile::default()
    };

    if leaf_size <= 1 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }

    let k = params.leaf_knn.min(leaf_size.saturating_sub(1));
    if k == 0 {
        profile.total_wall = total_start.elapsed();
        return Ok(profile);
    }

    let dim = dataset.dim();
    let block_rows = block_rows.max(1);
    let mut x = Array2::<f32>::zeros((leaf_size, dim));

    let load_start = Instant::now();
    {
        let mut x_view = x.slice_mut(ndarray::s![0..leaf_size, 0..dim]);
        let read_options = default_leaf_point_read_options(dataset, leaf_size);
        read_leaf_points_into_matrix(
            dataset,
            leaf,
            x_view.view_mut(),
            &read_options,
            &mut profile.io_stats,
        )?;
    }
    profile.load += load_start.elapsed();

    let x_view = x.view();
    let norms: Vec<f32> = if metric == Metric::L2 {
        x_view.map_axis(Axis(1), |row| row.dot(&row)).to_vec()
    } else {
        Vec::new()
    };

    if should_use_leaf_adsampling_executor(params, metric, leaf_size) {
        let vectors = x
            .as_slice_memory_order()
            .expect("leaf matrix is contiguous");
        let result =
            compute_leaf_adsampling_for_params(vectors, leaf_size, dim, k, params, ads_runtime)?;
        profile.distance += result.telemetry.compute_time();
        profile.record_leaf_adsampling(leaf_size, &result.telemetry);
        emit_row_topk_edges(
            &result.row_topk,
            sketches,
            leaf,
            params,
            edge_sink,
            &mut profile,
        )?;
        emit_lune_witnesses_from_vec_topk_l2(
            &result.row_topk,
            vectors,
            dim,
            leaf,
            edge_sink,
            &mut profile,
        )?;
        profile.total_wall = total_start.elapsed();
        profile.record_leaf_size_bucket(leaf_size, Some(&result.telemetry));
        return Ok(profile);
    }

    let block_starts: Vec<usize> = (0..leaf_size).step_by(block_rows).collect();
    let wave_width = choose_parallel_block_wave_width(
        leaf_size,
        block_rows,
        params.leaf_knn,
        block_starts.len(),
    );

    let sketch_cache = {
        let pf_start = Instant::now();
        let cache = LeafSketchCache::prefetch(sketches, leaf)?;
        profile.sketch_prefetch += pf_start.elapsed();
        profile.sketch_rows_prefetched += leaf.len();
        cache
    };

    for wave in block_starts.chunks(wave_width) {
        let mut results: Vec<_> = wave
            .par_iter()
            .map(|&block_start| {
                process_leaf_parallel_large_block(
                    dataset,
                    metric,
                    &sketch_cache,
                    leaf,
                    params,
                    &x_view,
                    &norms,
                    block_start,
                    block_rows,
                    k,
                    edge_sink.wants_lune_witnesses(),
                )
            })
            .collect::<AnnResult<Vec<_>>>()?;
        results.sort_unstable_by_key(|result| result.block_start);

        for mut result in results {
            let flush_start = Instant::now();
            flush_pending_edges(&mut result.edges, edge_sink)?;
            edge_sink.flush_lune_witnesses(&mut result.lune_witnesses)?;
            result.profile.flush += flush_start.elapsed();
            profile.merge(result.profile);
        }
    }

    profile.total_wall = total_start.elapsed();
    profile.record_leaf_size_bucket(leaf_size, None);
    Ok(profile)
}

const BLOCK_SIZE: usize = 1_024;

pub(crate) fn choose_parallel_block_wave_width(
    leaf_size: usize,
    block_rows: usize,
    leaf_knn: usize,
    total_blocks: usize,
) -> usize {
    const TARGET_BLOCK_SCRATCH_BYTES: usize = 256 * 1024 * 1024;

    let bytes_per_block = block_rows
        .saturating_mul(leaf_size)
        .saturating_mul(size_of::<f32>())
        .saturating_add(
            block_rows
                .saturating_mul(leaf_knn.max(1))
                .saturating_mul(2)
                .saturating_mul(size_of::<PendingEdge>()),
        );
    let by_memory = if bytes_per_block == 0 {
        rayon::current_num_threads().max(1)
    } else {
        cmp::max(1, TARGET_BLOCK_SCRATCH_BYTES / bytes_per_block)
    };

    cmp::max(
        1,
        total_blocks.min(rayon::current_num_threads().max(1).min(by_memory)),
    )
}

pub(crate) fn process_leaf_parallel_large_block(
    dataset: &dyn PointStore,
    metric: Metric,
    sketch_cache: &LeafSketchCache,
    leaf: &[u32],
    params: &ForgeANNParams,
    x_view: &ArrayView2<'_, f32>,
    norms: &[f32],
    block_start: usize,
    block_rows: usize,
    k: usize,
    emit_lune_witnesses: bool,
) -> AnnResult<ParallelBlockResult> {
    let leaf_size = leaf.len();
    let block_end = (block_start + block_rows).min(leaf_size);
    let block_size = block_end - block_start;
    let dim = dataset.dim();
    let mut profile = LeafProfile::default();
    let mut row_topk = vec![RowTopK::default(); block_size];

    let distance_start = Instant::now();
    if metric == Metric::L2 && emit_lune_witnesses {
        let mut block_dmat = vec![0.0f32; block_size * leaf_size];
        let x_block = x_view.slice(ndarray::s![block_start..block_end, 0..dim]);
        let mut dmat_view =
            ArrayViewMut2::from_shape((block_size, leaf_size), &mut block_dmat).unwrap();
        ndarray::linalg::general_mat_mul(1.0, &x_block, &x_view.t(), 0.0, &mut dmat_view);

        for i_block in 0..block_size {
            let i_local = block_start + i_block;
            let norm_i = norms[i_local];
            let topk = &mut row_topk[i_block];
            let row = &block_dmat[i_block * leaf_size..(i_block + 1) * leaf_size];

            for (j_local, &gram_ij) in row.iter().enumerate() {
                if i_local == j_local {
                    continue;
                }

                let mut dist2 = norm_i + norms[j_local] - 2.0 * gram_ij;
                if dist2 < 0.0 {
                    dist2 = 0.0;
                }
                topk.push(k, j_local, dist2);
            }
        }
    } else {
        for i_local in block_start..block_end {
            let i_global = leaf[i_local];
            let topk = &mut row_topk[i_local - block_start];
            for (j_local, &j_global) in leaf.iter().enumerate() {
                if i_local == j_local {
                    continue;
                }
                let d = dataset.get_distance(i_global, j_global, metric)?;
                topk.push(k, j_local, d);
            }
        }
    }
    profile.distance += distance_start.elapsed();

    let hash_start = Instant::now();
    let mut edges = Vec::with_capacity(block_size.saturating_mul(k.max(1)).saturating_mul(2));
    for i_local in block_start..block_end {
        let i_idx = leaf[i_local] as usize;
        let sketch_i = sketch_cache.row(i_local);

        for (rank, &(j_local, d)) in row_topk[i_local - block_start].iter().enumerate() {
            let j_global = leaf[j_local] as usize;
            let sketch_j = sketch_cache.row(j_local);

            let hash_ij = compute_hash(sketch_i, sketch_j, params.m_hash_bits);
            let hash_ji = compute_hash(sketch_j, sketch_i, params.m_hash_bits);
            let local_rank = rank.saturating_add(1).min(u8::MAX as usize) as u8;

            edges.push(PendingEdge {
                p: i_idx,
                c: j_global as u32,
                hash: hash_ij,
                dist: d,
                mandatory: false,
                local_rank,
                flags: PENDING_EDGE_DIRECT,
            });
            edges.push(PendingEdge {
                p: j_global,
                c: leaf[i_local],
                hash: hash_ji,
                dist: d,
                mandatory: false,
                local_rank,
                flags: PENDING_EDGE_MIRROR,
            });
        }
    }
    profile.hash += hash_start.elapsed();

    let vectors = x_view
        .as_slice_memory_order()
        .expect("leaf matrix is contiguous");
    let mut lune_witnesses = Vec::new();
    if metric == Metric::L2 {
        let emit_start = Instant::now();
        for i_local in block_start..block_end {
            let row = &row_topk[i_local - block_start];
            let topk = row.as_slice();
            for victim_pos in 1..topk.len() {
                let (victim_local, source_victim_dist2) = topk[victim_pos];
                let threshold = source_victim_dist2 / VIEW_LUNE_ALPHA_IMPL;
                for pivot_pos in 0..victim_pos {
                    let (pivot_local, _) = topk[pivot_pos];
                    let pivot_victim_dist2 =
                        l2_dist2_from_flat(vectors, dim, pivot_local, victim_local);
                    if pivot_victim_dist2 < threshold {
                        lune_witnesses.push(PendingLuneWitness {
                            src: leaf[i_local],
                            pivot: leaf[pivot_local],
                            victim: leaf[victim_local],
                            margin_q16: quantize_lune_margin_q16(threshold, pivot_victim_dist2),
                            pivot_rank: pivot_pos.saturating_add(1).min(u8::MAX as usize) as u8,
                            victim_rank: victim_pos.saturating_add(1).min(u8::MAX as usize) as u8,
                        });
                    }
                }
            }
        }
        profile.lune_witness_records += lune_witnesses.len();
        profile.lune_witness_emit += emit_start.elapsed();
    }

    Ok(ParallelBlockResult {
        block_start,
        profile,
        edges,
        lune_witnesses,
    })
}

pub(crate) fn flush_pending_edges(
    edges: &mut Vec<PendingEdge>,
    edge_sink: &dyn PendingEdgeSink,
) -> AnnResult<()> {
    edge_sink.flush_pending_edges(edges)
}

pub(crate) fn flush_pending_edges_to_reservoirs(
    edges: &mut Vec<PendingEdge>,
    reservoirs: &[Mutex<HashPruneReservoir>],
) {
    if edges.is_empty() {
        return;
    }

    edges.sort_unstable_by_key(|e| e.p);

    let mut i = 0;
    while i < edges.len() {
        let p_idx = edges[i].p;
        let mut res = reservoirs[p_idx].lock();

        while i < edges.len() && edges[i].p == p_idx {
            let edge = &edges[i];
            res.insert(edge.c, edge.hash, edge.dist);
            i += 1;
        }
    }
    edges.clear();
}

fn process_leaf_blockwise(
    dataset: &dyn PointStore,
    metric: Metric,
    sketches: &dyn SketchAccessor,
    leaf: &[u32],
    params: &ForgeANNParams,
    edge_sink: &dyn PendingEdgeSink,
    scratch: &mut LeafScratch,
    ads_runtime: Option<&LeafAdsOperatorRuntime>,
    profile: &mut LeafProfile,
) -> AnnResult<()> {
    let leaf_size = leaf.len();
    if leaf_size <= 1 {
        return Ok(());
    }

    let k = params.leaf_knn.min(leaf_size.saturating_sub(1));
    if k == 0 {
        return Ok(());
    }

    let dim = dataset.dim();
    if scratch.x.nrows() < leaf_size || scratch.x.ncols() < dim {
        scratch.x = Array2::<f32>::zeros((leaf_size.max(scratch.x.nrows()), dim));
    }

    let load_start = Instant::now();
    let mut x_view = scratch.x.slice_mut(ndarray::s![0..leaf_size, 0..dim]);
    let read_options = default_leaf_point_read_options(dataset, leaf_size);
    read_leaf_points_into_matrix(
        dataset,
        leaf,
        x_view.view_mut(),
        &read_options,
        &mut profile.io_stats,
    )?;
    profile.load += load_start.elapsed();

    if should_use_leaf_adsampling_executor(params, metric, leaf_size) {
        let vectors = x_view
            .as_slice_memory_order()
            .expect("leaf matrix is contiguous");
        let result =
            compute_leaf_adsampling_for_params(vectors, leaf_size, dim, k, params, ads_runtime)?;
        profile.distance += result.telemetry.compute_time();
        profile.record_leaf_adsampling(leaf_size, &result.telemetry);
        emit_row_topk_edges(&result.row_topk, sketches, leaf, params, edge_sink, profile)?;
        emit_lune_witnesses_from_vec_topk_l2(
            &result.row_topk,
            vectors,
            dim,
            leaf,
            edge_sink,
            profile,
        )?;
        return Ok(());
    }

    let distance_start = Instant::now();
    let norms: Vec<f32> = if metric == Metric::L2 {
        x_view.map_axis(Axis(1), |row| row.dot(&row)).to_vec()
    } else {
        Vec::new()
    };

    if scratch.row_topk.len() < leaf_size {
        scratch.row_topk.resize_with(leaf_size, RowTopK::default);
    }
    for row in scratch.row_topk.iter_mut().take(leaf_size) {
        row.clear();
    }

    for block_start in (0..leaf_size).step_by(BLOCK_SIZE) {
        let block_end = (block_start + BLOCK_SIZE).min(leaf_size);
        let block_size = block_end - block_start;

        if metric == Metric::L2 {
            scratch.block_dmat.clear();
            scratch.block_dmat.resize(block_size * leaf_size, 0.0f32);

            let x_block = x_view.slice(ndarray::s![block_start..block_end, 0..dim]);
            let mut dmat_view =
                ArrayViewMut2::from_shape((block_size, leaf_size), &mut scratch.block_dmat)
                    .unwrap();
            ndarray::linalg::general_mat_mul(1.0, &x_block, &x_view.t(), 0.0, &mut dmat_view);

            for i_block in 0..block_size {
                let i_local = block_start + i_block;
                let norm_i = norms[i_local];
                let row = &scratch.block_dmat[i_block * leaf_size..(i_block + 1) * leaf_size];
                let topk = &mut scratch.row_topk[i_local];

                for (j_local, &gram_ij) in row.iter().enumerate() {
                    if i_local == j_local {
                        continue;
                    }

                    let mut dist2 = norm_i + norms[j_local] - 2.0 * gram_ij;
                    if dist2 < 0.0 {
                        dist2 = 0.0;
                    }
                    topk.push(k, j_local, dist2);
                }
            }
        } else {
            for i_local in block_start..block_end {
                let i_global = leaf[i_local];
                let topk = &mut scratch.row_topk[i_local];
                for (j_local, &j_global) in leaf.iter().enumerate() {
                    if i_local == j_local {
                        continue;
                    }
                    let d = dataset.get_distance(i_global, j_global, metric)?;
                    topk.push(k, j_local, d);
                }
            }
        }
    }
    profile.distance += distance_start.elapsed();

    if metric == Metric::L2 {
        let vectors = x_view
            .as_slice_memory_order()
            .expect("leaf matrix is contiguous");
        emit_lune_witnesses_from_row_topk_vectors(
            &scratch.row_topk,
            vectors,
            dim,
            leaf,
            edge_sink,
            profile,
        )?;
    }

    scratch.edges.clear();
    let hash_start = Instant::now();
    let sketch_cache = {
        let pf_start = Instant::now();
        let cache = LeafSketchCache::prefetch(sketches, leaf)?;
        profile.sketch_prefetch += pf_start.elapsed();
        profile.sketch_rows_prefetched += leaf.len();
        cache
    };
    for (i_local, &i_global) in leaf.iter().enumerate() {
        let i_idx = i_global as usize;
        let sketch_i = sketch_cache.row(i_local);
        for (rank, &(j_local, d)) in scratch.row_topk[i_local].iter().enumerate() {
            let j_global = leaf[j_local] as usize;
            let sketch_j = sketch_cache.row(j_local);

            let hash_ij = compute_hash(sketch_i, sketch_j, params.m_hash_bits);
            let hash_ji = compute_hash(sketch_j, sketch_i, params.m_hash_bits);
            let local_rank = rank.saturating_add(1).min(u8::MAX as usize) as u8;

            scratch.edges.push(PendingEdge {
                p: i_idx,
                c: j_global as u32,
                hash: hash_ij,
                dist: d,
                mandatory: false,
                local_rank,
                flags: PENDING_EDGE_DIRECT,
            });
            scratch.edges.push(PendingEdge {
                p: j_global,
                c: i_global,
                hash: hash_ji,
                dist: d,
                mandatory: false,
                local_rank,
                flags: PENDING_EDGE_MIRROR,
            });
        }

        if scratch.edges.len() >= 16384 {
            let flush_start = Instant::now();
            scratch.flush_edges(edge_sink)?;
            profile.flush += flush_start.elapsed();
        }
    }
    profile.hash += hash_start.elapsed();

    let flush_start = Instant::now();
    scratch.flush_edges(edge_sink)?;
    profile.flush += flush_start.elapsed();
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use parking_lot::Mutex;

    use super::{
        HashPruneReservoir, LeafScratch, ReservoirEdgeSink, process_leaf,
        process_leaf_parallel_large_profiled, process_leaf_profiled_with_sink_with_ads_runtime,
    };
    use crate::common::Metric;
    use crate::forgeann::ForgeANNParams;
    use crate::forgeann::adsampling::LeafAdsOperatorRuntime;
    use crate::forgeann::hash_prune::SketchStore;
    use crate::forgeann::point_store::InmemDatasetPointStore;
    use crate::model::InmemDataset;

    fn build_test_dataset(num_points: usize, dim: usize) -> InmemDataset<f32> {
        let mut dataset = InmemDataset::new(num_points, 1.0, dim).unwrap();
        for point in 0..num_points {
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = point as f32 * 0.25 + axis as f32 * 0.5;
            }
        }
        dataset
    }

    fn build_test_sketches(num_points: usize, width: usize) -> SketchStore {
        let mut data = Vec::with_capacity(num_points * width);
        for point in 0..num_points {
            for bit in 0..width {
                data.push(point as f32 * 0.1 + bit as f32);
            }
        }
        SketchStore::from_test_data(data, width)
    }

    fn collect_neighbors(reservoirs: Vec<Mutex<HashPruneReservoir>>) -> Vec<Vec<u32>> {
        reservoirs
            .into_iter()
            .map(|reservoir| {
                let mut neighbors = reservoir.into_inner().into_neighbors();
                neighbors.sort_unstable();
                neighbors
            })
            .collect()
    }

    #[test]
    fn leaf_profile_merge_accumulates_all_phases() {
        let mut left = super::LeafProfile::default();
        left.total_wall = Duration::from_millis(10);
        left.distance = Duration::from_millis(6);

        let mut right = super::LeafProfile::default();
        right.total_wall = Duration::from_millis(7);
        right.load = Duration::from_millis(2);
        right.flush = Duration::from_millis(3);
        right.leaves = 2;

        left.merge(right);

        assert_eq!(left.total_wall, Duration::from_millis(17));
        assert_eq!(left.distance, Duration::from_millis(6));
        assert_eq!(left.load, Duration::from_millis(2));
        assert_eq!(left.flush, Duration::from_millis(3));
        assert_eq!(left.leaves, 2);
    }

    #[test]
    fn parallel_large_leaf_matches_serial_leaf_neighbors() {
        let num_points = 96usize;
        let dim = 8usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let sketches = build_test_sketches(num_points, 12);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 12;
        params.leaf_knn = 2;
        params.max_full_matrix_leaf_size = 32;

        let serial_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let parallel_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();

        let mut serial_scratch = LeafScratch::new(leaf.len(), dim, params.leaf_knn);
        process_leaf(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &serial_reservoirs,
            &mut serial_scratch,
        )
        .unwrap();

        let parallel_profile = process_leaf_parallel_large_profiled(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &parallel_reservoirs,
            16,
        )
        .unwrap();

        let serial_neighbors = collect_neighbors(serial_reservoirs);
        let parallel_neighbors = collect_neighbors(parallel_reservoirs);

        assert_eq!(parallel_profile.blockwise_leaves, 1);
        assert_eq!(serial_neighbors, parallel_neighbors);
    }

    #[test]
    fn production_leaf_adsampling_profile_records_large_leaf() {
        let num_points = 640usize;
        let dim = 8usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let sketches = build_test_sketches(num_points, 12);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut adsampling_params = ForgeANNParams::default();
        adsampling_params.m_hash_bits = 12;
        adsampling_params.leaf_knn = 2;
        adsampling_params.adsampling_group_dims = 4;
        adsampling_params.adsampling_epsilon = 100.0;

        let adsampling_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(adsampling_params.l_max)))
            .collect();

        let mut adsampling_scratch = LeafScratch::new(leaf.len(), dim, adsampling_params.leaf_knn);
        let adsampling_profile = super::process_leaf_profiled(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &adsampling_params,
            &adsampling_reservoirs,
            &mut adsampling_scratch,
        )
        .unwrap();

        assert_eq!(adsampling_profile.leaf_adsampling_leaves, 1);
        assert_eq!(adsampling_profile.leaf_adsampling_rows, num_points);
        assert!(adsampling_profile.leaf_adsampling_seed_evals > 0);
        assert!(!adsampling_profile.leaf_ads_tiling_enabled);
        assert_eq!(adsampling_profile.leaf_ads_tiled_leaves, 0);
    }

    #[test]
    fn leaf_adsampling_tiled_runtime_records_profile() {
        let num_points = 640usize;
        let dim = 8usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let sketches = build_test_sketches(num_points, 12);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 12;
        params.leaf_knn = 2;
        params.adsampling_group_dims = 4;
        params.adsampling_epsilon = 100.0;
        params.leaf_ads_tiling_enable = true;
        params.leaf_ads_cpu_budget = 4;
        params.leaf_ads_target_tile_ms = 1;
        params.leaf_ads_min_tile_rows = 64;
        params.leaf_ads_max_tile_rows = 64;
        params.leaf_ads_split_threshold = 64;

        let reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let edge_sink = ReservoirEdgeSink::new(&reservoirs);
        let runtime = LeafAdsOperatorRuntime::from_params(&params, 4);
        let mut scratch = LeafScratch::new(leaf.len(), dim, params.leaf_knn);
        let profile = process_leaf_profiled_with_sink_with_ads_runtime(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &edge_sink,
            &mut scratch,
            Some(&runtime),
        )
        .unwrap();

        assert!(profile.leaf_ads_tiling_enabled);
        assert_eq!(profile.leaf_ads_tiled_leaves, 1);
        assert_eq!(profile.leaf_ads_tiled_rows, num_points);
        assert!(profile.leaf_ads_tiles > 1);
        assert_eq!(profile.leaf_ads_tile_rows_min, 64);
        assert_eq!(profile.leaf_ads_tile_rows_max, 64);
        assert_eq!(profile.leaf_ads_cpu_budget, 4);
        assert!(profile.leaf_ads_active_workers_peak >= 1);
    }

    #[test]
    fn leaf_profile_records_exact_and_adsampling_bucket_timings() {
        let num_points = 512usize;
        let dim = 4usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let sketches = build_test_sketches(num_points, 12);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut exact_params = ForgeANNParams::default();
        exact_params.m_hash_bits = 12;
        exact_params.leaf_knn = 2;
        exact_params.max_full_matrix_leaf_size = 1024;
        let exact_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(exact_params.l_max)))
            .collect();
        let mut exact_scratch = LeafScratch::new(leaf.len(), dim, exact_params.leaf_knn);
        let exact_profile = super::process_leaf_profiled(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &exact_params,
            &exact_reservoirs,
            &mut exact_scratch,
        )
        .unwrap();

        let ads_points = 640usize;
        let ads_dataset = build_test_dataset(ads_points, dim);
        let ads_store = InmemDatasetPointStore::new(&ads_dataset, ads_points);
        let ads_sketches = build_test_sketches(ads_points, 12);
        let ads_leaf: Vec<u32> = (0..ads_points as u32).collect();

        let mut ads_params = exact_params.clone();
        ads_params.adsampling_group_dims = 4;
        ads_params.adsampling_epsilon = 100.0;
        let ads_reservoirs: Vec<_> = (0..ads_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(ads_params.l_max)))
            .collect();
        let mut ads_scratch = LeafScratch::new(ads_leaf.len(), dim, ads_params.leaf_knn);
        let ads_profile = super::process_leaf_profiled(
            &ads_store,
            Metric::L2,
            &ads_sketches,
            &ads_leaf,
            &ads_params,
            &ads_reservoirs,
            &mut ads_scratch,
        )
        .unwrap();

        let bucket = super::leaf_size_bucket_index(num_points).unwrap();
        assert_eq!(bucket, 1);
        assert_eq!(exact_profile.leaf_size_buckets[bucket].exact_leaves, 1);
        assert_eq!(exact_profile.leaf_size_buckets[bucket].adsampling_leaves, 0);
        assert!(exact_profile.leaf_size_buckets[bucket].exact_wall > Duration::ZERO);
        assert_eq!(ads_profile.leaf_size_buckets[bucket].exact_leaves, 0);
        assert_eq!(ads_profile.leaf_size_buckets[bucket].adsampling_leaves, 1);
        assert!(ads_profile.leaf_size_buckets[bucket].adsampling_scan > Duration::ZERO);
    }

    #[test]
    fn cpu_leaf_path_matches_serial_leaf_neighbors() {
        let num_points = 96usize;
        let dim = 8usize;
        let dataset = build_test_dataset(num_points, dim);
        let store = InmemDatasetPointStore::new(&dataset, num_points);
        let sketches = build_test_sketches(num_points, 12);
        let leaf: Vec<u32> = (0..num_points as u32).collect();

        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 12;
        params.leaf_knn = 2;
        params.max_full_matrix_leaf_size = 32;

        let serial_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();
        let backend_reservoirs: Vec<_> = (0..num_points)
            .map(|_| Mutex::new(HashPruneReservoir::new(params.l_max)))
            .collect();

        let mut serial_scratch = LeafScratch::new(leaf.len(), dim, params.leaf_knn);
        process_leaf(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &serial_reservoirs,
            &mut serial_scratch,
        )
        .unwrap();

        let backend_profile = super::process_leaf_parallel_large_profiled(
            &store,
            Metric::L2,
            &sketches,
            &leaf,
            &params,
            &backend_reservoirs,
            16,
        )
        .unwrap();

        let serial_neighbors = collect_neighbors(serial_reservoirs);
        let backend_neighbors = collect_neighbors(backend_reservoirs);

        assert_eq!(backend_profile.blockwise_leaves, 1);
        assert_eq!(serial_neighbors, backend_neighbors);
    }
}
