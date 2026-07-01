pub(crate) mod adsampling;
pub mod params;
pub use params::ForgeANNParams;

pub mod direct_io;
pub mod external_store;
pub(crate) mod final_prune;
pub mod hash_prune;
pub(crate) mod io_planned_forgeann;
pub(crate) mod io_runtime;
pub mod leaf_build;
pub(crate) mod point_pipeline;
pub mod point_store;
pub use point_store::PointBatchStats;
pub mod rbc_partition;
pub(crate) mod sampling;
mod scheduler;
pub mod spill_hashprune;
pub(crate) mod spine_overlay_prune;
pub(crate) mod view_lune_prune;

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};
use parking_lot::Mutex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tracing::info;

use self::direct_io::DirectIoConfig;
use self::external_store::{DiskSketchStore, ResidentPrefixSketchAccessor};
use self::hash_prune::{HashPruneReservoir, SketchAccessor};
use self::leaf_build::{
    LEAF_SIZE_BUCKETS, LeafProfile, PendingEdgeSink, ReservoirEdgeSink, leaf_size_bucket_label,
};
use self::point_pipeline::PointPipelineConfig;
use self::point_store::{
    DirectPointStore, InmemDatasetPointStore, PointStore, uring_gather_counters,
};
use self::rbc_partition::{
    ASSIGNMENT_DEPTH_BUCKETS, ASSIGNMENT_FALLBACK_REASONS, ASSIGNMENT_LEADER_BUCKETS,
    AssignmentDecisionProfile, RbcPartitionTelemetry, assignment_depth_bucket_label,
    assignment_fallback_reason_label, assignment_leader_bucket_label, rbc_partition_streaming,
    rbc_partition_streaming_with_dirs, root_fanout_profile_json,
};
use self::scheduler::{LeafTaskContext, SchedulerBudget, SchedulerTelemetry, run_scheduler_scope};
use self::spill_hashprune::{
    ExternalHashPruneReducer, InMemorySpillEdgeSink, SpillEdgeSink, SpillReduceStats, SpillRunStats,
};
use self::spine_overlay_prune::SpineOverlayEdgeRecorder;
use self::view_lune_prune::ViewLuneEdgeRecorder;
use crate::common::{AnnError, AnnResult};
use crate::model::InmemDataset;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ResidentReservoirPlan {
    resident_points: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OomMemoryPlan {
    total_budget_bytes: usize,
    reservoir_budget_bytes: usize,
    scratch_budget_bytes: usize,
    io_headroom_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OomArtifactDirs {
    root_dir: PathBuf,
    sketch_dir: PathBuf,
    spill_dir: PathBuf,
    partition_dir: PathBuf,
    vector_dir: PathBuf,
}

impl OomArtifactDirs {
    fn unique_dirs(&self) -> Vec<&Path> {
        let mut dirs = Vec::new();
        for dir in [
            self.root_dir.as_path(),
            self.sketch_dir.as_path(),
            self.spill_dir.as_path(),
            self.partition_dir.as_path(),
            self.vector_dir.as_path(),
        ] {
            if !dirs.iter().any(|seen| *seen == dir) {
                dirs.push(dir);
            }
        }
        dirs
    }
}

impl OomMemoryPlan {
    fn from_params(params: &ForgeANNParams) -> Self {
        Self::from_budget(params.effective_oom_memory_budget_bytes())
    }

    fn from_budget(total_budget_bytes: usize) -> Self {
        // The OOM budget is a scheduling envelope for the SOTA resident-subtree
        // path, not a hard partitioned RSS cap. Keep resident reservoirs and leaf
        // scratch at their historical waterlines.
        let reservoir_budget_bytes = total_budget_bytes.saturating_mul(45) / 100;
        let scratch_budget_bytes = total_budget_bytes.saturating_mul(15) / 100;
        let allocated = reservoir_budget_bytes.saturating_add(scratch_budget_bytes);
        let io_headroom_bytes = total_budget_bytes.saturating_sub(allocated);

        Self {
            total_budget_bytes,
            reservoir_budget_bytes,
            scratch_budget_bytes,
            io_headroom_bytes,
        }
    }
}

struct HybridEdgeSink<'a> {
    resident_points: usize,
    reservoirs: &'a [Mutex<HashPruneReservoir>],
    spill_sink: &'a dyn PendingEdgeSink,
}

struct NullEdgeSink;

impl PendingEdgeSink for NullEdgeSink {
    fn flush_pending_edges(&self, edges: &mut Vec<self::leaf_build::PendingEdge>) -> AnnResult<()> {
        edges.clear();
        Ok(())
    }
}

impl PendingEdgeSink for HybridEdgeSink<'_> {
    fn flush_pending_edges(&self, edges: &mut Vec<self::leaf_build::PendingEdge>) -> AnnResult<()> {
        if edges.is_empty() {
            return Ok(());
        }

        let mut resident_edges = Vec::new();
        let mut spill_edges = Vec::new();
        for edge in edges.drain(..) {
            if edge.p < self.resident_points {
                resident_edges.push(edge);
            } else {
                spill_edges.push(edge);
            }
        }

        if !resident_edges.is_empty() {
            self::leaf_build::flush_pending_edges_to_reservoirs(
                &mut resident_edges,
                self.reservoirs,
            );
        }
        if !spill_edges.is_empty() {
            self.spill_sink.flush_pending_edges(&mut spill_edges)?;
        }
        Ok(())
    }
}

/// 使用 ForgeANN 在内存中构建邻接表形式的图。
///
/// - `dataset`：已经通过 InmemDataset 加载好的向量数据；
/// - `num_points`：参与构图的点数（通常等于输入数据集行数）；
/// - `num_threads`：用于叶子并行处理的线程数（0 表示 Rayon 默认线程数）。
/// - `on_node_done`：回调函数，当某个节点的构图完成时调用，传入 `(u32, Vec<u32>)`。
pub fn build_forgeann_graph<F>(
    dataset: &InmemDataset<f32>,
    num_points: usize,
    num_threads: u32,
    params: &ForgeANNParams,
    on_node_done: F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    let store = InmemDatasetPointStore::new(dataset, num_points);
    build_forgeann_graph_with_store(&store, num_points, num_threads, params, on_node_done)
        .map(|_| ())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ForgeANNBuildSummary {
    pub graph_start_hint: Option<u32>,
}

pub fn build_forgeann_graph_with_store<F>(
    dataset: &dyn PointStore,
    num_points: usize,
    num_threads: u32,
    params: &ForgeANNParams,
    on_node_done: F,
) -> AnnResult<ForgeANNBuildSummary>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    build_forgeann_graph_with_store_optional_spine_overlay_recorder(
        dataset,
        num_points,
        num_threads,
        params,
        None,
        on_node_done,
    )
}

pub(crate) fn build_forgeann_graph_with_store_and_spine_overlay_recorder<F>(
    dataset: &dyn PointStore,
    num_points: usize,
    num_threads: u32,
    params: &ForgeANNParams,
    spine_overlay_recorder: &dyn SpineOverlayEdgeRecorder,
    on_node_done: F,
) -> AnnResult<ForgeANNBuildSummary>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    build_forgeann_graph_with_store_optional_spine_overlay_recorder(
        dataset,
        num_points,
        num_threads,
        params,
        Some(spine_overlay_recorder),
        on_node_done,
    )
}

pub(crate) fn build_forgeann_graph_with_store_and_view_lune_recorder<F>(
    dataset: &dyn PointStore,
    num_points: usize,
    num_threads: u32,
    params: &ForgeANNParams,
    view_lune_recorder: &dyn ViewLuneEdgeRecorder,
    on_node_done: F,
) -> AnnResult<ForgeANNBuildSummary>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    build_forgeann_graph_with_store_optional_prune_recorders(
        dataset,
        num_points,
        num_threads,
        params,
        None,
        Some(view_lune_recorder),
        on_node_done,
    )
}

fn build_forgeann_graph_with_store_optional_spine_overlay_recorder<F>(
    dataset: &dyn PointStore,
    num_points: usize,
    num_threads: u32,
    params: &ForgeANNParams,
    external_spine_overlay_recorder: Option<&dyn SpineOverlayEdgeRecorder>,
    on_node_done: F,
) -> AnnResult<ForgeANNBuildSummary>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    build_forgeann_graph_with_store_optional_prune_recorders(
        dataset,
        num_points,
        num_threads,
        params,
        external_spine_overlay_recorder,
        None,
        on_node_done,
    )
}

fn build_forgeann_graph_with_store_optional_prune_recorders<F>(
    dataset: &dyn PointStore,
    num_points: usize,
    num_threads: u32,
    params: &ForgeANNParams,
    external_spine_overlay_recorder: Option<&dyn SpineOverlayEdgeRecorder>,
    external_view_lune_recorder: Option<&dyn ViewLuneEdgeRecorder>,
    mut on_node_done: F,
) -> AnnResult<ForgeANNBuildSummary>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    if num_points == 0 {
        return Ok(ForgeANNBuildSummary::default());
    }

    assert!(params.m_hash_bits <= 16, "m_hash_bits must be <= 16");
    // 当前 Vertex::compare 仅实现 L2，度量通过 Metric::L2 传入。
    let metric = params.metric();
    let effective_threads = if num_threads == 0 {
        rayon::current_num_threads()
    } else {
        num_threads as usize
    };
    let detailed_profiling = detailed_forgeann_profiling_enabled();
    info!(
        "ForgeANN build start: num_points={}, num_threads={}",
        num_points, num_threads
    );
    if detailed_profiling {
        info!("ForgeANN detailed profiling enabled via FORGEANN_PROFILE");
    }
    log_memory_telemetry(
        "start",
        num_points,
        params,
        dataset.dim(),
        0,
        0,
        effective_threads,
    );

    let blas_env_backup = BlasThreadEnv::capture();

    let result = (|| {
        let adsampling_sidecar_store =
            prepare_adsampling_assignment_store(dataset, num_points, params)?;
        let adsampling_assignment_dataset = adsampling_sidecar_store.as_deref();

        if params.oom_enable {
            return build_forgeann_graph_oom(
                dataset,
                adsampling_assignment_dataset,
                num_points,
                effective_threads,
                params,
                detailed_profiling,
                external_spine_overlay_recorder,
                external_view_lune_recorder,
                &mut on_node_done,
            );
        }

        // 1) 预计算所有点的 m 维 sketch（一次性完成）。 RBC 之前必须完成，因为 process_leaf
        //    依赖它。
        let t_sketch = Instant::now();
        let sketches =
            hash_prune::compute_sketches_from_store(dataset, num_points, params, num_threads)?;
        let sketch_bytes = estimate_sketch_bytes(&sketches);
        info!(
            "ForgeANN compute_sketches done: elapsed={:?}",
            t_sketch.elapsed()
        );

        log_memory_telemetry(
            "after sketches",
            num_points,
            params,
            dataset.dim(),
            sketch_bytes,
            0,
            effective_threads,
        );

        // 2) 为每个点分配 HashPrune 水库（带互斥锁以支持多线程叶处理）。 也必须预先分配。
        let t_res = Instant::now();
        let mut reservoirs: Vec<Mutex<HashPruneReservoir>> = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            reservoirs.push(Mutex::new(HashPruneReservoir::new(params.l_max)));
        }
        let reservoir_bytes = estimate_reservoir_bytes(num_points, params.l_max);
        info!(
            "ForgeANN init reservoirs done: num_points={}, elapsed={:?}",
            num_points,
            t_res.elapsed()
        );
        log_memory_telemetry(
            "after reservoirs",
            num_points,
            params,
            dataset.dim(),
            sketch_bytes,
            reservoir_bytes,
            effective_threads,
        );

        // 3) 启动统一调度的 RBC + Leaf 处理。 RBC 递归与叶子构建共用同一个受限 Rayon
        //    池，num_threads 是整个 ForgeANN 构建阶段的硬上限。叶子任务采用 scope 内 FIFO
        //    调度，并在饱和时回退为 inline， 避免额外 OS consumer 线程和双倍线程占用。
        let t_rbc_leaf = Instant::now();
        let all_indices: Vec<u32> = (0..num_points as u32).collect();

        // 进度条
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} [{elapsed_precise}] {msg}")
                .unwrap(),
        );
        pb.set_message("ForgeANN streaming processing...");

        let leaf_scratch_bytes = estimate_leaf_scratch_bytes(
            params.kernel_safe_leaf_size(),
            dataset.dim(),
            params.leaf_knn,
        );
        let scheduler_budget = apply_immediate_leaf_drain_budget(
            SchedulerBudget::for_worker_count(effective_threads, params.kernel_safe_leaf_size()),
            effective_threads,
        );
        info!(
            "ForgeANN scheduler leaf drain policy: producer_leaf_backlog_soft_limit={}",
            scheduler_budget.producer_leaf_backlog_soft_limit
        );

        // 在线程并行处理叶子之前，将底层 BLAS/OpenMP 线程数钉死为 1，避免 oversubscription。
        BlasThreadEnv::set_single_threaded();

        // 单个 Rayon 池承载 RBC 递归与叶子处理，线程数受 --num-threads 硬限制。
        let pool = build_forgeann_rayon_pool(effective_threads)?;

        let edge_sink = ReservoirEdgeSink::new(reservoirs.as_slice());
        let point_pipeline_config = PointPipelineConfig::from_params(params);
        let point_pipeline_config = point_pipeline_config
            .enabled
            .then_some(point_pipeline_config);
        let leaf_context = LeafTaskContext {
            dataset,
            metric,
            sketches: &sketches,
            params,
            edge_sink: &edge_sink,
            spine_overlay_recorder: external_spine_overlay_recorder,
            view_lune_recorder: external_view_lune_recorder,
            point_pipeline_config: point_pipeline_config.as_ref(),
            detailed_profiling,
            #[cfg(test)]
            observer: None,
        };
        let producer_start = Instant::now();
        let mut rbc_telemetry = RbcPartitionTelemetry::default();
        let (scheduler_profile, scheduler_telemetry) =
            run_scheduler_scope(&pool, scheduler_budget, leaf_context, &pb, |emitter| {
                let mut rng = StdRng::seed_from_u64(params.random_seed);
                rbc_telemetry = rbc_partition_streaming(
                    dataset,
                    adsampling_assignment_dataset,
                    &all_indices,
                    metric,
                    params,
                    effective_threads as u32,
                    &mut rng,
                    None,
                    emitter,
                )?;
                Ok(())
            })?;
        let producer_wall = producer_start.elapsed();

        let streaming_profile = StreamingProfile {
            producer_wall,
            total_wall: t_rbc_leaf.elapsed(),
            leaf: scheduler_profile,
            telemetry: scheduler_telemetry,
            rbc: rbc_telemetry,
        };

        pb.finish_with_message("ForgeANN streaming processing done");
        info!(
            "ForgeANN streaming (RBC+Leaf) done: elapsed={:?}",
            t_rbc_leaf.elapsed()
        );
        if detailed_profiling {
            let (uring_batches, uring_windows, uring_bytes, uring_fallbacks) =
                uring_gather_counters();
            info!(
                "ForgeANN streaming profile: producer_wall={:?} total_wall={:?} leaf_parallelism={:.2} leaf_total_wall={:?} leaf_idle_estimate={:?} leaves={} points={} blockwise_leaves={} peak_inflight_leaf_tasks={} peak_leaf_backlog={} producer_help_drains={} producer_help_yields={} inline_leaf_fallbacks={} large_leaf_count={} large_leaf_block_tasks={} batched_leaf_drains={} batched_leaf_leaves={} batched_leaf_unique_points={} batched_leaf_load_ms={}",
                streaming_profile.producer_wall,
                streaming_profile.total_wall,
                streaming_profile.leaf_parallelism(),
                streaming_profile.leaf.total_wall,
                streaming_profile.leaf_idle_time(),
                streaming_profile.leaf.leaves,
                streaming_profile.leaf.points,
                streaming_profile.leaf.blockwise_leaves,
                streaming_profile.telemetry.peak_inflight_leaf_tasks,
                streaming_profile.telemetry.peak_leaf_backlog,
                streaming_profile.telemetry.producer_help_drains,
                streaming_profile.telemetry.producer_help_yields,
                streaming_profile.telemetry.inline_leaf_fallbacks,
                streaming_profile.telemetry.large_leaf_count,
                streaming_profile.telemetry.large_leaf_block_tasks,
                streaming_profile.telemetry.batched_leaf_drains,
                streaming_profile.telemetry.batched_leaf_leaves,
                streaming_profile.telemetry.batched_leaf_unique_points,
                streaming_profile.telemetry.batched_leaf_load_ms,
            );
            info!(
                "ForgeANN leaf profile: load={:?} distance={:?} topk={:?} hash={:?} flush={:?} lune_witness_emit={:?} lune_witness_records={}",
                streaming_profile.leaf.load,
                streaming_profile.leaf.distance,
                streaming_profile.leaf.topk,
                streaming_profile.leaf.hash,
                streaming_profile.leaf.flush,
                streaming_profile.leaf.lune_witness_emit,
                streaming_profile.leaf.lune_witness_records,
            );
            info!(
                "ForgeANN point io_uring gather: batches={} windows={} bytes={} fallbacks={}",
                uring_batches, uring_windows, uring_bytes, uring_fallbacks,
            );
        }
        log_memory_telemetry(
            "after RBC+leaf",
            num_points,
            params,
            dataset.dim(),
            sketch_bytes,
            reservoir_bytes,
            effective_threads,
        );
        info!(
            "ForgeANN leaf scratch estimate: max_leaf_size={} per_worker_bytes={} active_scratch_bytes_estimate={}",
            params.kernel_safe_leaf_size(),
            leaf_scratch_bytes,
            leaf_scratch_bytes.saturating_mul(effective_threads),
        );

        // 4) 把 HashPrune 水库转换成邻接表 Vec<Vec<u32>>。 逐个转换并调用回调。
        let t_graph = Instant::now();
        for (i, res) in reservoirs.into_iter().enumerate() {
            let neighbors = res.into_inner().into_neighbors();
            on_node_done(i as u32, neighbors)?;
        }
        info!("ForgeANN build graph done: elapsed={:?}", t_graph.elapsed());
        log_memory_telemetry(
            "after graph",
            num_points,
            params,
            dataset.dim(),
            sketch_bytes,
            0,
            effective_threads,
        );

        Ok(ForgeANNBuildSummary::default())
    })();

    blas_env_backup.restore();

    result
}

fn prepare_adsampling_assignment_store(
    dataset: &dyn PointStore,
    num_points: usize,
    params: &ForgeANNParams,
) -> AnnResult<Option<Box<dyn PointStore>>> {
    if !params.root_adsampling_enabled() {
        return Ok(None);
    }

    let Some(path) = params.adsampling_rotation_sidecar.as_deref() else {
        return Ok(None);
    };
    let io_cfg = if params.strict_oom_io_enabled() {
        DirectIoConfig::enabled_with_alignment(4096)
    } else {
        DirectIoConfig::disabled()
    };
    let store: Box<dyn PointStore> = Box::new(DirectPointStore::open_with_config(path, io_cfg)?);
    if store.dim() != dataset.dim() {
        return Err(AnnError::log_index_config_error(
            "adsampling_rotation_sidecar".to_string(),
            format!(
                "ADSampling sidecar dim {} does not match dataset dim {}",
                store.dim(),
                dataset.dim()
            ),
        ));
    }
    if store.len() < num_points {
        return Err(AnnError::log_index_config_error(
            "adsampling_rotation_sidecar".to_string(),
            format!(
                "ADSampling sidecar has {} points but build needs {}",
                store.len(),
                num_points
            ),
        ));
    }
    info!(
        "ForgeANN ADSampling sidecar opened: backend=direct path={:?} dim={}",
        path,
        store.dim(),
    );
    Ok(Some(store))
}

fn build_forgeann_graph_oom<F>(
    dataset: &dyn PointStore,
    adsampling_assignment_dataset: Option<&dyn PointStore>,
    num_points: usize,
    requested_threads: usize,
    params: &ForgeANNParams,
    detailed_profiling: bool,
    external_spine_overlay_recorder: Option<&dyn SpineOverlayEdgeRecorder>,
    external_view_lune_recorder: Option<&dyn ViewLuneEdgeRecorder>,
    on_node_done: &mut F,
) -> AnnResult<ForgeANNBuildSummary>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    let artifact_dirs = prepare_oom_artifact_dirs(params)?;
    let sketch_path = artifact_dirs.sketch_dir.join("sketches.bin");
    let candidate_path = artifact_dirs.spill_dir.join("candidate.run");
    let profile_path = params.oom_profile_json.clone();
    let overall_start = Instant::now();
    let mut memory_report = OomMemoryReport::default();
    memory_report.capture_stage("start");
    let memory_plan = OomMemoryPlan::from_params(params);
    let memory_budget_bytes = memory_plan.total_budget_bytes;
    let streaming_params = params.clone();
    let execution_threads = derive_oom_execution_threads(requested_threads, params, dataset.dim());
    let pool_threads = execution_threads.pool_threads;
    let leaf_budget_threads = execution_threads.leaf_budget_threads;
    info!(
        "ForgeANN OOM thread model: requested_threads={} pool_threads={} leaf_budget_threads={} memory_budget_bytes={} reservoir_budget_bytes={} scratch_budget_bytes={} io_headroom_bytes={}",
        requested_threads,
        pool_threads,
        leaf_budget_threads,
        memory_budget_bytes,
        memory_plan.reservoir_budget_bytes,
        memory_plan.scratch_budget_bytes,
        memory_plan.io_headroom_bytes,
    );

    let sketch_persist_start = Instant::now();
    let disk_sketches = DiskSketchStore::build_from_point_store(
        &sketch_path,
        dataset,
        num_points,
        &streaming_params,
        pool_threads as u32,
    )?;
    let graph_start_hint = disk_sketches.sampled_medoid_start();
    let sketch_persist_wall = sketch_persist_start.elapsed();
    let sketch_bytes = disk_sketches.resident_bytes();
    let spill_bytes = disk_sketches.file_bytes();

    log_memory_telemetry(
        "oom after sketches",
        num_points,
        params,
        dataset.dim(),
        sketch_bytes,
        0,
        leaf_budget_threads,
    );
    memory_report.capture_stage("after_sketches");

    let view_lune_metadata_only =
        external_view_lune_recorder.is_some() && params.view_lune_prune_enable;
    let shard_points = derive_spill_shard_points(num_points, params, memory_budget_bytes);
    let (resident_plan, effective_reservoir_budget_bytes) =
        if let Some(cap_bytes) = params.oom_resident_reservoir_cap_bytes {
            (
                plan_resident_reservoir_prefix_from_cap(num_points, params, cap_bytes),
                cap_bytes,
            )
        } else {
            (
                plan_resident_reservoir_prefix(
                    num_points,
                    leaf_budget_threads,
                    params,
                    dataset.dim(),
                    sketch_bytes,
                    shard_points,
                    memory_plan.reservoir_budget_bytes,
                ),
                memory_plan.reservoir_budget_bytes,
            )
        };
    info!(
        "ForgeANN OOM resident reservoir plan: resident_points={} total_points={} plan_budget_bytes={} base_reservoir_budget_bytes={} cap_bytes={:?} shard_points={}",
        resident_plan.resident_points,
        num_points,
        effective_reservoir_budget_bytes,
        memory_plan.reservoir_budget_bytes,
        params.oom_resident_reservoir_cap_bytes,
        shard_points
    );
    let mut effective_sketch_bytes = sketch_bytes;
    memory_report.capture_stage("after_root");

    let (
        streaming_profile,
        streaming_wall,
        spill_stats,
        reduce_stats,
        reduce_wall,
        emit_wall,
        resident_reservoir_bytes,
    ) = if view_lune_metadata_only {
        let resident_sketches = if should_use_full_resident_sketch_cache(
            &streaming_params,
            disk_sketches.file_bytes(),
        ) {
            let sketch_load_start = Instant::now();
            let resident_sketches =
                ResidentPrefixSketchAccessor::from_disk_prefix(&disk_sketches, num_points)?;
            info!(
                "ForgeANN OOM ViewLune metadata-only full resident sketch cache enabled: rows={} resident_bytes={} load_elapsed={:?}",
                num_points,
                resident_sketches.resident_bytes(),
                sketch_load_start.elapsed()
            );
            Some(resident_sketches)
        } else {
            info!(
                "ForgeANN OOM ViewLune metadata-only full resident sketch cache disabled: sketch_file_bytes={} sketch_cache_budget_bytes={}",
                disk_sketches.file_bytes(),
                streaming_params.oom_sketch_cache_bytes
            );
            None
        };
        let total_sketch_bytes = sketch_bytes.saturating_add(
            resident_sketches
                .as_ref()
                .map(|sketches| sketches.resident_bytes())
                .unwrap_or(0),
        );
        effective_sketch_bytes = total_sketch_bytes;
        let streaming_sketches: &dyn SketchAccessor =
            if let Some(sketches) = resident_sketches.as_ref() {
                sketches
            } else {
                &disk_sketches
            };
        let null_sink = NullEdgeSink;
        let streaming_start = Instant::now();
        let streaming_profile = run_streaming_partition_pipeline(
            dataset,
            adsampling_assignment_dataset,
            num_points,
            pool_threads,
            leaf_budget_threads,
            &streaming_params,
            &artifact_dirs,
            streaming_sketches,
            detailed_profiling,
            &null_sink,
            None,
            external_view_lune_recorder,
        )?;
        let streaming_wall = streaming_start.elapsed();
        info!(
            "ForgeANN OOM ViewLune metadata-only streaming done: elapsed={:?}",
            streaming_wall
        );
        (
            streaming_profile,
            streaming_wall,
            SpillRunStats::default(),
            SpillReduceStats::default(),
            Duration::ZERO,
            Duration::ZERO,
            0,
        )
    } else if resident_plan.resident_points == num_points {
        let t_res = Instant::now();
        let mut reservoirs: Vec<Mutex<HashPruneReservoir>> = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            reservoirs.push(Mutex::new(HashPruneReservoir::new(params.l_max)));
        }
        let reservoir_bytes = estimate_reservoir_bytes(num_points, params.l_max);
        info!(
            "ForgeANN OOM resident-reservoir mode enabled: num_points={} reservoir_bytes_estimate={} alloc_elapsed={:?}",
            num_points,
            reservoir_bytes,
            t_res.elapsed()
        );
        let resident_sketches = if should_use_full_resident_sketch_cache(
            &streaming_params,
            disk_sketches.file_bytes(),
        ) {
            let sketch_load_start = Instant::now();
            let resident_sketches =
                ResidentPrefixSketchAccessor::from_disk_prefix(&disk_sketches, num_points)?;
            info!(
                "ForgeANN OOM full resident sketch cache enabled: rows={} resident_bytes={} load_elapsed={:?}",
                num_points,
                resident_sketches.resident_bytes(),
                sketch_load_start.elapsed()
            );
            Some(resident_sketches)
        } else {
            info!(
                "ForgeANN OOM full resident sketch cache disabled: sketch_file_bytes={} sketch_cache_budget_bytes={}",
                disk_sketches.file_bytes(),
                streaming_params.oom_sketch_cache_bytes
            );
            None
        };
        let total_sketch_bytes = sketch_bytes.saturating_add(
            resident_sketches
                .as_ref()
                .map(|sketches| sketches.resident_bytes())
                .unwrap_or(0),
        );
        effective_sketch_bytes = total_sketch_bytes;
        log_memory_telemetry(
            "oom after resident reservoirs",
            num_points,
            params,
            dataset.dim(),
            total_sketch_bytes,
            reservoir_bytes,
            leaf_budget_threads,
        );

        let edge_sink = ReservoirEdgeSink::new(reservoirs.as_slice());
        let streaming_sketches: &dyn SketchAccessor =
            if let Some(sketches) = resident_sketches.as_ref() {
                sketches
            } else {
                &disk_sketches
            };
        let streaming_start = Instant::now();
        let streaming_profile = run_streaming_partition_pipeline(
            dataset,
            adsampling_assignment_dataset,
            num_points,
            pool_threads,
            leaf_budget_threads,
            &streaming_params,
            &artifact_dirs,
            streaming_sketches,
            detailed_profiling,
            &edge_sink,
            external_spine_overlay_recorder,
            external_view_lune_recorder,
        )?;
        let streaming_wall = streaming_start.elapsed();
        info!(
            "ForgeANN OOM resident-reservoir streaming done: elapsed={:?}",
            streaming_wall
        );
        let emit_start = Instant::now();
        for (i, res) in reservoirs.into_iter().enumerate() {
            let neighbors = res.into_inner().into_neighbors();
            on_node_done(i as u32, neighbors)?;
        }
        let emit_wall = emit_start.elapsed();
        log_memory_telemetry(
            "oom after resident emit",
            num_points,
            params,
            dataset.dim(),
            total_sketch_bytes,
            0,
            leaf_budget_threads,
        );

        (
            streaming_profile,
            streaming_wall,
            SpillRunStats::default(),
            SpillReduceStats::default(),
            Duration::ZERO,
            emit_wall,
            reservoir_bytes,
        )
    } else if resident_plan.resident_points > 0 {
        let resident_points = resident_plan.resident_points;
        let t_res = Instant::now();
        let mut reservoirs: Vec<Mutex<HashPruneReservoir>> = Vec::with_capacity(resident_points);
        for _ in 0..resident_points {
            reservoirs.push(Mutex::new(HashPruneReservoir::new(params.l_max)));
        }
        let reservoir_bytes = estimate_reservoir_bytes(resident_points, params.l_max);
        info!(
            "ForgeANN OOM hybrid resident prefix enabled: resident_points={} total_points={} reservoir_bytes_estimate={} alloc_elapsed={:?}",
            resident_points,
            num_points,
            reservoir_bytes,
            t_res.elapsed()
        );
        let resident_sketches =
            ResidentPrefixSketchAccessor::from_disk_prefix(&disk_sketches, resident_points)?;
        let resident_sketch_bytes = resident_sketches.resident_bytes();
        let total_sketch_bytes = sketch_bytes.saturating_add(resident_sketch_bytes);
        effective_sketch_bytes = total_sketch_bytes;
        log_memory_telemetry(
            "oom after hybrid resident reservoirs",
            num_points,
            params,
            dataset.dim(),
            total_sketch_bytes,
            reservoir_bytes,
            leaf_budget_threads,
        );
        let use_in_memory_spill = false;
        let (streaming_profile, streaming_wall, spill_stats, reduce_stats, reduce_wall, emit_wall) =
            if use_in_memory_spill {
                info!(
                    "ForgeANN OOM hybrid resident prefix using in-memory spill: resident_points={} total_points={} shard_points={} spill_cache_bytes={}",
                    resident_points, num_points, shard_points, params.oom_spill_cache_bytes
                );
                let spill_sink = InMemorySpillEdgeSink::create_sharded(
                    params.oom_spill_cache_bytes,
                    shard_points,
                );
                let hybrid_sink = HybridEdgeSink {
                    resident_points,
                    reservoirs: reservoirs.as_slice(),
                    spill_sink: &spill_sink,
                };
                let streaming_start = Instant::now();
                let streaming_profile = run_streaming_partition_pipeline(
                    dataset,
                    adsampling_assignment_dataset,
                    num_points,
                    pool_threads,
                    leaf_budget_threads,
                    &streaming_params,
                    &artifact_dirs,
                    &resident_sketches,
                    detailed_profiling,
                    &hybrid_sink,
                    external_spine_overlay_recorder,
                    external_view_lune_recorder,
                )?;
                let streaming_wall = streaming_start.elapsed();
                let spill_artifacts = spill_sink.finish()?;
                let spill_stats =
                    ExternalHashPruneReducer::summarize_in_memory_spill_artifacts(&spill_artifacts);

                let emit_start = Instant::now();
                for (i, res) in reservoirs.into_iter().enumerate() {
                    let neighbors = res.into_inner().into_neighbors();
                    on_node_done(i as u32, neighbors)?;
                }
                let emit_wall = emit_start.elapsed();

                let reduce_start = Instant::now();
                let reduce_stats =
                    ExternalHashPruneReducer::reduce_in_memory_spill_sharded_in_order_profiled_from(
                        resident_points,
                        num_points,
                        params.l_max,
                        spill_artifacts,
                        |point, neighbors| on_node_done(point, neighbors),
                    )?;
                let reduce_wall = reduce_start.elapsed();
                (
                    streaming_profile,
                    streaming_wall,
                    spill_stats,
                    reduce_stats,
                    reduce_wall,
                    emit_wall,
                )
            } else {
                let io_cfg = if params.strict_oom_io_enabled() {
                    DirectIoConfig::enabled_with_alignment(4096)
                } else {
                    DirectIoConfig::disabled()
                };
                let spill_sink = SpillEdgeSink::create_sharded_with_config(
                    &candidate_path,
                    params.oom_spill_cache_bytes,
                    shard_points,
                    io_cfg,
                )?;
                let hybrid_sink = HybridEdgeSink {
                    resident_points,
                    reservoirs: reservoirs.as_slice(),
                    spill_sink: &spill_sink,
                };
                let streaming_start = Instant::now();
                let streaming_profile = run_streaming_partition_pipeline(
                    dataset,
                    adsampling_assignment_dataset,
                    num_points,
                    pool_threads,
                    leaf_budget_threads,
                    &streaming_params,
                    &artifact_dirs,
                    &resident_sketches,
                    detailed_profiling,
                    &hybrid_sink,
                    external_spine_overlay_recorder,
                    external_view_lune_recorder,
                )?;
                let streaming_wall = streaming_start.elapsed();
                let spill_artifacts = spill_sink.finish()?;
                let spill_stats =
                    ExternalHashPruneReducer::summarize_spill_artifacts(&spill_artifacts);

                let emit_start = Instant::now();
                for (i, res) in reservoirs.into_iter().enumerate() {
                    let neighbors = res.into_inner().into_neighbors();
                    on_node_done(i as u32, neighbors)?;
                }
                let emit_wall = emit_start.elapsed();

                let reduce_start = Instant::now();
                let reduce_parallel_groups =
                    ForgeANNParams::OOM_POINT_PIPELINE_IO_THREADS.min(pool_threads.max(1));
                info!(
                    "ForgeANN OOM external reduce starting: point_start={} num_points={} shard_points={} parallel_groups={}",
                    resident_points, num_points, shard_points, reduce_parallel_groups
                );
                let reduce_stats =
                    ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_from_parallel(
                        resident_points,
                        num_points,
                        params.l_max,
                        shard_points,
                        &spill_artifacts,
                        reduce_parallel_groups,
                        |point, neighbors| on_node_done(point, neighbors),
                    )?;
                let reduce_wall = reduce_start.elapsed();
                (
                    streaming_profile,
                    streaming_wall,
                    spill_stats,
                    reduce_stats,
                    reduce_wall,
                    emit_wall,
                )
            };

        (
            streaming_profile,
            streaming_wall,
            spill_stats,
            reduce_stats,
            reduce_wall,
            emit_wall,
            reservoir_bytes,
        )
    } else {
        let io_cfg = if params.strict_oom_io_enabled() {
            DirectIoConfig::enabled_with_alignment(4096)
        } else {
            DirectIoConfig::disabled()
        };
        let spill_sink = SpillEdgeSink::create_sharded_with_config(
            &candidate_path,
            params.oom_spill_cache_bytes,
            shard_points,
            io_cfg,
        )?;
        let streaming_start = Instant::now();
        let streaming_profile = run_streaming_partition_pipeline(
            dataset,
            adsampling_assignment_dataset,
            num_points,
            pool_threads,
            leaf_budget_threads,
            &streaming_params,
            &artifact_dirs,
            &disk_sketches,
            detailed_profiling,
            &spill_sink,
            external_spine_overlay_recorder,
            external_view_lune_recorder,
        )?;
        let streaming_wall = streaming_start.elapsed();
        let spill_artifacts = spill_sink.finish()?;
        let spill_stats = ExternalHashPruneReducer::summarize_spill_artifacts(&spill_artifacts);

        let reduce_start = Instant::now();
        let reduce_parallel_groups =
            ForgeANNParams::OOM_POINT_PIPELINE_IO_THREADS.min(pool_threads.max(1));
        info!(
            "ForgeANN OOM external reduce starting: point_start={} num_points={} shard_points={} parallel_groups={}",
            0, num_points, shard_points, reduce_parallel_groups
        );
        let reduce_stats =
            ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_from_parallel(
                0,
                num_points,
                params.l_max,
                shard_points,
                &spill_artifacts,
                reduce_parallel_groups,
                |point, neighbors| on_node_done(point, neighbors),
            )?;
        let reduce_wall = reduce_start.elapsed();
        info!(
            "ForgeANN OOM external-reduce mode done: streaming_elapsed={:?} reduce_elapsed={:?}",
            streaming_wall, reduce_wall
        );

        (
            streaming_profile,
            streaming_wall,
            spill_stats,
            reduce_stats,
            reduce_wall,
            Duration::ZERO,
            0,
        )
    };

    let (uring_batches, uring_windows, uring_bytes, uring_fallbacks) = uring_gather_counters();
    trim_process_allocator("oom after streaming/reduce");
    memory_report.capture_stage("after_streaming");
    memory_report.capture_stage("after_reduce");
    memory_report.capture_stage("final");
    memory_report.refresh_summary_from_stages();
    if detailed_profiling {
        info!(
            "ForgeANN OOM point io_uring gather: batches={} windows={} bytes={} fallbacks={}",
            uring_batches, uring_windows, uring_bytes, uring_fallbacks,
        );
    }

    if let Some(path) = profile_path {
        write_oom_profile(
            &path,
            &OomProfile {
                num_points,
                dim: dataset.dim(),
                point_store_backend: "direct",
                memory_budget_bytes,
                effective_threads: pool_threads,
                pool_threads,
                leaf_budget_threads,
                reduce_shard_points: shard_points,
                sketch_bytes: effective_sketch_bytes,
                sketch_file_bytes: spill_bytes,
                spill: spill_stats,
                reduce: reduce_stats,
                sketch_persist_wall,
                streaming_wall,
                reduce_wall,
                emit_wall,
                total_wall: overall_start.elapsed(),
                scheduler: streaming_profile.telemetry,
                rbc: streaming_profile.rbc,
                leaf: streaming_profile.leaf,
                memory: memory_report,
                temp_files: collect_oom_temp_file_report(&artifact_dirs)?,
                uring_batches,
                uring_windows,
                uring_bytes,
                uring_fallbacks,
            },
        )?;
    }

    if resident_reservoir_bytes > 0 {
        info!(
            "ForgeANN OOM completed with resident reservoirs: estimated_bytes={resident_reservoir_bytes} budget_bytes={memory_budget_bytes}"
        );
    }

    if !params.oom_keep_artifacts {
        cleanup_oom_artifacts_in_dirs(&artifact_dirs)?;
    }

    Ok(ForgeANNBuildSummary { graph_start_hint })
}

fn run_streaming_partition_pipeline(
    dataset: &dyn PointStore,
    adsampling_assignment_dataset: Option<&dyn PointStore>,
    num_points: usize,
    pool_threads: usize,
    leaf_budget_threads: usize,
    params: &ForgeANNParams,
    artifact_dirs: &OomArtifactDirs,
    sketches: &dyn SketchAccessor,
    detailed_profiling: bool,
    edge_sink: &dyn PendingEdgeSink,
    spine_overlay_recorder: Option<&dyn SpineOverlayEdgeRecorder>,
    view_lune_recorder: Option<&dyn ViewLuneEdgeRecorder>,
) -> AnnResult<StreamingProfile> {
    let t_rbc_leaf = Instant::now();

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} [{elapsed_precise}] {msg}")
            .unwrap(),
    );
    pb.set_message("ForgeANN streaming processing...");

    let leaf_scratch_bytes = estimate_leaf_scratch_bytes(
        params.kernel_safe_leaf_size(),
        dataset.dim(),
        params.leaf_knn,
    );
    let scheduler_budget = apply_immediate_leaf_drain_budget(
        SchedulerBudget::for_memory_budget(
            leaf_budget_threads,
            params.kernel_safe_leaf_size(),
            params.effective_oom_memory_budget_bytes(),
        ),
        leaf_budget_threads,
    );
    info!(
        "ForgeANN scheduler leaf drain policy: producer_leaf_backlog_soft_limit={}",
        scheduler_budget.producer_leaf_backlog_soft_limit
    );

    BlasThreadEnv::set_single_threaded();

    let pool = build_forgeann_rayon_pool(pool_threads)?;

    let point_pipeline_config = PointPipelineConfig::from_params(params);
    let point_pipeline_config = point_pipeline_config
        .enabled
        .then_some(point_pipeline_config);
    let leaf_context = LeafTaskContext {
        dataset,
        metric: params.metric(),
        sketches,
        params,
        edge_sink,
        spine_overlay_recorder,
        view_lune_recorder,
        point_pipeline_config: point_pipeline_config.as_ref(),
        detailed_profiling,
        #[cfg(test)]
        observer: None,
    };
    let producer_start = Instant::now();
    let mut rbc_telemetry = RbcPartitionTelemetry::default();
    let (scheduler_profile, scheduler_telemetry) =
        run_scheduler_scope(&pool, scheduler_budget, leaf_context, &pb, |emitter| {
            let all_indices: Vec<u32> = (0..num_points as u32).collect();
            let mut rng = StdRng::seed_from_u64(params.random_seed);
            rbc_telemetry = rbc_partition_streaming_with_dirs(
                dataset,
                adsampling_assignment_dataset,
                &all_indices,
                params.metric(),
                params,
                pool_threads as u32,
                &mut rng,
                Some(artifact_dirs.partition_dir.as_path()),
                Some(artifact_dirs.vector_dir.as_path()),
                emitter,
            )?;
            Ok(())
        })?;
    let producer_wall = producer_start.elapsed();

    pb.finish_with_message("ForgeANN streaming processing done");
    info!(
        "ForgeANN streaming (RBC+Leaf) done: elapsed={:?}",
        t_rbc_leaf.elapsed()
    );
    info!(
        "ForgeANN leaf scratch estimate: max_leaf_size={} per_worker_bytes={} active_scratch_bytes_estimate={}",
        params.kernel_safe_leaf_size(),
        leaf_scratch_bytes,
        leaf_scratch_bytes.saturating_mul(leaf_budget_threads),
    );
    info!(
        "ForgeANN streaming thread model: pool_threads={} leaf_budget_threads={}",
        pool_threads, leaf_budget_threads
    );
    if scheduler_profile.lune_witness_records > 0 {
        info!(
            "ForgeANN ViewLune leaf witness emit: elapsed_ms={:.3} records={}",
            scheduler_profile.lune_witness_emit.as_secs_f64() * 1000.0,
            scheduler_profile.lune_witness_records
        );
    }

    Ok(StreamingProfile {
        producer_wall,
        total_wall: t_rbc_leaf.elapsed(),
        leaf: scheduler_profile,
        telemetry: scheduler_telemetry,
        rbc: rbc_telemetry,
    })
}

fn prepare_oom_artifact_dirs(params: &ForgeANNParams) -> AnnResult<OomArtifactDirs> {
    let root_dir = if params.oom_temp_dir.as_os_str().is_empty() {
        std::env::temp_dir().join("forgeann-forgeann-oom")
    } else {
        params.oom_temp_dir.clone()
    };
    let sketch_dir = resolve_oom_artifact_dir(&root_dir, &params.oom_sketch_temp_dir);
    let spill_dir = resolve_oom_artifact_dir(&root_dir, &params.oom_spill_temp_dir);
    let partition_dir = resolve_oom_artifact_dir(&root_dir, &params.oom_partition_temp_dir);
    let vector_dir = resolve_oom_artifact_dir(&partition_dir, &params.oom_vector_temp_dir);
    let dirs = OomArtifactDirs {
        root_dir,
        sketch_dir,
        spill_dir,
        partition_dir,
        vector_dir,
    };
    for dir in dirs.unique_dirs() {
        fs::create_dir_all(dir)?;
    }
    Ok(dirs)
}

fn resolve_oom_artifact_dir(default_dir: &Path, override_dir: &Path) -> PathBuf {
    if override_dir.as_os_str().is_empty() {
        default_dir.to_path_buf()
    } else {
        override_dir.to_path_buf()
    }
}

fn cleanup_oom_artifacts(dir: &Path) -> AnnResult<()> {
    if !dir.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && is_oom_artifact_file(&path) {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

fn is_oom_artifact_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    name == "sketches.bin"
        || name == "candidate.run"
        || name == "candidate_segments.manifest"
        || (name.starts_with("candidate_shard") && name.ends_with(".log"))
        || parse_partition_depth(name).is_some()
}

fn cleanup_oom_artifacts_in_dirs(dirs: &OomArtifactDirs) -> AnnResult<()> {
    for dir in dirs.unique_dirs() {
        cleanup_oom_artifacts(dir)?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct OomProfile {
    num_points: usize,
    dim: usize,
    point_store_backend: &'static str,
    memory_budget_bytes: usize,
    effective_threads: usize,
    pool_threads: usize,
    leaf_budget_threads: usize,
    reduce_shard_points: usize,
    sketch_bytes: usize,
    sketch_file_bytes: usize,
    spill: SpillRunStats,
    reduce: SpillReduceStats,
    sketch_persist_wall: Duration,
    streaming_wall: Duration,
    reduce_wall: Duration,
    emit_wall: Duration,
    total_wall: Duration,
    scheduler: SchedulerTelemetry,
    rbc: RbcPartitionTelemetry,
    leaf: LeafProfile,
    memory: OomMemoryReport,
    temp_files: TempFileReport,
    uring_batches: u64,
    uring_windows: u64,
    uring_bytes: u64,
    uring_fallbacks: u64,
}

#[derive(Clone, Debug, Default)]
struct OomMemoryReport {
    rss_kb_samples: Vec<usize>,
    rss_hwm_kb: Option<usize>,
    vm_peak_kb: Option<usize>,
    rss_by_stage: BTreeMap<String, ProcessMemorySnapshot>,
}

const OOM_MEMORY_STAGE_NAMES: [&str; 6] = [
    "start",
    "after_sketches",
    "after_root",
    "after_streaming",
    "after_reduce",
    "final",
];

impl OomMemoryReport {
    fn capture_stage(&mut self, stage: &str) {
        self.rss_by_stage
            .insert(stage.to_string(), read_process_memory_snapshot());
    }

    fn refresh_summary_from_stages(&mut self) {
        for stage in OOM_MEMORY_STAGE_NAMES {
            self.rss_by_stage.entry(stage.to_string()).or_default();
        }
        self.rss_kb_samples = self
            .rss_by_stage
            .values()
            .filter_map(|snapshot| snapshot.rss_bytes.map(|bytes| bytes / 1024))
            .collect();
        self.rss_hwm_kb = self
            .rss_by_stage
            .values()
            .filter_map(|snapshot| snapshot.hwm_bytes.map(|bytes| bytes / 1024))
            .max();
        self.vm_peak_kb = self
            .rss_by_stage
            .values()
            .filter_map(|snapshot| snapshot.vm_peak_bytes.map(|bytes| bytes / 1024))
            .max();
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FileCategoryStats {
    files: usize,
    bytes: u64,
}

#[derive(Clone, Debug, Default)]
struct TempFileReport {
    total: FileCategoryStats,
    candidate: FileCategoryStats,
    partition_depth0: FileCategoryStats,
    partition_depth1: FileCategoryStats,
    partition_depth2: FileCategoryStats,
    partition_depth3: FileCategoryStats,
    other: FileCategoryStats,
}

fn leaf_size_bucket_profile_json(profile: &LeafProfile) -> String {
    let mut entries = Vec::with_capacity(LEAF_SIZE_BUCKETS);
    for bucket_idx in 0..LEAF_SIZE_BUCKETS {
        let bucket = profile.leaf_size_buckets[bucket_idx];
        let exact_ms_per_row = if bucket.exact_rows == 0 {
            0.0
        } else {
            bucket.exact_wall.as_secs_f64() * 1000.0 / bucket.exact_rows as f64
        };
        let adsampling_ms_per_row = if bucket.adsampling_rows == 0 {
            0.0
        } else {
            bucket.adsampling_wall.as_secs_f64() * 1000.0 / bucket.adsampling_rows as f64
        };
        entries.push(format!(
            "{{ \"label\": \"{}\", \"leaves\": {}, \"rows\": {}, \"exact_leaves\": {}, \"exact_rows\": {}, \"exact_wall_ms\": {}, \"exact_ms_per_row\": {:.6}, \"exact_distance_ms\": {}, \"exact_topk_ms\": {}, \"adsampling_leaves\": {}, \"adsampling_rows\": {}, \"adsampling_wall_ms\": {}, \"adsampling_ms_per_row\": {:.6}, \"adsampling_layout_ms\": {}, \"adsampling_seed_ms\": {}, \"adsampling_scan_ms\": {}, \"adsampling_seed_evals\": {}, \"adsampling_full_evals\": {}, \"adsampling_pruned_evals\": {}, \"adsampling_group_evals\": {}, \"adsampling_simd_group_calls\": {}, \"adsampling_simd_active_lane_evals\": {}, \"adsampling_scalar_group_evals\": {} }}",
            leaf_size_bucket_label(bucket_idx),
            bucket.leaves,
            bucket.rows,
            bucket.exact_leaves,
            bucket.exact_rows,
            bucket.exact_wall.as_millis(),
            exact_ms_per_row,
            bucket.exact_distance.as_millis(),
            bucket.exact_topk.as_millis(),
            bucket.adsampling_leaves,
            bucket.adsampling_rows,
            bucket.adsampling_wall.as_millis(),
            adsampling_ms_per_row,
            bucket.adsampling_layout.as_millis(),
            bucket.adsampling_seed.as_millis(),
            bucket.adsampling_scan.as_millis(),
            bucket.adsampling_seed_evals,
            bucket.adsampling_full_evals,
            bucket.adsampling_pruned_evals,
            bucket.adsampling_group_evals,
            bucket.adsampling_simd_group_calls,
            bucket.adsampling_simd_active_lane_evals,
            bucket.adsampling_scalar_group_evals,
        ));
    }
    entries.join(", ")
}

fn assignment_decision_profile_json(profile: &AssignmentDecisionProfile) -> String {
    let mut depth_entries = Vec::with_capacity(ASSIGNMENT_DEPTH_BUCKETS);
    for depth_bucket in 0..ASSIGNMENT_DEPTH_BUCKETS {
        let mut leader_entries = Vec::with_capacity(ASSIGNMENT_LEADER_BUCKETS);
        for leader_bucket in 0..ASSIGNMENT_LEADER_BUCKETS {
            let bucket = profile.depth_leader_buckets[depth_bucket][leader_bucket];
            let adsampling_avg_recall = if bucket.adsampling_recall_samples == 0 {
                0.0
            } else {
                bucket.adsampling_recall_sum / bucket.adsampling_recall_samples as f64
            };
            let gemm_ms_per_work_m = if bucket.work == 0 || bucket.gemm_nodes == 0 {
                0.0
            } else {
                bucket.gemm_wall.as_secs_f64() * 1000.0 / (bucket.work as f64 / 1_000_000.0)
            };
            let adsampling_ms_per_work_m = if bucket.work == 0 || bucket.adsampling_nodes == 0 {
                0.0
            } else {
                bucket.adsampling_wall.as_secs_f64() * 1000.0 / (bucket.work as f64 / 1_000_000.0)
            };
            leader_entries.push(format!(
                "{{ \"label\": \"{}\", \"nodes\": {}, \"points\": {}, \"work\": {}, \"max_points\": {}, \"max_leaders\": {}, \"max_work\": {}, \"gemm_nodes\": {}, \"gemm_points\": {}, \"gemm_wall_ms\": {}, \"gemm_ms_per_work_m\": {:.6}, \"adsampling_nodes\": {}, \"adsampling_points\": {}, \"adsampling_wall_ms\": {}, \"adsampling_ms_per_work_m\": {:.6}, \"adsampling_avg_recall_at_fanout\": {:.6}, \"adsampling_recall_samples\": {}, \"adsampling_mismatches\": {} }}",
                assignment_leader_bucket_label(leader_bucket),
                bucket.nodes,
                bucket.points,
                bucket.work,
                bucket.max_points,
                bucket.max_leaders,
                bucket.max_work,
                bucket.gemm_nodes,
                bucket.gemm_points,
                bucket.gemm_wall.as_millis(),
                gemm_ms_per_work_m,
                bucket.adsampling_nodes,
                bucket.adsampling_points,
                bucket.adsampling_wall.as_millis(),
                adsampling_ms_per_work_m,
                adsampling_avg_recall,
                bucket.adsampling_recall_samples,
                bucket.adsampling_mismatches,
            ));
        }
        depth_entries.push(format!(
            "{{ \"label\": \"{}\", \"leader_buckets\": [{}] }}",
            assignment_depth_bucket_label(depth_bucket),
            leader_entries.join(", ")
        ));
    }

    let mut fallback_entries = Vec::with_capacity(ASSIGNMENT_FALLBACK_REASONS);
    for idx in 0..ASSIGNMENT_FALLBACK_REASONS {
        fallback_entries.push(format!(
            "\"{}\": {}",
            assignment_fallback_reason_label(idx),
            profile.fallback_reasons[idx]
        ));
    }

    format!(
        "{{ \"depth_leader_buckets\": [{}], \"fallback_reasons\": {{ {} }} }}",
        depth_entries.join(", "),
        fallback_entries.join(", ")
    )
}

fn memory_stage_profile_json(report: &OomMemoryReport) -> String {
    let entries = OOM_MEMORY_STAGE_NAMES
        .iter()
        .map(|stage| {
            let snapshot = report.rss_by_stage.get(*stage).copied().unwrap_or_default();
            format!(
                "\"{}\": {{ \"rss_kb\": {}, \"rss_hwm_kb\": {}, \"vm_peak_kb\": {} }}",
                stage,
                format_optional_kb(snapshot.rss_bytes),
                format_optional_kb(snapshot.hwm_bytes),
                format_optional_kb(snapshot.vm_peak_bytes),
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ {entries} }}")
}

fn validate_oom_profile(profile: &OomProfile) -> AnnResult<()> {
    if profile.rbc.io_planned_forgeann.enabled
        && !profile.rbc.io_planned_forgeann.dry_run
        && profile.rbc.io_planned_forgeann.planned_nodes > 0
        && profile.rbc.io_planned_forgeann.io_pain_by_depth.is_empty()
    {
        return Err(AnnError::log_index_config_error(
            "io_pain_by_depth".to_string(),
            "actual io-planned ForgeANN profile has empty io_pain_by_depth".to_string(),
        ));
    }
    Ok(())
}

fn write_oom_profile(path: &Path, profile: &OomProfile) -> AnnResult<()> {
    validate_oom_profile(profile)?;
    let mut file = fs::File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"num_points\": {},", profile.num_points)?;
    writeln!(file, "  \"dim\": {},", profile.dim)?;
    writeln!(
        file,
        "  \"point_store_backend\": \"{}\",",
        profile.point_store_backend
    )?;
    writeln!(
        file,
        "  \"memory_budget_bytes\": {},",
        profile.memory_budget_bytes
    )?;
    writeln!(
        file,
        "  \"effective_threads\": {},",
        profile.effective_threads
    )?;
    writeln!(file, "  \"pool_threads\": {},", profile.pool_threads)?;
    writeln!(
        file,
        "  \"leaf_budget_threads\": {},",
        profile.leaf_budget_threads
    )?;
    writeln!(
        file,
        "  \"reduce_shard_points\": {},",
        profile.reduce_shard_points
    )?;
    writeln!(file, "  \"sketch_bytes\": {},", profile.sketch_bytes)?;
    writeln!(
        file,
        "  \"sketch_file_bytes\": {},",
        profile.sketch_file_bytes
    )?;
    writeln!(
        file,
        "  \"spill\": {{ \"part_files\": {}, \"segment_count\": {}, \"shard_groups\": {}, \"max_parts_per_shard\": {}, \"max_segments_per_shard\": {}, \"record_bytes\": {}, \"manifest_bytes\": {} }},",
        profile.spill.part_files,
        profile.spill.segment_count,
        profile.spill.shard_groups,
        profile.spill.max_parts_per_shard,
        profile.spill.max_segments_per_shard,
        profile.spill.record_bytes,
        profile.spill.manifest_bytes,
    )?;
    writeln!(
        file,
        "  \"reduce\": {{ \"shard_groups\": {}, \"segments_scanned\": {}, \"part_files_opened\": {}, \"max_segments_in_group\": {} }},",
        profile.reduce.shard_groups,
        profile.reduce.segments_scanned,
        profile.reduce.part_files_opened,
        profile.reduce.max_segments_in_group,
    )?;
    writeln!(
        file,
        "  \"stage_ms\": {{ \"sketch_persist\": {}, \"streaming\": {}, \"reduce\": {}, \"emit\": {}, \"total\": {} }},",
        profile.sketch_persist_wall.as_millis(),
        profile.streaming_wall.as_millis(),
        profile.reduce_wall.as_millis(),
        profile.emit_wall.as_millis(),
        profile.total_wall.as_millis(),
    )?;
    let rss_by_stage_json = memory_stage_profile_json(&profile.memory);
    writeln!(
        file,
        "  \"memory\": {{ \"rss_kb_samples\": [{}], \"rss_hwm_kb\": {}, \"vm_peak_kb\": {}, \"rss_by_stage\": {} }},",
        profile
            .memory
            .rss_kb_samples
            .iter()
            .map(|sample| sample.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        format_optional_number(profile.memory.rss_hwm_kb),
        format_optional_number(profile.memory.vm_peak_kb),
        rss_by_stage_json,
    )?;
    writeln!(
        file,
        "  \"temp_files\": {{ \"total\": {{ \"files\": {}, \"bytes\": {} }}, \"candidate\": {{ \"files\": {}, \"bytes\": {} }}, \"partition_depth0\": {{ \"files\": {}, \"bytes\": {} }}, \"partition_depth1\": {{ \"files\": {}, \"bytes\": {} }}, \"partition_depth2\": {{ \"files\": {}, \"bytes\": {} }}, \"partition_depth3\": {{ \"files\": {}, \"bytes\": {} }}, \"other\": {{ \"files\": {}, \"bytes\": {} }} }},",
        profile.temp_files.total.files,
        profile.temp_files.total.bytes,
        profile.temp_files.candidate.files,
        profile.temp_files.candidate.bytes,
        profile.temp_files.partition_depth0.files,
        profile.temp_files.partition_depth0.bytes,
        profile.temp_files.partition_depth1.files,
        profile.temp_files.partition_depth1.bytes,
        profile.temp_files.partition_depth2.files,
        profile.temp_files.partition_depth2.bytes,
        profile.temp_files.partition_depth3.files,
        profile.temp_files.partition_depth3.bytes,
        profile.temp_files.other.files,
        profile.temp_files.other.bytes,
    )?;
    writeln!(
        file,
        "  \"io_uring\": {{ \"point_gather_batches\": {}, \"point_gather_windows\": {}, \"point_gather_bytes\": {}, \"point_gather_fallbacks\": {} }},",
        profile.uring_batches, profile.uring_windows, profile.uring_bytes, profile.uring_fallbacks,
    )?;
    writeln!(
        file,
        "  \"scheduler\": {{ \"peak_inflight_leaf_tasks\": {}, \"peak_leaf_backlog\": {}, \"leaf_backlog\": {}, \"producer_leaf_backlog_soft_limit\": {}, \"outstanding_leaf_tasks\": {}, \"inflight_leaf_tasks\": {}, \"active_leaf_drainers\": {}, \"leaf_drainer_limit\": {}, \"inline_leaf_fallbacks\": {}, \"producer_help_drains\": {}, \"producer_help_drain_ms\": {}, \"backlog_full_help_drains\": {}, \"backlog_full_help_ms\": {}, \"producer_help_yields\": {}, \"large_leaf_count\": {}, \"large_leaf_block_tasks\": {}, \"active_large_assignment_peak\": {}, \"large_assignment_guard_count\": {}, \"leaf_cap_large_assignment_hits\": {}, \"leaf_cap_producer_hits\": {}, \"leaf_cap_done_hits\": {}, \"batched_leaf_drains\": {}, \"batched_leaf_leaves\": {}, \"batched_leaf_points\": {}, \"batched_leaf_unique_points\": {}, \"batched_leaf_load_ms\": {}, \"leaf_batch_hydration_limit\": {}, \"leaf_batch_post_producer_scale\": {}, \"peak_leaf_batch_leaves\": {}, \"peak_leaf_batch_points\": {}, \"peak_leaf_batch_unique_points\": {}, \"active_leaf_batch_hydrations\": {}, \"peak_leaf_batch_hydrations\": {}, \"leaf_batch_hydration_wait_ms\": {}, \"work_graph_enabled\": {}, \"queued_work_ms_peak\": {}, \"queued_mem_bytes_peak\": {}, \"replay_admitted_runs\": {}, \"replay_paused_ms\": {}, \"worker_busy_ms\": {}, \"worker_idle_ms\": {} }},",
        profile.scheduler.peak_inflight_leaf_tasks,
        profile.scheduler.peak_leaf_backlog,
        profile.scheduler.leaf_backlog,
        profile.scheduler.producer_leaf_backlog_soft_limit,
        profile.scheduler.outstanding_leaf_tasks,
        profile.scheduler.inflight_leaf_tasks,
        profile.scheduler.active_leaf_drainers,
        profile.scheduler.leaf_drainer_limit,
        profile.scheduler.inline_leaf_fallbacks,
        profile.scheduler.producer_help_drains,
        profile.scheduler.producer_help_drain_ms,
        profile.scheduler.backlog_full_help_drains,
        profile.scheduler.backlog_full_help_ms,
        profile.scheduler.producer_help_yields,
        profile.scheduler.large_leaf_count,
        profile.scheduler.large_leaf_block_tasks,
        profile.scheduler.active_large_assignment_peak,
        profile.scheduler.large_assignment_guard_count,
        profile.scheduler.leaf_cap_large_assignment_hits,
        profile.scheduler.leaf_cap_producer_hits,
        profile.scheduler.leaf_cap_done_hits,
        profile.scheduler.batched_leaf_drains,
        profile.scheduler.batched_leaf_leaves,
        profile.scheduler.batched_leaf_points,
        profile.scheduler.batched_leaf_unique_points,
        profile.scheduler.batched_leaf_load_ms,
        profile.scheduler.leaf_batch_hydration_limit,
        profile.scheduler.leaf_batch_post_producer_scale,
        profile.scheduler.peak_leaf_batch_leaves,
        profile.scheduler.peak_leaf_batch_points,
        profile.scheduler.peak_leaf_batch_unique_points,
        profile.scheduler.active_leaf_batch_hydrations,
        profile.scheduler.peak_leaf_batch_hydrations,
        profile.scheduler.leaf_batch_hydration_wait_ms,
        profile.scheduler.work_graph_enabled,
        profile.scheduler.queued_work_ms_peak,
        profile.scheduler.queued_mem_bytes_peak,
        profile.scheduler.replay_admitted_runs,
        profile.scheduler.replay_paused_ms,
        profile.scheduler.worker_busy_ms,
        profile.scheduler.worker_idle_ms,
    )?;
    let root_json = root_fanout_profile_json(&profile.rbc.root_fanout);
    let root_body = root_json
        .trim()
        .strip_prefix("{")
        .and_then(|value| value.strip_suffix("}"))
        .map(str::trim)
        .unwrap_or("\"root_fanout\": null");
    writeln!(file, "  {root_body},")?;
    writeln!(
        file,
        "  \"assignment_decisions\": {},",
        assignment_decision_profile_json(&profile.rbc.assignment_decisions),
    )?;
    writeln!(
        file,
        "  \"ads_scheduler\": {{ \"ads_large_tasks\": {}, \"ads_chunks_total\": {}, \"ads_waves\": {}, \"ads_exact_fallback_small\": {}, \"ads_exact_fallback_nested\": {}, \"ads_parallelism_collapse_count\": {} }},",
        profile.rbc.ads_scheduler.ads_large_tasks,
        profile.rbc.ads_scheduler.ads_chunks_total,
        profile.rbc.ads_scheduler.ads_waves,
        profile.rbc.ads_scheduler.ads_exact_fallback_small,
        profile.rbc.ads_scheduler.ads_exact_fallback_nested,
        profile.rbc.ads_scheduler.ads_parallelism_collapse_count,
    )?;
    writeln!(
        file,
        "  \"io_planned_forgeann\": {},",
        profile.rbc.io_planned_forgeann.profile_json(),
    )?;
    writeln!(
        file,
        "  \"point_pipeline\": {{ \"batches\": {}, \"points\": {}, \"requested_rows\": {}, \"physical_rows\": {}, \"planned_windows\": {}, \"logical_bytes\": {}, \"physical_bytes\": {}, \"avg_rows_per_window\": {:.3}, \"avg_read_size_bytes\": {:.3}, \"read_amplification\": {:.3}, \"direct_read_calls_avoided\": {}, \"read_wall_ms\": {}, \"consumer_wait_ms\": {}, \"producer_wait_ms\": {}, \"budget_wait_ms\": {}, \"ready_queue_depth_peak\": {}, \"permit_peak_bytes\": {}, \"window_cache\": {{ \"hits\": {}, \"misses\": {}, \"inserts\": {}, \"evictions\": {}, \"used_peak_bytes\": {}, \"saved_direct_read_calls\": {}, \"logical_bytes\": {}, \"physical_bytes\": {} }} }},",
        profile.rbc.point_pipeline.batches,
        profile.rbc.point_pipeline.points,
        profile.rbc.point_pipeline.requested_rows,
        profile.rbc.point_pipeline.physical_rows,
        profile.rbc.point_pipeline.planned_windows,
        profile.rbc.point_pipeline.logical_bytes,
        profile.rbc.point_pipeline.physical_bytes,
        profile.rbc.point_pipeline.avg_rows_per_window(),
        profile.rbc.point_pipeline.avg_read_size_bytes(),
        profile.rbc.point_pipeline.read_amplification(),
        profile.rbc.point_pipeline.direct_read_calls_avoided,
        profile.rbc.point_pipeline.read_wall.as_millis(),
        profile.rbc.point_pipeline.consumer_wait.as_millis(),
        profile.rbc.point_pipeline.producer_wait.as_millis(),
        profile.rbc.point_pipeline.budget_wait.as_millis(),
        profile.rbc.point_pipeline.ready_queue_depth_peak,
        profile.rbc.point_pipeline.permit_peak_bytes,
        profile.rbc.point_pipeline.window_cache.hits,
        profile.rbc.point_pipeline.window_cache.misses,
        profile.rbc.point_pipeline.window_cache.inserts,
        profile.rbc.point_pipeline.window_cache.evictions,
        profile.rbc.point_pipeline.window_cache.used_peak_bytes,
        profile
            .rbc
            .point_pipeline
            .window_cache
            .saved_direct_read_calls,
        profile.rbc.point_pipeline.window_cache.logical_bytes,
        profile.rbc.point_pipeline.window_cache.physical_bytes,
    )?;
    let leaf_size_buckets_json = leaf_size_bucket_profile_json(&profile.leaf);
    let leaf_ads_tile_rows_avg = if profile.leaf.leaf_ads_tiles == 0 {
        0.0
    } else {
        profile.leaf.leaf_ads_tile_rows_total as f64 / profile.leaf.leaf_ads_tiles as f64
    };
    let leaf_ads_ewma_json = profile
        .leaf
        .leaf_ads_ewma_ns_per_row_by_bucket
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        file,
        "  \"leaf\": {{ \"leaves\": {}, \"points\": {}, \"blockwise_leaves\": {}, \"total_wall_ms\": {}, \"load_ms\": {}, \"distance_ms\": {}, \"topk_ms\": {}, \"hash_ms\": {}, \"flush_ms\": {}, \"leaf_adsampling_leaves\": {}, \"leaf_adsampling_rows\": {}, \"leaf_adsampling_layout_ms\": {}, \"leaf_adsampling_seed_ms\": {}, \"leaf_adsampling_scan_ms\": {}, \"leaf_adsampling_seed_evals\": {}, \"leaf_adsampling_full_evals\": {}, \"leaf_adsampling_pruned_evals\": {}, \"leaf_adsampling_group_evals\": {}, \"leaf_adsampling_simd_group_calls\": {}, \"leaf_adsampling_simd_active_lane_evals\": {}, \"leaf_adsampling_scalar_group_evals\": {}, \"leaf_ads_tiling_enabled\": {}, \"leaf_ads_wavefront_pairmask_enabled\": {}, \"leaf_ads_work_graph_enabled\": {}, \"leaf_ads_tiled_leaves\": {}, \"leaf_ads_tiled_rows\": {}, \"leaf_ads_tiles\": {}, \"leaf_ads_tile_rows_min\": {}, \"leaf_ads_tile_rows_max\": {}, \"leaf_ads_tile_rows_avg\": {:.3}, \"leaf_ads_tile_wall_ms\": {}, \"leaf_ads_tile_wait_ms\": {}, \"leaf_ads_handle_requeues\": {}, \"leaf_ads_cpu_budget\": {}, \"leaf_ads_active_context_peak\": {}, \"leaf_ads_active_workers_peak\": {}, \"leaf_ads_ewma_ns_per_row_by_bucket\": [{}], \"size_buckets\": [{}] }}",
        profile.leaf.leaves,
        profile.leaf.points,
        profile.leaf.blockwise_leaves,
        profile.leaf.total_wall.as_millis(),
        profile.leaf.load.as_millis(),
        profile.leaf.distance.as_millis(),
        profile.leaf.topk.as_millis(),
        profile.leaf.hash.as_millis(),
        profile.leaf.flush.as_millis(),
        profile.leaf.leaf_adsampling_leaves,
        profile.leaf.leaf_adsampling_rows,
        profile.leaf.leaf_adsampling_layout.as_millis(),
        profile.leaf.leaf_adsampling_seed.as_millis(),
        profile.leaf.leaf_adsampling_scan.as_millis(),
        profile.leaf.leaf_adsampling_seed_evals,
        profile.leaf.leaf_adsampling_full_evals,
        profile.leaf.leaf_adsampling_pruned_evals,
        profile.leaf.leaf_adsampling_group_evals,
        profile.leaf.leaf_adsampling_simd_group_calls,
        profile.leaf.leaf_adsampling_simd_active_lane_evals,
        profile.leaf.leaf_adsampling_scalar_group_evals,
        profile.leaf.leaf_ads_tiling_enabled,
        profile.leaf.leaf_ads_wavefront_pairmask_enabled,
        profile.leaf.leaf_ads_work_graph_enabled,
        profile.leaf.leaf_ads_tiled_leaves,
        profile.leaf.leaf_ads_tiled_rows,
        profile.leaf.leaf_ads_tiles,
        profile.leaf.leaf_ads_tile_rows_min,
        profile.leaf.leaf_ads_tile_rows_max,
        leaf_ads_tile_rows_avg,
        profile.leaf.leaf_ads_tile_wall.as_millis(),
        profile.leaf.leaf_ads_tile_wait.as_millis(),
        profile.leaf.leaf_ads_handle_requeues,
        profile.leaf.leaf_ads_cpu_budget,
        profile.leaf.leaf_ads_active_context_peak,
        profile.leaf.leaf_ads_active_workers_peak,
        leaf_ads_ewma_json,
        leaf_size_buckets_json,
    )?;
    writeln!(file, "}}")?;
    Ok(())
}

fn detailed_forgeann_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FORGEANN_PROFILE")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                matches!(
                    value.as_str(),
                    "1" | "true" | "yes" | "on" | "detail" | "detailed"
                )
            })
            .unwrap_or(false)
    })
}

fn apply_immediate_leaf_drain_budget(
    budget: SchedulerBudget,
    worker_count: usize,
) -> SchedulerBudget {
    let _ = worker_count;
    budget
}

const DEFAULT_FORGEANN_RAYON_STACK_MB: usize = 32;
const MIN_FORGEANN_RAYON_STACK_MB: usize = 8;
const MAX_FORGEANN_RAYON_STACK_MB: usize = 256;
const FORGEANN_RAYON_STACK_MB_ENV: &str = "FORGEANN_RAYON_STACK_MB";

fn normalize_forgeann_rayon_stack_mb(configured_mb: Option<usize>) -> usize {
    configured_mb
        .unwrap_or(DEFAULT_FORGEANN_RAYON_STACK_MB)
        .clamp(MIN_FORGEANN_RAYON_STACK_MB, MAX_FORGEANN_RAYON_STACK_MB)
}

fn forgeann_rayon_worker_stack_bytes() -> usize {
    let configured_mb = std::env::var(FORGEANN_RAYON_STACK_MB_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok());
    normalize_forgeann_rayon_stack_mb(configured_mb) * 1024 * 1024
}

fn build_forgeann_rayon_pool(num_threads: usize) -> AnnResult<rayon::ThreadPool> {
    let stack_bytes = forgeann_rayon_worker_stack_bytes();
    info!(
        "ForgeANN Rayon pool: threads={} worker_stack_bytes={}",
        num_threads, stack_bytes
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .stack_size(stack_bytes)
        .build()
        .map_err(|e| {
            AnnError::log_index_error(format!(
                "Failed to create Rayon pool with worker_stack_bytes={}: {}",
                stack_bytes, e
            ))
        })
}

#[derive(Clone, Debug, Default)]
struct StreamingProfile {
    producer_wall: Duration,
    total_wall: Duration,
    leaf: LeafProfile,
    telemetry: SchedulerTelemetry,
    rbc: RbcPartitionTelemetry,
}

impl StreamingProfile {
    fn leaf_idle_time(&self) -> Duration {
        self.total_wall.saturating_sub(self.leaf.total_wall)
    }

    fn leaf_parallelism(&self) -> f64 {
        if self.total_wall.is_zero() {
            0.0
        } else {
            self.leaf.total_wall.as_secs_f64() / self.total_wall.as_secs_f64()
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BlasThreadEnv {
    values: HashMap<&'static str, Option<String>>,
}

impl BlasThreadEnv {
    const KEYS: [&'static str; 6] = [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "MKL_DYNAMIC",
    ];

    fn capture() -> Self {
        let mut values = HashMap::with_capacity(Self::KEYS.len());
        for key in Self::KEYS {
            values.insert(key, std::env::var(key).ok());
        }
        Self { values }
    }

    fn set_single_threaded() {
        for key in [
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "GOTO_NUM_THREADS",
        ] {
            unsafe {
                std::env::set_var(key, "1");
            }
        }
        unsafe {
            std::env::set_var("MKL_DYNAMIC", "0");
        }
    }

    fn restore(&self) {
        for key in Self::KEYS {
            match self.values.get(key).and_then(|value| value.as_ref()) {
                Some(value) => unsafe {
                    std::env::set_var(key, value);
                },
                None => unsafe {
                    std::env::remove_var(key);
                },
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProcessMemorySnapshot {
    rss_bytes: Option<usize>,
    hwm_bytes: Option<usize>,
    vm_peak_bytes: Option<usize>,
}

fn log_memory_telemetry(
    phase: &str,
    num_points: usize,
    params: &ForgeANNParams,
    dim: usize,
    sketch_bytes: usize,
    reservoir_bytes: usize,
    worker_count: usize,
) {
    let snapshot = read_process_memory_snapshot();
    let per_worker_scratch =
        estimate_leaf_scratch_bytes(params.kernel_safe_leaf_size(), dim, params.leaf_knn);
    let active_scratch_bytes = per_worker_scratch.saturating_mul(worker_count.max(1));

    info!(
        "ForgeANN memory telemetry [{}]: points={} rss_bytes={} vm_hwm_bytes={} sketch_bytes={} reservoir_bytes={} active_scratch_bytes_estimate={} max_full_matrix_leaf_size={}",
        phase,
        num_points,
        format_optional_bytes(snapshot.rss_bytes),
        format_optional_bytes(snapshot.hwm_bytes),
        sketch_bytes,
        reservoir_bytes,
        active_scratch_bytes,
        params.kernel_safe_leaf_size(),
    );
}

#[cfg(target_os = "linux")]
fn trim_process_allocator(label: &str) {
    let result = unsafe { libc::malloc_trim(0) };
    tracing::debug!(
        "ForgeANN allocator trim [{}]: malloc_trim_result={}",
        label,
        result
    );
}

#[cfg(not(target_os = "linux"))]
fn trim_process_allocator(_label: &str) {}

fn read_process_memory_snapshot() -> ProcessMemorySnapshot {
    #[cfg(target_os = "linux")]
    {
        return parse_proc_self_status()
            .map(
                |(rss_bytes, hwm_bytes, vm_peak_bytes)| ProcessMemorySnapshot {
                    rss_bytes,
                    hwm_bytes,
                    vm_peak_bytes,
                },
            )
            .unwrap_or_default();
    }

    #[cfg(not(target_os = "linux"))]
    {
        ProcessMemorySnapshot::default()
    }
}

#[cfg(target_os = "linux")]
fn parse_proc_self_status() -> Option<(Option<usize>, Option<usize>, Option<usize>)> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let mut rss_bytes = None;
    let mut hwm_bytes = None;
    let mut vm_peak_bytes = None;

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            rss_bytes = parse_status_kib_field(line);
        } else if line.starts_with("VmHWM:") {
            hwm_bytes = parse_status_kib_field(line);
        } else if line.starts_with("VmPeak:") {
            vm_peak_bytes = parse_status_kib_field(line);
        }
    }

    Some((rss_bytes, hwm_bytes, vm_peak_bytes))
}

#[cfg(target_os = "linux")]
fn parse_status_kib_field(line: &str) -> Option<usize> {
    let value_kib = line.split_whitespace().nth(1)?.parse::<usize>().ok()?;
    value_kib.checked_mul(1024)
}

fn format_optional_bytes(value: Option<usize>) -> String {
    value
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_number(value: Option<usize>) -> String {
    value
        .map(|number| number.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn format_optional_kb(value: Option<usize>) -> String {
    value
        .map(|bytes| (bytes / 1024).to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn collect_temp_file_report(dir: &Path) -> AnnResult<TempFileReport> {
    let mut report = TempFileReport::default();
    if !dir.exists() {
        return Ok(report);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let bytes = entry.metadata()?.len();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        record_file_category(&mut report.total, bytes);
        if (name.starts_with("candidate_shard") && name.ends_with(".log"))
            || name == "candidate_segments.manifest"
        {
            record_file_category(&mut report.candidate, bytes);
        } else if let Some(depth) = parse_partition_depth(name) {
            match depth {
                0 => record_file_category(&mut report.partition_depth0, bytes),
                1 => record_file_category(&mut report.partition_depth1, bytes),
                2 => record_file_category(&mut report.partition_depth2, bytes),
                3 => record_file_category(&mut report.partition_depth3, bytes),
                _ => record_file_category(&mut report.other, bytes),
            }
        } else {
            record_file_category(&mut report.other, bytes);
        }
    }

    Ok(report)
}

fn collect_oom_temp_file_report(dirs: &OomArtifactDirs) -> AnnResult<TempFileReport> {
    let mut report = TempFileReport::default();
    for dir in dirs.unique_dirs() {
        merge_temp_file_report(&mut report, collect_temp_file_report(dir)?);
    }
    Ok(report)
}

fn merge_temp_file_report(report: &mut TempFileReport, other: TempFileReport) {
    merge_file_category(&mut report.total, other.total);
    merge_file_category(&mut report.candidate, other.candidate);
    merge_file_category(&mut report.partition_depth0, other.partition_depth0);
    merge_file_category(&mut report.partition_depth1, other.partition_depth1);
    merge_file_category(&mut report.partition_depth2, other.partition_depth2);
    merge_file_category(&mut report.partition_depth3, other.partition_depth3);
    merge_file_category(&mut report.other, other.other);
}

fn merge_file_category(stats: &mut FileCategoryStats, other: FileCategoryStats) {
    stats.files += other.files;
    stats.bytes += other.bytes;
}

fn record_file_category(stats: &mut FileCategoryStats, bytes: u64) {
    stats.files += 1;
    stats.bytes += bytes;
}

fn parse_partition_depth(name: &str) -> Option<usize> {
    let suffix = name.strip_prefix("partition_d")?;
    let depth = suffix.get(0..2)?.parse::<usize>().ok()?;
    Some(depth)
}

fn estimate_sketch_bytes(sketches: &dyn SketchAccessor) -> usize {
    sketches.resident_bytes()
}

fn should_use_full_resident_sketch_cache(
    params: &ForgeANNParams,
    sketch_file_bytes: usize,
) -> bool {
    sketch_file_bytes > 0
        && params.oom_sketch_cache_bytes > 0
        && sketch_file_bytes <= params.oom_sketch_cache_bytes
}

fn derive_spill_shard_points(
    num_points: usize,
    params: &ForgeANNParams,
    memory_budget_bytes: usize,
) -> usize {
    let spill_budget = if params.oom_spill_cache_bytes == 0 {
        (memory_budget_bytes / 8).max(1)
    } else {
        params.oom_spill_cache_bytes.max(1)
    };
    let reservoir_bytes_per_point = std::mem::size_of::<HashPruneReservoir>()
        + params
            .l_max
            .saturating_mul(size_of::<u32>() + size_of::<u16>() + size_of::<u16>());
    let bytes_per_point = reservoir_bytes_per_point.max(1);
    (spill_budget / bytes_per_point).clamp(1, num_points.max(1))
}

fn plan_resident_reservoir_prefix(
    num_points: usize,
    effective_threads: usize,
    params: &ForgeANNParams,
    dim: usize,
    sketch_bytes: usize,
    shard_points: usize,
    reservoir_budget_bytes: usize,
) -> ResidentReservoirPlan {
    let memory_budget_bytes = reservoir_budget_bytes;
    let active_scratch_bytes =
        estimate_leaf_scratch_bytes(params.kernel_safe_leaf_size(), dim, params.leaf_knn)
            .saturating_mul(effective_threads.max(1));
    let fixed_headroom = (memory_budget_bytes / 10).max(512 * 1024 * 1024);
    let shard_points = shard_points.max(1);
    let available_for_reservoirs = memory_budget_bytes
        .saturating_sub(active_scratch_bytes)
        .saturating_sub(sketch_bytes)
        .saturating_sub(fixed_headroom);
    let bytes_per_point = estimate_reservoir_bytes(1, params.l_max).max(1);
    let resident_points_raw = available_for_reservoirs / bytes_per_point;
    let resident_points = if resident_points_raw >= num_points {
        num_points
    } else {
        resident_points_raw
            .checked_div(shard_points)
            .unwrap_or(0)
            .saturating_mul(shard_points)
            .min(num_points)
    };
    ResidentReservoirPlan { resident_points }
}

fn plan_resident_reservoir_prefix_from_cap(
    num_points: usize,
    params: &ForgeANNParams,
    reservoir_cap_bytes: usize,
) -> ResidentReservoirPlan {
    let bytes_per_point = estimate_reservoir_bytes(1, params.l_max).max(1);
    let resident_points = (reservoir_cap_bytes / bytes_per_point).min(num_points);
    ResidentReservoirPlan { resident_points }
}

fn derive_oom_worker_count(requested_threads: usize, params: &ForgeANNParams, dim: usize) -> usize {
    let requested_threads = requested_threads.max(1);
    let memory_budget_bytes = params.effective_oom_memory_budget_bytes();
    let per_worker_bytes =
        estimate_leaf_scratch_bytes(params.kernel_safe_leaf_size(), dim, params.leaf_knn)
            // The scratch estimate already accounts for the large per-leaf matrices. Keep a
            // smaller fixed reserve so OOM mode can admit more concurrent leaf workers when the
            // scratch pool is already being reused aggressively.
            .saturating_add(32 * 1024 * 1024)
            .max(1);
    let shared_reserve_bytes = memory_budget_bytes
        .saturating_mul(2)
        .checked_div(3)
        .unwrap_or(memory_budget_bytes);
    let worker_budget = memory_budget_bytes
        .saturating_sub(shared_reserve_bytes)
        .max(per_worker_bytes);
    let budget_threads = (worker_budget / per_worker_bytes).max(1);
    requested_threads.min(budget_threads.max(1))
}

fn estimate_reservoir_bytes(num_points: usize, l_max: usize) -> usize {
    let slot_bytes = size_of::<u32>() + size_of::<u16>() + size_of::<u16>();
    let reservoir_struct_bytes =
        size_of::<HashPruneReservoir>() + size_of::<Mutex<HashPruneReservoir>>();
    num_points.saturating_mul(reservoir_struct_bytes + l_max.saturating_mul(slot_bytes))
}

fn estimate_leaf_scratch_bytes(max_leaf_size: usize, dim: usize, leaf_knn: usize) -> usize {
    let leaf_size = max_leaf_size.max(1);
    let dmat = leaf_size
        .saturating_mul(leaf_size)
        .saturating_mul(size_of::<f32>());
    let cand = leaf_size.saturating_mul(size_of::<(usize, f32)>());
    let x = leaf_size
        .saturating_mul(dim)
        .saturating_mul(size_of::<f32>());
    let pending_edge_size =
        size_of::<usize>() + size_of::<u32>() + size_of::<u16>() + size_of::<f32>();
    let edges = leaf_size
        .saturating_mul(leaf_knn.max(1).saturating_mul(2))
        .saturating_mul(pending_edge_size);
    dmat.saturating_add(cand)
        .saturating_add(x)
        .saturating_add(edges)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OomExecutionThreads {
    pool_threads: usize,
    leaf_budget_threads: usize,
}

fn derive_oom_execution_threads(
    requested_threads: usize,
    params: &ForgeANNParams,
    dim: usize,
) -> OomExecutionThreads {
    let pool_threads = requested_threads.max(1);
    let leaf_budget_threads = derive_oom_worker_count(pool_threads, params, dim);
    OomExecutionThreads {
        pool_threads,
        leaf_budget_threads,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;
    use std::{env, fs};

    use tempfile::tempdir;

    use super::ForgeANNParams;
    use crate::model::InmemDataset;

    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

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

    fn collect_graph(
        dataset: &InmemDataset<f32>,
        num_points: usize,
        params: &ForgeANNParams,
    ) -> Vec<Vec<u32>> {
        let mut graph = vec![Vec::new(); num_points];
        super::build_forgeann_graph(dataset, num_points, 1, params, |point, neighbors| {
            graph[point as usize] = neighbors;
            Ok(())
        })
        .unwrap();

        for neighbors in &mut graph {
            neighbors.sort_unstable();
        }
        graph
    }

    #[test]
    fn streaming_profile_idle_time_saturates_at_zero() {
        let mut profile = super::StreamingProfile::default();
        profile.total_wall = Duration::from_millis(25);
        profile.leaf.total_wall = Duration::from_millis(40);

        assert_eq!(profile.leaf_idle_time(), Duration::ZERO);
    }

    #[test]
    fn scheduler_budget_uses_bounded_leaf_backlog() {
        let budget = super::scheduler::SchedulerBudget::for_worker_count(42, 3500);

        assert_eq!(budget.worker_count, 42);
        assert_eq!(budget.leaf_backlog_capacity, 42 * 2048);
        assert_eq!(budget.producer_leaf_backlog_soft_limit, 0);
        assert_eq!(budget.large_leaf_min_size, 7000);
        assert_eq!(budget.leaf_block_rows, 1024);
    }

    #[test]
    fn oom_scheduler_budget_allows_d1_leaf_burst_without_inline_help() {
        let budget =
            super::scheduler::SchedulerBudget::for_memory_budget(42, 3500, 32 * 1024 * 1024 * 1024);

        assert_eq!(budget.worker_count, 42);
        assert_eq!(budget.producer_leaf_backlog_soft_limit, 0);
        assert!(
            budget.leaf_backlog_capacity >= 700_000,
            "large OOM D1 materialization can emit hundreds of thousands of leaves before child processing"
        );
        assert!(budget.leaf_backlog_capacity <= 1_048_576);
    }

    #[test]
    fn forgeann_rayon_worker_stack_policy_clamps_diagnostic_override() {
        assert_eq!(super::normalize_forgeann_rayon_stack_mb(None), 32);
        assert_eq!(super::normalize_forgeann_rayon_stack_mb(Some(1)), 8);
        assert_eq!(super::normalize_forgeann_rayon_stack_mb(Some(64)), 64);
        assert_eq!(super::normalize_forgeann_rayon_stack_mb(Some(1024)), 256);
    }

    #[test]
    fn blas_thread_env_round_trips_extended_thread_vars() {
        let _guard = ENV_LOCK.lock().unwrap();
        let original = super::BlasThreadEnv::capture();

        unsafe {
            env::set_var("OPENBLAS_NUM_THREADS", "8");
            env::set_var("OMP_NUM_THREADS", "9");
            env::set_var("MKL_NUM_THREADS", "10");
            env::set_var("BLIS_NUM_THREADS", "11");
            env::set_var("GOTO_NUM_THREADS", "12");
            env::set_var("MKL_DYNAMIC", "1");
        }
        let snapshot = super::BlasThreadEnv::capture();

        super::BlasThreadEnv::set_single_threaded();

        assert_eq!(env::var("OPENBLAS_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("OMP_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("MKL_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("BLIS_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("GOTO_NUM_THREADS").unwrap(), "1");
        assert_eq!(env::var("MKL_DYNAMIC").unwrap(), "0");

        snapshot.restore();

        assert_eq!(env::var("OPENBLAS_NUM_THREADS").unwrap(), "8");
        assert_eq!(env::var("OMP_NUM_THREADS").unwrap(), "9");
        assert_eq!(env::var("MKL_NUM_THREADS").unwrap(), "10");
        assert_eq!(env::var("BLIS_NUM_THREADS").unwrap(), "11");
        assert_eq!(env::var("GOTO_NUM_THREADS").unwrap(), "12");
        assert_eq!(env::var("MKL_DYNAMIC").unwrap(), "1");

        original.restore();
    }

    #[test]
    fn oom_build_keeps_external_artifacts_when_requested() {
        let dataset = build_test_dataset(1024, 8);
        let temp = tempdir().unwrap();
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        params.oom_keep_artifacts = true;
        params.c_min = 2;
        params.c_max = 4;
        params.max_depth = 6;
        params.max_leaders = 32;
        params.fanout_top = 4;
        params.fanout_second = 2;
        params.oom_temp_dir = PathBuf::from(temp.path());

        let mut visited = 0usize;
        super::build_forgeann_graph(&dataset, 1024, 1, &params, |_, neighbors| {
            let _ = neighbors;
            visited += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(visited, 1024);
        let entries: Vec<_> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.iter().any(|name| name.contains("sketch")),
            "expected OOM build to persist sketch artifacts, found {entries:?}"
        );
        assert!(
            entries
                .iter()
                .any(|name| name.starts_with("partition_d") && name.ends_with("_runs.bin")),
            "expected OOM build to persist append-only partition run artifacts, found {entries:?}"
        );
        assert!(
            entries.iter().all(|name| !name.contains("_child_")),
            "expected OOM build to avoid per-child partition run files, found {entries:?}"
        );
        let has_candidate_artifacts = entries.iter().any(|name| name.contains("candidate"));
        let memory_plan = super::OomMemoryPlan::from_params(&params);
        assert!(
            has_candidate_artifacts
                || super::plan_resident_reservoir_prefix(
                    1024,
                    1,
                    &params,
                    dataset.dim,
                    0,
                    1024,
                    memory_plan.reservoir_budget_bytes,
                )
                .resident_points
                    > 0,
            "expected candidate artifacts unless resident reservoir mode is active, found {entries:?}"
        );
    }

    #[test]
    fn oom_artifact_dirs_fallback_to_root_and_create_overrides() {
        let root = tempdir().unwrap();
        let sketch = tempdir().unwrap();
        let spill = tempdir().unwrap();
        let vector = tempdir().unwrap();

        let mut params = ForgeANNParams::default();
        params.oom_temp_dir = root.path().join("root-artifacts");
        params.oom_sketch_temp_dir = sketch.path().join("sketch-artifacts");
        params.oom_spill_temp_dir = spill.path().join("spill-artifacts");
        params.oom_vector_temp_dir = vector.path().join("vector-artifacts");

        let dirs = super::prepare_oom_artifact_dirs(&params).unwrap();

        assert_eq!(dirs.root_dir, params.oom_temp_dir);
        assert_eq!(dirs.sketch_dir, params.oom_sketch_temp_dir);
        assert_eq!(dirs.spill_dir, params.oom_spill_temp_dir);
        assert_eq!(dirs.partition_dir, params.oom_temp_dir);
        assert_eq!(dirs.vector_dir, params.oom_vector_temp_dir);
        for dir in [
            &dirs.root_dir,
            &dirs.sketch_dir,
            &dirs.spill_dir,
            &dirs.partition_dir,
            &dirs.vector_dir,
        ] {
            assert!(dir.is_dir(), "expected {} to exist", dir.display());
        }
    }

    #[test]
    fn oom_artifact_cleanup_preserves_unrelated_files_in_split_dirs() {
        let root = tempdir().unwrap();
        let sketch = tempdir().unwrap();
        let spill = tempdir().unwrap();
        let partition = tempdir().unwrap();
        let vector = tempdir().unwrap();

        let dirs = super::OomArtifactDirs {
            root_dir: root.path().to_path_buf(),
            sketch_dir: sketch.path().to_path_buf(),
            spill_dir: spill.path().to_path_buf(),
            partition_dir: partition.path().to_path_buf(),
            vector_dir: vector.path().to_path_buf(),
        };
        fs::write(dirs.sketch_dir.join("sketches.bin"), [1u8]).unwrap();
        fs::write(
            dirs.spill_dir.join("candidate_shard00000_part000.log"),
            [2u8],
        )
        .unwrap();
        fs::write(dirs.spill_dir.join("candidate_segments.manifest"), [3u8]).unwrap();
        fs::write(dirs.partition_dir.join("partition_d01_runs.bin"), [4u8]).unwrap();
        fs::write(
            dirs.vector_dir.join("partition_d02_vectors_00000001.bin"),
            [5u8],
        )
        .unwrap();
        fs::write(dirs.vector_dir.join("wiki_base.fbin"), [6u8]).unwrap();
        fs::write(dirs.spill_dir.join("notes.txt"), [7u8]).unwrap();

        super::cleanup_oom_artifacts_in_dirs(&dirs).unwrap();

        assert!(!dirs.sketch_dir.join("sketches.bin").exists());
        assert!(
            !dirs
                .spill_dir
                .join("candidate_shard00000_part000.log")
                .exists()
        );
        assert!(!dirs.spill_dir.join("candidate_segments.manifest").exists());
        assert!(!dirs.partition_dir.join("partition_d01_runs.bin").exists());
        assert!(
            !dirs
                .vector_dir
                .join("partition_d02_vectors_00000001.bin")
                .exists()
        );
        assert!(dirs.vector_dir.join("wiki_base.fbin").exists());
        assert!(dirs.spill_dir.join("notes.txt").exists());
    }

    #[test]
    fn oom_build_matches_in_memory_neighbors_on_small_dataset() {
        let dataset = build_test_dataset(48, 8);

        let in_memory_params = ForgeANNParams::default();

        let mut oom_params = in_memory_params.clone();
        oom_params.oom_enable = true;
        oom_params.oom_keep_artifacts = false;
        oom_params.oom_temp_dir = tempdir().unwrap().keep();

        let in_memory_graph = collect_graph(&dataset, 48, &in_memory_params);
        let oom_graph = collect_graph(&dataset, 48, &oom_params);

        assert_eq!(oom_graph, in_memory_graph);
    }

    #[test]
    fn oom_build_skips_candidate_spill_when_budget_can_hold_full_reservoirs() {
        let dataset = build_test_dataset(128, 8);
        let temp = tempdir().unwrap();
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        params.oom_keep_artifacts = true;
        params.oom_temp_dir = PathBuf::from(temp.path());

        let mut visited = 0usize;
        super::build_forgeann_graph(&dataset, 128, 4, &params, |_, neighbors| {
            let _ = neighbors;
            visited += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(visited, 128);
        let entries: Vec<_> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.iter().all(|name| !name.contains("candidate")),
            "expected OOM build to skip candidate spill artifacts when the full reservoir fits in memory, found {entries:?}"
        );
    }

    #[test]
    fn oom_build_cleans_up_external_artifacts_by_default() {
        let dataset = build_test_dataset(32, 8);
        let temp = tempdir().unwrap();
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        params.oom_keep_artifacts = false;
        params.oom_temp_dir = PathBuf::from(temp.path());

        let mut visited = 0usize;
        super::build_forgeann_graph(&dataset, 32, 1, &params, |_, neighbors| {
            let _ = neighbors;
            visited += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(visited, 32);
        let entries: Vec<_> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.is_empty(),
            "expected OOM build to clean artifacts by default, found {entries:?}"
        );
    }

    #[test]
    fn oom_profile_json_includes_memory_temp_file_and_stage_sections() {
        let dir = tempdir().unwrap();
        let profile_path = dir.path().join("oom_profile.json");
        let artifact_dir = dir.path().join("artifacts");
        fs::create_dir_all(&artifact_dir).unwrap();
        fs::write(artifact_dir.join("sketches.bin"), vec![0u8; 16]).unwrap();
        fs::write(
            artifact_dir.join("candidate_shard00000_part000.log"),
            vec![0u8; 16],
        )
        .unwrap();
        fs::write(
            artifact_dir.join("candidate_segments.manifest"),
            vec![0u8; 8],
        )
        .unwrap();
        fs::write(artifact_dir.join("partition_d01_runs.bin"), vec![0u8; 8]).unwrap();

        let profile = super::OomProfile {
            num_points: 32,
            dim: 8,
            point_store_backend: "direct",
            memory_budget_bytes: 12 * 1024 * 1024 * 1024,
            effective_threads: 4,
            pool_threads: 4,
            leaf_budget_threads: 3,
            reduce_shard_points: 128,
            sketch_bytes: 128,
            sketch_file_bytes: 256,
            spill: super::SpillRunStats {
                part_files: 1,
                segment_count: 2,
                shard_groups: 1,
                max_parts_per_shard: 1,
                max_segments_per_shard: 2,
                record_bytes: 16,
                manifest_bytes: 8,
            },
            reduce: super::SpillReduceStats {
                shard_groups: 1,
                segments_scanned: 2,
                part_files_opened: 1,
                max_segments_in_group: 2,
            },
            sketch_persist_wall: Duration::from_millis(11),
            streaming_wall: Duration::from_millis(22),
            reduce_wall: Duration::from_millis(33),
            emit_wall: Duration::from_millis(44),
            total_wall: Duration::from_millis(55),
            scheduler: super::SchedulerTelemetry::default(),
            rbc: super::RbcPartitionTelemetry {
                root_fanout: super::rbc_partition::RootFanoutProfile {
                    policy: "fixed".to_string(),
                    selection_semantics: "fixed".to_string(),
                    enabled: true,
                    fixed_fanout: 8,
                    total_points: 32,
                    kept_hist: vec![0, 10, 8, 0, 6, 0, 0, 0, 8],
                    avg_fanout: 2.0,
                    selector_ms: 12,
                    projected_assignment_bytes: 256,
                    projected_partition_d00_bytes_pre_dedup: 256,
                    root_leader_hash: 12345,
                    ..super::rbc_partition::RootFanoutProfile::default()
                },
                assignment_decisions: Box::new({
                    let mut decisions = super::rbc_partition::AssignmentDecisionProfile::default();
                    decisions.record_gemm(
                        1,
                        134_093,
                        751,
                        2,
                        Duration::from_millis(42),
                        Some("min-leaders"),
                    );
                    decisions
                }),
                ads_scheduler: Default::default(),
                io_planned_forgeann: Default::default(),
                point_pipeline: Default::default(),
            },
            leaf: super::LeafProfile::default(),
            memory: super::OomMemoryReport {
                rss_kb_samples: vec![128, 256],
                rss_hwm_kb: Some(512),
                vm_peak_kb: Some(1024),
                rss_by_stage: Default::default(),
            },
            temp_files: super::collect_temp_file_report(&artifact_dir).unwrap(),
            uring_batches: 1,
            uring_windows: 2,
            uring_bytes: 4096,
            uring_fallbacks: 0,
        };

        super::write_oom_profile(&profile_path, &profile).unwrap();
        let text = fs::read_to_string(&profile_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["point_store_backend"], "direct");
        assert!(text.contains("\"memory\""));
        assert!(text.contains("\"rss_kb_samples\""));
        assert!(text.contains("\"rss_by_stage\""));
        for stage in [
            "start",
            "after_sketches",
            "after_root",
            "after_streaming",
            "after_reduce",
            "final",
        ] {
            assert!(
                json["memory"]["rss_by_stage"][stage].is_object(),
                "missing memory stage {stage}"
            );
            assert!(
                json["memory"]["rss_by_stage"][stage]["vm_peak_kb"].is_number()
                    || json["memory"]["rss_by_stage"][stage]["vm_peak_kb"].is_null()
            );
            assert!(
                json["memory"]["rss_by_stage"][stage]["rss_hwm_kb"].is_number()
                    || json["memory"]["rss_by_stage"][stage]["rss_hwm_kb"].is_null()
            );
        }
        assert!(text.contains("\"vm_peak_kb\""));
        assert!(text.contains("\"memory_budget_bytes\""));
        assert!(text.contains("\"effective_threads\""));
        assert!(text.contains("\"pool_threads\""));
        assert!(text.contains("\"leaf_budget_threads\""));
        assert!(text.contains("\"reduce_shard_points\""));
        assert!(text.contains("\"spill\""));
        assert!(text.contains("\"part_files\""));
        assert!(text.contains("\"segment_count\""));
        assert!(text.contains("\"record_bytes\""));
        assert!(text.contains("\"manifest_bytes\""));
        assert!(text.contains("\"reduce\""));
        assert!(text.contains("\"segments_scanned\""));
        assert!(text.contains("\"part_files_opened\""));
        assert!(text.contains("\"temp_files\""));
        assert!(text.contains("\"candidate\""));
        assert!(text.contains("\"partition_depth1\""));
        assert!(text.contains("\"stage_ms\""));
        assert!(text.contains("\"streaming\""));
        assert!(text.contains("\"total\""));
        assert!(text.contains("\"io_uring\""));
        assert!(text.contains("\"point_gather_batches\""));
        assert!(text.contains("\"point_gather_fallbacks\""));
        assert!(text.contains("\"root_fanout\""));
        assert!(text.contains("\"policy\": \"fixed\""));
        assert!(text.contains("\"selection_semantics\": \"fixed\""));
        assert!(text.contains("\"enabled\": true"));
        assert!(text.contains("\"fixed_fanout\": 8"));
        assert!(text.contains("\"selector_ms\": 12"));
        assert!(text.contains("\"avg_fanout\": 2.0"));
        assert!(text.contains("\"root_leader_hash\": 12345"));
        assert!(text.contains("\"io_planned_forgeann\": null"));
        assert!(text.contains("\"point_pipeline\""));
        assert!(text.contains("\"read_amplification\""));
        assert!(text.contains("\"assignment_decisions\""));
        assert_eq!(
            json["assignment_decisions"]["depth_leader_buckets"]
                .as_array()
                .unwrap()
                .len(),
            super::rbc_partition::ASSIGNMENT_DEPTH_BUCKETS
        );
        assert_eq!(
            json["assignment_decisions"]["depth_leader_buckets"][1]["leader_buckets"][2]
                ["gemm_nodes"]
                .as_u64()
                .unwrap(),
            1
        );
        assert_eq!(
            json["assignment_decisions"]["fallback_reasons"]["min-leaders"]
                .as_u64()
                .unwrap(),
            1
        );
        assert_eq!(
            json["leaf"]["size_buckets"].as_array().unwrap().len(),
            super::LEAF_SIZE_BUCKETS
        );
        assert_eq!(
            json["leaf"]["size_buckets"][0]["label"].as_str().unwrap(),
            "[256,512)"
        );
        assert!(text.contains("\"leaf_adsampling_simd_active_lane_evals\""));
        assert!(text.contains("\"leaf_ads_tiling_enabled\""));
        assert!(text.contains("\"leaf_ads_wavefront_pairmask_enabled\""));
        assert!(text.contains("\"leaf_ads_work_graph_enabled\""));
        assert!(text.contains("\"leaf_ads_handle_requeues\""));
        assert!(text.contains("\"leaf_ads_ewma_ns_per_row_by_bucket\""));
        assert!(text.contains("\"queued_work_ms_peak\""));
        assert!(text.contains("\"worker_busy_ms\""));
    }

    #[test]
    fn collect_temp_file_report_classifies_partition_candidate_and_other_bytes() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("sketches.bin"), vec![0u8; 10]).unwrap();
        fs::write(
            dir.path().join("candidate_shard00000_part000.log"),
            vec![0u8; 11],
        )
        .unwrap();
        fs::write(dir.path().join("candidate_segments.manifest"), vec![0u8; 7]).unwrap();
        fs::write(dir.path().join("partition_d00_runs.bin"), vec![0u8; 12]).unwrap();
        fs::write(dir.path().join("partition_d02_runs.bin"), vec![0u8; 13]).unwrap();
        fs::write(dir.path().join("misc.tmp"), vec![0u8; 14]).unwrap();

        let report = super::collect_temp_file_report(dir.path()).unwrap();
        assert_eq!(report.total.files, 6);
        assert_eq!(report.total.bytes, 67);
        assert_eq!(report.candidate.files, 2);
        assert_eq!(report.candidate.bytes, 18);
        assert_eq!(report.partition_depth0.files, 1);
        assert_eq!(report.partition_depth0.bytes, 12);
        assert_eq!(report.partition_depth2.files, 1);
        assert_eq!(report.partition_depth2.bytes, 13);
        assert_eq!(report.other.files, 2);
        assert_eq!(report.other.bytes, 24);
    }

    #[test]
    fn derive_oom_worker_count_obeys_memory_budget() {
        let mut params = ForgeANNParams::default();
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 8192;

        let tight = super::derive_oom_worker_count(64, &params, 768);
        assert!(tight < 64);
        assert!(tight >= 1);
    }

    #[test]
    fn derive_oom_worker_count_reserves_budget_for_non_leaf_work() {
        let mut params = ForgeANNParams::default();
        params.oom_memory_budget_bytes = 4 * 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let workers = super::derive_oom_worker_count(42, &params, 768);
        assert!(
            workers < 42,
            "worker count should be clamped under a 4 GiB OOM budget"
        );
        assert!(
            workers > 3,
            "the 4 GiB budget should allow more leaf workers than the 1 GiB clamp"
        );
        assert!(workers >= 1);
    }

    #[test]
    fn oom_memory_plan_reserves_subtree_budget_without_clamping_leaf_threads() {
        let mut params = ForgeANNParams::default();
        params.oom_memory_budget_bytes = 28 * 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let plan = super::OomMemoryPlan::from_params(&params);

        assert!(plan.reservoir_budget_bytes < plan.total_budget_bytes);
        assert!(plan.scratch_budget_bytes > 0);
        assert_eq!(super::derive_oom_worker_count(42, &params, 768), 42);
    }

    #[test]
    fn oom_memory_plan_preserves_resident_subtree_budget() {
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 32 * 1024 * 1024 * 1024;

        let plan = super::OomMemoryPlan::from_params(&params);

        assert_eq!(
            plan.reservoir_budget_bytes,
            params.effective_oom_memory_budget_bytes() * 45 / 100
        );
        assert_eq!(
            plan.scratch_budget_bytes,
            params.effective_oom_memory_budget_bytes() * 15 / 100
        );
    }

    #[test]
    fn resident_reservoir_plan_does_not_shard_align_when_full_budget_fits() {
        let mut params = ForgeANNParams::default();
        params.oom_memory_budget_bytes = 64 * 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;
        params.l_max = 64;
        let num_points = 35_167_920;
        let dim = 768;
        let leaf_threads = 42;
        let sketch_bytes = 736_280_232;
        let shard_points = 7_669_584;

        let memory_plan = super::OomMemoryPlan::from_params(&params);
        let plan = super::plan_resident_reservoir_prefix(
            num_points,
            leaf_threads,
            &params,
            dim,
            sketch_bytes,
            shard_points,
            memory_plan.reservoir_budget_bytes,
        );

        assert_eq!(
            plan.resident_points, num_points,
            "when the reservoir budget can hold all points, the plan should select all points instead of rounding down to a shard boundary"
        );
    }

    #[test]
    fn full_resident_sketch_cache_requires_explicit_sufficient_budget() {
        let mut params = ForgeANNParams::default();
        params.oom_sketch_cache_bytes = 0;
        assert!(!super::should_use_full_resident_sketch_cache(
            &params,
            256 * 1024 * 1024
        ));

        params.oom_sketch_cache_bytes = 128 * 1024 * 1024;
        assert!(!super::should_use_full_resident_sketch_cache(
            &params,
            256 * 1024 * 1024
        ));

        params.oom_sketch_cache_bytes = 512 * 1024 * 1024;
        assert!(super::should_use_full_resident_sketch_cache(
            &params,
            256 * 1024 * 1024
        ));
    }

    #[test]
    fn one_gib_oom_budget_clamps_leaf_budget_threads_but_keeps_more_parallelism() {
        let mut params = ForgeANNParams::default();
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let workers = super::derive_oom_worker_count(12, &params, 768);
        assert_eq!(
            workers, 3,
            "the worker budget still clamps under a 1 GiB OOM budget, but the tighter scratch reserve should allow more parallelism than before"
        );
    }

    #[test]
    fn oom_execution_threads_keep_full_pool_width_while_clamping_leaf_budget_threads() {
        let mut params = ForgeANNParams::default();
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let threads = super::derive_oom_execution_threads(12, &params, 768);
        assert_eq!(threads.pool_threads, 12);
        assert_eq!(threads.leaf_budget_threads, 3);
    }

    #[test]
    fn resident_prefix_planning_uses_leaf_budget_threads_not_pool_width() {
        let mut params = ForgeANNParams::default();
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;

        let threads = super::derive_oom_execution_threads(12, &params, 768);
        let memory_plan = super::OomMemoryPlan::from_params(&params);
        let plan_with_leaf_budget = super::plan_resident_reservoir_prefix(
            5_000_000,
            threads.leaf_budget_threads,
            &params,
            768,
            0,
            100_000,
            memory_plan.reservoir_budget_bytes,
        );
        let plan_with_pool_width = super::plan_resident_reservoir_prefix(
            5_000_000,
            threads.pool_threads,
            &params,
            768,
            0,
            100_000,
            memory_plan.reservoir_budget_bytes,
        );

        assert!(
            plan_with_leaf_budget.resident_points >= plan_with_pool_width.resident_points,
            "resident planning should be based on leaf-budget scratch, not the wider RBC/root pool width"
        );
    }

    #[test]
    fn derive_spill_shard_points_uses_memory_budget_when_spill_budget_is_unset() {
        let mut params = ForgeANNParams::default();
        params.oom_memory_budget_bytes = 1024 * 1024 * 1024;
        params.oom_spill_cache_bytes = 0;

        let shard_points = super::derive_spill_shard_points(
            1_000_000,
            &params,
            params.effective_oom_memory_budget_bytes(),
        );
        assert!(shard_points > 1);
        assert!(shard_points <= 1_000_000);
    }

    #[test]
    fn resident_reservoir_plan_can_select_partial_prefix_when_budget_is_between_full_and_zero() {
        let mut params = ForgeANNParams::default();
        params.oom_memory_budget_bytes = 4 * 1024 * 1024 * 1024;
        params.max_full_matrix_leaf_size = 3500;
        let shard_points = 100_000;

        let plan = super::plan_resident_reservoir_prefix(
            10_000_000,
            8,
            &params,
            768,
            0,
            shard_points,
            super::OomMemoryPlan::from_params(&params).reservoir_budget_bytes,
        );

        assert!(
            plan.resident_points > 0,
            "expected some prefix to remain resident"
        );
        assert!(
            plan.resident_points < 10_000_000,
            "expected budget to be insufficient for full residency"
        );
        assert_eq!(
            plan.resident_points % shard_points,
            0,
            "resident prefix should be shard aligned"
        );
    }

    #[test]
    fn explicit_resident_reservoir_cap_is_not_shard_aligned() {
        let mut params = ForgeANNParams::default();
        params.l_max = 64;
        let desired_points = 12_345;
        let cap_bytes = super::estimate_reservoir_bytes(desired_points, params.l_max);

        let plan = super::plan_resident_reservoir_prefix_from_cap(5_000_000, &params, cap_bytes);

        assert_eq!(
            plan.resident_points, desired_points,
            "explicit cap should mean resident reservoir bytes, not legacy shard-aligned planning budget"
        );
    }

    #[test]
    fn oom_build_hybrid_resident_prefix_emits_each_point_once_and_keeps_spill_artifacts() {
        let num_points = 5_000usize;
        let dim = 8usize;
        let dataset = build_test_dataset(num_points, dim);
        let temp = tempdir().unwrap();
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        params.oom_keep_artifacts = true;
        params.l_max = 4_096;
        params.max_full_matrix_leaf_size = 256;
        params.c_min = 16;
        params.c_max = 32;
        params.max_depth = 6;
        params.max_leaders = 32;
        params.fanout_top = 4;
        params.fanout_second = 2;
        params.oom_memory_budget_bytes = 1280 * 1024 * 1024;
        params.oom_spill_cache_bytes = 16 * 1024 * 1024;
        params.oom_temp_dir = PathBuf::from(temp.path());

        let budget = params.effective_oom_memory_budget_bytes();
        let shard_points = super::derive_spill_shard_points(num_points, &params, budget);
        let effective_threads = super::derive_oom_worker_count(1, &params, dim);
        let resident_plan = super::plan_resident_reservoir_prefix(
            num_points,
            effective_threads,
            &params,
            dim,
            0,
            shard_points,
            super::OomMemoryPlan::from_params(&params).reservoir_budget_bytes,
        );
        assert!(
            resident_plan.resident_points > 0 && resident_plan.resident_points < num_points,
            "test configuration should force hybrid resident+spill mode, got resident_points={} num_points={} shard_points={}",
            resident_plan.resident_points,
            num_points,
            shard_points
        );

        let mut seen = vec![0usize; num_points];
        super::build_forgeann_graph(&dataset, num_points, 1, &params, |point, _neighbors| {
            seen[point as usize] += 1;
            Ok(())
        })
        .unwrap();

        assert!(
            seen.iter().all(|count| *count == 1),
            "expected every point to be emitted exactly once"
        );
        let entries: Vec<_> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.iter().any(|name| name.contains("candidate")),
            "expected hybrid mode to keep spill artifacts for the tail, found {entries:?}"
        );
    }
}
