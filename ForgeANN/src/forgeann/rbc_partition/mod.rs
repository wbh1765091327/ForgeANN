pub(crate) use std::collections::{BTreeMap, BinaryHeap, HashMap};
pub(crate) use std::fs::{self, File, OpenOptions};
pub(crate) use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
pub(crate) use std::mem::size_of;
pub(crate) use std::os::unix::fs::FileExt;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
pub(crate) use std::sync::{Arc, mpsc};
pub(crate) use std::time::{Duration, Instant};

pub(crate) use crossbeam::channel as crossbeam_channel;
pub(crate) use indicatif::{ProgressBar, ProgressStyle};
pub(crate) use ndarray::{ArrayView2, ArrayViewMut2};
pub(crate) use parking_lot::{Condvar, Mutex};
pub(crate) use rand::rngs::StdRng;
pub(crate) use rand::{Rng, SeedableRng};
pub(crate) use rayon;
pub(crate) use rayon::prelude::*;
pub(crate) use sha2::{Digest, Sha256};
pub(crate) use tempfile::NamedTempFile;

pub(crate) use super::adsampling::{
    AdSamplingChunkResult, AdSamplingConfig, AdSamplingLeaderLayout,
    AdSamplingPointAssignmentChunk, AdSamplingProfile, adsampling_depth_fallback_reason,
    adsampling_seed_indices, adsampling_seed_mask, assign_loaded_point_chunk_adsampling_layout,
    assign_loaded_point_indexed_adsampling_layout, assign_point_leaders_adsampling_streaming,
    classify_adsampling_task, duration_ms, should_use_adsampling_assignment,
};
pub(crate) use super::direct_io::{DirectIoConfig, DirectIoFile};
pub(crate) use super::io_planned_forgeann::{
    IoCostModel, IoPainSample, IoPlanConfig, IoPlannerStats, NodeAction, StorageState,
};
pub(crate) use super::params::ForgeANNParams;
pub(crate) use super::point_pipeline::PointPipelineStats;
#[cfg(test)]
pub(crate) use super::point_store::ResidentSubsetPointStore;
pub(crate) use super::point_store::{
    PointBatchStats, PointStore, VectorRunPointStore, WindowedGatherOptions,
};
pub(crate) use super::sampling::sample_set_bottomk;
pub(crate) use super::scheduler::{
    LargeAssignmentGuard, LeafDrainerLimitGuard, LeafEmitter, SchedulerSignals, SchedulerTelemetry,
};
pub(crate) use crate::common::{AnnError, AnnResult, Metric};

pub(crate) mod assignment_profile;
pub(crate) mod child_run_io;
pub(crate) mod gemm_tile;
pub(crate) mod memory_context;
pub(crate) mod prefetch_pipeline;

pub(crate) mod telemetry;
pub(crate) mod types;

// Re-exports for ForgeANN production callers.
pub use assignment_profile::{
    ASSIGNMENT_DEPTH_BUCKETS, ASSIGNMENT_FALLBACK_REASONS, ASSIGNMENT_LEADER_BUCKETS,
    AssignmentDecisionProfile, GemmProfile, assignment_depth_bucket_label,
    assignment_fallback_reason_index, assignment_fallback_reason_label,
    assignment_leader_bucket_label,
};
pub(super) use assignment_profile::{
    AdSamplingSchedulerRuntime, AdSamplingSchedulerStats, AssignmentDecisionRecord, SeededCluster,
};
pub(super) use child_run_io::{
    AssignmentContext, ExternalRunStore, PartitionStats, compute_clusters_gemm,
    compute_clusters_gemm_budgeted_with_context, rbc_recurse_parallel,
    read_child_run_chain_from_path, read_child_runs_batched_from_path, should_dedup_cluster,
};
pub(crate) use child_run_io::{rbc_partition_streaming, rbc_partition_streaming_with_dirs};
pub(crate) use gemm_tile::{
    choose_gemm_tile_sizes_with_override_and_budget, update_block_best_from_gram_tile,
    update_block_topk_from_gram_tile,
};
pub(super) use memory_context::{
    BufferedMergedChild, DepthWaveChildRunSchedule, ExternalPartitionAttempt,
    MaterializedMergedChild, MergedRawGroup, RootFanoutState, initialize_root_fanout_state,
    io_plan_config_for_params, select_root_leaders,
};
pub(crate) use prefetch_pipeline::StackTopK;
pub(super) use prefetch_pipeline::{
    PrefetchPipelineKind, StrictPrefetchGate, StrictPrefetchPipelineConfig,
    StrictPrefetchPipelineProfile, choose_compute_block_points_for_batch,
    choose_prefetch_batch_points_for_budget, choose_prefetch_queue_depth_for_budget,
    choose_spool_prefetch_workers, for_each_prefetched_point_batch_profiled,
    should_use_strict_prefetch_for_gemm_assignment, split_seeded_clusters_balanced,
};
pub(crate) use telemetry::{RbcPartitionTelemetry, RootFanoutProfile, root_fanout_profile_json};
// Bring types into scope for internal use within this module
pub(super) use types::*;
// Re-export types needed by external callers
pub(crate) use types::{
    ChildRun, LeafReason, PartitionResult, RbcPhase, RunExtent, VECTOR_RUN_FILE_COUNTER,
};

fn recurse_child_points_external(
    dataset: &dyn PointStore,
    points: Vec<u32>,
    depth: usize,
    parent_n: usize,
    metric: Metric,
    params: &ForgeANNParams,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    seed: u64,
    pb: &ProgressBar,
    external_run_store: &Arc<Mutex<ExternalRunStore>>,
    root_fanout_state: &RootFanoutState,
    assignment_context: &AssignmentContext<'_>,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<PartitionStats> {
    rbc_recurse_parallel(
        dataset,
        points,
        depth,
        parent_n,
        metric,
        params,
        adaptive_c_max,
        min_recurse_size,
        seed,
        pb,
        Some(external_run_store),
        root_fanout_state,
        assignment_context,
        leaf_emitter,
    )
}

fn recurse_child_points_vector_run(
    dataset: &dyn PointStore,
    points: Vec<u32>,
    depth: usize,
    parent_n: usize,
    metric: Metric,
    params: &ForgeANNParams,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    seed: u64,
    pb: &ProgressBar,
    external_run_store: &Arc<Mutex<ExternalRunStore>>,
    root_fanout_state: &RootFanoutState,
    assignment_context: &AssignmentContext<'_>,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<(PartitionStats, usize, Duration)> {
    let ordinal = VECTOR_RUN_FILE_COUNTER.fetch_add(1, Ordering::AcqRel);
    let path = { external_run_store.lock().vector_path(depth, ordinal) };
    let materialize_start = Instant::now();
    let vector_store = VectorRunPointStore::materialize(dataset, &points, &path, depth)?;
    let materialize_wall = materialize_start.elapsed();
    let temp_bytes = vector_store.byte_len();
    let child_result = rbc_recurse_parallel(
        &vector_store,
        points,
        depth,
        parent_n,
        metric,
        params,
        adaptive_c_max,
        min_recurse_size,
        seed,
        pb,
        Some(external_run_store),
        root_fanout_state,
        assignment_context,
        leaf_emitter,
    )
    .map(|child_stats| (child_stats, temp_bytes, materialize_wall));
    drop(vector_store);
    cleanup_vector_run_file(&path, child_result)
}

fn cleanup_vector_run_file(
    path: &Path,
    result: AnnResult<(PartitionStats, usize, Duration)>,
) -> AnnResult<(PartitionStats, usize, Duration)> {
    let cleanup_result = match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    };

    match (result, cleanup_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(primary), Ok(())) => Err(primary),
        (Ok(_), Err(cleanup)) => Err(AnnError::log_index_error(format!(
            "Failed to remove vector-run temp file {}: {cleanup}",
            path.display()
        ))),
        (Err(primary), Err(cleanup)) => {
            tracing::warn!(
                "Failed to remove vector-run temp file {} after primary error: {}",
                path.display(),
                cleanup
            );
            Err(primary)
        }
    }
}

#[cfg(test)]
mod vector_run_cleanup_tests {
    use std::fs::File;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn vector_run_cleanup_removes_file_after_success() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("partition_d02_vectors_00000001.bin");
        File::create(&path).unwrap();

        let result = cleanup_vector_run_file(
            &path,
            Ok((PartitionStats::new(2, 4), 128, Duration::from_millis(3))),
        )
        .unwrap();

        assert_eq!(result.1, 128);
        assert!(!path.exists());
    }

    #[test]
    fn vector_run_cleanup_removes_file_after_primary_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("partition_d02_vectors_00000002.bin");
        File::create(&path).unwrap();

        let result = cleanup_vector_run_file(
            &path,
            Err(AnnError::log_index_error("synthetic failure".to_string())),
        );

        assert!(result.is_err());
        assert!(!path.exists());
    }
}

fn point_store_storage_state(dataset: &dyn PointStore) -> StorageState {
    if dataset.is_resident_subset() {
        StorageState::Resident
    } else if dataset.is_vector_run() {
        StorageState::VectorRun
    } else {
        StorageState::RawIds
    }
}

fn recurse_child_run_maybe_resident(
    dataset: &dyn PointStore,
    child_run: ChildRun,
    depth: usize,
    parent_n: usize,
    metric: Metric,
    params: &ForgeANNParams,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    pb: &ProgressBar,
    external_run_store: &Arc<Mutex<ExternalRunStore>>,
    root_fanout_state: &RootFanoutState,
    assignment_context: &AssignmentContext<'_>,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<PartitionStats> {
    let expected_leaves = (child_run.len / adaptive_c_max.max(1)).max(4);
    let mut stats = PartitionStats::new(params.max_depth, expected_leaves);
    let (base_dir, io_cfg) = {
        let guard = external_run_store.lock();
        (guard.base_dir.clone(), guard.io_config())
    };
    let points = if params.io_planned_forgeann_enabled() {
        let (mut batched, batch_stats) = read_child_runs_batched_from_path(
            &base_dir,
            io_cfg,
            depth - 1,
            std::slice::from_ref(&child_run),
        )?;
        stats
            .telemetry
            .io_planned_forgeann
            .record_child_extent_batching(
                batch_stats.logical_extents,
                batch_stats.coalesced_reads,
                batch_stats.logical_bytes,
                batch_stats.physical_bytes,
                batch_stats.header_read_savings,
            );
        batched.pop().unwrap_or_default()
    } else {
        read_child_run_chain_from_path(&base_dir, io_cfg, depth - 1, &child_run.extents)?
    };

    let io_config = io_plan_config_for_params(params);
    let io_plan = if io_config.enabled {
        let state = point_store_storage_state(dataset);
        let resident_used_bytes = 0;
        // Read temp budget from the atomic counter (not hardcoded 0).
        let temp_used_bytes = {
            let guard = external_run_store.lock();
            guard
                .io_plan_budget
                .as_ref()
                .map(|b| b.temp_reserved() as usize)
                .unwrap_or(0)
        };
        let row_bytes = dataset.dim().saturating_mul(size_of::<f32>());
        let fanout = params.adaptive_fanout(points.len(), depth).max(1);
        let observed_pain = assignment_context.observed_io_pain_for_depth(depth);
        let plan = if let Some(pain) = observed_pain {
            IoCostModel::default().plan_node_with_pain(
                io_config,
                state,
                depth,
                points.len(),
                fanout,
                row_bytes,
                resident_used_bytes,
                temp_used_bytes,
                pain,
            )
        } else {
            IoCostModel::default().plan_node(
                io_config,
                state,
                depth,
                points.len(),
                fanout,
                row_bytes,
                resident_used_bytes,
                temp_used_bytes,
            )
        };
        stats
            .telemetry
            .io_planned_forgeann
            .record_plan(&plan, io_config);
        Some(plan)
    } else {
        None
    };

    if let Some(plan) = io_plan.as_ref().filter(|_| io_config.dry_run) {
        let _ = plan;
        stats
            .telemetry
            .io_planned_forgeann
            .record_external_execution();
        let child_stats = recurse_child_points_external(
            dataset,
            points,
            depth,
            parent_n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            child_run.seed,
            pb,
            external_run_store,
            root_fanout_state,
            assignment_context,
            leaf_emitter,
        )?;
        stats.merge_from(child_stats);
        return Ok(stats);
    }

    if let Some(plan) = io_plan.as_ref() {
        match plan.selected {
            NodeAction::ExternalStreaming | NodeAction::ExactFallback => {
                stats
                    .telemetry
                    .io_planned_forgeann
                    .record_external_execution();
                if plan.selected == NodeAction::ExactFallback {
                    stats
                        .telemetry
                        .io_planned_forgeann
                        .record_exact_fallback_execution();
                }
                let child_stats = recurse_child_points_external(
                    dataset,
                    points,
                    depth,
                    parent_n,
                    metric,
                    params,
                    adaptive_c_max,
                    min_recurse_size,
                    child_run.seed,
                    pb,
                    external_run_store,
                    root_fanout_state,
                    assignment_context,
                    leaf_emitter,
                )?;
                stats.merge_from(child_stats);
                return Ok(stats);
            }
            NodeAction::PrefetchStreaming => {
                stats
                    .telemetry
                    .io_planned_forgeann
                    .record_prefetch_streaming_execution();
                let child_stats = recurse_child_points_external(
                    dataset,
                    points,
                    depth,
                    parent_n,
                    metric,
                    params,
                    adaptive_c_max,
                    min_recurse_size,
                    child_run.seed,
                    pb,
                    external_run_store,
                    root_fanout_state,
                    assignment_context,
                    leaf_emitter,
                )?;
                stats.merge_from(child_stats);
                return Ok(stats);
            }
            NodeAction::HotVectorRun => {
                // Atomically reserve temp budget before materialization.
                let temp_bytes_estimate = points
                    .len()
                    .saturating_mul(dataset.dim().saturating_mul(std::mem::size_of::<f32>()));
                let budget_reserved = {
                    let guard = external_run_store.lock();
                    guard
                        .io_plan_budget
                        .as_ref()
                        .map(|b| b.try_reserve_temp(temp_bytes_estimate as u64))
                        .unwrap_or(true)
                };
                if !budget_reserved {
                    // Budget exceeded — fall back to external.
                    stats
                        .telemetry
                        .io_planned_forgeann
                        .record_vector_to_external_fallback();
                    let child_stats = recurse_child_points_external(
                        dataset,
                        points,
                        depth,
                        parent_n,
                        metric,
                        params,
                        adaptive_c_max,
                        min_recurse_size,
                        child_run.seed,
                        pb,
                        external_run_store,
                        root_fanout_state,
                        assignment_context,
                        leaf_emitter,
                    )?;
                    stats.merge_from(child_stats);
                    return Ok(stats);
                }
                let (child_stats, temp_bytes, materialize_wall) = recurse_child_points_vector_run(
                    dataset,
                    points,
                    depth,
                    parent_n,
                    metric,
                    params,
                    adaptive_c_max,
                    min_recurse_size,
                    child_run.seed,
                    pb,
                    external_run_store,
                    root_fanout_state,
                    assignment_context,
                    leaf_emitter,
                )?;
                // Record actual bytes written and release reservation.
                {
                    let guard = external_run_store.lock();
                    if let Some(b) = guard.io_plan_budget.as_ref() {
                        b.record_temp_written(temp_bytes as u64);
                        b.release_temp(temp_bytes_estimate as u64);
                    }
                }
                stats
                    .telemetry
                    .io_planned_forgeann
                    .record_vector_run_execution(temp_bytes, materialize_wall);
                stats.merge_from(child_stats);
                return Ok(stats);
            }
        }
    }

    if io_config.enabled {
        stats
            .telemetry
            .io_planned_forgeann
            .record_external_execution();
    }
    let child_stats = recurse_child_points_external(
        dataset,
        points,
        depth,
        parent_n,
        metric,
        params,
        adaptive_c_max,
        min_recurse_size,
        child_run.seed,
        pb,
        external_run_store,
        root_fanout_state,
        assignment_context,
        leaf_emitter,
    )?;
    stats.merge_from(child_stats);
    Ok(stats)
}

pub(super) struct RbcRootPartitionPlan<'a> {
    pub total_size: usize,
    pub adaptive_c_max: usize,
    pub min_recurse_size: usize,
    pub root_fanout_state: RootFanoutState,
    pub assignment_context: AssignmentContext<'a>,
    pub expected_leaves: usize,
    pub seed: u64,
}

pub(super) fn prepare_rbc_root_partition_plan<'a>(
    dataset: &dyn PointStore,
    adsampling_dataset: Option<&'a dyn PointStore>,
    indices: &[u32],
    metric: Metric,
    params: &'a ForgeANNParams,
    rng: &mut (impl Rng + Send),
) -> AnnResult<Option<RbcRootPartitionPlan<'a>>> {
    let total_size = indices.len();
    if total_size == 0 {
        return Ok(None);
    }
    let adaptive_c_max = params.adaptive_c_max(total_size);
    let min_recurse_size = adaptive_c_max.saturating_mul(2);
    let root_leaders = select_root_leaders(indices, params, rng);
    let root_fanout_state = if root_leaders.is_empty() {
        RootFanoutState::fixed(params.fanout_top)
    } else {
        initialize_root_fanout_state(dataset, indices, &root_leaders, params, metric, rng)?
    };
    let assignment_context = AssignmentContext::new_for_assignment_with_adsampling_dataset(
        0,
        params,
        adsampling_dataset,
    );
    let expected_leaves = (total_size / adaptive_c_max).max(1000);
    let seed: u64 = rng.random();
    Ok(Some(RbcRootPartitionPlan {
        total_size,
        adaptive_c_max,
        min_recurse_size,
        root_fanout_state,
        assignment_context,
        expected_leaves,
        seed,
    }))
}

pub(super) fn execute_rbc_root_partition_plan(
    dataset: &dyn PointStore,
    indices: &[u32],
    metric: Metric,
    params: &ForgeANNParams,
    external_run_store: Option<Arc<Mutex<ExternalRunStore>>>,
    leaf_emitter: &dyn LeafEmitter,
    plan: RbcRootPartitionPlan<'_>,
) -> AnnResult<RbcPartitionTelemetry> {
    let RbcRootPartitionPlan {
        total_size,
        adaptive_c_max,
        min_recurse_size,
        root_fanout_state,
        assignment_context,
        expected_leaves,
        seed,
    } = plan;

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .unwrap(),
    );

    tracing::info!(
        "RBC partition starting: {} points, adaptive_c_max={}, min_recurse={}",
        format_count(total_size),
        format_count(adaptive_c_max),
        format_count(min_recurse_size),
    );

    pb.set_message(format!(
        "leaves=0 depth=0 c_max={} min_recurse={}",
        format_count(adaptive_c_max),
        format_count(min_recurse_size),
    ));

    let mut stats = rbc_recurse_parallel(
        dataset,
        indices.to_vec(),
        0,
        total_size,
        metric,
        params,
        adaptive_c_max,
        min_recurse_size,
        seed,
        &pb,
        external_run_store.as_ref(),
        &root_fanout_state,
        &assignment_context,
        leaf_emitter,
    )?;
    stats.max_cluster_size_seen = stats.max_cluster_size_seen.max(total_size);
    stats.telemetry.root_fanout = root_fanout_state.profile();
    stats.telemetry.ads_scheduler = assignment_context.ads_scheduler_stats();
    assignment_context.merge_observed_io_pain_into(&mut stats.telemetry.io_planned_forgeann);
    // Ensure expected_leaves capacity is reflected
    stats
        .leaf_sizes
        .reserve(expected_leaves.saturating_sub(stats.leaf_sizes.len()));

    pb.finish_and_clear();
    log_final_summary(&stats, total_size, adaptive_c_max, min_recurse_size);

    Ok(stats.telemetry)
}

fn rbc_partition_impl_streaming(
    dataset: &dyn PointStore,
    adsampling_dataset: Option<&dyn PointStore>,
    indices: &[u32],
    metric: Metric,
    params: &ForgeANNParams,
    _num_threads: u32,
    rng: &mut (impl Rng + Send),
    external_run_store: Option<Arc<Mutex<ExternalRunStore>>>,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<RbcPartitionTelemetry> {
    let Some(plan) =
        prepare_rbc_root_partition_plan(dataset, adsampling_dataset, indices, metric, params, rng)?
    else {
        return Ok(RbcPartitionTelemetry::default());
    };
    execute_rbc_root_partition_plan(
        dataset,
        indices,
        metric,
        params,
        external_run_store,
        leaf_emitter,
        plan,
    )
}

fn emit_leaf(
    dataset: &dyn PointStore,
    metric: Metric,
    params: &ForgeANNParams,
    leaf: &mut Vec<u32>,
    depth: usize,
    reason: LeafReason,
    dedup: bool,
    max_leaf_size: usize,
    stats: &mut PartitionStats,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<()> {
    let before_dedup = leaf.len();
    if dedup {
        leaf.sort_unstable();
        leaf.dedup();
    }
    stats.record_assignments(before_dedup, leaf.len());

    if leaf.is_empty() {
        return Ok(());
    }

    emit_leaf_with_safety(
        dataset,
        metric,
        params,
        leaf,
        depth,
        reason,
        max_leaf_size.max(1),
        stats,
        leaf_emitter,
    )
}

fn forced_leaf_split_seed(leaf: &[u32], depth: usize, reason: LeafReason) -> u64 {
    let reason_tag = match reason {
        LeafReason::NaturalSize => 0_u64,
        LeafReason::MaxDepth => 1_u64,
        LeafReason::ShrinkRatio => 2_u64,
        LeafReason::SmallDeep => 3_u64,
        LeafReason::MinRecurse => 4_u64,
        LeafReason::PartitionFallback => 5_u64,
    };
    let first = leaf.first().copied().unwrap_or_default() as u64;
    let last = leaf.last().copied().unwrap_or_default() as u64;
    (depth as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (leaf.len() as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ first.rotate_left(17)
        ^ last.rotate_left(41)
        ^ reason_tag
}

fn forced_split_oversized_leaf(
    dataset: &dyn PointStore,
    metric: Metric,
    params: &ForgeANNParams,
    leaf: Vec<u32>,
    depth: usize,
    hard_cap: usize,
    seed: u64,
) -> AnnResult<Vec<Vec<u32>>> {
    let n = leaf.len();
    if n <= hard_cap {
        return Ok(vec![leaf]);
    }

    let max_leaders = params.max_leaders.max(2).min(n);
    let mut num_leaders = n.div_ceil(hard_cap).clamp(2, max_leaders);
    let local_fanout_base = params.adaptive_fanout(n, depth).max(1);
    let mut best_split: Option<Vec<Vec<u32>>> = None;
    let mut best_max_child = usize::MAX;
    let mut rng = StdRng::seed_from_u64(seed);

    for _ in 0..FORCED_LEAF_SPLIT_RETRY_LIMIT {
        let sample_seed: u64 = rng.random();
        let leaders = sample_set_bottomk(&leaf, num_leaders, sample_seed);
        let local_fanout = local_fanout_base.min(leaders.len()).max(1);

        let clusters = if metric == Metric::L2 && n >= 256 {
            if params.strict_oom_prefetch_pipeline_enabled() {
                let context = AssignmentContext::new_for_assignment_with_adsampling_dataset(
                    depth, params, None,
                );
                compute_clusters_gemm_budgeted_with_context(
                    dataset,
                    &leaf,
                    &leaders,
                    local_fanout,
                    params
                        .oom_enable
                        .then(|| params.effective_oom_memory_budget_bytes()),
                    Some(&context),
                )?
                .0
            } else {
                compute_clusters_gemm(dataset, &leaf, &leaders, local_fanout)?.0
            }
        } else {
            let avg_cluster_size = (n / num_leaders).max(16);
            let mut clusters: Vec<Vec<u32>> = (0..num_leaders)
                .map(|_| Vec::with_capacity(avg_cluster_size))
                .collect();
            for &idx in &leaf {
                let mut top_k = StackTopK::new(local_fanout);
                for (lid, &lid_global) in leaders.iter().enumerate() {
                    let d = dataset.get_distance(idx, lid_global, metric)?;
                    top_k.push(d, lid);
                }
                for &(_, lid) in top_k.iter() {
                    clusters[lid].push(idx);
                }
            }
            clusters
        };

        let mut merged = merge_clusters(clusters, params.c_min.min(hard_cap), hard_cap);
        if should_dedup_cluster(params, depth, local_fanout) {
            merged.par_iter_mut().for_each(|cluster| {
                cluster.sort_unstable();
                cluster.dedup();
            });
        }
        merged.retain(|cluster| !cluster.is_empty());

        let max_child = merged.iter().map(Vec::len).max().unwrap_or(0);
        if merged.len() > 1 && max_child < n {
            if max_child <= hard_cap {
                return Ok(merged);
            }
            if max_child < best_max_child {
                best_max_child = max_child;
                best_split = Some(merged);
            }
        }

        if num_leaders >= max_leaders {
            break;
        }

        let next_leaders = (num_leaders.saturating_mul(2))
            .min(max_leaders)
            .max(num_leaders + 1);
        if next_leaders == num_leaders {
            break;
        }
        num_leaders = next_leaders;
    }

    if let Some(split) = best_split {
        return Ok(split);
    }

    Ok(leaf.chunks(hard_cap).map(|chunk| chunk.to_vec()).collect())
}

fn emit_leaf_with_safety(
    dataset: &dyn PointStore,
    metric: Metric,
    params: &ForgeANNParams,
    leaf: &mut Vec<u32>,
    depth: usize,
    reason: LeafReason,
    max_leaf_size: usize,
    stats: &mut PartitionStats,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<()> {
    if leaf.is_empty() {
        return Ok(());
    }

    let forced_leaf_hard_cap = FORCED_LEAF_HARD_CAP.min(max_leaf_size.max(1));
    if leaf.len() > forced_leaf_hard_cap {
        stats.forced_leaf_splits += 1;
        let split_seed = forced_leaf_split_seed(leaf, depth, reason);
        let split_leaves = forced_split_oversized_leaf(
            dataset,
            metric,
            params,
            std::mem::take(leaf),
            depth,
            forced_leaf_hard_cap,
            split_seed,
        )?;

        for mut split_leaf in split_leaves {
            emit_leaf_with_safety(
                dataset,
                metric,
                params,
                &mut split_leaf,
                depth,
                reason,
                max_leaf_size,
                stats,
                leaf_emitter,
            )?;
        }
        return Ok(());
    }

    if leaf.len() > max_leaf_size {
        stats.record_oversized_leaf(leaf.len());
    }

    stats.record_leaf(depth, leaf.len(), reason);
    let leaf_to_emit = std::mem::take(leaf);
    if let Some(leaf_to_emit) = leaf_emitter.emit_leaf_from_dataset(dataset, leaf_to_emit)? {
        leaf_emitter.emit_leaf(leaf_to_emit)?;
    }
    Ok(())
}

fn log_final_summary(
    stats: &PartitionStats,
    total_size: usize,
    adaptive_c_max: usize,
    min_recurse_size: usize,
) {
    let duration = format_duration(stats.elapsed());
    let total_leaves = stats.total_leaves();
    let total_points = stats.total_points_in_leaves();
    let min_leaf_size = stats.leaf_sizes.iter().min().copied().unwrap_or(0);
    let max_leaf_size = stats.leaf_sizes.iter().max().copied().unwrap_or(0);
    let avg_leaf_size = if total_leaves > 0 {
        total_points as f64 / total_leaves as f64
    } else {
        0.0
    };

    let merge_reduction = if stats.pre_merge_clusters_total > 0 {
        100.0
            - (stats.post_merge_clusters_total as f64 / stats.pre_merge_clusters_total as f64
                * 100.0)
    } else {
        0.0
    };
    let avg_leaders = if stats.partition_attempts > 0 {
        stats.leader_sum as f64 / stats.partition_attempts as f64
    } else {
        0.0
    };
    let avg_fanout = if stats.partition_attempts > 0 {
        stats.fanout_sum as f64 / stats.partition_attempts as f64
    } else {
        0.0
    };

    tracing::info!("========== RBC Partition Complete ==========");
    tracing::info!("Duration: {}", duration);
    tracing::info!("Points: {}", format_count(total_size));
    tracing::info!("Leaf points emitted: {}", format_count(total_points));
    tracing::info!("Leaves: {}", format_count(total_leaves));
    tracing::info!("Adaptive c_max: {}", format_count(adaptive_c_max));
    tracing::info!("Min recurse size: {}", format_count(min_recurse_size));
    tracing::info!("Max depth: {}", format_count(stats.max_depth_seen));
    tracing::info!("Max stack: {}", format_count(stats.max_stack_size));
    tracing::info!(
        "Max cluster size: {}",
        format_count(stats.max_cluster_size_seen)
    );
    tracing::info!(
        "Oversized leaf events: {} (forced_splits={}, total_points={}, largest={})",
        format_count(stats.oversized_leaf_events),
        format_count(stats.forced_leaf_splits),
        format_count(stats.oversized_leaf_points),
        format_count(stats.largest_oversized_leaf),
    );

    tracing::info!("Leaf Termination Reasons:");
    tracing::info!(
        "  Natural size (<=c_max): {} ({:.1}%)",
        format_count(stats.natural_leaf_size),
        percentage(stats.natural_leaf_size, total_leaves),
    );
    tracing::info!(
        "  Min recurse (<=2xc_max): {} ({:.1}%)",
        format_count(stats.early_stop_min_recurse),
        percentage(stats.early_stop_min_recurse, total_leaves),
    );
    tracing::info!(
        "  Shrink ratio insufficient: {} ({:.1}%)",
        format_count(stats.early_stop_shrink_ratio),
        percentage(stats.early_stop_shrink_ratio, total_leaves),
    );
    tracing::info!(
        "  Small deep cluster: {} ({:.1}%)",
        format_count(stats.early_stop_small_deep),
        percentage(stats.early_stop_small_deep, total_leaves),
    );
    tracing::info!(
        "  Max depth reached: {} ({:.1}%)",
        format_count(stats.early_stop_max_depth),
        percentage(stats.early_stop_max_depth, total_leaves),
    );
    tracing::info!(
        "  Partition fallback: {} ({:.1}%)",
        format_count(stats.partition_fallback_leaves),
        percentage(stats.partition_fallback_leaves, total_leaves),
    );

    tracing::info!("Partitioning Efficiency:");
    tracing::info!(
        "  Total attempts: {}",
        format_count(stats.partition_attempts)
    );
    tracing::info!(
        "  Success (first try): {} ({:.1}%)",
        format_count(stats.partition_success_first),
        percentage(stats.partition_success_first, stats.partition_attempts),
    );
    tracing::info!(
        "  Success (after retry): {} ({:.1}%)",
        format_count(stats.partition_success_retry),
        percentage(stats.partition_success_retry, stats.partition_attempts),
    );
    tracing::info!(
        "  Failed (empty merge): {} ({:.1}%)",
        format_count(stats.partition_failed_empty),
        percentage(stats.partition_failed_empty, stats.partition_attempts),
    );
    tracing::info!(
        "  Failed (no split): {} ({:.1}%)",
        format_count(stats.partition_failed_nosplit),
        percentage(stats.partition_failed_nosplit, stats.partition_attempts),
    );
    tracing::info!(
        "  Retry escalations: {}",
        format_count(stats.retry_escalations)
    );

    tracing::info!("Merge Effectiveness:");
    tracing::info!(
        "  Pre-merge clusters: {}",
        format_count(stats.pre_merge_clusters_total)
    );
    tracing::info!(
        "  Post-merge clusters: {} ({:.1}% reduction)",
        format_count(stats.post_merge_clusters_total),
        merge_reduction,
    );
    tracing::info!(
        "  Empty clusters dropped: {}",
        format_count(stats.empty_clusters_dropped)
    );
    tracing::info!(
        "  No-op merges: {} ({:.1}%)",
        format_count(stats.no_op_merges),
        percentage(stats.no_op_merges, stats.partition_attempts),
    );

    tracing::info!("Adaptive Parameters:");
    tracing::info!("  Leaders per attempt (avg): {:.1}", avg_leaders);
    tracing::info!("  Max leaders used: {}", format_count(stats.leader_max));
    tracing::info!("  Fanout per attempt (avg): {:.1}", avg_fanout);
    tracing::info!("  Max fanout used: {}", format_count(stats.fanout_max));

    tracing::info!("Overlap Analysis:");
    tracing::info!(
        "  Assignments before dedup: {}",
        format_count(stats.assignments_before_dedup),
    );
    tracing::info!(
        "  Assignments after dedup: {}",
        format_count(stats.assignments_after_dedup),
    );
    tracing::info!("  Overlap ratio: {:.1}%", stats.overlap_ratio());
    tracing::info!("Dedup Activity:");
    tracing::info!(
        "  Runs: {} | Skipped: {}",
        format_count(stats.dedup_runs),
        format_count(stats.dedup_skipped),
    );
    tracing::info!(
        "  Assignments before/after: {} -> {} (removed={} / {:.3}%)",
        format_count(stats.dedup_assignments_before),
        format_count(stats.dedup_assignments_after),
        format_count(stats.dedup_assignments_removed),
        percentage(
            stats.dedup_assignments_removed,
            stats.dedup_assignments_before
        ),
    );
    tracing::info!("Phase Timing:");
    tracing::info!("  Cur dedup: {}", format_duration(stats.cur_dedup_time));
    tracing::info!(
        "  Cluster assign: {}",
        format_duration(stats.cluster_assign_time)
    );
    tracing::info!("  Merge clusters: {}", format_duration(stats.merge_time));
    tracing::info!(
        "  Merged dedup: {}",
        format_duration(stats.merged_dedup_time)
    );
    tracing::info!(
        "  GEMM profile: total={} build_x={} gemm={} topk={} merge={} blocks={}",
        format_duration(stats.gemm_profile.total_wall),
        format_duration(stats.gemm_profile.build_x),
        format_duration(stats.gemm_profile.gemm),
        format_duration(stats.gemm_profile.topk),
        format_duration(stats.gemm_profile.merge),
        format_count(stats.gemm_profile.blocks),
    );
    tracing::info!(
        "                spool_write={} flush={}",
        format_duration(stats.gemm_profile.spool_write),
        format_duration(stats.gemm_profile.flush),
    );
    tracing::info!(
        "  Prefetch pipeline: mode={} queue_depth={} budget={} used_peak={} batches={} io={} consumer_wait={} producer_wait={} budget_wait={} budget_fallbacks={} read_amp={:.3} avg_read_size={}",
        stats.gemm_profile.prefetch.pipeline.as_str(),
        stats.gemm_profile.prefetch.queue_depth,
        format_count(stats.gemm_profile.prefetch.prefetch_budget_bytes),
        format_count(stats.gemm_profile.prefetch.prefetch_used_peak_bytes),
        format_count(stats.gemm_profile.prefetch.batches),
        format_duration(stats.gemm_profile.prefetch.io_wall),
        format_duration(stats.gemm_profile.prefetch.consumer_wait),
        format_duration(stats.gemm_profile.prefetch.producer_wait),
        format_duration(stats.gemm_profile.prefetch.budget_wait),
        format_count(stats.gemm_profile.prefetch.fallback_budget_exhausted),
        stats.gemm_profile.prefetch.read_amplification(),
        format_count(stats.gemm_profile.prefetch.avg_read_size_bytes() as usize),
    );
    tracing::info!(
        "  ADS scheduler: large_tasks={} chunks_total={} waves={} exact_fallback_small={} exact_fallback_nested={} collapse_count={}",
        format_count(stats.telemetry.ads_scheduler.ads_large_tasks),
        format_count(stats.telemetry.ads_scheduler.ads_chunks_total),
        format_count(stats.telemetry.ads_scheduler.ads_waves),
        format_count(stats.telemetry.ads_scheduler.ads_exact_fallback_small),
        format_count(stats.telemetry.ads_scheduler.ads_exact_fallback_nested),
        format_count(stats.telemetry.ads_scheduler.ads_parallelism_collapse_count),
    );

    tracing::info!("Leaf Size Distribution:");
    for (label, count) in leaf_size_buckets(&stats.leaf_sizes) {
        tracing::info!(
            "  {}: {} ({:.1}%)",
            label,
            format_count(count),
            percentage(count, total_leaves),
        );
    }
    tracing::info!(
        "  Min/Avg/Max leaf size: {}/{:.1}/{}",
        format_count(min_leaf_size),
        avg_leaf_size,
        format_count(max_leaf_size),
    );

    tracing::info!("==============================================");

    if tracing::enabled!(tracing::Level::DEBUG) {
        tracing::debug!("Depth Distribution:");
        for (depth, depth_stats) in stats.depth_stats.iter().enumerate() {
            if depth_stats.leaves_generated == 0 && depth_stats.clusters_processed == 0 {
                continue;
            }
            let avg_points = if depth_stats.leaves_generated > 0 {
                depth_stats.total_points as f64 / depth_stats.leaves_generated as f64
            } else {
                0.0
            };
            tracing::debug!(
                "  Depth {}: leaves={} processed={} points={} avg_leaf_size={:.1} min={} max={}",
                depth,
                format_count(depth_stats.leaves_generated),
                format_count(depth_stats.clusters_processed),
                format_count(depth_stats.total_points),
                avg_points,
                format_count(depth_stats.min_cluster_size()),
                format_count(depth_stats.max_cluster_size),
            );
        }
    }
}

fn log_root_fanout_summary(profile: &RootFanoutProfile) {
    if !profile.observed() {
        return;
    }
    tracing::info!(
        "[rbc/root-fanout] policy={} semantics={} enabled={} fixed_fanout={} avg_fanout={:.3} selector_ms={} projected_assignment_bytes={} projected_partition_d00_bytes_pre_dedup={} root_leader_hash={}",
        profile.policy,
        profile.selection_semantics,
        profile.enabled,
        profile.fixed_fanout,
        profile.avg_fanout,
        profile.selector_ms,
        profile.projected_assignment_bytes,
        profile.projected_partition_d00_bytes_pre_dedup,
        profile.root_leader_hash,
    );
}

fn leaf_size_buckets(leaf_sizes: &[usize]) -> [(&'static str, usize); 6] {
    let mut counts = [0usize; 6];
    for &size in leaf_sizes {
        let idx = if size <= 1_000 {
            0
        } else if size <= 5_000 {
            1
        } else if size <= 10_000 {
            2
        } else if size <= 20_000 {
            3
        } else if size <= 30_000 {
            4
        } else {
            5
        };
        counts[idx] += 1;
    }

    [
        ("[1-1K]", counts[0]),
        ("[1K-5K]", counts[1]),
        ("[5K-10K]", counts[2]),
        ("[10K-20K]", counts[3]),
        ("[20K-30K]", counts[4]),
        ("[30K+]", counts[5]),
    ]
}

fn percentage(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if hours > 0 {
        format!("{hours}h{minutes:02}m{seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn format_count(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);

    for (idx, ch) in digits.chars().rev().enumerate() {
        if idx != 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }

    out.chars().rev().collect()
}

fn merge_clusters(mut clusters: Vec<Vec<u32>>, c_min: usize, c_max: usize) -> Vec<Vec<u32>> {
    clusters.retain(|cluster| !cluster.is_empty());
    if clusters.is_empty() {
        return clusters;
    }

    clusters.sort_by_key(Vec::len);

    let mut result: Vec<Vec<u32>> = Vec::new();
    let mut current_small: Option<Vec<u32>> = None;

    for cluster in clusters {
        let size = cluster.len();
        if size >= c_min {
            if let Some(mut small) = current_small.take() {
                if small.len() + size <= c_max {
                    let mut merged = cluster;
                    merged.append(&mut small);
                    result.push(merged);
                } else {
                    result.push(small);
                    result.push(cluster);
                }
            } else {
                result.push(cluster);
            }
        } else {
            match current_small {
                Some(small) => {
                    if small.len() + size <= c_max {
                        let mut merged = small;
                        merged.extend_from_slice(&cluster);
                        current_small = Some(merged);
                    } else {
                        result.push(small);
                        current_small = Some(cluster);
                    }
                }
                None => {
                    current_small = Some(cluster);
                }
            }
        }
    }

    if let Some(small) = current_small {
        if let Some(last) = result.last_mut() {
            if last.len() + small.len() <= c_max {
                last.extend_from_slice(&small);
            } else {
                result.push(small);
            }
        } else {
            result.push(small);
        }
    }

    result
}
