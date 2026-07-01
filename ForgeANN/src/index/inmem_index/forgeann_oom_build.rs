use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use super::{DirectMemGraphWriter, FixedDegreeMemGraphWriter, load_fixed_degree_mem_graph};
use crate::common::{AnnError, AnnResult, Metric};
use crate::forgeann::direct_io::DirectIoConfig;
use crate::forgeann::final_prune::final_robust_prune_rows;
use crate::forgeann::point_store::{DirectPointStore, PointStore};
use crate::forgeann::spine_overlay_prune::{
    RuntimeSpineOverlayAccumulator, SpineOverlayPruneOptions,
    build_spine_overlay_prune_graph_from_runtime_accumulator, validate_spine_overlay_options,
};
use crate::forgeann::view_lune_prune::{
    RuntimeViewLuneAccumulator, SpillingViewLuneRecorder, ViewLunePruneOptions,
    ViewLuneRawGraphShard, validate_view_lune_options,
};
use crate::forgeann::{
    ForgeANNParams, build_forgeann_graph_with_store,
    build_forgeann_graph_with_store_and_spine_overlay_recorder,
    build_forgeann_graph_with_store_and_view_lune_recorder,
};
use crate::model::InmemDataset;
use crate::model::graph::AdjacencyList;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeAnnGraphBuildConfig {
    pub metric: Metric,
    pub dim: usize,
    pub aligned_dim: usize,
    pub num_points: usize,
    pub max_degree: usize,
    pub num_threads: u32,
}

impl ForgeAnnGraphBuildConfig {
    pub fn new(
        metric: Metric,
        dim: usize,
        aligned_dim: usize,
        num_points: usize,
        max_degree: usize,
        num_threads: u32,
    ) -> AnnResult<Self> {
        if num_points == 0 {
            return Err(AnnError::log_index_config_error(
                "num_points".to_string(),
                "ForgeANN graph build requires at least one point".to_string(),
            ));
        }
        if dim == 0 {
            return Err(AnnError::log_index_config_error(
                "dim".to_string(),
                "ForgeANN graph build requires positive dimension".to_string(),
            ));
        }
        if aligned_dim < dim {
            return Err(AnnError::log_index_config_error(
                "aligned_dim".to_string(),
                "aligned_dim must be >= dim".to_string(),
            ));
        }
        if max_degree == 0 {
            return Err(AnnError::log_index_config_error(
                "max_degree".to_string(),
                "ForgeANN graph build requires positive max_degree".to_string(),
            ));
        }

        Ok(Self {
            metric,
            dim,
            aligned_dim,
            num_points,
            max_degree,
            num_threads,
        })
    }
}

pub fn build_forgeann_graph_oom_to_file(
    dataset_path: &Path,
    config: &ForgeAnnGraphBuildConfig,
    params: &ForgeANNParams,
    graph_file: &Path,
) -> AnnResult<()> {
    let io_cfg = if params.strict_oom_io_enabled() {
        DirectIoConfig::enabled_with_alignment(4096)
    } else {
        DirectIoConfig::disabled()
    };
    let graph_dataset = DirectPointStore::open_with_config(dataset_path, io_cfg)?;
    if graph_dataset.len() != config.num_points || graph_dataset.dim() != config.dim {
        return Err(AnnError::log_index_config_error(
            "data_path".to_string(),
            format!(
                "Point store metadata mismatch: config rows={} dim={} store rows={} dim={}",
                config.num_points,
                config.dim,
                graph_dataset.len(),
                graph_dataset.dim()
            ),
        ));
    }
    tracing::info!(
        "ForgeANN production OOM graph build uses point-store I/O: direct_io={} alignment={}",
        graph_dataset.io_config().enabled,
        graph_dataset.io_config().alignment
    );
    if params.view_lune_prune_enable || params.view_lune_oracle_json.is_some() {
        if params.final_prune_enable || params.spine_overlay_prune_enable {
            return Err(AnnError::log_index_config_error(
                "view_lune_prune_enable".to_string(),
                "ViewLunePrune is mutually exclusive with final RobustPrune and SpineOverlayPrune"
                    .to_string(),
            ));
        }
        return build_forgeann_graph_oom_to_file_view_lune(
            &graph_dataset,
            config,
            params,
            graph_file,
        );
    }
    if params.spine_overlay_prune_enable {
        return build_forgeann_graph_oom_to_file_spine_overlay(
            &graph_dataset,
            config,
            params,
            graph_file,
        );
    }
    if !params.final_prune_enable {
        tracing::info!(
            "ForgeANN production OOM final RobustPrune disabled; writing no-final-prune graph directly"
        );
        let start = graph_start_from_params_or_medoid(&graph_dataset, config, params)?;
        let mut writer = DirectMemGraphWriter::create(graph_file, start, 0)?;
        let build_summary = build_forgeann_graph_with_store(
            &graph_dataset,
            config.num_points,
            config.num_threads,
            params,
            |vid, neighbors| {
                let mut adjacency = AdjacencyList::for_range(config.max_degree);
                for neighbor in neighbors.into_iter().take(config.max_degree) {
                    adjacency.push(neighbor);
                }
                writer.write_node(vid, adjacency.as_slice())?;
                Ok(())
            },
        )?;
        if let Some(hint) = build_summary.graph_start_hint {
            tracing::info!(
                "ForgeANN production OOM sampled medoid hint observed after direct writer start selection: hint={} writer_start={}",
                hint,
                start
            );
        }
        writer.finish()?;

        tracing::info!("ForgeANN lightweight OOM graph build done via direct point store");
        tracing::info!(
            "ForgeANN production no-final-prune OOM graph build done: num_points={} output={:?}",
            config.num_points,
            graph_file
        );
        return Ok(());
    }
    if params.strict_oom_io_enabled() {
        tracing::info!(
            "ForgeANN production OOM final RobustPrune reads the mmap dataset directly and is intentionally not strict-I/O"
        );
    }

    let pre_prune_graph_file = graph_file.with_extension("pre_prune.tmp");
    let pre_prune_graph = FixedDegreeMemGraphWriter::create(
        &pre_prune_graph_file,
        config.num_points,
        config.max_degree,
    )?;
    tracing::info!(
        "ForgeANN production OOM graph build writes pre-prune graph directly: path={:?} max_degree={}",
        pre_prune_graph_file,
        config.max_degree
    );
    let build_summary = build_forgeann_graph_with_store(
        &graph_dataset,
        config.num_points,
        config.num_threads,
        params,
        |vid, neighbors| {
            let mut adjacency = AdjacencyList::for_range(config.max_degree);
            for neighbor in neighbors.into_iter().take(config.max_degree) {
                adjacency.push(neighbor);
            }
            pre_prune_graph.write_node(vid, adjacency.as_slice())?;
            Ok(())
        },
    )?;
    pre_prune_graph.finish()?;

    let start = if let Some(start) = validate_graph_start_hint(params.graph_start_hint, config)? {
        tracing::info!(
            "ForgeANN production OOM graph start reuses explicit hint: start={}",
            start
        );
        start
    } else if let Some(start) = build_summary.graph_start_hint {
        tracing::info!(
            "ForgeANN production OOM graph start reuses sketch-build sampled medoid: start={}",
            start
        );
        start
    } else {
        tracing::warn!(
            "ForgeANN production OOM graph start hint unavailable; falling back to direct medoid scan"
        );
        graph_dataset.calculate_medoid_point_id_with_threads(Some(config.num_threads))?
    };

    let mut prune_dataset = InmemDataset::<f32>::new_memory_mapped(config.num_points, config.dim)?;
    prune_dataset.set_memory_mapped(dataset_path)?;
    prune_dataset.build_from_file(dataset_path, config.num_points)?;
    let mut graph_rows =
        load_fixed_degree_mem_graph(&pre_prune_graph_file, config.num_points, config.max_degree)?;

    let prune_start = Instant::now();
    final_robust_prune_rows(
        &prune_dataset,
        graph_rows.as_mut_slice(),
        config.max_degree,
        config.metric,
        config.num_threads,
    )?;
    tracing::info!(
        "ForgeANN production final RobustPrune done: elapsed_ms={:.3}",
        prune_start.elapsed().as_secs_f64() * 1000.0
    );

    let mut writer = DirectMemGraphWriter::create(graph_file, start, 0)?;
    for (vid, neighbors) in graph_rows.iter().enumerate() {
        let capped = neighbors
            .get(..config.max_degree.min(neighbors.len()))
            .unwrap_or(&neighbors);
        writer.write_node(vid as u32, capped)?;
    }
    writer.finish()?;
    if let Err(err) = std::fs::remove_file(&pre_prune_graph_file) {
        tracing::warn!(
            "ForgeANN production OOM failed to remove pre-prune graph temp file {:?}: {}",
            pre_prune_graph_file,
            err
        );
    }

    tracing::info!("ForgeANN lightweight OOM graph build done via direct point store");

    tracing::info!(
        "ForgeANN production OOM graph build done: num_points={} output={:?}",
        config.num_points,
        graph_file
    );
    Ok(())
}

fn build_forgeann_graph_oom_to_file_view_lune(
    graph_dataset: &DirectPointStore,
    config: &ForgeAnnGraphBuildConfig,
    params: &ForgeANNParams,
    graph_file: &Path,
) -> AnnResult<()> {
    let options = ViewLunePruneOptions {
        max_degree: config.max_degree,
        metric: config.metric,
        candidate_width_multiplier: params.view_lune_candidate_width_multiplier,
        max_witnesses_per_victim: params.view_lune_max_witnesses_per_victim,
        nearest_core: params.view_lune_nearest_core,
    };
    validate_view_lune_options(&options)?;

    tracing::info!(
        "ViewLunePrune OOM path enabled: executor={} oracle_json={:?} max_degree={} candidate_width_multiplier={:.3} max_witnesses_per_victim={} nearest_core={}",
        params.view_lune_prune_enable,
        params.view_lune_oracle_json,
        options.max_degree,
        options.candidate_width_multiplier,
        options.max_witnesses_per_victim,
        options.nearest_core,
    );

    if view_lune_spill_enabled(params) {
        return build_forgeann_graph_oom_to_file_view_lune_spill(
            graph_dataset,
            config,
            params,
            graph_file,
            options,
        );
    }

    let recorder = RuntimeViewLuneAccumulator::create(config.num_points, options)?;
    tracing::info!(
        "ViewLunePrune runtime accumulator initialized: rows={} candidate_width={}",
        config.num_points,
        recorder.candidate_width()
    );

    let need_base_rows = !params.view_lune_prune_enable || params.view_lune_oracle_json.is_some();
    let mut base_rows = need_base_rows.then(|| Vec::with_capacity(config.num_points));
    let streaming_start = Instant::now();
    let build_summary = build_forgeann_graph_with_store_and_view_lune_recorder(
        graph_dataset,
        config.num_points,
        config.num_threads,
        params,
        &recorder,
        |vid, neighbors| {
            if let Some(base_rows) = base_rows.as_mut() {
                if vid as usize != base_rows.len() {
                    return Err(AnnError::log_index_error(format!(
                        "ViewLunePrune expected sequential base row {}, got {}",
                        base_rows.len(),
                        vid
                    )));
                }
                base_rows.push(neighbors.into_iter().take(config.max_degree).collect());
            }
            Ok(())
        },
    )?;
    tracing::info!(
        "ViewLunePrune base no-final row capture: enabled={} elapsed_ms={:.3} rows={}",
        need_base_rows,
        streaming_start.elapsed().as_secs_f64() * 1000.0,
        base_rows.as_ref().map_or(0, Vec::len)
    );

    let start = if let Some(start) = validate_graph_start_hint(params.graph_start_hint, config)? {
        tracing::info!(
            "ForgeANN production OOM graph start reuses explicit hint: start={}",
            start
        );
        start
    } else if let Some(start) = build_summary.graph_start_hint {
        tracing::info!(
            "ForgeANN production OOM graph start reuses sketch-build sampled medoid: start={}",
            start
        );
        start
    } else {
        tracing::warn!(
            "ForgeANN production OOM graph start hint unavailable; falling back to direct medoid scan"
        );
        graph_dataset.calculate_medoid_point_id_with_threads(Some(config.num_threads))?
    };

    let base_rows_for_compare = base_rows
        .as_ref()
        .and_then(|rows| (rows.len() == config.num_points).then_some(rows.as_slice()));
    let compact_start = Instant::now();
    let compaction = recorder.compact_storage();
    tracing::info!(
        "ViewLunePrune accumulator compacted: elapsed_ms={:.3} candidate_len={} candidate_capacity_before={} candidate_capacity_after={} candidate_slots_released={} witness_row_len={} witness_row_capacity_before={} witness_row_capacity_after={} witness_row_slots_released={} heap_witness_capacity_before={} heap_witness_capacity_after={} heap_witness_slots_released={} estimated_bytes_released={}",
        compact_start.elapsed().as_secs_f64() * 1000.0,
        compaction.candidate_len,
        compaction.candidate_capacity_before,
        compaction.candidate_capacity_after,
        compaction.candidate_slots_released(),
        compaction.witness_row_len,
        compaction.witness_row_capacity_before,
        compaction.witness_row_capacity_after,
        compaction.witness_row_slots_released(),
        compaction.heap_witness_capacity_before,
        compaction.heap_witness_capacity_after,
        compaction.heap_witness_slots_released(),
        compaction.estimated_bytes_released(),
    );
    let reduce_start = Instant::now();
    let mut writer = DirectMemGraphWriter::create(graph_file, start, 0)?;
    let (stats, overlap_edges, overlap_rows) = if params.view_lune_prune_enable {
        recorder.reduce_all_rows_with(base_rows_for_compare, |vid, row| {
            writer.write_node(vid as u32, row)
        })?
    } else {
        let reduced = recorder.reduce_all_rows_with(base_rows_for_compare, |_, _| Ok(()))?;
        let Some(base_rows) = base_rows.as_ref() else {
            return Err(AnnError::log_index_error(
                "ViewLunePrune oracle-only path did not capture base rows".to_string(),
            ));
        };
        if base_rows.len() != config.num_points {
            return Err(AnnError::log_index_error(format!(
                "ViewLunePrune oracle-only path expected {} base rows, got {}",
                config.num_points,
                base_rows.len()
            )));
        }
        for (vid, row) in base_rows.iter().enumerate() {
            writer.write_node(vid as u32, row)?;
        }
        reduced
    };
    tracing::info!(
        "ViewLunePrune metadata reduce done: elapsed_ms={:.3} raw_candidates={} raw_witnesses={} filtered_witnesses={} merged_candidates={} victims_with_witness={} stale_witness_rows_pruned={} pruned_by_selected_witness={} fill_after_prune={} rows_modified={} final_edges={} accumulator_rss_estimate_bytes={} reducer_ms={:.3} final_vector_io_count=0",
        reduce_start.elapsed().as_secs_f64() * 1000.0,
        stats.raw_candidate_edges,
        stats.raw_witness_records,
        stats.filtered_witness_records,
        stats.merged_candidates,
        stats.victims_with_witness,
        stats.stale_witness_rows_pruned,
        stats.pruned_by_selected_witness,
        stats.fill_after_prune,
        stats.rows_modified,
        stats.final_edges,
        stats.estimated_accumulator_bytes,
        stats.reduce_wall.as_secs_f64() * 1000.0,
    );
    writer.finish()?;

    if let Some(path) = params.view_lune_oracle_json.as_ref() {
        let report = recorder.write_oracle_json_from_stats(
            path,
            stats,
            overlap_edges,
            overlap_rows,
            base_rows_for_compare.map_or(0, |rows| rows.len()),
        )?;
        tracing::info!(
            "ViewLunePrune oracle report written: path={:?} schema={} overlap_edges={} overlap_rows={} rows_compared={}",
            path,
            report.schema,
            report.overlap_with_base_edges,
            report.overlap_with_base_rows,
            report.rows_compared_with_base,
        );
    }

    tracing::info!("ForgeANN lightweight OOM graph build done via direct point store");
    tracing::info!(
        "ForgeANN production ViewLunePrune OOM graph build done: executor={} num_points={} output={:?} overlap_edges={} overlap_rows={}",
        params.view_lune_prune_enable,
        config.num_points,
        graph_file,
        overlap_edges,
        overlap_rows,
    );
    Ok(())
}

fn build_forgeann_graph_oom_to_file_view_lune_spill(
    graph_dataset: &DirectPointStore,
    config: &ForgeAnnGraphBuildConfig,
    params: &ForgeANNParams,
    graph_file: &Path,
    options: ViewLunePruneOptions,
) -> AnnResult<()> {
    let artifact_dir = view_lune_spill_artifact_dir(params, graph_file);
    std::fs::create_dir_all(&artifact_dir)?;
    let shard_count = SpillingViewLuneRecorder::default_shard_count(config.num_points);
    let recorder =
        SpillingViewLuneRecorder::create(config.num_points, options, &artifact_dir, shard_count)?;
    tracing::info!(
        "ViewLunePrune spill recorder initialized: rows={} shards={} shard_rows={} buffer_records={} candidate_digest={} candidate_digest_budget_bytes={} artifact_dir={:?}",
        config.num_points,
        recorder.shard_count(),
        recorder.shard_rows(),
        recorder.buffer_records(),
        recorder.candidate_digest_enabled(),
        recorder.candidate_digest_budget_bytes(),
        artifact_dir,
    );

    let streaming_start = Instant::now();
    let build_summary = build_forgeann_graph_with_store_and_view_lune_recorder(
        graph_dataset,
        config.num_points,
        config.num_threads,
        params,
        &recorder,
        |_vid, _neighbors| Ok(()),
    )?;
    recorder.finish_writes()?;
    tracing::info!(
        "ViewLunePrune spilled metadata capture done: elapsed_ms={:.3} rows={} shards={}",
        streaming_start.elapsed().as_secs_f64() * 1000.0,
        config.num_points,
        recorder.shard_count(),
    );

    let start = if let Some(start) = validate_graph_start_hint(params.graph_start_hint, config)? {
        tracing::info!(
            "ForgeANN production OOM graph start reuses explicit hint: start={}",
            start
        );
        start
    } else if let Some(start) = build_summary.graph_start_hint {
        tracing::info!(
            "ForgeANN production OOM graph start reuses sketch-build sampled medoid: start={}",
            start
        );
        start
    } else {
        tracing::warn!(
            "ForgeANN production OOM graph start hint unavailable; falling back to direct medoid scan"
        );
        graph_dataset.calculate_medoid_point_id_with_threads(Some(config.num_threads))?
    };

    let reduce_start = Instant::now();
    let graph_shard_dir = artifact_dir.join("graph_rows");
    let (stats, graph_shards) = recorder.reduce_all_rows_to_raw_graph_shards_parallel(
        &graph_shard_dir,
        config.num_threads as usize,
    )?;
    write_raw_graph_shards_to_mem_index(graph_file, start, &graph_shards)?;
    tracing::info!(
        "ViewLunePrune spilled metadata reduce done: elapsed_ms={:.3} raw_candidates={} raw_witnesses={} filtered_witnesses={} merged_candidates={} victims_with_witness={} stale_witness_rows_pruned={} pruned_by_selected_witness={} fill_after_prune={} rows_modified={} final_edges={} peak_shard_accumulator_bytes={} reducer_ms={:.3} graph_shards={} final_vector_io_count=0",
        reduce_start.elapsed().as_secs_f64() * 1000.0,
        stats.raw_candidate_edges,
        stats.raw_witness_records,
        stats.filtered_witness_records,
        stats.merged_candidates,
        stats.victims_with_witness,
        stats.stale_witness_rows_pruned,
        stats.pruned_by_selected_witness,
        stats.fill_after_prune,
        stats.rows_modified,
        stats.final_edges,
        stats.estimated_accumulator_bytes,
        stats.reduce_wall.as_secs_f64() * 1000.0,
        graph_shards.len(),
    );
    if !params.oom_keep_artifacts {
        recorder.cleanup()?;
        cleanup_view_lune_graph_shards(&graph_shards)?;
        if let Err(err) = std::fs::remove_dir(&graph_shard_dir) {
            if err.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    "ViewLunePrune failed to remove graph shard dir {:?}: {}",
                    graph_shard_dir,
                    err
                );
            }
        }
        if let Err(err) = std::fs::remove_dir(&artifact_dir) {
            if err.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    "ViewLunePrune failed to remove spill artifact dir {:?}: {}",
                    artifact_dir,
                    err
                );
            }
        }
    }

    tracing::info!("ForgeANN lightweight OOM graph build done via direct point store");
    tracing::info!(
        "ForgeANN production spilled ViewLunePrune OOM graph build done: num_points={} output={:?} overlap_edges={} overlap_rows={}",
        config.num_points,
        graph_file,
        0,
        0,
    );
    Ok(())
}

fn write_raw_graph_shards_to_mem_index(
    graph_file: &Path,
    start: u32,
    graph_shards: &[ViewLuneRawGraphShard],
) -> AnnResult<()> {
    let row_bytes: u64 = graph_shards.iter().map(|shard| shard.bytes).sum();
    let index_size = 24u64.checked_add(row_bytes).ok_or_else(|| {
        AnnError::log_index_error("ViewLune graph index size overflow".to_string())
    })?;
    let max_degree = graph_shards
        .iter()
        .map(|shard| shard.max_degree)
        .max()
        .unwrap_or(0);
    let mut writer = BufWriter::new(File::create(graph_file)?);
    writer.write_all(&index_size.to_le_bytes())?;
    writer.write_all(&max_degree.to_le_bytes())?;
    writer.write_all(&start.to_le_bytes())?;
    writer.write_all(&0u64.to_le_bytes())?;
    for shard in graph_shards {
        let mut reader = BufReader::new(File::open(&shard.path)?);
        std::io::copy(&mut reader, &mut writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn cleanup_view_lune_graph_shards(graph_shards: &[ViewLuneRawGraphShard]) -> AnnResult<()> {
    for shard in graph_shards {
        match std::fs::remove_file(&shard.path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

fn build_forgeann_graph_oom_to_file_spine_overlay(
    graph_dataset: &DirectPointStore,
    config: &ForgeAnnGraphBuildConfig,
    params: &ForgeANNParams,
    graph_file: &Path,
) -> AnnResult<()> {
    let options = SpineOverlayPruneOptions {
        max_degree: config.max_degree,
        metric: config.metric,
        overlay_budget: params.spine_overlay_budget,
        overlay_spine_fraction: params.spine_overlay_spine_fraction,
    };
    validate_spine_overlay_options(&options)?;

    let artifact_dir = spine_overlay_artifact_dir(params, graph_file);
    std::fs::create_dir_all(&artifact_dir)?;
    tracing::info!(
        "SpineOverlayPrune OOM final graph path enabled: artifact_dir={:?} max_degree={} overlay_budget={} overlay_spine_fraction={:.3}",
        artifact_dir,
        options.max_degree,
        options.overlay_budget,
        options.overlay_spine_fraction,
    );

    let recorder = RuntimeSpineOverlayAccumulator::create(config.num_points, options)?;
    tracing::info!(
        "SpineOverlayPrune runtime accumulator initialized: rows={} runtime_capacity={}",
        config.num_points,
        recorder.capacity()
    );

    let mut base_rows: Vec<Vec<u32>> = Vec::with_capacity(config.num_points);
    let streaming_start = Instant::now();
    let build_summary = build_forgeann_graph_with_store_and_spine_overlay_recorder(
        graph_dataset,
        config.num_points,
        config.num_threads,
        params,
        &recorder,
        |vid, neighbors| {
            if vid as usize != base_rows.len() {
                return Err(AnnError::log_index_error(format!(
                    "SpineOverlayPrune expected sequential base row {}, got {}",
                    base_rows.len(),
                    vid
                )));
            }
            base_rows.push(neighbors.into_iter().take(config.max_degree).collect());
            Ok(())
        },
    )?;
    tracing::info!(
        "SpineOverlayPrune captured base no-final rows: elapsed_ms={:.3} rows={}",
        streaming_start.elapsed().as_secs_f64() * 1000.0,
        base_rows.len()
    );

    let start = if let Some(start) = validate_graph_start_hint(params.graph_start_hint, config)? {
        tracing::info!(
            "ForgeANN production OOM graph start reuses explicit hint: start={}",
            start
        );
        start
    } else if let Some(start) = build_summary.graph_start_hint {
        tracing::info!(
            "ForgeANN production OOM graph start reuses sketch-build sampled medoid: start={}",
            start
        );
        start
    } else {
        tracing::warn!(
            "ForgeANN production OOM graph start hint unavailable; falling back to direct medoid scan"
        );
        graph_dataset.calculate_medoid_point_id_with_threads(Some(config.num_threads))?
    };

    let reduce_start = Instant::now();
    let mut writer = DirectMemGraphWriter::create(graph_file, start, 0)?;
    let stats = build_spine_overlay_prune_graph_from_runtime_accumulator(
        recorder,
        base_rows.as_slice(),
        config.num_points,
        options,
        |vid, row| writer.write_node(vid, row),
    )?;
    writer.finish()?;
    tracing::info!(
        "SpineOverlayPrune metadata reduce/scoring done: elapsed_ms={:.3} raw_edges={} merged_candidates={} overfull_rows={} final_edges={} rows_with_overlay={} overlay_edges_added={} base_refill_edges={} reducer_ms={:.3}",
        reduce_start.elapsed().as_secs_f64() * 1000.0,
        stats.raw_edges,
        stats.merged_candidates,
        stats.overfull_rows,
        stats.final_edges,
        stats.rows_with_overlay,
        stats.overlay_edges_added,
        stats.base_refill_edges,
        stats.reduce_wall.as_secs_f64() * 1000.0,
    );
    tracing::info!("ForgeANN lightweight OOM graph build done via direct point store");
    tracing::info!(
        "ForgeANN production SpineOverlayPrune OOM graph build done: num_points={} output={:?}",
        config.num_points,
        graph_file
    );
    Ok(())
}

fn validate_graph_start_hint(
    start: Option<u32>,
    config: &ForgeAnnGraphBuildConfig,
) -> AnnResult<Option<u32>> {
    if let Some(start) = start {
        if start as usize >= config.num_points {
            return Err(AnnError::log_index_config_error(
                "graph_start".to_string(),
                format!("graph_start must be < num_points ({})", config.num_points),
            ));
        }
    }
    Ok(start)
}

fn graph_start_from_params_or_medoid(
    graph_dataset: &DirectPointStore,
    config: &ForgeAnnGraphBuildConfig,
    params: &ForgeANNParams,
) -> AnnResult<u32> {
    if let Some(start) = validate_graph_start_hint(params.graph_start_hint, config)? {
        tracing::info!(
            "ForgeANN production OOM graph start reuses explicit hint: start={}",
            start
        );
        return Ok(start);
    }
    let start = graph_dataset.calculate_medoid_point_id_with_threads(Some(config.num_threads))?;
    tracing::info!(
        "ForgeANN production OOM graph start computed by direct medoid scan: start={}",
        start
    );
    Ok(start)
}

fn spine_overlay_artifact_dir(params: &ForgeANNParams, graph_file: &Path) -> std::path::PathBuf {
    if params.oom_temp_dir.as_os_str().is_empty() {
        graph_file.with_extension("spine_overlay_prune_artifacts")
    } else {
        params.oom_temp_dir.join("spine_overlay_prune")
    }
}

fn view_lune_spill_enabled(params: &ForgeANNParams) -> bool {
    params.view_lune_prune_enable
        && params.view_lune_oracle_json.is_none()
        && std::env::var("FORGEANN_VIEW_LUNE_SPILL_DISABLE")
            .ok()
            .as_deref()
            != Some("1")
}

fn view_lune_spill_artifact_dir(params: &ForgeANNParams, graph_file: &Path) -> std::path::PathBuf {
    if params.oom_temp_dir.as_os_str().is_empty() {
        graph_file.with_extension("view_lune_spill_artifacts")
    } else {
        params.oom_temp_dir.join("view_lune_spill")
    }
}
