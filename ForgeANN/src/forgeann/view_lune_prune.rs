use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rayon::prelude::*;

use super::leaf_build::{
    PENDING_EDGE_DIRECT, PENDING_EDGE_MIRROR, PendingEdge, PendingEdgeSink, PendingLuneWitness,
};
use crate::common::{AnnError, AnnResult, Metric};

#[allow(dead_code)]
const FINAL_PRUNE_ALPHA_IMPL: f32 = 1.2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewLunePruneOptions {
    pub max_degree: usize,
    pub metric: Metric,
    pub candidate_width_multiplier: f32,
    pub max_witnesses_per_victim: usize,
    pub nearest_core: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewLunePruneStats {
    pub raw_candidate_edges: usize,
    pub raw_witness_records: usize,
    pub merged_candidates: usize,
    pub victims_with_witness: usize,
    pub distinct_witness_families: usize,
    pub filtered_witness_records: usize,
    pub stale_witness_rows_pruned: usize,
    pub pruned_by_selected_witness: usize,
    pub fill_after_prune: usize,
    pub rows_modified: usize,
    pub final_edges: usize,
    pub rows_with_candidates: usize,
    pub rows_over_candidate_width: usize,
    pub leaf_witness_emit: Duration,
    pub reduce_wall: Duration,
    pub estimated_accumulator_bytes: usize,
}

impl ViewLunePruneStats {
    fn add_shard(&mut self, shard: ViewLunePruneStats) {
        self.merged_candidates += shard.merged_candidates;
        self.victims_with_witness += shard.victims_with_witness;
        self.filtered_witness_records += shard.filtered_witness_records;
        self.stale_witness_rows_pruned += shard.stale_witness_rows_pruned;
        self.pruned_by_selected_witness += shard.pruned_by_selected_witness;
        self.fill_after_prune += shard.fill_after_prune;
        self.rows_modified += shard.rows_modified;
        self.final_edges += shard.final_edges;
        self.rows_with_candidates += shard.rows_with_candidates;
        self.rows_over_candidate_width += shard.rows_over_candidate_width;
        self.leaf_witness_emit += shard.leaf_witness_emit;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewLuneAccumulatorCompactionStats {
    pub candidate_len: usize,
    pub candidate_capacity_before: usize,
    pub candidate_capacity_after: usize,
    pub witness_row_len: usize,
    pub witness_row_capacity_before: usize,
    pub witness_row_capacity_after: usize,
    pub heap_witness_capacity_before: usize,
    pub heap_witness_capacity_after: usize,
}

impl ViewLuneAccumulatorCompactionStats {
    pub fn candidate_slots_released(self) -> usize {
        self.candidate_capacity_before
            .saturating_sub(self.candidate_capacity_after)
    }

    pub fn witness_row_slots_released(self) -> usize {
        self.witness_row_capacity_before
            .saturating_sub(self.witness_row_capacity_after)
    }

    pub fn heap_witness_slots_released(self) -> usize {
        self.heap_witness_capacity_before
            .saturating_sub(self.heap_witness_capacity_after)
    }

    pub fn estimated_bytes_released(self) -> usize {
        self.candidate_slots_released()
            .saturating_mul(std::mem::size_of::<ViewLuneCandidate>())
            .saturating_add(
                self.witness_row_slots_released()
                    .saturating_mul(std::mem::size_of::<ViewLuneVictimWitnesses>()),
            )
            .saturating_add(
                self.heap_witness_slots_released()
                    .saturating_mul(std::mem::size_of::<RuntimeLuneWitness>()),
            )
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LuneWitness {
    pub src: u32,
    pub pivot: u32,
    pub victim: u32,
    pub margin_q16: u16,
    pub pivot_rank: u8,
    pub victim_rank: u8,
    pub view_family: u32,
}

#[derive(Clone, Debug)]
pub struct ViewLuneOracleReport {
    pub schema: &'static str,
    pub options: ViewLunePruneOptions,
    pub stats: ViewLunePruneStats,
    pub overlap_with_base_edges: usize,
    pub overlap_with_base_rows: usize,
    pub rows_compared_with_base: usize,
}

pub(crate) trait ViewLuneEdgeRecorder: Sync {
    fn write_edges(&self, view_family: u32, edges: &[PendingEdge]) -> AnnResult<()>;
    fn write_witnesses(&self, view_family: u32, witnesses: &[PendingLuneWitness]) -> AnnResult<()>;

    fn write_edges_in_place(
        &self,
        view_family: u32,
        edges: &mut Vec<PendingEdge>,
    ) -> AnnResult<()> {
        self.write_edges(view_family, edges.as_slice())
    }

    fn write_witnesses_in_place(
        &self,
        view_family: u32,
        witnesses: &mut Vec<PendingLuneWitness>,
    ) -> AnnResult<()> {
        self.write_witnesses(view_family, witnesses.as_slice())
    }
}

#[derive(Debug)]
pub(crate) struct SpillingViewLuneRecorder {
    shards: Vec<Mutex<ViewLuneSpillShard>>,
    num_points: usize,
    shard_rows: usize,
    buffer_records: usize,
    candidate_mode: ViewLuneCandidateSpillMode,
    candidate_digest_budget_bytes: usize,
    options: ViewLunePruneOptions,
    raw_candidate_edges: AtomicUsize,
    raw_witness_records: AtomicUsize,
    witness_families: Mutex<Vec<u32>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ViewLuneRawGraphShard {
    pub path: PathBuf,
    pub bytes: u64,
    pub max_degree: u32,
}

#[derive(Debug)]
struct ViewLuneShardReduceResult {
    stats: ViewLunePruneStats,
    graph_shard: ViewLuneRawGraphShard,
}

#[derive(Debug)]
struct ViewLuneSpillShard {
    edge_path: PathBuf,
    witness_path: PathBuf,
    candidate_run_dir: PathBuf,
    edge_writer: BufWriter<File>,
    witness_writer: BufWriter<File>,
    edge_records: usize,
    witness_records: usize,
    edge_buffer: Vec<BufferedSpilledEdge>,
    witness_buffer: Vec<PendingLuneWitness>,
    candidate_digest: Option<CandidateDigestRun>,
    candidate_run_paths: Vec<PathBuf>,
    candidate_run_bytes: u64,
    candidate_run_rows: usize,
    candidate_run_records: usize,
    candidate_run_flushes: usize,
}

#[derive(Clone, Copy, Debug)]
struct BufferedSpilledEdge {
    local_source: u32,
    dst: u32,
    dist: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewLuneCandidateSpillMode {
    RawEdges,
    DigestRuns,
}

#[derive(Debug)]
struct CandidateDigestRun {
    rows: Vec<CandidateDigestRow>,
    candidates: Vec<ViewLuneCandidate>,
    touched_rows: Vec<u32>,
    candidate_slots: usize,
    budget_slots: usize,
    candidate_width: usize,
}

#[derive(Debug)]
struct CandidateDigestRow {
    offset: u32,
    len: u16,
    capacity: u16,
    candidate_worst_idx: u16,
    touched: bool,
}

const DEFAULT_VIEW_LUNE_SPILL_SHARDS: usize = 16;
const DEFAULT_VIEW_LUNE_SPILL_BUFFER_RECORDS: usize = 1 << 20;
const DEFAULT_VIEW_LUNE_CANDIDATE_DIGEST_BUDGET_MB: usize = 64;
const VIEW_LUNE_EDGE_SPILL_RECORD_BYTES: usize = 12;
const VIEW_LUNE_WITNESS_SPILL_RECORD_BYTES: usize = 16;
const VIEW_LUNE_CANDIDATE_RUN_ROW_HEADER_BYTES: usize = 8;
const VIEW_LUNE_CANDIDATE_RUN_RECORD_BYTES: usize = 8;
const EMPTY_VIEW_LUNE_CANDIDATE: ViewLuneCandidate = ViewLuneCandidate {
    dst: 0,
    dist2: f32::INFINITY,
};

impl CandidateDigestRun {
    fn new(shard_len: usize, candidate_width: usize, budget_bytes: usize) -> Self {
        let slot_bytes = std::mem::size_of::<ViewLuneCandidate>().max(1);
        let budget_slots = (budget_bytes / slot_bytes).max(candidate_width.max(1));
        Self {
            rows: (0..shard_len)
                .map(|_| CandidateDigestRow::default())
                .collect(),
            candidates: Vec::new(),
            touched_rows: Vec::new(),
            candidate_slots: 0,
            budget_slots,
            candidate_width,
        }
    }

    fn insert(&mut self, local_source: u32, dst: u32, dist2: f32) {
        let local_source = local_source as usize;
        if local_source >= self.rows.len() {
            return;
        }
        let was_empty = self.rows[local_source].len == 0;
        let added_slot = self.insert_candidate(local_source, ViewLuneCandidate { dst, dist2 });
        if added_slot {
            self.candidate_slots += 1;
        }
        if was_empty && self.rows[local_source].len > 0 && !self.rows[local_source].touched {
            self.rows[local_source].touched = true;
            self.touched_rows.push(local_source as u32);
        }
    }

    fn should_flush(&self) -> bool {
        self.candidates.len() >= self.budget_slots
    }

    fn release_storage(&mut self) {
        for row in &mut self.rows {
            *row = CandidateDigestRow::default();
        }
        self.candidates.clear();
        self.candidates.shrink_to_fit();
        self.touched_rows.clear();
        self.touched_rows.shrink_to_fit();
        self.candidate_slots = 0;
    }

    fn insert_candidate(&mut self, local_source: usize, candidate: ViewLuneCandidate) -> bool {
        if self.candidate_width == 0 {
            return false;
        }
        let insert_idx = match self.binary_search_row_by_dst(local_source, candidate.dst) {
            Ok(idx) => {
                let candidate_idx = self.row_offset(local_source) + idx;
                let existing = &mut self.candidates[candidate_idx];
                existing.dist2 = existing.dist2.min(candidate.dist2);
                if self.rows[local_source].candidate_worst_index() == Some(idx) {
                    self.recompute_candidate_worst(local_source);
                }
                return false;
            }
            Err(idx) => idx,
        };
        let len = self.rows[local_source].len as usize;
        if len < self.candidate_width {
            self.ensure_row_capacity(local_source, len + 1);
            self.insert_row_slot(local_source, insert_idx, candidate);
            self.recompute_candidate_worst(local_source);
            return true;
        }
        let Some(worst_idx) = self.current_candidate_worst_index(local_source) else {
            return false;
        };
        let worst = self.row_candidate(local_source, worst_idx);
        if candidate_distance_cmp(&candidate, &worst).is_lt() {
            self.remove_row_slot(local_source, worst_idx);
            let insert_idx = if worst_idx < insert_idx {
                insert_idx.saturating_sub(1)
            } else {
                insert_idx
            };
            self.insert_row_slot(local_source, insert_idx, candidate);
            self.recompute_candidate_worst(local_source);
        }
        false
    }

    fn ensure_row_capacity(&mut self, local_source: usize, needed: usize) {
        let current_capacity = self.rows[local_source].capacity as usize;
        if current_capacity >= needed {
            return;
        }
        let target_capacity = if current_capacity == 0 {
            CANDIDATE_ROW_CAPACITY_GROWTH
                .max(needed)
                .min(self.candidate_width)
        } else {
            current_capacity
                .saturating_mul(2)
                .max(needed)
                .min(self.candidate_width)
        };
        let old_offset = self.rows[local_source].offset as usize;
        let old_len = self.rows[local_source].len as usize;
        let new_offset = self.candidates.len();
        self.candidates
            .resize(new_offset + target_capacity, EMPTY_VIEW_LUNE_CANDIDATE);
        for idx in 0..old_len {
            self.candidates[new_offset + idx] = self.candidates[old_offset + idx];
        }
        self.rows[local_source].offset = u32::try_from(new_offset).unwrap_or(u32::MAX);
        self.rows[local_source].capacity = u16::try_from(target_capacity).unwrap_or(u16::MAX);
    }

    fn binary_search_row_by_dst(&self, local_source: usize, dst: u32) -> Result<usize, usize> {
        let offset = self.row_offset(local_source);
        let len = self.rows[local_source].len as usize;
        self.candidates[offset..offset + len].binary_search_by_key(&dst, |candidate| candidate.dst)
    }

    fn insert_row_slot(
        &mut self,
        local_source: usize,
        insert_idx: usize,
        candidate: ViewLuneCandidate,
    ) {
        let offset = self.row_offset(local_source);
        let len = self.rows[local_source].len as usize;
        for idx in (insert_idx..len).rev() {
            self.candidates[offset + idx + 1] = self.candidates[offset + idx];
        }
        self.candidates[offset + insert_idx] = candidate;
        self.rows[local_source].len = u16::try_from(len + 1).unwrap_or(u16::MAX);
    }

    fn remove_row_slot(&mut self, local_source: usize, remove_idx: usize) {
        let offset = self.row_offset(local_source);
        let len = self.rows[local_source].len as usize;
        if remove_idx >= len {
            return;
        }
        for idx in remove_idx + 1..len {
            self.candidates[offset + idx - 1] = self.candidates[offset + idx];
        }
        self.rows[local_source].len = u16::try_from(len.saturating_sub(1)).unwrap_or(0);
        self.rows[local_source].candidate_worst_idx = INVALID_CANDIDATE_IDX;
    }

    fn row_offset(&self, local_source: usize) -> usize {
        self.rows[local_source].offset as usize
    }

    fn row_candidate(&self, local_source: usize, idx: usize) -> ViewLuneCandidate {
        self.candidates[self.row_offset(local_source) + idx]
    }

    fn current_candidate_worst_index(&mut self, local_source: usize) -> Option<usize> {
        if self.rows[local_source].candidate_worst_index().is_none() {
            self.recompute_candidate_worst(local_source);
        }
        self.rows[local_source].candidate_worst_index()
    }

    fn recompute_candidate_worst(&mut self, local_source: usize) {
        let offset = self.row_offset(local_source);
        let len = self.rows[local_source].len as usize;
        let Some((idx, _)) = self.candidates[offset..offset + len]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| candidate_distance_cmp(left, right))
        else {
            self.rows[local_source].candidate_worst_idx = INVALID_CANDIDATE_IDX;
            return;
        };
        self.rows[local_source].candidate_worst_idx =
            u16::try_from(idx).unwrap_or(INVALID_CANDIDATE_IDX);
    }
}

impl Default for CandidateDigestRow {
    fn default() -> Self {
        Self {
            offset: 0,
            len: 0,
            capacity: 0,
            candidate_worst_idx: INVALID_CANDIDATE_IDX,
            touched: false,
        }
    }
}

impl CandidateDigestRow {
    fn candidate_worst_index(&self) -> Option<usize> {
        let idx = usize::from(self.candidate_worst_idx);
        (self.candidate_worst_idx != INVALID_CANDIDATE_IDX && idx < usize::from(self.len))
            .then_some(idx)
    }
}

impl SpillingViewLuneRecorder {
    pub(crate) fn create(
        num_points: usize,
        options: ViewLunePruneOptions,
        dir: &Path,
        requested_shards: usize,
    ) -> AnnResult<Self> {
        let candidate_mode = if Self::default_candidate_digest_enabled() {
            ViewLuneCandidateSpillMode::DigestRuns
        } else {
            ViewLuneCandidateSpillMode::RawEdges
        };
        Self::create_with_mode(
            num_points,
            options,
            dir,
            requested_shards,
            candidate_mode,
            Self::default_candidate_digest_budget_bytes(),
        )
    }

    fn create_with_mode(
        num_points: usize,
        options: ViewLunePruneOptions,
        dir: &Path,
        requested_shards: usize,
        candidate_mode: ViewLuneCandidateSpillMode,
        candidate_digest_budget_bytes: usize,
    ) -> AnnResult<Self> {
        validate_view_lune_options(&options)?;
        fs::create_dir_all(dir)?;
        let shard_count = requested_shards.max(1).min(num_points.max(1));
        let shard_rows = num_points.div_ceil(shard_count);
        let candidate_width = candidate_width_for_options(&options);
        let buffer_records = Self::default_buffer_records();
        let candidate_digest_budget_bytes = candidate_digest_budget_bytes.max(1);
        let mut shards = Vec::with_capacity(shard_count);
        for shard_idx in 0..shard_count {
            let shard_start = shard_idx.saturating_mul(shard_rows);
            let shard_end = (shard_start + shard_rows).min(num_points);
            let shard_len = shard_end.saturating_sub(shard_start);
            let edge_path = dir.join(format!("view_lune_edges_{shard_idx:04}.bin"));
            let witness_path = dir.join(format!("view_lune_witnesses_{shard_idx:04}.bin"));
            let candidate_run_dir = dir.join(format!("view_lune_candidate_runs_{shard_idx:04}"));
            let candidate_digest = if candidate_mode == ViewLuneCandidateSpillMode::DigestRuns {
                fs::create_dir_all(&candidate_run_dir)?;
                Some(CandidateDigestRun::new(
                    shard_len,
                    candidate_width,
                    candidate_digest_budget_bytes,
                ))
            } else {
                None
            };
            let edge_writer = BufWriter::new(File::create(&edge_path)?);
            let witness_writer = BufWriter::new(File::create(&witness_path)?);
            shards.push(Mutex::new(ViewLuneSpillShard {
                edge_path,
                witness_path,
                candidate_run_dir,
                edge_writer,
                witness_writer,
                edge_records: 0,
                witness_records: 0,
                edge_buffer: Vec::new(),
                witness_buffer: Vec::new(),
                candidate_digest,
                candidate_run_paths: Vec::new(),
                candidate_run_bytes: 0,
                candidate_run_rows: 0,
                candidate_run_records: 0,
                candidate_run_flushes: 0,
            }));
        }
        Ok(Self {
            shards,
            num_points,
            shard_rows,
            buffer_records,
            candidate_mode,
            candidate_digest_budget_bytes,
            options,
            raw_candidate_edges: AtomicUsize::new(0),
            raw_witness_records: AtomicUsize::new(0),
            witness_families: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn default_shard_count(num_points: usize) -> usize {
        match std::env::var("FORGEANN_VIEW_LUNE_SPILL_SHARDS") {
            Ok(value) => value
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_VIEW_LUNE_SPILL_SHARDS),
            Err(_) => DEFAULT_VIEW_LUNE_SPILL_SHARDS.min(num_points.max(1)),
        }
    }

    pub(crate) fn default_buffer_records() -> usize {
        match std::env::var("FORGEANN_VIEW_LUNE_SPILL_BUFFER_RECORDS") {
            Ok(value) => value
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_VIEW_LUNE_SPILL_BUFFER_RECORDS),
            Err(_) => DEFAULT_VIEW_LUNE_SPILL_BUFFER_RECORDS,
        }
    }

    pub(crate) fn default_candidate_digest_enabled() -> bool {
        match std::env::var("FORGEANN_VIEW_LUNE_CANDIDATE_DIGEST_ENABLE") {
            Ok(value) => env_flag_enabled(&value),
            Err(_) => true,
        }
    }

    pub(crate) fn default_candidate_digest_budget_bytes() -> usize {
        match std::env::var("FORGEANN_VIEW_LUNE_CANDIDATE_DIGEST_BUDGET_MB") {
            Ok(value) => value
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_VIEW_LUNE_CANDIDATE_DIGEST_BUDGET_MB)
                .saturating_mul(1024 * 1024),
            Err(_) => DEFAULT_VIEW_LUNE_CANDIDATE_DIGEST_BUDGET_MB * 1024 * 1024,
        }
    }

    pub(crate) fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub(crate) fn shard_rows(&self) -> usize {
        self.shard_rows
    }

    pub(crate) fn buffer_records(&self) -> usize {
        self.buffer_records
    }

    pub(crate) fn candidate_digest_enabled(&self) -> bool {
        self.candidate_mode == ViewLuneCandidateSpillMode::DigestRuns
    }

    pub(crate) fn candidate_digest_budget_bytes(&self) -> usize {
        self.candidate_digest_budget_bytes
    }

    pub(crate) fn finish_writes(&self) -> AnnResult<()> {
        for shard in &self.shards {
            let mut shard = shard.lock();
            match self.candidate_mode {
                ViewLuneCandidateSpillMode::RawEdges => {
                    flush_buffered_spilled_edges(&mut shard)?;
                }
                ViewLuneCandidateSpillMode::DigestRuns => {
                    flush_candidate_digest_run(&mut shard)?;
                    if let Some(digest) = shard.candidate_digest.as_mut() {
                        digest.release_storage();
                    }
                }
            }
            flush_buffered_spilled_witnesses(&mut shard)?;
            shard.edge_writer.flush()?;
            shard.witness_writer.flush()?;
        }
        Ok(())
    }

    pub(crate) fn cleanup(&self) -> AnnResult<()> {
        for shard in &self.shards {
            let shard = shard.lock();
            remove_file_if_exists(&shard.edge_path)?;
            remove_file_if_exists(&shard.witness_path)?;
            for path in &shard.candidate_run_paths {
                remove_file_if_exists(path)?;
            }
            match fs::remove_dir(&shard.candidate_run_dir) {
                Ok(()) => {}
                Err(err)
                    if err.kind() == ErrorKind::NotFound
                        || err.kind() == ErrorKind::DirectoryNotEmpty => {}
                Err(err) => return Err(err.into()),
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn reduce_all_rows_with<F>(
        &self,
        base_rows: Option<&[Vec<u32>]>,
        mut on_row: F,
    ) -> AnnResult<(ViewLunePruneStats, usize, usize)>
    where
        F: FnMut(usize, &[u32]) -> AnnResult<()>,
    {
        self.finish_writes()?;
        let reduce_start = Instant::now();
        let mut combined = ViewLunePruneStats {
            raw_candidate_edges: self.raw_candidate_edges.load(AtomicOrdering::Relaxed),
            raw_witness_records: self.raw_witness_records.load(AtomicOrdering::Relaxed),
            distinct_witness_families: self.distinct_witness_families(),
            ..ViewLunePruneStats::default()
        };
        let mut overlap_edges = 0usize;
        let mut overlap_rows = 0usize;
        let mut peak_accumulator_bytes = 0usize;

        for shard_idx in 0..self.shards.len() {
            let shard_start = shard_idx.saturating_mul(self.shard_rows);
            if shard_start >= self.num_points {
                break;
            }
            let shard_end = (shard_start + self.shard_rows).min(self.num_points);
            let shard_len = shard_end - shard_start;
            let (
                edge_path,
                witness_path,
                edge_records,
                witness_records,
                candidate_run_paths,
                candidate_run_bytes,
                candidate_run_rows,
                candidate_run_flushes,
            ) = {
                let shard = self.shards[shard_idx].lock();
                (
                    shard.edge_path.clone(),
                    shard.witness_path.clone(),
                    shard.edge_records,
                    shard.witness_records,
                    shard.candidate_run_paths.clone(),
                    shard.candidate_run_bytes,
                    shard.candidate_run_rows,
                    shard.candidate_run_flushes,
                )
            };
            let accumulator = RuntimeViewLuneAccumulator::create(shard_len, self.options)?;
            let load_start = Instant::now();
            match self.candidate_mode {
                ViewLuneCandidateSpillMode::RawEdges => {
                    load_spilled_edges(&edge_path, &accumulator)?;
                }
                ViewLuneCandidateSpillMode::DigestRuns => {
                    load_spilled_candidate_runs(&candidate_run_paths, &accumulator)?;
                }
            }
            load_spilled_witnesses(&witness_path, &accumulator)?;
            let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
            let compact_start = Instant::now();
            let compaction = accumulator.compact_storage();
            let compact_ms = compact_start.elapsed().as_secs_f64() * 1000.0;
            let base_slice = base_rows.and_then(|rows| rows.get(shard_start..shard_end));
            let (mut shard_stats, shard_overlap_edges, shard_overlap_rows) = accumulator
                .reduce_all_rows_with(base_slice, |local_source, row| {
                    on_row(shard_start + local_source, row)
                })?;
            peak_accumulator_bytes =
                peak_accumulator_bytes.max(shard_stats.estimated_accumulator_bytes);
            tracing::info!(
                "ViewLunePrune spill shard reduced: shard={} range=[{}, {}) edge_records={} witness_records={} candidate_runs={} candidate_run_bytes={} candidate_run_rows={} load_ms={:.3} compact_ms={:.3} candidate_len={} witness_row_len={} accumulator_bytes={} released_bytes={}",
                shard_idx,
                shard_start,
                shard_end,
                edge_records,
                witness_records,
                candidate_run_flushes,
                candidate_run_bytes,
                candidate_run_rows,
                load_ms,
                compact_ms,
                shard_stats.merged_candidates,
                shard_stats.victims_with_witness,
                shard_stats.estimated_accumulator_bytes,
                compaction.estimated_bytes_released(),
            );
            shard_stats.raw_candidate_edges = 0;
            shard_stats.raw_witness_records = 0;
            shard_stats.distinct_witness_families = 0;
            combined.add_shard(shard_stats);
            overlap_edges += shard_overlap_edges;
            overlap_rows += shard_overlap_rows;
        }
        combined.estimated_accumulator_bytes = peak_accumulator_bytes;
        combined.reduce_wall = reduce_start.elapsed();
        Ok((combined, overlap_edges, overlap_rows))
    }

    pub(crate) fn reduce_all_rows_to_raw_graph_shards_parallel(
        &self,
        output_dir: &Path,
        requested_threads: usize,
    ) -> AnnResult<(ViewLunePruneStats, Vec<ViewLuneRawGraphShard>)> {
        self.finish_writes()?;
        fs::create_dir_all(output_dir)?;
        let reduce_start = Instant::now();
        let threads = requested_threads
            .max(1)
            .min(self.shards.len().max(1))
            .min(Self::default_reduce_threads(self.shards.len()));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|err| {
                AnnError::log_index_error(format!("ViewLunePrune reduce pool: {err}"))
            })?;
        let mut results = pool.install(|| {
            (0..self.shards.len())
                .into_par_iter()
                .map(|shard_idx| self.reduce_shard_to_raw_graph(output_dir, shard_idx))
                .collect::<AnnResult<Vec<_>>>()
        })?;
        results.sort_by_key(|result| result.graph_shard.path.clone());

        let mut combined = ViewLunePruneStats {
            raw_candidate_edges: self.raw_candidate_edges.load(AtomicOrdering::Relaxed),
            raw_witness_records: self.raw_witness_records.load(AtomicOrdering::Relaxed),
            distinct_witness_families: self.distinct_witness_families(),
            ..ViewLunePruneStats::default()
        };
        let mut peak_accumulator_bytes = 0usize;
        let mut graph_shards = Vec::with_capacity(results.len());
        for mut result in results {
            peak_accumulator_bytes =
                peak_accumulator_bytes.max(result.stats.estimated_accumulator_bytes);
            result.stats.raw_candidate_edges = 0;
            result.stats.raw_witness_records = 0;
            result.stats.distinct_witness_families = 0;
            combined.add_shard(result.stats);
            graph_shards.push(result.graph_shard);
        }
        combined.estimated_accumulator_bytes = peak_accumulator_bytes;
        combined.reduce_wall = reduce_start.elapsed();
        Ok((combined, graph_shards))
    }

    pub(crate) fn default_reduce_threads(shard_count: usize) -> usize {
        match std::env::var("FORGEANN_VIEW_LUNE_SPILL_REDUCE_THREADS") {
            Ok(value) => value
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .unwrap_or(8)
                .min(shard_count.max(1)),
            Err(_) => 8.min(shard_count.max(1)),
        }
    }

    fn reduce_shard_to_raw_graph(
        &self,
        output_dir: &Path,
        shard_idx: usize,
    ) -> AnnResult<ViewLuneShardReduceResult> {
        let shard_start = shard_idx.saturating_mul(self.shard_rows);
        let shard_end = (shard_start + self.shard_rows).min(self.num_points);
        let shard_len = shard_end.saturating_sub(shard_start);
        let (
            edge_path,
            witness_path,
            edge_records,
            witness_records,
            candidate_run_paths,
            candidate_run_bytes,
            candidate_run_rows,
            candidate_run_flushes,
        ) = {
            let shard = self.shards[shard_idx].lock();
            (
                shard.edge_path.clone(),
                shard.witness_path.clone(),
                shard.edge_records,
                shard.witness_records,
                shard.candidate_run_paths.clone(),
                shard.candidate_run_bytes,
                shard.candidate_run_rows,
                shard.candidate_run_flushes,
            )
        };
        let accumulator = RuntimeViewLuneAccumulator::create(shard_len, self.options)?;
        let load_start = Instant::now();
        match self.candidate_mode {
            ViewLuneCandidateSpillMode::RawEdges => {
                load_spilled_edges(&edge_path, &accumulator)?;
            }
            ViewLuneCandidateSpillMode::DigestRuns => {
                load_spilled_candidate_runs(&candidate_run_paths, &accumulator)?;
            }
        }
        load_spilled_witnesses(&witness_path, &accumulator)?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        let compact_start = Instant::now();
        let compaction = accumulator.compact_storage();
        let compact_ms = compact_start.elapsed().as_secs_f64() * 1000.0;
        let graph_path = output_dir.join(format!("view_lune_graph_rows_{shard_idx:04}.bin"));
        let graph_file = File::create(&graph_path)?;
        let mut graph_writer = BufWriter::new(graph_file);
        let mut graph_bytes = 0u64;
        let mut max_degree = 0u32;
        let (stats, _overlap_edges, _overlap_rows) =
            accumulator.reduce_all_rows_with(None, |_local_source, row| {
                write_raw_graph_row(&mut graph_writer, row, &mut graph_bytes, &mut max_degree)
            })?;
        graph_writer.flush()?;
        tracing::info!(
            "ViewLunePrune spill shard reduced: shard={} range=[{}, {}) edge_records={} witness_records={} candidate_runs={} candidate_run_bytes={} candidate_run_rows={} load_ms={:.3} compact_ms={:.3} candidate_len={} witness_row_len={} accumulator_bytes={} released_bytes={}",
            shard_idx,
            shard_start,
            shard_end,
            edge_records,
            witness_records,
            candidate_run_flushes,
            candidate_run_bytes,
            candidate_run_rows,
            load_ms,
            compact_ms,
            stats.merged_candidates,
            stats.victims_with_witness,
            stats.estimated_accumulator_bytes,
            compaction.estimated_bytes_released(),
        );
        Ok(ViewLuneShardReduceResult {
            stats,
            graph_shard: ViewLuneRawGraphShard {
                path: graph_path,
                bytes: graph_bytes,
                max_degree,
            },
        })
    }

    fn shard_index(&self, source: usize) -> Option<usize> {
        (source < self.num_points).then(|| (source / self.shard_rows).min(self.shards.len() - 1))
    }

    fn local_source(&self, source: usize, shard_idx: usize) -> u32 {
        (source - shard_idx.saturating_mul(self.shard_rows)) as u32
    }

    fn distinct_witness_families(&self) -> usize {
        let mut families = self.witness_families.lock();
        families.sort_unstable();
        families.dedup();
        families.len()
    }
}

pub(crate) struct ViewLuneTaggedEdgeSink<'a> {
    recorder: &'a dyn ViewLuneEdgeRecorder,
    view_family: u32,
    inner: &'a dyn PendingEdgeSink,
}

impl<'a> ViewLuneTaggedEdgeSink<'a> {
    pub(crate) fn new(
        recorder: &'a dyn ViewLuneEdgeRecorder,
        view_family: u32,
        inner: &'a dyn PendingEdgeSink,
    ) -> Self {
        Self {
            recorder,
            view_family,
            inner,
        }
    }
}

impl PendingEdgeSink for ViewLuneTaggedEdgeSink<'_> {
    fn flush_pending_edges(&self, edges: &mut Vec<PendingEdge>) -> AnnResult<()> {
        self.recorder
            .write_edges_in_place(self.view_family, edges)?;
        self.inner.flush_pending_edges(edges)
    }

    fn wants_lune_witnesses(&self) -> bool {
        true
    }

    fn flush_lune_witnesses(&self, witnesses: &mut Vec<PendingLuneWitness>) -> AnnResult<()> {
        self.recorder
            .write_witnesses_in_place(self.view_family, witnesses)?;
        witnesses.clear();
        Ok(())
    }
}

#[derive(Debug)]
pub struct RuntimeViewLuneAccumulator {
    rows: Vec<Mutex<RuntimeViewLuneRow>>,
    options: ViewLunePruneOptions,
    candidate_width: usize,
    raw_candidate_edges: AtomicUsize,
    raw_witness_records: AtomicUsize,
    filtered_witness_records: AtomicUsize,
    stale_witness_rows_pruned: AtomicUsize,
    rows_over_candidate_width: AtomicUsize,
    witness_families: Mutex<Vec<u32>>,
}

#[derive(Debug)]
struct RuntimeViewLuneRow {
    candidates: Vec<ViewLuneCandidate>,
    candidate_worst_idx: u16,
    witness_rows: Vec<ViewLuneVictimWitnesses>,
    witness_overflow: Vec<RuntimeLuneWitness>,
}

impl Default for RuntimeViewLuneRow {
    fn default() -> Self {
        Self {
            candidates: Vec::new(),
            candidate_worst_idx: INVALID_CANDIDATE_IDX,
            witness_rows: Vec::new(),
            witness_overflow: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ViewLuneCandidate {
    dst: u32,
    dist2: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ViewLuneVictimWitnesses {
    victim: u32,
    len: u16,
    overflow_capacity: u16,
    overflow_offset: u32,
    inline: [RuntimeLuneWitness; INLINE_WITNESS_CAP],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RuntimeLuneWitness {
    pivot: u32,
    rank_key: u32,
}

const INLINE_WITNESS_CAP: usize = 2;
const INVALID_CANDIDATE_IDX: u16 = u16::MAX;
const INVALID_WITNESS_OVERFLOW_OFFSET: u32 = u32::MAX;
const CANDIDATE_ROW_CAPACITY_GROWTH: usize = 8;
const WITNESS_ROW_CAPACITY_GROWTH: usize = 4;

#[derive(Clone, Copy, Debug)]
struct CandidateUpdate {
    dst: u32,
    dist2: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewLuneRowOutput {
    pub neighbors: Vec<u32>,
    pub pruned_by_selected_witness: usize,
    pub fill_after_prune: usize,
    pub modified_vs_base: bool,
    pub overlap_with_base: usize,
}

pub fn candidate_width_for_options(options: &ViewLunePruneOptions) -> usize {
    let max_degree = options.max_degree.max(1);
    let width = (max_degree as f32 * options.candidate_width_multiplier).ceil() as usize;
    width.clamp(max_degree, max_degree.saturating_mul(4))
}

pub fn validate_view_lune_options(options: &ViewLunePruneOptions) -> AnnResult<()> {
    if options.metric != Metric::L2 {
        return Err(AnnError::log_index_config_error(
            "view_lune_prune".to_string(),
            "ViewLunePrune is L2-only in this implementation".to_string(),
        ));
    }
    if options.max_degree == 0 {
        return Err(AnnError::log_index_config_error(
            "view_lune_max_degree".to_string(),
            "max_degree must be > 0".to_string(),
        ));
    }
    if !options.candidate_width_multiplier.is_finite() || options.candidate_width_multiplier <= 0.0
    {
        return Err(AnnError::log_index_config_error(
            "view_lune_candidate_width_multiplier".to_string(),
            "candidate width multiplier must be finite and > 0".to_string(),
        ));
    }
    if options.max_witnesses_per_victim == 0 {
        return Err(AnnError::log_index_config_error(
            "view_lune_max_witnesses_per_victim".to_string(),
            "max witnesses per victim must be > 0".to_string(),
        ));
    }
    if options.max_witnesses_per_victim > usize::from(u16::MAX) {
        return Err(AnnError::log_index_config_error(
            "view_lune_max_witnesses_per_victim".to_string(),
            "max witnesses per victim must fit in u16".to_string(),
        ));
    }
    if options.nearest_core > options.max_degree {
        return Err(AnnError::log_index_config_error(
            "view_lune_nearest_core".to_string(),
            "nearest core must not exceed max_degree".to_string(),
        ));
    }
    Ok(())
}

#[inline]
#[allow(dead_code)]
pub fn is_lune_witness(source_victim_dist2: f32, pivot_victim_dist2: f32) -> bool {
    pivot_victim_dist2 < source_victim_dist2 / FINAL_PRUNE_ALPHA_IMPL
}

impl RuntimeViewLuneAccumulator {
    pub fn create(num_points: usize, options: ViewLunePruneOptions) -> AnnResult<Self> {
        validate_view_lune_options(&options)?;
        let candidate_width = candidate_width_for_options(&options);
        let rows = (0..num_points)
            .map(|_| Mutex::new(RuntimeViewLuneRow::default()))
            .collect();
        Ok(Self {
            rows,
            options,
            candidate_width,
            raw_candidate_edges: AtomicUsize::new(0),
            raw_witness_records: AtomicUsize::new(0),
            filtered_witness_records: AtomicUsize::new(0),
            stale_witness_rows_pruned: AtomicUsize::new(0),
            rows_over_candidate_width: AtomicUsize::new(0),
            witness_families: Mutex::new(Vec::new()),
        })
    }

    pub fn candidate_width(&self) -> usize {
        self.candidate_width
    }

    pub fn compact_storage(&self) -> ViewLuneAccumulatorCompactionStats {
        let mut stats = ViewLuneAccumulatorCompactionStats::default();
        for row in &self.rows {
            let mut row = row.lock();
            row.compact_storage(&mut stats);
        }
        stats
    }

    #[allow(dead_code)]
    pub fn reduce_all_rows(
        &self,
        base_rows: Option<&[Vec<u32>]>,
    ) -> (Vec<Vec<u32>>, ViewLunePruneStats, usize, usize) {
        let mut outputs = Vec::with_capacity(self.rows.len());
        let (stats, overlap_edges, overlap_rows) = self
            .reduce_all_rows_with(base_rows, |_, row| {
                outputs.push(row.to_vec());
                Ok(())
            })
            .expect("collecting ViewLune rows should not fail");
        (outputs, stats, overlap_edges, overlap_rows)
    }

    pub fn reduce_all_rows_with<F>(
        &self,
        base_rows: Option<&[Vec<u32>]>,
        mut on_row: F,
    ) -> AnnResult<(ViewLunePruneStats, usize, usize)>
    where
        F: FnMut(usize, &[u32]) -> AnnResult<()>,
    {
        let reduce_start = Instant::now();
        let mut stats = self.snapshot_stats();
        let mut overlap_edges = 0usize;
        let mut overlap_rows = 0usize;
        for source in 0..self.rows.len() {
            let base_row = base_rows.and_then(|rows| rows.get(source).map(Vec::as_slice));
            let row_output = self.reduce_row(source, base_row);
            stats.pruned_by_selected_witness += row_output.pruned_by_selected_witness;
            stats.fill_after_prune += row_output.fill_after_prune;
            stats.final_edges += row_output.neighbors.len();
            if row_output.modified_vs_base {
                stats.rows_modified += 1;
            }
            overlap_edges += row_output.overlap_with_base;
            if let Some(base) = base_row {
                let base_prefix = &base[..base.len().min(self.options.max_degree)];
                if base_prefix == row_output.neighbors.as_slice() {
                    overlap_rows += 1;
                }
            }
            on_row(source, &row_output.neighbors)?;
        }
        stats.reduce_wall = reduce_start.elapsed();
        Ok((stats, overlap_edges, overlap_rows))
    }

    #[allow(dead_code)]
    pub fn write_oracle_json(
        &self,
        path: &Path,
        base_rows: Option<&[Vec<u32>]>,
        executor_outputs: Option<&[Vec<u32>]>,
    ) -> AnnResult<ViewLuneOracleReport> {
        let (stats, overlap_edges, overlap_rows, rows_compared) = if let Some(outputs) =
            executor_outputs
        {
            let mut stats = self.snapshot_stats();
            stats.final_edges = outputs.iter().map(Vec::len).sum();
            let mut overlap_edges = 0usize;
            let mut overlap_rows = 0usize;
            if let Some(base_rows) = base_rows {
                for (out, base) in outputs.iter().zip(base_rows) {
                    overlap_edges +=
                        count_overlap(out, &base[..base.len().min(self.options.max_degree)]);
                    if out.as_slice() == &base[..base.len().min(self.options.max_degree)] {
                        overlap_rows += 1;
                    }
                }
            }
            (
                stats,
                overlap_edges,
                overlap_rows,
                base_rows.map_or(0, |rows| rows.len()),
            )
        } else {
            let (_outputs, stats, overlap_edges, overlap_rows) = self.reduce_all_rows(base_rows);
            (
                stats,
                overlap_edges,
                overlap_rows,
                base_rows.map_or(0, |rows| rows.len()),
            )
        };
        let report = ViewLuneOracleReport {
            schema: "view_lune_prune_oracle_v1",
            options: self.options,
            stats,
            overlap_with_base_edges: overlap_edges,
            overlap_with_base_rows: overlap_rows,
            rows_compared_with_base: rows_compared,
        };
        fs::write(path, report.to_json())?;
        Ok(report)
    }

    pub fn write_oracle_json_from_stats(
        &self,
        path: &Path,
        stats: ViewLunePruneStats,
        overlap_edges: usize,
        overlap_rows: usize,
        rows_compared: usize,
    ) -> AnnResult<ViewLuneOracleReport> {
        let report = ViewLuneOracleReport {
            schema: "view_lune_prune_oracle_v1",
            options: self.options,
            stats,
            overlap_with_base_edges: overlap_edges,
            overlap_with_base_rows: overlap_rows,
            rows_compared_with_base: rows_compared,
        };
        fs::write(path, report.to_json())?;
        Ok(report)
    }

    pub fn snapshot_stats(&self) -> ViewLunePruneStats {
        let mut merged_candidates = 0usize;
        let mut victims_with_witness = 0usize;
        let mut heap_witness_capacity = 0usize;
        let mut rows_with_candidates = 0usize;
        for row in &self.rows {
            let row = row.lock();
            merged_candidates += row.candidates.len();
            rows_with_candidates += usize::from(!row.candidates.is_empty());
            victims_with_witness += row.witness_rows.len();
            heap_witness_capacity += row.witness_overflow.capacity();
        }
        let distinct_witness_families = {
            let mut families = self.witness_families.lock();
            families.sort_unstable();
            families.dedup();
            families.len()
        };
        ViewLunePruneStats {
            raw_candidate_edges: self.raw_candidate_edges.load(AtomicOrdering::Relaxed),
            raw_witness_records: self.raw_witness_records.load(AtomicOrdering::Relaxed),
            filtered_witness_records: self.filtered_witness_records.load(AtomicOrdering::Relaxed),
            stale_witness_rows_pruned: self.stale_witness_rows_pruned.load(AtomicOrdering::Relaxed),
            merged_candidates,
            victims_with_witness,
            distinct_witness_families,
            rows_with_candidates,
            rows_over_candidate_width: self.rows_over_candidate_width.load(AtomicOrdering::Relaxed),
            estimated_accumulator_bytes: self.estimated_bytes(
                merged_candidates,
                victims_with_witness,
                heap_witness_capacity,
            ),
            ..ViewLunePruneStats::default()
        }
    }

    fn estimated_bytes(
        &self,
        candidates: usize,
        victim_rows: usize,
        heap_witness_capacity: usize,
    ) -> usize {
        self.rows
            .len()
            .saturating_mul(std::mem::size_of::<Mutex<RuntimeViewLuneRow>>())
            .saturating_add(candidates.saturating_mul(std::mem::size_of::<ViewLuneCandidate>()))
            .saturating_add(
                victim_rows.saturating_mul(std::mem::size_of::<ViewLuneVictimWitnesses>()),
            )
            .saturating_add(
                heap_witness_capacity.saturating_mul(std::mem::size_of::<RuntimeLuneWitness>()),
            )
    }

    fn reduce_row(&self, source: usize, base_row: Option<&[u32]>) -> ViewLuneRowOutput {
        let mut row = self.rows[source].lock();
        row.candidates.sort_by(candidate_distance_cmp);
        row.candidate_worst_idx = INVALID_CANDIDATE_IDX;
        let candidates = row.candidates.as_slice();

        let baseline = base_row.map(|base| &base[..base.len().min(self.options.max_degree)]);

        if candidates.len() <= self.options.max_degree {
            let neighbors: Vec<u32> = candidates.iter().map(|candidate| candidate.dst).collect();
            let overlap = baseline.map_or(0, |baseline| count_overlap(&neighbors, baseline));
            return ViewLuneRowOutput {
                modified_vs_base: baseline.is_some_and(|baseline| neighbors != baseline),
                neighbors,
                pruned_by_selected_witness: 0,
                fill_after_prune: 0,
                overlap_with_base: overlap,
            };
        }

        let mut selected = Vec::with_capacity(self.options.max_degree);
        let mut skipped = 0usize;
        for candidate in candidates {
            if selected.contains(&candidate.dst) {
                continue;
            }
            if selected.len() < self.options.nearest_core {
                selected.push(candidate.dst);
            } else if row.has_selected_witness(candidate.dst, &selected) {
                skipped += 1;
            } else {
                selected.push(candidate.dst);
            }
            if selected.len() == self.options.max_degree {
                break;
            }
        }

        let mut fill_after_prune = 0usize;
        if selected.len() < self.options.max_degree {
            for candidate in candidates {
                if selected.len() == self.options.max_degree {
                    break;
                }
                if selected.contains(&candidate.dst) {
                    continue;
                }
                selected.push(candidate.dst);
                fill_after_prune += 1;
            }
        }

        selected.truncate(self.options.max_degree);
        let overlap = baseline.map_or(0, |baseline| count_overlap(&selected, baseline));
        ViewLuneRowOutput {
            modified_vs_base: baseline.is_some_and(|baseline| selected != baseline),
            neighbors: selected,
            pruned_by_selected_witness: skipped,
            fill_after_prune,
            overlap_with_base: overlap,
        }
    }
}

impl ViewLuneEdgeRecorder for RuntimeViewLuneAccumulator {
    fn write_edges(&self, _view_family: u32, edges: &[PendingEdge]) -> AnnResult<()> {
        let mut edges = edges.to_vec();
        self.write_edges_in_place(_view_family, &mut edges)
    }

    fn write_edges_in_place(
        &self,
        _view_family: u32,
        edges: &mut Vec<PendingEdge>,
    ) -> AnnResult<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let raw_edges = edges.len();
        edges.sort_unstable_by_key(|edge| (edge.p, edge.c));
        let mut idx = 0usize;
        while idx < edges.len() {
            let source = edges[idx].p;
            if source >= self.rows.len() {
                idx += 1;
                continue;
            }
            let mut row = self.rows[source].lock();
            while idx < edges.len() && edges[idx].p == source {
                let dst = edges[idx].c;
                let mut dist2 = f32::INFINITY;
                let mut valid = false;
                while idx < edges.len() && edges[idx].p == source && edges[idx].c == dst {
                    let edge = &edges[idx];
                    let is_direct = edge.flags & PENDING_EDGE_DIRECT != 0;
                    let is_mirror = edge.flags & PENDING_EDGE_MIRROR != 0;
                    if is_direct || is_mirror {
                        dist2 = dist2.min(edge.dist);
                        valid = true;
                    }
                    idx += 1;
                }
                if !valid {
                    continue;
                }
                let update = CandidateUpdate { dst, dist2 };
                let (overflowed, stale_rows_pruned) =
                    row.insert_candidate(update, self.candidate_width);
                if overflowed {
                    self.rows_over_candidate_width
                        .fetch_add(1, AtomicOrdering::Relaxed);
                }
                if stale_rows_pruned > 0 {
                    self.stale_witness_rows_pruned
                        .fetch_add(stale_rows_pruned, AtomicOrdering::Relaxed);
                }
            }
        }
        self.raw_candidate_edges
            .fetch_add(raw_edges, AtomicOrdering::Relaxed);
        Ok(())
    }

    fn write_witnesses(&self, view_family: u32, witnesses: &[PendingLuneWitness]) -> AnnResult<()> {
        let mut witnesses = witnesses.to_vec();
        self.write_witnesses_in_place(view_family, &mut witnesses)
    }

    fn write_witnesses_in_place(
        &self,
        view_family: u32,
        witnesses: &mut Vec<PendingLuneWitness>,
    ) -> AnnResult<()> {
        if witnesses.is_empty() {
            return Ok(());
        }
        self.witness_families.lock().push(view_family);
        let raw_witnesses = witnesses.len();
        let mut filtered_witnesses = 0usize;
        witnesses.sort_unstable_by_key(|witness| (witness.src, witness.victim, witness.pivot));
        let mut idx = 0usize;
        while idx < witnesses.len() {
            let source = witnesses[idx].src as usize;
            if source >= self.rows.len() {
                idx += 1;
                continue;
            }
            let mut row = self.rows[source].lock();
            while idx < witnesses.len() && witnesses[idx].src as usize == source {
                let victim = witnesses[idx].victim;
                let pivot = witnesses[idx].pivot;
                let mut witness = RuntimeLuneWitness {
                    pivot,
                    rank_key: witness_rank_key(
                        witnesses[idx].pivot_rank.max(1),
                        witnesses[idx].margin_q16,
                        witnesses[idx].victim_rank.max(1),
                    ),
                };
                idx += 1;
                while idx < witnesses.len()
                    && witnesses[idx].src as usize == source
                    && witnesses[idx].victim == victim
                    && witnesses[idx].pivot == pivot
                {
                    let next = RuntimeLuneWitness {
                        pivot,
                        rank_key: witness_rank_key(
                            witnesses[idx].pivot_rank.max(1),
                            witnesses[idx].margin_q16,
                            witnesses[idx].victim_rank.max(1),
                        ),
                    };
                    if witness_retention_cmp(&next, &witness).is_lt() {
                        witness = next;
                    }
                    idx += 1;
                }
                if row.has_candidate(victim) && row.has_candidate(pivot) {
                    row.insert_witness(victim, witness, self.options.max_witnesses_per_victim);
                } else {
                    filtered_witnesses += 1;
                }
            }
        }
        self.raw_witness_records
            .fetch_add(raw_witnesses, AtomicOrdering::Relaxed);
        if filtered_witnesses > 0 {
            self.filtered_witness_records
                .fetch_add(filtered_witnesses, AtomicOrdering::Relaxed);
        }
        Ok(())
    }
}

impl ViewLuneEdgeRecorder for SpillingViewLuneRecorder {
    fn write_edges(&self, _view_family: u32, edges: &[PendingEdge]) -> AnnResult<()> {
        let mut edges = edges.to_vec();
        self.write_edges_in_place(_view_family, &mut edges)
    }

    fn write_edges_in_place(
        &self,
        _view_family: u32,
        edges: &mut Vec<PendingEdge>,
    ) -> AnnResult<()> {
        if edges.is_empty() {
            return Ok(());
        }
        self.raw_candidate_edges
            .fetch_add(edges.len(), AtomicOrdering::Relaxed);
        edges.sort_unstable_by_key(|edge| (edge.p, edge.c));
        let mut idx = 0usize;
        while idx < edges.len() {
            let source = edges[idx].p;
            let Some(shard_idx) = self.shard_index(source) else {
                while idx < edges.len() && edges[idx].p == source {
                    idx += 1;
                }
                continue;
            };
            let mut shard = self.shards[shard_idx].lock();
            while idx < edges.len() && self.shard_index(edges[idx].p) == Some(shard_idx) {
                let source = edges[idx].p;
                let dst = edges[idx].c;
                let mut dist = f32::INFINITY;
                let mut valid = false;
                while idx < edges.len() && edges[idx].p == source && edges[idx].c == dst {
                    let edge = &edges[idx];
                    if edge.flags & (PENDING_EDGE_DIRECT | PENDING_EDGE_MIRROR) != 0 {
                        dist = dist.min(edge.dist);
                        valid = true;
                    }
                    idx += 1;
                }
                if valid {
                    let edge = BufferedSpilledEdge {
                        local_source: self.local_source(source, shard_idx),
                        dst,
                        dist,
                    };
                    match self.candidate_mode {
                        ViewLuneCandidateSpillMode::RawEdges => {
                            buffer_spilled_edge(&mut shard, self.buffer_records, edge)?;
                        }
                        ViewLuneCandidateSpillMode::DigestRuns => {
                            buffer_candidate_digest_edge(&mut shard, edge)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn write_witnesses(&self, view_family: u32, witnesses: &[PendingLuneWitness]) -> AnnResult<()> {
        let mut witnesses = witnesses.to_vec();
        self.write_witnesses_in_place(view_family, &mut witnesses)
    }

    fn write_witnesses_in_place(
        &self,
        view_family: u32,
        witnesses: &mut Vec<PendingLuneWitness>,
    ) -> AnnResult<()> {
        if witnesses.is_empty() {
            return Ok(());
        }
        self.raw_witness_records
            .fetch_add(witnesses.len(), AtomicOrdering::Relaxed);
        self.witness_families.lock().push(view_family);
        witnesses.sort_unstable_by_key(|witness| (witness.src, witness.victim, witness.pivot));
        let mut idx = 0usize;
        while idx < witnesses.len() {
            let source = witnesses[idx].src as usize;
            let Some(shard_idx) = self.shard_index(source) else {
                while idx < witnesses.len() && witnesses[idx].src as usize == source {
                    idx += 1;
                }
                continue;
            };
            let mut shard = self.shards[shard_idx].lock();
            while idx < witnesses.len()
                && self.shard_index(witnesses[idx].src as usize) == Some(shard_idx)
            {
                let source = witnesses[idx].src;
                let victim = witnesses[idx].victim;
                let pivot = witnesses[idx].pivot;
                let mut witness = witnesses[idx];
                idx += 1;
                while idx < witnesses.len()
                    && witnesses[idx].src == source
                    && witnesses[idx].victim == victim
                    && witnesses[idx].pivot == pivot
                {
                    if pending_witness_retention_key(&witnesses[idx])
                        < pending_witness_retention_key(&witness)
                    {
                        witness = witnesses[idx];
                    }
                    idx += 1;
                }
                buffer_spilled_witness(
                    &mut shard,
                    self.buffer_records,
                    PendingLuneWitness {
                        src: self.local_source(source as usize, shard_idx),
                        ..witness
                    },
                )?;
            }
        }
        Ok(())
    }
}

fn remove_file_if_exists(path: &Path) -> AnnResult<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

fn env_flag_enabled(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn buffer_candidate_digest_edge(
    shard: &mut ViewLuneSpillShard,
    edge: BufferedSpilledEdge,
) -> AnnResult<()> {
    let should_flush = {
        let Some(digest) = shard.candidate_digest.as_mut() else {
            return Err(AnnError::log_index_error(
                "ViewLunePrune candidate digest mode missing shard digest".to_string(),
            ));
        };
        digest.insert(edge.local_source, edge.dst, edge.dist);
        digest.should_flush()
    };
    if should_flush {
        flush_candidate_digest_run(shard)?;
    }
    Ok(())
}

fn flush_candidate_digest_run(shard: &mut ViewLuneSpillShard) -> AnnResult<()> {
    let Some(digest) = shard.candidate_digest.as_mut() else {
        return Ok(());
    };
    if digest.candidate_slots == 0 {
        return Ok(());
    }

    let run_path = shard
        .candidate_run_dir
        .join(format!("run_{:06}.bin", shard.candidate_run_flushes));
    let mut writer = BufWriter::new(File::create(&run_path)?);
    let mut run_bytes = 0u64;
    let mut run_records = 0usize;
    let mut run_rows = 0usize;

    digest.touched_rows.sort_unstable();
    digest.touched_rows.dedup();
    for &local_source in &digest.touched_rows {
        let local_source_idx = local_source as usize;
        let offset = digest.rows[local_source_idx].offset as usize;
        let len = digest.rows[local_source_idx].len as usize;
        if len == 0 {
            digest.rows[local_source_idx] = CandidateDigestRow::default();
            continue;
        }
        digest.candidates[offset..offset + len].sort_by(candidate_distance_cmp);
        let count = len as u32;
        writer.write_all(&local_source.to_le_bytes())?;
        writer.write_all(&count.to_le_bytes())?;
        run_bytes += VIEW_LUNE_CANDIDATE_RUN_ROW_HEADER_BYTES as u64;
        for candidate in &digest.candidates[offset..offset + len] {
            writer.write_all(&candidate.dst.to_le_bytes())?;
            writer.write_all(&candidate.dist2.to_bits().to_le_bytes())?;
        }
        run_records += len;
        run_rows += 1;
        run_bytes += (len * VIEW_LUNE_CANDIDATE_RUN_RECORD_BYTES) as u64;
        digest.rows[local_source_idx] = CandidateDigestRow::default();
    }
    writer.flush()?;

    digest.candidates.clear();
    digest.touched_rows.clear();
    digest.candidate_slots = 0;
    shard.candidate_run_paths.push(run_path);
    shard.candidate_run_bytes += run_bytes;
    shard.candidate_run_rows += run_rows;
    shard.candidate_run_records += run_records;
    shard.candidate_run_flushes += 1;
    shard.edge_records += run_records;
    Ok(())
}

fn buffer_spilled_edge(
    shard: &mut ViewLuneSpillShard,
    buffer_records: usize,
    edge: BufferedSpilledEdge,
) -> AnnResult<()> {
    shard.edge_buffer.push(edge);
    if shard.edge_buffer.len() >= buffer_records {
        flush_buffered_spilled_edges(shard)?;
    }
    Ok(())
}

fn flush_buffered_spilled_edges(shard: &mut ViewLuneSpillShard) -> AnnResult<()> {
    if shard.edge_buffer.is_empty() {
        return Ok(());
    }
    shard
        .edge_buffer
        .sort_unstable_by_key(|edge| (edge.local_source, edge.dst));
    let mut idx = 0usize;
    let mut written = 0usize;
    while idx < shard.edge_buffer.len() {
        let source = shard.edge_buffer[idx].local_source;
        let dst = shard.edge_buffer[idx].dst;
        let mut dist = f32::INFINITY;
        while idx < shard.edge_buffer.len()
            && shard.edge_buffer[idx].local_source == source
            && shard.edge_buffer[idx].dst == dst
        {
            dist = dist.min(shard.edge_buffer[idx].dist);
            idx += 1;
        }
        write_spilled_edge(&mut shard.edge_writer, source, dst, dist)?;
        written += 1;
    }
    shard.edge_records += written;
    shard.edge_buffer.clear();
    Ok(())
}

fn buffer_spilled_witness(
    shard: &mut ViewLuneSpillShard,
    buffer_records: usize,
    witness: PendingLuneWitness,
) -> AnnResult<()> {
    shard.witness_buffer.push(witness);
    if shard.witness_buffer.len() >= buffer_records {
        flush_buffered_spilled_witnesses(shard)?;
    }
    Ok(())
}

fn flush_buffered_spilled_witnesses(shard: &mut ViewLuneSpillShard) -> AnnResult<()> {
    if shard.witness_buffer.is_empty() {
        return Ok(());
    }
    shard
        .witness_buffer
        .sort_unstable_by_key(|witness| (witness.src, witness.victim, witness.pivot));
    let mut idx = 0usize;
    let mut written = 0usize;
    while idx < shard.witness_buffer.len() {
        let source = shard.witness_buffer[idx].src;
        let victim = shard.witness_buffer[idx].victim;
        let pivot = shard.witness_buffer[idx].pivot;
        let mut witness = shard.witness_buffer[idx];
        idx += 1;
        while idx < shard.witness_buffer.len()
            && shard.witness_buffer[idx].src == source
            && shard.witness_buffer[idx].victim == victim
            && shard.witness_buffer[idx].pivot == pivot
        {
            if pending_witness_retention_key(&shard.witness_buffer[idx])
                < pending_witness_retention_key(&witness)
            {
                witness = shard.witness_buffer[idx];
            }
            idx += 1;
        }
        write_spilled_witness(&mut shard.witness_writer, witness)?;
        written += 1;
    }
    shard.witness_records += written;
    shard.witness_buffer.clear();
    Ok(())
}

fn write_spilled_edge(
    writer: &mut BufWriter<File>,
    local_source: u32,
    dst: u32,
    dist: f32,
) -> AnnResult<()> {
    let mut record = [0u8; VIEW_LUNE_EDGE_SPILL_RECORD_BYTES];
    record[0..4].copy_from_slice(&local_source.to_le_bytes());
    record[4..8].copy_from_slice(&dst.to_le_bytes());
    record[8..12].copy_from_slice(&dist.to_bits().to_le_bytes());
    writer.write_all(&record)?;
    Ok(())
}

fn write_spilled_witness(
    writer: &mut BufWriter<File>,
    witness: PendingLuneWitness,
) -> AnnResult<()> {
    let mut record = [0u8; VIEW_LUNE_WITNESS_SPILL_RECORD_BYTES];
    record[0..4].copy_from_slice(&witness.src.to_le_bytes());
    record[4..8].copy_from_slice(&witness.pivot.to_le_bytes());
    record[8..12].copy_from_slice(&witness.victim.to_le_bytes());
    record[12..14].copy_from_slice(&witness.margin_q16.to_le_bytes());
    record[14] = witness.pivot_rank;
    record[15] = witness.victim_rank;
    writer.write_all(&record)?;
    Ok(())
}

fn load_spilled_edges(path: &Path, accumulator: &RuntimeViewLuneAccumulator) -> AnnResult<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut record = [0u8; VIEW_LUNE_EDGE_SPILL_RECORD_BYTES];
    let mut batch = Vec::with_capacity(65_536);
    loop {
        match reader.read_exact(&mut record) {
            Ok(()) => {
                batch.push(PendingEdge {
                    p: u32::from_le_bytes(record[0..4].try_into().unwrap()) as usize,
                    c: u32::from_le_bytes(record[4..8].try_into().unwrap()),
                    hash: 0,
                    dist: f32::from_bits(u32::from_le_bytes(record[8..12].try_into().unwrap())),
                    mandatory: false,
                    local_rank: 0,
                    flags: PENDING_EDGE_DIRECT,
                });
                if batch.len() == batch.capacity() {
                    accumulator.write_edges_in_place(0, &mut batch)?;
                    batch.clear();
                }
            }
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err.into()),
        }
    }
    if !batch.is_empty() {
        accumulator.write_edges_in_place(0, &mut batch)?;
    }
    Ok(())
}

fn load_spilled_candidate_runs(
    paths: &[PathBuf],
    accumulator: &RuntimeViewLuneAccumulator,
) -> AnnResult<()> {
    let mut row_header = [0u8; VIEW_LUNE_CANDIDATE_RUN_ROW_HEADER_BYTES];
    let mut candidate_record = [0u8; VIEW_LUNE_CANDIDATE_RUN_RECORD_BYTES];
    let mut batch = Vec::with_capacity(65_536);
    for path in paths {
        let mut reader = BufReader::new(File::open(path)?);
        loop {
            match reader.read_exact(&mut row_header) {
                Ok(()) => {
                    let local_source = u32::from_le_bytes(row_header[0..4].try_into().unwrap());
                    let count = u32::from_le_bytes(row_header[4..8].try_into().unwrap());
                    for _ in 0..count {
                        reader.read_exact(&mut candidate_record)?;
                        batch.push(PendingEdge {
                            p: local_source as usize,
                            c: u32::from_le_bytes(candidate_record[0..4].try_into().unwrap()),
                            hash: 0,
                            dist: f32::from_bits(u32::from_le_bytes(
                                candidate_record[4..8].try_into().unwrap(),
                            )),
                            mandatory: false,
                            local_rank: 0,
                            flags: PENDING_EDGE_DIRECT,
                        });
                        if batch.len() == batch.capacity() {
                            accumulator.write_edges_in_place(0, &mut batch)?;
                            batch.clear();
                        }
                    }
                }
                Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(err.into()),
            }
        }
    }
    if !batch.is_empty() {
        accumulator.write_edges_in_place(0, &mut batch)?;
    }
    Ok(())
}

fn load_spilled_witnesses(path: &Path, accumulator: &RuntimeViewLuneAccumulator) -> AnnResult<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut record = [0u8; VIEW_LUNE_WITNESS_SPILL_RECORD_BYTES];
    let mut batch = Vec::with_capacity(65_536);
    loop {
        match reader.read_exact(&mut record) {
            Ok(()) => {
                batch.push(PendingLuneWitness {
                    src: u32::from_le_bytes(record[0..4].try_into().unwrap()),
                    pivot: u32::from_le_bytes(record[4..8].try_into().unwrap()),
                    victim: u32::from_le_bytes(record[8..12].try_into().unwrap()),
                    margin_q16: u16::from_le_bytes(record[12..14].try_into().unwrap()),
                    pivot_rank: record[14],
                    victim_rank: record[15],
                });
                if batch.len() == batch.capacity() {
                    accumulator.write_witnesses_in_place(0, &mut batch)?;
                    batch.clear();
                }
            }
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err.into()),
        }
    }
    if !batch.is_empty() {
        accumulator.write_witnesses_in_place(0, &mut batch)?;
    }
    Ok(())
}

fn write_raw_graph_row(
    writer: &mut BufWriter<File>,
    neighbors: &[u32],
    bytes: &mut u64,
    max_degree: &mut u32,
) -> AnnResult<()> {
    let degree = neighbors.len() as u32;
    writer.write_all(&degree.to_le_bytes())?;
    for neighbor in neighbors {
        writer.write_all(&neighbor.to_le_bytes())?;
    }
    *bytes += ((neighbors.len() + 1) * std::mem::size_of::<u32>()) as u64;
    *max_degree = (*max_degree).max(degree);
    Ok(())
}

impl RuntimeViewLuneRow {
    fn compact_storage(&mut self, stats: &mut ViewLuneAccumulatorCompactionStats) {
        stats.candidate_len += self.candidates.len();
        stats.candidate_capacity_before += self.candidates.capacity();
        if self.candidates.capacity() > self.candidates.len() {
            self.candidates.shrink_to_fit();
        }
        stats.candidate_capacity_after += self.candidates.capacity();

        stats.witness_row_len += self.witness_rows.len();
        stats.witness_row_capacity_before += self.witness_rows.capacity();
        stats.heap_witness_capacity_before += self.witness_overflow.capacity();
        self.compact_witness_overflow();
        stats.heap_witness_capacity_after += self.witness_overflow.capacity();
        if self.witness_rows.capacity() > self.witness_rows.len() {
            self.witness_rows.shrink_to_fit();
        }
        stats.witness_row_capacity_after += self.witness_rows.capacity();
    }

    fn compact_witness_overflow(&mut self) {
        if self.witness_overflow.is_empty() {
            return;
        }
        let old_overflow = std::mem::take(&mut self.witness_overflow);
        let active_overflow_len: usize = self
            .witness_rows
            .iter()
            .filter(|row| row.has_overflow())
            .map(|row| usize::from(row.len))
            .sum();
        let mut compacted = Vec::with_capacity(active_overflow_len);
        for row in &mut self.witness_rows {
            if !row.has_overflow() {
                continue;
            }
            let len = usize::from(row.len);
            let old_offset = row.overflow_offset as usize;
            let new_offset = compacted.len();
            compacted.extend_from_slice(&old_overflow[old_offset..old_offset + len]);
            row.overflow_offset =
                u32::try_from(new_offset).unwrap_or(INVALID_WITNESS_OVERFLOW_OFFSET);
            row.overflow_capacity = row.len;
        }
        self.witness_overflow = compacted;
    }

    fn insert_candidate(&mut self, update: CandidateUpdate, capacity: usize) -> (bool, usize) {
        if capacity == 0 {
            return (false, self.remove_witness_row(update.dst));
        }
        let candidate = ViewLuneCandidate {
            dst: update.dst,
            dist2: update.dist2,
        };
        let insert_idx = match self
            .candidates
            .binary_search_by_key(&update.dst, |candidate| candidate.dst)
        {
            Ok(idx) => {
                let existing = &mut self.candidates[idx];
                existing.dist2 = existing.dist2.min(update.dist2);
                if self.candidate_worst_index() == Some(idx) {
                    self.recompute_candidate_worst();
                }
                return (false, 0);
            }
            Err(idx) => idx,
        };
        if self.candidates.len() < capacity {
            reserve_bounded_row_slot(
                &mut self.candidates,
                capacity,
                CANDIDATE_ROW_CAPACITY_GROWTH,
            );
            self.candidates.insert(insert_idx, candidate);
            self.recompute_candidate_worst();
            return (false, 0);
        }
        let Some(worst_idx) = self.current_candidate_worst_index() else {
            return (false, 0);
        };
        let worst = &self.candidates[worst_idx];
        if candidate_distance_cmp(&candidate, worst).is_lt() {
            let evicted_dst = worst.dst;
            self.candidates.remove(worst_idx);
            let stale_rows_pruned = self.remove_witness_row(evicted_dst);
            let insert_idx = if worst_idx < insert_idx {
                insert_idx.saturating_sub(1)
            } else {
                insert_idx
            };
            self.candidates.insert(insert_idx, candidate);
            self.recompute_candidate_worst();
            return (true, stale_rows_pruned);
        }
        (true, self.remove_witness_row(candidate.dst))
    }

    fn candidate_worst_index(&self) -> Option<usize> {
        let idx = usize::from(self.candidate_worst_idx);
        (self.candidate_worst_idx != INVALID_CANDIDATE_IDX && idx < self.candidates.len())
            .then_some(idx)
    }

    fn current_candidate_worst_index(&mut self) -> Option<usize> {
        if self.candidate_worst_index().is_none() {
            self.recompute_candidate_worst();
        }
        self.candidate_worst_index()
    }

    fn recompute_candidate_worst(&mut self) {
        let Some((idx, _)) = self
            .candidates
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| candidate_distance_cmp(left, right))
        else {
            self.candidate_worst_idx = INVALID_CANDIDATE_IDX;
            return;
        };
        self.candidate_worst_idx = u16::try_from(idx).unwrap_or(INVALID_CANDIDATE_IDX);
    }

    fn insert_witness(
        &mut self,
        victim: u32,
        witness: RuntimeLuneWitness,
        max_witnesses_per_victim: usize,
    ) {
        let victim_idx = match self
            .witness_rows
            .binary_search_by_key(&victim, |witness_row| witness_row.victim)
        {
            Ok(idx) => idx,
            Err(idx) => {
                reserve_bounded_row_slot(
                    &mut self.witness_rows,
                    self.candidates.len().max(1),
                    WITNESS_ROW_CAPACITY_GROWTH,
                );
                self.witness_rows.insert(
                    idx,
                    ViewLuneVictimWitnesses {
                        victim,
                        len: 0,
                        overflow_capacity: 0,
                        overflow_offset: INVALID_WITNESS_OVERFLOW_OFFSET,
                        inline: [RuntimeLuneWitness::default(); INLINE_WITNESS_CAP],
                    },
                );
                idx
            }
        };
        let victim_row = &mut self.witness_rows[victim_idx];
        victim_row.insert(
            witness,
            max_witnesses_per_victim,
            &mut self.witness_overflow,
        );
    }

    fn has_candidate(&self, dst: u32) -> bool {
        self.candidates
            .binary_search_by_key(&dst, |candidate| candidate.dst)
            .is_ok()
    }

    fn remove_witness_row(&mut self, victim: u32) -> usize {
        let Ok(idx) = self
            .witness_rows
            .binary_search_by_key(&victim, |witness_row| witness_row.victim)
        else {
            return 0;
        };
        self.witness_rows.remove(idx);
        1
    }

    fn has_selected_witness(&self, victim: u32, selected: &[u32]) -> bool {
        let Ok(idx) = self
            .witness_rows
            .binary_search_by_key(&victim, |witness_row| witness_row.victim)
        else {
            return false;
        };
        self.witness_slice(idx)
            .iter()
            .any(|witness| selected.contains(&witness.pivot))
    }

    fn witness_slice(&self, idx: usize) -> &[RuntimeLuneWitness] {
        self.witness_rows[idx].as_slice(&self.witness_overflow)
    }
}

fn reserve_bounded_row_slot<T>(items: &mut Vec<T>, max_capacity: usize, growth: usize) {
    let needed = items.len().saturating_add(1);
    if needed <= items.capacity() {
        return;
    }
    let max_capacity = max_capacity.max(needed);
    let growth = growth.max(1);
    let rounded = needed.saturating_add(growth - 1) / growth * growth;
    let target = rounded.min(max_capacity);
    if target > items.capacity() {
        items.reserve_exact(target - items.capacity());
    }
}

fn sort_inline_witnesses(items: &mut [RuntimeLuneWitness; INLINE_WITNESS_CAP], len: usize) {
    items[..len].sort_by(witness_retention_cmp);
}

fn candidate_distance_cmp(left: &ViewLuneCandidate, right: &ViewLuneCandidate) -> Ordering {
    left.dist2
        .total_cmp(&right.dist2)
        .then_with(|| left.dst.cmp(&right.dst))
}

fn witness_retention_cmp(left: &RuntimeLuneWitness, right: &RuntimeLuneWitness) -> Ordering {
    left.rank_key
        .cmp(&right.rank_key)
        .then_with(|| left.pivot.cmp(&right.pivot))
}

impl ViewLuneVictimWitnesses {
    fn has_overflow(&self) -> bool {
        self.overflow_offset != INVALID_WITNESS_OVERFLOW_OFFSET
    }

    fn as_slice<'a>(&'a self, overflow: &'a [RuntimeLuneWitness]) -> &'a [RuntimeLuneWitness] {
        let len = usize::from(self.len);
        if !self.has_overflow() {
            return &self.inline[..len];
        }
        let offset = self.overflow_offset as usize;
        &overflow[offset..offset + len]
    }

    fn insert(
        &mut self,
        witness: RuntimeLuneWitness,
        max_witnesses: usize,
        overflow: &mut Vec<RuntimeLuneWitness>,
    ) {
        let max_witnesses = max_witnesses.max(1);
        if self.has_overflow() {
            let offset = self.overflow_offset as usize;
            let capacity = usize::from(self.overflow_capacity);
            let len = insert_witness_slice(
                &mut overflow[offset..offset + capacity],
                usize::from(self.len),
                witness,
                max_witnesses.min(capacity),
            );
            self.len = u16::try_from(len).unwrap_or(u16::MAX);
            return;
        }

        if let Some(existing) = self.inline[..usize::from(self.len)]
            .iter_mut()
            .find(|existing| existing.pivot == witness.pivot)
        {
            if witness_retention_cmp(&witness, existing).is_lt() {
                *existing = witness;
                sort_inline_witnesses(&mut self.inline, usize::from(self.len));
            }
            return;
        }

        let inline_limit = max_witnesses.min(INLINE_WITNESS_CAP);
        let len = usize::from(self.len);
        if len < inline_limit {
            self.inline[len] = witness;
            self.len += 1;
            sort_inline_witnesses(&mut self.inline, len + 1);
            return;
        }

        if max_witnesses <= INLINE_WITNESS_CAP {
            let worst_idx = inline_limit.saturating_sub(1);
            if inline_limit > 0 && witness_retention_cmp(&witness, &self.inline[worst_idx]).is_lt()
            {
                self.inline[worst_idx] = witness;
                sort_inline_witnesses(&mut self.inline, inline_limit);
            }
            return;
        }

        let capacity = max_witnesses.min(usize::from(u16::MAX));
        let offset = overflow.len();
        overflow.resize(offset + capacity, RuntimeLuneWitness::default());
        overflow[offset..offset + len].copy_from_slice(&self.inline[..len]);
        let new_len = insert_witness_slice(
            &mut overflow[offset..offset + capacity],
            len,
            witness,
            capacity,
        );
        self.overflow_offset = u32::try_from(offset).unwrap_or(INVALID_WITNESS_OVERFLOW_OFFSET);
        self.overflow_capacity = u16::try_from(capacity).unwrap_or(u16::MAX);
        self.len = u16::try_from(new_len).unwrap_or(u16::MAX);
    }
}

fn insert_witness_slice(
    witnesses: &mut [RuntimeLuneWitness],
    len: usize,
    witness: RuntimeLuneWitness,
    max_witnesses: usize,
) -> usize {
    let max_witnesses = max_witnesses.min(witnesses.len()).max(1);
    let len = len.min(max_witnesses);
    if let Some(existing) = witnesses[..len]
        .iter_mut()
        .find(|existing| existing.pivot == witness.pivot)
    {
        if witness_retention_cmp(&witness, existing).is_lt() {
            *existing = witness;
            witnesses[..len].sort_by(witness_retention_cmp);
        }
        return len;
    }
    if len < max_witnesses {
        witnesses[len] = witness;
        witnesses[..len + 1].sort_by(witness_retention_cmp);
        return len + 1;
    }
    let worst_idx = max_witnesses - 1;
    if witness_retention_cmp(&witness, &witnesses[worst_idx]).is_lt() {
        witnesses[worst_idx] = witness;
        witnesses[..max_witnesses].sort_by(witness_retention_cmp);
    }
    len
}

fn witness_rank_key(pivot_rank: u8, margin_q16: u16, victim_rank: u8) -> u32 {
    (u32::from(pivot_rank) << 24) | (u32::from(u16::MAX - margin_q16) << 8) | u32::from(victim_rank)
}

fn pending_witness_retention_key(witness: &PendingLuneWitness) -> u32 {
    witness_rank_key(
        witness.pivot_rank.max(1),
        witness.margin_q16,
        witness.victim_rank.max(1),
    )
}

fn count_overlap(left: &[u32], right: &[u32]) -> usize {
    left.iter().filter(|dst| right.contains(dst)).count()
}

impl ViewLuneOracleReport {
    fn to_json(&self) -> String {
        let stats = self.stats;
        let opts = self.options;
        format!(
            concat!(
                "{{\n",
                "  \"schema\": \"{}\",\n",
                "  \"options\": {{\n",
                "    \"max_degree\": {},\n",
                "    \"candidate_width\": {},\n",
                "    \"candidate_width_multiplier\": {},\n",
                "    \"max_witnesses_per_victim\": {},\n",
                "    \"nearest_core\": {}\n",
                "  }},\n",
                "  \"metrics\": {{\n",
                "    \"raw_candidate_edges\": {},\n",
                "    \"raw_witness_records\": {},\n",
                "    \"filtered_witness_records\": {},\n",
                "    \"merged_candidates\": {},\n",
                "    \"victims_with_witness\": {},\n",
                "    \"stale_witness_rows_pruned\": {},\n",
                "    \"distinct_witness_families\": {},\n",
                "    \"pruned_by_selected_witness\": {},\n",
                "    \"fill_after_prune\": {},\n",
                "    \"rows_modified\": {},\n",
                "    \"final_edges\": {},\n",
                "    \"rows_with_candidates\": {},\n",
                "    \"rows_over_candidate_width\": {},\n",
                "    \"leaf_witness_emit_ms\": {},\n",
                "    \"reduce_wall_ms\": {},\n",
                "    \"estimated_accumulator_bytes\": {},\n",
                "    \"overlap_with_base_edges\": {},\n",
                "    \"overlap_with_base_rows\": {},\n",
                "    \"rows_compared_with_base\": {},\n",
                "    \"final_vector_io_count\": 0\n",
                "  }},\n",
                "  \"simulated_widths\": [1.25, 1.5, 2.0]\n",
                "}}\n"
            ),
            self.schema,
            opts.max_degree,
            candidate_width_for_options(&opts),
            opts.candidate_width_multiplier,
            opts.max_witnesses_per_victim,
            opts.nearest_core,
            stats.raw_candidate_edges,
            stats.raw_witness_records,
            stats.filtered_witness_records,
            stats.merged_candidates,
            stats.victims_with_witness,
            stats.stale_witness_rows_pruned,
            stats.distinct_witness_families,
            stats.pruned_by_selected_witness,
            stats.fill_after_prune,
            stats.rows_modified,
            stats.final_edges,
            stats.rows_with_candidates,
            stats.rows_over_candidate_width,
            stats.leaf_witness_emit.as_secs_f64() * 1000.0,
            stats.reduce_wall.as_secs_f64() * 1000.0,
            stats.estimated_accumulator_bytes,
            self.overlap_with_base_edges,
            self.overlap_with_base_rows,
            self.rows_compared_with_base,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(max_degree: usize) -> ViewLunePruneOptions {
        ViewLunePruneOptions {
            max_degree,
            metric: Metric::L2,
            candidate_width_multiplier: 2.0,
            max_witnesses_per_victim: 2,
            nearest_core: 0,
        }
    }

    fn runtime_witness(
        pivot: u32,
        margin_q16: u16,
        pivot_rank: u8,
        victim_rank: u8,
    ) -> RuntimeLuneWitness {
        RuntimeLuneWitness {
            pivot,
            rank_key: witness_rank_key(pivot_rank, margin_q16, victim_rank),
        }
    }

    fn pending_edge(dst: u32, dist: f32) -> PendingEdge {
        pending_edge_for(0, dst, dist)
    }

    fn pending_edge_for(src: usize, dst: u32, dist: f32) -> PendingEdge {
        PendingEdge {
            p: src,
            c: dst,
            hash: 0,
            dist,
            mandatory: false,
            local_rank: dst.min(u8::MAX as u32) as u8,
            flags: PENDING_EDGE_DIRECT,
        }
    }

    #[test]
    fn strict_lune_predicate_matches_squared_alpha_semantics() {
        assert!(is_lune_witness(12.0, 9.99));
        assert!(!is_lune_witness(12.0, 10.0));
        assert!(!is_lune_witness(12.0, 10.01));
    }

    #[test]
    fn witness_retention_dedups_and_caps_per_victim() {
        let mut row = RuntimeViewLuneRow::default();
        row.insert_witness(9, runtime_witness(3, 10, 2, 3), 2);
        row.insert_witness(9, runtime_witness(3, 30, 2, 3), 2);
        row.insert_witness(9, runtime_witness(4, 40, 1, 3), 2);
        row.insert_witness(9, runtime_witness(5, 50, 3, 3), 2);
        let witnesses = row.witness_slice(0);
        assert_eq!(witnesses.len(), 2);
        assert_eq!(witnesses[0].pivot, 4);
        assert_eq!(witnesses[1].pivot, 3);
        assert_eq!(witnesses[1].rank_key, witness_rank_key(2, 30, 3));
    }

    #[test]
    fn witness_retention_uses_inline_then_overflow_for_larger_caps() {
        let mut row = RuntimeViewLuneRow::default();
        for pivot in 0..2 {
            row.insert_witness(9, runtime_witness(pivot, 10, pivot as u8, 1), 4);
        }
        assert_eq!(row.witness_slice(0).len(), 2);
        assert!(!row.witness_rows[0].has_overflow());

        row.insert_witness(9, runtime_witness(2, 10, 2, 1), 4);
        assert_eq!(row.witness_slice(0).len(), 3);
        assert!(row.witness_rows[0].has_overflow());

        row.insert_witness(10, runtime_witness(0, 10, 0, 1), 6);
        for pivot in 1..6 {
            row.insert_witness(10, runtime_witness(pivot, 10, pivot as u8, 1), 6);
        }
        let overflow_idx = row
            .witness_rows
            .iter()
            .position(|witness_row| witness_row.victim == 10)
            .unwrap();
        assert!(row.witness_rows[overflow_idx].has_overflow());
        assert_eq!(row.witness_slice(overflow_idx).len(), 6);
    }

    #[test]
    fn witness_family_metric_is_tracked_outside_retained_records() {
        let acc = RuntimeViewLuneAccumulator::create(1, options(2)).unwrap();
        acc.write_edges(7, &[pending_edge(1, 1.0), pending_edge(2, 2.0)])
            .unwrap();
        let witness = PendingLuneWitness {
            src: 0,
            pivot: 1,
            victim: 2,
            margin_q16: 9,
            pivot_rank: 1,
            victim_rank: 2,
        };
        acc.write_witnesses(11, &[witness]).unwrap();
        acc.write_witnesses(12, &[witness]).unwrap();
        acc.write_witnesses(12, &[witness]).unwrap();
        let stats = acc.snapshot_stats();
        assert_eq!(stats.distinct_witness_families, 2);
        assert_eq!(stats.victims_with_witness, 1);
        assert_eq!(acc.rows[0].lock().witness_slice(0).len(), 1);
    }

    #[test]
    fn witnesses_without_retained_candidates_are_filtered() {
        let acc = RuntimeViewLuneAccumulator::create(1, options(1)).unwrap();
        acc.write_edges(7, &[pending_edge(1, 1.0)]).unwrap();
        acc.write_witnesses(
            7,
            &[PendingLuneWitness {
                src: 0,
                pivot: 1,
                victim: 2,
                margin_q16: 9,
                pivot_rank: 1,
                victim_rank: 2,
            }],
        )
        .unwrap();

        let stats = acc.snapshot_stats();
        assert_eq!(stats.raw_witness_records, 1);
        assert_eq!(stats.filtered_witness_records, 1);
        assert_eq!(stats.victims_with_witness, 0);
    }

    #[test]
    fn evicting_candidate_prunes_stale_victim_witness_row() {
        let acc = RuntimeViewLuneAccumulator::create(1, options(1)).unwrap();
        acc.write_edges(7, &[pending_edge(1, 1.0), pending_edge(2, 2.0)])
            .unwrap();
        acc.write_witnesses(
            7,
            &[PendingLuneWitness {
                src: 0,
                pivot: 1,
                victim: 2,
                margin_q16: 9,
                pivot_rank: 1,
                victim_rank: 2,
            }],
        )
        .unwrap();
        assert_eq!(acc.snapshot_stats().victims_with_witness, 1);

        acc.write_edges(7, &[pending_edge(3, 0.5)]).unwrap();
        let stats = acc.snapshot_stats();
        assert_eq!(stats.victims_with_witness, 0);
        assert_eq!(stats.stale_witness_rows_pruned, 1);
    }

    #[test]
    fn compact_storage_releases_row_capacity_without_changing_lengths() {
        let acc = RuntimeViewLuneAccumulator::create(1, options(4)).unwrap();
        acc.write_edges(
            7,
            &[
                pending_edge(1, 1.0),
                pending_edge(2, 2.0),
                pending_edge(3, 3.0),
            ],
        )
        .unwrap();
        acc.write_witnesses(
            7,
            &[
                PendingLuneWitness {
                    src: 0,
                    pivot: 1,
                    victim: 2,
                    margin_q16: 9,
                    pivot_rank: 1,
                    victim_rank: 2,
                },
                PendingLuneWitness {
                    src: 0,
                    pivot: 1,
                    victim: 3,
                    margin_q16: 9,
                    pivot_rank: 1,
                    victim_rank: 3,
                },
            ],
        )
        .unwrap();

        {
            let mut row = acc.rows[0].lock();
            row.candidates.reserve(8);
            row.witness_rows.reserve(8);
        }

        let compacted = acc.compact_storage();
        assert_eq!(compacted.candidate_len, 3);
        assert_eq!(compacted.witness_row_len, 2);
        assert!(compacted.candidate_capacity_before > compacted.candidate_capacity_after);
        assert!(compacted.witness_row_capacity_before > compacted.witness_row_capacity_after);
        assert!(compacted.estimated_bytes_released() > 0);
    }

    #[test]
    fn row_vectors_grow_in_bounded_chunks() {
        let mut row = RuntimeViewLuneRow::default();
        for dst in 1..=65 {
            row.insert_candidate(
                CandidateUpdate {
                    dst,
                    dist2: dst as f32,
                },
                128,
            );
        }
        assert!(row.candidates.capacity() >= 65);
        assert!(row.candidates.capacity() <= 72);

        for victim in 1..=65 {
            row.insert_witness(victim, runtime_witness(1, 10, 1, victim as u8), 4);
        }
        assert!(row.witness_rows.capacity() >= 65);
        assert!(row.witness_rows.capacity() <= 68);
    }

    #[test]
    fn candidate_retention_merges_duplicates_and_preserves_best_bounded_rows() {
        let mut row = RuntimeViewLuneRow::default();
        row.insert_candidate(CandidateUpdate { dst: 3, dist2: 3.0 }, 2);
        row.insert_candidate(CandidateUpdate { dst: 1, dist2: 1.0 }, 2);
        row.insert_candidate(CandidateUpdate { dst: 3, dist2: 0.5 }, 2);
        assert_eq!(row.candidates.len(), 2);
        let merged = row
            .candidates
            .iter()
            .find(|candidate| candidate.dst == 3)
            .unwrap();
        assert_eq!(merged.dist2, 0.5);

        row.insert_candidate(CandidateUpdate { dst: 2, dist2: 2.0 }, 2);
        assert!(row.candidates.iter().all(|candidate| candidate.dst != 2));

        row.insert_candidate(
            CandidateUpdate {
                dst: 4,
                dist2: 0.25,
            },
            2,
        );
        let dsts: Vec<_> = row
            .candidates
            .iter()
            .map(|candidate| candidate.dst)
            .collect();
        assert_eq!(dsts, vec![3, 4]);
    }

    #[test]
    fn candidate_worst_cache_refreshes_when_duplicate_improves_worst() {
        let mut row = RuntimeViewLuneRow::default();
        row.insert_candidate(
            CandidateUpdate {
                dst: 1,
                dist2: 10.0,
            },
            2,
        );
        row.insert_candidate(
            CandidateUpdate {
                dst: 2,
                dist2: 20.0,
            },
            2,
        );
        row.insert_candidate(CandidateUpdate { dst: 2, dist2: 1.0 }, 2);
        row.insert_candidate(CandidateUpdate { dst: 3, dist2: 5.0 }, 2);

        let dsts: Vec<_> = row
            .candidates
            .iter()
            .map(|candidate| candidate.dst)
            .collect();
        assert_eq!(dsts, vec![2, 3]);
    }

    #[test]
    fn candidate_digest_uses_compact_row_metadata() {
        assert!(std::mem::size_of::<CandidateDigestRow>() <= 16);
        assert!(
            std::mem::size_of::<CandidateDigestRow>()
                < std::mem::size_of::<Vec<ViewLuneCandidate>>()
        );

        let mut digest = CandidateDigestRun::new(2, 4, 32);
        digest.insert(0, 4, 4.0);
        digest.insert(0, 1, 1.0);
        digest.insert(0, 4, 0.5);
        digest.insert(0, 2, 2.0);
        digest.insert(0, 3, 3.0);
        digest.insert(0, 5, 5.0);

        assert_eq!(digest.rows[0].len, 4);
        assert_eq!(digest.candidate_slots, 4);
        let offset = digest.rows[0].offset as usize;
        let len = digest.rows[0].len as usize;
        let row = &digest.candidates[offset..offset + len];
        assert_eq!(
            row.iter()
                .map(|candidate| (candidate.dst, candidate.dist2))
                .collect::<Vec<_>>(),
            vec![(1, 1.0), (2, 2.0), (3, 3.0), (4, 0.5)]
        );

        digest.release_storage();
        assert_eq!(digest.candidates.capacity(), 0);
        assert_eq!(digest.touched_rows.capacity(), 0);
        assert!(digest.rows.iter().all(|row| row.len == 0 && !row.touched));
    }

    #[test]
    fn candidate_digest_spill_matches_runtime_accumulator() {
        let opts = options(2);
        let dir = std::env::temp_dir().join(format!(
            "view-lune-candidate-digest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let memory = RuntimeViewLuneAccumulator::create(2, opts).unwrap();
        let recorder = SpillingViewLuneRecorder::create_with_mode(
            2,
            opts,
            &dir,
            1,
            ViewLuneCandidateSpillMode::DigestRuns,
            32,
        )
        .unwrap();

        let batch1 = vec![
            pending_edge_for(0, 1, 1.0),
            pending_edge_for(0, 2, 2.0),
            pending_edge_for(0, 3, 3.0),
            pending_edge_for(0, 4, 4.0),
            pending_edge_for(0, 5, 5.0),
            pending_edge_for(1, 10, 1.5),
            pending_edge_for(1, 11, 2.5),
        ];
        let batch2 = vec![
            pending_edge_for(0, 4, 0.25),
            pending_edge_for(0, 6, 6.0),
            pending_edge_for(0, 7, 7.0),
            pending_edge_for(1, 12, 0.5),
            pending_edge_for(1, 13, 3.5),
        ];
        memory.write_edges(7, &batch1).unwrap();
        memory.write_edges(8, &batch2).unwrap();
        let mut spill_batch1 = batch1.clone();
        let mut spill_batch2 = batch2.clone();
        recorder.write_edges_in_place(7, &mut spill_batch1).unwrap();
        recorder.write_edges_in_place(8, &mut spill_batch2).unwrap();

        let witnesses = vec![
            PendingLuneWitness {
                src: 0,
                pivot: 4,
                victim: 1,
                margin_q16: 9,
                pivot_rank: 1,
                victim_rank: 2,
            },
            PendingLuneWitness {
                src: 0,
                pivot: 4,
                victim: 5,
                margin_q16: 9,
                pivot_rank: 1,
                victim_rank: 5,
            },
        ];
        memory.write_witnesses(9, &witnesses).unwrap();
        let mut spill_witnesses = witnesses.clone();
        recorder
            .write_witnesses_in_place(9, &mut spill_witnesses)
            .unwrap();

        let (memory_rows, memory_stats, _, _) = memory.reduce_all_rows(None);
        let mut spilled_rows = vec![Vec::new(); 2];
        let (spill_stats, _, _) = recorder
            .reduce_all_rows_with(None, |source, row| {
                spilled_rows[source] = row.to_vec();
                Ok(())
            })
            .unwrap();

        assert_eq!(spilled_rows, memory_rows);
        assert_eq!(spilled_rows[0], vec![4, 2]);
        assert_eq!(spilled_rows[1], vec![12, 10]);
        assert_eq!(
            spill_stats.raw_candidate_edges,
            memory_stats.raw_candidate_edges
        );
        assert_eq!(
            spill_stats.raw_witness_records,
            memory_stats.raw_witness_records
        );
        assert_eq!(
            spill_stats.filtered_witness_records,
            memory_stats.filtered_witness_records
        );

        recorder.cleanup().unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn selected_pivot_conditionally_prunes_victim() {
        let acc = RuntimeViewLuneAccumulator::create(1, options(2)).unwrap();
        acc.write_edges(
            7,
            &[
                PendingEdge {
                    p: 0,
                    c: 1,
                    hash: 0,
                    dist: 1.0,
                    mandatory: false,
                    local_rank: 1,
                    flags: PENDING_EDGE_DIRECT,
                },
                PendingEdge {
                    p: 0,
                    c: 2,
                    hash: 0,
                    dist: 2.0,
                    mandatory: false,
                    local_rank: 2,
                    flags: PENDING_EDGE_DIRECT,
                },
                PendingEdge {
                    p: 0,
                    c: 3,
                    hash: 0,
                    dist: 3.0,
                    mandatory: false,
                    local_rank: 3,
                    flags: PENDING_EDGE_DIRECT,
                },
            ],
        )
        .unwrap();
        acc.write_witnesses(
            7,
            &[PendingLuneWitness {
                src: 0,
                pivot: 1,
                victim: 2,
                margin_q16: 9,
                pivot_rank: 1,
                victim_rank: 2,
            }],
        )
        .unwrap();
        let row = acc.reduce_row(0, Some(&[1, 2]));
        assert_eq!(row.neighbors, vec![1, 3]);
        assert_eq!(row.pruned_by_selected_witness, 1);
    }

    #[test]
    fn nearest_core_protects_prefix() {
        let mut opts = options(2);
        opts.nearest_core = 2;
        let acc = RuntimeViewLuneAccumulator::create(1, opts).unwrap();
        acc.write_edges(
            7,
            &[
                PendingEdge {
                    p: 0,
                    c: 1,
                    hash: 0,
                    dist: 1.0,
                    mandatory: false,
                    local_rank: 1,
                    flags: PENDING_EDGE_DIRECT,
                },
                PendingEdge {
                    p: 0,
                    c: 2,
                    hash: 0,
                    dist: 2.0,
                    mandatory: false,
                    local_rank: 2,
                    flags: PENDING_EDGE_DIRECT,
                },
                PendingEdge {
                    p: 0,
                    c: 3,
                    hash: 0,
                    dist: 3.0,
                    mandatory: false,
                    local_rank: 3,
                    flags: PENDING_EDGE_DIRECT,
                },
            ],
        )
        .unwrap();
        acc.write_witnesses(
            7,
            &[PendingLuneWitness {
                src: 0,
                pivot: 1,
                victim: 2,
                margin_q16: 9,
                pivot_rank: 1,
                victim_rank: 2,
            }],
        )
        .unwrap();
        let row = acc.reduce_row(0, Some(&[1, 2]));
        assert_eq!(row.neighbors, vec![1, 2]);
        assert_eq!(row.pruned_by_selected_witness, 0);
    }

    #[test]
    fn refill_preserves_degree_uniqueness_and_distance_order() {
        let acc = RuntimeViewLuneAccumulator::create(1, options(3)).unwrap();
        let edges: Vec<_> = (1..=5)
            .map(|dst| PendingEdge {
                p: 0,
                c: dst,
                hash: 0,
                dist: dst as f32,
                mandatory: false,
                local_rank: dst as u8,
                flags: PENDING_EDGE_DIRECT,
            })
            .collect();
        acc.write_edges(7, &edges).unwrap();
        acc.write_witnesses(
            7,
            &[
                PendingLuneWitness {
                    src: 0,
                    pivot: 1,
                    victim: 2,
                    margin_q16: 9,
                    pivot_rank: 1,
                    victim_rank: 2,
                },
                PendingLuneWitness {
                    src: 0,
                    pivot: 1,
                    victim: 3,
                    margin_q16: 9,
                    pivot_rank: 1,
                    victim_rank: 3,
                },
                PendingLuneWitness {
                    src: 0,
                    pivot: 1,
                    victim: 4,
                    margin_q16: 9,
                    pivot_rank: 1,
                    victim_rank: 4,
                },
            ],
        )
        .unwrap();
        let row = acc.reduce_row(0, Some(&[1, 2, 3]));
        assert_eq!(row.neighbors, vec![1, 5, 2]);
        assert_eq!(row.neighbors.len(), 3);
        let mut uniq = row.neighbors.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 3);
    }

    #[test]
    fn mirror_edges_are_candidates_but_not_witnesses() {
        let acc = RuntimeViewLuneAccumulator::create(1, options(2)).unwrap();
        acc.write_edges(
            7,
            &[PendingEdge {
                p: 0,
                c: 42,
                hash: 0,
                dist: 1.0,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_MIRROR,
            }],
        )
        .unwrap();
        let row = acc.reduce_row(0, None);
        assert_eq!(row.neighbors, vec![42]);
    }
}
