use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use rayon::prelude::*;

use super::direct_io::{DirectIoConfig, DirectIoFile};
use super::hash_prune::HashPruneReservoir;
use super::leaf_build::{PENDING_EDGE_DIRECT, PendingEdge, PendingEdgeSink};
use crate::common::{AnnError, AnnResult};

const RUN_RECORD_BYTES: usize = std::mem::size_of::<SpillRecord16>();
const DEFAULT_SPILL_CACHE_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_SHARD_POINTS: usize = 1_000_000;
const SPILL_FLAG_MANDATORY: u16 = 1 << 0;
const DEFAULT_PART_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const SEGMENT_MANIFEST_NAME: &str = "candidate_segments.manifest";
const DIRECT_CANDIDATE_BUFFER_BYTES: usize = 1024 * 1024;
const REDUCE_NEIGHBOR_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpillRunStats {
    pub part_files: usize,
    pub segment_count: usize,
    pub shard_groups: usize,
    pub max_parts_per_shard: usize,
    pub max_segments_per_shard: usize,
    pub record_bytes: usize,
    pub manifest_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpillReduceStats {
    pub shard_groups: usize,
    pub segments_scanned: usize,
    pub part_files_opened: usize,
    pub max_segments_in_group: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpillPartFileDesc {
    pub shard_id: usize,
    pub file_id: u16,
    pub path: PathBuf,
    pub byte_len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillSegmentDesc {
    pub shard_id: u32,
    pub file_id: u16,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub record_count: u32,
    pub src_min: u32,
    pub src_max: u32,
    pub sorted_by_src: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpillArtifacts {
    pub part_files: Vec<SpillPartFileDesc>,
    pub segment_manifest: Vec<SpillSegmentDesc>,
    pub manifest_path: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InMemorySpillArtifacts {
    pub shard_points: usize,
    pub segments: Vec<InMemorySpillSegment>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InMemorySpillSegment {
    pub shard_id: usize,
    records: Vec<SpillRecord16>,
    src_min: u32,
    src_max: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct SpillRecord16 {
    src: u32,
    dst: u32,
    hash: u16,
    flags: u16,
    dist: f32,
}

impl SpillRecord16 {
    fn from_pending_edge(edge: &PendingEdge) -> AnnResult<Self> {
        let src = u32::try_from(edge.p).map_err(|_| {
            AnnError::log_index_error(format!(
                "Spill writer only supports u32 source ids, got p={}",
                edge.p
            ))
        })?;
        Ok(Self {
            src,
            dst: edge.c,
            hash: edge.hash,
            flags: if edge.mandatory {
                SPILL_FLAG_MANDATORY
            } else {
                0
            },
            dist: edge.dist,
        })
    }

    fn to_pending_edge(self) -> PendingEdge {
        PendingEdge {
            p: self.src as usize,
            c: self.dst,
            hash: self.hash,
            dist: self.dist,
            mandatory: (self.flags & SPILL_FLAG_MANDATORY) != 0,
            local_rank: 1,
            flags: PENDING_EDGE_DIRECT,
        }
    }
}

#[derive(Debug)]
pub struct CandidateRunWriter {
    inner: CandidateRunWriterInner,
    records: usize,
}

#[derive(Debug)]
enum CandidateRunWriterInner {
    Buffered(BufWriter<File>),
    Direct {
        file: DirectIoFile,
        offset: u64,
        buffer: Vec<u8>,
    },
}

impl CandidateRunWriter {
    pub fn create(path: &Path) -> AnnResult<Self> {
        Self::create_with_config(path, DirectIoConfig::disabled())
    }

    pub fn create_with_config(path: &Path, io_cfg: DirectIoConfig) -> AnnResult<Self> {
        let inner = if io_cfg.enabled {
            CandidateRunWriterInner::Direct {
                file: DirectIoFile::open_rw(path, io_cfg, true)?,
                offset: 0,
                buffer: Vec::with_capacity(DIRECT_CANDIDATE_BUFFER_BYTES),
            }
        } else {
            CandidateRunWriterInner::Buffered(BufWriter::new(File::create(path)?))
        };
        Ok(Self { inner, records: 0 })
    }

    pub fn push(&mut self, edge: &PendingEdge) -> AnnResult<()> {
        let record = SpillRecord16::from_pending_edge(edge)?;
        let mut bytes = [0u8; RUN_RECORD_BYTES];
        bytes[0..4].copy_from_slice(&record.src.to_le_bytes());
        bytes[4..8].copy_from_slice(&record.dst.to_le_bytes());
        bytes[8..10].copy_from_slice(&record.hash.to_le_bytes());
        bytes[10..12].copy_from_slice(&record.flags.to_le_bytes());
        bytes[12..16].copy_from_slice(&record.dist.to_le_bytes());
        match &mut self.inner {
            CandidateRunWriterInner::Buffered(writer) => {
                writer.write_all(&bytes)?;
            }
            CandidateRunWriterInner::Direct {
                file,
                offset,
                buffer,
            } => {
                buffer.extend_from_slice(&bytes);
                if buffer.len() >= DIRECT_CANDIDATE_BUFFER_BYTES {
                    flush_direct_candidate_buffer(file, offset, buffer)?;
                }
            }
        }
        self.records += 1;
        Ok(())
    }

    pub fn finish(mut self) -> AnnResult<()> {
        match &mut self.inner {
            CandidateRunWriterInner::Buffered(writer) => {
                writer.flush()?;
            }
            CandidateRunWriterInner::Direct {
                file,
                offset,
                buffer,
            } => {
                flush_direct_candidate_buffer(file, offset, buffer)?;
                file.set_len(*offset)?;
                file.sync_all()?;
            }
        }
        Ok(())
    }
}

fn flush_direct_candidate_buffer(
    file: &DirectIoFile,
    offset: &mut u64,
    buffer: &mut Vec<u8>,
) -> AnnResult<()> {
    if buffer.is_empty() {
        return Ok(());
    }
    file.write_all_at(buffer, *offset)?;
    *offset += buffer.len() as u64;
    buffer.clear();
    Ok(())
}

#[derive(Debug)]
struct SpillPartWriter {
    desc: SpillPartFileDesc,
    writer: CandidateRunWriter,
}

#[derive(Debug)]
struct SpillState {
    base_path: PathBuf,
    shard_points: usize,
    io_cfg: DirectIoConfig,
    buffers_by_shard: HashMap<usize, Vec<PendingEdge>>,
    part_writers_by_shard: HashMap<usize, SpillPartWriter>,
    next_part_id_by_shard: HashMap<usize, u16>,
    part_files: Vec<SpillPartFileDesc>,
    segment_manifest: Vec<SpillSegmentDesc>,
    manifest_path: PathBuf,
    flush_threshold_edges: usize,
    part_max_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct SpillEdgeSink {
    inner: Arc<Mutex<SpillState>>,
}

#[derive(Debug)]
struct InMemorySpillState {
    shard_points: usize,
    buffers_by_shard: HashMap<usize, Vec<SpillRecord16>>,
    segments: Vec<InMemorySpillSegment>,
}

#[derive(Debug, Clone)]
pub struct InMemorySpillEdgeSink {
    inner: Arc<Mutex<InMemorySpillState>>,
    shard_points: usize,
    flush_threshold_edges: usize,
}

impl SpillEdgeSink {
    pub fn create(path: &Path, spill_cache_bytes: usize) -> AnnResult<Self> {
        Self::create_sharded_with_config(
            path,
            spill_cache_bytes,
            DEFAULT_SHARD_POINTS,
            DirectIoConfig::disabled(),
        )
    }

    pub fn create_sharded(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
    ) -> AnnResult<Self> {
        Self::create_sharded_with_config(
            path,
            spill_cache_bytes,
            shard_points,
            DirectIoConfig::disabled(),
        )
    }

    pub fn create_sharded_with_config(
        path: &Path,
        spill_cache_bytes: usize,
        shard_points: usize,
        io_cfg: DirectIoConfig,
    ) -> AnnResult<Self> {
        let spill_cache_bytes = effective_spill_cache_bytes(spill_cache_bytes);
        let estimated_edge_bytes = std::mem::size_of::<PendingEdge>().max(1);
        let flush_threshold_edges = (spill_cache_bytes / estimated_edge_bytes)
            .max(1)
            .min(1_000_000);
        Ok(Self {
            inner: Arc::new(Mutex::new(SpillState {
                base_path: path.to_path_buf(),
                shard_points: shard_points.max(1),
                io_cfg,
                buffers_by_shard: HashMap::new(),
                part_writers_by_shard: HashMap::new(),
                next_part_id_by_shard: HashMap::new(),
                part_files: Vec::new(),
                segment_manifest: Vec::new(),
                manifest_path: path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(SEGMENT_MANIFEST_NAME),
                flush_threshold_edges,
                part_max_bytes: DEFAULT_PART_MAX_BYTES,
            })),
        })
    }

    pub fn finish(self) -> AnnResult<SpillArtifacts> {
        match Arc::try_unwrap(self.inner) {
            Ok(inner) => {
                let mut state = inner.into_inner();
                flush_all_buffers_to_parts(&mut state)?;
                finish_part_writers(&mut state)?;
                write_segment_manifest(&state)?;
                Ok(SpillArtifacts {
                    part_files: state.part_files,
                    segment_manifest: state.segment_manifest,
                    manifest_path: state.manifest_path,
                })
            }
            Err(shared) => {
                let mut state = shared.lock();
                flush_all_buffers_to_parts(&mut state)?;
                finish_part_writers(&mut state)?;
                write_segment_manifest(&state)?;
                Ok(SpillArtifacts {
                    part_files: state.part_files.clone(),
                    segment_manifest: state.segment_manifest.clone(),
                    manifest_path: state.manifest_path.clone(),
                })
            }
        }
    }
}

impl InMemorySpillEdgeSink {
    pub fn create_sharded(spill_cache_bytes: usize, shard_points: usize) -> Self {
        let spill_cache_bytes = effective_spill_cache_bytes(spill_cache_bytes);
        let flush_threshold_edges = (spill_cache_bytes / RUN_RECORD_BYTES.max(1))
            .max(1)
            .min(1_000_000);
        let shard_points = shard_points.max(1);
        Self {
            inner: Arc::new(Mutex::new(InMemorySpillState {
                shard_points,
                buffers_by_shard: HashMap::new(),
                segments: Vec::new(),
            })),
            shard_points,
            flush_threshold_edges,
        }
    }

    pub fn finish(self) -> AnnResult<InMemorySpillArtifacts> {
        match Arc::try_unwrap(self.inner) {
            Ok(inner) => {
                let mut state = inner.into_inner();
                flush_all_memory_buffers_to_segments(&mut state)?;
                Ok(InMemorySpillArtifacts {
                    shard_points: state.shard_points,
                    segments: state.segments,
                })
            }
            Err(shared) => {
                let mut state = shared.lock();
                flush_all_memory_buffers_to_segments(&mut state)?;
                Ok(InMemorySpillArtifacts {
                    shard_points: state.shard_points,
                    segments: state.segments.clone(),
                })
            }
        }
    }
}

impl PendingEdgeSink for SpillEdgeSink {
    fn flush_pending_edges(&self, edges: &mut Vec<PendingEdge>) -> AnnResult<()> {
        if edges.is_empty() {
            return Ok(());
        }

        edges.sort_unstable_by_key(|edge| edge.p);
        let mut state = self.inner.lock();
        let buffer_capacity = state.flush_threshold_edges.min(65_536);
        for edge in edges.drain(..) {
            let shard_id = edge.p / state.shard_points;
            let buffer = state
                .buffers_by_shard
                .entry(shard_id)
                .or_insert_with(|| Vec::with_capacity(buffer_capacity));
            buffer.push(edge);
            if buffer.len() >= state.flush_threshold_edges {
                flush_shard_buffer_to_part(&mut state, shard_id)?;
            }
        }
        Ok(())
    }
}

impl PendingEdgeSink for InMemorySpillEdgeSink {
    fn flush_pending_edges(&self, edges: &mut Vec<PendingEdge>) -> AnnResult<()> {
        if edges.is_empty() {
            return Ok(());
        }

        let groups = encode_spill_records_by_shard(edges, self.shard_points)?;
        let mut state = self.inner.lock();
        let flush_threshold_edges = self.flush_threshold_edges;
        let buffer_capacity = flush_threshold_edges.min(65_536);
        for (shard_id, records) in groups {
            let should_flush = {
                let buffer = state
                    .buffers_by_shard
                    .entry(shard_id)
                    .or_insert_with(|| Vec::with_capacity(buffer_capacity));
                buffer.extend(records);
                buffer.len() >= flush_threshold_edges
            };
            if should_flush {
                flush_memory_shard_buffer_to_segment(&mut state, shard_id)?;
            }
        }
        Ok(())
    }
}

fn encode_spill_records_by_shard(
    edges: &mut Vec<PendingEdge>,
    shard_points: usize,
) -> AnnResult<Vec<(usize, Vec<SpillRecord16>)>> {
    if edges.is_empty() {
        return Ok(Vec::new());
    }

    let shard_points = shard_points.max(1);
    edges.sort_unstable_by_key(|edge| edge.p);
    let mut groups: Vec<(usize, Vec<SpillRecord16>)> = Vec::new();
    for edge in edges.drain(..) {
        let shard_id = edge.p / shard_points;
        let record = SpillRecord16::from_pending_edge(&edge)?;
        if groups
            .last()
            .map(|(existing_shard, _)| *existing_shard != shard_id)
            .unwrap_or(true)
        {
            groups.push((shard_id, Vec::new()));
        }
        groups.last_mut().unwrap().1.push(record);
    }
    Ok(groups)
}

fn flush_all_buffers_to_parts(state: &mut SpillState) -> AnnResult<()> {
    let shard_ids: Vec<_> = state.buffers_by_shard.keys().copied().collect();
    for shard_id in shard_ids {
        flush_shard_buffer_to_part(state, shard_id)?;
    }
    Ok(())
}

fn flush_all_memory_buffers_to_segments(state: &mut InMemorySpillState) -> AnnResult<()> {
    let shard_ids: Vec<_> = state.buffers_by_shard.keys().copied().collect();
    for shard_id in shard_ids {
        flush_memory_shard_buffer_to_segment(state, shard_id)?;
    }
    Ok(())
}

fn flush_memory_shard_buffer_to_segment(
    state: &mut InMemorySpillState,
    shard_id: usize,
) -> AnnResult<()> {
    let Some(buffer) = state.buffers_by_shard.get_mut(&shard_id) else {
        return Ok(());
    };
    if buffer.is_empty() {
        return Ok(());
    }

    buffer.sort_unstable_by_key(|record| record.src);
    let records = std::mem::take(buffer);
    let src_min = records.first().map(|record| record.src).unwrap_or(0);
    let src_max = records.last().map(|record| record.src).unwrap_or(0);
    state.segments.push(InMemorySpillSegment {
        shard_id,
        records,
        src_min,
        src_max,
    });
    Ok(())
}

fn flush_shard_buffer_to_part(state: &mut SpillState, shard_id: usize) -> AnnResult<()> {
    let Some(buffer) = state.buffers_by_shard.get_mut(&shard_id) else {
        return Ok(());
    };
    if buffer.is_empty() {
        return Ok(());
    }

    buffer.sort_unstable_by_key(|edge| edge.p);
    let staged: Vec<PendingEdge> = std::mem::take(buffer);
    let src_min = staged.first().map(|edge| edge.p as u32).unwrap_or(0);
    let src_max = staged.last().map(|edge| edge.p as u32).unwrap_or(0);
    let byte_len = (staged.len() * RUN_RECORD_BYTES) as u64;
    ensure_part_writer(state, shard_id, byte_len)?;
    let part = state.part_writers_by_shard.get_mut(&shard_id).unwrap();
    let byte_offset = part.desc.byte_len;
    for edge in staged.iter() {
        part.writer.push(edge)?;
    }
    part.desc.byte_len += byte_len;
    state.segment_manifest.push(SpillSegmentDesc {
        shard_id: shard_id as u32,
        file_id: part.desc.file_id,
        byte_offset,
        byte_len,
        record_count: staged.len() as u32,
        src_min,
        src_max,
        sorted_by_src: true,
    });
    Ok(())
}

fn ensure_part_writer(
    state: &mut SpillState,
    shard_id: usize,
    pending_bytes: u64,
) -> AnnResult<()> {
    let need_rollover = state
        .part_writers_by_shard
        .get(&shard_id)
        .map(|part| part.desc.byte_len.saturating_add(pending_bytes) > state.part_max_bytes)
        .unwrap_or(true);
    if !need_rollover {
        return Ok(());
    }

    if let Some(part) = state.part_writers_by_shard.remove(&shard_id) {
        finalize_part_writer(state, part)?;
    }

    let file_id = {
        let next = state.next_part_id_by_shard.entry(shard_id).or_insert(0);
        let file_id = *next;
        *next = next.saturating_add(1);
        file_id
    };
    let path = part_path_for(&state.base_path, file_id as usize, shard_id);
    let writer = CandidateRunWriter::create_with_config(&path, state.io_cfg)?;
    state.part_writers_by_shard.insert(
        shard_id,
        SpillPartWriter {
            desc: SpillPartFileDesc {
                shard_id,
                file_id,
                path,
                byte_len: 0,
            },
            writer,
        },
    );
    Ok(())
}

fn finalize_part_writer(state: &mut SpillState, part: SpillPartWriter) -> AnnResult<()> {
    let SpillPartWriter { desc, writer } = part;
    writer.finish()?;
    state.part_files.push(desc);
    Ok(())
}

fn finish_part_writers(state: &mut SpillState) -> AnnResult<()> {
    let shard_ids: Vec<_> = state.part_writers_by_shard.keys().copied().collect();
    for shard_id in shard_ids {
        if let Some(part) = state.part_writers_by_shard.remove(&shard_id) {
            finalize_part_writer(state, part)?;
        }
    }
    Ok(())
}

fn part_path_for(base_path: &Path, part_index: usize, shard_id: usize) -> PathBuf {
    let parent = base_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = base_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("candidate");
    parent.join(format!("{stem}_shard{shard_id:05}_part{part_index:03}.log"))
}

fn write_segment_manifest(state: &SpillState) -> AnnResult<()> {
    let mut writer = BufWriter::new(File::create(&state.manifest_path)?);
    for segment in &state.segment_manifest {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{}",
            segment.shard_id,
            segment.file_id,
            segment.byte_offset,
            segment.byte_len,
            segment.record_count,
            segment.src_min,
            segment.src_max,
            u8::from(segment.sorted_by_src)
        )?;
    }
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Default)]
pub struct ExternalHashPruneReducer;

impl ExternalHashPruneReducer {
    pub fn reduce_spill_artifacts(
        num_points: usize,
        l_max: usize,
        artifacts: &SpillArtifacts,
    ) -> AnnResult<HashMap<u32, Vec<u32>>> {
        let mut reduced = HashMap::with_capacity(num_points);
        Self::reduce_spill_artifacts_sharded_in_order(
            num_points,
            l_max,
            num_points.max(1),
            artifacts,
            |point, neighbors| {
                reduced.insert(point, neighbors);
                Ok(())
            },
        )?;
        Ok(reduced)
    }

    pub fn reduce_spill_artifacts_ordered(
        num_points: usize,
        l_max: usize,
        artifacts: &SpillArtifacts,
    ) -> AnnResult<Vec<Vec<u32>>> {
        let mut ordered = vec![Vec::new(); num_points];
        Self::reduce_spill_artifacts_sharded_in_order(
            num_points,
            l_max,
            num_points.max(1),
            artifacts,
            |point, neighbors| {
                ordered[point as usize] = neighbors;
                Ok(())
            },
        )?;
        Ok(ordered)
    }

    pub fn reduce_spill_artifacts_sharded_in_order<F>(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<()>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_with_limit(
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            |point, neighbors| on_point_done(point, neighbors),
        )
        .map(|_| ())
    }

    pub fn reduce_spill_artifacts_sharded_in_order_from<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<()>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_from_with_limit(
            num_points,
            point_start,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            |point, neighbors| on_point_done(point, neighbors),
        )
        .map(|_| ())
    }

    pub fn file_point_counts(run_files: &[PathBuf]) -> AnnResult<BTreeMap<PathBuf, usize>> {
        let mut counts = BTreeMap::new();
        for path in run_files {
            let mut reader = BufReader::new(File::open(path)?);
            let mut count = 0usize;
            loop {
                let mut src_bytes = [0u8; 4];
                match reader.read_exact(&mut src_bytes) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(err) => return Err(err.into()),
                }

                let mut skip = [0u8; RUN_RECORD_BYTES - 4];
                reader.read_exact(&mut skip)?;
                count += 1;
            }
            counts.insert(path.clone(), count);
        }
        Ok(counts)
    }

    pub fn summarize_spill_artifacts(artifacts: &SpillArtifacts) -> SpillRunStats {
        summarize_spill_artifacts(artifacts)
    }

    pub fn reduce_spill_artifacts_sharded_in_order_profiled<F>(
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_with_limit(
            num_points,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            |point, neighbors| on_point_done(point, neighbors),
        )
    }

    pub fn reduce_spill_artifacts_sharded_in_order_profiled_from<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_from_with_limit(
            num_points,
            point_start,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            |point, neighbors| on_point_done(point, neighbors),
        )
    }

    pub fn reduce_spill_artifacts_sharded_in_order_profiled_from_parallel<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        points_per_shard: usize,
        artifacts: &SpillArtifacts,
        max_parallel_groups: usize,
        mut on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_segments_sharded_in_order_from_parallel(
            num_points,
            point_start,
            l_max,
            points_per_shard,
            artifacts,
            max_parallel_groups,
            |point, neighbors| on_point_done(point, neighbors),
        )
    }

    pub fn summarize_in_memory_spill_artifacts(
        artifacts: &InMemorySpillArtifacts,
    ) -> SpillRunStats {
        summarize_in_memory_spill_artifacts(artifacts)
    }

    pub fn reduce_in_memory_spill_sharded_in_order_profiled_from<F>(
        point_start: usize,
        num_points: usize,
        l_max: usize,
        artifacts: InMemorySpillArtifacts,
        mut on_point_done: F,
    ) -> AnnResult<SpillReduceStats>
    where
        F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
    {
        reduce_in_memory_segments_sharded_in_order_from(
            num_points,
            point_start,
            l_max,
            artifacts,
            |point, neighbors| on_point_done(point, neighbors),
        )
    }
}

fn reduce_segments_sharded_in_order_with_limit<F>(
    num_points: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    reduce_segments_sharded_in_order_from_with_limit(
        num_points,
        0,
        l_max,
        points_per_shard,
        artifacts,
        _max_open_files,
        on_point_done,
    )
}

fn reduce_segments_sharded_in_order_from_with_limit<F>(
    num_points: usize,
    point_start: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    _max_open_files: usize,
    mut on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    if num_points == 0 {
        return Ok(SpillReduceStats::default());
    }
    if point_start >= num_points {
        return Ok(SpillReduceStats::default());
    }

    let mut next_point = point_start;
    let groups = group_segments_by_shard(num_points, points_per_shard, artifacts)?;
    let stats = SpillReduceStats {
        shard_groups: groups.len(),
        segments_scanned: artifacts.segment_manifest.len(),
        part_files_opened: artifacts.part_files.len(),
        max_segments_in_group: groups
            .iter()
            .map(|group| group.segments.len())
            .max()
            .unwrap_or(0),
    };
    let mut result = Ok(());
    for group in &groups {
        if group.point_end <= point_start {
            continue;
        }
        let group_start = group.point_start.max(point_start);
        if next_point < group_start {
            if let Err(err) = emit_empty_point_range(next_point, group_start, &mut on_point_done) {
                result = Err(err);
                break;
            }
        }

        if let Err(err) = reduce_segment_group_in_range(
            group_start,
            group.point_end,
            l_max,
            points_per_shard
                .max(1)
                .min(group.point_end.saturating_sub(group_start).max(1)),
            group,
            artifacts,
            &mut on_point_done,
        ) {
            result = Err(err);
            break;
        }
        next_point = group.point_end;
    }

    if result.is_ok() && next_point < num_points {
        result = emit_empty_point_range(next_point, num_points, &mut on_point_done);
    }
    result.map(|_| stats)
}

fn reduce_segments_sharded_in_order_from_parallel<F>(
    num_points: usize,
    point_start: usize,
    l_max: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
    max_parallel_groups: usize,
    mut on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    if num_points == 0 || point_start >= num_points {
        return Ok(SpillReduceStats::default());
    }

    let groups = group_segments_by_shard(num_points, points_per_shard, artifacts)?;
    let stats = SpillReduceStats {
        shard_groups: groups.len(),
        segments_scanned: artifacts.segment_manifest.len(),
        part_files_opened: artifacts.part_files.len(),
        max_segments_in_group: groups
            .iter()
            .map(|group| group.segments.len())
            .max()
            .unwrap_or(0),
    };
    let active_groups: Vec<_> = groups
        .into_iter()
        .filter(|group| group.point_end > point_start)
        .collect();
    if active_groups.len() <= 1 || max_parallel_groups <= 1 {
        return reduce_segments_sharded_in_order_from_with_limit(
            num_points,
            point_start,
            l_max,
            points_per_shard,
            artifacts,
            usize::MAX,
            on_point_done,
        );
    }

    let parallelism = max_parallel_groups.max(1).min(active_groups.len());
    let temp_dir = artifacts
        .manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let mut reduced_groups = Vec::with_capacity(active_groups.len());
    for chunk in active_groups.chunks(parallelism) {
        let chunk_results: Vec<_> = chunk
            .par_iter()
            .map(|group| {
                let group_start = group.point_start.max(point_start);
                reduce_segment_group_to_neighbor_part(
                    group_start,
                    group.point_end,
                    l_max,
                    points_per_shard
                        .max(1)
                        .min(group.point_end.saturating_sub(group_start).max(1)),
                    group,
                    artifacts,
                    temp_dir,
                )
            })
            .collect();
        for result in chunk_results {
            reduced_groups.push(result?);
        }
    }
    reduced_groups.sort_unstable_by_key(|part| part.point_start);

    let mut result = Ok(());
    let mut next_point = point_start;
    for part in reduced_groups {
        if result.is_err() {
            let _ = fs::remove_file(&part.path);
            continue;
        }
        if next_point < part.point_start {
            result = emit_empty_point_range(next_point, part.point_start, &mut on_point_done);
            if result.is_err() {
                let _ = fs::remove_file(&part.path);
                continue;
            }
        }
        result = replay_neighbor_part(&part, &mut on_point_done);
        let _ = fs::remove_file(&part.path);
        if result.is_ok() {
            next_point = part.point_end;
        }
    }
    if result.is_ok() && next_point < num_points {
        result = emit_empty_point_range(next_point, num_points, &mut on_point_done);
    }
    result.map(|_| stats)
}

fn reduce_segment_group_in_range<F>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    points_per_chunk: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    if point_start >= point_end {
        return Ok(());
    }

    let points_per_chunk = points_per_chunk.max(1);
    let mut chunk_start = point_start;
    let mut chunk_end = (chunk_start + points_per_chunk).min(point_end);
    let mut reservoirs = allocate_reservoir_slab(chunk_end.saturating_sub(chunk_start), l_max);

    let mut file_lookup = BTreeMap::new();
    for part in &artifacts.part_files {
        file_lookup.insert((part.shard_id, part.file_id), part.path.clone());
    }

    for segment in &group.segments {
        let Some(path) = file_lookup.get(&(segment.shard_id as usize, segment.file_id)) else {
            return Err(AnnError::log_index_error(format!(
                "Missing spill part for shard={} file_id={}",
                segment.shard_id, segment.file_id
            )));
        };
        let mut reader = SegmentReader::open(path, *segment, DirectIoConfig::disabled())?;
        while let Some(edge) = reader.read_edge()? {
            while edge.p >= chunk_end {
                emit_shard(chunk_start, chunk_end, &mut reservoirs, on_point_done)?;
                chunk_start = chunk_end;
                chunk_end = (chunk_end + points_per_chunk).min(point_end);
                if chunk_start >= point_end {
                    return Err(AnnError::log_index_error(format!(
                        "Spill reducer encountered edge p={} outside shard range [{point_start}, {point_end})",
                        edge.p
                    )));
                }
                if chunk_start < point_end {
                    resize_or_reset_reservoirs(&mut reservoirs, chunk_end - chunk_start, l_max);
                }
            }

            if edge.p < chunk_start {
                if edge.p < point_start {
                    continue;
                }
                return Err(AnnError::log_index_error(format!(
                    "Spill reducer encountered out-of-order candidate edge p={} before current chunk start={chunk_start}",
                    edge.p
                )));
            }
            let local_idx = edge.p - chunk_start;
            reservoirs[local_idx].insert(edge.c, edge.hash, edge.dist);
        }
    }

    while chunk_start < point_end {
        emit_shard(chunk_start, chunk_end, &mut reservoirs, on_point_done)?;
        chunk_start = chunk_end;
        chunk_end = (chunk_end + points_per_chunk).min(point_end);

        if chunk_start < point_end {
            resize_or_reset_reservoirs(&mut reservoirs, chunk_end - chunk_start, l_max);
        }
    }

    Ok(())
}

#[derive(Debug)]
struct NeighborPart {
    point_start: usize,
    point_end: usize,
    path: PathBuf,
}

fn reduce_segment_group_to_neighbor_part(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    points_per_chunk: usize,
    group: &SegmentGroup,
    artifacts: &SpillArtifacts,
    temp_dir: &Path,
) -> AnnResult<NeighborPart> {
    if point_start >= point_end {
        return Err(AnnError::log_index_error(format!(
            "Cannot reduce empty spill group range [{point_start}, {point_end})"
        )));
    }

    let path = temp_dir.join(format!(
        ".candidate_reduce_{pid}_{start}_{end}.part",
        pid = std::process::id(),
        start = point_start,
        end = point_end
    ));
    let mut writer = BufWriter::with_capacity(REDUCE_NEIGHBOR_BUFFER_BYTES, File::create(&path)?);
    let points_per_chunk = points_per_chunk.max(1);
    let mut chunk_start = point_start;
    let mut chunk_end = (chunk_start + points_per_chunk).min(point_end);
    let mut reservoirs = allocate_reservoir_slab(chunk_end.saturating_sub(chunk_start), l_max);

    let mut file_lookup = BTreeMap::new();
    for part in &artifacts.part_files {
        file_lookup.insert((part.shard_id, part.file_id), part.path.clone());
    }

    for segment in &group.segments {
        let Some(path) = file_lookup.get(&(segment.shard_id as usize, segment.file_id)) else {
            return Err(AnnError::log_index_error(format!(
                "Missing spill part for shard={} file_id={}",
                segment.shard_id, segment.file_id
            )));
        };
        let mut reader = SegmentReader::open(path, *segment, DirectIoConfig::disabled())?;
        while let Some(edge) = reader.read_edge()? {
            if edge.p < point_start {
                continue;
            }
            if edge.p >= point_end {
                break;
            }
            while edge.p >= chunk_end {
                write_neighbor_chunk(chunk_start, chunk_end, &mut reservoirs, &mut writer)?;
                chunk_start = chunk_end;
                chunk_end = (chunk_end + points_per_chunk).min(point_end);
                if chunk_start >= point_end {
                    return Err(AnnError::log_index_error(format!(
                        "Spill reducer encountered edge p={} outside shard range [{point_start}, {point_end})",
                        edge.p
                    )));
                }
                resize_or_reset_reservoirs(&mut reservoirs, chunk_end - chunk_start, l_max);
            }

            if edge.p < chunk_start {
                return Err(AnnError::log_index_error(format!(
                    "Spill reducer encountered out-of-order candidate edge p={} before current chunk start={chunk_start}",
                    edge.p
                )));
            }
            let local_idx = edge.p - chunk_start;
            reservoirs[local_idx].insert(edge.c, edge.hash, edge.dist);
        }
    }

    while chunk_start < point_end {
        write_neighbor_chunk(chunk_start, chunk_end, &mut reservoirs, &mut writer)?;
        chunk_start = chunk_end;
        chunk_end = (chunk_end + points_per_chunk).min(point_end);
        if chunk_start < point_end {
            resize_or_reset_reservoirs(&mut reservoirs, chunk_end - chunk_start, l_max);
        }
    }
    writer.flush()?;

    Ok(NeighborPart {
        point_start,
        point_end,
        path,
    })
}

fn reduce_in_memory_segments_sharded_in_order_from<F>(
    num_points: usize,
    point_start: usize,
    l_max: usize,
    mut artifacts: InMemorySpillArtifacts,
    mut on_point_done: F,
) -> AnnResult<SpillReduceStats>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    if num_points == 0 || point_start >= num_points {
        return Ok(SpillReduceStats::default());
    }

    artifacts
        .segments
        .sort_unstable_by_key(|segment| (segment.shard_id, segment.src_min));
    let stats = SpillReduceStats {
        shard_groups: artifacts
            .segments
            .iter()
            .map(|segment| segment.shard_id)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        segments_scanned: artifacts.segments.len(),
        part_files_opened: 0,
        max_segments_in_group: max_in_memory_segments_per_shard(&artifacts),
    };

    let mut next_point = point_start;
    let mut i = 0usize;
    while i < artifacts.segments.len() {
        let shard_id = artifacts.segments[i].shard_id;
        let shard_start = shard_id.saturating_mul(artifacts.shard_points);
        if shard_start >= num_points {
            return Err(AnnError::log_index_error(format!(
                "In-memory spill reducer encountered shard_id={shard_id} outside num_points={num_points}"
            )));
        }
        let shard_end = (shard_start + artifacts.shard_points).min(num_points);
        if shard_end <= point_start {
            while i < artifacts.segments.len() && artifacts.segments[i].shard_id == shard_id {
                i += 1;
            }
            continue;
        }

        let group_start = shard_start.max(point_start);
        if next_point < group_start {
            emit_empty_point_range(next_point, group_start, &mut on_point_done)?;
        }

        let group_first = i;
        while i < artifacts.segments.len() && artifacts.segments[i].shard_id == shard_id {
            let src_min = artifacts.segments[i].src_min as usize;
            let src_max = artifacts.segments[i].src_max as usize;
            if src_min < shard_start || src_max >= shard_end {
                return Err(AnnError::log_index_error(format!(
                    "In-memory spill segment src_range=[{}, {}] outside shard range [{}, {}) for shard_id={shard_id}",
                    artifacts.segments[i].src_min,
                    artifacts.segments[i].src_max,
                    shard_start,
                    shard_end
                )));
            }
            i += 1;
        }

        reduce_in_memory_segment_group_in_range(
            group_start,
            shard_end,
            l_max,
            artifacts.shard_points,
            &mut artifacts.segments[group_first..i],
            &mut on_point_done,
        )?;
        next_point = shard_end;
    }

    if next_point < num_points {
        emit_empty_point_range(next_point, num_points, &mut on_point_done)?;
    }
    Ok(stats)
}

fn reduce_in_memory_segment_group_in_range<F>(
    point_start: usize,
    point_end: usize,
    l_max: usize,
    points_per_shard: usize,
    segments: &mut [InMemorySpillSegment],
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    if point_start >= point_end {
        return Ok(());
    }

    for segment in segments.iter_mut() {
        segment.records.sort_unstable_by_key(|record| record.src);
    }
    segments.sort_unstable_by_key(|segment| (segment.src_min, segment.src_max));

    let points_per_chunk = points_per_shard
        .max(1)
        .min(point_end.saturating_sub(point_start).max(1));
    let mut chunk_start = point_start;
    let mut chunk_end = (chunk_start + points_per_chunk).min(point_end);
    let mut reservoirs = allocate_reservoir_slab(chunk_end.saturating_sub(chunk_start), l_max);

    for segment in segments.iter() {
        for record in &segment.records {
            let edge = record.to_pending_edge();
            if edge.p < point_start {
                continue;
            }
            if edge.p >= point_end {
                break;
            }
            while edge.p >= chunk_end {
                emit_shard(chunk_start, chunk_end, &mut reservoirs, on_point_done)?;
                chunk_start = chunk_end;
                chunk_end = (chunk_end + points_per_chunk).min(point_end);
                if chunk_start >= point_end {
                    return Err(AnnError::log_index_error(format!(
                        "In-memory spill reducer encountered edge p={} outside shard range [{point_start}, {point_end})",
                        edge.p
                    )));
                }
                resize_or_reset_reservoirs(&mut reservoirs, chunk_end - chunk_start, l_max);
            }

            if edge.p < chunk_start {
                return Err(AnnError::log_index_error(format!(
                    "In-memory spill reducer encountered out-of-order candidate edge p={} before current chunk start={chunk_start}",
                    edge.p
                )));
            }
            let local_idx = edge.p - chunk_start;
            reservoirs[local_idx].insert(edge.c, edge.hash, edge.dist);
        }
    }

    while chunk_start < point_end {
        emit_shard(chunk_start, chunk_end, &mut reservoirs, on_point_done)?;
        chunk_start = chunk_end;
        chunk_end = (chunk_end + points_per_chunk).min(point_end);

        if chunk_start < point_end {
            resize_or_reset_reservoirs(&mut reservoirs, chunk_end - chunk_start, l_max);
        }
    }

    Ok(())
}

fn max_in_memory_segments_per_shard(artifacts: &InMemorySpillArtifacts) -> usize {
    let mut counts = BTreeMap::new();
    for segment in &artifacts.segments {
        *counts.entry(segment.shard_id).or_insert(0usize) += 1;
    }
    counts.values().copied().max().unwrap_or(0)
}

#[derive(Debug, Clone)]
struct SegmentGroup {
    point_start: usize,
    point_end: usize,
    segments: Vec<SpillSegmentDesc>,
}

fn group_segments_by_shard(
    num_points: usize,
    points_per_shard: usize,
    artifacts: &SpillArtifacts,
) -> AnnResult<Vec<SegmentGroup>> {
    if artifacts.segment_manifest.is_empty() {
        return Ok(Vec::new());
    }

    let mut grouped: BTreeMap<usize, Vec<SpillSegmentDesc>> = BTreeMap::new();
    for segment in &artifacts.segment_manifest {
        grouped
            .entry(segment.shard_id as usize)
            .or_default()
            .push(*segment);
    }

    let points_per_shard = points_per_shard.max(1);
    let mut groups = Vec::with_capacity(grouped.len());
    for (shard_id, mut segments) in grouped {
        let point_start = shard_id.saturating_mul(points_per_shard);
        if point_start >= num_points {
            return Err(AnnError::log_index_error(format!(
                "Spill reducer encountered shard_id={shard_id} outside num_points={num_points} with points_per_shard={points_per_shard}"
            )));
        }
        let point_end = (point_start + points_per_shard).min(num_points);
        for segment in &segments {
            let src_min = segment.src_min as usize;
            let src_max = segment.src_max as usize;
            if src_min < point_start || src_max >= point_end {
                return Err(AnnError::log_index_error(format!(
                    "Spill reducer encountered segment metadata src_range=[{}, {}] outside shard range [{}, {}) for shard_id={shard_id}",
                    segment.src_min, segment.src_max, point_start, point_end
                )));
            }
        }
        segments.sort_unstable_by_key(|segment| {
            (segment.src_min, segment.file_id, segment.byte_offset)
        });
        groups.push(SegmentGroup {
            point_start,
            point_end,
            segments,
        });
    }

    Ok(groups)
}

fn summarize_spill_artifacts(artifacts: &SpillArtifacts) -> SpillRunStats {
    let mut parts_per_shard: BTreeMap<usize, usize> = BTreeMap::new();
    let mut segments_per_shard: BTreeMap<usize, usize> = BTreeMap::new();
    for part in &artifacts.part_files {
        *parts_per_shard.entry(part.shard_id).or_insert(0) += 1;
    }
    for segment in &artifacts.segment_manifest {
        *segments_per_shard
            .entry(segment.shard_id as usize)
            .or_insert(0) += 1;
    }

    SpillRunStats {
        part_files: artifacts.part_files.len(),
        segment_count: artifacts.segment_manifest.len(),
        shard_groups: parts_per_shard
            .keys()
            .chain(segments_per_shard.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        max_parts_per_shard: parts_per_shard.values().copied().max().unwrap_or(0),
        max_segments_per_shard: segments_per_shard.values().copied().max().unwrap_or(0),
        record_bytes: RUN_RECORD_BYTES,
        manifest_bytes: fs::metadata(&artifacts.manifest_path)
            .map(|meta| meta.len())
            .unwrap_or(0),
    }
}

fn summarize_in_memory_spill_artifacts(artifacts: &InMemorySpillArtifacts) -> SpillRunStats {
    let mut segments_per_shard: BTreeMap<usize, usize> = BTreeMap::new();
    for segment in &artifacts.segments {
        *segments_per_shard.entry(segment.shard_id).or_insert(0) += 1;
    }
    let record_count: usize = artifacts
        .segments
        .iter()
        .map(|segment| segment.records.len())
        .sum();

    SpillRunStats {
        part_files: 0,
        segment_count: artifacts.segments.len(),
        shard_groups: segments_per_shard.len(),
        max_parts_per_shard: 0,
        max_segments_per_shard: segments_per_shard.values().copied().max().unwrap_or(0),
        record_bytes: RUN_RECORD_BYTES,
        manifest_bytes: record_count.saturating_mul(RUN_RECORD_BYTES) as u64,
    }
}

fn effective_spill_cache_bytes(spill_cache_bytes: usize) -> usize {
    if spill_cache_bytes == 0 {
        DEFAULT_SPILL_CACHE_BYTES
    } else {
        spill_cache_bytes
    }
}

fn allocate_reservoir_slab(points: usize, l_max: usize) -> Vec<HashPruneReservoir> {
    (0..points)
        .map(|_| HashPruneReservoir::new(l_max))
        .collect()
}

fn resize_or_reset_reservoirs(
    reservoirs: &mut Vec<HashPruneReservoir>,
    points: usize,
    l_max: usize,
) {
    if reservoirs.len() < points {
        reservoirs.resize_with(points, || HashPruneReservoir::new(l_max));
    }

    for reservoir in reservoirs.iter_mut().take(points) {
        reservoir.clear();
    }
}

fn emit_shard<F>(
    shard_start: usize,
    shard_end: usize,
    reservoirs: &mut [HashPruneReservoir],
    on_point_done: &mut F,
) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    for point in shard_start..shard_end {
        let local_idx = point - shard_start;
        let mut neighbors = reservoirs[local_idx].drain_neighbors();
        neighbors.shrink_to_fit();
        on_point_done(point as u32, neighbors)?;
    }
    Ok(())
}

fn write_neighbor_chunk<W: Write>(
    shard_start: usize,
    shard_end: usize,
    reservoirs: &mut [HashPruneReservoir],
    writer: &mut W,
) -> AnnResult<()> {
    for point in shard_start..shard_end {
        let local_idx = point - shard_start;
        let neighbors = reservoirs[local_idx].drain_neighbors();
        writer.write_all(&(neighbors.len() as u32).to_le_bytes())?;
        for neighbor in neighbors {
            writer.write_all(&neighbor.to_le_bytes())?;
        }
    }
    Ok(())
}

fn replay_neighbor_part<F>(part: &NeighborPart, on_point_done: &mut F) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    let mut reader =
        BufReader::with_capacity(REDUCE_NEIGHBOR_BUFFER_BYTES, File::open(&part.path)?);
    for point in part.point_start..part.point_end {
        let mut degree_bytes = [0u8; 4];
        reader.read_exact(&mut degree_bytes)?;
        let degree = u32::from_le_bytes(degree_bytes) as usize;
        let mut neighbors = Vec::with_capacity(degree);
        for _ in 0..degree {
            let mut neighbor_bytes = [0u8; 4];
            reader.read_exact(&mut neighbor_bytes)?;
            neighbors.push(u32::from_le_bytes(neighbor_bytes));
        }
        on_point_done(point as u32, neighbors)?;
    }
    Ok(())
}

fn emit_empty_point_range<F>(start: usize, end: usize, on_point_done: &mut F) -> AnnResult<()>
where
    F: FnMut(u32, Vec<u32>) -> AnnResult<()>,
{
    for point in start..end {
        on_point_done(point as u32, Vec::new())?;
    }
    Ok(())
}

#[derive(Debug)]
enum SegmentReaderInner {
    Buffered {
        reader: BufReader<File>,
        remaining: u64,
    },
    Direct {
        file: DirectIoFile,
        offset: u64,
        len: u64,
    },
}

#[derive(Debug)]
struct SegmentReader {
    inner: SegmentReaderInner,
}

impl SegmentReader {
    fn open(path: &Path, segment: SpillSegmentDesc, io_cfg: DirectIoConfig) -> AnnResult<Self> {
        let inner = if io_cfg.enabled {
            SegmentReaderInner::Direct {
                file: DirectIoFile::open_read(path, io_cfg)?,
                offset: segment.byte_offset,
                len: segment.byte_offset + segment.byte_len,
            }
        } else {
            let mut reader =
                BufReader::with_capacity(DIRECT_CANDIDATE_BUFFER_BYTES, File::open(path)?);
            reader.seek_relative(segment.byte_offset as i64)?;
            SegmentReaderInner::Buffered {
                reader,
                remaining: segment.byte_len,
            }
        };
        Ok(Self { inner })
    }

    fn read_edge(&mut self) -> AnnResult<Option<PendingEdge>> {
        let mut bytes = [0u8; RUN_RECORD_BYTES];
        match &mut self.inner {
            SegmentReaderInner::Buffered { reader, remaining } => {
                if *remaining == 0 {
                    return Ok(None);
                }
                if *remaining < RUN_RECORD_BYTES as u64 {
                    return Err(AnnError::log_index_error(format!(
                        "Candidate segment has trailing partial record with {} bytes remaining",
                        remaining
                    )));
                }
                match reader.read_exact(&mut bytes) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                    Err(err) => return Err(err.into()),
                }
                *remaining -= RUN_RECORD_BYTES as u64;
            }
            SegmentReaderInner::Direct { file, offset, len } => {
                if *offset >= *len {
                    return Ok(None);
                }
                if offset.saturating_add(RUN_RECORD_BYTES as u64) > *len {
                    return Err(AnnError::log_index_error(format!(
                        "Candidate run has trailing partial record at offset {} of length {}",
                        offset, len
                    )));
                }
                file.read_exact_at(&mut bytes, *offset)?;
                *offset += RUN_RECORD_BYTES as u64;
            }
        }

        Ok(Some(
            SpillRecord16 {
                src: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                dst: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                hash: u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
                flags: u16::from_le_bytes(bytes[10..12].try_into().unwrap()),
                dist: f32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            }
            .to_pending_edge(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::{
        CandidateRunWriter, ExternalHashPruneReducer, InMemorySpillEdgeSink, RUN_RECORD_BYTES,
        SPILL_FLAG_MANDATORY, SpillArtifacts, SpillEdgeSink, SpillPartFileDesc, SpillRecord16,
        SpillSegmentDesc, encode_spill_records_by_shard,
    };
    use crate::forgeann::direct_io::DirectIoConfig;
    use crate::forgeann::hash_prune::HashPruneReservoir;
    use crate::forgeann::leaf_build::{PENDING_EDGE_DIRECT, PendingEdge, PendingEdgeSink};

    fn reduce_in_memory(num_points: usize, l_max: usize, edges: &[PendingEdge]) -> Vec<Vec<u32>> {
        let mut reservoirs: Vec<_> = (0..num_points)
            .map(|_| HashPruneReservoir::new(l_max))
            .collect();
        for edge in edges {
            reservoirs[edge.p].insert(edge.c, edge.hash, edge.dist);
        }
        reservoirs
            .into_iter()
            .map(|reservoir| {
                let mut neighbors = reservoir.into_neighbors();
                neighbors.sort_unstable();
                neighbors
            })
            .collect()
    }

    fn build_artifacts_from_paths(
        paths: &[PathBuf],
        shard_ids: &[usize],
        point_ranges: &[(u32, u32)],
    ) -> SpillArtifacts {
        let part_files = paths
            .iter()
            .enumerate()
            .map(|(idx, path)| SpillPartFileDesc {
                shard_id: shard_ids[idx],
                file_id: idx as u16,
                path: path.clone(),
                byte_len: fs::metadata(path).unwrap().len(),
            })
            .collect();
        let segment_manifest = paths
            .iter()
            .enumerate()
            .map(|(idx, path)| SpillSegmentDesc {
                shard_id: shard_ids[idx] as u32,
                file_id: idx as u16,
                byte_offset: 0,
                byte_len: fs::metadata(path).unwrap().len(),
                record_count: (fs::metadata(path).unwrap().len() as usize / RUN_RECORD_BYTES)
                    as u32,
                src_min: point_ranges[idx].0,
                src_max: point_ranges[idx].1,
                sorted_by_src: true,
            })
            .collect();
        SpillArtifacts {
            part_files,
            segment_manifest,
            manifest_path: paths
                .first()
                .and_then(|path| path.parent())
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("candidate_segments.manifest"),
        }
    }

    #[test]
    fn spill_record16_is_fixed_width_and_preserves_mandatory_flag() {
        assert_eq!(RUN_RECORD_BYTES, 16);
        assert_eq!(std::mem::size_of::<SpillRecord16>(), 16);

        let edge = PendingEdge {
            p: 7,
            c: 42,
            hash: 9,
            dist: 0.125,
            mandatory: true,
            local_rank: 1,
            flags: PENDING_EDGE_DIRECT,
        };
        let record = SpillRecord16::from_pending_edge(&edge).unwrap();
        let round_trip = record.to_pending_edge();

        assert_eq!(round_trip.p, edge.p);
        assert_eq!(round_trip.c, edge.c);
        assert_eq!(round_trip.hash, edge.hash);
        assert_eq!(round_trip.dist, edge.dist);
        assert!(round_trip.mandatory);
    }

    #[test]
    fn candidate_run_writer_uses_fixed_16_byte_records() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate.run");
        let edges = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 1,
                dist: 0.1,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 2,
                c: 3,
                hash: 2,
                dist: 0.2,
                mandatory: true,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 4,
                c: 5,
                hash: 3,
                dist: 0.3,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];

        let mut writer = CandidateRunWriter::create(&run_path).unwrap();
        for edge in &edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let metadata = fs::metadata(&run_path).unwrap();
        assert_eq!(metadata.len(), (edges.len() * RUN_RECORD_BYTES) as u64);
    }

    #[test]
    fn candidate_run_writer_strict_io_round_trips_fixed_records() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate_strict.run");
        let edges = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 1,
                dist: 0.1,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 2,
                c: 3,
                hash: 2,
                dist: 0.2,
                mandatory: true,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 4,
                c: 5,
                hash: 3,
                dist: 0.3,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];

        let mut writer = CandidateRunWriter::create_with_config(
            &run_path,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        for edge in &edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let metadata = fs::metadata(&run_path).unwrap();
        assert_eq!(metadata.len(), (edges.len() * RUN_RECORD_BYTES) as u64);

        let segment = SpillSegmentDesc {
            shard_id: 0,
            file_id: 0,
            byte_offset: 0,
            byte_len: metadata.len(),
            record_count: edges.len() as u32,
            src_min: 0,
            src_max: 4,
            sorted_by_src: true,
        };
        let mut reader = super::SegmentReader::open(
            &run_path,
            segment,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        let mut round_trip = Vec::new();
        while let Some(edge) = reader.read_edge().unwrap() {
            round_trip.push(edge);
        }
        let round_trip_fields: Vec<_> = round_trip
            .iter()
            .map(|edge| (edge.p, edge.c, edge.hash, edge.dist, edge.mandatory))
            .collect();
        let edge_fields: Vec<_> = edges
            .iter()
            .map(|edge| (edge.p, edge.c, edge.hash, edge.dist, edge.mandatory))
            .collect();
        assert_eq!(round_trip_fields, edge_fields);
    }

    #[test]
    fn candidate_run_writer_strict_io_batches_records_before_direct_writes() {
        use crate::forgeann::direct_io::{
            direct_io_test_write_calls, reset_direct_io_test_counters,
        };

        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate_strict_batched.run");
        reset_direct_io_test_counters();

        let mut writer = CandidateRunWriter::create_with_config(
            &run_path,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        for i in 0..1000_u32 {
            writer
                .push(&PendingEdge {
                    p: i as usize,
                    c: i + 1,
                    hash: (i % 4096) as u16,
                    dist: i as f32,
                    mandatory: false,
                    local_rank: 1,
                    flags: PENDING_EDGE_DIRECT,
                })
                .unwrap();
        }
        writer.finish().unwrap();

        assert_eq!(
            fs::metadata(&run_path).unwrap().len(),
            (1000 * RUN_RECORD_BYTES) as u64
        );
        assert!(
            direct_io_test_write_calls() < 100,
            "strict candidate writer should batch records before direct I/O, saw {} writes",
            direct_io_test_write_calls()
        );
    }

    #[test]
    fn spill_reduce_matches_in_memory_hashprune_for_same_candidates() {
        let edges = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 1,
                dist: 1.0,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 0,
                c: 2,
                hash: 1,
                dist: 0.5,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 0,
                c: 3,
                hash: 7,
                dist: 0.8,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 1,
                c: 0,
                hash: 4,
                dist: 1.0,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 1,
                c: 2,
                hash: 9,
                dist: 0.7,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 2,
                c: 0,
                hash: 3,
                dist: 0.6,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 2,
                c: 1,
                hash: 3,
                dist: 0.9,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];

        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidates.run");
        let mut writer = CandidateRunWriter::create(&run_path).unwrap();
        for edge in &edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let artifacts = build_artifacts_from_paths(&[run_path], &[0], &[(0, 2)]);
        let reduced = ExternalHashPruneReducer::reduce_spill_artifacts(3, 4, &artifacts).unwrap();
        let expected = reduce_in_memory(3, 4, &edges);

        let reduced_sorted: Vec<Vec<u32>> = (0..3)
            .map(|point| {
                let mut neighbors = reduced.get(&(point as u32)).cloned().unwrap_or_default();
                neighbors.sort_unstable();
                neighbors
            })
            .collect();

        assert_eq!(reduced_sorted, expected);
    }

    #[test]
    fn spill_reduce_is_history_independent_across_multiple_run_files() {
        let first = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 2,
                dist: 0.9,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 0,
                c: 2,
                hash: 5,
                dist: 1.2,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 1,
                c: 0,
                hash: 3,
                dist: 0.4,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];
        let second = vec![
            PendingEdge {
                p: 0,
                c: 3,
                hash: 2,
                dist: 0.3,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 1,
                c: 2,
                hash: 8,
                dist: 0.6,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 1,
                c: 3,
                hash: 8,
                dist: 0.2,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];

        let dir = tempdir().unwrap();
        let path_a = dir.path().join("a.run");
        let path_b = dir.path().join("b.run");

        let mut writer_a = CandidateRunWriter::create(&path_a).unwrap();
        for edge in &first {
            writer_a.push(edge).unwrap();
        }
        writer_a.finish().unwrap();

        let mut writer_b = CandidateRunWriter::create(&path_b).unwrap();
        for edge in &second {
            writer_b.push(edge).unwrap();
        }
        writer_b.finish().unwrap();

        let mut combined = Vec::new();
        combined.extend_from_slice(&first);
        combined.extend_from_slice(&second);
        let expected = reduce_in_memory(2, 4, &combined);

        let artifacts = build_artifacts_from_paths(&[path_b, path_a], &[0, 0], &[(0, 1), (0, 1)]);
        let reduced = ExternalHashPruneReducer::reduce_spill_artifacts(2, 4, &artifacts).unwrap();
        let reduced_sorted: HashMap<u32, Vec<u32>> = reduced
            .into_iter()
            .map(|(point, mut neighbors)| {
                neighbors.sort_unstable();
                (point, neighbors)
            })
            .collect();

        assert_eq!(
            reduced_sorted.get(&0).cloned().unwrap_or_default(),
            expected[0]
        );
        assert_eq!(
            reduced_sorted.get(&1).cloned().unwrap_or_default(),
            expected[1]
        );
    }

    #[test]
    fn spill_reduce_streams_shards_in_point_order() {
        let shard_a = vec![
            PendingEdge {
                p: 0,
                c: 3,
                hash: 2,
                dist: 0.9,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 1,
                c: 4,
                hash: 4,
                dist: 0.7,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];
        let shard_b = vec![
            PendingEdge {
                p: 2,
                c: 6,
                hash: 7,
                dist: 0.1,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 3,
                c: 0,
                hash: 1,
                dist: 0.6,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];
        let shard_c = vec![
            PendingEdge {
                p: 4,
                c: 1,
                hash: 9,
                dist: 0.5,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 5,
                c: 2,
                hash: 10,
                dist: 0.4,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];

        let dir = tempdir().unwrap();
        let path_a = dir.path().join("a.run");
        let path_b = dir.path().join("b.run");
        let path_c = dir.path().join("c.run");

        let mut writer_a = CandidateRunWriter::create(&path_a).unwrap();
        for edge in &shard_a {
            writer_a.push(edge).unwrap();
        }
        writer_a.finish().unwrap();

        let mut writer_b = CandidateRunWriter::create(&path_b).unwrap();
        for edge in &shard_b {
            writer_b.push(edge).unwrap();
        }
        writer_b.finish().unwrap();

        let mut writer_c = CandidateRunWriter::create(&path_c).unwrap();
        for edge in &shard_c {
            writer_c.push(edge).unwrap();
        }
        writer_c.finish().unwrap();

        let mut expected_edges = Vec::new();
        expected_edges.extend_from_slice(&shard_a);
        expected_edges.extend_from_slice(&shard_b);
        expected_edges.extend_from_slice(&shard_c);
        let expected = reduce_in_memory(6, 4, &expected_edges);

        let mut streamed = Vec::new();
        let artifacts = build_artifacts_from_paths(
            &[path_b, path_c, path_a],
            &[1, 2, 0],
            &[(2, 3), (4, 5), (0, 1)],
        );
        ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order(
            6,
            4,
            2,
            &artifacts,
            |point, mut neighbors| {
                neighbors.sort_unstable();
                streamed.push((point, neighbors));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(streamed.len(), 6);
        assert_eq!(
            streamed.iter().map(|(point, _)| *point).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        for (point, neighbors) in streamed {
            assert_eq!(neighbors, expected[point as usize]);
        }
    }

    #[test]
    fn spill_parallel_reduce_matches_serial_order_and_cleans_temp_parts() {
        let shard_b = vec![
            PendingEdge {
                p: 2,
                c: 20,
                hash: 2,
                dist: 0.4,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 3,
                c: 30,
                hash: 3,
                dist: 0.3,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 3,
                c: 31,
                hash: 4,
                dist: 0.2,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];
        let shard_d = vec![
            PendingEdge {
                p: 6,
                c: 60,
                hash: 6,
                dist: 0.1,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 7,
                c: 70,
                hash: 7,
                dist: 0.5,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];

        let dir = tempdir().unwrap();
        let path_b = dir.path().join("b.run");
        let path_d = dir.path().join("d.run");
        for (path, edges) in [(&path_b, &shard_b), (&path_d, &shard_d)] {
            let mut writer = CandidateRunWriter::create(path).unwrap();
            for edge in edges {
                writer.push(edge).unwrap();
            }
            writer.finish().unwrap();
        }

        let mut all_edges = Vec::new();
        all_edges.extend_from_slice(&shard_b);
        all_edges.extend_from_slice(&shard_d);
        let expected = reduce_in_memory(8, 4, &all_edges);
        let artifacts = build_artifacts_from_paths(&[path_d, path_b], &[3, 1], &[(6, 7), (2, 3)]);

        let mut streamed = Vec::new();
        let stats =
            ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_profiled_from_parallel(
                2,
                8,
                4,
                2,
                &artifacts,
                4,
                |point, mut neighbors| {
                    neighbors.sort_unstable();
                    streamed.push((point, neighbors));
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(stats.shard_groups, 2);
        assert_eq!(
            streamed.iter().map(|(point, _)| *point).collect::<Vec<_>>(),
            vec![2, 3, 4, 5, 6, 7]
        );
        for (point, neighbors) in streamed {
            assert_eq!(neighbors, expected[point as usize]);
        }
        let leftover_reduce_parts = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .contains(".candidate_reduce_")
            })
            .count();
        assert_eq!(leftover_reduce_parts, 0);
    }

    #[test]
    fn spill_reduce_can_skip_pre_emitted_resident_prefix() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tail.run");
        let tail_edges = vec![
            PendingEdge {
                p: 4,
                c: 40,
                hash: 1,
                dist: 0.2,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 5,
                c: 50,
                hash: 2,
                dist: 0.3,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];
        let mut writer = CandidateRunWriter::create(&path).unwrap();
        for edge in &tail_edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let artifacts = build_artifacts_from_paths(&[path], &[2], &[(4, 5)]);
        let resident_prefix = 4usize;
        let mut emitted_points = vec![0u32, 1, 2, 3];
        ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order_from(
            resident_prefix,
            6,
            4,
            2,
            &artifacts,
            |point, _neighbors| {
                emitted_points.push(point);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            emitted_points,
            vec![0, 1, 2, 3, 4, 5],
            "range-limited reducer should not re-emit already-produced resident prefix points below {resident_prefix}"
        );
    }

    #[test]
    fn spill_reduce_rejects_segment_metadata_that_crosses_shard_ranges() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.run");
        let edges = vec![
            PendingEdge {
                p: 2,
                c: 10,
                hash: 1,
                dist: 0.2,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 3,
                c: 11,
                hash: 2,
                dist: 0.1,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];
        let mut writer = CandidateRunWriter::create(&path).unwrap();
        for edge in &edges {
            writer.push(edge).unwrap();
        }
        writer.finish().unwrap();

        let artifacts = build_artifacts_from_paths(&[path], &[0], &[(2, 3)]);
        let err = ExternalHashPruneReducer::reduce_spill_artifacts_sharded_in_order(
            4,
            4,
            2,
            &artifacts,
            |_, _| Ok(()),
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("segment metadata"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn zero_spill_budget_does_not_flush_every_pending_edge_batch() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate.run");
        let sink = SpillEdgeSink::create(&run_path, 0).unwrap();

        for idx in 0..10 {
            let mut batch = vec![PendingEdge {
                p: idx,
                c: (idx + 1) as u32,
                hash: idx as u16,
                dist: idx as f32 + 0.5,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            }];
            sink.flush_pending_edges(&mut batch).unwrap();
            assert!(batch.is_empty());
        }

        let artifacts = sink.finish().unwrap();
        assert_eq!(
            artifacts.part_files.len(),
            1,
            "zero spill budget should use a batched default cache"
        );

        let part_paths: Vec<_> = artifacts
            .part_files
            .iter()
            .map(|part| part.path.clone())
            .collect();
        let counts = ExternalHashPruneReducer::file_point_counts(&part_paths).unwrap();
        assert_eq!(counts.values().copied().sum::<usize>(), 10);
    }

    #[test]
    fn sharded_spill_sink_writes_part_files_and_manifest_by_source_shard() {
        let dir = tempdir().unwrap();
        let run_path = dir.path().join("candidate.run");
        let sink = SpillEdgeSink::create_sharded(&run_path, 1024, 4).unwrap();

        let mut batch = vec![
            PendingEdge {
                p: 0,
                c: 1,
                hash: 0,
                dist: 0.1,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 5,
                c: 2,
                hash: 1,
                dist: 0.2,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 9,
                c: 3,
                hash: 2,
                dist: 0.3,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];
        sink.flush_pending_edges(&mut batch).unwrap();
        let artifacts = sink.finish().unwrap();

        assert!(!artifacts.part_files.is_empty());
        assert!(!artifacts.segment_manifest.is_empty());
        let names: Vec<_> = artifacts
            .part_files
            .iter()
            .map(|part| {
                part.path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert!(names.iter().any(|name| name.contains("shard00000_part000")));
        assert!(names.iter().any(|name| name.contains("shard00001_part000")));
        assert!(names.iter().any(|name| name.contains("shard00002_part000")));
        assert!(
            artifacts
                .manifest_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("candidate_segments.manifest")
        );
    }

    #[test]
    fn in_memory_spill_sink_reduces_without_candidate_files() {
        let sink = InMemorySpillEdgeSink::create_sharded(1, 4);
        let mut batch = vec![
            PendingEdge {
                p: 5,
                c: 20,
                hash: 1,
                dist: 0.2,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 1,
                c: 10,
                hash: 2,
                dist: 0.1,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 5,
                c: 21,
                hash: 3,
                dist: 0.3,
                mandatory: true,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 9,
                c: 30,
                hash: 4,
                dist: 0.4,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];
        let expected = reduce_in_memory(12, 4, &batch);

        sink.flush_pending_edges(&mut batch).unwrap();
        assert!(batch.is_empty());
        let artifacts = sink.finish().unwrap();
        let spill_stats = ExternalHashPruneReducer::summarize_in_memory_spill_artifacts(&artifacts);
        assert_eq!(spill_stats.part_files, 0);
        assert_eq!(spill_stats.record_bytes, RUN_RECORD_BYTES);
        assert!(spill_stats.segment_count >= 3);

        let mut actual = vec![Vec::new(); 12];
        let reduce_stats =
            ExternalHashPruneReducer::reduce_in_memory_spill_sharded_in_order_profiled_from(
                0,
                12,
                4,
                artifacts,
                |point, mut neighbors| {
                    neighbors.sort_unstable();
                    actual[point as usize] = neighbors;
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(reduce_stats.part_files_opened, 0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn in_memory_spill_batch_encoding_groups_records_by_shard() {
        let mut batch = vec![
            PendingEdge {
                p: 9,
                c: 30,
                hash: 4,
                dist: 0.4,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 1,
                c: 10,
                hash: 2,
                dist: 0.1,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 5,
                c: 20,
                hash: 1,
                dist: 0.2,
                mandatory: false,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
            PendingEdge {
                p: 5,
                c: 21,
                hash: 3,
                dist: 0.3,
                mandatory: true,
                local_rank: 1,
                flags: PENDING_EDGE_DIRECT,
            },
        ];

        let groups = encode_spill_records_by_shard(&mut batch, 4).unwrap();

        assert!(batch.is_empty());
        assert_eq!(
            groups
                .iter()
                .map(|(shard_id, records)| (
                    *shard_id,
                    records.iter().map(|record| record.src).collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>(),
            vec![(0, vec![1]), (1, vec![5, 5]), (2, vec![9])]
        );
        assert_eq!(
            groups[1].1[1].flags & SPILL_FLAG_MANDATORY,
            SPILL_FLAG_MANDATORY
        );
    }

    #[test]
    fn spill_reduce_succeeds_without_preparing_compaction_runs() {
        let dir = tempdir().unwrap();
        let batches = vec![
            (
                0usize,
                vec![
                    PendingEdge {
                        p: 0,
                        c: 10,
                        hash: 1,
                        dist: 0.9,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 1,
                        c: 12,
                        hash: 4,
                        dist: 0.5,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
            (
                1usize,
                vec![
                    PendingEdge {
                        p: 2,
                        c: 20,
                        hash: 2,
                        dist: 0.8,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 3,
                        c: 30,
                        hash: 3,
                        dist: 0.7,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
            (
                0usize,
                vec![
                    PendingEdge {
                        p: 0,
                        c: 11,
                        hash: 1,
                        dist: 0.4,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 1,
                        c: 13,
                        hash: 5,
                        dist: 0.6,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
            (
                1usize,
                vec![
                    PendingEdge {
                        p: 2,
                        c: 21,
                        hash: 2,
                        dist: 0.1,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 3,
                        c: 31,
                        hash: 3,
                        dist: 0.3,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
            (
                2usize,
                vec![
                    PendingEdge {
                        p: 4,
                        c: 40,
                        hash: 6,
                        dist: 0.2,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 4,
                        c: 41,
                        hash: 7,
                        dist: 0.9,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
        ];

        let mut part_files = Vec::new();
        let mut segment_manifest = Vec::new();
        let mut all_edges = Vec::new();
        for (idx, (shard_id, batch)) in batches.iter().enumerate() {
            let path = dir.path().join(format!("batch_{idx}.run"));
            let mut writer = CandidateRunWriter::create(&path).unwrap();
            for edge in batch {
                writer.push(edge).unwrap();
            }
            writer.finish().unwrap();
            let byte_len = fs::metadata(&path).unwrap().len();
            part_files.push(super::SpillPartFileDesc {
                shard_id: *shard_id,
                file_id: idx as u16,
                path: path.clone(),
                byte_len,
            });
            segment_manifest.push(super::SpillSegmentDesc {
                shard_id: *shard_id as u32,
                file_id: idx as u16,
                byte_offset: 0,
                byte_len,
                record_count: batch.len() as u32,
                src_min: batch.iter().map(|edge| edge.p as u32).min().unwrap(),
                src_max: batch.iter().map(|edge| edge.p as u32).max().unwrap(),
                sorted_by_src: true,
            });
            all_edges.extend_from_slice(batch);
        }
        let expected = reduce_in_memory(5, 4, &all_edges);

        let mut streamed = Vec::new();
        super::reduce_segments_sharded_in_order_with_limit(
            5,
            4,
            2,
            &super::SpillArtifacts {
                part_files,
                segment_manifest,
                manifest_path: dir.path().join("candidate_segments.manifest"),
            },
            2,
            |point, mut neighbors| {
                neighbors.sort_unstable();
                streamed.push((point, neighbors));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            streamed.iter().map(|(point, _)| *point).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        for (point, neighbors) in streamed {
            assert_eq!(neighbors, expected[point as usize]);
        }
    }

    #[test]
    fn spill_reduce_shard_segments_stay_correct_under_low_open_file_budget() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join("candidate.run");
        let batches = vec![
            (
                0usize,
                0usize,
                vec![
                    PendingEdge {
                        p: 0,
                        c: 10,
                        hash: 1,
                        dist: 0.9,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 1,
                        c: 11,
                        hash: 2,
                        dist: 0.8,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
            (
                1usize,
                0usize,
                vec![
                    PendingEdge {
                        p: 2,
                        c: 12,
                        hash: 3,
                        dist: 0.7,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 3,
                        c: 13,
                        hash: 4,
                        dist: 0.6,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
            (
                2usize,
                0usize,
                vec![
                    PendingEdge {
                        p: 0,
                        c: 14,
                        hash: 1,
                        dist: 0.2,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 1,
                        c: 15,
                        hash: 5,
                        dist: 0.3,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
            (
                3usize,
                1usize,
                vec![
                    PendingEdge {
                        p: 4,
                        c: 20,
                        hash: 6,
                        dist: 0.5,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 5,
                        c: 21,
                        hash: 7,
                        dist: 0.4,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
            (
                4usize,
                1usize,
                vec![
                    PendingEdge {
                        p: 6,
                        c: 22,
                        hash: 8,
                        dist: 0.3,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 7,
                        c: 23,
                        hash: 9,
                        dist: 0.2,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
            (
                5usize,
                1usize,
                vec![
                    PendingEdge {
                        p: 4,
                        c: 24,
                        hash: 6,
                        dist: 0.1,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                    PendingEdge {
                        p: 7,
                        c: 25,
                        hash: 10,
                        dist: 0.9,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_DIRECT,
                    },
                ],
            ),
        ];

        let mut run_paths = Vec::new();
        let mut part_files = Vec::new();
        let mut segments = Vec::new();
        let mut all_edges = Vec::new();
        for (run_index, shard_id, edges) in batches {
            let path = super::part_path_for(&base_path, run_index, shard_id);
            let mut writer = CandidateRunWriter::create(&path).unwrap();
            for edge in &edges {
                writer.push(edge).unwrap();
            }
            writer.finish().unwrap();
            let byte_len = fs::metadata(&path).unwrap().len();
            run_paths.push(path);
            part_files.push(super::SpillPartFileDesc {
                shard_id,
                file_id: run_index as u16,
                path: run_paths.last().unwrap().clone(),
                byte_len,
            });
            segments.push(super::SpillSegmentDesc {
                shard_id: shard_id as u32,
                file_id: run_index as u16,
                byte_offset: 0,
                byte_len,
                record_count: edges.len() as u32,
                src_min: edges.iter().map(|edge| edge.p as u32).min().unwrap(),
                src_max: edges.iter().map(|edge| edge.p as u32).max().unwrap(),
                sorted_by_src: true,
            });
            all_edges.extend_from_slice(&edges);
        }

        let expected = reduce_in_memory(8, 4, &all_edges);
        let mut streamed = Vec::new();
        let artifacts = super::SpillArtifacts {
            part_files,
            segment_manifest: segments,
            manifest_path: dir.path().join("candidate_segments.manifest"),
        };
        super::reduce_segments_sharded_in_order_with_limit(
            8,
            4,
            4,
            &artifacts,
            2,
            |point, mut neighbors| {
                neighbors.sort_unstable();
                streamed.push((point, neighbors));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            streamed.iter().map(|(point, _)| *point).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        for (point, neighbors) in streamed {
            assert_eq!(neighbors, expected[point as usize]);
        }
    }

    #[test]
    fn spill_artifact_summary_counts_parts_segments_and_shards() {
        let dir = tempdir().unwrap();
        let shard0 = dir.path().join("candidate_shard00000_part000.log");
        let shard1 = dir.path().join("candidate_shard00001_part000.log");
        fs::write(&shard0, vec![0u8; 32]).unwrap();
        fs::write(&shard1, vec![0u8; 48]).unwrap();

        let artifacts = super::SpillArtifacts {
            part_files: vec![
                super::SpillPartFileDesc {
                    shard_id: 0,
                    file_id: 0,
                    path: shard0,
                    byte_len: 32,
                },
                super::SpillPartFileDesc {
                    shard_id: 1,
                    file_id: 1,
                    path: shard1,
                    byte_len: 48,
                },
            ],
            segment_manifest: vec![
                super::SpillSegmentDesc {
                    shard_id: 0,
                    file_id: 0,
                    byte_offset: 0,
                    byte_len: 16,
                    record_count: 1,
                    src_min: 0,
                    src_max: 0,
                    sorted_by_src: true,
                },
                super::SpillSegmentDesc {
                    shard_id: 0,
                    file_id: 0,
                    byte_offset: 16,
                    byte_len: 16,
                    record_count: 1,
                    src_min: 1,
                    src_max: 1,
                    sorted_by_src: true,
                },
                super::SpillSegmentDesc {
                    shard_id: 1,
                    file_id: 1,
                    byte_offset: 0,
                    byte_len: 48,
                    record_count: 3,
                    src_min: 4,
                    src_max: 6,
                    sorted_by_src: true,
                },
            ],
            manifest_path: dir.path().join("candidate_segments.manifest"),
        };

        let stats = super::summarize_spill_artifacts(&artifacts);
        assert_eq!(stats.part_files, 2);
        assert_eq!(stats.segment_count, 3);
        assert_eq!(stats.shard_groups, 2);
        assert_eq!(stats.max_parts_per_shard, 1);
        assert_eq!(stats.max_segments_per_shard, 2);
    }
}
