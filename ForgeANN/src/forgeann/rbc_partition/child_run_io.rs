use serde::{Deserialize, Serialize};

use super::*;
use crate::forgeann::hash_prune::{ResidentSubsetSketchAccessor, SketchAccessor};
use crate::forgeann::io_runtime::BoundedReadPlan;
use crate::forgeann::leaf_build::LeafProfile;
use crate::forgeann::point_pipeline::{PointPipelineConfig, hydrate_resident_subset_with_stats};
use crate::forgeann::point_store::{WindowedGatherOptions, WindowedGatherStats};

pub(crate) struct ExternalRunStore {
    pub base_dir: PathBuf,
    pub vector_dir: PathBuf,
    pub io_cfg: DirectIoConfig,
    pub writers: HashMap<usize, RunAppender>,
    /// Global atomic temp budget for io-planned vector-run. `None` when io-planning disabled.
    pub io_plan_budget:
        Option<std::sync::Arc<crate::forgeann::io_planned_forgeann::IoPlanBudgetState>>,
}

#[derive(Debug)]
pub(crate) enum RunAppender {
    Buffered(BufWriter<File>),
    Direct { file: DirectIoFile, offset: u64 },
}

impl ExternalRunStore {
    pub(crate) fn new(base_dir: &Path, io_cfg: DirectIoConfig) -> Self {
        Self::new_with_vector_dir(base_dir, base_dir, io_cfg)
    }

    pub(crate) fn new_with_vector_dir(
        base_dir: &Path,
        vector_dir: &Path,
        io_cfg: DirectIoConfig,
    ) -> Self {
        Self {
            base_dir: base_dir.to_path_buf(),
            vector_dir: vector_dir.to_path_buf(),
            io_cfg,
            writers: HashMap::new(),
            io_plan_budget: None,
        }
    }

    fn with_io_plan_budget(
        mut self,
        budget: std::sync::Arc<crate::forgeann::io_planned_forgeann::IoPlanBudgetState>,
    ) -> Self {
        self.io_plan_budget = Some(budget);
        self
    }

    fn path_for_depth(&self, depth: usize) -> PathBuf {
        self.base_dir
            .join(format!("partition_d{depth:02}_runs.bin"))
    }

    pub(crate) fn vector_path(&self, depth: usize, ordinal: usize) -> PathBuf {
        self.vector_dir
            .join(format!("partition_d{depth:02}_vectors_{ordinal:08}.bin"))
    }

    pub(crate) fn io_config(&self) -> DirectIoConfig {
        self.io_cfg
    }

    pub(crate) fn append_points(&mut self, depth: usize, points: &[u32]) -> AnnResult<RunExtent> {
        let path = self.path_for_depth(depth);
        let io_cfg = self.io_cfg;
        let writer = self.writers.entry(depth).or_insert_with(|| {
            if io_cfg.enabled {
                let offset = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
                RunAppender::Direct {
                    file: DirectIoFile::open_rw(&path, io_cfg, false).unwrap(),
                    offset,
                }
            } else {
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .read(true)
                    .open(&path)
                    .unwrap();
                RunAppender::Buffered(BufWriter::new(file))
            }
        });

        let byte_offset = match writer {
            RunAppender::Buffered(writer) => {
                let byte_offset = writer.seek(SeekFrom::End(0))?;
                writer.write_all(&(points.len() as u64).to_le_bytes())?;
                for &point in points {
                    writer.write_all(&point.to_le_bytes())?;
                }
                writer.flush()?;
                byte_offset
            }
            RunAppender::Direct { file, offset } => {
                let byte_offset = *offset;
                let mut bytes =
                    Vec::with_capacity(RUN_EXTENT_HEADER_BYTES + points.len() * size_of::<u32>());
                bytes.extend_from_slice(&(points.len() as u64).to_le_bytes());
                for &point in points {
                    bytes.extend_from_slice(&point.to_le_bytes());
                }
                file.write_all_at(&bytes, *offset)?;
                *offset += bytes.len() as u64;
                byte_offset
            }
        };
        Ok(RunExtent {
            byte_offset,
            len: points.len(),
        })
    }

    fn read_points(&self, depth: usize, extent: RunExtent) -> AnnResult<Vec<u32>> {
        read_run_points_from_path(&self.path_for_depth(depth), self.io_cfg, extent)
    }

    pub(crate) fn finalize(&mut self) -> AnnResult<()> {
        for writer in self.writers.values_mut() {
            match writer {
                RunAppender::Buffered(writer) => {
                    writer.flush()?;
                }
                RunAppender::Direct { file, offset } => {
                    file.set_len(*offset)?;
                    file.sync_all()?;
                }
            }
        }
        Ok(())
    }
}

impl Drop for ExternalRunStore {
    fn drop(&mut self) {
        let _ = self.finalize();
    }
}

pub(crate) fn write_child_run(
    store: &mut ExternalRunStore,
    depth: usize,
    points: &[u32],
) -> AnnResult<RunExtent> {
    store.append_points(depth, points)
}

pub(crate) fn read_run_points_from_path(
    path: &Path,
    io_cfg: DirectIoConfig,
    extent: RunExtent,
) -> AnnResult<Vec<u32>> {
    if io_cfg.enabled {
        let file = DirectIoFile::open_read(path, io_cfg)?;
        let mut header = [0u8; RUN_EXTENT_HEADER_BYTES];
        file.read_exact_at(&mut header, extent.byte_offset)?;
        let len = u64::from_le_bytes(header) as usize;
        let mut bytes = vec![0u8; len * size_of::<u32>()];
        file.read_exact_at(
            &mut bytes,
            extent.byte_offset + RUN_EXTENT_HEADER_BYTES as u64,
        )?;
        let mut points = vec![0_u32; len];
        for (idx, point) in points.iter_mut().enumerate() {
            let start = idx * size_of::<u32>();
            *point = u32::from_le_bytes(bytes[start..start + size_of::<u32>()].try_into().unwrap());
        }
        return Ok(points);
    }

    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(SeekFrom::Start(extent.byte_offset))?;
    let mut header = [0u8; RUN_EXTENT_HEADER_BYTES];
    reader.read_exact(&mut header)?;
    let len = u64::from_le_bytes(header) as usize;
    let mut points = vec![0_u32; len];
    let mut buf = [0u8; size_of::<u32>()];
    for point in &mut points {
        reader.read_exact(&mut buf)?;
        *point = u32::from_le_bytes(buf);
    }
    Ok(points)
}

pub(crate) fn read_child_run(
    store: &ExternalRunStore,
    depth: usize,
    extent: RunExtent,
) -> AnnResult<Vec<u32>> {
    let points = store.read_points(depth, extent)?;
    debug_assert_eq!(points.len(), extent.len);
    Ok(points)
}

pub(crate) fn read_child_run_chain(
    store: &ExternalRunStore,
    depth: usize,
    extents: &[RunExtent],
) -> AnnResult<Vec<u32>> {
    let total_len = extents.iter().map(|extent| extent.len).sum();
    let mut points = Vec::with_capacity(total_len);
    for &extent in extents {
        let mut chunk = read_child_run(store, depth, extent)?;
        points.append(&mut chunk);
    }
    Ok(points)
}

pub(crate) fn read_child_run_chain_from_path(
    base_dir: &Path,
    io_cfg: DirectIoConfig,
    depth: usize,
    extents: &[RunExtent],
) -> AnnResult<Vec<u32>> {
    let total_len = extents.iter().map(|extent| extent.len).sum();
    let mut points = Vec::with_capacity(total_len);
    let path = base_dir.join(format!("partition_d{depth:02}_runs.bin"));
    for &extent in extents {
        let mut chunk = read_run_points_from_path(&path, io_cfg, extent)?;
        points.append(&mut chunk);
    }
    Ok(points)
}

pub(crate) const D2_TRACE_DUMP_COMPLETE: &str = "forgeann_d2_trace_dump_complete";

fn write_d2_trace_dump(
    path: &Path,
    base_dir: &Path,
    io_cfg: DirectIoConfig,
    source_depth: usize,
    replay_depth: usize,
    child_runs: &[ChildRun],
    num_points: usize,
    row_bytes: usize,
) -> AnnResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let start = Instant::now();
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);
    let total_occurrences = child_runs.iter().map(|run| run.len).sum::<usize>();
    writer.write_all(b"FAD2TRC1")?;
    writer.write_all(&1_u64.to_le_bytes())?;
    writer.write_all(&(child_runs.len() as u64).to_le_bytes())?;
    writer.write_all(&(num_points as u64).to_le_bytes())?;
    writer.write_all(&(row_bytes as u64).to_le_bytes())?;
    writer.write_all(&(total_occurrences as u64).to_le_bytes())?;
    writer.write_all(&(source_depth as u64).to_le_bytes())?;
    writer.write_all(&(replay_depth as u64).to_le_bytes())?;

    let mut offset = 0_u64;
    let mut loaded_occurrences = 0usize;
    for (run_id, run) in child_runs.iter().enumerate() {
        writer.write_all(&(run_id as u64).to_le_bytes())?;
        writer.write_all(&offset.to_le_bytes())?;
        writer.write_all(&(run.len as u64).to_le_bytes())?;
        offset = offset.saturating_add(run.len as u64);
    }
    for run in child_runs {
        let ids = read_child_run_chain_from_path(base_dir, io_cfg, source_depth, &run.extents)?;
        debug_assert_eq!(ids.len(), run.len);
        loaded_occurrences = loaded_occurrences.saturating_add(ids.len());
        for id in ids {
            writer.write_all(&id.to_le_bytes())?;
        }
    }
    writer.flush()?;
    tracing::info!(
        "[d2-trace-dump] path={} source_depth={} replay_depth={} child_runs={} total_occurrences={} row_bytes={} elapsed_ms={}",
        path.display(),
        source_depth,
        replay_depth,
        child_runs.len(),
        loaded_occurrences,
        row_bytes,
        start.elapsed().as_millis(),
    );
    Ok(())
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ChildRunExtentBatchStats {
    pub logical_extents: usize,
    pub coalesced_reads: usize,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub header_read_savings: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ChildExtentRead {
    pub run_idx: usize,
    pub extent_idx: usize,
    pub extent: RunExtent,
    pub start: u64,
    pub end: u64,
}

pub(crate) fn read_child_runs_batched_from_path(
    base_dir: &Path,
    io_cfg: DirectIoConfig,
    depth: usize,
    runs: &[ChildRun],
) -> AnnResult<(Vec<Vec<u32>>, ChildRunExtentBatchStats)> {
    let path = base_dir.join(format!("partition_d{depth:02}_runs.bin"));
    let mut reads = Vec::new();
    for (run_idx, run) in runs.iter().enumerate() {
        for (extent_idx, &extent) in run.extents.iter().enumerate() {
            let logical_bytes =
                RUN_EXTENT_HEADER_BYTES + extent.len.saturating_mul(size_of::<u32>());
            let start = extent.byte_offset;
            let end = start.saturating_add(logical_bytes as u64);
            reads.push(ChildExtentRead {
                run_idx,
                extent_idx,
                extent,
                start,
                end,
            });
        }
    }

    reads.sort_unstable_by_key(|read| (read.start, read.end));
    let mut output: Vec<Vec<u32>> = runs.iter().map(|run| Vec::with_capacity(run.len)).collect();
    let mut stats = ChildRunExtentBatchStats {
        logical_extents: reads.len(),
        logical_bytes: reads
            .iter()
            .map(|read| read.end.saturating_sub(read.start))
            .sum(),
        ..ChildRunExtentBatchStats::default()
    };

    let mut idx = 0usize;
    while idx < reads.len() {
        let group_start = reads[idx].start;
        let mut group_end = reads[idx].end;
        let mut end_idx = idx + 1;
        while end_idx < reads.len() && reads[end_idx].start <= group_end {
            group_end = group_end.max(reads[end_idx].end);
            end_idx += 1;
        }

        let mut bytes = vec![0u8; group_end.saturating_sub(group_start) as usize];
        if io_cfg.enabled {
            let file = DirectIoFile::open_read(&path, io_cfg)?;
            file.read_exact_at(&mut bytes, group_start)?;
        } else {
            let file = File::open(&path)?;
            read_exact_at_file(&file, &path, &mut bytes, group_start)?;
        }
        stats.coalesced_reads += 1;
        stats.physical_bytes = stats
            .physical_bytes
            .saturating_add(group_end.saturating_sub(group_start));

        let mut group_reads: Vec<ChildExtentRead> = reads[idx..end_idx].to_vec();
        group_reads.sort_unstable_by_key(|read| (read.run_idx, read.extent_idx));
        for read in group_reads {
            let offset = read.start.saturating_sub(group_start) as usize;
            let header_end = offset + RUN_EXTENT_HEADER_BYTES;
            let len = u64::from_le_bytes(bytes[offset..header_end].try_into().unwrap()) as usize;
            debug_assert_eq!(len, read.extent.len);
            let payload_start = header_end;
            let payload_end = payload_start + len.saturating_mul(size_of::<u32>());
            let mut payload = Vec::with_capacity(len);
            for chunk in bytes[payload_start..payload_end].chunks_exact(size_of::<u32>()) {
                payload.push(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
            output[read.run_idx].extend(payload);
        }
        idx = end_idx;
    }

    stats.header_read_savings = stats.logical_extents.saturating_sub(stats.coalesced_reads);
    Ok((output, stats))
}

pub(crate) fn read_exact_at_file(
    file: &File,
    path: &Path,
    mut buf: &mut [u8],
    mut offset: u64,
) -> AnnResult<()> {
    while !buf.is_empty() {
        let read = file.read_at(buf, offset)?;
        if read == 0 {
            return Err(AnnError::log_index_error(format!(
                "Unexpected EOF while reading {} at offset {}",
                path.display(),
                offset
            )));
        }
        let (_, rest) = buf.split_at_mut(read);
        buf = rest;
        offset += read as u64;
    }
    Ok(())
}

pub(crate) fn merge_cluster_plan(
    raw_counts: &[usize],
    c_min: usize,
    c_max: usize,
) -> Vec<MergedRawGroup> {
    let mut clusters: Vec<(u16, usize)> = raw_counts
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(raw_child, len)| (len > 0).then_some((raw_child as u16, len)))
        .collect();
    if clusters.is_empty() {
        return Vec::new();
    }

    clusters.sort_by_key(|&(_, len)| len);

    let mut result: Vec<MergedRawGroup> = Vec::new();
    let mut current_small: Option<MergedRawGroup> = None;

    for (raw_child, size) in clusters {
        let cluster = MergedRawGroup {
            raw_children: vec![raw_child],
            raw_len: size,
        };
        if size >= c_min {
            if let Some(small) = current_small.take() {
                if small.raw_len + size <= c_max {
                    let mut merged_children = vec![raw_child];
                    merged_children.extend_from_slice(&small.raw_children);
                    result.push(MergedRawGroup {
                        raw_children: merged_children,
                        raw_len: size + small.raw_len,
                    });
                } else {
                    result.push(small);
                    result.push(cluster);
                }
            } else {
                result.push(cluster);
            }
        } else {
            match current_small.take() {
                Some(small) => {
                    if small.raw_len + size <= c_max {
                        let mut merged_children = small.raw_children;
                        merged_children.push(raw_child);
                        current_small = Some(MergedRawGroup {
                            raw_children: merged_children,
                            raw_len: small.raw_len + size,
                        });
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
            if last.raw_len + small.raw_len <= c_max {
                last.raw_children.extend_from_slice(&small.raw_children);
                last.raw_len += small.raw_len;
            } else {
                result.push(small);
            }
        } else {
            result.push(small);
        }
    }

    result
}

fn percentile_sorted(sorted: &[usize], numerator: usize, denominator: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = sorted.len().saturating_sub(1).saturating_mul(numerator) / denominator.max(1);
    sorted[idx]
}

fn log_external_child_distribution(
    depth: usize,
    leaders: usize,
    fanout: usize,
    raw_counts: &[usize],
    merge_groups: &[MergedRawGroup],
    c_min: usize,
    c_max: usize,
) {
    let mut raw_nonempty: Vec<usize> = raw_counts
        .iter()
        .copied()
        .filter(|&count| count > 0)
        .collect();
    raw_nonempty.sort_unstable();
    let raw_total = raw_nonempty.iter().sum::<usize>();
    let raw_max = raw_nonempty.last().copied().unwrap_or(0);
    let raw_gt_cmax = raw_nonempty.iter().filter(|&&count| count > c_max).count();

    let mut merged_lens: Vec<usize> = merge_groups.iter().map(|group| group.raw_len).collect();
    merged_lens.sort_unstable();
    let merged_max = merged_lens.last().copied().unwrap_or(0);
    let merged_gt_cmax = merged_lens.iter().filter(|&&count| count > c_max).count();
    let merged_gt_10m = merged_lens
        .iter()
        .filter(|&&count| count >= 10_000_000)
        .count();

    tracing::info!(
        "[rbc/external-child-distribution] depth={} leaders={} fanout={} c_min={} c_max={} raw_nonempty={} raw_total={} raw_p50={} raw_p90={} raw_p99={} raw_max={} raw_gt_cmax={} merged_groups={} merged_p50={} merged_p90={} merged_p99={} merged_max={} merged_gt_cmax={} merged_gt_10m={}",
        depth,
        leaders,
        fanout,
        c_min,
        c_max,
        raw_nonempty.len(),
        raw_total,
        percentile_sorted(&raw_nonempty, 50, 100),
        percentile_sorted(&raw_nonempty, 90, 100),
        percentile_sorted(&raw_nonempty, 99, 100),
        raw_max,
        raw_gt_cmax,
        merged_lens.len(),
        percentile_sorted(&merged_lens, 50, 100),
        percentile_sorted(&merged_lens, 90, 100),
        percentile_sorted(&merged_lens, 99, 100),
        merged_max,
        merged_gt_cmax,
        merged_gt_10m,
    );
}

pub(crate) fn assign_record_fanout(local_fanout: usize) -> usize {
    local_fanout.max(1).min(32)
}

pub(crate) fn assign_record_width(record_cap: usize) -> usize {
    1 + 2 * assign_record_fanout(record_cap)
}

pub(crate) fn write_assign_record(
    writer: &mut BufWriter<File>,
    leaders: &[u16],
    record_cap: usize,
) -> AnnResult<()> {
    let record_cap = assign_record_fanout(record_cap);
    if leaders.len() > record_cap {
        return Err(AnnError::log_index_error(format!(
            "Assignment record contains {} leaders above cap {}",
            leaders.len(),
            record_cap
        )));
    }
    writer.write_all(&[leaders.len() as u8])?;
    for &leader in leaders {
        writer.write_all(&leader.to_le_bytes())?;
    }
    for _ in leaders.len()..record_cap {
        writer.write_all(&0u16.to_le_bytes())?;
    }
    Ok(())
}

pub(crate) fn read_assign_record(
    reader: &mut BufReader<File>,
    out: &mut [u16],
) -> AnnResult<usize> {
    let mut count_buf = [0u8; 1];
    reader.read_exact(&mut count_buf)?;
    let count = count_buf[0] as usize;
    if count > out.len() {
        return Err(AnnError::log_index_error(format!(
            "Assignment record count {count} exceeds record cap {}",
            out.len()
        )));
    }
    let mut buf = [0u8; size_of::<u16>()];
    for slot in out.iter_mut() {
        reader.read_exact(&mut buf)?;
        *slot = u16::from_le_bytes(buf);
    }
    Ok(count)
}

pub(crate) fn append_assign_record_bytes(
    buf: &mut Vec<u8>,
    leaders: &[u16],
    record_cap: usize,
) -> AnnResult<()> {
    let record_cap = assign_record_fanout(record_cap);
    if leaders.len() > record_cap {
        return Err(AnnError::log_index_error(format!(
            "Assignment record contains {} leaders above cap {}",
            leaders.len(),
            record_cap
        )));
    }
    buf.push(leaders.len() as u8);
    for &leader in leaders {
        buf.extend_from_slice(&leader.to_le_bytes());
    }
    for _ in leaders.len()..record_cap {
        buf.extend_from_slice(&0u16.to_le_bytes());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AssignmentSpoolSegment {
    pub offset: u64,
    pub len: u64,
}

pub(crate) trait AssignmentRecordSource {
    fn read_record(&mut self, out: &mut [u16]) -> AnnResult<usize>;
}

pub(crate) struct ContiguousAssignmentRecordSource {
    pub reader: BufReader<File>,
}

impl AssignmentRecordSource for ContiguousAssignmentRecordSource {
    fn read_record(&mut self, out: &mut [u16]) -> AnnResult<usize> {
        read_assign_record(&mut self.reader, out)
    }
}

pub(crate) struct SegmentedAssignmentRecordSource<'a> {
    pub reader: BufReader<File>,
    pub segments: &'a [AssignmentSpoolSegment],
    pub next_segment_idx: usize,
    pub remaining_bytes: u64,
    pub record_width: u64,
}

impl<'a> SegmentedAssignmentRecordSource<'a> {
    pub(crate) fn new(
        spool: &NamedTempFile,
        segments: &'a [AssignmentSpoolSegment],
        record_cap: usize,
    ) -> AnnResult<Self> {
        Ok(Self {
            reader: BufReader::new(spool.reopen()?),
            segments,
            next_segment_idx: 0,
            remaining_bytes: 0,
            record_width: assign_record_width(record_cap) as u64,
        })
    }

    fn seek_next_segment(&mut self) -> AnnResult<bool> {
        while self.remaining_bytes == 0 {
            let Some(segment) = self.segments.get(self.next_segment_idx).copied() else {
                return Ok(false);
            };
            self.next_segment_idx += 1;
            if segment.len == 0 {
                continue;
            }
            if segment.len % self.record_width != 0 {
                return Err(AnnError::log_index_error(format!(
                    "Assignment spool segment length {} is not aligned to record width {}",
                    segment.len, self.record_width
                )));
            }
            self.reader.seek(SeekFrom::Start(segment.offset))?;
            self.remaining_bytes = segment.len;
            return Ok(true);
        }
        Ok(true)
    }
}

impl AssignmentRecordSource for SegmentedAssignmentRecordSource<'_> {
    fn read_record(&mut self, out: &mut [u16]) -> AnnResult<usize> {
        if !self.seek_next_segment()? {
            return Err(AnnError::log_index_error(
                "Assignment spool segments ended before all points were materialized".to_string(),
            ));
        }
        let len = read_assign_record(&mut self.reader, out)?;
        self.remaining_bytes = self.remaining_bytes.saturating_sub(self.record_width);
        Ok(len)
    }
}

pub(crate) fn flush_materialized_child_buffer(
    store: &mut ExternalRunStore,
    depth: usize,
    child: &mut BufferedMergedChild,
) -> AnnResult<()> {
    if child.buffer.is_empty() {
        return Ok(());
    }
    let extent = write_child_run(store, depth, &child.buffer)?;
    child.extents.push(extent);
    child.buffer.clear();
    Ok(())
}

pub(crate) fn materialize_merged_children_from_spool_with_inline_limit(
    cur: &[u32],
    depth: usize,
    fanout: usize,
    merge_groups: &[MergedRawGroup],
    spool: &NamedTempFile,
    store: &mut ExternalRunStore,
    inline_limit: Option<usize>,
) -> AnnResult<Vec<MaterializedMergedChild>> {
    let mut source = ContiguousAssignmentRecordSource {
        reader: BufReader::new(spool.reopen()?),
    };
    materialize_merged_children_from_assignment_records(
        cur,
        depth,
        fanout,
        merge_groups,
        &mut source,
        store,
        inline_limit,
    )
}

pub(crate) fn materialize_merged_children_from_spool_segments_with_inline_limit(
    cur: &[u32],
    depth: usize,
    fanout: usize,
    merge_groups: &[MergedRawGroup],
    spool: &NamedTempFile,
    segments: &[AssignmentSpoolSegment],
    store: &mut ExternalRunStore,
    inline_limit: Option<usize>,
) -> AnnResult<Vec<MaterializedMergedChild>> {
    let record_cap = assign_record_fanout(fanout);
    let mut source = SegmentedAssignmentRecordSource::new(spool, segments, record_cap)?;
    materialize_merged_children_from_assignment_records(
        cur,
        depth,
        fanout,
        merge_groups,
        &mut source,
        store,
        inline_limit,
    )
}

pub(crate) fn materialize_merged_children_from_assignment_records(
    cur: &[u32],
    depth: usize,
    fanout: usize,
    merge_groups: &[MergedRawGroup],
    assignment_source: &mut impl AssignmentRecordSource,
    store: &mut ExternalRunStore,
    inline_limit: Option<usize>,
) -> AnnResult<Vec<MaterializedMergedChild>> {
    let raw_child_count = merge_groups
        .iter()
        .flat_map(|group| group.raw_children.iter())
        .copied()
        .max()
        .map(|idx| idx as usize + 1)
        .unwrap_or(0);
    let mut raw_to_merged = vec![u16::MAX; raw_child_count];
    for (merged_idx, group) in merge_groups.iter().enumerate() {
        for &raw_child in &group.raw_children {
            raw_to_merged[raw_child as usize] = merged_idx as u16;
        }
    }

    let record_cap = assign_record_fanout(fanout);
    let buffer_cap = MATERIALIZE_CHILD_BUFFER_POINTS.max(record_cap);
    let mut children: Vec<BufferedMergedChild> = merge_groups
        .iter()
        .map(|group| {
            let inline = inline_limit.map_or(false, |limit| group.raw_len <= limit);
            BufferedMergedChild {
                extents: Vec::new(),
                buffer: Vec::with_capacity(if inline {
                    group.raw_len.min(buffer_cap)
                } else {
                    buffer_cap
                }),
                inline,
            }
        })
        .collect();
    let batch_points = MATERIALIZE_BATCH_POINTS.max(buffer_cap);
    let mut assignment_batch = vec![0u16; batch_points * record_cap];
    let mut assignment_counts = vec![0usize; batch_points];

    for point_batch in cur.chunks(batch_points) {
        let used = point_batch.len() * record_cap;
        for (row, record) in assignment_batch[..used].chunks_mut(record_cap).enumerate() {
            assignment_counts[row] = assignment_source.read_record(record)?;
        }

        let per_point_merged: Vec<([u16; 32], u8)> = assignment_batch[..used]
            .par_chunks(record_cap)
            .zip(assignment_counts[..point_batch.len()].par_iter())
            .map(|(record, &record_len)| {
                let mut merged_for_point = [u16::MAX; 32];
                let mut unique_count = 0usize;
                for &raw_child in &record[..record_len] {
                    let merged_idx = raw_to_merged[raw_child as usize];
                    if merged_for_point[..unique_count].contains(&merged_idx) {
                        continue;
                    }
                    merged_for_point[unique_count] = merged_idx;
                    unique_count += 1;
                }
                (merged_for_point, unique_count as u8)
            })
            .collect();

        for (point, (merged_children, unique_count)) in
            point_batch.iter().copied().zip(per_point_merged.iter())
        {
            for &merged_idx in &merged_children[..usize::from(*unique_count)] {
                let child = &mut children[merged_idx as usize];
                child.buffer.push(point);
                if !child.inline && child.buffer.len() >= buffer_cap {
                    flush_materialized_child_buffer(store, depth, child)?;
                }
            }
        }
    }

    let mut materialized = Vec::with_capacity(children.len());
    for mut child in children {
        if !child.inline {
            flush_materialized_child_buffer(store, depth, &mut child)?;
        }
        let points = child.inline.then_some(child.buffer);
        materialized.push(MaterializedMergedChild {
            extents: child.extents,
            points,
        });
    }
    Ok(materialized)
}

pub(crate) fn compute_clusters_scalar_to_spool(
    dataset: &dyn PointStore,
    cur: &[u32],
    leaders: &[u32],
    local_fanout: usize,
    metric: Metric,
    spool_writer: &mut BufWriter<File>,
    raw_counts: &mut [usize],
) -> AnnResult<()> {
    let fanout = assign_record_fanout(local_fanout.min(leaders.len().max(1)));
    if fanout == 1 {
        for &idx in cur {
            let mut best_lid = 0usize;
            let mut best_dist = f32::INFINITY;
            for (lid, &lid_global) in leaders.iter().enumerate() {
                let d = dataset.get_distance(idx, lid_global, metric)?;
                if d < best_dist {
                    best_dist = d;
                    best_lid = lid;
                }
            }
            raw_counts[best_lid] += 1;
            let leader = [best_lid as u16];
            write_assign_record(spool_writer, &leader, fanout)?;
        }
    } else {
        for &idx in cur {
            let mut top_k = StackTopK::new(fanout);
            for (lid, &lid_global) in leaders.iter().enumerate() {
                let d = dataset.get_distance(idx, lid_global, metric)?;
                top_k.push(d, lid);
            }
            let mut leaders_for_point = [0u16; 32];
            let mut len = 0usize;
            for &(_, lid) in top_k.iter() {
                raw_counts[lid] += 1;
                leaders_for_point[len] = lid as u16;
                len += 1;
            }
            write_assign_record(spool_writer, &leaders_for_point[..len], fanout)?;
        }
    }
    spool_writer.flush()?;
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct DepthStats {
    pub clusters_processed: usize,
    pub leaves_generated: usize,
    pub total_points: usize,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
}

impl DepthStats {
    pub(crate) fn new() -> Self {
        Self {
            clusters_processed: 0,
            leaves_generated: 0,
            total_points: 0,
            min_cluster_size: usize::MAX,
            max_cluster_size: 0,
        }
    }

    fn observe_size(&mut self, size: usize) {
        self.min_cluster_size = self.min_cluster_size.min(size);
        self.max_cluster_size = self.max_cluster_size.max(size);
    }

    pub(crate) fn min_cluster_size(&self) -> usize {
        if self.min_cluster_size == usize::MAX {
            0
        } else {
            self.min_cluster_size
        }
    }
}

#[derive(Debug)]
pub(crate) struct PartitionStats {
    pub depth_stats: Vec<DepthStats>,
    pub max_depth_seen: usize,

    pub early_stop_max_depth: usize,
    pub early_stop_shrink_ratio: usize,
    pub early_stop_small_deep: usize,
    pub early_stop_min_recurse: usize,
    pub natural_leaf_size: usize,
    pub partition_fallback_leaves: usize,

    pub partition_attempts: usize,
    pub partition_success_first: usize,
    pub partition_success_retry: usize,
    pub partition_failed_empty: usize,
    pub partition_failed_nosplit: usize,
    pub retry_escalations: usize,

    pub pre_merge_clusters_total: usize,
    pub post_merge_clusters_total: usize,
    pub empty_clusters_dropped: usize,
    pub no_op_merges: usize,

    pub leader_sum: usize,
    pub leader_max: usize,
    pub fanout_sum: usize,
    pub fanout_max: usize,

    pub assignments_before_dedup: usize,
    pub assignments_after_dedup: usize,
    pub dedup_runs: usize,
    pub dedup_skipped: usize,
    pub dedup_assignments_before: usize,
    pub dedup_assignments_after: usize,
    pub dedup_assignments_removed: usize,
    pub cur_dedup_time: Duration,
    pub cluster_assign_time: Duration,
    pub merge_time: Duration,
    pub merged_dedup_time: Duration,
    pub gemm_profile: GemmProfile,
    pub telemetry: RbcPartitionTelemetry,

    pub max_stack_size: usize,
    pub max_cluster_size_seen: usize,
    pub oversized_leaf_events: usize,
    pub oversized_leaf_points: usize,
    pub forced_leaf_splits: usize,
    pub largest_oversized_leaf: usize,

    pub leaf_sizes: Vec<usize>,
    pub start_time: Instant,
}

impl PartitionStats {
    pub(crate) fn new(max_depth: usize, expected_leaves: usize) -> Self {
        Self {
            depth_stats: (0..=max_depth).map(|_| DepthStats::new()).collect(),
            max_depth_seen: 0,
            early_stop_max_depth: 0,
            early_stop_shrink_ratio: 0,
            early_stop_small_deep: 0,
            early_stop_min_recurse: 0,
            natural_leaf_size: 0,
            partition_fallback_leaves: 0,
            partition_attempts: 0,
            partition_success_first: 0,
            partition_success_retry: 0,
            partition_failed_empty: 0,
            partition_failed_nosplit: 0,
            retry_escalations: 0,
            pre_merge_clusters_total: 0,
            post_merge_clusters_total: 0,
            empty_clusters_dropped: 0,
            no_op_merges: 0,
            leader_sum: 0,
            leader_max: 0,
            fanout_sum: 0,
            fanout_max: 0,
            assignments_before_dedup: 0,
            assignments_after_dedup: 0,
            dedup_runs: 0,
            dedup_skipped: 0,
            dedup_assignments_before: 0,
            dedup_assignments_after: 0,
            dedup_assignments_removed: 0,
            cur_dedup_time: Duration::ZERO,
            cluster_assign_time: Duration::ZERO,
            merge_time: Duration::ZERO,
            merged_dedup_time: Duration::ZERO,
            gemm_profile: GemmProfile::default(),
            telemetry: RbcPartitionTelemetry::default(),
            max_stack_size: 0,
            max_cluster_size_seen: 0,
            oversized_leaf_events: 0,
            oversized_leaf_points: 0,
            forced_leaf_splits: 0,
            largest_oversized_leaf: 0,
            leaf_sizes: Vec::with_capacity(expected_leaves),
            start_time: Instant::now(),
        }
    }

    fn ensure_depth(&mut self, depth: usize) {
        if depth >= self.depth_stats.len() {
            self.depth_stats.resize_with(depth + 1, DepthStats::new);
        }
    }

    pub(crate) fn record_leaf(&mut self, depth: usize, size: usize, reason: LeafReason) {
        self.ensure_depth(depth);
        self.max_depth_seen = self.max_depth_seen.max(depth);
        self.max_cluster_size_seen = self.max_cluster_size_seen.max(size);

        let depth_stats = &mut self.depth_stats[depth];
        depth_stats.leaves_generated += 1;
        depth_stats.total_points += size;
        depth_stats.observe_size(size);

        self.leaf_sizes.push(size);

        match reason {
            LeafReason::NaturalSize => self.natural_leaf_size += 1,
            LeafReason::MaxDepth => self.early_stop_max_depth += 1,
            LeafReason::ShrinkRatio => self.early_stop_shrink_ratio += 1,
            LeafReason::SmallDeep => self.early_stop_small_deep += 1,
            LeafReason::MinRecurse => self.early_stop_min_recurse += 1,
            LeafReason::PartitionFallback => self.partition_fallback_leaves += 1,
        }
    }

    fn record_partition_attempt(
        &mut self,
        depth: usize,
        size: usize,
        leaders: usize,
        fanout: usize,
    ) {
        self.ensure_depth(depth);
        self.partition_attempts += 1;
        self.max_depth_seen = self.max_depth_seen.max(depth);
        self.max_cluster_size_seen = self.max_cluster_size_seen.max(size);

        let depth_stats = &mut self.depth_stats[depth];
        depth_stats.clusters_processed += 1;
        depth_stats.observe_size(size);

        self.leader_sum += leaders;
        self.leader_max = self.leader_max.max(leaders);
        self.fanout_sum += fanout;
        self.fanout_max = self.fanout_max.max(fanout);
    }

    fn record_partition_result(&mut self, result: PartitionResult) {
        match result {
            PartitionResult::SuccessFirst => self.partition_success_first += 1,
            PartitionResult::SuccessRetry => self.partition_success_retry += 1,
            PartitionResult::FailedEmpty => self.partition_failed_empty += 1,
            PartitionResult::FailedNoSplit => self.partition_failed_nosplit += 1,
        }
    }

    fn record_merge(&mut self, pre: usize, post: usize, empty: usize) {
        self.pre_merge_clusters_total += pre;
        self.post_merge_clusters_total += post;
        self.empty_clusters_dropped += empty;
        if pre == post {
            self.no_op_merges += 1;
        }
    }

    fn record_retry_escalation(&mut self) {
        self.retry_escalations += 1;
    }

    pub(crate) fn record_assignments(&mut self, before_dedup: usize, after_dedup: usize) {
        self.assignments_before_dedup += before_dedup;
        self.assignments_after_dedup += after_dedup;
    }

    fn record_dedup(&mut self, executed: bool, before_dedup: usize, after_dedup: usize) {
        if executed {
            self.dedup_runs += 1;
            self.dedup_assignments_before += before_dedup;
            self.dedup_assignments_after += after_dedup;
            self.dedup_assignments_removed += before_dedup.saturating_sub(after_dedup);
        } else {
            self.dedup_skipped += 1;
        }
    }

    fn record_phase_time(&mut self, phase: RbcPhase, duration: Duration) {
        match phase {
            RbcPhase::CurDedup => self.cur_dedup_time += duration,
            RbcPhase::ClusterAssign => self.cluster_assign_time += duration,
            RbcPhase::MergeClusters => self.merge_time += duration,
            RbcPhase::MergedDedup => self.merged_dedup_time += duration,
        }
    }

    #[cfg(test)]
    pub(crate) fn record_gemm_profile(&mut self, profile: GemmProfile) {
        self.record_gemm_profile_with_context(profile, None);
    }

    fn record_gemm_profile_with_context(
        &mut self,
        profile: GemmProfile,
        assignment_context: Option<&AssignmentContext<'_>>,
    ) {
        let point_pipeline = profile.point_pipeline.clone();
        let point_pipeline_for_io = point_pipeline.clone();
        let prefetch_for_io = profile.prefetch;
        let decision_depth = profile
            .assignment_decision
            .as_ref()
            .map(|decision| decision.depth);
        if let Some(decision) = profile.assignment_decision {
            self.telemetry.assignment_decisions.record(decision);
        }
        self.gemm_profile.merge(profile);
        self.telemetry.point_pipeline.merge(point_pipeline);
        if let Some(depth) = decision_depth.filter(|_| self.telemetry.io_planned_forgeann.enabled) {
            self.telemetry
                .io_planned_forgeann
                .record_depth_point_pipeline(depth, &point_pipeline_for_io);
            self.record_depth_strict_prefetch_profile(depth, &prefetch_for_io);
        }
        if let (Some(depth), Some(context)) = (decision_depth, assignment_context) {
            context.record_depth_io_pain_from_profiles(
                depth,
                &point_pipeline_for_io,
                &prefetch_for_io,
            );
        }
    }

    fn record_depth_strict_prefetch_profile(
        &mut self,
        depth: usize,
        profile: &StrictPrefetchPipelineProfile,
    ) {
        let read_calls = profile
            .range_reads
            .saturating_add(profile.point_reads)
            .min(usize::MAX as u64) as usize;
        if profile.batches == 0
            && read_calls == 0
            && profile.logical_bytes == 0
            && profile.physical_bytes == 0
        {
            return;
        }
        self.telemetry.io_planned_forgeann.record_depth_io_pain(
            depth,
            read_calls,
            profile.batches,
            profile.logical_bytes,
            profile.physical_bytes,
            profile.io_wall.as_secs_f64() * 1000.0,
            profile.consumer_wait.as_secs_f64() * 1000.0,
            profile.producer_wait.as_secs_f64() * 1000.0,
            profile.prefetch_used_peak_bytes,
        );
    }

    pub(crate) fn record_oversized_leaf(&mut self, size: usize) {
        self.oversized_leaf_events += 1;
        self.oversized_leaf_points += size;
        self.largest_oversized_leaf = self.largest_oversized_leaf.max(size);
    }

    pub(crate) fn merge_from(&mut self, mut other: PartitionStats) {
        // depth_stats: extend or merge per-depth
        for (depth, ds) in other.depth_stats.into_iter().enumerate() {
            self.ensure_depth(depth);
            let mine = &mut self.depth_stats[depth];
            mine.clusters_processed += ds.clusters_processed;
            mine.leaves_generated += ds.leaves_generated;
            mine.total_points += ds.total_points;
            if ds.min_cluster_size < mine.min_cluster_size {
                mine.min_cluster_size = ds.min_cluster_size;
            }
            if ds.max_cluster_size > mine.max_cluster_size {
                mine.max_cluster_size = ds.max_cluster_size;
            }
        }
        self.max_depth_seen = self.max_depth_seen.max(other.max_depth_seen);
        self.early_stop_max_depth += other.early_stop_max_depth;
        self.early_stop_shrink_ratio += other.early_stop_shrink_ratio;
        self.early_stop_small_deep += other.early_stop_small_deep;
        self.early_stop_min_recurse += other.early_stop_min_recurse;
        self.natural_leaf_size += other.natural_leaf_size;
        self.partition_fallback_leaves += other.partition_fallback_leaves;
        self.partition_attempts += other.partition_attempts;
        self.partition_success_first += other.partition_success_first;
        self.partition_success_retry += other.partition_success_retry;
        self.partition_failed_empty += other.partition_failed_empty;
        self.partition_failed_nosplit += other.partition_failed_nosplit;
        self.retry_escalations += other.retry_escalations;
        self.pre_merge_clusters_total += other.pre_merge_clusters_total;
        self.post_merge_clusters_total += other.post_merge_clusters_total;
        self.empty_clusters_dropped += other.empty_clusters_dropped;
        self.no_op_merges += other.no_op_merges;
        self.leader_sum += other.leader_sum;
        self.leader_max = self.leader_max.max(other.leader_max);
        self.fanout_sum += other.fanout_sum;
        self.fanout_max = self.fanout_max.max(other.fanout_max);
        self.assignments_before_dedup += other.assignments_before_dedup;
        self.assignments_after_dedup += other.assignments_after_dedup;
        self.dedup_runs += other.dedup_runs;
        self.dedup_skipped += other.dedup_skipped;
        self.dedup_assignments_before += other.dedup_assignments_before;
        self.dedup_assignments_after += other.dedup_assignments_after;
        self.dedup_assignments_removed += other.dedup_assignments_removed;
        self.cur_dedup_time += other.cur_dedup_time;
        self.cluster_assign_time += other.cluster_assign_time;
        self.merge_time += other.merge_time;
        self.merged_dedup_time += other.merged_dedup_time;
        self.gemm_profile.merge(other.gemm_profile);
        self.telemetry.merge(other.telemetry);
        self.max_stack_size = self.max_stack_size.max(other.max_stack_size);
        self.max_cluster_size_seen = self.max_cluster_size_seen.max(other.max_cluster_size_seen);
        self.oversized_leaf_events += other.oversized_leaf_events;
        self.oversized_leaf_points += other.oversized_leaf_points;
        self.forced_leaf_splits += other.forced_leaf_splits;
        self.largest_oversized_leaf = self
            .largest_oversized_leaf
            .max(other.largest_oversized_leaf);
        self.leaf_sizes.append(&mut other.leaf_sizes);
    }

    pub(crate) fn total_leaves(&self) -> usize {
        self.leaf_sizes.len()
    }

    pub(crate) fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub(crate) fn total_points_in_leaves(&self) -> usize {
        self.leaf_sizes.iter().sum()
    }

    pub(crate) fn overlap_ratio(&self) -> f64 {
        percentage(
            self.assignments_before_dedup
                .saturating_sub(self.assignments_after_dedup),
            self.assignments_before_dedup,
        )
    }
}

pub(crate) fn rbc_partition_streaming(
    dataset: &dyn PointStore,
    adsampling_dataset: Option<&dyn PointStore>,
    indices: &[u32],
    metric: Metric,
    params: &ForgeANNParams,
    num_threads: u32,
    rng: &mut (impl Rng + Send),
    external_child_dir: Option<&Path>,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<RbcPartitionTelemetry> {
    rbc_partition_streaming_with_dirs(
        dataset,
        adsampling_dataset,
        indices,
        metric,
        params,
        num_threads,
        rng,
        external_child_dir,
        None,
        leaf_emitter,
    )
}

pub(crate) fn rbc_partition_streaming_with_dirs(
    dataset: &dyn PointStore,
    adsampling_dataset: Option<&dyn PointStore>,
    indices: &[u32],
    metric: Metric,
    params: &ForgeANNParams,
    num_threads: u32,
    rng: &mut (impl Rng + Send),
    external_child_dir: Option<&Path>,
    external_vector_dir: Option<&Path>,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<RbcPartitionTelemetry> {
    let external_run_store = external_child_dir.map(|dir| {
        let io_cfg = if params.strict_oom_io_enabled() {
            DirectIoConfig::enabled_with_alignment(4096)
        } else {
            DirectIoConfig::disabled()
        };
        let mut store = if let Some(vector_dir) = external_vector_dir {
            ExternalRunStore::new_with_vector_dir(dir, vector_dir, io_cfg)
        } else {
            ExternalRunStore::new(dir, io_cfg)
        };
        if params.io_planned_forgeann_enabled() && params.io_plan_temp_budget_bytes() > 0 {
            let budget = std::sync::Arc::new(
                crate::forgeann::io_planned_forgeann::IoPlanBudgetState::new(
                    params.io_plan_temp_budget_bytes() as u64,
                ),
            );
            store = store.with_io_plan_budget(budget);
        }
        Arc::new(Mutex::new(store))
    });
    let manifest_path = if let Some(manifest_path) = d0_runs_manifest_path_from_env() {
        Some(manifest_path)
    } else if d0_runs_cache_dir_from_env().is_some() {
        if external_run_store.is_none() {
            return Err(AnnError::log_index_error(format!(
                "{D0_RUNS_CACHE_DIR_ENV} requires an external child-run directory"
            )));
        }
        let Some(plan) = prepare_rbc_root_partition_plan(
            dataset,
            adsampling_dataset,
            indices,
            metric,
            params,
            rng,
        )?
        else {
            return Ok(RbcPartitionTelemetry::default());
        };
        let cache_manifest = d0_runs_cache_manifest_path(
            dataset,
            metric,
            params,
            plan.total_size,
            plan.adaptive_c_max,
            plan.min_recurse_size,
            plan.seed,
            &plan.root_fanout_state.profile(),
        )?;
        if cache_manifest.exists() {
            tracing::info!(
                "[rbc/d0-cache-hit] manifest={} root_seed={} root_leader_hash={}",
                cache_manifest.display(),
                plan.seed,
                plan.root_fanout_state.profile().root_leader_hash,
            );
            return rbc_partition_with_reused_d0_plan(
                dataset,
                indices,
                metric,
                params,
                external_run_store.as_ref(),
                leaf_emitter,
                &cache_manifest,
                plan,
            );
        }
        tracing::info!(
            "[rbc/d0-cache-miss] manifest={} root_seed={} root_leader_hash={}",
            cache_manifest.display(),
            plan.seed,
            plan.root_fanout_state.profile().root_leader_hash,
        );
        return execute_rbc_root_partition_plan(
            dataset,
            indices,
            metric,
            params,
            external_run_store,
            leaf_emitter,
            plan,
        );
    } else {
        None
    };

    if let Some(manifest_path) = manifest_path {
        let Some(plan) = prepare_rbc_root_partition_plan(
            dataset,
            adsampling_dataset,
            indices,
            metric,
            params,
            rng,
        )?
        else {
            return Ok(RbcPartitionTelemetry::default());
        };
        return rbc_partition_with_reused_d0_plan(
            dataset,
            indices,
            metric,
            params,
            external_run_store.as_ref(),
            leaf_emitter,
            &manifest_path,
            plan,
        );
    }
    rbc_partition_impl_streaming(
        dataset,
        adsampling_dataset,
        indices,
        metric,
        params,
        num_threads,
        rng,
        external_run_store,
        leaf_emitter,
    )
}

fn rbc_partition_with_reused_d0_plan(
    dataset: &dyn PointStore,
    _indices: &[u32],
    metric: Metric,
    params: &ForgeANNParams,
    external_run_store: Option<&Arc<Mutex<ExternalRunStore>>>,
    leaf_emitter: &dyn LeafEmitter,
    manifest_path: &Path,
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
    let external_run_store = external_run_store.ok_or_else(|| {
        AnnError::log_index_error(format!(
            "{D0_RUNS_MANIFEST_ENV} or {D0_RUNS_CACHE_DIR_ENV} requires an external child-run directory"
        ))
    })?;
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .unwrap(),
    );
    pb.set_message(format!(
        "reused-d0 depth=1 c_max={} min_recurse={}",
        format_count(adaptive_c_max),
        format_count(min_recurse_size),
    ));
    let root_profile = root_fanout_state.profile();
    let (child_runs, root_leaves, manifest_root_profile) = load_reused_d0_child_runs(
        manifest_path,
        dataset,
        metric,
        params,
        total_size,
        adaptive_c_max,
        min_recurse_size,
        seed,
        &root_profile,
        external_run_store,
    )?;
    let root_leaf_stats = emit_reused_d0_root_leaves(
        dataset,
        metric,
        params,
        &root_leaves,
        external_run_store,
        leaf_emitter,
    )?;
    let child_context = assignment_context.for_depth(1);
    tracing::info!(
        "[rbc/d0-reuse-start] manifest={} child_runs={} root_leaf_count={} root_leaf_points={} depth=1 parent_n={} adaptive_c_max={} min_recurse={}",
        manifest_path.display(),
        child_runs.len(),
        root_leaves.len(),
        root_leaves.iter().map(|leaf| leaf.len).sum::<usize>(),
        total_size,
        adaptive_c_max,
        min_recurse_size,
    );
    let mut stats = parallel_join_child_runs(
        dataset,
        child_runs,
        1,
        total_size,
        metric,
        params,
        adaptive_c_max,
        min_recurse_size,
        &pb,
        external_run_store,
        &root_fanout_state,
        &child_context,
        leaf_emitter,
    )?;
    stats.merge_from(root_leaf_stats);
    stats.max_cluster_size_seen = stats.max_cluster_size_seen.max(total_size);
    stats.telemetry.root_fanout = manifest_root_profile;
    stats.telemetry.ads_scheduler = assignment_context.ads_scheduler_stats();
    assignment_context.merge_observed_io_pain_into(&mut stats.telemetry.io_planned_forgeann);
    stats
        .leaf_sizes
        .reserve(expected_leaves.saturating_sub(stats.leaf_sizes.len()));
    pb.finish_and_clear();
    tracing::info!(
        "[rbc/d0-reuse-done] manifest={} elapsed_ms={} leaves={} total_leaf_points={}",
        manifest_path.display(),
        stats.elapsed().as_millis(),
        stats.leaf_sizes.len(),
        stats.total_points_in_leaves(),
    );
    log_final_summary(&stats, total_size, adaptive_c_max, min_recurse_size);
    Ok(stats.telemetry)
}

/// Per-worker scratch buffers for GEMM-based cluster assignment.
/// Per-worker scratch buffers for GEMM-based cluster assignment.
/// Reused across point-tile chunks to avoid repeated large heap allocations.
pub(crate) struct GemmScratch {
    pub x_data: Vec<f32>,
    pub x_norms: Vec<f32>,
    pub gram_data: Vec<f32>,
    pub topk_rows: Vec<StackTopK>,
    pub local_clusters: Vec<Vec<u32>>,
}

impl GemmScratch {
    pub(crate) fn new(
        point_tile: usize,
        leader_tile: usize,
        dim: usize,
        nl: usize,
        fanout: usize,
    ) -> Self {
        Self {
            x_data: Vec::with_capacity(point_tile * dim),
            x_norms: Vec::with_capacity(point_tile),
            gram_data: Vec::with_capacity(point_tile * leader_tile),
            topk_rows: (0..point_tile).map(|_| StackTopK::new(fanout)).collect(),
            local_clusters: (0..nl).map(|_| Vec::new()).collect(),
        }
    }
}

pub(crate) struct GemmFoldState {
    pub scratch: GemmScratch,
    pub clusters: Vec<Vec<u32>>,
    pub profile: GemmProfile,
}

pub(crate) fn default_rbc_windowed_options(
    dataset: &dyn PointStore,
    tile_rows: usize,
) -> WindowedGatherOptions {
    let row_bytes = dataset.dim().saturating_mul(size_of::<f32>());
    let mut max_window_bytes = row_bytes
        .saturating_mul(tile_rows.max(1))
        .min(4 * 1024 * 1024)
        .max(256 * 1024);
    let mut max_gap_rows = if row_bytes <= 4096 { 1 } else { 0 };
    if dataset.prefers_coalesced_window_reads() {
        max_window_bytes = env_usize("FORGEANN_RBC_MAX_WINDOW_BYTES")
            .or_else(|| env_usize("FORGEANN_STRICT_GATHER_MAX_WINDOW_BYTES"))
            .unwrap_or(max_window_bytes)
            .max(max_window_bytes);
        max_gap_rows = env_usize("FORGEANN_RBC_MAX_GAP_ROWS")
            .or_else(|| env_usize("FORGEANN_STRICT_GATHER_MAX_GAP_ROWS"))
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(4);
    }
    WindowedGatherOptions {
        max_gap_rows,
        max_window_bytes,
        alignment_bytes: 4096,
        sort_ids: true,
    }
}

pub(crate) fn enter_large_assignment_guard<'a>(
    leaf_emitter: &'a dyn LeafEmitter,
    active: bool,
) -> Option<LargeAssignmentGuard<'a>> {
    active
        .then(|| leaf_emitter.scheduler_signals())
        .flatten()
        .map(LargeAssignmentGuard::enter)
}

pub(crate) fn enter_leaf_drainer_limit_guard<'a>(
    leaf_emitter: &'a dyn LeafEmitter,
    limit: usize,
    backpressure_limit: usize,
) -> Option<LeafDrainerLimitGuard<'a>> {
    leaf_emitter.scheduler_signals().map(|signals| {
        LeafDrainerLimitGuard::enter_with_backpressure_limit(signals, limit, backpressure_limit)
    })
}

pub(crate) fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn env_bool(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
}

fn env_bytes_from_gib(name: &str) -> Option<usize> {
    env_f64(name)
        .filter(|value| *value >= 0.0)
        .map(|value| (value * 1024.0 * 1024.0 * 1024.0).round() as usize)
}

const D0_RUNS_MANIFEST_ENV: &str = "FORGEANN_REUSE_D0_RUNS_MANIFEST";
const D0_RUNS_CACHE_DIR_ENV: &str = "FORGEANN_D0_RUNS_CACHE_DIR";
const D0_RUNS_MANIFEST_FILE: &str = "partition_d00_runs.manifest.json";
const D0_RUNS_FILE: &str = "partition_d00_runs.bin";
const D0_RUNS_MANIFEST_FORMAT: &str = "forgeann-d0-runs";
const D0_RUNS_MANIFEST_VERSION: u32 = 1;

#[derive(Debug)]
struct D0CapturedRootLeaf {
    points: Vec<u32>,
    depth: usize,
    reason: LeafReason,
}

#[derive(Default, Debug)]
struct D0RootLeafRecorder {
    leaves: Mutex<Vec<D0CapturedRootLeaf>>,
}

impl D0RootLeafRecorder {
    fn record(&self, points: Vec<u32>, depth: usize, reason: LeafReason) {
        if points.is_empty() {
            return;
        }
        self.leaves.lock().push(D0CapturedRootLeaf {
            points,
            depth,
            reason,
        });
    }

    fn take(&self) -> Vec<D0CapturedRootLeaf> {
        std::mem::take(&mut *self.leaves.lock())
    }
}

#[derive(Clone, Debug)]
struct D0RootLeafRun {
    len: usize,
    depth: usize,
    reason: LeafReason,
    extents: Vec<RunExtent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct D0RunsManifest {
    format: String,
    version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dataset_identity: Option<String>,
    num_points: usize,
    dim: usize,
    metric: String,
    random_seed: u64,
    root_seed: u64,
    c_min: usize,
    c_max: usize,
    max_depth: usize,
    max_leaders: usize,
    fanout_top: usize,
    fanout_second: usize,
    psamp_fraction_bits: u64,
    min_shrink_ratio_bits: u32,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    root_leaf_count: usize,
    root_leaf_points: usize,
    #[serde(default)]
    root_leaves: Vec<D0RunsManifestRootLeaf>,
    root_fanout: RootFanoutProfile,
    run_file: String,
    run_file_bytes: u64,
    run_file_sha256: String,
    child_runs: Vec<D0RunsManifestChild>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct D0RunsManifestChild {
    len: usize,
    seed: u64,
    extents: Vec<D0RunsManifestExtent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct D0RunsManifestRootLeaf {
    len: usize,
    depth: usize,
    reason: String,
    extents: Vec<D0RunsManifestExtent>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct D0RunsManifestExtent {
    byte_offset: u64,
    len: usize,
}

fn metric_label(metric: Metric) -> &'static str {
    match metric {
        Metric::L2 => "l2",
        Metric::Cosine => "cosine",
        Metric::Ip => "ip",
    }
}

fn leaf_reason_label(reason: LeafReason) -> &'static str {
    match reason {
        LeafReason::NaturalSize => "natural_size",
        LeafReason::MaxDepth => "max_depth",
        LeafReason::ShrinkRatio => "shrink_ratio",
        LeafReason::SmallDeep => "small_deep",
        LeafReason::MinRecurse => "min_recurse",
        LeafReason::PartitionFallback => "partition_fallback",
    }
}

fn leaf_reason_from_label(label: &str) -> Option<LeafReason> {
    match label {
        "natural_size" => Some(LeafReason::NaturalSize),
        "max_depth" => Some(LeafReason::MaxDepth),
        "shrink_ratio" => Some(LeafReason::ShrinkRatio),
        "small_deep" => Some(LeafReason::SmallDeep),
        "min_recurse" => Some(LeafReason::MinRecurse),
        "partition_fallback" => Some(LeafReason::PartitionFallback),
        _ => None,
    }
}

fn emit_leaf_with_d0_root_capture(
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
    root_leaf_recorder: Option<&D0RootLeafRecorder>,
) -> AnnResult<()> {
    if let Some(recorder) = root_leaf_recorder {
        let mut persisted = leaf.clone();
        if dedup {
            persisted.sort_unstable();
            persisted.dedup();
        }
        recorder.record(persisted, depth, reason);
    }
    emit_leaf(
        dataset,
        metric,
        params,
        leaf,
        depth,
        reason,
        dedup,
        max_leaf_size,
        stats,
        leaf_emitter,
    )
}

fn d0_runs_manifest_path_from_env() -> Option<PathBuf> {
    std::env::var_os(D0_RUNS_MANIFEST_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn d0_runs_cache_dir_from_env() -> Option<PathBuf> {
    std::env::var_os(D0_RUNS_CACHE_DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn d0_runs_cache_key(
    dataset_identity: &str,
    metric: Metric,
    params: &ForgeANNParams,
    total_size: usize,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    root_seed: u64,
    root_fanout: &RootFanoutProfile,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"forgeann-d0-runs-cache-v1\0");
    hasher.update(dataset_identity.as_bytes());
    hasher.update([0]);
    hasher.update(metric_label(metric).as_bytes());
    hasher.update(total_size.to_le_bytes());
    hasher.update(params.c_min.to_le_bytes());
    hasher.update(params.c_max.to_le_bytes());
    hasher.update(params.max_depth.to_le_bytes());
    hasher.update(params.max_leaders.to_le_bytes());
    hasher.update(params.fanout_top.to_le_bytes());
    hasher.update(params.fanout_second.to_le_bytes());
    hasher.update(params.psamp_fraction.to_bits().to_le_bytes());
    hasher.update(params.min_shrink_ratio.to_bits().to_le_bytes());
    hasher.update(params.random_seed.to_le_bytes());
    hasher.update(adaptive_c_max.to_le_bytes());
    hasher.update(min_recurse_size.to_le_bytes());
    hasher.update(root_seed.to_le_bytes());
    hasher.update(root_fanout.fixed_fanout.to_le_bytes());
    hasher.update(root_fanout.root_leader_hash.to_le_bytes());
    format!("{:x}", hasher.finalize())
}

fn d0_runs_cache_manifest_path_for_dir(
    cache_dir: &Path,
    dataset_identity: &str,
    metric: Metric,
    params: &ForgeANNParams,
    total_size: usize,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    root_seed: u64,
    root_fanout: &RootFanoutProfile,
) -> PathBuf {
    cache_dir
        .join(d0_runs_cache_key(
            dataset_identity,
            metric,
            params,
            total_size,
            adaptive_c_max,
            min_recurse_size,
            root_seed,
            root_fanout,
        ))
        .join(D0_RUNS_MANIFEST_FILE)
}

fn d0_runs_cache_manifest_path(
    dataset: &dyn PointStore,
    metric: Metric,
    params: &ForgeANNParams,
    total_size: usize,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    root_seed: u64,
    root_fanout: &RootFanoutProfile,
) -> AnnResult<PathBuf> {
    let cache_dir = d0_runs_cache_dir_from_env()
        .ok_or_else(|| AnnError::log_index_error(format!("{D0_RUNS_CACHE_DIR_ENV} is not set")))?;
    let dataset_identity = dataset.source_identity().ok_or_else(|| {
        AnnError::log_index_error(format!(
            "{D0_RUNS_CACHE_DIR_ENV} requires a point store with stable source identity"
        ))
    })?;
    Ok(d0_runs_cache_manifest_path_for_dir(
        &cache_dir,
        &dataset_identity,
        metric,
        params,
        total_size,
        adaptive_c_max,
        min_recurse_size,
        root_seed,
        root_fanout,
    ))
}

fn sha256_file_hex(path: &Path) -> AnnResult<String> {
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn resolve_manifest_run_file(manifest_path: &Path, manifest: &D0RunsManifest) -> PathBuf {
    let run_file = Path::new(&manifest.run_file);
    if run_file.is_absolute() {
        run_file.to_path_buf()
    } else {
        manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(run_file)
    }
}

fn install_reused_d0_run_file(
    source: &Path,
    target_dir: &Path,
    manifest: &D0RunsManifest,
) -> AnnResult<PathBuf> {
    let source_meta = fs::metadata(source)?;
    if source_meta.len() != manifest.run_file_bytes {
        return Err(AnnError::log_index_error(format!(
            "D0 reuse run file size mismatch: path={} expected_bytes={} actual_bytes={}",
            source.display(),
            manifest.run_file_bytes,
            source_meta.len()
        )));
    }
    let source_hash = sha256_file_hex(source)?;
    if source_hash != manifest.run_file_sha256 {
        return Err(AnnError::log_index_error(format!(
            "D0 reuse run file hash mismatch: path={} expected_sha256={} actual_sha256={}",
            source.display(),
            manifest.run_file_sha256,
            source_hash
        )));
    }

    fs::create_dir_all(target_dir)?;
    let target = target_dir.join(D0_RUNS_FILE);
    if target == source {
        return Ok(target);
    }
    if target.exists() {
        let target_meta = fs::metadata(&target)?;
        if target_meta.len() == manifest.run_file_bytes
            && sha256_file_hex(&target)? == manifest.run_file_sha256
        {
            return Ok(target);
        }
        return Err(AnnError::log_index_error(format!(
            "D0 reuse target already exists with different contents: {}",
            target.display()
        )));
    }

    match fs::hard_link(source, &target) {
        Ok(()) => Ok(target),
        Err(hard_link_err) => match std::os::unix::fs::symlink(source, &target) {
            Ok(()) => Ok(target),
            Err(symlink_err) => Err(AnnError::log_index_error(format!(
                "Failed to link reused D0 run file from {} to {}: hard_link={hard_link_err} symlink={symlink_err}",
                source.display(),
                target.display()
            ))),
        },
    }
}

fn install_d0_cache_run_file(
    source: &Path,
    target_dir: &Path,
    manifest: &D0RunsManifest,
) -> AnnResult<PathBuf> {
    let source_meta = fs::metadata(source)?;
    if source_meta.len() != manifest.run_file_bytes {
        return Err(AnnError::log_index_error(format!(
            "D0 cache source run file size mismatch: path={} expected_bytes={} actual_bytes={}",
            source.display(),
            manifest.run_file_bytes,
            source_meta.len()
        )));
    }
    let source_hash = sha256_file_hex(source)?;
    if source_hash != manifest.run_file_sha256 {
        return Err(AnnError::log_index_error(format!(
            "D0 cache source run file hash mismatch: path={} expected_sha256={} actual_sha256={}",
            source.display(),
            manifest.run_file_sha256,
            source_hash
        )));
    }

    fs::create_dir_all(target_dir)?;
    let target = target_dir.join(D0_RUNS_FILE);
    if target == source {
        return Ok(target);
    }
    if target.exists() {
        let target_meta = fs::metadata(&target)?;
        if target_meta.len() == manifest.run_file_bytes
            && sha256_file_hex(&target)? == manifest.run_file_sha256
        {
            return Ok(target);
        }
        return Err(AnnError::log_index_error(format!(
            "D0 cache run file already exists with different contents: {}",
            target.display()
        )));
    }

    match fs::hard_link(source, &target) {
        Ok(()) => {}
        Err(hard_link_err) => {
            tracing::warn!(
                "[rbc/d0-cache-copy] source={} target={} reason=hard_link_failed error={}",
                source.display(),
                target.display(),
                hard_link_err,
            );
            fs::copy(source, &target)?;
        }
    }
    let target_meta = fs::metadata(&target)?;
    if target_meta.len() != manifest.run_file_bytes
        || sha256_file_hex(&target)? != manifest.run_file_sha256
    {
        return Err(AnnError::log_index_error(format!(
            "D0 cache run file verification failed after install: {}",
            target.display()
        )));
    }
    Ok(target)
}

fn validate_d0_manifest_extents(
    manifest_file_bytes: u64,
    extents: &[D0RunsManifestExtent],
    len: usize,
    label: &str,
    mismatches: &mut Vec<String>,
) {
    let extent_len_sum = extents.iter().map(|extent| extent.len).sum::<usize>();
    if extent_len_sum != len {
        mismatches.push(format!("{label} len={len} extent_len_sum={extent_len_sum}"));
    }
    for extent in extents {
        let bytes =
            RUN_EXTENT_HEADER_BYTES.saturating_add(extent.len.saturating_mul(size_of::<u32>()));
        let Some(end) = extent.byte_offset.checked_add(bytes as u64) else {
            mismatches.push(format!("{label} extent overflow"));
            continue;
        };
        if end > manifest_file_bytes {
            mismatches.push(format!(
                "{label} extent_end={end} run_file_bytes={manifest_file_bytes}"
            ));
        }
    }
}

fn validate_d0_runs_manifest(
    manifest: &D0RunsManifest,
    manifest_path: &Path,
    dataset: &dyn PointStore,
    metric: Metric,
    params: &ForgeANNParams,
    total_size: usize,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    root_seed: u64,
    root_fanout: &RootFanoutProfile,
) -> AnnResult<()> {
    let mut mismatches = Vec::new();
    if manifest.format != D0_RUNS_MANIFEST_FORMAT {
        mismatches.push(format!("format={}", manifest.format));
    }
    if manifest.version != D0_RUNS_MANIFEST_VERSION {
        mismatches.push(format!("version={}", manifest.version));
    }
    if let Some(manifest_identity) = &manifest.dataset_identity {
        match dataset.source_identity() {
            Some(current_identity) if &current_identity == manifest_identity => {}
            Some(current_identity) => mismatches.push(format!(
                "dataset_identity manifest={} current={}",
                manifest_identity, current_identity
            )),
            None => mismatches.push("dataset_identity_unavailable".to_string()),
        }
    }
    if manifest.num_points != total_size {
        mismatches.push(format!(
            "num_points manifest={} current={}",
            manifest.num_points, total_size
        ));
    }
    if manifest.dim != dataset.dim() {
        mismatches.push(format!(
            "dim manifest={} current={}",
            manifest.dim,
            dataset.dim()
        ));
    }
    if manifest.metric != metric_label(metric) {
        mismatches.push(format!(
            "metric manifest={} current={}",
            manifest.metric,
            metric_label(metric)
        ));
    }
    if manifest.random_seed != params.random_seed {
        mismatches.push(format!(
            "random_seed manifest={} current={}",
            manifest.random_seed, params.random_seed
        ));
    }
    if manifest.root_seed != root_seed {
        mismatches.push(format!(
            "root_seed manifest={} current={}",
            manifest.root_seed, root_seed
        ));
    }
    if manifest.c_min != params.c_min {
        mismatches.push(format!(
            "c_min manifest={} current={}",
            manifest.c_min, params.c_min
        ));
    }
    if manifest.c_max != params.c_max {
        mismatches.push(format!(
            "c_max manifest={} current={}",
            manifest.c_max, params.c_max
        ));
    }
    if manifest.max_depth != params.max_depth {
        mismatches.push(format!(
            "max_depth manifest={} current={}",
            manifest.max_depth, params.max_depth
        ));
    }
    if manifest.max_leaders != params.max_leaders {
        mismatches.push(format!(
            "max_leaders manifest={} current={}",
            manifest.max_leaders, params.max_leaders
        ));
    }
    if manifest.fanout_top != params.fanout_top {
        mismatches.push(format!(
            "fanout_top manifest={} current={}",
            manifest.fanout_top, params.fanout_top
        ));
    }
    if manifest.fanout_second != params.fanout_second {
        mismatches.push(format!(
            "fanout_second manifest={} current={}",
            manifest.fanout_second, params.fanout_second
        ));
    }
    if manifest.psamp_fraction_bits != params.psamp_fraction.to_bits() {
        mismatches.push("psamp_fraction_bits".to_string());
    }
    if manifest.min_shrink_ratio_bits != params.min_shrink_ratio.to_bits() {
        mismatches.push("min_shrink_ratio_bits".to_string());
    }
    if manifest.adaptive_c_max != adaptive_c_max {
        mismatches.push(format!(
            "adaptive_c_max manifest={} current={}",
            manifest.adaptive_c_max, adaptive_c_max
        ));
    }
    if manifest.min_recurse_size != min_recurse_size {
        mismatches.push(format!(
            "min_recurse_size manifest={} current={}",
            manifest.min_recurse_size, min_recurse_size
        ));
    }
    if manifest.root_leaf_count != manifest.root_leaves.len() {
        mismatches.push(format!(
            "root_leaf_count manifest={} actual={}",
            manifest.root_leaf_count,
            manifest.root_leaves.len()
        ));
    }
    let root_leaf_points = manifest
        .root_leaves
        .iter()
        .map(|leaf| leaf.len)
        .sum::<usize>();
    if manifest.root_leaf_points != root_leaf_points {
        mismatches.push(format!(
            "root_leaf_points manifest={} actual={root_leaf_points}",
            manifest.root_leaf_points
        ));
    }
    if manifest.root_fanout.fixed_fanout != root_fanout.fixed_fanout
        || manifest.root_fanout.root_leader_hash != root_fanout.root_leader_hash
    {
        mismatches.push(format!(
            "root_fanout manifest_fixed={} current_fixed={} manifest_hash={} current_hash={}",
            manifest.root_fanout.fixed_fanout,
            root_fanout.fixed_fanout,
            manifest.root_fanout.root_leader_hash,
            root_fanout.root_leader_hash
        ));
    }
    if manifest.child_runs.is_empty() {
        mismatches.push("child_runs_empty".to_string());
    }

    for (run_idx, run) in manifest.child_runs.iter().enumerate() {
        validate_d0_manifest_extents(
            manifest.run_file_bytes,
            &run.extents,
            run.len,
            &format!("child_run[{run_idx}]"),
            &mut mismatches,
        );
    }

    for (leaf_idx, leaf) in manifest.root_leaves.iter().enumerate() {
        if leaf_reason_from_label(&leaf.reason).is_none() {
            mismatches.push(format!(
                "root_leaf[{leaf_idx}] unknown_reason={}",
                leaf.reason
            ));
        }
        validate_d0_manifest_extents(
            manifest.run_file_bytes,
            &leaf.extents,
            leaf.len,
            &format!("root_leaf[{leaf_idx}]"),
            &mut mismatches,
        );
    }

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(AnnError::log_index_error(format!(
            "D0 reuse manifest is incompatible: path={} mismatches={}",
            manifest_path.display(),
            mismatches.join(", ")
        )))
    }
}

fn append_d0_root_leaves(
    store: &mut ExternalRunStore,
    leaves: Vec<D0CapturedRootLeaf>,
) -> AnnResult<Vec<D0RootLeafRun>> {
    let mut root_leaves = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        let extent = store.append_points(0, &leaf.points)?;
        root_leaves.push(D0RootLeafRun {
            len: leaf.points.len(),
            depth: leaf.depth,
            reason: leaf.reason,
            extents: vec![extent],
        });
    }
    Ok(root_leaves)
}

fn write_d0_runs_manifest(
    base_dir: &Path,
    dataset: &dyn PointStore,
    metric: Metric,
    params: &ForgeANNParams,
    total_size: usize,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    root_seed: u64,
    root_fanout: RootFanoutProfile,
    child_runs: &[ChildRun],
    root_leaves: &[D0RootLeafRun],
) -> AnnResult<Option<PathBuf>> {
    if child_runs.is_empty() {
        tracing::warn!("[rbc/d0-reuse-manifest-skip] reason=no-child-runs");
        return Ok(None);
    }

    let run_path = base_dir.join(D0_RUNS_FILE);
    let run_file_bytes = fs::metadata(&run_path)?.len();
    let run_file_sha256 = sha256_file_hex(&run_path)?;
    let root_leaf_count = root_leaves.len();
    let root_leaf_points = root_leaves.iter().map(|leaf| leaf.len).sum::<usize>();
    let manifest = D0RunsManifest {
        format: D0_RUNS_MANIFEST_FORMAT.to_string(),
        version: D0_RUNS_MANIFEST_VERSION,
        dataset_identity: dataset.source_identity(),
        num_points: total_size,
        dim: dataset.dim(),
        metric: metric_label(metric).to_string(),
        random_seed: params.random_seed,
        root_seed,
        c_min: params.c_min,
        c_max: params.c_max,
        max_depth: params.max_depth,
        max_leaders: params.max_leaders,
        fanout_top: params.fanout_top,
        fanout_second: params.fanout_second,
        psamp_fraction_bits: params.psamp_fraction.to_bits(),
        min_shrink_ratio_bits: params.min_shrink_ratio.to_bits(),
        adaptive_c_max,
        min_recurse_size,
        root_leaf_count,
        root_leaf_points,
        root_leaves: root_leaves
            .iter()
            .map(|leaf| D0RunsManifestRootLeaf {
                len: leaf.len,
                depth: leaf.depth,
                reason: leaf_reason_label(leaf.reason).to_string(),
                extents: leaf
                    .extents
                    .iter()
                    .map(|extent| D0RunsManifestExtent {
                        byte_offset: extent.byte_offset,
                        len: extent.len,
                    })
                    .collect(),
            })
            .collect(),
        root_fanout,
        run_file: D0_RUNS_FILE.to_string(),
        run_file_bytes,
        run_file_sha256,
        child_runs: child_runs
            .iter()
            .map(|run| D0RunsManifestChild {
                len: run.len,
                seed: run.seed,
                extents: run
                    .extents
                    .iter()
                    .map(|extent| D0RunsManifestExtent {
                        byte_offset: extent.byte_offset,
                        len: extent.len,
                    })
                    .collect(),
            })
            .collect(),
    };
    let manifest_path = base_dir.join(D0_RUNS_MANIFEST_FILE);
    let manifest_json = serde_json::to_vec_pretty(&manifest).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to encode D0 reuse manifest {}: {err}",
            manifest_path.display()
        ))
    })?;
    fs::write(&manifest_path, &manifest_json)?;
    publish_d0_runs_manifest_to_cache(base_dir, &manifest, &manifest_json)?;
    tracing::info!(
        "[rbc/d0-reuse-manifest] path={} run_file={} run_file_bytes={} child_runs={} root_leaf_count={} root_leaf_points={} root_seed={} root_leader_hash={}",
        manifest_path.display(),
        run_path.display(),
        run_file_bytes,
        child_runs.len(),
        manifest.root_leaf_count,
        manifest.root_leaf_points,
        root_seed,
        manifest.root_fanout.root_leader_hash,
    );
    Ok(Some(manifest_path))
}

fn publish_d0_runs_manifest_to_cache(
    base_dir: &Path,
    manifest: &D0RunsManifest,
    manifest_json: &[u8],
) -> AnnResult<()> {
    let Some(cache_dir) = d0_runs_cache_dir_from_env() else {
        return Ok(());
    };
    publish_d0_runs_manifest_to_cache_dir(&cache_dir, base_dir, manifest, manifest_json)
}

fn publish_d0_runs_manifest_to_cache_dir(
    cache_dir: &Path,
    base_dir: &Path,
    manifest: &D0RunsManifest,
    manifest_json: &[u8],
) -> AnnResult<()> {
    let Some(dataset_identity) = manifest.dataset_identity.as_deref() else {
        return Err(AnnError::log_index_error(format!(
            "{D0_RUNS_CACHE_DIR_ENV} requires a point store with stable source identity"
        )));
    };
    let cache_manifest_path = d0_runs_cache_manifest_path_for_dir(
        &cache_dir,
        dataset_identity,
        match manifest.metric.as_str() {
            "l2" => Metric::L2,
            "cosine" => Metric::Cosine,
            "ip" => Metric::Ip,
            other => {
                return Err(AnnError::log_index_error(format!(
                    "Unsupported D0 manifest metric for cache publish: {other}"
                )));
            }
        },
        &ForgeANNParams {
            c_min: manifest.c_min,
            c_max: manifest.c_max,
            max_depth: manifest.max_depth,
            max_leaders: manifest.max_leaders,
            fanout_top: manifest.fanout_top,
            fanout_second: manifest.fanout_second,
            psamp_fraction: f64::from_bits(manifest.psamp_fraction_bits),
            min_shrink_ratio: f32::from_bits(manifest.min_shrink_ratio_bits),
            random_seed: manifest.random_seed,
            ..ForgeANNParams::default()
        },
        manifest.num_points,
        manifest.adaptive_c_max,
        manifest.min_recurse_size,
        manifest.root_seed,
        &manifest.root_fanout,
    );
    let cache_entry_dir = cache_manifest_path
        .parent()
        .ok_or_else(|| AnnError::log_index_error("D0 cache manifest has no parent".to_string()))?;
    fs::create_dir_all(cache_entry_dir)?;
    let source_run_file = base_dir.join(D0_RUNS_FILE);
    let _ = install_d0_cache_run_file(&source_run_file, cache_entry_dir, manifest)?;
    if cache_manifest_path.exists() {
        let existing = fs::read(&cache_manifest_path)?;
        if existing != manifest_json {
            return Err(AnnError::log_index_error(format!(
                "D0 cache manifest already exists with different contents: {}",
                cache_manifest_path.display()
            )));
        }
    } else {
        fs::write(&cache_manifest_path, manifest_json)?;
    }
    tracing::info!(
        "[rbc/d0-cache-publish] manifest={} run_file={} run_file_bytes={} child_runs={} root_leaf_count={}",
        cache_manifest_path.display(),
        cache_entry_dir.join(D0_RUNS_FILE).display(),
        manifest.run_file_bytes,
        manifest.child_runs.len(),
        manifest.root_leaf_count,
    );
    Ok(())
}

fn load_reused_d0_child_runs(
    manifest_path: &Path,
    dataset: &dyn PointStore,
    metric: Metric,
    params: &ForgeANNParams,
    total_size: usize,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    root_seed: u64,
    root_fanout: &RootFanoutProfile,
    external_run_store: &Arc<Mutex<ExternalRunStore>>,
) -> AnnResult<(Vec<ChildRun>, Vec<D0RootLeafRun>, RootFanoutProfile)> {
    let reader = BufReader::new(File::open(manifest_path)?);
    let manifest: D0RunsManifest = serde_json::from_reader(reader).map_err(|err| {
        AnnError::log_index_error(format!(
            "Failed to parse D0 reuse manifest {}: {err}",
            manifest_path.display()
        ))
    })?;
    validate_d0_runs_manifest(
        &manifest,
        manifest_path,
        dataset,
        metric,
        params,
        total_size,
        adaptive_c_max,
        min_recurse_size,
        root_seed,
        root_fanout,
    )?;
    let source_run_file = resolve_manifest_run_file(manifest_path, &manifest);
    let target_dir = { external_run_store.lock().base_dir.clone() };
    let target_run_file = install_reused_d0_run_file(&source_run_file, &target_dir, &manifest)?;
    let child_runs = manifest
        .child_runs
        .iter()
        .map(|run| ChildRun {
            len: run.len,
            seed: run.seed,
            extents: run
                .extents
                .iter()
                .map(|extent| RunExtent {
                    byte_offset: extent.byte_offset,
                    len: extent.len,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let root_leaves = manifest
        .root_leaves
        .iter()
        .map(|leaf| {
            let reason = leaf_reason_from_label(&leaf.reason).ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "D0 reuse manifest contains unknown root leaf reason: {}",
                    leaf.reason
                ))
            })?;
            Ok(D0RootLeafRun {
                len: leaf.len,
                depth: leaf.depth,
                reason,
                extents: leaf
                    .extents
                    .iter()
                    .map(|extent| RunExtent {
                        byte_offset: extent.byte_offset,
                        len: extent.len,
                    })
                    .collect(),
            })
        })
        .collect::<AnnResult<Vec<_>>>()?;
    tracing::info!(
        "[rbc/d0-reuse] manifest={} source_run_file={} active_run_file={} child_runs={} root_leaf_count={} root_leaf_points={} run_file_bytes={} root_seed={} root_leader_hash={}",
        manifest_path.display(),
        source_run_file.display(),
        target_run_file.display(),
        child_runs.len(),
        root_leaves.len(),
        root_leaves.iter().map(|leaf| leaf.len).sum::<usize>(),
        manifest.run_file_bytes,
        root_seed,
        manifest.root_fanout.root_leader_hash,
    );
    Ok((child_runs, root_leaves, manifest.root_fanout))
}

fn emit_reused_d0_root_leaves(
    dataset: &dyn PointStore,
    metric: Metric,
    params: &ForgeANNParams,
    root_leaves: &[D0RootLeafRun],
    external_run_store: &Arc<Mutex<ExternalRunStore>>,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<PartitionStats> {
    let mut stats = PartitionStats::new(params.max_depth, root_leaves.len().max(4));
    if root_leaves.is_empty() {
        return Ok(stats);
    }

    let emit_start = Instant::now();
    let store = external_run_store.lock();
    for root_leaf in root_leaves {
        let mut leaf = read_child_run_chain(&store, 0, &root_leaf.extents)?;
        if leaf.len() != root_leaf.len {
            return Err(AnnError::log_index_error(format!(
                "D0 reuse root leaf length mismatch: expected={} actual={}",
                root_leaf.len,
                leaf.len()
            )));
        }
        emit_leaf(
            dataset,
            metric,
            params,
            &mut leaf,
            root_leaf.depth,
            root_leaf.reason,
            false,
            params.kernel_safe_leaf_size(),
            &mut stats,
            leaf_emitter,
        )?;
    }
    drop(store);
    tracing::info!(
        "[rbc/d0-reuse-root-leaves] leaves={} points={} emit_ms={}",
        root_leaves.len(),
        root_leaves.iter().map(|leaf| leaf.len).sum::<usize>(),
        emit_start.elapsed().as_millis(),
    );
    Ok(stats)
}

#[derive(Clone, Debug)]
struct D1ResidentSubtreeConfig {
    budget_bytes: usize,
    max_run_bytes: usize,
    whole_scope_budget_bytes: usize,
    max_dataset_resident_ratio: f64,
    budget_explicit: bool,
    max_run_explicit: bool,
    max_grouped_dataset_read_amp: f64,
    min_buffer_ratio: f64,
    whole_scope_min_buffer_ratio: f64,
    max_read_amplification: f64,
    max_window_bytes: usize,
    min_unique_points: usize,
    min_expected_leaves: usize,
    max_native_expected_leaves: usize,
    io_threads: usize,
    group_pipeline_slots: usize,
    enable_source: &'static str,
}

impl D1ResidentSubtreeConfig {
    const AUTO_GROUPED_DATASET_READ_AMP_HEADROOM: f64 = 1.4;
    const AUTO_GROUPED_DATASET_READ_AMP_MAX: f64 = 8.0;
    const AUTO_GROUPED_DATASET_READ_AMP_MIN: f64 = 2.0;
    const DEFAULT_GROUP_PIPELINE_SLOTS: usize = 2;
    const DEFAULT_IO_THREADS: usize = 8;

    fn from_params(params: &ForgeANNParams, adaptive_c_max: usize) -> Option<Self> {
        if !params.oom_enable || env_bool("FORGEANN_D1_RESIDENT_SUBTREE_DISABLE") {
            return None;
        }
        let enable_source = if env_bool("FORGEANN_D1_RESIDENT_SUBTREE_ENABLE") {
            "env"
        } else {
            "auto"
        };
        let budget_env = env_bytes_from_gib("FORGEANN_D1_RESIDENT_SUBTREE_BUDGET_GB")
            .or_else(|| env_usize("FORGEANN_D1_RESIDENT_SUBTREE_BUDGET_BYTES"));
        let budget_explicit = budget_env.is_some();
        let budget_bytes = budget_env
            .unwrap_or_else(|| params.effective_oom_memory_budget_bytes() / 4)
            .max(1);
        let max_run_env = env_bytes_from_gib("FORGEANN_D1_RESIDENT_SUBTREE_MAX_RUN_GB")
            .or_else(|| env_usize("FORGEANN_D1_RESIDENT_SUBTREE_MAX_RUN_BYTES"));
        let max_run_explicit = max_run_env.is_some();
        let max_run_bytes = max_run_env.unwrap_or(budget_bytes).max(1);
        let whole_scope_budget_bytes =
            env_bytes_from_gib("FORGEANN_D1_RESIDENT_SUBTREE_SCOPE_BUDGET_GB")
                .or_else(|| env_usize("FORGEANN_D1_RESIDENT_SUBTREE_SCOPE_BUDGET_BYTES"))
                .unwrap_or_else(|| params.effective_oom_memory_budget_bytes().max(budget_bytes))
                .max(1);
        let max_dataset_resident_ratio = env_f64("FORGEANN_D1_RESIDENT_SUBTREE_MAX_RESIDENT_RATIO")
            .or_else(|| env_f64("FORGEANN_D1_RESIDENT_SUBTREE_MAX_DATA_RATIO"))
            .unwrap_or(0.20)
            .clamp(0.01, 1.0);
        let max_grouped_dataset_read_amp =
            env_f64("FORGEANN_D1_RESIDENT_SUBTREE_MAX_GROUPED_DATA_READ_AMP")
                .unwrap_or_else(|| {
                    Self::default_grouped_dataset_read_amp(max_dataset_resident_ratio)
                })
                .max(1.0);
        let min_buffer_ratio = env_f64("FORGEANN_D1_RESIDENT_SUBTREE_MIN_BUFFER_RATIO")
            .unwrap_or(0.70)
            .clamp(0.01, 1.0);
        let whole_scope_min_buffer_ratio =
            env_f64("FORGEANN_D1_RESIDENT_SUBTREE_SCOPE_MIN_BUFFER_RATIO")
                .unwrap_or(min_buffer_ratio)
                .clamp(0.01, 1.0);
        let max_read_amplification = env_f64("FORGEANN_D1_RESIDENT_SUBTREE_MAX_READ_AMP")
            .unwrap_or(ForgeANNParams::OOM_POINT_PIPELINE_MAX_READ_AMPLIFICATION)
            .max(1.0);
        let max_window_bytes = env_usize("FORGEANN_D1_RESIDENT_SUBTREE_MAX_WINDOW_BYTES")
            .unwrap_or(ForgeANNParams::OOM_POINT_PIPELINE_MAX_WINDOW_BYTES)
            .max(1);
        let min_unique_points = env_usize("FORGEANN_D1_RESIDENT_SUBTREE_MIN_POINTS")
            .unwrap_or_else(|| adaptive_c_max.saturating_add(1))
            .max(1);
        let min_expected_leaves = env_usize("FORGEANN_D1_RESIDENT_SUBTREE_MIN_LEAVES")
            .unwrap_or(2)
            .max(1);
        let max_native_expected_leaves =
            env_usize("FORGEANN_D1_RESIDENT_SUBTREE_MAX_NATIVE_LEAVES")
                .unwrap_or(1024)
                .max(min_expected_leaves);
        let io_threads = env_usize("FORGEANN_D1_RESIDENT_SUBTREE_IO_THREADS")
            .unwrap_or(Self::DEFAULT_IO_THREADS)
            .max(1);
        let group_pipeline_slots =
            if env_bool("FORGEANN_D1_RESIDENT_SUBTREE_GROUP_PIPELINE_DISABLE") {
                1
            } else {
                env_usize("FORGEANN_D1_RESIDENT_SUBTREE_GROUP_PIPELINE_SLOTS")
                    .unwrap_or(Self::DEFAULT_GROUP_PIPELINE_SLOTS)
                    .clamp(1, 4)
            };
        Some(Self {
            budget_bytes,
            max_run_bytes,
            whole_scope_budget_bytes,
            max_dataset_resident_ratio,
            budget_explicit,
            max_run_explicit,
            max_grouped_dataset_read_amp,
            min_buffer_ratio,
            whole_scope_min_buffer_ratio,
            max_read_amplification,
            max_window_bytes,
            min_unique_points,
            min_expected_leaves,
            max_native_expected_leaves,
            io_threads,
            group_pipeline_slots,
            enable_source,
        })
    }

    fn max_resident_bytes_for(&self, parent_n: usize, dim: usize) -> usize {
        let dataset_bytes = parent_n
            .saturating_mul(dim.max(1))
            .saturating_mul(size_of::<f32>());
        ((dataset_bytes as f64) * self.max_dataset_resident_ratio).floor() as usize
    }

    fn default_grouped_dataset_read_amp(max_dataset_resident_ratio: f64) -> f64 {
        let ratio = max_dataset_resident_ratio.clamp(0.01, 1.0);
        (Self::AUTO_GROUPED_DATASET_READ_AMP_HEADROOM / ratio).clamp(
            Self::AUTO_GROUPED_DATASET_READ_AMP_MIN,
            Self::AUTO_GROUPED_DATASET_READ_AMP_MAX,
        )
    }

    fn grouped_budget_bytes_for(&self, parent_n: usize, dim: usize) -> usize {
        let ratio_cap = self.max_resident_bytes_for(parent_n, dim).max(1);
        let configured = if self.budget_explicit || self.max_run_explicit {
            self.max_run_bytes
        } else {
            ratio_cap
        };
        configured.min(ratio_cap).max(1)
    }

    fn limiter_budget_bytes_for(&self, parent_n: usize, dim: usize) -> usize {
        if self.budget_explicit {
            self.budget_bytes
        } else {
            self.grouped_budget_bytes_for(parent_n, dim)
        }
    }

    fn pipelined_group_budget_bytes_for(&self, parent_n: usize, dim: usize) -> usize {
        let base_group_budget = self.grouped_budget_bytes_for(parent_n, dim).max(1);
        let slots = self.group_pipeline_slots.max(1);
        if slots <= 1 {
            return base_group_budget;
        }
        let slot_budget = self
            .limiter_budget_bytes_for(parent_n, dim)
            .max(1)
            .saturating_div(slots)
            .max(d1_resident_subtree_estimated_bytes(dim, 1));
        base_group_budget.min(slot_budget).max(1)
    }

    fn grouped_dataset_read_amp(&self, physical_bytes: u64, parent_n: usize, dim: usize) -> f64 {
        let dataset_bytes = parent_n
            .saturating_mul(dim.max(1))
            .saturating_mul(size_of::<f32>())
            .max(1) as f64;
        physical_bytes as f64 / dataset_bytes
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct D1ResidentSubtreeLimiterState {
    live_bytes: usize,
    peak_live_bytes: usize,
    wait_count: usize,
    wait_ns: u128,
}

#[derive(Debug)]
struct D1ResidentSubtreeLimiter {
    budget_bytes: usize,
    state: Mutex<D1ResidentSubtreeLimiterState>,
    condvar: Condvar,
}

impl D1ResidentSubtreeLimiter {
    fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            state: Mutex::new(D1ResidentSubtreeLimiterState::default()),
            condvar: Condvar::new(),
        }
    }

    fn acquire(self: &Arc<Self>, bytes: usize) -> D1ResidentSubtreePermit {
        let bytes = bytes.max(1);
        let wait_start = Instant::now();
        let mut waited = false;
        let mut state = self.state.lock();
        while !self.can_admit(state.live_bytes, bytes) {
            waited = true;
            self.condvar.wait(&mut state);
        }
        if waited {
            state.wait_count = state.wait_count.saturating_add(1);
            state.wait_ns = state
                .wait_ns
                .saturating_add(wait_start.elapsed().as_nanos());
        }
        state.live_bytes = state.live_bytes.saturating_add(bytes);
        state.peak_live_bytes = state.peak_live_bytes.max(state.live_bytes);
        D1ResidentSubtreePermit {
            limiter: Arc::clone(self),
            bytes,
            released: false,
        }
    }

    fn snapshot(&self) -> D1ResidentSubtreeLimiterState {
        *self.state.lock()
    }

    fn can_admit(&self, live_bytes: usize, bytes: usize) -> bool {
        self.budget_bytes == 0
            || (bytes > self.budget_bytes && live_bytes == 0)
            || (bytes <= self.budget_bytes && live_bytes.saturating_add(bytes) <= self.budget_bytes)
    }

    fn release(&self, bytes: usize) {
        let mut state = self.state.lock();
        state.live_bytes = state.live_bytes.saturating_sub(bytes);
        drop(state);
        self.condvar.notify_all();
    }
}

struct D1ResidentSubtreePermit {
    limiter: Arc<D1ResidentSubtreeLimiter>,
    bytes: usize,
    released: bool,
}

impl Drop for D1ResidentSubtreePermit {
    fn drop(&mut self) {
        if !self.released {
            self.limiter.release(self.bytes);
            self.released = true;
        }
    }
}

struct PermitBackedPointStore {
    inner: Arc<dyn PointStore>,
    _permit: D1ResidentSubtreePermit,
}

impl PointStore for PermitBackedPointStore {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn point_id_capacity(&self) -> usize {
        self.inner.point_id_capacity()
    }

    fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
        self.inner.read_point_into(pid, out)
    }

    fn read_range_into(&self, start_pid: u32, count: usize, out: &mut [f32]) -> AnnResult<()> {
        self.inner.read_range_into(start_pid, count, out)
    }

    fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        self.inner.read_points_into(ids, out)
    }

    fn read_points_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        self.inner.read_points_into_stats(ids, out, stats)
    }

    fn read_points_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        self.inner
            .read_points_windowed_into_stats(ids, out, options, stats)
    }

    fn read_points_windowed_into_batch_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        self.inner
            .read_points_windowed_into_batch_stats(ids, out, options, stats)
    }

    fn read_bounded_plan_into_batch_stats(
        &self,
        plan: &BoundedReadPlan,
        out: &mut [f32],
        stats: &mut PointBatchStats,
    ) -> AnnResult<bool> {
        self.inner
            .read_bounded_plan_into_batch_stats(plan, out, stats)
    }

    fn is_vector_run(&self) -> bool {
        self.inner.is_vector_run()
    }

    fn is_resident_subset(&self) -> bool {
        self.inner.is_resident_subset()
    }

    fn resident_rows(&self) -> Option<(&[u32], &[f32])> {
        self.inner.resident_rows()
    }

    fn prefers_coalesced_window_reads(&self) -> bool {
        self.inner.prefers_coalesced_window_reads()
    }

    fn disables_prefetch_pipeline(&self) -> bool {
        self.inner.disables_prefetch_pipeline()
    }

    fn source_identity(&self) -> Option<String> {
        self.inner.source_identity()
    }

    fn discard_cached_pages(&self) -> AnnResult<()> {
        self.inner.discard_cached_pages()
    }
}

fn d1_resident_subtree_estimated_bytes(dim: usize, point_count: usize) -> usize {
    let vector_bytes = d1_resident_subtree_vector_bytes(dim, point_count);
    let id_bytes = size_of::<u32>();
    vector_bytes.saturating_add(point_count.max(1).saturating_mul(id_bytes))
}

fn d1_resident_subtree_vector_bytes(dim: usize, point_count: usize) -> usize {
    let row_bytes = dim.max(1).saturating_mul(size_of::<f32>());
    point_count.max(1).saturating_mul(row_bytes)
}

fn d1_resident_subtree_sketch_bytes(width: usize, point_count: usize) -> usize {
    point_count
        .max(1)
        .saturating_mul(width.max(1))
        .saturating_mul(size_of::<f32>())
        .saturating_add(point_count.max(1).saturating_mul(size_of::<u32>()))
        .saturating_add(size_of::<ResidentSubsetSketchAccessor>())
}

fn d1_resident_subtree_vector_budget_for_total_resident_budget(
    total_budget_bytes: usize,
    dim: usize,
    sketch_width: usize,
) -> usize {
    if sketch_width == 0 {
        return total_budget_bytes.max(1);
    }
    let vector_row_bytes = dim
        .max(1)
        .saturating_mul(size_of::<f32>())
        .saturating_add(size_of::<u32>())
        .max(1);
    let sketch_row_bytes = sketch_width
        .max(1)
        .saturating_mul(size_of::<f32>())
        .saturating_add(size_of::<u32>());
    let row_total_bytes = vector_row_bytes.saturating_add(sketch_row_bytes).max(1);
    total_budget_bytes
        .saturating_sub(size_of::<ResidentSubsetSketchAccessor>())
        .saturating_mul(vector_row_bytes)
        .saturating_div(row_total_bytes)
        .max(d1_resident_subtree_estimated_bytes(dim, 1))
}

#[derive(Debug)]
struct D1ResidentSubtreeGroup {
    group_index: usize,
    pairs: Vec<(ChildRun, Vec<u32>)>,
    unique_ids: Vec<u32>,
    source_runs: usize,
    selected_points: usize,
    logical_bytes: u64,
    physical_bytes: u64,
    planned_windows: usize,
    buffer_ratio: f64,
    resident_bytes: usize,
    expected_leaves: usize,
}

#[derive(Debug)]
struct D1ResidentSubtreeGrouping {
    groups: Vec<D1ResidentSubtreeGroup>,
    levelwise_pairs: Vec<(ChildRun, Vec<u32>)>,
    candidates: usize,
    selected_runs: usize,
    rejected_runs: usize,
}

#[derive(Debug)]
struct D1ResidentSubtreeScopeReject {
    pairs: Vec<(ChildRun, Vec<u32>)>,
    reason: &'static str,
    source_runs: usize,
    selected_points: usize,
    unique_points: usize,
    resident_bytes: usize,
    logical_bytes: u64,
    physical_bytes: u64,
    planned_windows: usize,
    buffer_ratio: f64,
}

struct D1ResidentPendingGroup {
    pairs: Vec<(ChildRun, Vec<u32>)>,
    unique_seen: std::collections::HashSet<u32>,
    unique_ids: Vec<u32>,
    selected_points: usize,
    expected_leaves: usize,
}

impl D1ResidentPendingGroup {
    fn new() -> Self {
        Self {
            pairs: Vec::new(),
            unique_seen: std::collections::HashSet::new(),
            unique_ids: Vec::new(),
            selected_points: 0,
            expected_leaves: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    fn estimated_resident_bytes_with(&self, additional_unique: usize, dim: usize) -> usize {
        d1_resident_subtree_estimated_bytes(
            dim,
            self.unique_seen.len().saturating_add(additional_unique),
        )
    }

    fn expected_leaves_with(&self, additional_points: usize, adaptive_c_max: usize) -> usize {
        self.selected_points
            .saturating_add(additional_points)
            .div_ceil(adaptive_c_max.max(1))
            .max(1)
    }

    fn push(
        &mut self,
        child_run: ChildRun,
        points: Vec<u32>,
        run_unique_ids: Vec<u32>,
        adaptive_c_max: usize,
    ) {
        self.selected_points = self.selected_points.saturating_add(points.len());
        self.expected_leaves = self.selected_points.div_ceil(adaptive_c_max.max(1)).max(1);
        for pid in run_unique_ids {
            if self.unique_seen.insert(pid) {
                self.unique_ids.push(pid);
            }
        }
        self.pairs.push((child_run, points));
    }

    fn into_group_or_levelwise(
        mut self,
        group_index: usize,
        config: &D1ResidentSubtreeConfig,
        dim: usize,
        group_budget_bytes: usize,
    ) -> Result<D1ResidentSubtreeGroup, Vec<(ChildRun, Vec<u32>)>> {
        if self.pairs.is_empty() {
            return Err(Vec::new());
        }
        self.unique_ids.sort_unstable();
        self.unique_ids.shrink_to_fit();
        let resident_bytes = d1_resident_subtree_estimated_bytes(dim, self.unique_ids.len());
        if resident_bytes > group_budget_bytes {
            return Err(self.pairs);
        }
        if self.expected_leaves > config.max_native_expected_leaves {
            return Err(self.pairs);
        }
        let row_bytes = dim.max(1).saturating_mul(size_of::<f32>());
        let planner = crate::forgeann::io_runtime::BoundedReadPlanner {
            row_bytes,
            max_window_bytes: config.max_window_bytes,
            max_read_amplification: config.max_read_amplification,
        };
        let plan = planner.plan(&self.unique_ids);
        let logical_bytes = plan.logical_bytes();
        let physical_bytes = plan.physical_bytes();
        let buffer_ratio = if physical_bytes == 0 {
            1.0
        } else {
            logical_bytes as f64 / physical_bytes as f64
        };
        let source_runs = self.pairs.len();
        Ok(D1ResidentSubtreeGroup {
            group_index,
            pairs: self.pairs,
            unique_ids: self.unique_ids,
            source_runs,
            selected_points: self.selected_points,
            logical_bytes,
            physical_bytes,
            planned_windows: plan.windows.len(),
            buffer_ratio,
            resident_bytes,
            expected_leaves: self.expected_leaves.max(4),
        })
    }
}

fn finalize_d1_resident_pending_group(
    pending: D1ResidentPendingGroup,
    groups: &mut Vec<D1ResidentSubtreeGroup>,
    levelwise_pairs: &mut Vec<(ChildRun, Vec<u32>)>,
    rejected_runs: &mut usize,
    config: &D1ResidentSubtreeConfig,
    dim: usize,
    group_budget_bytes: usize,
) {
    if pending.is_empty() {
        return;
    }
    let pending_runs = pending.pairs.len();
    match pending.into_group_or_levelwise(groups.len(), config, dim, group_budget_bytes) {
        Ok(group) => groups.push(group),
        Err(mut pairs) => {
            *rejected_runs = rejected_runs.saturating_add(pending_runs);
            levelwise_pairs.append(&mut pairs);
        }
    }
}

#[cfg(test)]
fn group_d1_resident_subtree_inputs(
    pairs: Vec<(ChildRun, Vec<u32>)>,
    config: &D1ResidentSubtreeConfig,
    dim: usize,
    parent_n: usize,
    adaptive_c_max: usize,
) -> D1ResidentSubtreeGrouping {
    group_d1_resident_subtree_inputs_with_budget(
        pairs,
        config,
        dim,
        adaptive_c_max,
        config.grouped_budget_bytes_for(parent_n, dim),
    )
}

fn group_d1_resident_subtree_inputs_with_budget(
    pairs: Vec<(ChildRun, Vec<u32>)>,
    config: &D1ResidentSubtreeConfig,
    dim: usize,
    adaptive_c_max: usize,
    group_budget_bytes: usize,
) -> D1ResidentSubtreeGrouping {
    let mut groups = Vec::new();
    let mut levelwise_pairs = Vec::new();
    let mut candidates = 0usize;
    let mut selected_runs = 0usize;
    let mut rejected_runs = 0usize;
    let mut pending = D1ResidentPendingGroup::new();

    for (child_run, points) in pairs {
        if points.is_empty() {
            continue;
        }
        candidates = candidates.saturating_add(1);

        let mut run_unique_ids = points.clone();
        run_unique_ids.sort_unstable();
        run_unique_ids.dedup();
        let additional_unique = run_unique_ids
            .iter()
            .filter(|&&pid| !pending.unique_seen.contains(&pid))
            .count();
        let run_expected_leaves = points.len().div_ceil(adaptive_c_max.max(1)).max(1);
        if run_expected_leaves > config.max_native_expected_leaves {
            if !pending.is_empty() {
                finalize_d1_resident_pending_group(
                    pending,
                    &mut groups,
                    &mut levelwise_pairs,
                    &mut rejected_runs,
                    config,
                    dim,
                    group_budget_bytes,
                );
                pending = D1ResidentPendingGroup::new();
            }
            selected_runs = selected_runs.saturating_add(1);
            rejected_runs = rejected_runs.saturating_add(1);
            levelwise_pairs.push((child_run, points));
            continue;
        }
        let projected_bytes = pending.estimated_resident_bytes_with(additional_unique, dim);
        let projected_expected_leaves = pending.expected_leaves_with(points.len(), adaptive_c_max);
        if !pending.is_empty()
            && (projected_bytes > group_budget_bytes
                || projected_expected_leaves > config.max_native_expected_leaves)
        {
            finalize_d1_resident_pending_group(
                pending,
                &mut groups,
                &mut levelwise_pairs,
                &mut rejected_runs,
                config,
                dim,
                group_budget_bytes,
            );
            pending = D1ResidentPendingGroup::new();
        }
        selected_runs = selected_runs.saturating_add(1);
        pending.push(child_run, points, run_unique_ids, adaptive_c_max);
    }

    finalize_d1_resident_pending_group(
        pending,
        &mut groups,
        &mut levelwise_pairs,
        &mut rejected_runs,
        config,
        dim,
        group_budget_bytes,
    );

    D1ResidentSubtreeGrouping {
        groups,
        levelwise_pairs,
        candidates,
        selected_runs,
        rejected_runs,
    }
}

fn d1_resident_grouping_physical_bytes(grouping: &D1ResidentSubtreeGrouping) -> u64 {
    grouping
        .groups
        .iter()
        .map(|group| group.physical_bytes)
        .sum::<u64>()
}

fn maybe_build_d1_resident_subtree_scope(
    pairs: Vec<(ChildRun, Vec<u32>)>,
    config: &D1ResidentSubtreeConfig,
    dim: usize,
    parent_n: usize,
    adaptive_c_max: usize,
) -> Result<D1ResidentSubtreeGroup, D1ResidentSubtreeScopeReject> {
    let reject = |pairs: Vec<(ChildRun, Vec<u32>)>,
                  reason: &'static str,
                  source_runs: usize,
                  selected_points: usize,
                  unique_points: usize,
                  resident_bytes: usize,
                  logical_bytes: u64,
                  physical_bytes: u64,
                  planned_windows: usize,
                  buffer_ratio: f64| {
        D1ResidentSubtreeScopeReject {
            pairs,
            reason,
            source_runs,
            selected_points,
            unique_points,
            resident_bytes,
            logical_bytes,
            physical_bytes,
            planned_windows,
            buffer_ratio,
        }
    };

    if pairs.is_empty() {
        return Err(reject(pairs, "empty", 0, 0, 0, 0, 0, 0, 0, 0.0));
    }

    let source_runs = pairs.len();
    let selected_points = pairs.iter().map(|(_, points)| points.len()).sum::<usize>();
    let expected_leaves = pairs
        .iter()
        .map(|(_, points)| points.len().div_ceil(adaptive_c_max.max(1)).max(1))
        .sum::<usize>();
    if selected_points == 0 || expected_leaves < config.min_expected_leaves {
        return Err(reject(
            pairs,
            "too_few_leaves",
            source_runs,
            selected_points,
            0,
            0,
            0,
            0,
            0,
            0.0,
        ));
    }

    let max_resident_bytes = config.max_resident_bytes_for(parent_n, dim).max(1);
    let mut unique_seen = std::collections::HashSet::new();
    let mut unique_ids = Vec::new();
    let mut budget_reject = None;
    'collect_unique: for (_, points) in &pairs {
        for &pid in points {
            if unique_seen.insert(pid) {
                unique_ids.push(pid);
                let resident_bytes = d1_resident_subtree_estimated_bytes(dim, unique_ids.len());
                if resident_bytes > config.whole_scope_budget_bytes {
                    budget_reject = Some(("scope_budget", unique_ids.len(), resident_bytes));
                    break 'collect_unique;
                }
                let resident_vector_bytes = d1_resident_subtree_vector_bytes(dim, unique_ids.len());
                if resident_vector_bytes > max_resident_bytes {
                    budget_reject = Some(("resident_ratio", unique_ids.len(), resident_bytes));
                    break 'collect_unique;
                }
            }
        }
    }
    if let Some((reason, unique_points, resident_bytes)) = budget_reject {
        return Err(reject(
            pairs,
            reason,
            source_runs,
            selected_points,
            unique_points,
            resident_bytes,
            0,
            0,
            0,
            0.0,
        ));
    }
    if unique_ids.len() < config.min_unique_points {
        let unique_points = unique_ids.len();
        return Err(reject(
            pairs,
            "too_few_unique_points",
            source_runs,
            selected_points,
            unique_points,
            0,
            0,
            0,
            0,
            0.0,
        ));
    }
    unique_ids.sort_unstable();
    unique_ids.shrink_to_fit();

    let resident_bytes = d1_resident_subtree_estimated_bytes(dim, unique_ids.len());
    let resident_vector_bytes = d1_resident_subtree_vector_bytes(dim, unique_ids.len());
    if resident_vector_bytes > max_resident_bytes {
        return Err(reject(
            pairs,
            "resident_ratio",
            source_runs,
            selected_points,
            unique_ids.len(),
            resident_bytes,
            0,
            0,
            0,
            0.0,
        ));
    }
    let row_bytes = dim.max(1).saturating_mul(size_of::<f32>());
    let planner = crate::forgeann::io_runtime::BoundedReadPlanner {
        row_bytes,
        max_window_bytes: config.max_window_bytes,
        max_read_amplification: config.max_read_amplification,
    };
    let plan = planner.plan(&unique_ids);
    let logical_bytes = plan.logical_bytes();
    let physical_bytes = plan.physical_bytes();
    let buffer_ratio = if physical_bytes == 0 {
        1.0
    } else {
        logical_bytes as f64 / physical_bytes as f64
    };

    Ok(D1ResidentSubtreeGroup {
        group_index: 0,
        pairs,
        unique_ids,
        source_runs,
        selected_points,
        logical_bytes,
        physical_bytes,
        planned_windows: plan.windows.len(),
        buffer_ratio,
        resident_bytes,
        expected_leaves: expected_leaves.max(4),
    })
}

#[cfg(test)]
mod d1_resident_subtree_tests {
    use super::*;

    #[test]
    fn external_run_store_routes_vector_runs_to_vector_dir() {
        let run_dir = tempfile::tempdir().unwrap();
        let vector_dir = tempfile::tempdir().unwrap();
        let store = ExternalRunStore::new_with_vector_dir(
            run_dir.path(),
            vector_dir.path(),
            DirectIoConfig::disabled(),
        );

        assert_eq!(
            store.path_for_depth(1),
            run_dir.path().join("partition_d01_runs.bin")
        );
        assert_eq!(
            store.vector_path(2, 7),
            vector_dir.path().join("partition_d02_vectors_00000007.bin")
        );
    }

    struct DimOnlyPointStore {
        len: usize,
        dim: usize,
    }

    impl PointStore for DimOnlyPointStore {
        fn len(&self) -> usize {
            self.len
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn read_point_into(&self, _pid: u32, out: &mut [f32]) -> AnnResult<()> {
            out.fill(0.0);
            Ok(())
        }

        fn read_range_into(
            &self,
            _start_pid: u32,
            _count: usize,
            out: &mut [f32],
        ) -> AnnResult<()> {
            out.fill(0.0);
            Ok(())
        }

        fn read_points_into(&self, _ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
            out.fill(0.0);
            Ok(())
        }
    }

    #[derive(Default)]
    struct CollectingLeafEmitter {
        leaves: Mutex<Vec<Vec<u32>>>,
    }

    impl CollectingLeafEmitter {
        fn leaf_count(&self) -> usize {
            self.leaves.lock().len()
        }
    }

    impl LeafEmitter for CollectingLeafEmitter {
        fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
            self.leaves.lock().push(leaf);
            Ok(())
        }
    }

    #[derive(Default)]
    struct CountingResidentLeafEmitter {
        inline: AtomicUsize,
    }

    impl LeafEmitter for CountingResidentLeafEmitter {
        fn emit_leaf(&self, _leaf: Vec<u32>) -> AnnResult<()> {
            Ok(())
        }

        fn emit_leaf_inline_from_dataset(
            &self,
            dataset: &dyn PointStore,
            _leaf: Vec<u32>,
        ) -> AnnResult<Option<Vec<u32>>> {
            assert!(dataset.is_resident_subset());
            self.inline.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }

    #[test]
    fn resident_subtree_leaf_emitter_submits_local_morsels() {
        let resident_store: Arc<dyn PointStore> = Arc::new(
            ResidentSubsetPointStore::new(
                vec![0, 1, 2, 3],
                2,
                vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            )
            .unwrap(),
        );
        let inner = CountingResidentLeafEmitter::default();
        let queue = Arc::new(D1ResidentLeafMorselQueue::new(2));
        let metrics = Arc::new(D1ResidentLeafMorselMetrics::default());
        let emitter = ResidentDatasetLeafEmitter::new(
            &inner,
            D1ResidentLeafMorselSink {
                queue: Arc::clone(&queue),
                metrics: Arc::clone(&metrics),
            },
        );

        emitter
            .emit_leaf_from_dataset(resident_store.as_ref(), vec![0, 1])
            .unwrap();
        emitter
            .emit_leaf_from_dataset(resident_store.as_ref(), vec![2, 3])
            .unwrap();

        assert_eq!(queue.pop(&metrics).unwrap(), vec![0, 1]);
        assert_eq!(queue.pop(&metrics).unwrap(), vec![2, 3]);
        assert_eq!(inner.inline.load(Ordering::Relaxed), 0);
        assert_eq!(emitter.resident_leaf_submissions(), 2);
        assert_eq!(metrics.submitted_leaves.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.queue_depth_peak.load(Ordering::Relaxed), 2);
        assert_eq!(
            emitter.resident_leaf_completion_mode(),
            "local_morsel_priority_work_batch"
        );
    }

    #[test]
    fn resident_leaf_morsel_queue_prioritizes_larger_leaves() {
        let queue = Arc::new(D1ResidentLeafMorselQueue::new(4));
        let metrics = Arc::new(D1ResidentLeafMorselMetrics::default());
        let sink = D1ResidentLeafMorselSink {
            queue: Arc::clone(&queue),
            metrics: Arc::clone(&metrics),
        };
        sink.submit(vec![0, 1]).unwrap();
        sink.submit(vec![2, 3, 4, 5]).unwrap();
        sink.submit(vec![6, 7, 8]).unwrap();
        drop(sink);

        assert_eq!(queue.pop(&metrics).unwrap(), vec![2, 3, 4, 5]);
        assert_eq!(queue.pop(&metrics).unwrap(), vec![6, 7, 8]);
        assert_eq!(queue.pop(&metrics).unwrap(), vec![0, 1]);
        assert!(queue.pop(&metrics).is_none());
        assert_eq!(metrics.queue_depth.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn resident_leaf_morsel_worker_drains_inline_batches() {
        let resident_store: Arc<dyn PointStore> = Arc::new(
            ResidentSubsetPointStore::new(
                vec![0, 1, 2, 3],
                2,
                vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            )
            .unwrap(),
        );
        let inner = CountingResidentLeafEmitter::default();
        let queue = Arc::new(D1ResidentLeafMorselQueue::new(4));
        let metrics = Arc::new(D1ResidentLeafMorselMetrics::default());
        let sink = D1ResidentLeafMorselSink {
            queue: Arc::clone(&queue),
            metrics: Arc::clone(&metrics),
        };
        sink.submit(vec![0, 1]).unwrap();
        sink.submit(vec![2, 3]).unwrap();
        drop(sink);

        d1_resident_leaf_morsel_worker_loop(
            queue,
            Arc::clone(&metrics),
            &inner,
            resident_store,
            None,
            D1ResidentLeafMorselPolicy {
                workers: 1,
                queue_capacity: 4,
                max_batch_leaves: 4,
                max_batch_points: 16,
                max_batch_work: 16,
            },
        )
        .unwrap();

        assert_eq!(inner.inline.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.drained_leaves.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.drain_batches.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.queue_depth.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn resident_leaf_morsel_worker_keeps_heavy_leaf_singleton() {
        let resident_store: Arc<dyn PointStore> = Arc::new(
            ResidentSubsetPointStore::new(
                vec![0, 1, 2, 3, 4, 5, 6],
                1,
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .unwrap(),
        );
        let inner = CountingResidentLeafEmitter::default();
        let queue = Arc::new(D1ResidentLeafMorselQueue::new(4));
        let metrics = Arc::new(D1ResidentLeafMorselMetrics::default());
        let sink = D1ResidentLeafMorselSink {
            queue: Arc::clone(&queue),
            metrics: Arc::clone(&metrics),
        };
        sink.submit(vec![0, 1]).unwrap();
        sink.submit(vec![2, 3, 4]).unwrap();
        sink.submit(vec![5, 6]).unwrap();
        drop(sink);

        d1_resident_leaf_morsel_worker_loop(
            queue,
            Arc::clone(&metrics),
            &inner,
            resident_store,
            None,
            D1ResidentLeafMorselPolicy {
                workers: 1,
                queue_capacity: 4,
                max_batch_leaves: 4,
                max_batch_points: 10,
                max_batch_work: 9,
            },
        )
        .unwrap();

        assert_eq!(inner.inline.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.drained_leaves.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.drain_batches.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.drain_batch_points_peak.load(Ordering::Relaxed), 4);
        assert_eq!(metrics.drain_batch_work_peak.load(Ordering::Relaxed), 9);
        assert_eq!(metrics.queue_depth.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn resident_leaf_morsel_policy_default_work_uses_four_leaf_budget() {
        let mut params = ForgeANNParams::default();
        params.max_full_matrix_leaf_size = 10;
        let policy = D1ResidentLeafMorselPolicy::from_params(&params);
        assert_eq!(policy.max_batch_work, 400);
    }

    #[test]
    fn permit_backed_resident_store_releases_with_last_arc() {
        let limiter = Arc::new(D1ResidentSubtreeLimiter::new(128));
        let permit = limiter.acquire(64);
        let resident_store: Arc<dyn PointStore> = Arc::new(PermitBackedPointStore {
            inner: Arc::new(
                ResidentSubsetPointStore::new(vec![0, 1], 2, vec![0.0, 0.0, 1.0, 1.0]).unwrap(),
            ),
            _permit: permit,
        });
        assert_eq!(limiter.snapshot().live_bytes, 64);

        let queued_store = Arc::clone(&resident_store);
        drop(resident_store);
        assert_eq!(limiter.snapshot().live_bytes, 64);

        drop(queued_store);
        assert_eq!(limiter.snapshot().live_bytes, 0);
    }

    fn matching_d0_manifest(
        params: &ForgeANNParams,
        root_profile: RootFanoutProfile,
    ) -> D0RunsManifest {
        D0RunsManifest {
            format: D0_RUNS_MANIFEST_FORMAT.to_string(),
            version: D0_RUNS_MANIFEST_VERSION,
            dataset_identity: None,
            num_points: 10,
            dim: 2,
            metric: metric_label(Metric::L2).to_string(),
            random_seed: params.random_seed,
            root_seed: 99,
            c_min: params.c_min,
            c_max: params.c_max,
            max_depth: params.max_depth,
            max_leaders: params.max_leaders,
            fanout_top: params.fanout_top,
            fanout_second: params.fanout_second,
            psamp_fraction_bits: params.psamp_fraction.to_bits(),
            min_shrink_ratio_bits: params.min_shrink_ratio.to_bits(),
            adaptive_c_max: params.adaptive_c_max(10),
            min_recurse_size: params.adaptive_c_max(10).saturating_mul(2),
            root_leaf_count: 0,
            root_leaf_points: 0,
            root_leaves: Vec::new(),
            root_fanout: root_profile,
            run_file: D0_RUNS_FILE.to_string(),
            run_file_bytes: 64,
            run_file_sha256: "0".repeat(64),
            child_runs: vec![D0RunsManifestChild {
                len: 2,
                seed: 123,
                extents: vec![D0RunsManifestExtent {
                    byte_offset: 0,
                    len: 2,
                }],
            }],
        }
    }

    #[test]
    fn d0_runs_manifest_validation_rejects_param_mismatch() {
        let mut params = ForgeANNParams::default();
        params.fanout_top = 6;
        let root_profile = RootFanoutProfile::fixed(params.fanout_top);
        let mut manifest = matching_d0_manifest(&params, root_profile.clone());
        manifest.fanout_top = 8;
        let dataset = DimOnlyPointStore { len: 10, dim: 2 };

        let err = validate_d0_runs_manifest(
            &manifest,
            Path::new("/tmp/d0.manifest.json"),
            &dataset,
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            &root_profile,
        )
        .expect_err("manifest with different fanout must be rejected");

        assert!(format!("{err}").contains("fanout_top"));
    }

    #[test]
    fn d0_runs_manifest_validation_rejects_extent_past_file() {
        let params = ForgeANNParams::default();
        let root_profile = RootFanoutProfile::fixed(params.fanout_top);
        let mut manifest = matching_d0_manifest(&params, root_profile.clone());
        manifest.run_file_bytes = 12;
        let dataset = DimOnlyPointStore { len: 10, dim: 2 };

        let err = validate_d0_runs_manifest(
            &manifest,
            Path::new("/tmp/d0.manifest.json"),
            &dataset,
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            &root_profile,
        )
        .expect_err("manifest extent past run file must be rejected");

        assert!(format!("{err}").contains("extent_end"));
    }

    #[test]
    fn d0_runs_cache_path_is_stable_and_parameter_scoped() {
        let cache_dir = Path::new("/tmp/forgeann-d0-cache");
        let mut params = ForgeANNParams::default();
        params.fanout_top = 6;
        let root_profile = RootFanoutProfile::fixed(params.fanout_top);
        let first = d0_runs_cache_manifest_path_for_dir(
            cache_dir,
            "dataset-a",
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            &root_profile,
        );
        let again = d0_runs_cache_manifest_path_for_dir(
            cache_dir,
            "dataset-a",
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            &root_profile,
        );
        params.fanout_top = 8;
        let changed = d0_runs_cache_manifest_path_for_dir(
            cache_dir,
            "dataset-a",
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            &RootFanoutProfile::fixed(params.fanout_top),
        );

        assert_eq!(first, again);
        assert_ne!(first, changed);
        assert_eq!(
            first.file_name().and_then(|name| name.to_str()),
            Some(D0_RUNS_MANIFEST_FILE)
        );
    }

    #[test]
    fn d0_runs_cache_publish_links_manifest_and_run_file() {
        let source_dir = tempfile::tempdir().unwrap();
        let cache_dir = tempfile::tempdir().unwrap();
        let mut source_store = ExternalRunStore::new(source_dir.path(), DirectIoConfig::disabled());
        let extent = source_store.append_points(0, &[1, 2, 3]).unwrap();
        source_store.finalize().unwrap();

        let params = ForgeANNParams::default();
        let root_profile = RootFanoutProfile::fixed(params.fanout_top);
        let mut manifest = matching_d0_manifest(&params, root_profile.clone());
        manifest.dataset_identity = Some("dataset-a".to_string());
        manifest.run_file_bytes = fs::metadata(source_dir.path().join(D0_RUNS_FILE))
            .unwrap()
            .len();
        manifest.run_file_sha256 = sha256_file_hex(&source_dir.path().join(D0_RUNS_FILE)).unwrap();
        manifest.child_runs = vec![D0RunsManifestChild {
            len: 3,
            seed: 123,
            extents: vec![D0RunsManifestExtent {
                byte_offset: extent.byte_offset,
                len: extent.len,
            }],
        }];
        let manifest_json = serde_json::to_vec_pretty(&manifest).unwrap();
        let cache_manifest = d0_runs_cache_manifest_path_for_dir(
            cache_dir.path(),
            "dataset-a",
            Metric::L2,
            &params,
            manifest.num_points,
            manifest.adaptive_c_max,
            manifest.min_recurse_size,
            manifest.root_seed,
            &root_profile,
        );

        publish_d0_runs_manifest_to_cache_dir(
            cache_dir.path(),
            source_dir.path(),
            &manifest,
            &manifest_json,
        )
        .unwrap();
        publish_d0_runs_manifest_to_cache_dir(
            cache_dir.path(),
            source_dir.path(),
            &manifest,
            &manifest_json,
        )
        .unwrap();

        assert!(cache_manifest.exists());
        assert!(cache_manifest.parent().unwrap().join(D0_RUNS_FILE).exists());
        assert_eq!(fs::read(cache_manifest).unwrap(), manifest_json);
    }

    #[test]
    fn d0_manifest_validation_rejects_dataset_identity_mismatch() {
        let params = ForgeANNParams::default();
        let root_profile = RootFanoutProfile::fixed(params.fanout_top);
        let mut manifest = matching_d0_manifest(&params, root_profile.clone());
        manifest.dataset_identity = Some("test-dataset".to_string());
        let dataset = DimOnlyPointStore { len: 10, dim: 2 };

        let err = validate_d0_runs_manifest(
            &manifest,
            Path::new("/tmp/d0.manifest.json"),
            &dataset,
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            &root_profile,
        )
        .expect_err("manifest with identity must require matching dataset identity");

        assert!(format!("{err}").contains("dataset_identity_unavailable"));
    }

    #[test]
    fn d0_runs_manifest_write_records_child_extents_and_hash() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ExternalRunStore::new(dir.path(), DirectIoConfig::disabled());
        let extent = store.append_points(0, &[1, 2]).unwrap();
        store.finalize().unwrap();
        let params = ForgeANNParams::default();
        let dataset = DimOnlyPointStore { len: 10, dim: 2 };
        let root_profile = RootFanoutProfile::fixed(params.fanout_top);
        let child_run = ChildRun {
            extents: vec![extent],
            len: 2,
            seed: 123,
        };

        let manifest_path = write_d0_runs_manifest(
            dir.path(),
            &dataset,
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            root_profile,
            &[child_run],
            &[],
        )
        .unwrap()
        .expect("large-root manifest should be written");
        let text = fs::read_to_string(manifest_path).unwrap();
        let manifest: D0RunsManifest = serde_json::from_str(&text).unwrap();

        assert_eq!(manifest.child_runs.len(), 1);
        assert_eq!(manifest.child_runs[0].seed, 123);
        assert_eq!(
            manifest.child_runs[0].extents[0].byte_offset,
            extent.byte_offset
        );
        assert_eq!(manifest.run_file_sha256.len(), 64);
    }

    #[test]
    fn d0_runs_manifest_load_installs_run_file_and_restores_children() {
        let source_dir = tempfile::tempdir().unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let mut source_store = ExternalRunStore::new(source_dir.path(), DirectIoConfig::disabled());
        let extent = source_store.append_points(0, &[1, 2, 3]).unwrap();
        source_store.finalize().unwrap();
        let params = ForgeANNParams::default();
        let dataset = DimOnlyPointStore { len: 10, dim: 2 };
        let root_profile = RootFanoutProfile::fixed(params.fanout_top);
        let manifest_path = write_d0_runs_manifest(
            source_dir.path(),
            &dataset,
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            root_profile.clone(),
            &[ChildRun {
                extents: vec![extent],
                len: 3,
                seed: 123,
            }],
            &[],
        )
        .unwrap()
        .unwrap();
        let target_store = Arc::new(Mutex::new(ExternalRunStore::new(
            target_dir.path(),
            DirectIoConfig::disabled(),
        )));

        let (child_runs, root_leaves, loaded_root_profile) = load_reused_d0_child_runs(
            &manifest_path,
            &dataset,
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            &root_profile,
            &target_store,
        )
        .unwrap();

        assert_eq!(loaded_root_profile.fixed_fanout, root_profile.fixed_fanout);
        assert_eq!(child_runs.len(), 1);
        assert!(root_leaves.is_empty());
        assert_eq!(child_runs[0].len, 3);
        assert_eq!(child_runs[0].seed, 123);
        assert!(target_dir.path().join(D0_RUNS_FILE).exists());
    }

    #[test]
    fn d0_runs_manifest_roundtrips_root_leaves() {
        let source_dir = tempfile::tempdir().unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let mut source_store = ExternalRunStore::new(source_dir.path(), DirectIoConfig::disabled());
        let child_extent = source_store.append_points(0, &[1, 2, 3]).unwrap();
        let root_extent = source_store.append_points(0, &[7, 8]).unwrap();
        source_store.finalize().unwrap();
        let params = ForgeANNParams::default();
        let dataset = DimOnlyPointStore { len: 10, dim: 2 };
        let root_profile = RootFanoutProfile::fixed(params.fanout_top);
        let root_leaf = D0RootLeafRun {
            len: 2,
            depth: 1,
            reason: LeafReason::NaturalSize,
            extents: vec![root_extent],
        };
        let manifest_path = write_d0_runs_manifest(
            source_dir.path(),
            &dataset,
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            root_profile.clone(),
            &[ChildRun {
                extents: vec![child_extent],
                len: 3,
                seed: 123,
            }],
            &[root_leaf],
        )
        .unwrap()
        .unwrap();
        let target_store = Arc::new(Mutex::new(ExternalRunStore::new(
            target_dir.path(),
            DirectIoConfig::disabled(),
        )));

        let (child_runs, root_leaves, _) = load_reused_d0_child_runs(
            &manifest_path,
            &dataset,
            Metric::L2,
            &params,
            10,
            params.adaptive_c_max(10),
            params.adaptive_c_max(10).saturating_mul(2),
            99,
            &root_profile,
            &target_store,
        )
        .unwrap();

        assert_eq!(child_runs.len(), 1);
        assert_eq!(root_leaves.len(), 1);
        assert_eq!(root_leaves[0].len, 2);
        assert_eq!(root_leaves[0].depth, 1);
        assert_eq!(
            leaf_reason_label(root_leaves[0].reason),
            leaf_reason_label(LeafReason::NaturalSize)
        );
        let restored = {
            let guard = target_store.lock();
            read_child_run_chain(&guard, 0, &root_leaves[0].extents).unwrap()
        };
        assert_eq!(restored, vec![7, 8]);
    }

    fn resident_subtree_pipeline_test_config(slots: usize) -> D1ResidentSubtreeConfig {
        D1ResidentSubtreeConfig {
            budget_bytes: 1 << 30,
            max_run_bytes: 1 << 30,
            whole_scope_budget_bytes: 1 << 30,
            max_dataset_resident_ratio: 0.20,
            budget_explicit: false,
            max_run_explicit: false,
            max_grouped_dataset_read_amp: 2.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 1,
            min_expected_leaves: 1,
            max_native_expected_leaves: usize::MAX,
            io_threads: 4,
            group_pipeline_slots: slots,
            enable_source: "test",
        }
    }

    #[test]
    fn resident_subtree_group_preserves_loaded_child_points() {
        let config = D1ResidentSubtreeConfig {
            budget_bytes: 1 << 30,
            max_run_bytes: 1 << 30,
            whole_scope_budget_bytes: 1 << 30,
            max_dataset_resident_ratio: 1.0,
            budget_explicit: true,
            max_run_explicit: true,
            max_grouped_dataset_read_amp: 2.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 1,
            min_expected_leaves: 1,
            max_native_expected_leaves: usize::MAX,
            io_threads: 1,
            group_pipeline_slots: 1,
            enable_source: "test",
        };
        let child_run = ChildRun {
            extents: Vec::new(),
            len: 4,
            seed: 17,
        };
        let points = vec![0, 1, 2, 3];

        let grouping =
            group_d1_resident_subtree_inputs(vec![(child_run, points.clone())], &config, 2, 64, 2);

        assert_eq!(grouping.levelwise_pairs.len(), 0);
        assert_eq!(grouping.groups.len(), 1);
        assert_eq!(grouping.groups[0].source_runs, 1);
        assert_eq!(grouping.groups[0].pairs[0].1, points);
    }

    #[test]
    fn resident_subtree_grouping_keeps_small_runs_native() {
        let config = D1ResidentSubtreeConfig {
            budget_bytes: 1 << 30,
            max_run_bytes: 1 << 30,
            whole_scope_budget_bytes: 1 << 30,
            max_dataset_resident_ratio: 1.0,
            budget_explicit: true,
            max_run_explicit: true,
            max_grouped_dataset_read_amp: 1.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 64,
            min_expected_leaves: 64,
            max_native_expected_leaves: usize::MAX,
            io_threads: 1,
            group_pipeline_slots: 1,
            enable_source: "test",
        };
        let child_run = ChildRun {
            extents: Vec::new(),
            len: 2,
            seed: 17,
        };
        let grouping =
            group_d1_resident_subtree_inputs(vec![(child_run, vec![7, 9])], &config, 2, 64, 8);

        assert_eq!(grouping.groups.len(), 1);
        assert_eq!(grouping.levelwise_pairs.len(), 0);
        assert_eq!(grouping.rejected_runs, 0);
        assert_eq!(grouping.selected_runs, 1);
    }

    #[test]
    fn resident_subtree_scope_selects_overlapping_runs_once() {
        let config = D1ResidentSubtreeConfig {
            budget_bytes: 1 << 30,
            max_run_bytes: 1 << 30,
            whole_scope_budget_bytes: 1 << 30,
            max_dataset_resident_ratio: 1.0,
            budget_explicit: true,
            max_run_explicit: true,
            max_grouped_dataset_read_amp: 2.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 1,
            min_expected_leaves: 1,
            max_native_expected_leaves: usize::MAX,
            io_threads: 1,
            group_pipeline_slots: 1,
            enable_source: "test",
        };
        let run_a = ChildRun {
            extents: Vec::new(),
            len: 4,
            seed: 17,
        };
        let run_b = ChildRun {
            extents: Vec::new(),
            len: 4,
            seed: 23,
        };

        let group = maybe_build_d1_resident_subtree_scope(
            vec![(run_a, vec![0, 1, 2, 3]), (run_b, vec![2, 3, 4, 5])],
            &config,
            2,
            64,
            2,
        )
        .expect("whole scope group");

        assert_eq!(group.source_runs, 2);
        assert_eq!(group.selected_points, 8);
        assert_eq!(group.unique_ids, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(group.pairs[0].1, vec![0, 1, 2, 3]);
        assert_eq!(group.pairs[1].1, vec![2, 3, 4, 5]);
    }

    #[test]
    fn resident_subtree_scope_low_buffer_ratio_is_native_telemetry() {
        let config = D1ResidentSubtreeConfig {
            budget_bytes: 1 << 30,
            max_run_bytes: 1 << 30,
            whole_scope_budget_bytes: 1 << 30,
            max_dataset_resident_ratio: 1.0,
            budget_explicit: true,
            max_run_explicit: true,
            max_grouped_dataset_read_amp: 1.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1_000.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 1,
            min_expected_leaves: 1,
            max_native_expected_leaves: usize::MAX,
            io_threads: 1,
            group_pipeline_slots: 1,
            enable_source: "test",
        };
        let child_run = ChildRun {
            extents: Vec::new(),
            len: 2,
            seed: 17,
        };

        let group = maybe_build_d1_resident_subtree_scope(
            vec![(child_run, vec![0, 1_000])],
            &config,
            2,
            2_000,
            2,
        )
        .expect("low-buffer whole scope should remain native");

        assert_eq!(group.source_runs, 1);
        assert!(group.buffer_ratio < config.whole_scope_min_buffer_ratio);
        assert_eq!(group.unique_ids, vec![0, 1_000]);
    }

    #[test]
    fn resident_subtree_scope_respects_budget_gate() {
        let config = D1ResidentSubtreeConfig {
            budget_bytes: 1 << 30,
            max_run_bytes: 1 << 30,
            whole_scope_budget_bytes: 1,
            max_dataset_resident_ratio: 1.0,
            budget_explicit: true,
            max_run_explicit: true,
            max_grouped_dataset_read_amp: 2.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 1,
            min_expected_leaves: 1,
            max_native_expected_leaves: usize::MAX,
            io_threads: 1,
            group_pipeline_slots: 1,
            enable_source: "test",
        };
        let child_run = ChildRun {
            extents: Vec::new(),
            len: 4,
            seed: 17,
        };

        let rejected = maybe_build_d1_resident_subtree_scope(
            vec![(child_run, vec![0, 1, 2, 3])],
            &config,
            2,
            8,
            2,
        )
        .expect_err("scope should exceed budget");

        assert_eq!(rejected.reason, "scope_budget");
        assert_eq!(rejected.pairs.len(), 1);
        assert_eq!(rejected.pairs[0].1, vec![0, 1, 2, 3]);
        assert!(rejected.unique_points < 4);
    }

    #[test]
    fn resident_subtree_scope_respects_dataset_ratio_gate() {
        let config = D1ResidentSubtreeConfig {
            budget_bytes: 1 << 30,
            max_run_bytes: 1 << 30,
            whole_scope_budget_bytes: 1 << 30,
            max_dataset_resident_ratio: 0.20,
            budget_explicit: true,
            max_run_explicit: true,
            max_grouped_dataset_read_amp: 2.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 1,
            min_expected_leaves: 1,
            max_native_expected_leaves: usize::MAX,
            io_threads: 1,
            group_pipeline_slots: 1,
            enable_source: "test",
        };
        let child_run = ChildRun {
            extents: Vec::new(),
            len: 21,
            seed: 17,
        };

        let rejected = maybe_build_d1_resident_subtree_scope(
            vec![(child_run, (0..21).collect::<Vec<_>>())],
            &config,
            2,
            100,
            2,
        )
        .expect_err("scope should exceed resident ratio");

        assert_eq!(rejected.reason, "resident_ratio");
        assert!(rejected.resident_bytes > config.max_resident_bytes_for(100, 2));
    }

    #[test]
    fn resident_subtree_scope_ratio_gate_counts_vector_payload_not_ids() {
        let config = D1ResidentSubtreeConfig {
            budget_bytes: 1 << 30,
            max_run_bytes: 1 << 30,
            whole_scope_budget_bytes: 1 << 30,
            max_dataset_resident_ratio: 0.40,
            budget_explicit: true,
            max_run_explicit: true,
            max_grouped_dataset_read_amp: 2.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 1,
            min_expected_leaves: 1,
            max_native_expected_leaves: usize::MAX,
            io_threads: 1,
            group_pipeline_slots: 1,
            enable_source: "test",
        };
        let child_run = ChildRun {
            extents: Vec::new(),
            len: 4,
            seed: 17,
        };

        let group = maybe_build_d1_resident_subtree_scope(
            vec![(child_run, vec![0, 1, 2, 3])],
            &config,
            2,
            10,
            2,
        )
        .expect("resident ratio should gate vector payload, not sorted-id metadata");
        let cap = config.max_resident_bytes_for(10, 2);
        let vector_bytes = d1_resident_subtree_vector_bytes(2, group.unique_ids.len());
        let indexed_estimate = group.unique_ids.len().saturating_mul(
            2usize
                .saturating_mul(size_of::<f32>())
                .saturating_add(size_of::<u32>())
                .saturating_add(size_of::<(u32, usize)>()),
        );

        assert_eq!(group.resident_bytes, 48);
        assert_eq!(vector_bytes, cap);
        assert!(group.resident_bytes > cap);
        assert!(indexed_estimate > cap);
    }

    #[test]
    fn resident_subtree_default_group_budget_uses_ratio_cap() {
        let config = D1ResidentSubtreeConfig {
            budget_bytes: 8,
            max_run_bytes: 8,
            whole_scope_budget_bytes: 1 << 30,
            max_dataset_resident_ratio: 0.20,
            budget_explicit: false,
            max_run_explicit: false,
            max_grouped_dataset_read_amp: 2.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 1,
            min_expected_leaves: 1,
            max_native_expected_leaves: usize::MAX,
            io_threads: 1,
            group_pipeline_slots: 1,
            enable_source: "test",
        };
        let child_run = ChildRun {
            extents: Vec::new(),
            len: 4,
            seed: 17,
        };

        assert_eq!(config.grouped_budget_bytes_for(100, 2), 160);
        let grouping = group_d1_resident_subtree_inputs(
            vec![(child_run, vec![0, 1, 2, 3])],
            &config,
            2,
            100,
            2,
        );

        assert_eq!(grouping.groups.len(), 1);
        assert_eq!(grouping.rejected_runs, 0);
    }

    #[test]
    fn resident_subtree_pipeline_group_budget_uses_slot_share() {
        let config = resident_subtree_pipeline_test_config(2);
        let base_budget = config.grouped_budget_bytes_for(100, 2);
        let pipeline_budget = config.pipelined_group_budget_bytes_for(100, 2);

        assert_eq!(base_budget, 160);
        assert_eq!(pipeline_budget, 80);
    }

    #[test]
    fn resident_subtree_default_group_pipeline_slots_is_two() {
        assert_eq!(D1ResidentSubtreeConfig::DEFAULT_GROUP_PIPELINE_SLOTS, 2);
    }

    #[test]
    fn resident_subtree_default_io_threads_is_eight() {
        assert_eq!(D1ResidentSubtreeConfig::DEFAULT_IO_THREADS, 8);
    }

    #[test]
    fn resident_subtree_pipeline_grouping_keeps_two_groups_within_cap() {
        let config = resident_subtree_pipeline_test_config(2);
        let pipeline_budget = config.pipelined_group_budget_bytes_for(100, 2);
        let limiter_budget = config.limiter_budget_bytes_for(100, 2);
        let runs = (0..4)
            .map(|idx| {
                let child_run = ChildRun {
                    extents: Vec::new(),
                    len: 2,
                    seed: idx as u64,
                };
                let base = (idx * 2) as u32;
                (child_run, vec![base, base + 1])
            })
            .collect::<Vec<_>>();

        let grouping =
            group_d1_resident_subtree_inputs_with_budget(runs, &config, 2, 2, pipeline_budget);

        assert!(grouping.groups.len() >= 2);
        for group in &grouping.groups {
            assert!(group.resident_bytes <= pipeline_budget);
        }
        assert!(
            grouping.groups[0]
                .resident_bytes
                .saturating_add(grouping.groups[1].resident_bytes)
                <= limiter_budget
        );
        for (idx, group) in grouping.groups.iter().enumerate() {
            assert_eq!(group.group_index, idx);
        }
    }

    #[test]
    fn resident_subtree_grouping_preserves_planner_order() {
        let config = resident_subtree_pipeline_test_config(1);
        let group_budget = d1_resident_subtree_estimated_bytes(2, 5);
        let runs = vec![
            (
                ChildRun {
                    extents: Vec::new(),
                    len: 2,
                    seed: 1,
                },
                vec![0, 1],
            ),
            (
                ChildRun {
                    extents: Vec::new(),
                    len: 5,
                    seed: 2,
                },
                vec![10, 11, 12, 13, 14],
            ),
            (
                ChildRun {
                    extents: Vec::new(),
                    len: 3,
                    seed: 3,
                },
                vec![20, 21, 22],
            ),
        ];

        let grouping =
            group_d1_resident_subtree_inputs_with_budget(runs, &config, 2, 2, group_budget);

        assert_eq!(
            grouping
                .groups
                .iter()
                .map(|group| group.selected_points)
                .collect::<Vec<_>>(),
            vec![2, 5, 3]
        );
        for (idx, group) in grouping.groups.iter().enumerate() {
            assert_eq!(group.group_index, idx);
        }
    }

    #[test]
    fn resident_subtree_grouping_rejects_oversized_single_run() {
        let config = resident_subtree_pipeline_test_config(1);
        let group_budget = d1_resident_subtree_estimated_bytes(2, 3);
        let runs = vec![
            (
                ChildRun {
                    extents: Vec::new(),
                    len: 5,
                    seed: 1,
                },
                vec![0, 1, 2, 3, 4],
            ),
            (
                ChildRun {
                    extents: Vec::new(),
                    len: 7,
                    seed: 2,
                },
                vec![10, 11, 12, 13, 14, 15, 16],
            ),
        ];

        let grouping =
            group_d1_resident_subtree_inputs_with_budget(runs, &config, 2, 2, group_budget);

        assert!(grouping.groups.is_empty());
        assert_eq!(grouping.rejected_runs, 2);
        assert_eq!(
            grouping
                .levelwise_pairs
                .iter()
                .map(|(_, points)| points.len())
                .collect::<Vec<_>>(),
            vec![5, 7]
        );
    }

    #[test]
    fn resident_subtree_grouping_routes_root_like_runs_to_levelwise() {
        let mut config = resident_subtree_pipeline_test_config(1);
        config.max_native_expected_leaves = 2;
        let group_budget = d1_resident_subtree_estimated_bytes(2, 32);
        let oversized = ChildRun {
            extents: Vec::new(),
            len: 7,
            seed: 7,
        };
        let small = ChildRun {
            extents: Vec::new(),
            len: 2,
            seed: 2,
        };

        let grouping = group_d1_resident_subtree_inputs_with_budget(
            vec![
                (oversized, vec![10, 11, 12, 13, 14, 15, 16]),
                (small, vec![1, 2]),
            ],
            &config,
            2,
            3,
            group_budget,
        );

        assert_eq!(grouping.groups.len(), 1);
        assert_eq!(grouping.groups[0].selected_points, 2);
        assert_eq!(grouping.levelwise_pairs.len(), 1);
        assert_eq!(grouping.levelwise_pairs[0].1.len(), 7);
        assert_eq!(grouping.rejected_runs, 1);
    }

    #[test]
    fn resident_subtree_grouping_splits_before_group_becomes_root_like() {
        let mut config = resident_subtree_pipeline_test_config(1);
        config.max_native_expected_leaves = 2;
        let group_budget = d1_resident_subtree_estimated_bytes(2, 32);
        let runs = (0..4)
            .map(|idx| {
                let child_run = ChildRun {
                    extents: Vec::new(),
                    len: 2,
                    seed: idx as u64,
                };
                let base = (idx * 10) as u32;
                (child_run, vec![base, base + 1])
            })
            .collect::<Vec<_>>();

        let grouping =
            group_d1_resident_subtree_inputs_with_budget(runs, &config, 2, 3, group_budget);

        assert_eq!(grouping.levelwise_pairs.len(), 0);
        assert_eq!(grouping.rejected_runs, 0);
        assert_eq!(grouping.groups.len(), 2);
        assert_eq!(
            grouping
                .groups
                .iter()
                .map(|group| group.selected_points)
                .collect::<Vec<_>>(),
            vec![6, 2]
        );
    }

    #[test]
    fn resident_subtree_pipeline_keeps_requested_slots_when_read_amp_is_high() {
        let mut config = resident_subtree_pipeline_test_config(2);
        config.max_grouped_dataset_read_amp = 0.01;
        let pipeline_budget = config.pipelined_group_budget_bytes_for(100, 2);
        let runs = (0..3)
            .map(|idx| {
                let child_run = ChildRun {
                    extents: Vec::new(),
                    len: 2,
                    seed: idx as u64,
                };
                let base = (idx * 2) as u32;
                (child_run, vec![base, base + 1])
            })
            .collect::<Vec<_>>();

        let grouping =
            group_d1_resident_subtree_inputs_with_budget(runs, &config, 2, 2, pipeline_budget);
        let grouped_physical_bytes = d1_resident_grouping_physical_bytes(&grouping);
        let grouped_read_amp = config.grouped_dataset_read_amp(grouped_physical_bytes, 100, 2);

        assert!(grouped_read_amp > config.max_grouped_dataset_read_amp);
        assert!(!grouping.groups.is_empty());
        assert_eq!(grouping.levelwise_pairs.len(), 0);
        assert_eq!(
            grouping.selected_runs + grouping.rejected_runs,
            grouping.candidates
        );
    }

    #[test]
    fn resident_subtree_slots_one_matches_default_grouping_budget() {
        let config = resident_subtree_pipeline_test_config(1);
        let child_run = ChildRun {
            extents: Vec::new(),
            len: 4,
            seed: 17,
        };
        let pairs = vec![(child_run, vec![0, 1, 2, 3])];
        let default_grouping = group_d1_resident_subtree_inputs(pairs.clone(), &config, 2, 100, 2);
        let explicit_grouping = group_d1_resident_subtree_inputs_with_budget(
            pairs,
            &config,
            2,
            2,
            config.grouped_budget_bytes_for(100, 2),
        );

        assert_eq!(
            default_grouping.groups.len(),
            explicit_grouping.groups.len()
        );
        assert_eq!(
            default_grouping.selected_runs,
            explicit_grouping.selected_runs
        );
        assert_eq!(
            default_grouping.rejected_runs,
            explicit_grouping.rejected_runs
        );
    }

    #[test]
    fn resident_subtree_grouped_read_amp_is_telemetry_only() {
        let config = D1ResidentSubtreeConfig {
            budget_bytes: 8,
            max_run_bytes: 8,
            whole_scope_budget_bytes: 1 << 30,
            max_dataset_resident_ratio: 0.20,
            budget_explicit: false,
            max_run_explicit: false,
            max_grouped_dataset_read_amp: 2.0,
            min_buffer_ratio: 1.0,
            whole_scope_min_buffer_ratio: 1.0,
            max_read_amplification: 1.0,
            max_window_bytes: 1 << 20,
            min_unique_points: 1,
            min_expected_leaves: 1,
            max_native_expected_leaves: usize::MAX,
            io_threads: 1,
            group_pipeline_slots: 1,
            enable_source: "test",
        };

        assert!(
            config.grouped_dataset_read_amp(2_000, 100, 2) > config.max_grouped_dataset_read_amp
        );
        assert!(
            config.grouped_dataset_read_amp(1_000, 1_000, 2) < config.max_grouped_dataset_read_amp
        );

        let child_run = ChildRun {
            extents: Vec::new(),
            len: 4,
            seed: 17,
        };
        let grouping = group_d1_resident_subtree_inputs_with_budget(
            vec![(child_run, vec![0, 1, 2, 3])],
            &config,
            2,
            2,
            config.grouped_budget_bytes_for(100, 2),
        );

        assert_eq!(grouping.groups.len(), 1);
        assert_eq!(grouping.levelwise_pairs.len(), 0);
        assert_eq!(grouping.rejected_runs, 0);
    }

    #[test]
    fn resident_subtree_default_grouped_read_amp_scales_with_resident_ratio() {
        let full_resident = D1ResidentSubtreeConfig::default_grouped_dataset_read_amp(1.0);
        let twenty_percent = D1ResidentSubtreeConfig::default_grouped_dataset_read_amp(0.20);
        let tiny_resident = D1ResidentSubtreeConfig::default_grouped_dataset_read_amp(0.01);

        assert!((full_resident - 2.0).abs() < 1e-12);
        assert!((twenty_percent - 7.0).abs() < 1e-12);
        assert!((tiny_resident - 8.0).abs() < 1e-12);
    }

    #[test]
    fn resident_subtree_recursion_does_not_write_external_child_runs() {
        let n = 96usize;
        let dim = 1usize;
        let ids = (0..n as u32).collect::<Vec<_>>();
        let data = ids.iter().map(|&id| id as f32).collect::<Vec<_>>();
        let dataset = ResidentSubsetPointStore::new(ids.clone(), dim, data).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let external_run_store = Arc::new(Mutex::new(ExternalRunStore::new(
            dir.path(),
            DirectIoConfig::disabled(),
        )));
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        params.c_min = 2;
        params.c_max = 8;
        params.max_depth = 4;
        params.max_leaders = 2;
        params.psamp_fraction = 0.1;
        params.fanout_top = 1;
        params.fanout_second = 1;
        let adaptive_c_max = params.adaptive_c_max(n);
        let min_recurse_size = adaptive_c_max.saturating_mul(2);
        let pb = ProgressBar::hidden();
        let root_fanout_state = RootFanoutState::fixed(params.fanout_top);
        let assignment_context =
            AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None);
        let leaf_emitter = CollectingLeafEmitter::default();

        let stats = rbc_recurse_parallel(
            &dataset,
            ids,
            1,
            n,
            Metric::L2,
            &params,
            adaptive_c_max,
            min_recurse_size,
            123,
            &pb,
            Some(&external_run_store),
            &root_fanout_state,
            &assignment_context,
            &leaf_emitter,
        )
        .unwrap();
        external_run_store.lock().finalize().unwrap();

        assert!(!stats.leaf_sizes.is_empty());
        assert!(leaf_emitter.leaf_count() > 0);
        assert!(
            fs::read_dir(dir.path()).unwrap().next().is_none(),
            "resident recursion must not spill child-run files"
        );
    }

    #[test]
    fn resident_source_offsets_map_sorted_and_unsorted_run_points() {
        let resident_ids = vec![3, 5, 8, 13, 21];

        assert_eq!(
            d1_resident_source_offsets(&resident_ids, &[3, 8, 21]).unwrap(),
            vec![0, 2, 4]
        );
        assert_eq!(
            d1_resident_source_offsets(&resident_ids, &[21, 5, 13]).unwrap(),
            vec![4, 1, 3]
        );
        assert!(d1_resident_source_offsets(&resident_ids, &[3, 4]).is_err());
    }

    #[test]
    fn resident_scan_compute_tasks_cover_group_runs_in_stable_order() {
        let runs = vec![
            D1LevelScanRunWork {
                points: vec![0, 1, 2, 3, 4],
                seed: 0,
                raw_counts: Vec::new(),
                assignment_segments: Vec::new(),
                leaders: 0,
                fanout: 0,
                max_leaders: 0,
            },
            D1LevelScanRunWork {
                points: vec![10, 11, 12],
                seed: 0,
                raw_counts: Vec::new(),
                assignment_segments: Vec::new(),
                leaders: 0,
                fanout: 0,
                max_leaders: 0,
            },
        ];

        let tasks = build_d1_resident_level_scan_compute_tasks(&runs, 2);

        assert_eq!(
            tasks,
            vec![
                D1LevelScanComputeTask {
                    run_idx: 0,
                    point_start: 0,
                    point_end: 2,
                    run_point_start: 0,
                },
                D1LevelScanComputeTask {
                    run_idx: 0,
                    point_start: 2,
                    point_end: 4,
                    run_point_start: 2,
                },
                D1LevelScanComputeTask {
                    run_idx: 0,
                    point_start: 4,
                    point_end: 5,
                    run_point_start: 4,
                },
                D1LevelScanComputeTask {
                    run_idx: 1,
                    point_start: 0,
                    point_end: 2,
                    run_point_start: 0,
                },
                D1LevelScanComputeTask {
                    run_idx: 1,
                    point_start: 2,
                    point_end: 3,
                    run_point_start: 2,
                },
            ]
        );
    }
}

struct D1ResidentSubtreeGroupResult {
    stats: PartitionStats,
    hydrate_wall: Duration,
    build_wall: Duration,
    resident_leaf_wait: Duration,
    load_wall: Duration,
    sketch_hydrate_wall: Duration,
    sketch_stats: WindowedGatherStats,
    pipeline_stats: PointPipelineStats,
    point_calls: u64,
    range_calls: u64,
    range_rows_read: u64,
    bytes_read: u64,
    resident_leaf_submissions: usize,
    resident_leaf_drained: usize,
    resident_leaf_batches: usize,
    resident_leaf_queue_depth_peak: usize,
    resident_leaf_send_wait_ms: u64,
    resident_leaf_batch_points_peak: usize,
    resident_leaf_batch_work_peak: u64,
    resident_leaf_profile: D1ResidentLeafProfileDelta,
}

#[derive(Clone, Copy, Debug)]
struct D1ResidentLeafMorselPolicy {
    workers: usize,
    queue_capacity: usize,
    max_batch_leaves: usize,
    max_batch_points: usize,
    max_batch_work: u128,
}

impl D1ResidentLeafMorselPolicy {
    const DEFAULT_BATCH_WORK_MULTIPLIER: u128 = 4;

    fn from_params(params: &ForgeANNParams) -> Self {
        let worker_threads = rayon::current_num_threads().max(1);
        let default_workers = worker_threads;
        let workers = env_usize("FORGEANN_D1_RESIDENT_SUBTREE_LEAF_MORSEL_WORKERS")
            .unwrap_or(default_workers)
            .clamp(1, worker_threads.max(1));
        let (default_max_batch_leaves, default_max_batch_points) =
            d1_resident_leaf_batch_limits(params, worker_threads);
        let max_batch_leaves = env_usize("FORGEANN_D1_RESIDENT_SUBTREE_LEAF_MORSEL_MAX_LEAVES")
            .unwrap_or(default_max_batch_leaves)
            .max(1);
        let max_batch_points = env_usize("FORGEANN_D1_RESIDENT_SUBTREE_LEAF_MORSEL_MAX_POINTS")
            .unwrap_or(default_max_batch_points)
            .max(1);
        let default_max_batch_work = leaf_morsel_quadratic_work(params.kernel_safe_leaf_size())
            .saturating_mul(Self::DEFAULT_BATCH_WORK_MULTIPLIER);
        let max_batch_work = env_usize("FORGEANN_D1_RESIDENT_SUBTREE_LEAF_MORSEL_MAX_WORK")
            .map(|value| value as u128)
            .unwrap_or(default_max_batch_work)
            .max(1);
        let queue_capacity = env_usize("FORGEANN_D1_RESIDENT_SUBTREE_LEAF_MORSEL_QUEUE_LEAVES")
            .unwrap_or_else(|| {
                workers
                    .saturating_mul(max_batch_leaves)
                    .saturating_mul(8)
                    .clamp(max_batch_leaves, 65_536)
            })
            .max(1);
        Self {
            workers,
            queue_capacity,
            max_batch_leaves,
            max_batch_points,
            max_batch_work,
        }
    }
}

fn leaf_morsel_quadratic_work(points: usize) -> u128 {
    let points = points as u128;
    points.saturating_mul(points)
}

fn leaf_morsel_work_to_u64(work: u128) -> u64 {
    work.min(u64::MAX as u128) as u64
}

#[derive(Default)]
struct D1ResidentLeafMorselMetrics {
    submitted_leaves: AtomicUsize,
    drained_leaves: AtomicUsize,
    drain_batches: AtomicUsize,
    drain_batch_points_peak: AtomicUsize,
    drain_batch_work_peak: AtomicU64,
    queue_depth: AtomicUsize,
    queue_depth_peak: AtomicUsize,
    send_wait_ns: AtomicU64,
    error: Mutex<Option<String>>,
}

impl D1ResidentLeafMorselMetrics {
    fn check_error(&self) -> AnnResult<()> {
        if let Some(message) = self.error.lock().clone() {
            Err(AnnError::log_index_error(message))
        } else {
            Ok(())
        }
    }

    fn record_error(&self, err: AnnError) {
        let mut guard = self.error.lock();
        if guard.is_none() {
            *guard = Some(err.to_string());
        }
    }

    fn note_pending_leaf(&self) {
        let depth = self.queue_depth.fetch_add(1, Ordering::AcqRel) + 1;
        update_atomic_max_usize(&self.queue_depth_peak, depth);
    }

    fn note_submitted_leaf(&self) {
        self.submitted_leaves.fetch_add(1, Ordering::Relaxed);
    }

    fn note_dequeued_leaf(&self) {
        let mut depth = self.queue_depth.load(Ordering::Acquire);
        while depth > 0 {
            match self.queue_depth.compare_exchange_weak(
                depth,
                depth - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(next) => depth = next,
            }
        }
    }

    fn note_drained_batch(&self, leaves: usize, points: usize, work: u128) {
        self.drained_leaves.fetch_add(leaves, Ordering::Relaxed);
        self.drain_batches.fetch_add(1, Ordering::Relaxed);
        update_atomic_max_usize(&self.drain_batch_points_peak, points);
        update_atomic_max_u64(&self.drain_batch_work_peak, leaf_morsel_work_to_u64(work));
    }
}

struct D1ResidentLeafMorselItem {
    points: usize,
    seq: u64,
    leaf: Vec<u32>,
}

impl PartialEq for D1ResidentLeafMorselItem {
    fn eq(&self, other: &Self) -> bool {
        self.points == other.points && self.seq == other.seq
    }
}

impl Eq for D1ResidentLeafMorselItem {}

impl PartialOrd for D1ResidentLeafMorselItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for D1ResidentLeafMorselItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.points
            .cmp(&other.points)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

#[derive(Default)]
struct D1ResidentLeafMorselQueueState {
    heap: BinaryHeap<D1ResidentLeafMorselItem>,
    closed: bool,
    next_seq: u64,
}

struct D1ResidentLeafMorselQueue {
    capacity: usize,
    state: Mutex<D1ResidentLeafMorselQueueState>,
    condvar: Condvar,
}

impl D1ResidentLeafMorselQueue {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            state: Mutex::new(D1ResidentLeafMorselQueueState::default()),
            condvar: Condvar::new(),
        }
    }

    fn submit(&self, leaf: Vec<u32>, metrics: &D1ResidentLeafMorselMetrics) -> AnnResult<()> {
        metrics.check_error()?;
        let send_start = Instant::now();
        let mut waited = false;
        let mut guard = self.state.lock();
        loop {
            if guard.closed {
                return Err(AnnError::log_index_error(
                    "D1 resident leaf morsel queue closed while submitting leaf".to_string(),
                ));
            }
            if guard.heap.len() < self.capacity {
                let seq = guard.next_seq;
                guard.next_seq = guard.next_seq.wrapping_add(1);
                metrics.note_pending_leaf();
                guard.heap.push(D1ResidentLeafMorselItem {
                    points: leaf.len(),
                    seq,
                    leaf,
                });
                metrics.note_submitted_leaf();
                if waited {
                    metrics
                        .send_wait_ns
                        .fetch_add(duration_nanos_u64(send_start.elapsed()), Ordering::Relaxed);
                }
                self.condvar.notify_one();
                return Ok(());
            }
            waited = true;
            self.condvar.wait_for(&mut guard, Duration::from_millis(10));
            drop(guard);
            metrics.check_error()?;
            guard = self.state.lock();
        }
    }

    fn pop(&self, metrics: &D1ResidentLeafMorselMetrics) -> Option<Vec<u32>> {
        let mut guard = self.state.lock();
        loop {
            if let Some(item) = guard.heap.pop() {
                metrics.note_dequeued_leaf();
                self.condvar.notify_one();
                return Some(item.leaf);
            }
            if guard.closed {
                return None;
            }
            self.condvar.wait(&mut guard);
        }
    }

    fn try_pop_for_batch(
        &self,
        metrics: &D1ResidentLeafMorselMetrics,
        current_points: usize,
        current_work: u128,
        policy: D1ResidentLeafMorselPolicy,
    ) -> Option<Vec<u32>> {
        let mut guard = self.state.lock();
        let item = guard.heap.peek()?;
        let next_points = item.points;
        if current_points.saturating_add(next_points) > policy.max_batch_points {
            return None;
        }
        let next_work = leaf_morsel_quadratic_work(next_points);
        if current_work.saturating_add(next_work) > policy.max_batch_work {
            return None;
        }
        let item = guard.heap.pop()?;
        metrics.note_dequeued_leaf();
        self.condvar.notify_one();
        Some(item.leaf)
    }

    fn close(&self) {
        let mut guard = self.state.lock();
        guard.closed = true;
        self.condvar.notify_all();
    }
}

struct D1ResidentLeafMorselSink {
    queue: Arc<D1ResidentLeafMorselQueue>,
    metrics: Arc<D1ResidentLeafMorselMetrics>,
}

impl D1ResidentLeafMorselSink {
    fn submit(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.queue.submit(leaf, &self.metrics)
    }
}

impl Drop for D1ResidentLeafMorselSink {
    fn drop(&mut self) {
        self.queue.close();
    }
}

#[derive(Default)]
struct D1ResidentLeafMorselStats {
    wait: Duration,
    drained_leaves: usize,
    drain_batches: usize,
    queue_depth_peak: usize,
    send_wait_ms: u64,
    batch_points_peak: usize,
    batch_work_peak: u64,
}

#[derive(Clone, Debug, Default)]
struct D1ResidentLeafProfileDelta {
    leaves: usize,
    points: usize,
    total_wall: Duration,
    load: Duration,
    distance: Duration,
    topk: Duration,
    hash: Duration,
    flush: Duration,
    sketch_prefetch: Duration,
    sketch_rows_prefetched: usize,
    ads_leaves: usize,
    ads_rows: usize,
    ads_layout: Duration,
    ads_seed: Duration,
    ads_scan: Duration,
    ads_seed_evals: u64,
    ads_full_evals: u64,
    ads_pruned_evals: u64,
    ads_group_evals: u64,
    ads_simd_group_calls: u64,
    ads_simd_active_lane_evals: u64,
    ads_scalar_group_evals: u64,
}

impl D1ResidentLeafProfileDelta {
    fn from_snapshots(before: Option<&LeafProfile>, after: Option<&LeafProfile>) -> Self {
        let (Some(before), Some(after)) = (before, after) else {
            return Self::default();
        };
        Self {
            leaves: after.leaves.saturating_sub(before.leaves),
            points: after.points.saturating_sub(before.points),
            total_wall: duration_saturating_sub(after.total_wall, before.total_wall),
            load: duration_saturating_sub(after.load, before.load),
            distance: duration_saturating_sub(after.distance, before.distance),
            topk: duration_saturating_sub(after.topk, before.topk),
            hash: duration_saturating_sub(after.hash, before.hash),
            flush: duration_saturating_sub(after.flush, before.flush),
            sketch_prefetch: duration_saturating_sub(after.sketch_prefetch, before.sketch_prefetch),
            sketch_rows_prefetched: after
                .sketch_rows_prefetched
                .saturating_sub(before.sketch_rows_prefetched),
            ads_leaves: after
                .leaf_adsampling_leaves
                .saturating_sub(before.leaf_adsampling_leaves),
            ads_rows: after
                .leaf_adsampling_rows
                .saturating_sub(before.leaf_adsampling_rows),
            ads_layout: duration_saturating_sub(
                after.leaf_adsampling_layout,
                before.leaf_adsampling_layout,
            ),
            ads_seed: duration_saturating_sub(
                after.leaf_adsampling_seed,
                before.leaf_adsampling_seed,
            ),
            ads_scan: duration_saturating_sub(
                after.leaf_adsampling_scan,
                before.leaf_adsampling_scan,
            ),
            ads_seed_evals: after
                .leaf_adsampling_seed_evals
                .saturating_sub(before.leaf_adsampling_seed_evals),
            ads_full_evals: after
                .leaf_adsampling_full_evals
                .saturating_sub(before.leaf_adsampling_full_evals),
            ads_pruned_evals: after
                .leaf_adsampling_pruned_evals
                .saturating_sub(before.leaf_adsampling_pruned_evals),
            ads_group_evals: after
                .leaf_adsampling_group_evals
                .saturating_sub(before.leaf_adsampling_group_evals),
            ads_simd_group_calls: after
                .leaf_adsampling_simd_group_calls
                .saturating_sub(before.leaf_adsampling_simd_group_calls),
            ads_simd_active_lane_evals: after
                .leaf_adsampling_simd_active_lane_evals
                .saturating_sub(before.leaf_adsampling_simd_active_lane_evals),
            ads_scalar_group_evals: after
                .leaf_adsampling_scalar_group_evals
                .saturating_sub(before.leaf_adsampling_scalar_group_evals),
        }
    }

    fn merge(&mut self, other: &Self) {
        self.leaves = self.leaves.saturating_add(other.leaves);
        self.points = self.points.saturating_add(other.points);
        self.total_wall += other.total_wall;
        self.load += other.load;
        self.distance += other.distance;
        self.topk += other.topk;
        self.hash += other.hash;
        self.flush += other.flush;
        self.sketch_prefetch += other.sketch_prefetch;
        self.sketch_rows_prefetched = self
            .sketch_rows_prefetched
            .saturating_add(other.sketch_rows_prefetched);
        self.ads_leaves = self.ads_leaves.saturating_add(other.ads_leaves);
        self.ads_rows = self.ads_rows.saturating_add(other.ads_rows);
        self.ads_layout += other.ads_layout;
        self.ads_seed += other.ads_seed;
        self.ads_scan += other.ads_scan;
        self.ads_seed_evals = self.ads_seed_evals.saturating_add(other.ads_seed_evals);
        self.ads_full_evals = self.ads_full_evals.saturating_add(other.ads_full_evals);
        self.ads_pruned_evals = self.ads_pruned_evals.saturating_add(other.ads_pruned_evals);
        self.ads_group_evals = self.ads_group_evals.saturating_add(other.ads_group_evals);
        self.ads_simd_group_calls = self
            .ads_simd_group_calls
            .saturating_add(other.ads_simd_group_calls);
        self.ads_simd_active_lane_evals = self
            .ads_simd_active_lane_evals
            .saturating_add(other.ads_simd_active_lane_evals);
        self.ads_scalar_group_evals = self
            .ads_scalar_group_evals
            .saturating_add(other.ads_scalar_group_evals);
    }
}

fn duration_saturating_sub(after: Duration, before: Duration) -> Duration {
    after.checked_sub(before).unwrap_or(Duration::ZERO)
}

fn d1_resident_leaf_morsel_worker_loop(
    queue: Arc<D1ResidentLeafMorselQueue>,
    metrics: Arc<D1ResidentLeafMorselMetrics>,
    inner: &dyn LeafEmitter,
    dataset: Arc<dyn PointStore>,
    sketches: Option<Arc<ResidentSubsetSketchAccessor>>,
    policy: D1ResidentLeafMorselPolicy,
) -> AnnResult<()> {
    while metrics.check_error().is_ok() {
        let first = match queue.pop(&metrics) {
            Some(leaf) => leaf,
            None => break,
        };
        let mut batch = Vec::with_capacity(policy.max_batch_leaves.min(64).max(1));
        let mut batch_points = first.len();
        let mut batch_work = leaf_morsel_quadratic_work(first.len());
        batch.push(first);
        while batch.len() < policy.max_batch_leaves && batch_points < policy.max_batch_points {
            match queue.try_pop_for_batch(&metrics, batch_points, batch_work, policy) {
                Some(leaf) => {
                    batch_points = batch_points.saturating_add(leaf.len());
                    batch_work = batch_work.saturating_add(leaf_morsel_quadratic_work(leaf.len()));
                    batch.push(leaf);
                }
                None => break,
            }
        }
        let batch_len = batch.len();
        for leaf in batch {
            let emitted = if let Some(sketches) = sketches.as_ref() {
                inner.emit_leaf_inline_from_dataset_with_sketches(
                    dataset.as_ref(),
                    sketches.as_ref(),
                    leaf,
                )?
            } else {
                inner.emit_leaf_inline_from_dataset(dataset.as_ref(), leaf)?
            };
            if let Some(leaf) = emitted {
                inner.emit_leaf(leaf)?;
            }
        }
        metrics.note_drained_batch(batch_len, batch_points, batch_work);
    }
    metrics.check_error()
}

fn finish_d1_resident_leaf_morsels(
    handles: Vec<std::thread::ScopedJoinHandle<'_, AnnResult<()>>>,
    metrics: &Arc<D1ResidentLeafMorselMetrics>,
) -> AnnResult<D1ResidentLeafMorselStats> {
    let wait_start = Instant::now();
    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => metrics.record_error(err),
            Err(_) => metrics.record_error(AnnError::log_index_error(
                "D1 resident leaf morsel worker thread panicked".to_string(),
            )),
        }
    }
    metrics.check_error()?;
    Ok(D1ResidentLeafMorselStats {
        wait: wait_start.elapsed(),
        drained_leaves: metrics.drained_leaves.load(Ordering::Relaxed),
        drain_batches: metrics.drain_batches.load(Ordering::Relaxed),
        queue_depth_peak: metrics.queue_depth_peak.load(Ordering::Relaxed),
        send_wait_ms: metrics.send_wait_ns.load(Ordering::Relaxed) / 1_000_000,
        batch_points_peak: metrics.drain_batch_points_peak.load(Ordering::Relaxed),
        batch_work_peak: metrics.drain_batch_work_peak.load(Ordering::Relaxed),
    })
}

struct D1ResidentSubtreeHydratedGroup {
    group_index: usize,
    pairs: Vec<(ChildRun, Vec<u32>)>,
    unique_points: usize,
    source_runs: usize,
    selected_points: usize,
    resident_bytes: usize,
    resident_vector_bytes: usize,
    resident_sketch_bytes: usize,
    buffer_ratio: f64,
    hydrate_wall: Duration,
    load_wall: Duration,
    sketch_hydrate_wall: Duration,
    sketch_stats: WindowedGatherStats,
    pipeline_stats: PointPipelineStats,
    point_calls: u64,
    range_calls: u64,
    range_rows_read: u64,
    bytes_read: u64,
    resident_store: Arc<dyn PointStore>,
    resident_sketches: Option<Arc<ResidentSubsetSketchAccessor>>,
}

#[derive(Default)]
struct D1ResidentSubtreePipelineTelemetry {
    hydrate_send_wait_ns: AtomicU64,
    build_recv_wait_ns: AtomicU64,
    ready_queue_depth_peak: AtomicUsize,
}

fn duration_nanos_u64(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn d1_resident_hydrate_config(
    config: &D1ResidentSubtreeConfig,
    params: &ForgeANNParams,
    row_bytes: usize,
) -> PointPipelineConfig {
    let mut hydrate_config = PointPipelineConfig::from_params(params);
    hydrate_config.enabled = true;
    hydrate_config.io_threads = config.io_threads;
    hydrate_config.queue_depth = hydrate_config.queue_depth.max(config.io_threads);
    hydrate_config.budget_bytes = hydrate_config.budget_bytes.max(row_bytes);
    hydrate_config.max_window_bytes = config.max_window_bytes;
    hydrate_config.max_read_amplification = config.max_read_amplification;
    hydrate_config.min_leaf_points = 1;
    hydrate_config.window_cache_bytes = 0;
    hydrate_config
}

fn hydrate_d1_resident_subtree_sketches(
    sketches: &dyn SketchAccessor,
    unique_ids: &[u32],
    config: &D1ResidentSubtreeConfig,
) -> AnnResult<(ResidentSubsetSketchAccessor, Duration, WindowedGatherStats)> {
    let width = sketches.width();
    let mut data = vec![0.0f32; unique_ids.len().saturating_mul(width)];
    let mut stats = WindowedGatherStats::default();
    let start = Instant::now();
    sketches.read_rows_bounded_into_stats(
        unique_ids,
        &mut data,
        config.max_window_bytes,
        config.max_read_amplification,
        &mut stats,
    )?;
    let wall = start.elapsed();
    let resident =
        ResidentSubsetSketchAccessor::new(unique_ids.to_vec(), sketches.rows(), width, data)?;
    Ok((resident, wall, stats))
}

fn hydrate_d1_resident_subtree_group(
    dataset: &dyn PointStore,
    sketches: Option<&dyn SketchAccessor>,
    group: D1ResidentSubtreeGroup,
    config: &D1ResidentSubtreeConfig,
    params: &ForgeANNParams,
    limiter: &Arc<D1ResidentSubtreeLimiter>,
    depth: usize,
    row_bytes: usize,
) -> AnnResult<D1ResidentSubtreeHydratedGroup> {
    let D1ResidentSubtreeGroup {
        group_index,
        pairs,
        unique_ids,
        source_runs,
        selected_points,
        logical_bytes,
        physical_bytes,
        planned_windows,
        buffer_ratio,
        resident_bytes,
        expected_leaves: _,
    } = group;
    let resident_vector_bytes = resident_bytes;
    let resident_sketch_bytes = sketches
        .map(|sketches| d1_resident_subtree_sketch_bytes(sketches.width(), unique_ids.len()))
        .unwrap_or(0);
    let resident_total_bytes = resident_vector_bytes.saturating_add(resident_sketch_bytes);
    let resident_permit = limiter.acquire(resident_total_bytes);
    let hydrate_config = d1_resident_hydrate_config(config, params, row_bytes);

    tracing::info!(
        "[adsampling/d1-resident-subtree-group-start] depth={} group={} source_runs={} unique_points={} selected_points={} resident_bytes={} resident_vector_bytes={} resident_sketch_bytes={} logical_bytes={} physical_bytes={} planned_windows={} buffer_ratio={:.4}",
        depth,
        group_index,
        source_runs,
        unique_ids.len(),
        selected_points,
        resident_total_bytes,
        resident_vector_bytes,
        resident_sketch_bytes,
        logical_bytes,
        physical_bytes,
        planned_windows,
        buffer_ratio,
    );
    let hydrate_start = Instant::now();
    let hydrated = hydrate_resident_subset_with_stats(dataset, &unique_ids, &hydrate_config)?;
    let mut sketch_hydrate_wall = Duration::ZERO;
    let mut sketch_stats = WindowedGatherStats::default();
    let resident_sketches = if let Some(sketches) = sketches {
        let (resident_sketches, wall, stats) =
            hydrate_d1_resident_subtree_sketches(sketches, &unique_ids, config)?;
        sketch_hydrate_wall = wall;
        sketch_stats = stats;
        Some(Arc::new(resident_sketches))
    } else {
        None
    };
    let hydrate_wall = hydrate_start.elapsed();
    tracing::info!(
        "[adsampling/d1-resident-subtree-group-hydrated] depth={} group={} hydrate_ms={} load_ms={} sketch_hydrate_ms={} sketch_rows={} sketch_physical_bytes={} sketch_windows={} sketch_read_amp={:.3} pipeline_batches={} physical_bytes={} read_amp={:.3} point_calls={} range_calls={} range_rows_read={} bytes_read={}",
        depth,
        group_index,
        hydrate_wall.as_millis(),
        hydrated.load.as_millis(),
        sketch_hydrate_wall.as_millis(),
        sketch_stats.rows_requested,
        sketch_stats.physical_bytes,
        sketch_stats.windows_submitted,
        if sketch_stats.logical_bytes == 0 {
            0.0
        } else {
            sketch_stats.physical_bytes as f64 / sketch_stats.logical_bytes as f64
        },
        hydrated.pipeline_stats.batches,
        hydrated.pipeline_stats.physical_bytes,
        hydrated.pipeline_stats.read_amplification(),
        hydrated.io_stats.point_calls,
        hydrated.io_stats.range_calls,
        hydrated.io_stats.range_rows_read,
        hydrated.io_stats.bytes_read,
    );
    let resident_store: Arc<dyn PointStore> = Arc::new(PermitBackedPointStore {
        inner: Arc::new(hydrated.store),
        _permit: resident_permit,
    });
    Ok(D1ResidentSubtreeHydratedGroup {
        group_index,
        pairs,
        unique_points: unique_ids.len(),
        source_runs,
        selected_points,
        resident_bytes: resident_total_bytes,
        resident_vector_bytes,
        resident_sketch_bytes,
        buffer_ratio,
        hydrate_wall,
        load_wall: hydrated.load,
        sketch_hydrate_wall,
        sketch_stats,
        pipeline_stats: hydrated.pipeline_stats,
        point_calls: hydrated.io_stats.point_calls,
        range_calls: hydrated.io_stats.range_calls,
        range_rows_read: hydrated.io_stats.range_rows_read,
        bytes_read: hydrated.io_stats.bytes_read,
        resident_store,
        resident_sketches,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_d1_resident_subtree_group(
    hydrated: D1ResidentSubtreeHydratedGroup,
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
) -> AnnResult<D1ResidentSubtreeGroupResult> {
    let D1ResidentSubtreeHydratedGroup {
        group_index,
        pairs,
        unique_points,
        source_runs,
        selected_points,
        resident_bytes,
        resident_vector_bytes,
        resident_sketch_bytes,
        buffer_ratio,
        hydrate_wall,
        load_wall,
        sketch_hydrate_wall,
        sketch_stats,
        pipeline_stats,
        point_calls,
        range_calls,
        range_rows_read,
        bytes_read,
        resident_store,
        resident_sketches,
    } = hydrated;
    let leaf_policy = D1ResidentLeafMorselPolicy::from_params(params);
    let resident_leaf_profile_before = leaf_emitter.leaf_profile_snapshot();
    let (mut group_stats, build_wall, resident_leaf_submissions, resident_leaf_mode, morsel_stats) =
        std::thread::scope(|scope| -> AnnResult<_> {
            let leaf_queue = Arc::new(D1ResidentLeafMorselQueue::new(leaf_policy.queue_capacity));
            let leaf_metrics = Arc::new(D1ResidentLeafMorselMetrics::default());
            let mut leaf_handles = Vec::with_capacity(leaf_policy.workers);
            for _ in 0..leaf_policy.workers {
                let worker_queue = Arc::clone(&leaf_queue);
                let worker_metrics = Arc::clone(&leaf_metrics);
                let worker_store = Arc::clone(&resident_store);
                let worker_sketches = resident_sketches.as_ref().map(Arc::clone);
                leaf_handles.push(scope.spawn(move || {
                    d1_resident_leaf_morsel_worker_loop(
                        worker_queue,
                        worker_metrics,
                        leaf_emitter,
                        worker_store,
                        worker_sketches,
                        leaf_policy,
                    )
                }));
            }

            let resident_leaf_emitter = ResidentDatasetLeafEmitter::new(
                leaf_emitter,
                D1ResidentLeafMorselSink {
                    queue: Arc::clone(&leaf_queue),
                    metrics: Arc::clone(&leaf_metrics),
                },
            );
            let build_start = Instant::now();
            let group_stats = parallel_join_child_runs_d1_level_scan_ads_from_pairs(
                resident_store.as_ref(),
                pairs,
                depth,
                parent_n,
                metric,
                params,
                adaptive_c_max,
                min_recurse_size,
                pb,
                external_run_store,
                root_fanout_state,
                assignment_context,
                &resident_leaf_emitter,
                Duration::ZERO,
                None,
            );
            let build_wall = build_start.elapsed();
            let resident_leaf_submissions = resident_leaf_emitter.resident_leaf_submissions();
            let resident_leaf_mode = resident_leaf_emitter.resident_leaf_completion_mode();
            drop(resident_leaf_emitter);

            let morsel_stats = finish_d1_resident_leaf_morsels(leaf_handles, &leaf_metrics)?;
            let group_stats = group_stats?;
            Ok((
                group_stats,
                build_wall,
                resident_leaf_submissions,
                resident_leaf_mode,
                morsel_stats,
            ))
        })?;
    let resident_leaf_profile_after = leaf_emitter.leaf_profile_snapshot();
    let resident_leaf_profile = D1ResidentLeafProfileDelta::from_snapshots(
        resident_leaf_profile_before.as_ref(),
        resident_leaf_profile_after.as_ref(),
    );
    let resident_leaf_wait = morsel_stats.wait;
    group_stats
        .telemetry
        .point_pipeline
        .merge(pipeline_stats.clone());

    tracing::info!(
        "[adsampling/d1-resident-subtree-group] depth={} group={} source_runs={} unique_points={} selected_points={} resident_bytes={} resident_vector_bytes={} resident_sketch_bytes={} buffer_ratio={:.4} hydrate_ms={} hydrate_load_ms={} sketch_hydrate_ms={} sketch_rows={} sketch_physical_bytes={} sketch_windows={} sketch_read_amp={:.3} build_ms={} resident_leaf_wait_ms={} resident_leaf_mode={} resident_leaf_submissions={} resident_leaf_workers={} resident_leaf_queue_capacity={} resident_leaf_queue_peak={} resident_leaf_batches={} resident_leaf_drained={} resident_leaf_send_wait_ms={} resident_leaf_batch_points_peak={} resident_leaf_batch_work_peak={} resident_leaf_batch_work_limit={} resident_leaf_profile_leaves={} resident_leaf_profile_points={} resident_leaf_profile_wall_ms={} resident_leaf_profile_load_ms={} resident_leaf_profile_distance_ms={} resident_leaf_profile_topk_ms={} resident_leaf_profile_hash_ms={} resident_leaf_profile_flush_ms={} resident_leaf_profile_sketch_ms={} resident_leaf_profile_sketch_rows={} resident_leaf_ads_leaves={} resident_leaf_ads_rows={} resident_leaf_ads_layout_ms={} resident_leaf_ads_seed_ms={} resident_leaf_ads_scan_ms={} resident_leaf_ads_seed_evals={} resident_leaf_ads_full_evals={} resident_leaf_ads_pruned_evals={} resident_leaf_ads_group_evals={} resident_leaf_ads_simd_group_calls={} resident_leaf_ads_simd_active_lane_evals={} resident_leaf_ads_scalar_group_evals={}",
        depth,
        group_index,
        source_runs,
        unique_points,
        selected_points,
        resident_bytes,
        resident_vector_bytes,
        resident_sketch_bytes,
        buffer_ratio,
        hydrate_wall.as_millis(),
        load_wall.as_millis(),
        sketch_hydrate_wall.as_millis(),
        sketch_stats.rows_requested,
        sketch_stats.physical_bytes,
        sketch_stats.windows_submitted,
        if sketch_stats.logical_bytes == 0 {
            0.0
        } else {
            sketch_stats.physical_bytes as f64 / sketch_stats.logical_bytes as f64
        },
        build_wall.as_millis(),
        resident_leaf_wait.as_millis(),
        resident_leaf_mode,
        resident_leaf_submissions,
        leaf_policy.workers,
        leaf_policy.queue_capacity,
        morsel_stats.queue_depth_peak,
        morsel_stats.drain_batches,
        morsel_stats.drained_leaves,
        morsel_stats.send_wait_ms,
        morsel_stats.batch_points_peak,
        morsel_stats.batch_work_peak,
        leaf_morsel_work_to_u64(leaf_policy.max_batch_work),
        resident_leaf_profile.leaves,
        resident_leaf_profile.points,
        resident_leaf_profile.total_wall.as_millis(),
        resident_leaf_profile.load.as_millis(),
        resident_leaf_profile.distance.as_millis(),
        resident_leaf_profile.topk.as_millis(),
        resident_leaf_profile.hash.as_millis(),
        resident_leaf_profile.flush.as_millis(),
        resident_leaf_profile.sketch_prefetch.as_millis(),
        resident_leaf_profile.sketch_rows_prefetched,
        resident_leaf_profile.ads_leaves,
        resident_leaf_profile.ads_rows,
        resident_leaf_profile.ads_layout.as_millis(),
        resident_leaf_profile.ads_seed.as_millis(),
        resident_leaf_profile.ads_scan.as_millis(),
        resident_leaf_profile.ads_seed_evals,
        resident_leaf_profile.ads_full_evals,
        resident_leaf_profile.ads_pruned_evals,
        resident_leaf_profile.ads_group_evals,
        resident_leaf_profile.ads_simd_group_calls,
        resident_leaf_profile.ads_simd_active_lane_evals,
        resident_leaf_profile.ads_scalar_group_evals,
    );

    Ok(D1ResidentSubtreeGroupResult {
        stats: group_stats,
        hydrate_wall,
        build_wall,
        resident_leaf_wait,
        load_wall,
        sketch_hydrate_wall,
        sketch_stats,
        pipeline_stats,
        point_calls,
        range_calls,
        range_rows_read,
        bytes_read,
        resident_leaf_submissions,
        resident_leaf_drained: morsel_stats.drained_leaves,
        resident_leaf_batches: morsel_stats.drain_batches,
        resident_leaf_queue_depth_peak: morsel_stats.queue_depth_peak,
        resident_leaf_send_wait_ms: morsel_stats.send_wait_ms,
        resident_leaf_batch_points_peak: morsel_stats.batch_points_peak,
        resident_leaf_batch_work_peak: morsel_stats.batch_work_peak,
        resident_leaf_profile,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_d1_resident_subtree_groups(
    dataset: &dyn PointStore,
    groups: Vec<D1ResidentSubtreeGroup>,
    config: &D1ResidentSubtreeConfig,
    group_pipeline_slots: usize,
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
    let expected_leaves = groups
        .iter()
        .map(|group| group.expected_leaves)
        .sum::<usize>()
        .max(4);
    let mut stats = PartitionStats::new(params.max_depth, expected_leaves);
    if groups.is_empty() {
        return Ok(stats);
    }

    let group_count = groups.len();
    let source_runs = groups.iter().map(|group| group.source_runs).sum::<usize>();
    let selected_points = groups
        .iter()
        .map(|group| group.selected_points)
        .sum::<usize>();
    let selected_unique_points = groups
        .iter()
        .map(|group| group.unique_ids.len())
        .sum::<usize>();
    let global_sketches = leaf_emitter.sketch_accessor();
    let sketch_width = global_sketches
        .map(|sketches| sketches.width())
        .unwrap_or(0);
    let logical_bytes = groups.iter().map(|group| group.logical_bytes).sum::<u64>();
    let physical_bytes = groups.iter().map(|group| group.physical_bytes).sum::<u64>();
    let resident_bytes = groups
        .iter()
        .map(|group| group.resident_bytes)
        .sum::<usize>();
    let planned_windows = groups
        .iter()
        .map(|group| group.planned_windows)
        .sum::<usize>();
    let min_buffer_ratio = groups
        .iter()
        .map(|group| group.buffer_ratio)
        .fold(1.0_f64, f64::min);
    let avg_buffer_ratio = if physical_bytes == 0 {
        1.0
    } else {
        logical_bytes as f64 / physical_bytes as f64
    };
    let max_resident_bytes = config.max_resident_bytes_for(parent_n, dataset.dim());
    let limiter_budget_bytes = config.limiter_budget_bytes_for(parent_n, dataset.dim());
    let grouped_budget_bytes = d1_resident_subtree_vector_budget_for_total_resident_budget(
        config.grouped_budget_bytes_for(parent_n, dataset.dim()),
        dataset.dim(),
        sketch_width,
    );
    let requested_group_pipeline_slots = group_pipeline_slots
        .max(1)
        .min(config.group_pipeline_slots.max(1));
    let effective_group_pipeline_slots = requested_group_pipeline_slots
        .min(group_count.max(1))
        .max(1);
    let planned_group_budget_bytes = if effective_group_pipeline_slots > 1 {
        d1_resident_subtree_vector_budget_for_total_resident_budget(
            config.pipelined_group_budget_bytes_for(parent_n, dataset.dim()),
            dataset.dim(),
            sketch_width,
        )
    } else {
        grouped_budget_bytes
    };
    let ready_queue_capacity = effective_group_pipeline_slots.max(1);
    tracing::info!(
        "[adsampling/d1-resident-subtree-plan] depth={} groups={} source_runs={} selected_points={} selected_unique_points={} resident_bytes={} logical_bytes={} physical_bytes={} planned_windows={} avg_buffer_ratio={:.4} min_buffer_ratio={:.4} max_resident_ratio={:.4} max_resident_bytes={} effective_group_budget_bytes={} planned_group_budget_bytes={} effective_limiter_budget_bytes={} budget_bytes={} max_group_bytes={} scope_budget_bytes={} io_threads={} group_pipeline_slots={} requested_group_pipeline_slots={} ready_queue_capacity={} enable_source={} consume_scope=d1-subtree",
        depth,
        group_count,
        source_runs,
        selected_points,
        selected_unique_points,
        resident_bytes,
        logical_bytes,
        physical_bytes,
        planned_windows,
        avg_buffer_ratio,
        min_buffer_ratio,
        config.max_dataset_resident_ratio,
        max_resident_bytes,
        grouped_budget_bytes,
        planned_group_budget_bytes,
        limiter_budget_bytes,
        config.budget_bytes,
        config.max_run_bytes,
        config.whole_scope_budget_bytes,
        config.io_threads,
        effective_group_pipeline_slots,
        requested_group_pipeline_slots,
        ready_queue_capacity,
        config.enable_source,
    );

    let limiter = Arc::new(D1ResidentSubtreeLimiter::new(limiter_budget_bytes));
    let total_start = Instant::now();
    let row_bytes = dataset.dim().max(1).saturating_mul(size_of::<f32>());
    let mut hydrate_wall = Duration::ZERO;
    let mut build_wall = Duration::ZERO;
    let mut resident_leaf_wait = Duration::ZERO;
    let mut load_wall = Duration::ZERO;
    let mut sketch_hydrate_wall = Duration::ZERO;
    let mut sketch_stats = WindowedGatherStats::default();
    let mut point_pipeline = PointPipelineStats::default();
    let mut point_calls = 0_u64;
    let mut range_calls = 0_u64;
    let mut range_rows_read = 0_u64;
    let mut bytes_read = 0_u64;
    let mut resident_leaf_submissions = 0usize;
    let mut resident_leaf_drained = 0usize;
    let mut resident_leaf_batches = 0usize;
    let mut resident_leaf_queue_depth_peak = 0usize;
    let mut resident_leaf_send_wait_ms = 0u64;
    let mut resident_leaf_batch_points_peak = 0usize;
    let mut resident_leaf_batch_work_peak = 0u64;
    let mut resident_leaf_profile = D1ResidentLeafProfileDelta::default();
    let pipeline_telemetry = Arc::new(D1ResidentSubtreePipelineTelemetry::default());

    macro_rules! record_group_result {
        ($result:expr) => {{
            let result = $result;
            hydrate_wall += result.hydrate_wall;
            build_wall += result.build_wall;
            resident_leaf_wait += result.resident_leaf_wait;
            load_wall += result.load_wall;
            sketch_hydrate_wall += result.sketch_hydrate_wall;
            sketch_stats.merge(&result.sketch_stats);
            point_pipeline.merge(result.pipeline_stats);
            point_calls = point_calls.saturating_add(result.point_calls);
            range_calls = range_calls.saturating_add(result.range_calls);
            range_rows_read = range_rows_read.saturating_add(result.range_rows_read);
            bytes_read = bytes_read.saturating_add(result.bytes_read);
            resident_leaf_submissions =
                resident_leaf_submissions.saturating_add(result.resident_leaf_submissions);
            resident_leaf_drained =
                resident_leaf_drained.saturating_add(result.resident_leaf_drained);
            resident_leaf_batches =
                resident_leaf_batches.saturating_add(result.resident_leaf_batches);
            resident_leaf_queue_depth_peak =
                resident_leaf_queue_depth_peak.max(result.resident_leaf_queue_depth_peak);
            resident_leaf_send_wait_ms =
                resident_leaf_send_wait_ms.saturating_add(result.resident_leaf_send_wait_ms);
            resident_leaf_batch_points_peak =
                resident_leaf_batch_points_peak.max(result.resident_leaf_batch_points_peak);
            resident_leaf_batch_work_peak =
                resident_leaf_batch_work_peak.max(result.resident_leaf_batch_work_peak);
            resident_leaf_profile.merge(&result.resident_leaf_profile);
            stats.merge_from(result.stats);
        }};
    }

    if effective_group_pipeline_slots <= 1 {
        for group in groups {
            let hydrated = hydrate_d1_resident_subtree_group(
                dataset,
                global_sketches,
                group,
                config,
                params,
                &limiter,
                depth,
                row_bytes,
            )?;
            let result = build_d1_resident_subtree_group(
                hydrated,
                depth,
                parent_n,
                metric,
                params,
                adaptive_c_max,
                min_recurse_size,
                pb,
                external_run_store,
                root_fanout_state,
                assignment_context,
                leaf_emitter,
            )?;
            record_group_result!(result);
        }
    } else {
        std::thread::scope(|scope| -> AnnResult<()> {
            let (hydrated_tx, hydrated_rx) = crossbeam_channel::bounded::<
                AnnResult<D1ResidentSubtreeHydratedGroup>,
            >(ready_queue_capacity);
            let abort_pipeline = Arc::new(AtomicBool::new(false));
            let producer_limiter = Arc::clone(&limiter);
            let producer_telemetry = Arc::clone(&pipeline_telemetry);
            let producer_abort = Arc::clone(&abort_pipeline);
            let producer_handle = scope.spawn(move || {
                for group in groups {
                    if producer_abort.load(Ordering::Acquire) {
                        break;
                    }
                    let hydrated = hydrate_d1_resident_subtree_group(
                        dataset,
                        global_sketches,
                        group,
                        config,
                        params,
                        &producer_limiter,
                        depth,
                        row_bytes,
                    );
                    let hydrated_ok = hydrated.is_ok();
                    let send_start = Instant::now();
                    let mut pending = hydrated;
                    loop {
                        if producer_abort.load(Ordering::Acquire) {
                            return;
                        }
                        match hydrated_tx.send_timeout(pending, Duration::from_millis(100)) {
                            Ok(()) => break,
                            Err(crossbeam_channel::SendTimeoutError::Timeout(next_pending)) => {
                                pending = next_pending;
                            }
                            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => return,
                        }
                    }
                    producer_telemetry
                        .hydrate_send_wait_ns
                        .fetch_add(duration_nanos_u64(send_start.elapsed()), Ordering::Relaxed);
                    update_atomic_max_usize(
                        &producer_telemetry.ready_queue_depth_peak,
                        hydrated_tx.len(),
                    );
                    if !hydrated_ok {
                        break;
                    }
                }
            });

            let mut pipeline_error = None;
            for expected_group_index in 0..group_count {
                let recv_start = Instant::now();
                let hydrated = match hydrated_rx.recv() {
                    Ok(hydrated) => hydrated,
                    Err(_) => {
                        pipeline_error = Some(AnnError::log_index_error(
                            "D1 resident subtree hydrate queue closed before all groups built"
                                .to_string(),
                        ));
                        break;
                    }
                };
                pipeline_telemetry
                    .build_recv_wait_ns
                    .fetch_add(duration_nanos_u64(recv_start.elapsed()), Ordering::Relaxed);
                let hydrated = match hydrated {
                    Ok(hydrated) => hydrated,
                    Err(err) => {
                        pipeline_error = Some(err);
                        break;
                    }
                };
                if hydrated.group_index != expected_group_index {
                    pipeline_error = Some(AnnError::log_index_error(format!(
                        "D1 resident subtree group order violation: expected {}, got {}",
                        expected_group_index, hydrated.group_index
                    )));
                    break;
                }
                match build_d1_resident_subtree_group(
                    hydrated,
                    depth,
                    parent_n,
                    metric,
                    params,
                    adaptive_c_max,
                    min_recurse_size,
                    pb,
                    external_run_store,
                    root_fanout_state,
                    assignment_context,
                    leaf_emitter,
                ) {
                    Ok(result) => record_group_result!(result),
                    Err(err) => {
                        pipeline_error = Some(err);
                        break;
                    }
                }
            }
            drop(hydrated_rx);
            if pipeline_error.is_some() {
                abort_pipeline.store(true, Ordering::Release);
            }
            producer_handle.join().map_err(|_| {
                AnnError::log_index_error(
                    "D1 resident subtree hydrate producer thread panicked".to_string(),
                )
            })?;
            match pipeline_error {
                Some(err) => Err(err),
                None => Ok(()),
            }
        })?;
    }

    let limiter_state = limiter.snapshot();
    let hydrate_send_wait_ms = pipeline_telemetry
        .hydrate_send_wait_ns
        .load(Ordering::Relaxed)
        / 1_000_000;
    let build_recv_wait_ms = pipeline_telemetry
        .build_recv_wait_ns
        .load(Ordering::Relaxed)
        / 1_000_000;
    let ready_queue_depth_peak = pipeline_telemetry
        .ready_queue_depth_peak
        .load(Ordering::Relaxed);
    tracing::info!(
        "[adsampling/d1-resident-subtree-pipeline] depth={} requested_slots={} effective_slots={} ready_queue_capacity={} hydrate_send_wait_ms={} build_wait_ms={} ready_queue_depth_peak={} resident_wait_ms={} resident_peak_bytes={}",
        depth,
        requested_group_pipeline_slots,
        effective_group_pipeline_slots,
        ready_queue_capacity,
        hydrate_send_wait_ms,
        build_recv_wait_ms,
        ready_queue_depth_peak,
        limiter_state.wait_ns / 1_000_000,
        limiter_state.peak_live_bytes,
    );
    tracing::info!(
        "[adsampling/d1-resident-subtree] depth={} groups={} source_runs={} points={} unique_points={} elapsed_ms={} hydrate_ms={} hydrate_load_ms={} sketch_hydrate_ms={} sketch_rows={} sketch_physical_bytes={} sketch_windows={} sketch_read_amp={:.3} build_ms={} resident_leaf_wait_ms={} point_pipeline_batches={} point_pipeline_physical_bytes={} point_pipeline_read_amp={:.3} point_calls={} range_calls={} range_rows_read={} bytes_read={} resident_leaf_submissions={} resident_leaf_drained={} resident_leaf_batches={} resident_leaf_queue_depth_peak={} resident_leaf_send_wait_ms={} resident_leaf_batch_points_peak={} resident_leaf_batch_work_peak={} resident_leaf_profile_leaves={} resident_leaf_profile_points={} resident_leaf_profile_wall_ms={} resident_leaf_profile_load_ms={} resident_leaf_profile_distance_ms={} resident_leaf_profile_topk_ms={} resident_leaf_profile_hash_ms={} resident_leaf_profile_flush_ms={} resident_leaf_profile_sketch_ms={} resident_leaf_profile_sketch_rows={} resident_leaf_ads_leaves={} resident_leaf_ads_rows={} resident_leaf_ads_layout_ms={} resident_leaf_ads_seed_ms={} resident_leaf_ads_scan_ms={} resident_leaf_ads_seed_evals={} resident_leaf_ads_full_evals={} resident_leaf_ads_pruned_evals={} resident_leaf_ads_group_evals={} resident_leaf_ads_simd_group_calls={} resident_leaf_ads_simd_active_lane_evals={} resident_leaf_ads_scalar_group_evals={} resident_peak_bytes={} resident_wait_count={} resident_wait_ms={} group_pipeline_slots={} ready_queue_depth_peak={}",
        depth,
        group_count,
        source_runs,
        selected_points,
        selected_unique_points,
        total_start.elapsed().as_millis(),
        hydrate_wall.as_millis(),
        load_wall.as_millis(),
        sketch_hydrate_wall.as_millis(),
        sketch_stats.rows_requested,
        sketch_stats.physical_bytes,
        sketch_stats.windows_submitted,
        if sketch_stats.logical_bytes == 0 {
            0.0
        } else {
            sketch_stats.physical_bytes as f64 / sketch_stats.logical_bytes as f64
        },
        build_wall.as_millis(),
        resident_leaf_wait.as_millis(),
        point_pipeline.batches,
        point_pipeline.physical_bytes,
        point_pipeline.read_amplification(),
        point_calls,
        range_calls,
        range_rows_read,
        bytes_read,
        resident_leaf_submissions,
        resident_leaf_drained,
        resident_leaf_batches,
        resident_leaf_queue_depth_peak,
        resident_leaf_send_wait_ms,
        resident_leaf_batch_points_peak,
        resident_leaf_batch_work_peak,
        resident_leaf_profile.leaves,
        resident_leaf_profile.points,
        resident_leaf_profile.total_wall.as_millis(),
        resident_leaf_profile.load.as_millis(),
        resident_leaf_profile.distance.as_millis(),
        resident_leaf_profile.topk.as_millis(),
        resident_leaf_profile.hash.as_millis(),
        resident_leaf_profile.flush.as_millis(),
        resident_leaf_profile.sketch_prefetch.as_millis(),
        resident_leaf_profile.sketch_rows_prefetched,
        resident_leaf_profile.ads_leaves,
        resident_leaf_profile.ads_rows,
        resident_leaf_profile.ads_layout.as_millis(),
        resident_leaf_profile.ads_seed.as_millis(),
        resident_leaf_profile.ads_scan.as_millis(),
        resident_leaf_profile.ads_seed_evals,
        resident_leaf_profile.ads_full_evals,
        resident_leaf_profile.ads_pruned_evals,
        resident_leaf_profile.ads_group_evals,
        resident_leaf_profile.ads_simd_group_calls,
        resident_leaf_profile.ads_simd_active_lane_evals,
        resident_leaf_profile.ads_scalar_group_evals,
        limiter_state.peak_live_bytes,
        limiter_state.wait_count,
        limiter_state.wait_ns / 1_000_000,
        effective_group_pipeline_slots,
        ready_queue_depth_peak,
    );

    Ok(stats)
}

#[derive(Clone)]
pub(crate) struct AssignmentContext<'a> {
    depth: usize,
    params: ForgeANNParams,
    adsampling_dataset: Option<&'a dyn PointStore>,
    pub(crate) scheduler_runtime: Arc<AdSamplingSchedulerRuntime>,
    strict_prefetch_gate: Option<Arc<StrictPrefetchGate>>,
    io_pain: Option<Arc<Mutex<IoPlannerStats>>>,
    depth_wave_assignment: bool,
}

impl<'a> AssignmentContext<'a> {
    pub(crate) fn new_for_assignment_with_adsampling_dataset(
        depth: usize,
        params: &ForgeANNParams,
        adsampling_dataset: Option<&'a dyn PointStore>,
    ) -> Self {
        Self {
            depth,
            params: params.clone(),
            adsampling_dataset,
            scheduler_runtime: Arc::new(AdSamplingSchedulerRuntime::default()),
            strict_prefetch_gate: params.strict_oom_prefetch_pipeline_enabled().then(|| {
                Arc::new(StrictPrefetchGate::new(
                    params.effective_oom_strict_prefetch_budget_bytes(),
                ))
            }),
            io_pain: params
                .io_planned_forgeann_enabled()
                .then(|| Arc::new(Mutex::new(IoPlannerStats::default()))),
            depth_wave_assignment: false,
        }
    }

    fn for_depth(&self, depth: usize) -> Self {
        Self {
            depth,
            params: self.params.clone(),
            adsampling_dataset: self.adsampling_dataset,
            scheduler_runtime: Arc::clone(&self.scheduler_runtime),
            strict_prefetch_gate: self.strict_prefetch_gate.clone(),
            io_pain: self.io_pain.clone(),
            depth_wave_assignment: false,
        }
    }

    pub(crate) fn with_depth_wave_assignment(mut self) -> Self {
        self.depth_wave_assignment = true;
        self
    }

    pub(crate) fn ads_scheduler_stats(&self) -> AdSamplingSchedulerStats {
        self.scheduler_runtime.snapshot()
    }

    fn record_ads_exact_fallback(&self, reason: Option<&str>) {
        self.scheduler_runtime.record_exact_fallback_reason(reason);
    }

    pub(crate) fn record_adsampling_scheduler_profile(&self, profile: &AdSamplingProfile) {
        self.scheduler_runtime
            .record_ads_profile(&self.params, profile);
    }

    fn record_depth_io_pain_from_profiles(
        &self,
        depth: usize,
        point_pipeline: &PointPipelineStats,
        prefetch: &StrictPrefetchPipelineProfile,
    ) {
        let Some(io_pain) = &self.io_pain else {
            return;
        };
        let config = io_plan_config_for_params(&self.params);
        let mut guard = io_pain.lock();
        guard.record_config(config);
        guard.record_depth_point_pipeline(depth, point_pipeline);
        let read_calls = prefetch
            .range_reads
            .saturating_add(prefetch.point_reads)
            .min(usize::MAX as u64) as usize;
        guard.record_depth_io_pain(
            depth,
            read_calls,
            prefetch.batches,
            prefetch.logical_bytes,
            prefetch.physical_bytes,
            prefetch.io_wall.as_secs_f64() * 1000.0,
            prefetch.consumer_wait.as_secs_f64() * 1000.0,
            prefetch.producer_wait.as_secs_f64() * 1000.0,
            prefetch.prefetch_used_peak_bytes,
        );
    }

    pub(crate) fn observed_io_pain_for_depth(&self, depth: usize) -> Option<IoPainSample> {
        self.io_pain
            .as_ref()
            .and_then(|io_pain| io_pain.lock().pain_sample_for_depth(depth))
    }

    pub(crate) fn merge_observed_io_pain_into(&self, stats: &mut IoPlannerStats) {
        let Some(io_pain) = &self.io_pain else {
            return;
        };
        let guard = io_pain.lock();
        stats.merge_depth_io_pain_from(&guard);
    }

    fn depth_wave_ads_eligible(&self, points: usize, leaders: usize, fanout: usize) -> bool {
        if !self.depth_wave_assignment
            || self.depth == 0
            || self.depth > ForgeANNParams::ADS_WAVE_DEPTHS
            || self.depth > ForgeANNParams::ADS_DEPTH_MAX_DEPTH
            || points < 1
            || leaders < 1
            || (points as u128).saturating_mul(leaders as u128) < 1
            || fanout == 0
            || self.params.adsampling_group_dims == 0
        {
            return false;
        }
        true
    }

    pub(crate) fn should_use_adsampling_assignment(
        &self,
        points: usize,
        leaders: usize,
        fanout: usize,
    ) -> bool {
        if self.depth_wave_ads_eligible(points, leaders, fanout) {
            return true;
        }
        if self.depth > 0 && self.scheduler_runtime.depth_ads_disabled() {
            return false;
        }
        should_use_adsampling_assignment(&self.params, self.depth, points, leaders, fanout)
    }

    pub(crate) fn assignment_gemm_fallback_reason(
        &self,
        points: usize,
        leaders: usize,
        fanout: usize,
    ) -> Option<&'static str> {
        if self.depth_wave_ads_eligible(points, leaders, fanout) {
            return None;
        }
        if self.depth > 0 && self.scheduler_runtime.depth_ads_disabled() {
            return Some("scheduler-collapse");
        }
        adsampling_depth_fallback_reason(&self.params, self.depth, points, leaders, fanout)
    }

    fn adsampling_dataset_for_assignment<'b>(
        &'b self,
        default_dataset: &'b dyn PointStore,
    ) -> &'b dyn PointStore {
        if self.depth == 0 {
            self.adsampling_dataset.unwrap_or(default_dataset)
        } else {
            default_dataset
        }
    }

    pub(crate) fn strict_prefetch_config(&self) -> Option<StrictPrefetchPipelineConfig> {
        self.params
            .strict_oom_prefetch_pipeline_enabled()
            .then(|| StrictPrefetchPipelineConfig {
                enabled: true,
                queue_depth: ForgeANNParams::OOM_STRICT_PREFETCH_QUEUE_DEPTH,
                budget_bytes: self.params.effective_oom_strict_prefetch_budget_bytes(),
                max_window_bytes: ForgeANNParams::OOM_STRICT_PREFETCH_MAX_WINDOW_BYTES,
                max_gap_rows: ForgeANNParams::OOM_STRICT_PREFETCH_MAX_GAP_ROWS,
            })
    }

    fn strict_prefetch_gate(&self) -> Option<&StrictPrefetchGate> {
        self.strict_prefetch_gate.as_deref()
    }
}

pub(crate) struct LoadedAdSamplingBlock {
    pub block_start: usize,
    pub result: AdSamplingChunkResult,
    pub wall: Duration,
}

pub(crate) fn update_atomic_max_usize(max_value: &AtomicUsize, value: usize) {
    let mut current = max_value.load(Ordering::Relaxed);
    while value > current {
        match max_value.compare_exchange_weak(current, value, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

pub(crate) fn update_atomic_max_u64(max_value: &AtomicU64, value: u64) {
    let mut current = max_value.load(Ordering::Relaxed);
    while value > current {
        match max_value.compare_exchange_weak(current, value, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

pub(crate) fn merge_adsampling_chunk_result_into_profile(
    profile: &mut AdSamplingProfile,
    result: &AdSamplingChunkResult,
    chunk_wall: Duration,
) {
    profile.chunk_wall_accumulated_ms += duration_ms(chunk_wall);
    profile.chunk_wall_max_ms = profile.chunk_wall_max_ms.max(duration_ms(chunk_wall));
    profile.seed_ms += duration_ms(result.seed);
    profile.seed_accumulated_ms += duration_ms(result.seed);
    profile.seed_wall_ms = profile.seed_wall_ms.max(duration_ms(result.seed));
    profile.scan_ms += duration_ms(result.scan);
    profile.scan_accumulated_ms += duration_ms(result.scan);
    profile.full_evals += result.full_evals;
    profile.pruned_evals += result.pruned_evals;
    profile.group_evals += result.group_evals;
    profile.simd_group_calls += result.simd_group_calls;
    profile.simd_active_lane_evals += result.simd_active_lane_evals;
    profile.scalar_group_evals += result.scalar_group_evals;
    profile.validation_sources += result.validation_sources;
    profile.validation_mismatches += result.validation_mismatches;
    profile.validation_recall_hits += result.validation_recall_hits;
    profile.validation_recall_total += result.validation_recall_total;
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assign_point_leaders_adsampling_prefetched_with_context(
    dataset: &dyn PointStore,
    cur: &[u32],
    leaders: &[u32],
    local_fanout: usize,
    params: &ForgeANNParams,
    depth: usize,
    memory_budget_bytes: Option<usize>,
    strict_config: Option<StrictPrefetchPipelineConfig>,
    strict_gate: Option<&StrictPrefetchGate>,
    mut visit_chunk: impl FnMut(&AdSamplingPointAssignmentChunk) -> AnnResult<()> + Send,
) -> AnnResult<(AdSamplingProfile, StrictPrefetchPipelineProfile)> {
    if leaders.is_empty() {
        return Err(crate::common::AnnError::log_index_error(
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
    let num_workers = rayon::current_num_threads().max(1);
    let (point_tile_size, _) =
        choose_gemm_tile_sizes_with_override_and_budget(leaders.len(), dim, memory_budget_bytes);
    let batch_points = choose_prefetch_batch_points_for_budget(
        point_tile_size,
        dim,
        choose_spool_prefetch_workers(num_workers, memory_budget_bytes),
        memory_budget_bytes,
    );
    let point_read_options = default_rbc_windowed_options(dataset, point_tile_size);
    let mut prefetch_profile = StrictPrefetchPipelineProfile::default();
    let mut profile = AdSamplingProfile {
        depth,
        points: cur.len(),
        leaders: leaders.len(),
        fanout,
        epsilon: config.epsilon,
        group_dims: config.group_dims,
        seed_exact_m: seed_count,
        layout_ms: duration_ms(layout_elapsed),
        called_inside_rayon_worker: rayon::current_thread_index().is_some(),
        scheduler_mode: "depth-wave".to_string(),
        ..AdSamplingProfile::default()
    };

    let active_chunks = Arc::new(AtomicUsize::new(0));
    let active_chunk_max = Arc::new(AtomicUsize::new(0));
    let active_chunk_start_sum = Arc::new(AtomicUsize::new(0));
    let completed_chunks = Arc::new(AtomicUsize::new(0));
    let completed_rows = Arc::new(AtomicUsize::new(0));
    let completed_batches = Arc::new(AtomicUsize::new(0));
    let mut next_source_start = 0usize;
    std::thread::scope(|scope| -> AnnResult<()> {
        let (progress_stop_tx, progress_stop_rx) = crossbeam_channel::bounded::<()>(1);
        let progress_active_chunks = Arc::clone(&active_chunks);
        let progress_completed_chunks = Arc::clone(&completed_chunks);
        let progress_completed_rows = Arc::clone(&completed_rows);
        let progress_completed_batches = Arc::clone(&completed_batches);
        let progress_start = total_start;
        let progress_handle = scope.spawn(move || {
            loop {
                match progress_stop_rx.recv_timeout(Duration::from_secs(30)) {
                    Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                }
                let rows_done = progress_completed_rows.load(Ordering::Relaxed);
                tracing::info!(
                    "[adsampling/assign-progress] depth={} points={} leaders={} fanout={} scheduler_mode=depth-wave completed_rows={} completed_chunks={} completed_batches={} active_chunks={} elapsed_ms={}",
                    depth,
                    cur.len(),
                    leaders.len(),
                    fanout,
                    rows_done,
                    progress_completed_chunks.load(Ordering::Relaxed),
                    progress_completed_batches.load(Ordering::Relaxed),
                    progress_active_chunks.load(Ordering::Relaxed),
                    progress_start.elapsed().as_millis(),
                );
            }
        });

        let assign_result = for_each_prefetched_point_batch_profiled(
            dataset,
            cur,
            batch_points,
            choose_prefetch_queue_depth_for_budget(batch_points, dim, memory_budget_bytes),
            point_read_options,
            strict_config,
            strict_gate,
            Some(&mut prefetch_profile),
            |prefetched| {
                let source_start = next_source_start;
                next_source_start += prefetched.len;
                let compute_block_points = choose_compute_block_points_for_batch(
                    point_tile_size,
                    prefetched.len,
                    num_workers,
                );
                profile.chunks += prefetched.len.div_ceil(compute_block_points);

                let mut blocks = (0..prefetched.len)
                    .into_par_iter()
                    .step_by(compute_block_points)
                    .map(|block_start| {
                        let block_end = (block_start + compute_block_points).min(prefetched.len);
                        let point_ids = &prefetched.ids[block_start..block_end];
                        let point_data = &prefetched.data[block_start * dim..block_end * dim];
                        let active = active_chunks.fetch_add(1, Ordering::AcqRel) + 1;
                        update_atomic_max_usize(&active_chunk_max, active);
                        active_chunk_start_sum.fetch_add(active, Ordering::Relaxed);
                        let block_timer = Instant::now();
                        let result = assign_loaded_point_chunk_adsampling_layout(
                            point_ids,
                            point_data,
                            source_start + block_start,
                            dim,
                            &leader_layout,
                            fanout,
                            config,
                            &seed_indices,
                            &seeded,
                        );
                        let wall = block_timer.elapsed();
                        active_chunks.fetch_sub(1, Ordering::AcqRel);
                        if result.is_ok() {
                            completed_chunks.fetch_add(1, Ordering::Relaxed);
                            completed_rows.fetch_add(point_ids.len(), Ordering::Relaxed);
                        }
                        result.map(|result| LoadedAdSamplingBlock {
                            block_start,
                            result,
                            wall,
                        })
                    })
                    .collect::<AnnResult<Vec<_>>>()?;
                blocks.sort_unstable_by_key(|block| block.block_start);
                profile.ordered_pending_max = profile.ordered_pending_max.max(blocks.len());

                for block in blocks {
                    merge_adsampling_chunk_result_into_profile(
                        &mut profile,
                        &block.result,
                        block.wall,
                    );
                    let visit_start = Instant::now();
                    visit_chunk(&block.result.chunk)?;
                    profile.visit_chunk_ms += duration_ms(visit_start.elapsed());
                }
                completed_batches.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        );
        let _ = progress_stop_tx.send(());
        progress_handle.join().map_err(|_| {
            AnnError::log_index_error("ADSampling assignment progress thread panicked".to_string())
        })?;
        assign_result
    })?;

    profile.chunk_active_max = active_chunk_max.load(Ordering::Relaxed);
    let active_start_sum = active_chunk_start_sum.load(Ordering::Relaxed);
    if profile.chunks > 0 {
        profile.chunk_active_start_avg = active_start_sum as f64 / profile.chunks as f64;
    }
    profile.validation_recall_at_fanout = if profile.validation_recall_total == 0 {
        0.0
    } else {
        profile.validation_recall_hits as f64 / profile.validation_recall_total as f64
    };
    profile.total_ms = duration_ms(total_start.elapsed());
    if profile.total_ms > 0.0 {
        profile.effective_parallelism = profile.chunk_wall_accumulated_ms / profile.total_ms;
    }

    Ok((profile, prefetch_profile))
}

pub(crate) fn log_adsampling_profile(profile: &AdSamplingProfile) {
    tracing::info!(
        "[adsampling/assign] depth={} points={} leaders={} fanout={} epsilon={:.3} group_dims={} seed_exact_m={} scheduler_mode={} fallback_reason={} total_ms={:.3} effective_parallelism={:.3} layout_ms={:.3} seed_wall_ms={:.3} seed_accum_ms={:.3} scan_accum_ms={:.3} chunks={} inside_rayon={} chunk_wall_accum_ms={:.3} chunk_wall_max_ms={:.3} chunk_send_block_ms={:.3} chunk_active_max={} chunk_active_start_avg={:.3} recv_wait_ms={:.3} recv_empty_polls={} ordered_pending_max={} read_ms={:.3} compute_ms={:.3} apply_ms={:.3} loaded_queue_depth_max={} computed_queue_depth_max={} visit_chunk_ms={:.3} full_evals={} pruned_evals={} group_evals={} simd_group_calls={} simd_active_lane_evals={} scalar_group_evals={} validation_sources={} recall_at_fanout={:.4} mismatches={} point_pipeline_batches={} point_pipeline_ready_peak={} point_pipeline_read_amp={:.3} point_pipeline_consumer_wait_ms={} point_pipeline_producer_wait_ms={} point_pipeline_budget_wait_ms={} point_pipeline_avg_read_size={:.1}",
        profile.depth,
        profile.points,
        profile.leaders,
        profile.fanout,
        profile.epsilon,
        profile.group_dims,
        profile.seed_exact_m,
        profile.scheduler_mode,
        profile.fallback_reason.as_deref().unwrap_or("none"),
        profile.total_ms,
        profile.effective_parallelism,
        profile.layout_ms,
        profile.seed_wall_ms,
        profile.seed_accumulated_ms,
        profile.scan_accumulated_ms,
        profile.chunks,
        profile.called_inside_rayon_worker,
        profile.chunk_wall_accumulated_ms,
        profile.chunk_wall_max_ms,
        profile.chunk_send_block_ms,
        profile.chunk_active_max,
        profile.chunk_active_start_avg,
        profile.recv_wait_ms,
        profile.recv_empty_polls,
        profile.ordered_pending_max,
        profile.read_ms,
        profile.compute_ms,
        profile.apply_ms,
        profile.loaded_queue_depth_max,
        profile.computed_queue_depth_max,
        profile.visit_chunk_ms,
        profile.full_evals,
        profile.pruned_evals,
        profile.group_evals,
        profile.simd_group_calls,
        profile.simd_active_lane_evals,
        profile.scalar_group_evals,
        profile.validation_sources,
        profile.validation_recall_at_fanout,
        profile.validation_mismatches,
        profile.point_pipeline.batches,
        profile.point_pipeline.ready_queue_depth_peak,
        profile.point_pipeline.read_amplification(),
        profile.point_pipeline.consumer_wait.as_millis(),
        profile.point_pipeline.producer_wait.as_millis(),
        profile.point_pipeline.budget_wait.as_millis(),
        profile.point_pipeline.avg_read_size_bytes(),
    );
}

pub(crate) fn log_adsampling_depth_fallback(
    context: &AssignmentContext<'_>,
    points: usize,
    leaders: usize,
    fanout: usize,
    path: &'static str,
) {
    if context.depth == 0 {
        return;
    }
    if let Some(reason) = context.assignment_gemm_fallback_reason(points, leaders, fanout) {
        tracing::debug!(
            "[adsampling/depth-gate] path={} depth={} points={} leaders={} fanout={} work={} reason={}",
            path,
            context.depth,
            points,
            leaders,
            fanout,
            (points as u128).saturating_mul(leaders as u128),
            reason
        );
    }
}

pub(crate) fn assignment_gemm_fallback_reason(
    context: &AssignmentContext<'_>,
    points: usize,
    leaders: usize,
    fanout: usize,
) -> Option<&'static str> {
    if context.depth == 0 {
        return None;
    }
    context.assignment_gemm_fallback_reason(points, leaders, fanout)
}

impl GemmFoldState {
    pub(crate) fn new(
        point_tile: usize,
        leader_tile: usize,
        dim: usize,
        nl: usize,
        fanout: usize,
    ) -> Self {
        Self {
            scratch: GemmScratch::new(point_tile, leader_tile, dim, nl, fanout),
            clusters: (0..nl).map(|_| Vec::new()).collect(),
            profile: GemmProfile::default(),
        }
    }
}

pub(crate) fn merge_cluster_buffers(dst: &mut [Vec<u32>], src: Vec<Vec<u32>>) {
    assert_eq!(dst.len(), src.len());
    for (dst_cluster, mut src_cluster) in dst.iter_mut().zip(src) {
        dst_cluster.append(&mut src_cluster);
    }
}

pub(crate) fn merge_cluster_buffers_in_place(dst: &mut [Vec<u32>], src: &mut [Vec<u32>]) {
    assert_eq!(dst.len(), src.len());
    for (dst_cluster, src_cluster) in dst.iter_mut().zip(src.iter_mut()) {
        dst_cluster.append(src_cluster);
    }
}

pub(crate) fn borrow_gram_tile(
    gram_data: &mut Vec<f32>,
    rows: usize,
    cols: usize,
) -> ArrayViewMut2<'_, f32> {
    let gram_len = rows * cols;
    gram_data.resize(gram_len, 0.0);
    gram_data[..gram_len].fill(0.0);
    ArrayViewMut2::from_shape((rows, cols), &mut gram_data[..gram_len]).unwrap()
}

/// GEMM-based cluster assignment for L2 metric when n >= 256.
///
/// Processes points in blocks to avoid OOM on large inputs (e.g. 35M × 10K leaders).
/// Each block is processed by a rayon thread: build local X matrix, GEMM against shared
/// leader matrix, top-k assignment, then merge into global clusters.
pub(crate) fn compute_clusters_gemm(
    dataset: &dyn PointStore,
    cur: &[u32],
    leaders: &[u32],
    local_fanout: usize,
) -> AnnResult<(Vec<Vec<u32>>, GemmProfile)> {
    compute_clusters_gemm_budgeted(dataset, cur, leaders, local_fanout, None)
}

pub(crate) fn compute_clusters_gemm_budgeted(
    dataset: &dyn PointStore,
    cur: &[u32],
    leaders: &[u32],
    local_fanout: usize,
    memory_budget_bytes: Option<usize>,
) -> AnnResult<(Vec<Vec<u32>>, GemmProfile)> {
    compute_clusters_gemm_budgeted_with_context(
        dataset,
        cur,
        leaders,
        local_fanout,
        memory_budget_bytes,
        None,
    )
}

pub(crate) fn compute_clusters_gemm_budgeted_with_context(
    dataset: &dyn PointStore,
    cur: &[u32],
    leaders: &[u32],
    local_fanout: usize,
    memory_budget_bytes: Option<usize>,
    assignment_context: Option<&AssignmentContext<'_>>,
) -> AnnResult<(Vec<Vec<u32>>, GemmProfile)> {
    if let Some(context) = assignment_context {
        let use_adsampling =
            context.should_use_adsampling_assignment(cur.len(), leaders.len(), local_fanout);
        if use_adsampling {
            let fanout = assign_record_fanout(local_fanout.min(leaders.len().max(1)));
            let mut clusters = (0..leaders.len()).map(|_| Vec::new()).collect::<Vec<_>>();
            let adsampling_dataset = context.adsampling_dataset_for_assignment(dataset);
            let (adsampling_profile, prefetch_profile) = if context.depth > 0 {
                assign_point_leaders_adsampling_prefetched_with_context(
                    adsampling_dataset,
                    cur,
                    leaders,
                    fanout,
                    &context.params,
                    context.depth,
                    memory_budget_bytes,
                    context.strict_prefetch_config(),
                    context.strict_prefetch_gate(),
                    |chunk| {
                        for (local_idx, leaders_for_point) in
                            chunk.leaders_by_point.iter().enumerate()
                        {
                            let pid = cur[chunk.source_start + local_idx];
                            for &leader_idx in leaders_for_point {
                                clusters[leader_idx].push(pid);
                            }
                        }
                        Ok(())
                    },
                )?
            } else {
                let profile = assign_point_leaders_adsampling_streaming(
                    adsampling_dataset,
                    cur,
                    leaders,
                    fanout,
                    &context.params,
                    context.depth,
                    |chunk| {
                        for (local_idx, leaders_for_point) in
                            chunk.leaders_by_point.iter().enumerate()
                        {
                            let pid = cur[chunk.source_start + local_idx];
                            for &leader_idx in leaders_for_point {
                                clusters[leader_idx].push(pid);
                            }
                        }
                        Ok(())
                    },
                )?;
                (profile, StrictPrefetchPipelineProfile::default())
            };
            context.record_adsampling_scheduler_profile(&adsampling_profile);
            log_adsampling_profile(&adsampling_profile);
            let profile = GemmProfile {
                total_wall: Duration::from_secs_f64(adsampling_profile.total_ms / 1000.0),
                topk: Duration::from_secs_f64(adsampling_profile.total_ms / 1000.0),
                blocks: 1,
                ..GemmProfile::default()
            };
            let mut profile = profile;
            profile
                .point_pipeline
                .merge(adsampling_profile.point_pipeline.clone());
            profile.prefetch.merge(prefetch_profile);
            profile.assignment_decision = Some(AssignmentDecisionRecord {
                depth: context.depth,
                points: cur.len(),
                leaders: leaders.len(),
                fanout,
                wall: profile.total_wall,
                adsampling: true,
                fallback_reason: None,
                recall_at_fanout: adsampling_profile.validation_recall_at_fanout,
                mismatches: adsampling_profile.validation_mismatches,
            });
            return Ok((clusters, profile));
        } else {
            log_adsampling_depth_fallback(
                context,
                cur.len(),
                leaders.len(),
                local_fanout,
                "in-memory",
            );
        }
    }

    let total_start = Instant::now();
    let nl = leaders.len();
    let dim = dataset.dim();
    let (point_tile_size, leader_tile_size) =
        choose_gemm_tile_sizes_with_override_and_budget(nl, dim, memory_budget_bytes);
    let leader_read_options =
        default_rbc_windowed_options(dataset, nl.min(leader_tile_size.max(1)));
    let point_read_options = default_rbc_windowed_options(dataset, point_tile_size);

    // Pre-compute leader matrix and norms (shared, read-only)
    let mut l_data = vec![0f32; nl * dim];
    let mut leader_io_stats = PointBatchStats::default();
    dataset.read_points_windowed_into_batch_stats(
        leaders,
        &mut l_data,
        &leader_read_options,
        &mut leader_io_stats,
    )?;
    let l_norms: Vec<f32> = l_data
        .chunks(dim)
        .map(|r| r.iter().map(|&v| v * v).sum())
        .collect();
    let l_mat = ArrayView2::from_shape((nl, dim), &l_data).unwrap();

    let num_workers = rayon::current_num_threads().max(1);
    let batch_points = choose_prefetch_batch_points_for_budget(
        point_tile_size,
        dim,
        num_workers,
        memory_budget_bytes,
    );
    let mut state = GemmFoldState::new(point_tile_size, leader_tile_size, dim, nl, local_fanout);
    let mut io_stats = leader_io_stats;

    // Stage batched point reads on the producer thread so worker hot loops focus on
    // GEMM/top-k/merge. This is the smallest execution-model change that starts hiding
    // point-store latency without changing clustering semantics or introducing a separate async
    // runtime.
    let strict_prefetch_config =
        assignment_context.and_then(|context| context.strict_prefetch_config());
    let strict_prefetch_gate =
        assignment_context.and_then(|context| context.strict_prefetch_gate());
    let mut prefetch_profile = StrictPrefetchPipelineProfile::default();
    for_each_prefetched_point_batch_profiled(
        dataset,
        cur,
        batch_points,
        choose_prefetch_queue_depth_for_budget(batch_points, dim, memory_budget_bytes),
        point_read_options,
        strict_prefetch_config,
        strict_prefetch_gate,
        Some(&mut prefetch_profile),
        |prefetched| {
            io_stats.point_calls += prefetched.io_stats.point_calls;
            io_stats.range_calls += prefetched.io_stats.range_calls;
            io_stats.range_rows_read += prefetched.io_stats.range_rows_read;
            io_stats.bytes_read += prefetched.io_stats.bytes_read;
            state.profile.build_x += prefetched.load;
            let compute_block_points =
                choose_compute_block_points_for_batch(point_tile_size, prefetched.len, num_workers);

            let batch_state = (0..prefetched.len)
                .into_par_iter()
                .step_by(compute_block_points)
                .map(|block_start_idx| {
                    let block_end_idx =
                        (block_start_idx + compute_block_points).min(prefetched.len);
                    let block = &prefetched.ids[block_start_idx..block_end_idx];
                    let x_slice = &prefetched.data[block_start_idx * dim..block_end_idx * dim];
                    let block_n = block.len();
                    let mut local_state = GemmFoldState::new(
                        compute_block_points,
                        leader_tile_size,
                        dim,
                        nl,
                        local_fanout,
                    );
                    let mut block_profile = GemmProfile {
                        blocks: 1,
                        ..GemmProfile::default()
                    };
                    let block_start = Instant::now();

                    let scratch = &mut local_state.scratch;
                    let build_x_start = Instant::now();
                    scratch.x_data.clear();
                    scratch.x_data.extend_from_slice(x_slice);
                    scratch.x_norms.resize(block_n, 0f32);
                    for (i, chunk) in scratch.x_data.chunks(dim).enumerate() {
                        scratch.x_norms[i] = chunk.iter().map(|&v| v * v).sum();
                    }
                    let x_mat =
                        ArrayView2::from_shape((block_n, dim), scratch.x_data.as_slice()).unwrap();
                    block_profile.build_x += build_x_start.elapsed();

                    if local_fanout == 1 {
                        let topk_start = Instant::now();
                        let mut best_distances = vec![f32::INFINITY; block_n];
                        let mut best_leaders = vec![0usize; block_n];
                        block_profile.topk += topk_start.elapsed();

                        for leader_offset in (0..nl).step_by(leader_tile_size) {
                            let leader_end = (leader_offset + leader_tile_size).min(nl);
                            let leader_view =
                                l_mat.slice(ndarray::s![leader_offset..leader_end, ..]);
                            let tile_cols = leader_end - leader_offset;
                            let mut gram_tile =
                                borrow_gram_tile(&mut scratch.gram_data, block_n, tile_cols);
                            let gemm_start = Instant::now();
                            ndarray::linalg::general_mat_mul(
                                1.0,
                                &x_mat,
                                &leader_view.t(),
                                0.0,
                                &mut gram_tile,
                            );
                            block_profile.gemm += gemm_start.elapsed();
                            let topk_update_start = Instant::now();
                            update_block_best_from_gram_tile(
                                &mut best_distances,
                                &mut best_leaders,
                                &scratch.x_norms[..block_n],
                                &l_norms,
                                gram_tile.view(),
                                leader_offset,
                            );
                            block_profile.topk += topk_update_start.elapsed();
                        }

                        let merge_start = Instant::now();
                        scratch.local_clusters.resize_with(nl, Vec::new);
                        for v in scratch.local_clusters[..nl].iter_mut() {
                            v.clear();
                        }
                        for (i, &best_lid) in best_leaders.iter().enumerate() {
                            scratch.local_clusters[best_lid].push(block[i]);
                        }

                        merge_cluster_buffers_in_place(
                            &mut local_state.clusters,
                            &mut scratch.local_clusters,
                        );
                        block_profile.merge += merge_start.elapsed();
                    } else {
                        let topk_start = Instant::now();
                        scratch
                            .topk_rows
                            .resize_with(block_n, || StackTopK::new(local_fanout));
                        for topk in scratch.topk_rows[..block_n].iter_mut() {
                            *topk = StackTopK::new(local_fanout);
                        }
                        block_profile.topk += topk_start.elapsed();

                        for leader_offset in (0..nl).step_by(leader_tile_size) {
                            let leader_end = (leader_offset + leader_tile_size).min(nl);
                            let leader_view =
                                l_mat.slice(ndarray::s![leader_offset..leader_end, ..]);
                            let tile_cols = leader_end - leader_offset;
                            let mut gram_tile =
                                borrow_gram_tile(&mut scratch.gram_data, block_n, tile_cols);
                            let gemm_start = Instant::now();
                            ndarray::linalg::general_mat_mul(
                                1.0,
                                &x_mat,
                                &leader_view.t(),
                                0.0,
                                &mut gram_tile,
                            );
                            block_profile.gemm += gemm_start.elapsed();
                            let topk_update_start = Instant::now();
                            update_block_topk_from_gram_tile(
                                &mut scratch.topk_rows[..block_n],
                                &scratch.x_norms[..block_n],
                                &l_norms,
                                gram_tile.view(),
                                leader_offset,
                            );
                            block_profile.topk += topk_update_start.elapsed();
                        }

                        let merge_start = Instant::now();
                        scratch.local_clusters.resize_with(nl, Vec::new);
                        for v in scratch.local_clusters[..nl].iter_mut() {
                            v.clear();
                        }
                        for (i, top_k) in scratch.topk_rows[..block_n].iter().enumerate() {
                            for &(_, lid) in top_k.iter() {
                                scratch.local_clusters[lid].push(block[i]);
                            }
                        }

                        merge_cluster_buffers_in_place(
                            &mut local_state.clusters,
                            &mut scratch.local_clusters,
                        );
                        block_profile.merge += merge_start.elapsed();
                    }
                    block_profile.total_wall = block_start.elapsed();
                    local_state.profile.merge(block_profile);
                    Ok::<GemmFoldState, crate::common::AnnError>(local_state)
                })
                .try_reduce(
                    || {
                        GemmFoldState::new(
                            compute_block_points,
                            leader_tile_size,
                            dim,
                            nl,
                            local_fanout,
                        )
                    },
                    |mut left, right| -> AnnResult<GemmFoldState> {
                        merge_cluster_buffers(&mut left.clusters, right.clusters);
                        left.profile.merge(right.profile);
                        Ok(left)
                    },
                )?;

            merge_cluster_buffers(&mut state.clusters, batch_state.clusters);
            state.profile.merge(batch_state.profile);
            Ok(())
        },
    )?;
    state.profile.prefetch.merge(prefetch_profile);

    let mut profile = state.profile;
    profile.total_wall = total_start.elapsed();
    if let Some(context) = assignment_context {
        let fallback_reason =
            assignment_gemm_fallback_reason(context, cur.len(), leaders.len(), local_fanout);
        context.record_ads_exact_fallback(fallback_reason);
        profile.assignment_decision = Some(AssignmentDecisionRecord {
            depth: context.depth,
            points: cur.len(),
            leaders: leaders.len(),
            fanout: local_fanout,
            wall: profile.total_wall,
            adsampling: false,
            fallback_reason: fallback_reason.map(assignment_fallback_reason_index),
            recall_at_fanout: 0.0,
            mismatches: 0,
        });
    }
    let io = &io_stats;
    let prefetch = &profile.prefetch;
    tracing::info!(
        "[rbc/par] point_reads={} range_reads={} range_rows={} hit_ratio={:.2} prefetch_pipeline={} prefetch_queue_depth={} prefetch_budget_bytes={} prefetch_used_peak_bytes={} prefetch_batches={} prefetch_io_ms={} prefetch_consumer_wait_ms={} prefetch_producer_wait_ms={} prefetch_budget_wait_ms={} prefetch_fallback_budget_exhausted={} prefetch_logical_bytes={} prefetch_physical_bytes={} prefetch_read_amplification={:.3} prefetch_avg_read_size={}",
        io.point_calls,
        io.range_calls,
        io.range_rows_read,
        io.range_hit_ratio(),
        prefetch.pipeline.as_str(),
        prefetch.queue_depth,
        prefetch.prefetch_budget_bytes,
        prefetch.prefetch_used_peak_bytes,
        prefetch.batches,
        prefetch.io_wall.as_millis(),
        prefetch.consumer_wait.as_millis(),
        prefetch.producer_wait.as_millis(),
        prefetch.budget_wait.as_millis(),
        prefetch.fallback_budget_exhausted,
        prefetch.logical_bytes,
        prefetch.physical_bytes,
        prefetch.read_amplification(),
        prefetch.avg_read_size_bytes(),
    );
    Ok((state.clusters, profile))
}

pub(crate) fn compute_clusters_gemm_to_spool_budgeted_with_context(
    dataset: &dyn PointStore,
    cur: &[u32],
    leaders: &[u32],
    local_fanout: usize,
    memory_budget_bytes: Option<usize>,
    spool_writer: &mut BufWriter<File>,
    raw_counts: &mut [usize],
    assignment_context: Option<&AssignmentContext<'_>>,
) -> AnnResult<GemmProfile> {
    if let Some(context) = assignment_context {
        let use_adsampling =
            context.should_use_adsampling_assignment(cur.len(), leaders.len(), local_fanout);
        if use_adsampling {
            let fanout = assign_record_fanout(local_fanout.min(leaders.len().max(1)));
            raw_counts.fill(0);
            let adsampling_dataset = context.adsampling_dataset_for_assignment(dataset);
            let mut write_chunk =
                |chunk: &crate::forgeann::adsampling::AdSamplingPointAssignmentChunk| {
                    for leaders_for_point in &chunk.leaders_by_point {
                        let mut record = [0u16; 32];
                        for (idx, &leader_idx) in leaders_for_point.iter().enumerate() {
                            raw_counts[leader_idx] += 1;
                            record[idx] = leader_idx as u16;
                        }
                        write_assign_record(
                            spool_writer,
                            &record[..leaders_for_point.len()],
                            fanout,
                        )?;
                    }
                    Ok(())
                };
            let (adsampling_profile, prefetch_profile) = if context.depth > 0 {
                assign_point_leaders_adsampling_prefetched_with_context(
                    adsampling_dataset,
                    cur,
                    leaders,
                    fanout,
                    &context.params,
                    context.depth,
                    memory_budget_bytes,
                    context.strict_prefetch_config(),
                    context.strict_prefetch_gate(),
                    &mut write_chunk,
                )?
            } else {
                let profile = assign_point_leaders_adsampling_streaming(
                    adsampling_dataset,
                    cur,
                    leaders,
                    fanout,
                    &context.params,
                    context.depth,
                    &mut write_chunk,
                )?;
                (profile, StrictPrefetchPipelineProfile::default())
            };
            context.record_adsampling_scheduler_profile(&adsampling_profile);
            log_adsampling_profile(&adsampling_profile);
            let flush_start = Instant::now();
            spool_writer.flush()?;
            let mut profile = GemmProfile {
                total_wall: Duration::from_secs_f64(adsampling_profile.total_ms / 1000.0),
                topk: Duration::from_secs_f64(adsampling_profile.total_ms / 1000.0),
                flush: flush_start.elapsed(),
                blocks: 1,
                ..GemmProfile::default()
            };
            profile
                .point_pipeline
                .merge(adsampling_profile.point_pipeline.clone());
            profile.prefetch.merge(prefetch_profile);
            profile.total_wall += profile.flush;
            profile.assignment_decision = Some(AssignmentDecisionRecord {
                depth: context.depth,
                points: cur.len(),
                leaders: leaders.len(),
                fanout,
                wall: profile.total_wall,
                adsampling: true,
                fallback_reason: None,
                recall_at_fanout: adsampling_profile.validation_recall_at_fanout,
                mismatches: adsampling_profile.validation_mismatches,
            });
            return Ok(profile);
        } else {
            log_adsampling_depth_fallback(context, cur.len(), leaders.len(), local_fanout, "spool");
        }
    }

    let total_start = Instant::now();
    let nl = leaders.len();
    let dim = dataset.dim();
    let fanout = assign_record_fanout(local_fanout.min(nl.max(1)));
    let (point_tile_size, leader_tile_size) =
        choose_gemm_tile_sizes_with_override_and_budget(nl, dim, memory_budget_bytes);
    let leader_read_options =
        default_rbc_windowed_options(dataset, nl.min(leader_tile_size.max(1)));
    let point_read_options = default_rbc_windowed_options(dataset, point_tile_size);

    // Shared leader data is read once and reused across worker-local point tiles.
    let mut l_data = vec![0f32; nl * dim];
    let mut leader_io_stats = PointBatchStats::default();
    dataset.read_points_windowed_into_batch_stats(
        leaders,
        &mut l_data,
        &leader_read_options,
        &mut leader_io_stats,
    )?;
    let l_norms: Vec<f32> = l_data
        .chunks(dim)
        .map(|r| r.iter().map(|&v| v * v).sum())
        .collect();
    let l_mat = ArrayView2::from_shape((nl, dim), &l_data).unwrap();
    let num_workers = rayon::current_num_threads().max(1);
    let batch_points = choose_prefetch_batch_points_for_budget(
        point_tile_size,
        dim,
        choose_spool_prefetch_workers(num_workers, memory_budget_bytes),
        memory_budget_bytes,
    );

    pub(crate) struct BatchFold {
        pub scratch: GemmScratch,
        pub profile: GemmProfile,
        pub raw_counts: Vec<usize>,
        pub assign_chunks: Vec<Vec<u8>>,
    }

    let mut profile = GemmProfile::default();
    let mut io_stats = leader_io_stats;
    let mut merged_raw_counts = vec![0usize; nl];

    let strict_prefetch_config =
        assignment_context.and_then(|context| context.strict_prefetch_config());
    let strict_prefetch_gate =
        assignment_context.and_then(|context| context.strict_prefetch_gate());
    let mut prefetch_profile = StrictPrefetchPipelineProfile::default();
    for_each_prefetched_point_batch_profiled(
        dataset,
        cur,
        batch_points,
        choose_prefetch_queue_depth_for_budget(batch_points, dim, memory_budget_bytes),
        point_read_options,
        strict_prefetch_config,
        strict_prefetch_gate,
        Some(&mut prefetch_profile),
        |prefetched| {
            io_stats.point_calls += prefetched.io_stats.point_calls;
            io_stats.range_calls += prefetched.io_stats.range_calls;
            io_stats.range_rows_read += prefetched.io_stats.range_rows_read;
            io_stats.bytes_read += prefetched.io_stats.bytes_read;
            profile.build_x += prefetched.load;
            let compute_block_points =
                choose_compute_block_points_for_batch(point_tile_size, prefetched.len, num_workers);

            let folds: Vec<BatchFold> = (0..prefetched.len)
                .into_par_iter()
                .step_by(compute_block_points)
                .map(|block_start_idx| {
                    let block_end_idx =
                        (block_start_idx + compute_block_points).min(prefetched.len);
                    let block = &prefetched.ids[block_start_idx..block_end_idx];
                    let x_slice = &prefetched.data[block_start_idx * dim..block_end_idx * dim];
                    let mut fold = BatchFold {
                        scratch: GemmScratch::new(
                            compute_block_points,
                            leader_tile_size,
                            dim,
                            nl,
                            fanout,
                        ),
                        profile: GemmProfile::default(),
                        raw_counts: vec![0usize; nl],
                        assign_chunks: Vec::new(),
                    };
                    let block_n = block.len();
                    let scratch = &mut fold.scratch;
                    let mut block_profile = GemmProfile {
                        blocks: 1,
                        ..GemmProfile::default()
                    };
                    let block_start = Instant::now();

                    let build_x_start = Instant::now();
                    scratch.x_data.clear();
                    scratch.x_data.extend_from_slice(x_slice);
                    scratch.x_norms.resize(block_n, 0f32);
                    for (i, chunk) in scratch.x_data.chunks(dim).enumerate() {
                        scratch.x_norms[i] = chunk.iter().map(|&v| v * v).sum();
                    }
                    let x_mat =
                        ArrayView2::from_shape((block_n, dim), scratch.x_data.as_slice()).unwrap();
                    block_profile.build_x += build_x_start.elapsed();

                    let mut assign_buf = Vec::with_capacity(block_n * assign_record_width(fanout));

                    if fanout == 1 {
                        let mut best_distances = vec![f32::INFINITY; block_n];
                        let mut best_leaders = vec![0usize; block_n];

                        for leader_offset in (0..nl).step_by(leader_tile_size) {
                            let leader_end = (leader_offset + leader_tile_size).min(nl);
                            let leader_view =
                                l_mat.slice(ndarray::s![leader_offset..leader_end, ..]);
                            let tile_cols = leader_end - leader_offset;
                            let mut gram_tile =
                                borrow_gram_tile(&mut scratch.gram_data, block_n, tile_cols);
                            let gemm_start = Instant::now();
                            ndarray::linalg::general_mat_mul(
                                1.0,
                                &x_mat,
                                &leader_view.t(),
                                0.0,
                                &mut gram_tile,
                            );
                            block_profile.gemm += gemm_start.elapsed();
                            let topk_update_start = Instant::now();
                            update_block_best_from_gram_tile(
                                &mut best_distances,
                                &mut best_leaders,
                                &scratch.x_norms[..block_n],
                                &l_norms,
                                gram_tile.view(),
                                leader_offset,
                            );
                            block_profile.topk += topk_update_start.elapsed();
                        }

                        for &best_lid in &best_leaders {
                            fold.raw_counts[best_lid] += 1;
                            append_assign_record_bytes(
                                &mut assign_buf,
                                &[best_lid as u16],
                                fanout,
                            )?;
                        }
                    } else {
                        let topk_start = Instant::now();
                        scratch
                            .topk_rows
                            .resize_with(block_n, || StackTopK::new(fanout));
                        for topk in scratch.topk_rows[..block_n].iter_mut() {
                            *topk = StackTopK::new(fanout);
                        }
                        block_profile.topk += topk_start.elapsed();

                        for leader_offset in (0..nl).step_by(leader_tile_size) {
                            let leader_end = (leader_offset + leader_tile_size).min(nl);
                            let leader_view =
                                l_mat.slice(ndarray::s![leader_offset..leader_end, ..]);
                            let tile_cols = leader_end - leader_offset;
                            let mut gram_tile =
                                borrow_gram_tile(&mut scratch.gram_data, block_n, tile_cols);
                            let gemm_start = Instant::now();
                            ndarray::linalg::general_mat_mul(
                                1.0,
                                &x_mat,
                                &leader_view.t(),
                                0.0,
                                &mut gram_tile,
                            );
                            block_profile.gemm += gemm_start.elapsed();
                            let topk_update_start = Instant::now();
                            update_block_topk_from_gram_tile(
                                &mut scratch.topk_rows[..block_n],
                                &scratch.x_norms[..block_n],
                                &l_norms,
                                gram_tile.view(),
                                leader_offset,
                            );
                            block_profile.topk += topk_update_start.elapsed();
                        }

                        for top_k in &scratch.topk_rows[..block_n] {
                            let mut leaders_for_point = [0u16; 32];
                            let mut len = 0usize;
                            for &(_, lid) in top_k.iter() {
                                fold.raw_counts[lid] += 1;
                                leaders_for_point[len] = lid as u16;
                                len += 1;
                            }
                            append_assign_record_bytes(
                                &mut assign_buf,
                                &leaders_for_point[..len],
                                fanout,
                            )?;
                        }
                    }

                    block_profile.merge += Duration::ZERO;
                    block_profile.total_wall = block_start.elapsed();
                    fold.profile.merge(block_profile);
                    fold.assign_chunks.push(assign_buf);
                    Ok::<BatchFold, crate::common::AnnError>(fold)
                })
                .collect::<AnnResult<Vec<_>>>()?;

            for fold in folds {
                profile.merge(fold.profile);
                for (lid, count) in fold.raw_counts.into_iter().enumerate() {
                    merged_raw_counts[lid] += count;
                }
                for bytes in fold.assign_chunks {
                    let spool_write_start = Instant::now();
                    spool_writer.write_all(&bytes)?;
                    profile.spool_write += spool_write_start.elapsed();
                }
            }
            Ok(())
        },
    )?;
    profile.prefetch.merge(prefetch_profile);

    raw_counts.copy_from_slice(&merged_raw_counts);

    let flush_start = Instant::now();
    spool_writer.flush()?;
    profile.flush += flush_start.elapsed();
    profile.total_wall = total_start.elapsed();
    if let Some(context) = assignment_context {
        let fallback_reason =
            assignment_gemm_fallback_reason(context, cur.len(), leaders.len(), fanout);
        context.record_ads_exact_fallback(fallback_reason);
        profile.assignment_decision = Some(AssignmentDecisionRecord {
            depth: context.depth,
            points: cur.len(),
            leaders: leaders.len(),
            fanout,
            wall: profile.total_wall,
            adsampling: false,
            fallback_reason: fallback_reason.map(assignment_fallback_reason_index),
            recall_at_fanout: 0.0,
            mismatches: 0,
        });
    }
    let prefetch = &profile.prefetch;
    tracing::info!(
        "[rbc/par-spool] point_reads={} range_reads={} range_rows={} hit_ratio={:.2} workers={} build_x_ms={} gemm_ms={} topk_ms={} merge_ms={} spool_write_ms={} flush_ms={} total_ms={} prefetch_pipeline={} prefetch_queue_depth={} prefetch_budget_bytes={} prefetch_used_peak_bytes={} prefetch_batches={} prefetch_io_ms={} prefetch_consumer_wait_ms={} prefetch_producer_wait_ms={} prefetch_budget_wait_ms={} prefetch_fallback_budget_exhausted={} prefetch_logical_bytes={} prefetch_physical_bytes={} prefetch_read_amplification={:.3} prefetch_avg_read_size={}",
        io_stats.point_calls,
        io_stats.range_calls,
        io_stats.range_rows_read,
        io_stats.range_hit_ratio(),
        num_workers,
        profile.build_x.as_millis(),
        profile.gemm.as_millis(),
        profile.topk.as_millis(),
        profile.merge.as_millis(),
        profile.spool_write.as_millis(),
        profile.flush.as_millis(),
        profile.total_wall.as_millis(),
        prefetch.pipeline.as_str(),
        prefetch.queue_depth,
        prefetch.prefetch_budget_bytes,
        prefetch.prefetch_used_peak_bytes,
        prefetch.batches,
        prefetch.io_wall.as_millis(),
        prefetch.consumer_wait.as_millis(),
        prefetch.producer_wait.as_millis(),
        prefetch.budget_wait.as_millis(),
        prefetch.fallback_budget_exhausted,
        prefetch.logical_bytes,
        prefetch.physical_bytes,
        prefetch.read_amplification(),
        prefetch.avg_read_size_bytes(),
    );
    Ok(profile)
}

#[inline]
pub(crate) fn should_dedup_cluster(
    _params: &ForgeANNParams,
    depth: usize,
    local_fanout: usize,
) -> bool {
    depth <= 1 || local_fanout > 1
}

#[inline]
pub(crate) fn should_use_external_assign(
    params: &ForgeANNParams,
    depth: usize,
    cluster_len: usize,
    has_external_run_store: bool,
) -> bool {
    params.oom_enable
        && has_external_run_store
        && depth <= EXTERNAL_ASSIGN_DEPTH_LIMIT
        && (depth == 0 || cluster_len >= MIN_EXTERNAL_ASSIGN_POINTS_NON_ROOT)
}

pub(crate) fn partition_external_once(
    dataset: &dyn PointStore,
    cur: &[u32],
    depth: usize,
    metric: Metric,
    params: &ForgeANNParams,
    adaptive_c_max: usize,
    leaders: &[u32],
    local_fanout: usize,
    compute_budget_bytes: Option<usize>,
    external_run_store: &Arc<Mutex<ExternalRunStore>>,
    assignment_context: &AssignmentContext<'_>,
    stats: &mut PartitionStats,
) -> AnnResult<ExternalPartitionAttempt> {
    let fanout = assign_record_fanout(local_fanout.max(1).min(leaders.len().max(1)));
    let assign_start = Instant::now();
    let assign_spool = tempfile::NamedTempFile::new()?;
    let file = assign_spool.reopen()?;
    let mut writer = BufWriter::new(file);
    let mut raw_counts = vec![0usize; leaders.len().max(1)];

    // Fixed-only assignment: root experiment variants were removed.
    let use_gemm = metric == Metric::L2 && cur.len() >= 256;
    if use_gemm {
        let profile = compute_clusters_gemm_to_spool_budgeted_with_context(
            dataset,
            cur,
            leaders,
            fanout,
            compute_budget_bytes.or_else(|| {
                params
                    .oom_enable
                    .then(|| params.effective_oom_memory_budget_bytes())
            }),
            &mut writer,
            &mut raw_counts,
            Some(assignment_context),
        )?;
        stats.record_gemm_profile_with_context(profile, Some(assignment_context));
    } else {
        compute_clusters_scalar_to_spool(
            dataset,
            cur,
            leaders,
            fanout,
            metric,
            &mut writer,
            &mut raw_counts,
        )?;
    }
    stats.record_phase_time(RbcPhase::ClusterAssign, assign_start.elapsed());

    let raw_cluster_count = raw_counts.len();
    let empty_clusters = raw_counts.iter().filter(|&&count| count == 0).count();
    let pre_merge_clusters = raw_cluster_count.saturating_sub(empty_clusters);
    let assignments_before_merge = raw_counts.iter().sum::<usize>();

    let merge_start = Instant::now();
    let merge_groups = merge_cluster_plan(&raw_counts, params.c_min, adaptive_c_max);
    stats.record_phase_time(RbcPhase::MergeClusters, merge_start.elapsed());
    stats.record_merge(pre_merge_clusters, merge_groups.len(), empty_clusters);
    stats.record_assignments(assignments_before_merge, assignments_before_merge);
    log_external_child_distribution(
        depth,
        leaders.len(),
        fanout,
        &raw_counts,
        &merge_groups,
        params.c_min,
        adaptive_c_max,
    );

    // Persist spool and child runs in the shared external store.
    {
        let _ = external_run_store;
        let _ = depth;
    }

    Ok(ExternalPartitionAttempt {
        assign_spool,
        fanout,
        merge_groups,
    })
}

/// Parallel recursive RBC partitioning.
/// When a cluster splits into multiple sub-clusters, they are processed in parallel
/// using rayon::join (binary tree of joins for arbitrary fan-out).
pub(crate) fn rbc_recurse_parallel(
    dataset: &dyn PointStore,
    mut cur: Vec<u32>,
    depth: usize,
    parent_n: usize,
    metric: Metric,
    params: &ForgeANNParams,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    seed: u64,
    pb: &ProgressBar,
    external_run_store: Option<&Arc<Mutex<ExternalRunStore>>>,
    root_fanout_state: &RootFanoutState,
    assignment_context: &AssignmentContext<'_>,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<PartitionStats> {
    let expected_leaves = (cur.len() / adaptive_c_max.max(1)).max(4);
    let mut stats = PartitionStats::new(params.max_depth, expected_leaves);
    let d0_root_leaf_recorder =
        (depth == 0 && external_run_store.is_some()).then(D0RootLeafRecorder::default);
    let d0_root_leaf_recorder_ref = d0_root_leaf_recorder.as_ref();

    stats.max_depth_seen = stats.max_depth_seen.max(depth);
    stats.max_cluster_size_seen = stats.max_cluster_size_seen.max(cur.len());

    if cur.is_empty() {
        return Ok(stats);
    }

    let original_n = cur.len();
    let mut n = original_n;

    if depth >= params.max_depth {
        emit_leaf_with_d0_root_capture(
            dataset,
            metric,
            params,
            &mut cur,
            depth,
            LeafReason::MaxDepth,
            true,
            params.kernel_safe_leaf_size(),
            &mut stats,
            leaf_emitter,
            d0_root_leaf_recorder_ref,
        )?;
        return Ok(stats);
    }

    if depth >= 3 {
        let shrink_ratio = parent_n as f32 / n as f32;
        if shrink_ratio < params.min_shrink_ratio {
            emit_leaf_with_d0_root_capture(
                dataset,
                metric,
                params,
                &mut cur,
                depth,
                LeafReason::ShrinkRatio,
                true,
                params.kernel_safe_leaf_size(),
                &mut stats,
                leaf_emitter,
                d0_root_leaf_recorder_ref,
            )?;
            return Ok(stats);
        }
    }

    let local_fanout = params.adaptive_fanout(n, depth);
    let dedup_cur = depth != 0 && should_dedup_cluster(params, depth, local_fanout);

    if depth != 0 {
        let before_dedup = cur.len();
        let dedup_start = Instant::now();
        if dedup_cur {
            cur.sort_unstable();
            cur.dedup();
        }
        stats.record_phase_time(RbcPhase::CurDedup, dedup_start.elapsed());
        stats.record_dedup(dedup_cur, before_dedup, cur.len());
        if cur.is_empty() {
            stats.record_assignments(original_n, 0);
            return Ok(stats);
        }
        stats.record_assignments(original_n, cur.len());
    } else {
        stats.record_assignments(n, n);
    }

    n = cur.len();

    if n <= adaptive_c_max {
        emit_leaf_with_d0_root_capture(
            dataset,
            metric,
            params,
            &mut cur,
            depth,
            LeafReason::NaturalSize,
            false,
            params.kernel_safe_leaf_size(),
            &mut stats,
            leaf_emitter,
            d0_root_leaf_recorder_ref,
        )?;
        return Ok(stats);
    }

    if depth >= 3 && n <= min_recurse_size {
        emit_leaf_with_d0_root_capture(
            dataset,
            metric,
            params,
            &mut cur,
            depth,
            LeafReason::MinRecurse,
            false,
            params.kernel_safe_leaf_size(),
            &mut stats,
            leaf_emitter,
            d0_root_leaf_recorder_ref,
        )?;
        return Ok(stats);
    }

    if n < adaptive_c_max * 3 && depth >= 5 {
        emit_leaf_with_d0_root_capture(
            dataset,
            metric,
            params,
            &mut cur,
            depth,
            LeafReason::SmallDeep,
            false,
            params.kernel_safe_leaf_size(),
            &mut stats,
            leaf_emitter,
            d0_root_leaf_recorder_ref,
        )?;
        return Ok(stats);
    }

    let adaptive_psamp = params.adaptive_psamp_fraction(n, depth);
    let max_leaders = params.max_leaders.min(n);
    let mut num_leaders = ((adaptive_psamp * n as f64).round() as usize).max(2);
    num_leaders = num_leaders.min(max_leaders);

    let mut merged: Vec<Vec<u32>> = Vec::new();
    let mut external_attempt: Option<ExternalPartitionAttempt> = None;
    let mut attempts_for_current = 0usize;
    // Derive a deterministic RNG from seed + depth + cluster size to avoid needing &mut Rng
    let mut rng =
        StdRng::seed_from_u64(seed ^ (depth as u64).wrapping_mul(6364136223846793005) ^ (n as u64));

    for _ in 0..6 {
        attempts_for_current += 1;

        let leaders: Vec<u32> = if depth == 0 && attempts_for_current == 1 {
            root_fanout_state
                .forced_root_leaders
                .clone()
                .unwrap_or_else(|| {
                    let sample_seed: u64 = rng.random();
                    sample_set_bottomk(&cur, num_leaders, sample_seed)
                })
        } else {
            let sample_seed: u64 = rng.random();
            sample_set_bottomk(&cur, num_leaders, sample_seed)
        };
        num_leaders = leaders.len();

        let mut local_fanout = local_fanout;
        if num_leaders > 1 {
            local_fanout = local_fanout.min(num_leaders);
        } else {
            local_fanout = 1;
        }

        stats.record_partition_attempt(depth, n, num_leaders, local_fanout);
        let assignment_class = classify_adsampling_task(params, n, num_leaders);
        let _large_assignment_guard = enter_large_assignment_guard(
            leaf_emitter,
            assignment_class.large_ads || assignment_class.huge_ads,
        );

        let use_external_assign = !dataset.is_resident_subset()
            && should_use_external_assign(params, depth, n, external_run_store.is_some());
        if use_external_assign {
            let attempt = partition_external_once(
                dataset,
                &cur,
                depth,
                metric,
                params,
                adaptive_c_max,
                &leaders,
                local_fanout,
                params
                    .oom_enable
                    .then(|| params.effective_oom_memory_budget_bytes()),
                external_run_store.expect("external run store must exist for external assign"),
                assignment_context,
                &mut stats,
            )?;
            if attempt.merge_groups.len() <= 1 && num_leaders < max_leaders {
                stats.record_retry_escalation();
                num_leaders = (num_leaders.saturating_mul(2)).min(max_leaders);
                continue;
            }
            external_attempt = Some(attempt);
            break;
        } else {
            let use_gemm = metric == Metric::L2 && n >= 256;
            let assign_start = Instant::now();
            let clusters = if use_gemm {
                let compute_budget_bytes = params
                    .oom_enable
                    .then(|| params.effective_oom_memory_budget_bytes());
                let use_adsampling = assignment_context.should_use_adsampling_assignment(
                    n,
                    leaders.len(),
                    local_fanout,
                );
                let use_strict_prefetch_gemm = should_use_strict_prefetch_for_gemm_assignment(
                    &assignment_context.params,
                    n,
                    leaders.len(),
                    dataset.dim(),
                    compute_budget_bytes,
                    rayon::current_num_threads().max(1),
                );
                let use_context_gemm = use_adsampling || use_strict_prefetch_gemm;
                let (clusters, mut gemm_profile) = if use_context_gemm {
                    compute_clusters_gemm_budgeted_with_context(
                        dataset,
                        &cur,
                        &leaders,
                        local_fanout,
                        compute_budget_bytes,
                        Some(assignment_context),
                    )?
                } else {
                    log_adsampling_depth_fallback(
                        assignment_context,
                        n,
                        leaders.len(),
                        local_fanout,
                        "backend",
                    );
                    compute_clusters_gemm_budgeted(
                        dataset,
                        &cur,
                        &leaders,
                        local_fanout,
                        compute_budget_bytes,
                    )?
                };
                if !use_adsampling && !use_context_gemm {
                    let fallback_reason = assignment_gemm_fallback_reason(
                        assignment_context,
                        n,
                        leaders.len(),
                        local_fanout,
                    );
                    assignment_context.record_ads_exact_fallback(fallback_reason);
                    gemm_profile.assignment_decision = Some(AssignmentDecisionRecord {
                        depth: assignment_context.depth,
                        points: n,
                        leaders: leaders.len(),
                        fanout: local_fanout,
                        wall: gemm_profile.total_wall,
                        adsampling: false,
                        fallback_reason: fallback_reason.map(assignment_fallback_reason_index),
                        recall_at_fanout: 0.0,
                        mismatches: 0,
                    });
                }
                stats.record_gemm_profile_with_context(gemm_profile, Some(assignment_context));
                clusters
            } else {
                let avg_cluster_size = (n / num_leaders).max(16);
                let mut clusters: Vec<Vec<u32>> = (0..num_leaders)
                    .map(|_| Vec::with_capacity(avg_cluster_size))
                    .collect();
                if local_fanout == 1 {
                    for &idx in &cur {
                        let mut best_lid = 0usize;
                        let mut best_dist = f32::INFINITY;
                        for (lid, &lid_global) in leaders.iter().enumerate() {
                            let d = dataset.get_distance(idx, lid_global, metric)?;
                            if d < best_dist {
                                best_dist = d;
                                best_lid = lid;
                            }
                        }
                        clusters[best_lid].push(idx);
                    }
                } else {
                    for &idx in &cur {
                        let mut top_k = StackTopK::new(local_fanout);
                        for (lid, &lid_global) in leaders.iter().enumerate() {
                            let d = dataset.get_distance(idx, lid_global, metric)?;
                            top_k.push(d, lid);
                        }
                        for &(_, lid) in top_k.iter() {
                            clusters[lid].push(idx);
                        }
                    }
                }
                clusters
            };
            stats.record_phase_time(RbcPhase::ClusterAssign, assign_start.elapsed());

            let raw_cluster_count = clusters.len();
            let empty_clusters = clusters.iter().filter(|c| c.is_empty()).count();
            let pre_merge_clusters = raw_cluster_count.saturating_sub(empty_clusters);
            let assignments_before_merge = clusters.iter().map(Vec::len).sum::<usize>();

            let merge_start = Instant::now();
            merged = merge_clusters(clusters, params.c_min, adaptive_c_max);
            stats.record_phase_time(RbcPhase::MergeClusters, merge_start.elapsed());
            let post_merge_clusters = merged.len();
            stats.record_merge(pre_merge_clusters, post_merge_clusters, empty_clusters);
            stats.record_assignments(assignments_before_merge, assignments_before_merge);

            if merged.len() <= 1 && num_leaders < max_leaders {
                stats.record_retry_escalation();
                num_leaders = (num_leaders.saturating_mul(2)).min(max_leaders);
                continue;
            }
        }
        break;
    }

    let dedup_merged = should_dedup_cluster(params, depth, local_fanout);
    let mut to_recurse: Vec<SeededCluster> = Vec::new();
    let mut external_recurse_runs: Vec<ChildRun> = Vec::new();
    let child_seed_base = seed
        .wrapping_add(depth as u64 + 1)
        .wrapping_mul(0x9e3779b97f4a7c15);

    if let Some(attempt) = external_attempt {
        if attempt.merge_groups.is_empty() {
            stats.record_partition_result(PartitionResult::FailedEmpty);
            emit_leaf_with_d0_root_capture(
                dataset,
                metric,
                params,
                &mut cur,
                depth,
                LeafReason::PartitionFallback,
                false,
                params.kernel_safe_leaf_size(),
                &mut stats,
                leaf_emitter,
                d0_root_leaf_recorder_ref,
            )?;
            return Ok(stats);
        }

        if attempt.merge_groups.len() == 1 {
            stats.record_partition_result(PartitionResult::FailedNoSplit);
            emit_leaf_with_d0_root_capture(
                dataset,
                metric,
                params,
                &mut cur,
                depth,
                LeafReason::PartitionFallback,
                false,
                params.kernel_safe_leaf_size(),
                &mut stats,
                leaf_emitter,
                d0_root_leaf_recorder_ref,
            )?;
            return Ok(stats);
        }

        if attempts_for_current == 1 {
            stats.record_partition_result(PartitionResult::SuccessFirst);
        } else {
            stats.record_partition_result(PartitionResult::SuccessRetry);
        }

        let mut guard = external_run_store
            .expect("external run store must exist for external assign")
            .lock();
        let materialized = materialize_merged_children_from_spool_with_inline_limit(
            &cur,
            depth,
            attempt.fanout,
            &attempt.merge_groups,
            &attempt.assign_spool,
            &mut guard,
            Some(adaptive_c_max),
        )?;
        drop(guard);

        for (cluster_idx, child) in materialized.into_iter().enumerate() {
            let mut cluster = if let Some(points) = child.points {
                points
            } else {
                let guard = external_run_store
                    .expect("external run store must exist for external materialization")
                    .lock();
                read_child_run_chain(&guard, depth, &child.extents)?
            };
            if dedup_merged {
                let dedup_start = Instant::now();
                let before = cluster.len();
                cluster.sort_unstable();
                cluster.dedup();
                let after = cluster.len();
                stats.record_phase_time(RbcPhase::MergedDedup, dedup_start.elapsed());
                stats.record_dedup(true, before, after);
                stats.record_assignments(before, after);
            } else {
                stats.record_dedup(false, cluster.len(), cluster.len());
                stats.record_assignments(cluster.len(), cluster.len());
            }
            if cluster.is_empty() {
                continue;
            }
            if cluster.len() > adaptive_c_max {
                let cluster_seed = child_seed_base
                    ^ (cluster_idx as u64).wrapping_mul(0xbf58476d1ce4e5b9)
                    ^ (cluster.len() as u64).rotate_left(17);
                if depth <= EXTERNAL_CHILD_DEPTH_LIMIT {
                    let extent = {
                        let mut guard = external_run_store
                            .expect("external run store must exist for child run write")
                            .lock();
                        write_child_run(&mut guard, depth, &cluster)?
                    };
                    external_recurse_runs.push(ChildRun {
                        extents: vec![extent],
                        len: cluster.len(),
                        seed: cluster_seed,
                    });
                } else {
                    to_recurse.push(SeededCluster {
                        points: cluster,
                        seed: cluster_seed,
                    });
                }
            } else {
                emit_leaf_with_d0_root_capture(
                    dataset,
                    metric,
                    params,
                    &mut cluster,
                    depth + 1,
                    LeafReason::NaturalSize,
                    false,
                    params.kernel_safe_leaf_size(),
                    &mut stats,
                    leaf_emitter,
                    d0_root_leaf_recorder_ref,
                )?;
            }
        }
    } else {
        if dedup_merged {
            // Parallelize sort+dedup across clusters (significant win at depth 0 with large
            // clusters)
            let dedup_start = Instant::now();
            let dedup_results: Vec<(usize, usize)> = merged
                .par_iter_mut()
                .map(|cluster| {
                    let before = cluster.len();
                    cluster.sort_unstable();
                    cluster.dedup();
                    (before, cluster.len())
                })
                .collect();
            let dedup_elapsed = dedup_start.elapsed();
            stats.record_phase_time(RbcPhase::MergedDedup, dedup_elapsed);
            for (before_dedup, after_dedup) in dedup_results {
                stats.record_dedup(true, before_dedup, after_dedup);
                stats.record_assignments(before_dedup, after_dedup);
            }
        } else {
            for cluster in &merged {
                stats.record_dedup(false, cluster.len(), cluster.len());
                stats.record_assignments(cluster.len(), cluster.len());
            }
        }
        merged.retain(|cluster| !cluster.is_empty());

        if merged.is_empty() {
            stats.record_partition_result(PartitionResult::FailedEmpty);
            emit_leaf_with_d0_root_capture(
                dataset,
                metric,
                params,
                &mut cur,
                depth,
                LeafReason::PartitionFallback,
                false,
                params.kernel_safe_leaf_size(),
                &mut stats,
                leaf_emitter,
                d0_root_leaf_recorder_ref,
            )?;
            return Ok(stats);
        }

        if merged.len() == 1 {
            stats.record_partition_result(PartitionResult::FailedNoSplit);
            let mut leaf = merged.pop().unwrap();
            emit_leaf_with_d0_root_capture(
                dataset,
                metric,
                params,
                &mut leaf,
                depth,
                LeafReason::PartitionFallback,
                false,
                params.kernel_safe_leaf_size(),
                &mut stats,
                leaf_emitter,
                d0_root_leaf_recorder_ref,
            )?;
            return Ok(stats);
        }

        if attempts_for_current == 1 {
            stats.record_partition_result(PartitionResult::SuccessFirst);
        } else {
            stats.record_partition_result(PartitionResult::SuccessRetry);
        }

        // Separate small clusters (emit as leaves) from large ones (recurse)
        for (cluster_idx, cluster) in merged.into_iter().enumerate() {
            if cluster.len() > adaptive_c_max {
                let cluster_seed = child_seed_base
                    ^ (cluster_idx as u64).wrapping_mul(0xbf58476d1ce4e5b9)
                    ^ (cluster.len() as u64).rotate_left(17);
                if let Some(run_store) = external_run_store
                    .filter(|_| !dataset.is_resident_subset())
                    .filter(|_| depth <= EXTERNAL_CHILD_DEPTH_LIMIT)
                {
                    let extent = {
                        let mut guard = run_store.lock();
                        write_child_run(&mut guard, depth, &cluster)?
                    };
                    external_recurse_runs.push(ChildRun {
                        extents: vec![extent],
                        len: cluster.len(),
                        seed: cluster_seed,
                    });
                } else {
                    to_recurse.push(SeededCluster {
                        points: cluster,
                        seed: cluster_seed,
                    });
                }
            } else {
                let mut leaf = cluster;
                emit_leaf_with_d0_root_capture(
                    dataset,
                    metric,
                    params,
                    &mut leaf,
                    depth + 1,
                    LeafReason::NaturalSize,
                    false,
                    params.kernel_safe_leaf_size(),
                    &mut stats,
                    leaf_emitter,
                    d0_root_leaf_recorder_ref,
                )?;
            }
        }
    }

    if depth == 0 {
        log_root_fanout_summary(&root_fanout_state.profile());
    }

    let child_assignment_context = assignment_context.for_depth(depth + 1);

    if !external_recurse_runs.is_empty() {
        let external_run_store =
            external_run_store.expect("external run store must exist for external recurse runs");
        if depth == 0 {
            let mut guard = external_run_store.lock();
            let root_leaves = d0_root_leaf_recorder
                .as_ref()
                .map(D0RootLeafRecorder::take)
                .unwrap_or_default();
            let root_leaf_runs = append_d0_root_leaves(&mut guard, root_leaves)?;
            guard.finalize()?;
            let base_dir = guard.base_dir.clone();
            drop(guard);
            let _ = write_d0_runs_manifest(
                &base_dir,
                dataset,
                metric,
                params,
                n,
                adaptive_c_max,
                min_recurse_size,
                seed,
                root_fanout_state.profile(),
                &external_recurse_runs,
                &root_leaf_runs,
            )?;
        }
        let child_stats = parallel_join_child_runs(
            dataset,
            external_recurse_runs,
            depth + 1,
            n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            pb,
            external_run_store,
            root_fanout_state,
            &child_assignment_context,
            leaf_emitter,
        )?;
        stats.merge_from(child_stats);

        pb.inc(1);
        return Ok(stats);
    }

    // Process sub-clusters in parallel using rayon::join (binary tree)
    let child_stats = parallel_join_clusters(
        dataset,
        to_recurse,
        depth + 1,
        n,
        metric,
        params,
        adaptive_c_max,
        min_recurse_size,
        pb,
        external_run_store,
        root_fanout_state,
        &child_assignment_context,
        leaf_emitter,
    )?;
    stats.merge_from(child_stats);

    // Update progress bar
    pb.inc(1);

    Ok(stats)
}

/// Recursively process a list of clusters in parallel using rayon::join (binary tree
/// decomposition).
pub(crate) fn parallel_join_clusters(
    dataset: &dyn PointStore,
    mut clusters: Vec<SeededCluster>,
    depth: usize,
    parent_n: usize,
    metric: Metric,
    params: &ForgeANNParams,
    adaptive_c_max: usize,
    min_recurse_size: usize,
    pb: &ProgressBar,
    external_run_store: Option<&Arc<Mutex<ExternalRunStore>>>,
    root_fanout_state: &RootFanoutState,
    assignment_context: &AssignmentContext<'_>,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<PartitionStats> {
    let expected_leaves = (clusters
        .iter()
        .map(|cluster| cluster.points.len())
        .sum::<usize>()
        / adaptive_c_max.max(1))
    .max(4);
    let mut stats = PartitionStats::new(params.max_depth, expected_leaves);

    if clusters.is_empty() {
        return Ok(stats);
    }

    if clusters.len() == 1 {
        let cluster = clusters.pop().unwrap();
        let child_stats = rbc_recurse_parallel(
            dataset,
            cluster.points,
            depth,
            parent_n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            cluster.seed,
            pb,
            external_run_store,
            root_fanout_state,
            assignment_context,
            leaf_emitter,
        )?;
        stats.merge_from(child_stats);
        return Ok(stats);
    }

    // Split into two halves and process in parallel
    let (left, right) = split_seeded_clusters_balanced(clusters);

    let (left_result, right_result) = rayon::join(
        || {
            parallel_join_clusters(
                dataset,
                left,
                depth,
                parent_n,
                metric,
                params,
                adaptive_c_max,
                min_recurse_size,
                pb,
                external_run_store,
                root_fanout_state,
                assignment_context,
                leaf_emitter,
            )
        },
        || {
            parallel_join_clusters(
                dataset,
                right,
                depth,
                parent_n,
                metric,
                params,
                adaptive_c_max,
                min_recurse_size,
                pb,
                external_run_store,
                root_fanout_state,
                assignment_context,
                leaf_emitter,
            )
        },
    );

    stats.merge_from(left_result?);
    stats.merge_from(right_result?);
    Ok(stats)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct D1LevelScanOccurrence {
    pub run_idx: usize,
    pub local_idx: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct D1LevelScanHeapEntry {
    pub point: u32,
    pub run_idx: usize,
    pub local_idx: usize,
}

impl Ord for D1LevelScanHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .point
            .cmp(&self.point)
            .then_with(|| other.run_idx.cmp(&self.run_idx))
            .then_with(|| other.local_idx.cmp(&self.local_idx))
    }
}

impl PartialOrd for D1LevelScanHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) struct D1LevelScanBatchPlan {
    pub point_ids: Vec<u32>,
    pub occurrences_by_point: Vec<Vec<D1LevelScanOccurrence>>,
}

pub(crate) struct D1LevelScanLoadedBatch {
    pub batch_idx: usize,
    pub plan: D1LevelScanBatchPlan,
    pub data: Vec<f32>,
    pub io_stats: PointBatchStats,
    pub read_wall: Duration,
}

pub(crate) struct D1LevelScanRunCompute {
    pub leader_layout: AdSamplingLeaderLayout,
    pub config: AdSamplingConfig,
    pub seed_indices: Vec<usize>,
    pub seeded: Vec<bool>,
    pub fanout: usize,
}

pub(crate) struct D1LevelScanRunWork {
    pub points: Vec<u32>,
    pub seed: u64,
    pub raw_counts: Vec<usize>,
    pub assignment_segments: Vec<AssignmentSpoolSegment>,
    pub leaders: usize,
    pub fanout: usize,
    pub max_leaders: usize,
}

pub(crate) struct D1LevelScanAssignmentSpool {
    pub spool: NamedTempFile,
    pub writer: BufWriter<File>,
    pub write_offset: u64,
}

impl D1LevelScanAssignmentSpool {
    pub(crate) fn new_in(dir: &Path) -> AnnResult<Self> {
        fs::create_dir_all(dir)?;
        let spool = NamedTempFile::new_in(dir)?;
        let writer = BufWriter::new(spool.reopen()?);
        Ok(Self {
            spool,
            writer,
            write_offset: 0,
        })
    }

    fn append(&mut self, bytes: &[u8]) -> AnnResult<Option<AssignmentSpoolSegment>> {
        if bytes.is_empty() {
            return Ok(None);
        }
        let segment = AssignmentSpoolSegment {
            offset: self.write_offset,
            len: bytes.len() as u64,
        };
        self.writer.write_all(bytes)?;
        self.write_offset = self.write_offset.saturating_add(segment.len);
        Ok(Some(segment))
    }
}

#[derive(Default)]
pub(crate) struct D1ResidentLeafBatch {
    pub leaves: Vec<Vec<u32>>,
    pub points: usize,
}

fn nanos_to_millis(nanos: u64) -> u128 {
    u128::from(nanos) / 1_000_000
}

fn leaf_worker_capacity(worker_threads: usize) -> usize {
    worker_threads.max(1).saturating_sub(1).max(1)
}

fn d1_materialize_leaf_drainer_limit_for_policy(worker_threads: usize) -> Option<usize> {
    if let Some(requested) = env_usize("FORGEANN_D1_MATERIALIZE_LEAF_DRAINER_LIMIT") {
        if requested == 0 {
            return None;
        }
        return Some(requested.clamp(1, leaf_worker_capacity(worker_threads)));
    }

    Some(
        worker_threads
            .max(1)
            .saturating_mul(3)
            .div_ceil(4)
            .clamp(1, leaf_worker_capacity(worker_threads)),
    )
}

fn d1_materialize_leaf_drainer_backpressure_limit(
    worker_threads: usize,
    leaf_drainer_limit: usize,
) -> usize {
    leaf_drainer_limit.clamp(1, leaf_worker_capacity(worker_threads))
}

fn d1_materialize_leaf_backlog_soft_limit_for_policy() -> Option<usize> {
    env_usize("FORGEANN_D1_MATERIALIZE_LEAF_BACKLOG_SOFT_LIMIT").filter(|&limit| limit > 0)
}

fn d1_resident_leaf_batch_limits(params: &ForgeANNParams, worker_threads: usize) -> (usize, usize) {
    if !params.leaf_batch_drain_enable {
        return (1, 0);
    }

    let worker_threads = worker_threads.max(1);
    (
        params
            .leaf_batch_drain_max_leaves
            .max(worker_threads.saturating_mul(2))
            .max(1),
        params
            .leaf_batch_drain_max_points
            .max(worker_threads.saturating_mul(params.kernel_safe_leaf_size()))
            .max(1),
    )
}

struct ResidentDatasetLeafEmitter<'a> {
    inner: &'a dyn LeafEmitter,
    local_sink: D1ResidentLeafMorselSink,
    resident_leaf_submissions: AtomicUsize,
}

impl<'a> ResidentDatasetLeafEmitter<'a> {
    fn new(inner: &'a dyn LeafEmitter, local_sink: D1ResidentLeafMorselSink) -> Self {
        Self {
            inner,
            local_sink,
            resident_leaf_submissions: AtomicUsize::new(0),
        }
    }

    fn emit_resident_leaf(&self, leaf: Vec<u32>) -> AnnResult<Option<Vec<u32>>> {
        self.local_sink.submit(leaf)?;
        self.resident_leaf_submissions
            .fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }

    fn resident_leaf_submissions(&self) -> usize {
        self.resident_leaf_submissions.load(Ordering::Relaxed)
    }

    fn resident_leaf_completion_mode(&self) -> &'static str {
        "local_morsel_priority_work_batch"
    }
}

impl LeafEmitter for ResidentDatasetLeafEmitter<'_> {
    fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.inner.emit_leaf(leaf)
    }

    fn emit_leaf_deferred(&self, leaf: Vec<u32>) -> AnnResult<()> {
        self.inner.emit_leaf_deferred(leaf)
    }

    fn emit_leaf_from_dataset(
        &self,
        dataset: &dyn PointStore,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        if dataset.is_resident_subset() {
            self.emit_resident_leaf(leaf)
        } else {
            self.inner.emit_leaf_from_dataset(dataset, leaf)
        }
    }

    fn emit_leaf_inline_from_dataset(
        &self,
        dataset: &dyn PointStore,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        if dataset.is_resident_subset() {
            self.emit_resident_leaf(leaf)
        } else {
            self.inner.emit_leaf_inline_from_dataset(dataset, leaf)
        }
    }

    fn wait_for_leaf_backlog_below(&self, target_backlog: usize) -> AnnResult<bool> {
        self.inner.wait_for_leaf_backlog_below(target_backlog)
    }

    fn scheduler_signals(&self) -> Option<&dyn SchedulerSignals> {
        self.inner.scheduler_signals()
    }

    fn scheduler_telemetry(&self) -> Option<SchedulerTelemetry> {
        self.inner.scheduler_telemetry()
    }
}

pub(crate) struct D1MaterializeLeafEmitter<'a> {
    pub inner: &'a dyn LeafEmitter,
    pub backlog_soft_limit: Option<usize>,
    pub resident_batch: Mutex<D1ResidentLeafBatch>,
    pub resident_batch_max_leaves: usize,
    pub resident_batch_max_points: usize,
    pub resident_batch_flushes: AtomicUsize,
    pub emit_calls: AtomicUsize,
    pub emit_ns: AtomicU64,
    pub throttle_calls: AtomicUsize,
    pub throttle_ns: AtomicU64,
}

impl<'a> D1MaterializeLeafEmitter<'a> {
    pub(crate) fn new_with_backlog_soft_limit(
        inner: &'a dyn LeafEmitter,
        backlog_soft_limit: Option<usize>,
    ) -> Self {
        Self {
            inner,
            backlog_soft_limit,
            resident_batch: Mutex::new(D1ResidentLeafBatch::default()),
            resident_batch_max_leaves: ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MAX_LEAVES,
            resident_batch_max_points: ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MAX_POINTS,
            resident_batch_flushes: AtomicUsize::new(0),
            emit_calls: AtomicUsize::new(0),
            emit_ns: AtomicU64::new(0),
            throttle_calls: AtomicUsize::new(0),
            throttle_ns: AtomicU64::new(0),
        }
    }

    pub(crate) fn emit_calls(&self) -> usize {
        self.emit_calls.load(Ordering::Relaxed)
    }

    fn emit_ms(&self) -> u128 {
        nanos_to_millis(self.emit_ns.load(Ordering::Relaxed))
    }

    pub(crate) fn throttle_calls(&self) -> usize {
        self.throttle_calls.load(Ordering::Relaxed)
    }

    fn throttle_ms(&self) -> u128 {
        nanos_to_millis(self.throttle_ns.load(Ordering::Relaxed))
    }

    fn resident_batch_flushes(&self) -> usize {
        self.resident_batch_flushes.load(Ordering::Relaxed)
    }

    fn configure_resident_batch_for_d1(&mut self, params: &ForgeANNParams, worker_threads: usize) {
        let (max_leaves, max_points) = d1_resident_leaf_batch_limits(params, worker_threads);
        self.resident_batch_max_leaves = max_leaves;
        self.resident_batch_max_points = max_points;
    }

    fn flush_resident_batch(&self, dataset: &dyn PointStore) -> AnnResult<()> {
        let batch = {
            let mut guard = self.resident_batch.lock();
            if guard.leaves.is_empty() {
                return Ok(());
            }
            let leaves = std::mem::take(&mut guard.leaves);
            guard.points = 0;
            leaves
        };
        let emit_start = Instant::now();
        let result = self
            .inner
            .emit_leaf_batch_inline_from_dataset(dataset, batch);
        self.emit_ns.fetch_add(
            emit_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.resident_batch_flushes.fetch_add(1, Ordering::Relaxed);
        result
    }

    fn emit_resident_leaf_batched(
        &self,
        dataset: &dyn PointStore,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        debug_assert!(dataset.is_resident_subset());
        let emitted = self.inner.emit_leaf_from_dataset(dataset, leaf)?;
        if let Some(leaf) = emitted {
            self.inner.emit_leaf_deferred(leaf)?;
        }
        self.emit_calls.fetch_add(1, Ordering::Relaxed);
        self.wait_for_backlog_soft_limit()?;
        Ok(None)
    }

    fn wait_for_backlog_soft_limit(&self) -> AnnResult<()> {
        let Some(backlog_soft_limit) = self.backlog_soft_limit else {
            return Ok(());
        };
        let throttle_start = Instant::now();
        let waited = self.inner.wait_for_leaf_backlog_below(backlog_soft_limit)?;
        if waited {
            self.throttle_ns.fetch_add(
                throttle_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            self.throttle_calls.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

impl LeafEmitter for D1MaterializeLeafEmitter<'_> {
    fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
        let emit_start = Instant::now();
        let result = self.inner.emit_leaf_deferred(leaf);
        self.emit_ns.fetch_add(
            emit_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.emit_calls.fetch_add(1, Ordering::Relaxed);
        result?;
        self.wait_for_backlog_soft_limit()
    }

    fn emit_leaf_from_dataset(
        &self,
        dataset: &dyn PointStore,
        leaf: Vec<u32>,
    ) -> AnnResult<Option<Vec<u32>>> {
        let emit_start = Instant::now();
        let result = if dataset.is_resident_subset() {
            self.emit_resident_leaf_batched(dataset, leaf)
        } else {
            self.inner.emit_leaf_from_dataset(dataset, leaf)
        };
        if matches!(result, Ok(None)) {
            self.emit_ns.fetch_add(
                emit_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            if !dataset.is_resident_subset() {
                self.emit_calls.fetch_add(1, Ordering::Relaxed);
            }
        }
        result
    }

    fn scheduler_signals(&self) -> Option<&dyn SchedulerSignals> {
        self.inner.scheduler_signals()
    }

    fn scheduler_telemetry(&self) -> Option<SchedulerTelemetry> {
        self.inner.scheduler_telemetry()
    }
}

#[derive(Default)]
pub(crate) struct D1LevelScanRunBatchInput {
    pub point_ids: Vec<u32>,
    pub source_offsets: Vec<usize>,
    pub local_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct D1LevelScanComputeTask {
    pub run_idx: usize,
    pub point_start: usize,
    pub point_end: usize,
    pub run_point_start: usize,
}

pub(crate) struct D1LevelScanRunBatchOutput {
    pub run_idx: usize,
    pub run_point_start: usize,
    pub assignment_bytes: Vec<u8>,
    pub raw_counts: Vec<usize>,
    pub chunk: AdSamplingChunkResult,
    pub chunk_wall: Duration,
}

pub(crate) struct D1LevelScanComputedBatch {
    pub batch_idx: usize,
    pub outputs: Vec<D1LevelScanRunBatchOutput>,
    pub io_stats: PointBatchStats,
    pub read_wall: Duration,
    pub compute_wall: Duration,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
}

#[derive(Clone)]
pub(crate) struct D1AdsComputeSharedTelemetry {
    pub active_chunks: Arc<AtomicUsize>,
    pub active_chunk_max: Arc<AtomicUsize>,
    pub active_chunk_start_sum: Arc<AtomicU64>,
    pub loaded_queue_depth_peak: Arc<AtomicUsize>,
    pub computed_queue_depth_peak: Arc<AtomicUsize>,
    pub read_batches: Arc<AtomicUsize>,
    pub computed_batches: Arc<AtomicUsize>,
}

impl D1AdsComputeSharedTelemetry {
    pub(crate) fn new() -> Self {
        Self {
            active_chunks: Arc::new(AtomicUsize::new(0)),
            active_chunk_max: Arc::new(AtomicUsize::new(0)),
            active_chunk_start_sum: Arc::new(AtomicU64::new(0)),
            loaded_queue_depth_peak: Arc::new(AtomicUsize::new(0)),
            computed_queue_depth_peak: Arc::new(AtomicUsize::new(0)),
            read_batches: Arc::new(AtomicUsize::new(0)),
            computed_batches: Arc::new(AtomicUsize::new(0)),
        }
    }
}

pub(crate) struct D1SequentialAdsPipelineReport {
    pub prefetch: StrictPrefetchPipelineProfile,
    pub compute_wall: Duration,
    pub apply_wall: Duration,
    pub batches: usize,
}

fn push_next_d1_level_scan_heap_entry(
    heap: &mut BinaryHeap<D1LevelScanHeapEntry>,
    runs: &[D1LevelScanRunWork],
    run_idx: usize,
    next_local_idx: usize,
) {
    if let Some(&point) = runs
        .get(run_idx)
        .and_then(|run| run.points.get(next_local_idx))
    {
        heap.push(D1LevelScanHeapEntry {
            point,
            run_idx,
            local_idx: next_local_idx,
        });
    }
}

pub(crate) fn next_d1_level_scan_batch(
    heap: &mut BinaryHeap<D1LevelScanHeapEntry>,
    runs: &[D1LevelScanRunWork],
    max_unique_points: usize,
) -> Option<D1LevelScanBatchPlan> {
    if heap.is_empty() {
        return None;
    }
    let max_unique_points = max_unique_points.max(1);
    let mut point_ids = Vec::with_capacity(max_unique_points);
    let mut occurrences_by_point = Vec::with_capacity(max_unique_points);

    while point_ids.len() < max_unique_points {
        let first = match heap.pop() {
            Some(entry) => entry,
            None => break,
        };
        let point = first.point;
        let mut occurrences = vec![D1LevelScanOccurrence {
            run_idx: first.run_idx,
            local_idx: first.local_idx,
        }];
        push_next_d1_level_scan_heap_entry(heap, runs, first.run_idx, first.local_idx + 1);

        while heap.peek().is_some_and(|entry| entry.point == point) {
            let entry = heap.pop().unwrap();
            occurrences.push(D1LevelScanOccurrence {
                run_idx: entry.run_idx,
                local_idx: entry.local_idx,
            });
            push_next_d1_level_scan_heap_entry(heap, runs, entry.run_idx, entry.local_idx + 1);
        }

        point_ids.push(point);
        occurrences_by_point.push(occurrences);
    }

    Some(D1LevelScanBatchPlan {
        point_ids,
        occurrences_by_point,
    })
}

fn read_d1_level_scan_batch(
    dataset: &dyn PointStore,
    batch_idx: usize,
    plan: D1LevelScanBatchPlan,
    read_options: WindowedGatherOptions,
) -> AnnResult<D1LevelScanLoadedBatch> {
    let read_start = Instant::now();
    let mut data = vec![0.0f32; plan.point_ids.len().saturating_mul(dataset.dim())];
    let mut io_stats = PointBatchStats::default();
    dataset.read_points_windowed_into_batch_stats(
        &plan.point_ids,
        &mut data,
        &read_options,
        &mut io_stats,
    )?;
    Ok(D1LevelScanLoadedBatch {
        batch_idx,
        plan,
        data,
        io_stats,
        read_wall: read_start.elapsed(),
    })
}

pub(crate) fn d1_ads_compute_chunk_points(
    points: usize,
    workers: usize,
    chunks_per_worker: usize,
) -> usize {
    pub(crate) const MIN_CHUNK_POINTS: usize = 256;
    pub(crate) const MAX_CHUNK_POINTS: usize = 8192;

    if points == 0 {
        return MIN_CHUNK_POINTS;
    }
    let target_chunks = workers
        .max(1)
        .saturating_mul(chunks_per_worker.max(1))
        .max(1);
    let chunk_points = points.div_ceil(target_chunks);
    chunk_points.clamp(MIN_CHUNK_POINTS, MAX_CHUNK_POINTS)
}

pub(crate) fn build_d1_level_scan_compute_tasks(
    grouped: &[Option<D1LevelScanRunBatchInput>],
    chunk_points: usize,
) -> Vec<D1LevelScanComputeTask> {
    let chunk_points = chunk_points.max(1);
    let mut tasks = Vec::new();
    for (run_idx, input) in grouped.iter().enumerate() {
        let Some(input) = input else {
            continue;
        };
        debug_assert_eq!(input.point_ids.len(), input.source_offsets.len());
        debug_assert_eq!(input.point_ids.len(), input.local_indices.len());
        for point_start in (0..input.point_ids.len()).step_by(chunk_points) {
            let point_end = (point_start + chunk_points).min(input.point_ids.len());
            let Some(&run_point_start) = input.local_indices.get(point_start) else {
                continue;
            };
            tasks.push(D1LevelScanComputeTask {
                run_idx,
                point_start,
                point_end,
                run_point_start,
            });
        }
    }
    tasks
}

pub(crate) fn build_d1_resident_level_scan_compute_tasks(
    runs: &[D1LevelScanRunWork],
    chunk_points: usize,
) -> Vec<D1LevelScanComputeTask> {
    let chunk_points = chunk_points.max(1);
    let mut tasks = Vec::new();
    for (run_idx, run) in runs.iter().enumerate() {
        for point_start in (0..run.points.len()).step_by(chunk_points) {
            let point_end = (point_start + chunk_points).min(run.points.len());
            tasks.push(D1LevelScanComputeTask {
                run_idx,
                point_start,
                point_end,
                run_point_start: point_start,
            });
        }
    }
    tasks
}

pub(crate) fn drain_ordered_d1_level_scan_batches<T>(
    pending: &mut BTreeMap<usize, T>,
    next_apply_batch_idx: &mut usize,
    mut apply: impl FnMut(T) -> AnnResult<()>,
) -> AnnResult<usize> {
    let mut drained = 0usize;
    while let Some(batch) = pending.remove(next_apply_batch_idx) {
        apply(batch)?;
        *next_apply_batch_idx += 1;
        drained += 1;
    }
    Ok(drained)
}

fn d1_level_scan_computed_reorder_capacity(
    pipeline_inflight: usize,
    batch_points: usize,
    dim: usize,
    memory_budget_bytes: Option<usize>,
) -> usize {
    let requested = pipeline_inflight.max(1).saturating_mul(2).max(1);
    clamp_d1_level_scan_pipeline_inflight(requested, batch_points, dim, memory_budget_bytes)
        .max(pipeline_inflight.max(1))
}

#[allow(clippy::too_many_arguments)]
fn fill_d1_sequential_ads_plan_queue(
    sender: &mut Option<crossbeam_channel::Sender<(usize, D1LevelScanBatchPlan)>>,
    pending_plan: &mut Option<(usize, D1LevelScanBatchPlan)>,
    heap: &mut BinaryHeap<D1LevelScanHeapEntry>,
    runs: &[D1LevelScanRunWork],
    batch_points: usize,
    next_batch_idx: &mut usize,
    sent_batches: &mut usize,
    applied_batches: usize,
    max_outstanding_batches: usize,
) -> AnnResult<bool> {
    let Some(tx) = sender.as_ref().cloned() else {
        return Ok(true);
    };

    while sent_batches.saturating_sub(applied_batches) < max_outstanding_batches.max(1) {
        let item = if let Some(item) = pending_plan.take() {
            item
        } else {
            let Some(plan) = next_d1_level_scan_batch(heap, runs, batch_points) else {
                *sender = None;
                return Ok(true);
            };
            let batch_idx = *next_batch_idx;
            *next_batch_idx += 1;
            (batch_idx, plan)
        };

        match tx.try_send(item) {
            Ok(()) => {
                *sent_batches += 1;
            }
            Err(crossbeam_channel::TrySendError::Full(item)) => {
                *pending_plan = Some(item);
                return Ok(false);
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                return Err(AnnError::log_index_error(
                    "D1 sequential ADS reader stopped before accepting the next plan".to_string(),
                ));
            }
        }
    }

    Ok(false)
}

fn compute_d1_level_scan_loaded_batch(
    loaded: D1LevelScanLoadedBatch,
    dim: usize,
    compute_runs: &[D1LevelScanRunCompute],
    _params: &ForgeANNParams,
    telemetry: &D1AdsComputeSharedTelemetry,
) -> AnnResult<D1LevelScanComputedBatch> {
    let batch_idx = loaded.batch_idx;
    let read_wall = loaded.read_wall;
    let mut grouped = (0..compute_runs.len())
        .map(|_| None)
        .collect::<Vec<Option<D1LevelScanRunBatchInput>>>();
    for (point_offset, occurrences) in loaded.plan.occurrences_by_point.iter().enumerate() {
        let point_id = loaded.plan.point_ids[point_offset];
        for occurrence in occurrences {
            let input = grouped[occurrence.run_idx].get_or_insert_with(Default::default);
            input.point_ids.push(point_id);
            input.source_offsets.push(point_offset);
            input.local_indices.push(occurrence.local_idx);
        }
    }
    let total_points = grouped
        .iter()
        .filter_map(Option::as_ref)
        .map(|input| input.point_ids.len())
        .sum::<usize>();
    let chunk_points = d1_ads_compute_chunk_points(
        total_points,
        rayon::current_num_threads().max(1),
        ForgeANNParams::ADS_CHUNKS_PER_WORKER,
    );
    let tasks = build_d1_level_scan_compute_tasks(&grouped, chunk_points);

    let compute_start = Instant::now();
    let mut outputs = tasks
        .into_par_iter()
        .map(|task| -> AnnResult<D1LevelScanRunBatchOutput> {
            let run = &compute_runs[task.run_idx];
            let input = grouped[task.run_idx]
                .as_ref()
                .expect("D1 ADS compute task must reference an existing run input");
            let chunk_start = Instant::now();
            let active = telemetry.active_chunks.fetch_add(1, Ordering::AcqRel) + 1;
            update_atomic_max_usize(&telemetry.active_chunk_max, active);
            telemetry
                .active_chunk_start_sum
                .fetch_add(active as u64, Ordering::Relaxed);
            let chunk = assign_loaded_point_indexed_adsampling_layout(
                &input.point_ids[task.point_start..task.point_end],
                &loaded.data,
                &input.source_offsets[task.point_start..task.point_end],
                task.run_point_start,
                dim,
                &run.leader_layout,
                run.fanout,
                run.config,
                &run.seed_indices,
                &run.seeded,
            );
            let chunk_wall = chunk_start.elapsed();
            telemetry.active_chunks.fetch_sub(1, Ordering::AcqRel);
            build_d1_level_scan_output(chunk?, chunk_wall, run, task.run_idx, task.run_point_start)
        })
        .collect::<AnnResult<Vec<_>>>()?;
    outputs.sort_unstable_by_key(|output| (output.run_idx, output.run_point_start));
    let compute_wall = compute_start.elapsed();

    let logical_bytes = loaded
        .plan
        .point_ids
        .len()
        .saturating_mul(dim)
        .saturating_mul(size_of::<f32>()) as u64;
    let physical_bytes = loaded.io_stats.bytes_read;
    Ok(D1LevelScanComputedBatch {
        batch_idx,
        outputs,
        io_stats: loaded.io_stats,
        read_wall,
        compute_wall,
        logical_bytes,
        physical_bytes,
    })
}

fn build_d1_level_scan_output(
    chunk: AdSamplingChunkResult,
    chunk_wall: Duration,
    run: &D1LevelScanRunCompute,
    run_idx: usize,
    run_point_start: usize,
) -> AnnResult<D1LevelScanRunBatchOutput> {
    let mut assignment_bytes = Vec::with_capacity(
        chunk
            .chunk
            .leaders_by_point
            .len()
            .saturating_mul(assign_record_width(run.fanout)),
    );
    let mut raw_counts = vec![0usize; run.seeded.len()];
    for leaders_for_point in &chunk.chunk.leaders_by_point {
        let mut record = [0u16; 32];
        for (idx, &leader_idx) in leaders_for_point.iter().enumerate() {
            raw_counts[leader_idx] += 1;
            record[idx] = leader_idx as u16;
        }
        append_assign_record_bytes(
            &mut assignment_bytes,
            &record[..leaders_for_point.len()],
            run.fanout,
        )?;
    }
    Ok(D1LevelScanRunBatchOutput {
        run_idx,
        run_point_start,
        assignment_bytes,
        raw_counts,
        chunk,
        chunk_wall,
    })
}

fn d1_resident_ids_are_strictly_increasing(ids: &[u32]) -> bool {
    ids.windows(2).all(|window| window[0] < window[1])
}

fn d1_point_ids_are_nondecreasing(ids: &[u32]) -> bool {
    ids.windows(2).all(|window| window[0] <= window[1])
}

fn d1_resident_source_offsets(resident_ids: &[u32], point_ids: &[u32]) -> AnnResult<Vec<usize>> {
    let mut offsets = Vec::with_capacity(point_ids.len());
    if d1_point_ids_are_nondecreasing(point_ids) {
        let mut cursor = 0usize;
        for &pid in point_ids {
            while cursor < resident_ids.len() && resident_ids[cursor] < pid {
                cursor += 1;
            }
            if resident_ids.get(cursor).copied() != Some(pid) {
                return Err(AnnError::log_index_error(format!(
                    "D1 resident ADS point id {pid} is not present in resident buffer"
                )));
            }
            offsets.push(cursor);
        }
    } else {
        for &pid in point_ids {
            let offset = resident_ids.binary_search(&pid).map_err(|_| {
                AnnError::log_index_error(format!(
                    "D1 resident ADS point id {pid} is not present in resident buffer"
                ))
            })?;
            offsets.push(offset);
        }
    }
    Ok(offsets)
}

#[allow(clippy::too_many_arguments)]
fn run_d1_resident_ads_buffer_pipeline(
    dataset: &dyn PointStore,
    _params: &ForgeANNParams,
    depth: usize,
    dim: usize,
    work_runs: &mut [D1LevelScanRunWork],
    compute_runs: &[D1LevelScanRunCompute],
    assignment_spool: &mut D1LevelScanAssignmentSpool,
    profile: &mut AdSamplingProfile,
) -> AnnResult<D1SequentialAdsPipelineReport> {
    let Some((resident_ids, resident_data)) = dataset.resident_rows() else {
        return Err(AnnError::log_index_error(
            "D1 resident ADS buffer pipeline requires a resident point store".to_string(),
        ));
    };
    if !d1_resident_ids_are_strictly_increasing(resident_ids) {
        return Err(AnnError::log_index_error(
            "D1 resident ADS buffer requires sorted unique resident ids".to_string(),
        ));
    }
    if dim == 0 || resident_data.len() != resident_ids.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "D1 resident ADS buffer shape mismatch: ids={} data={} dim={dim}",
            resident_ids.len(),
            resident_data.len()
        )));
    }
    if work_runs.len() != compute_runs.len() {
        return Err(AnnError::log_index_error(format!(
            "D1 resident ADS run metadata mismatch: work_runs={} compute_runs={}",
            work_runs.len(),
            compute_runs.len()
        )));
    }

    let total_start = Instant::now();
    let telemetry = D1AdsComputeSharedTelemetry::new();
    let workers = rayon::current_num_threads().max(1);
    let mut prefetch = StrictPrefetchPipelineProfile {
        pipeline: PrefetchPipelineKind::Local,
        ..StrictPrefetchPipelineProfile::default()
    };
    let mut compute_wall_total = Duration::ZERO;
    let mut apply_wall_total = Duration::ZERO;
    let total_points = work_runs.iter().map(|run| run.points.len()).sum::<usize>();
    let chunk_points =
        d1_ads_compute_chunk_points(total_points, workers, ForgeANNParams::ADS_CHUNKS_PER_WORKER);
    let tasks = build_d1_resident_level_scan_compute_tasks(work_runs, chunk_points);
    let task_count = tasks.len();

    profile.scheduler_mode = "depth-wave-resident".to_string();
    if !tasks.is_empty() {
        let compute_start = Instant::now();
        let mut outputs = tasks
            .into_par_iter()
            .map(|task| -> AnnResult<D1LevelScanRunBatchOutput> {
                let work = &work_runs[task.run_idx];
                let run = &compute_runs[task.run_idx];
                let point_ids = &work.points[task.point_start..task.point_end];
                let source_offsets = d1_resident_source_offsets(resident_ids, point_ids)?;
                let chunk_start = Instant::now();
                let active = telemetry.active_chunks.fetch_add(1, Ordering::AcqRel) + 1;
                update_atomic_max_usize(&telemetry.active_chunk_max, active);
                telemetry
                    .active_chunk_start_sum
                    .fetch_add(active as u64, Ordering::Relaxed);
                let chunk = assign_loaded_point_indexed_adsampling_layout(
                    point_ids,
                    resident_data,
                    &source_offsets,
                    task.run_point_start,
                    dim,
                    &run.leader_layout,
                    run.fanout,
                    run.config,
                    &run.seed_indices,
                    &run.seeded,
                );
                let chunk_wall = chunk_start.elapsed();
                telemetry.active_chunks.fetch_sub(1, Ordering::AcqRel);
                build_d1_level_scan_output(
                    chunk?,
                    chunk_wall,
                    run,
                    task.run_idx,
                    task.run_point_start,
                )
            })
            .collect::<AnnResult<Vec<_>>>()?;
        outputs.sort_unstable_by_key(|output| (output.run_idx, output.run_point_start));
        let compute_wall = compute_start.elapsed();
        compute_wall_total += compute_wall;
        profile.compute_ms += duration_ms(compute_wall);

        let logical_bytes = total_points
            .saturating_mul(dim)
            .saturating_mul(size_of::<f32>()) as u64;
        let computed = D1LevelScanComputedBatch {
            batch_idx: 0,
            outputs,
            io_stats: PointBatchStats::default(),
            read_wall: Duration::ZERO,
            compute_wall,
            logical_bytes,
            physical_bytes: 0,
        };
        let apply_start = Instant::now();
        apply_d1_level_scan_computed_batch(computed, work_runs, assignment_spool, profile)?;
        let apply_wall = apply_start.elapsed();
        apply_wall_total += apply_wall;
        profile.apply_ms += duration_ms(apply_wall);
        prefetch.batches += 1;
        prefetch.logical_bytes = prefetch.logical_bytes.saturating_add(logical_bytes);
    }

    profile.chunk_active_max = profile
        .chunk_active_max
        .max(telemetry.active_chunk_max.load(Ordering::Relaxed));
    let active_start_sum = telemetry.active_chunk_start_sum.load(Ordering::Relaxed);
    if profile.chunks > 0 {
        profile.chunk_active_start_avg = active_start_sum as f64 / profile.chunks as f64;
    }
    profile.validation_recall_at_fanout = if profile.validation_recall_total == 0 {
        0.0
    } else {
        profile.validation_recall_hits as f64 / profile.validation_recall_total as f64
    };
    profile.total_ms = duration_ms(total_start.elapsed());
    if profile.total_ms > 0.0 {
        profile.effective_parallelism = profile.chunk_wall_accumulated_ms / profile.total_ms;
    }
    tracing::info!(
        "[adsampling/d1-sequential-ads-resident-buffer] depth={} runs={} resident_rows={} resident_bytes={} batches={} tasks={} chunk_points={} compute_ms={} apply_ms={}",
        depth,
        work_runs.len(),
        resident_ids.len(),
        resident_data.len().saturating_mul(size_of::<f32>()),
        prefetch.batches,
        task_count,
        chunk_points,
        compute_wall_total.as_millis(),
        apply_wall_total.as_millis(),
    );

    let batches = prefetch.batches;
    Ok(D1SequentialAdsPipelineReport {
        prefetch,
        compute_wall: compute_wall_total,
        apply_wall: apply_wall_total,
        batches,
    })
}

fn apply_d1_level_scan_computed_batch(
    computed: D1LevelScanComputedBatch,
    work_runs: &mut [D1LevelScanRunWork],
    assignment_spool: &mut D1LevelScanAssignmentSpool,
    profile: &mut AdSamplingProfile,
) -> AnnResult<()> {
    profile.chunk_active_max = profile.chunk_active_max.max(
        computed
            .outputs
            .len()
            .min(rayon::current_num_threads().max(1)),
    );

    let mut spool_write_wall = Duration::ZERO;
    for output in computed.outputs {
        profile.chunks += 1;
        merge_adsampling_chunk_result_into_profile(profile, &output.chunk, output.chunk_wall);
        let write_start = Instant::now();
        if let Some(segment) = assignment_spool.append(&output.assignment_bytes)? {
            work_runs[output.run_idx].assignment_segments.push(segment);
        }
        spool_write_wall += write_start.elapsed();
        for (idx, count) in output.raw_counts.into_iter().enumerate() {
            work_runs[output.run_idx].raw_counts[idx] += count;
        }
    }
    profile.visit_chunk_ms += duration_ms(spool_write_wall);
    let _ = computed.batch_idx;
    Ok(())
}

pub(crate) fn schedule_depth_wave_child_runs(
    params: &ForgeANNParams,
    child_runs: Vec<ChildRun>,
    depth: usize,
) -> DepthWaveChildRunSchedule {
    let mut wave_runs = Vec::new();
    let exact_runs = Vec::new();

    wave_runs.extend(child_runs);
    let _ = (params, depth);
    DepthWaveChildRunSchedule {
        wave_runs,
        exact_runs,
    }
}

pub(crate) fn should_use_depth_wave_child_run_scheduler(
    _params: &ForgeANNParams,
    depth: usize,
) -> bool {
    depth == 1 && depth <= ForgeANNParams::ADS_WAVE_DEPTHS
}

pub(crate) fn parallel_join_child_runs(
    dataset: &dyn PointStore,
    child_runs: Vec<ChildRun>,
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
    if should_use_depth_wave_child_run_scheduler(params, depth) {
        return parallel_join_child_runs_depth_wave(
            dataset,
            child_runs,
            depth,
            parent_n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            pb,
            external_run_store,
            root_fanout_state,
            assignment_context,
            leaf_emitter,
        );
    }

    parallel_join_child_runs_inline(
        dataset,
        child_runs,
        depth,
        parent_n,
        metric,
        params,
        adaptive_c_max,
        min_recurse_size,
        pb,
        external_run_store,
        root_fanout_state,
        assignment_context,
        leaf_emitter,
    )
}

pub(crate) fn d1_level_scan_batch_points(
    _params: &ForgeANNParams,
    dim: usize,
    memory_budget_bytes: Option<usize>,
) -> usize {
    let workers = rayon::current_num_threads().max(1);
    let point_tile =
        choose_prefetch_batch_points_for_budget(4096, dim, workers, memory_budget_bytes);
    let balanced_window = 262_144usize;
    let max_window = 1_048_576usize;
    let default_points = memory_budget_bytes
        .map(|budget| {
            let row_bytes = dim.max(1).saturating_mul(size_of::<f32>()).max(1);
            let window_bytes = balanced_window.saturating_mul(row_bytes);
            if budget / 4 >= window_bytes {
                balanced_window
            } else {
                point_tile
            }
        })
        .unwrap_or(balanced_window);
    env_usize("FORGEANN_D1_LEVEL_SCAN_BATCH_POINTS")
        .unwrap_or(default_points)
        .max(4096)
        .min(max_window)
}

pub(crate) fn d1_level_scan_memory_budget_bytes(params: &ForgeANNParams) -> Option<usize> {
    let oom_budget = params
        .oom_enable
        .then(|| params.effective_oom_memory_budget_bytes());

    oom_budget
}

pub(crate) fn clamp_d1_level_scan_pipeline_inflight(
    requested: usize,
    batch_points: usize,
    dim: usize,
    memory_budget_bytes: Option<usize>,
) -> usize {
    let requested = requested.max(1);
    let batch_bytes = batch_points
        .max(1)
        .saturating_mul(dim.max(1))
        .saturating_mul(size_of::<f32>())
        .max(1);
    let Some(budget) = memory_budget_bytes.filter(|&budget| budget > 0) else {
        return requested;
    };
    let pipeline_budget = (budget / 4).max(batch_bytes);
    requested.min((pipeline_budget / batch_bytes).max(1))
}

fn d1_level_scan_pipeline_inflight(
    batch_points: usize,
    dim: usize,
    memory_budget_bytes: Option<usize>,
) -> usize {
    let workers = rayon::current_num_threads().max(1);
    let requested = env_usize("FORGEANN_D1_LEVEL_SCAN_INFLIGHT")
        .unwrap_or_else(|| workers.div_ceil(12).clamp(2, 4));
    clamp_d1_level_scan_pipeline_inflight(requested, batch_points, dim, memory_budget_bytes)
}

#[allow(clippy::too_many_arguments)]
fn run_d1_sequential_ads_pipeline(
    dataset: &dyn PointStore,
    params: &ForgeANNParams,
    depth: usize,
    dim: usize,
    work_runs: &mut [D1LevelScanRunWork],
    compute_runs: &[D1LevelScanRunCompute],
    assignment_spool: &mut D1LevelScanAssignmentSpool,
    profile: &mut AdSamplingProfile,
    batch_points: usize,
    pipeline_inflight: usize,
    computed_reorder_capacity: usize,
    read_options: WindowedGatherOptions,
) -> AnnResult<D1SequentialAdsPipelineReport> {
    let scan_start = Instant::now();
    let telemetry = D1AdsComputeSharedTelemetry::new();
    let cancel = Arc::new(AtomicBool::new(false));
    let mut heap = BinaryHeap::new();
    for (run_idx, run) in work_runs.iter().enumerate() {
        if let Some(&point) = run.points.first() {
            heap.push(D1LevelScanHeapEntry {
                point,
                run_idx,
                local_idx: 0,
            });
        }
    }

    let mut prefetch_profile = StrictPrefetchPipelineProfile {
        pipeline: PrefetchPipelineKind::Local,
        queue_depth: pipeline_inflight,
        prefetch_budget_bytes: batch_points
            .saturating_mul(dim)
            .saturating_mul(size_of::<f32>())
            .saturating_mul(pipeline_inflight),
        ..StrictPrefetchPipelineProfile::default()
    };
    let mut compute_wall_total = Duration::ZERO;
    let mut apply_wall_total = Duration::ZERO;
    let mut next_batch_idx = 0usize;
    let mut sent_batches = 0usize;
    let mut applied_batches = 0usize;
    let mut next_apply_batch_idx = 0usize;
    let mut pending_plan = None;
    let mut pending_apply = BTreeMap::new();
    let max_outstanding_batches = computed_reorder_capacity.max(pipeline_inflight).max(1);

    std::thread::scope(|scope| -> AnnResult<()> {
        let (plan_tx, plan_rx) =
            crossbeam_channel::bounded::<(usize, D1LevelScanBatchPlan)>(pipeline_inflight.max(1));
        let mut plan_tx = Some(plan_tx);
        let (loaded_tx, loaded_rx) = crossbeam_channel::bounded::<AnnResult<D1LevelScanLoadedBatch>>(
            pipeline_inflight.max(1),
        );
        let (computed_tx, computed_rx) = crossbeam_channel::bounded::<
            AnnResult<D1LevelScanComputedBatch>,
        >(computed_reorder_capacity.max(1));

        let reader_loaded_tx = loaded_tx.clone();
        let reader_telemetry = telemetry.clone();
        let reader_cancel = Arc::clone(&cancel);
        let reader_handle = scope.spawn(move || {
            while !reader_cancel.load(Ordering::Acquire) {
                let Ok((batch_idx, plan)) = plan_rx.recv() else {
                    break;
                };
                let loaded = read_d1_level_scan_batch(dataset, batch_idx, plan, read_options);
                let loaded_ok = loaded.is_ok();
                if reader_loaded_tx.send(loaded).is_err() {
                    break;
                }
                if loaded_ok {
                    reader_telemetry
                        .read_batches
                        .fetch_add(1, Ordering::Relaxed);
                    update_atomic_max_usize(
                        &reader_telemetry.loaded_queue_depth_peak,
                        reader_loaded_tx.len(),
                    );
                } else {
                    break;
                }
            }
        });
        drop(loaded_tx);

        let compute_worker_count = pipeline_inflight
            .max(1)
            .min(rayon::current_num_threads().max(1));
        let mut compute_handles = Vec::with_capacity(compute_worker_count);
        for _ in 0..compute_worker_count {
            let worker_loaded_rx = loaded_rx.clone();
            let worker_computed_tx = computed_tx.clone();
            let worker_telemetry = telemetry.clone();
            let worker_cancel = Arc::clone(&cancel);
            compute_handles.push(scope.spawn(move || {
                while !worker_cancel.load(Ordering::Acquire) {
                    let Ok(loaded) = worker_loaded_rx.recv() else {
                        break;
                    };
                    let computed = loaded.and_then(|loaded| {
                        compute_d1_level_scan_loaded_batch(
                            loaded,
                            dim,
                            compute_runs,
                            params,
                            &worker_telemetry,
                        )
                    });
                    let computed_ok = computed.is_ok();
                    if worker_computed_tx.send(computed).is_err() {
                        break;
                    }
                    if computed_ok {
                        worker_telemetry
                            .computed_batches
                            .fetch_add(1, Ordering::Relaxed);
                        update_atomic_max_usize(
                            &worker_telemetry.computed_queue_depth_peak,
                            worker_computed_tx.len(),
                        );
                    } else {
                        break;
                    }
                }
            }));
        }
        drop(computed_tx);

        let mut planner_exhausted = fill_d1_sequential_ads_plan_queue(
            &mut plan_tx,
            &mut pending_plan,
            &mut heap,
            work_runs,
            batch_points,
            &mut next_batch_idx,
            &mut sent_batches,
            applied_batches,
            max_outstanding_batches,
        )?;
        let mut pipeline_error = None;
        let mut last_progress_log = Instant::now();

        while !(planner_exhausted && applied_batches == sent_batches) {
            let recv_start = Instant::now();
            let computed = match computed_rx.recv() {
                Ok(computed) => computed,
                Err(_) => {
                    if planner_exhausted && applied_batches == sent_batches {
                        break;
                    }
                    pipeline_error = Some(AnnError::log_index_error(
                        "D1 sequential ADS compute queue closed before all batches applied"
                            .to_string(),
                    ));
                    break;
                }
            };
            let recv_wait = recv_start.elapsed();
            profile.recv_wait_ms += duration_ms(recv_wait);

            let computed = match computed {
                Ok(computed) => computed,
                Err(err) => {
                    pipeline_error = Some(err);
                    break;
                }
            };
            pending_apply.insert(computed.batch_idx, computed);
            profile.ordered_pending_max = profile.ordered_pending_max.max(pending_apply.len());
            profile.computed_queue_depth_max = profile
                .computed_queue_depth_max
                .max(computed_rx.len())
                .max(pending_apply.len());

            let mut recv_wait_charged = false;
            drain_ordered_d1_level_scan_batches(
                &mut pending_apply,
                &mut next_apply_batch_idx,
                |computed| {
                    let read_wall = computed.read_wall;
                    let compute_wall = computed.compute_wall;
                    let io_stats = computed.io_stats.clone();
                    let logical_bytes = computed.logical_bytes;
                    let physical_bytes = computed.physical_bytes;
                    let apply_start = Instant::now();
                    apply_d1_level_scan_computed_batch(
                        computed,
                        work_runs,
                        assignment_spool,
                        profile,
                    )?;
                    let apply_wall = apply_start.elapsed();
                    apply_wall_total += apply_wall;
                    compute_wall_total += compute_wall;
                    profile.read_ms += duration_ms(read_wall);
                    profile.compute_ms += duration_ms(compute_wall);
                    profile.apply_ms += duration_ms(apply_wall);
                    prefetch_profile.batches += 1;
                    prefetch_profile.io_wall += read_wall;
                    if !recv_wait_charged {
                        prefetch_profile.consumer_wait += recv_wait;
                        recv_wait_charged = true;
                    }
                    prefetch_profile.range_reads += io_stats.range_calls;
                    prefetch_profile.point_reads += io_stats.point_calls;
                    prefetch_profile.logical_bytes =
                        prefetch_profile.logical_bytes.saturating_add(logical_bytes);
                    prefetch_profile.physical_bytes = prefetch_profile
                        .physical_bytes
                        .saturating_add(physical_bytes);
                    applied_batches += 1;
                    Ok(())
                },
            )?;

            planner_exhausted = fill_d1_sequential_ads_plan_queue(
                &mut plan_tx,
                &mut pending_plan,
                &mut heap,
                work_runs,
                batch_points,
                &mut next_batch_idx,
                &mut sent_batches,
                applied_batches,
                max_outstanding_batches,
            )?;

            profile.loaded_queue_depth_max = profile
                .loaded_queue_depth_max
                .max(loaded_rx.len())
                .max(telemetry.loaded_queue_depth_peak.load(Ordering::Relaxed));
            profile.computed_queue_depth_max = profile
                .computed_queue_depth_max
                .max(computed_rx.len())
                .max(telemetry.computed_queue_depth_peak.load(Ordering::Relaxed));

            if last_progress_log.elapsed() >= Duration::from_secs(30) {
                tracing::info!(
                    "[adsampling/d1-sequential-ads-progress] depth={} planned_batches={} read_batches={} computed_batches={} applied_batches={} loaded_queue_depth={} pending_apply_batches={} elapsed_ms={} read_ms={:.3} compute_ms={} apply_ms={:.3} spool_bytes={} active_chunk_peak={}",
                    depth,
                    sent_batches,
                    telemetry.read_batches.load(Ordering::Relaxed),
                    telemetry.computed_batches.load(Ordering::Relaxed),
                    applied_batches,
                    loaded_rx.len(),
                    pending_apply.len(),
                    scan_start.elapsed().as_millis(),
                    profile.read_ms,
                    compute_wall_total.as_millis(),
                    profile.apply_ms,
                    assignment_spool.write_offset,
                    telemetry.active_chunk_max.load(Ordering::Relaxed),
                );
                last_progress_log = Instant::now();
            }
        }

        if pipeline_error.is_some() {
            cancel.store(true, Ordering::Release);
            plan_tx = None;
            pending_plan = None;
            while computed_rx.recv().is_ok() {}
        }
        drop(plan_tx);
        drop(pending_plan);
        drop(loaded_rx);
        drop(computed_rx);

        reader_handle.join().map_err(|_| {
            AnnError::log_index_error("D1 sequential ADS reader thread panicked".to_string())
        })?;
        for handle in compute_handles {
            handle.join().map_err(|_| {
                AnnError::log_index_error(
                    "D1 sequential ADS compute worker thread panicked".to_string(),
                )
            })?;
        }

        match pipeline_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    })?;

    profile.chunk_active_max = profile
        .chunk_active_max
        .max(telemetry.active_chunk_max.load(Ordering::Relaxed));
    let active_start_sum = telemetry.active_chunk_start_sum.load(Ordering::Relaxed);
    if profile.chunks > 0 {
        profile.chunk_active_start_avg = active_start_sum as f64 / profile.chunks as f64;
    }
    profile.loaded_queue_depth_max = profile
        .loaded_queue_depth_max
        .max(telemetry.loaded_queue_depth_peak.load(Ordering::Relaxed));
    profile.computed_queue_depth_max = profile
        .computed_queue_depth_max
        .max(telemetry.computed_queue_depth_peak.load(Ordering::Relaxed));
    profile.total_ms = duration_ms(scan_start.elapsed());
    if profile.total_ms > 0.0 {
        profile.effective_parallelism = profile.chunk_wall_accumulated_ms / profile.total_ms;
    }
    profile.validation_recall_at_fanout = if profile.validation_recall_total == 0 {
        0.0
    } else {
        profile.validation_recall_hits as f64 / profile.validation_recall_total as f64
    };

    Ok(D1SequentialAdsPipelineReport {
        prefetch: prefetch_profile,
        compute_wall: compute_wall_total,
        apply_wall: apply_wall_total,
        batches: applied_batches,
    })
}

#[allow(clippy::too_many_arguments)]
fn parallel_join_child_runs_d1_level_scan_ads(
    dataset: &dyn PointStore,
    child_runs: Vec<ChildRun>,
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
    let expected_leaves =
        (child_runs.iter().map(|run| run.len).sum::<usize>() / adaptive_c_max.max(1)).max(4);
    if child_runs.is_empty() {
        return Ok(PartitionStats::new(params.max_depth, expected_leaves));
    }

    let (base_dir, io_cfg) = {
        let guard = external_run_store.lock();
        (guard.base_dir.clone(), guard.io_config())
    };
    let read_runs_start = Instant::now();
    let (child_points, batch_stats) =
        read_child_runs_batched_from_path(&base_dir, io_cfg, depth - 1, &child_runs)?;
    let read_child_runs_wall = read_runs_start.elapsed();
    let input_pairs = child_runs.into_iter().zip(child_points).collect::<Vec<_>>();

    parallel_join_child_runs_d1_level_scan_ads_from_pairs(
        dataset,
        input_pairs,
        depth,
        parent_n,
        metric,
        params,
        adaptive_c_max,
        min_recurse_size,
        pb,
        external_run_store,
        root_fanout_state,
        assignment_context,
        leaf_emitter,
        read_child_runs_wall,
        Some(batch_stats),
    )
}

#[allow(clippy::too_many_arguments)]
fn parallel_join_child_runs_d1_level_scan_ads_from_pairs(
    dataset: &dyn PointStore,
    input_pairs: Vec<(ChildRun, Vec<u32>)>,
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
    read_child_runs_wall: Duration,
    child_extent_batch_stats: Option<ChildRunExtentBatchStats>,
) -> AnnResult<PartitionStats> {
    let expected_leaves = (input_pairs
        .iter()
        .map(|(_, points)| points.len())
        .sum::<usize>()
        / adaptive_c_max.max(1))
    .max(4);
    let mut stats = PartitionStats::new(params.max_depth, expected_leaves);
    if input_pairs.is_empty() {
        return Ok(stats);
    }
    if let Some(batch_stats) = child_extent_batch_stats {
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
    }
    let (base_dir, io_cfg) = {
        let guard = external_run_store.lock();
        (guard.base_dir.clone(), guard.io_config())
    };
    let d1_level_leaf_emitter: &dyn LeafEmitter = leaf_emitter;
    let scan_context = assignment_context
        .for_depth(depth)
        .with_depth_wave_assignment();
    let mut work_runs = Vec::new();
    let mut compute_runs = Vec::new();
    let mut fallback_clusters = Vec::new();
    let mut resident_subtree_selected_runs = 0usize;
    let mut resident_subtree_groups = 0usize;
    let mut resident_subtree_rejected = 0usize;
    let dim = dataset.dim();
    let memory_budget_bytes = d1_level_scan_memory_budget_bytes(params);
    let resident_subtree_config = if dataset.is_resident_subset() {
        None
    } else {
        D1ResidentSubtreeConfig::from_params(params, adaptive_c_max)
    };
    let profile_config =
        AdSamplingConfig::depth_from_params(params, params.fanout_for_depth(depth).max(1));
    let mut profile = AdSamplingProfile {
        depth,
        epsilon: profile_config.epsilon,
        group_dims: profile_config.group_dims,
        scheduler_mode: "depth-wave".to_string(),
        called_inside_rayon_worker: rayon::current_thread_index().is_some(),
        ..AdSamplingProfile::default()
    };

    let prep_start = Instant::now();
    let scan_pairs = if let Some(config) = resident_subtree_config.as_ref() {
        match maybe_build_d1_resident_subtree_scope(
            input_pairs,
            config,
            dim,
            parent_n,
            adaptive_c_max,
        ) {
            Ok(scope_group) => {
                resident_subtree_selected_runs = scope_group.source_runs;
                resident_subtree_groups = 1;
                tracing::info!(
                    "[adsampling/d1-resident-subtree-filter] depth={} candidates={} selected_runs={} groups={} rejected=0 scope_mode=whole min_buffer_ratio={:.4} scope_min_buffer_ratio={:.4} max_resident_ratio={:.4} max_resident_bytes={} effective_group_budget_bytes={} max_native_expected_leaves={} max_read_amp={:.3} budget_bytes={} max_group_bytes={} scope_budget_bytes={} enable_source={}",
                    depth,
                    resident_subtree_selected_runs,
                    resident_subtree_selected_runs,
                    resident_subtree_groups,
                    config.min_buffer_ratio,
                    config.whole_scope_min_buffer_ratio,
                    config.max_dataset_resident_ratio,
                    config.max_resident_bytes_for(parent_n, dim),
                    config.grouped_budget_bytes_for(parent_n, dim),
                    config.max_native_expected_leaves,
                    config.max_read_amplification,
                    config.budget_bytes,
                    config.max_run_bytes,
                    config.whole_scope_budget_bytes,
                    config.enable_source,
                );
                let resident_stats = process_d1_resident_subtree_groups(
                    dataset,
                    vec![scope_group],
                    config,
                    1,
                    depth,
                    parent_n,
                    metric,
                    params,
                    adaptive_c_max,
                    min_recurse_size,
                    pb,
                    external_run_store,
                    root_fanout_state,
                    assignment_context,
                    leaf_emitter,
                )?;
                stats.merge_from(resident_stats);
                Vec::new()
            }
            Err(scope_reject) => {
                if env_bool("FORGEANN_D1_RESIDENT_SUBTREE_REQUIRE_WHOLE") {
                    tracing::error!(
                        "[adsampling/d1-resident-subtree-native-reject] depth={} reason={} scope_runs={} scope_points={} scope_unique_points={} scope_resident_bytes={} scope_logical_bytes={} scope_physical_bytes={} scope_planned_windows={} scope_buffer_ratio={:.4} max_resident_ratio={:.4} max_resident_bytes={} max_read_amp={:.3} budget_bytes={} max_group_bytes={} scope_budget_bytes={} enable_source={}",
                        depth,
                        scope_reject.reason,
                        scope_reject.source_runs,
                        scope_reject.selected_points,
                        scope_reject.unique_points,
                        scope_reject.resident_bytes,
                        scope_reject.logical_bytes,
                        scope_reject.physical_bytes,
                        scope_reject.planned_windows,
                        scope_reject.buffer_ratio,
                        config.max_dataset_resident_ratio,
                        config.max_resident_bytes_for(parent_n, dim),
                        config.max_read_amplification,
                        config.budget_bytes,
                        config.max_run_bytes,
                        config.whole_scope_budget_bytes,
                        config.enable_source,
                    );
                    return Err(AnnError::log_index_error(format!(
                        "D1 resident subtree native whole-scope required but rejected: reason={} scope_runs={} scope_points={} scope_unique_points={} scope_resident_bytes={} max_resident_bytes={}",
                        scope_reject.reason,
                        scope_reject.source_runs,
                        scope_reject.selected_points,
                        scope_reject.unique_points,
                        scope_reject.resident_bytes,
                        config.max_resident_bytes_for(parent_n, dim),
                    )));
                }
                let input_pairs = scope_reject.pairs;
                let requested_group_pipeline_slots = config.group_pipeline_slots.max(1);
                let sketch_width = leaf_emitter
                    .sketch_accessor()
                    .map(|sketches| sketches.width())
                    .unwrap_or(0);
                let raw_single_slot_group_budget_bytes =
                    config.grouped_budget_bytes_for(parent_n, dim);
                let raw_pipeline_group_budget_bytes =
                    config.pipelined_group_budget_bytes_for(parent_n, dim);
                let single_slot_group_budget_bytes =
                    d1_resident_subtree_vector_budget_for_total_resident_budget(
                        raw_single_slot_group_budget_bytes,
                        dim,
                        sketch_width,
                    );
                let pipeline_group_budget_bytes =
                    d1_resident_subtree_vector_budget_for_total_resident_budget(
                        raw_pipeline_group_budget_bytes,
                        dim,
                        sketch_width,
                    );
                let mut group_pipeline_slots = requested_group_pipeline_slots;
                let group_budget_bytes = if group_pipeline_slots > 1 {
                    pipeline_group_budget_bytes
                } else {
                    single_slot_group_budget_bytes
                };
                let grouping = group_d1_resident_subtree_inputs_with_budget(
                    input_pairs,
                    config,
                    dim,
                    adaptive_c_max,
                    group_budget_bytes,
                );
                let grouped_physical_bytes = d1_resident_grouping_physical_bytes(&grouping);
                let grouped_read_amp =
                    config.grouped_dataset_read_amp(grouped_physical_bytes, parent_n, dim);
                if grouping.groups.len() < 2 {
                    group_pipeline_slots = 1;
                }
                resident_subtree_selected_runs = grouping.selected_runs;
                resident_subtree_rejected = grouping.rejected_runs;
                resident_subtree_groups = grouping.groups.len();
                tracing::info!(
                    "[adsampling/d1-resident-subtree-filter] depth={} candidates={} selected_runs={} groups={} rejected={} scope_mode=grouped scope_reject={} scope_runs={} scope_points={} scope_unique_points={} scope_resident_bytes={} scope_logical_bytes={} scope_physical_bytes={} scope_planned_windows={} scope_buffer_ratio={:.4} grouped_physical_bytes={} grouped_dataset_read_amp={:.3} grouped_dataset_read_amp_reference={:.3} min_buffer_ratio={:.4} scope_min_buffer_ratio={:.4} max_resident_ratio={:.4} max_resident_bytes={} effective_group_budget_bytes={} single_slot_group_budget_bytes={} pipeline_group_budget_bytes={} max_native_expected_leaves={} requested_group_pipeline_slots={} group_pipeline_slots={} max_read_amp={:.3} budget_bytes={} max_group_bytes={} scope_budget_bytes={} enable_source={}",
                    depth,
                    grouping.candidates,
                    resident_subtree_selected_runs,
                    resident_subtree_groups,
                    resident_subtree_rejected,
                    scope_reject.reason,
                    scope_reject.source_runs,
                    scope_reject.selected_points,
                    scope_reject.unique_points,
                    scope_reject.resident_bytes,
                    scope_reject.logical_bytes,
                    scope_reject.physical_bytes,
                    scope_reject.planned_windows,
                    scope_reject.buffer_ratio,
                    grouped_physical_bytes,
                    grouped_read_amp,
                    config.max_grouped_dataset_read_amp,
                    config.min_buffer_ratio,
                    config.whole_scope_min_buffer_ratio,
                    config.max_dataset_resident_ratio,
                    config.max_resident_bytes_for(parent_n, dim),
                    group_budget_bytes,
                    single_slot_group_budget_bytes,
                    pipeline_group_budget_bytes,
                    config.max_native_expected_leaves,
                    requested_group_pipeline_slots,
                    group_pipeline_slots,
                    config.max_read_amplification,
                    config.budget_bytes,
                    config.max_run_bytes,
                    config.whole_scope_budget_bytes,
                    config.enable_source,
                );
                let levelwise_pairs = grouping.levelwise_pairs;
                if !levelwise_pairs.is_empty() {
                    let levelwise_points = levelwise_pairs
                        .iter()
                        .map(|(_, points)| points.len())
                        .sum::<usize>();
                    let levelwise_max_points = levelwise_pairs
                        .iter()
                        .map(|(_, points)| points.len())
                        .max()
                        .unwrap_or(0);
                    tracing::warn!(
                        "[adsampling/d1-resident-subtree-levelwise-selected] depth={} runs={} points={} max_points={} reason=oversized_or_resident_budget max_native_expected_leaves={}",
                        depth,
                        levelwise_pairs.len(),
                        levelwise_points,
                        levelwise_max_points,
                        config.max_native_expected_leaves,
                    );
                }
                if !grouping.groups.is_empty() {
                    let resident_stats = process_d1_resident_subtree_groups(
                        dataset,
                        grouping.groups,
                        config,
                        group_pipeline_slots,
                        depth,
                        parent_n,
                        metric,
                        params,
                        adaptive_c_max,
                        min_recurse_size,
                        pb,
                        external_run_store,
                        root_fanout_state,
                        assignment_context,
                        leaf_emitter,
                    )?;
                    stats.merge_from(resident_stats);
                }
                levelwise_pairs
            }
        }
    } else {
        input_pairs
    };

    for (child_run, points) in scan_pairs {
        let original_n = points.len();
        stats.max_depth_seen = stats.max_depth_seen.max(depth);
        stats.max_cluster_size_seen = stats.max_cluster_size_seen.max(original_n);
        if original_n == 0 {
            continue;
        }

        if depth >= params.max_depth {
            let mut points = points;
            emit_leaf(
                dataset,
                metric,
                params,
                &mut points,
                depth,
                LeafReason::MaxDepth,
                true,
                params.kernel_safe_leaf_size(),
                &mut stats,
                d1_level_leaf_emitter,
            )?;
            continue;
        }

        let mut points = points;

        let mut local_fanout = params.adaptive_fanout(original_n, depth);
        let dedup_cur = should_dedup_cluster(params, depth, local_fanout);
        let dedup_start = Instant::now();
        if dedup_cur {
            points.sort_unstable();
            points.dedup();
        }
        stats.record_phase_time(RbcPhase::CurDedup, dedup_start.elapsed());
        stats.record_dedup(dedup_cur, original_n, points.len());
        stats.record_assignments(original_n, points.len());
        let n = points.len();
        if n == 0 {
            continue;
        }
        if n <= adaptive_c_max {
            emit_leaf(
                dataset,
                metric,
                params,
                &mut points,
                depth,
                LeafReason::NaturalSize,
                false,
                params.kernel_safe_leaf_size(),
                &mut stats,
                d1_level_leaf_emitter,
            )?;
            continue;
        }
        if depth >= 3 && n <= min_recurse_size {
            emit_leaf(
                dataset,
                metric,
                params,
                &mut points,
                depth,
                LeafReason::MinRecurse,
                false,
                params.kernel_safe_leaf_size(),
                &mut stats,
                d1_level_leaf_emitter,
            )?;
            continue;
        }

        let adaptive_psamp = params.adaptive_psamp_fraction(n, depth);
        let max_leaders = params.max_leaders.min(n);
        let mut num_leaders = ((adaptive_psamp * n as f64).round() as usize)
            .max(2)
            .min(max_leaders);
        let mut rng = StdRng::seed_from_u64(
            child_run.seed ^ (depth as u64).wrapping_mul(6364136223846793005) ^ (n as u64),
        );
        let sample_seed: u64 = rng.random();
        let leaders = sample_set_bottomk(&points, num_leaders, sample_seed);
        num_leaders = leaders.len();
        if num_leaders > 1 {
            local_fanout = local_fanout.min(num_leaders);
        } else {
            local_fanout = 1;
        }
        stats.record_partition_attempt(depth, n, num_leaders, local_fanout);

        if metric != Metric::L2
            || !scan_context.should_use_adsampling_assignment(n, num_leaders, local_fanout)
        {
            fallback_clusters.push(SeededCluster {
                points,
                seed: child_run.seed,
            });
            continue;
        }

        let fanout = assign_record_fanout(local_fanout.min(num_leaders.max(1)));
        let layout_start = Instant::now();
        let leader_layout = AdSamplingLeaderLayout::build(dataset, &leaders)?;
        profile.layout_ms += duration_ms(layout_start.elapsed());
        let config = AdSamplingConfig::depth_from_params(params, fanout);
        let seed_indices = adsampling_seed_indices(num_leaders, config.seed_exact, fanout);
        let seeded = adsampling_seed_mask(num_leaders, &seed_indices);
        profile.points += n;
        profile.leaders += num_leaders;
        profile.fanout = profile.fanout.max(fanout);
        profile.seed_exact_m = profile.seed_exact_m.max(seed_indices.len());

        compute_runs.push(D1LevelScanRunCompute {
            leader_layout,
            config,
            seed_indices,
            seeded,
            fanout,
        });
        work_runs.push(D1LevelScanRunWork {
            points,
            seed: child_run.seed,
            raw_counts: vec![0usize; num_leaders],
            assignment_segments: Vec::new(),
            leaders: num_leaders,
            fanout,
            max_leaders,
        });
    }

    tracing::info!(
        "[adsampling/d1-sequential-ads-plan] depth={} manifest_runs={} scan_runs={} fallback_runs={} resident_subtree_groups={} resident_subtree_source_runs={} resident_subtree_rejected_runs={} scan_points={} read_child_runs_ms={} prep_ms={}",
        depth,
        work_runs.len() + fallback_clusters.len(),
        work_runs.len(),
        fallback_clusters.len(),
        resident_subtree_groups,
        resident_subtree_selected_runs,
        resident_subtree_rejected,
        profile.points,
        read_child_runs_wall.as_millis(),
        prep_start.elapsed().as_millis(),
    );

    let mut assignment_spool = D1LevelScanAssignmentSpool::new_in(&base_dir)?;
    if !work_runs.is_empty() {
        let scan_start = Instant::now();
        let pipeline_report = if dataset.resident_rows().is_some() {
            tracing::info!(
                "[adsampling/d1-sequential-ads-pipeline] depth={} batch_points=0 loaded_inflight=0 computed_reorder_capacity=0 compute_workers={} shared_spool=true resident_buffer=true",
                depth,
                rayon::current_num_threads().max(1),
            );
            run_d1_resident_ads_buffer_pipeline(
                dataset,
                params,
                depth,
                dim,
                &mut work_runs,
                &compute_runs,
                &mut assignment_spool,
                &mut profile,
            )?
        } else {
            let batch_points = d1_level_scan_batch_points(params, dim, memory_budget_bytes);
            let pipeline_inflight =
                d1_level_scan_pipeline_inflight(batch_points, dim, memory_budget_bytes);
            let computed_reorder_capacity = d1_level_scan_computed_reorder_capacity(
                pipeline_inflight,
                batch_points,
                dim,
                memory_budget_bytes,
            );
            let read_options = default_rbc_windowed_options(dataset, batch_points);
            tracing::info!(
                "[adsampling/d1-sequential-ads-pipeline] depth={} batch_points={} loaded_inflight={} computed_reorder_capacity={} compute_workers={} shared_spool=true resident_buffer=false",
                depth,
                batch_points,
                pipeline_inflight,
                computed_reorder_capacity,
                pipeline_inflight
                    .max(1)
                    .min(rayon::current_num_threads().max(1)),
            );
            run_d1_sequential_ads_pipeline(
                dataset,
                params,
                depth,
                dim,
                &mut work_runs,
                &compute_runs,
                &mut assignment_spool,
                &mut profile,
                batch_points,
                pipeline_inflight,
                computed_reorder_capacity,
                read_options,
            )?
        };
        let flush_start = Instant::now();
        assignment_spool.writer.flush()?;
        let assignment_flush_wall = flush_start.elapsed();
        tracing::info!(
            "[adsampling/d1-sequential-ads-apply] depth={} batches={} apply_ms={:.3} flush_ms={} spool_bytes={}",
            depth,
            pipeline_report.batches,
            pipeline_report.apply_wall.as_secs_f64() * 1000.0,
            assignment_flush_wall.as_millis(),
            assignment_spool.write_offset,
        );
        scan_context.record_adsampling_scheduler_profile(&profile);
        log_adsampling_profile(&profile);

        let mut gemm_profile = GemmProfile {
            total_wall: Duration::from_secs_f64(profile.total_ms / 1000.0),
            topk: pipeline_report.compute_wall,
            blocks: profile.chunks,
            prefetch: pipeline_report.prefetch,
            assignment_decision: Some(AssignmentDecisionRecord {
                depth,
                points: profile.points,
                leaders: profile.leaders,
                fanout: profile.fanout,
                wall: Duration::from_secs_f64(profile.total_ms / 1000.0),
                adsampling: true,
                fallback_reason: None,
                recall_at_fanout: profile.validation_recall_at_fanout,
                mismatches: profile.validation_mismatches,
            }),
            ..GemmProfile::default()
        };
        gemm_profile.flush = assignment_flush_wall;
        stats.record_phase_time(RbcPhase::ClusterAssign, scan_start.elapsed());
        stats.record_gemm_profile_with_context(gemm_profile, Some(&scan_context));
    }

    let native_inline_child_clusters = dataset.is_resident_subset();
    let mut next_depth_runs = Vec::new();
    let mut native_next_clusters = Vec::new();
    let mut native_next_points = 0usize;
    let materialize_stage_start = Instant::now();
    let mut d1_merge_wall = Duration::ZERO;
    let mut d1_materialize_wall = Duration::ZERO;
    let mut d1_read_child_wall = Duration::ZERO;
    let mut d1_dedup_wall = Duration::ZERO;
    let mut d1_write_child_wall = Duration::ZERO;
    let mut d1_leaf_emit_wall = Duration::ZERO;
    let mut d1_retry_recurse_wall = Duration::ZERO;
    let mut d1_materialized_children = 0usize;
    let mut d1_written_child_runs = 0usize;
    let d1_work_run_count = work_runs.len();
    let d1_scheduler_telemetry_start = leaf_emitter.scheduler_telemetry();
    let worker_threads = rayon::current_num_threads().max(1);
    let d1_producer_leaf_backlog_soft_limit = d1_scheduler_telemetry_start
        .map(|telemetry| telemetry.producer_leaf_backlog_soft_limit)
        .unwrap_or(0);
    let d1_leaf_drainer_limit = d1_materialize_leaf_drainer_limit_for_policy(worker_threads);
    let d1_leaf_drainer_backpressure_limit = d1_leaf_drainer_limit
        .map(|limit| d1_materialize_leaf_drainer_backpressure_limit(worker_threads, limit));
    let d1_leaf_backlog_soft_limit = d1_materialize_leaf_backlog_soft_limit_for_policy();
    let mut d1_materialize_leaf_emitter = D1MaterializeLeafEmitter::new_with_backlog_soft_limit(
        d1_level_leaf_emitter,
        d1_leaf_backlog_soft_limit,
    );
    d1_materialize_leaf_emitter.configure_resident_batch_for_d1(params, worker_threads);
    tracing::info!(
        "[adsampling/d1-materialize-start] depth={} runs={} native_inline_child_clusters={} producer_leaf_backlog_soft_limit={} leaf_drainer_limit={} leaf_drainer_backpressure_limit={} leaf_backlog_soft_limit={} resident_leaf_batch_max_leaves={} resident_leaf_batch_max_points={}",
        depth,
        d1_work_run_count,
        native_inline_child_clusters,
        d1_producer_leaf_backlog_soft_limit,
        d1_leaf_drainer_limit.unwrap_or(0),
        d1_leaf_drainer_backpressure_limit.unwrap_or(0),
        d1_leaf_backlog_soft_limit.unwrap_or(0),
        d1_materialize_leaf_emitter.resident_batch_max_leaves,
        d1_materialize_leaf_emitter.resident_batch_max_points,
    );
    let mut d1_materialize_progress_last = Instant::now();
    std::thread::scope(|scope| -> AnnResult<()> {
        let _ = scope;
        let _d1_leaf_drainer_limit_guard = d1_leaf_drainer_limit
            .zip(d1_leaf_drainer_backpressure_limit)
            .and_then(|(limit, backpressure_limit)| {
                enter_leaf_drainer_limit_guard(leaf_emitter, limit, backpressure_limit)
            });
        for (run_idx, mut run) in work_runs.into_iter().enumerate() {
            if run_idx > 0 && d1_materialize_progress_last.elapsed() >= Duration::from_secs(30) {
                tracing::info!(
                    "[adsampling/d1-materialize-progress] depth={} completed_runs={}/{} materialized_children={} next_depth_runs={} leaf_emit_calls={} leaf_emit_ms={} elapsed_ms={}",
                    depth,
                    run_idx,
                    d1_work_run_count,
                    d1_materialized_children,
                    next_depth_runs.len(),
                    d1_materialize_leaf_emitter.emit_calls(),
                    d1_materialize_leaf_emitter.emit_ms(),
                    materialize_stage_start.elapsed().as_millis(),
                );
                d1_materialize_progress_last = Instant::now();
            }
            let raw_cluster_count = run.raw_counts.len();
            let empty_clusters = run.raw_counts.iter().filter(|&&count| count == 0).count();
            let pre_merge_clusters = raw_cluster_count.saturating_sub(empty_clusters);
            let assignments_before_merge = run.raw_counts.iter().sum::<usize>();
            let merge_start = Instant::now();
            let merge_groups = merge_cluster_plan(&run.raw_counts, params.c_min, adaptive_c_max);
            let merge_wall = merge_start.elapsed();
            d1_merge_wall += merge_wall;
            stats.record_phase_time(RbcPhase::MergeClusters, merge_wall);
            stats.record_merge(pre_merge_clusters, merge_groups.len(), empty_clusters);
            stats.record_assignments(assignments_before_merge, assignments_before_merge);

            if merge_groups.is_empty() {
                stats.record_partition_result(PartitionResult::FailedEmpty);
                let leaf_start = Instant::now();
                emit_leaf(
                    dataset,
                    metric,
                    params,
                    &mut run.points,
                    depth,
                    LeafReason::PartitionFallback,
                    false,
                    params.kernel_safe_leaf_size(),
                    &mut stats,
                    &d1_materialize_leaf_emitter,
                )?;
                d1_leaf_emit_wall += leaf_start.elapsed();
                continue;
            }

            if merge_groups.len() == 1 {
                if run.leaders < run.max_leaders {
                    stats.record_retry_escalation();
                    let child_start = Instant::now();
                    let child_stats = if native_inline_child_clusters {
                        rbc_recurse_parallel(
                            dataset,
                            run.points,
                            depth,
                            parent_n,
                            metric,
                            params,
                            adaptive_c_max,
                            min_recurse_size,
                            run.seed,
                            pb,
                            None,
                            root_fanout_state,
                            &scan_context,
                            d1_level_leaf_emitter,
                        )?
                    } else {
                        recurse_child_points_external(
                            dataset,
                            run.points,
                            depth,
                            parent_n,
                            metric,
                            params,
                            adaptive_c_max,
                            min_recurse_size,
                            run.seed,
                            pb,
                            external_run_store,
                            root_fanout_state,
                            &scan_context,
                            d1_level_leaf_emitter,
                        )?
                    };
                    d1_retry_recurse_wall += child_start.elapsed();
                    stats.merge_from(child_stats);
                } else {
                    stats.record_partition_result(PartitionResult::FailedNoSplit);
                    let leaf_start = Instant::now();
                    emit_leaf(
                        dataset,
                        metric,
                        params,
                        &mut run.points,
                        depth,
                        LeafReason::PartitionFallback,
                        false,
                        params.kernel_safe_leaf_size(),
                        &mut stats,
                        &d1_materialize_leaf_emitter,
                    )?;
                    d1_leaf_emit_wall += leaf_start.elapsed();
                }
                continue;
            }

            stats.record_partition_result(PartitionResult::SuccessFirst);
            let materialize_start = Instant::now();
            let materialize_inline_limit = if native_inline_child_clusters {
                Some(usize::MAX)
            } else {
                Some(adaptive_c_max)
            };
            let materialized = {
                let mut guard = external_run_store.lock();
                materialize_merged_children_from_spool_segments_with_inline_limit(
                    &run.points,
                    depth,
                    run.fanout,
                    &merge_groups,
                    &assignment_spool.spool,
                    &run.assignment_segments,
                    &mut guard,
                    materialize_inline_limit,
                )?
            };
            d1_materialize_wall += materialize_start.elapsed();
            d1_materialized_children += materialized.len();
            let dedup_merged = should_dedup_cluster(params, depth, run.fanout);
            let child_seed_base = run
                .seed
                .wrapping_add(depth as u64 + 1)
                .wrapping_mul(0x9e3779b97f4a7c15);
            for (cluster_idx, child) in materialized.into_iter().enumerate() {
                let mut cluster = if let Some(points) = child.points {
                    points
                } else {
                    let read_child_start = Instant::now();
                    let guard = external_run_store.lock();
                    let points = read_child_run_chain(&guard, depth, &child.extents)?;
                    d1_read_child_wall += read_child_start.elapsed();
                    points
                };
                if dedup_merged {
                    let dedup_start = Instant::now();
                    let before = cluster.len();
                    cluster.sort_unstable();
                    cluster.dedup();
                    let after = cluster.len();
                    let dedup_wall = dedup_start.elapsed();
                    d1_dedup_wall += dedup_wall;
                    stats.record_phase_time(RbcPhase::MergedDedup, dedup_wall);
                    stats.record_dedup(true, before, after);
                    stats.record_assignments(before, after);
                } else {
                    stats.record_dedup(false, cluster.len(), cluster.len());
                    stats.record_assignments(cluster.len(), cluster.len());
                }
                if cluster.is_empty() {
                    continue;
                }
                let cluster_seed = child_seed_base
                    ^ (cluster_idx as u64).wrapping_mul(0xbf58476d1ce4e5b9)
                    ^ (cluster.len() as u64).rotate_left(17);
                if cluster.len() > adaptive_c_max {
                    if native_inline_child_clusters {
                        let cluster_len = cluster.len();
                        native_next_points = native_next_points.saturating_add(cluster_len);
                        native_next_clusters.push(SeededCluster {
                            points: cluster,
                            seed: cluster_seed,
                        });
                    } else if depth <= EXTERNAL_CHILD_DEPTH_LIMIT {
                        let cluster_len = cluster.len();
                        let write_child_start = Instant::now();
                        let extent = {
                            let mut guard = external_run_store.lock();
                            write_child_run(&mut guard, depth, &cluster)?
                        };
                        d1_write_child_wall += write_child_start.elapsed();
                        d1_written_child_runs += 1;
                        let child_run = ChildRun {
                            extents: vec![extent],
                            len: cluster_len,
                            seed: cluster_seed,
                        };
                        next_depth_runs.push(child_run);
                    } else {
                        fallback_clusters.push(SeededCluster {
                            points: cluster,
                            seed: cluster_seed,
                        });
                    }
                } else {
                    let leaf_start = Instant::now();
                    emit_leaf(
                        dataset,
                        metric,
                        params,
                        &mut cluster,
                        depth + 1,
                        LeafReason::NaturalSize,
                        false,
                        params.kernel_safe_leaf_size(),
                        &mut stats,
                        &d1_materialize_leaf_emitter,
                    )?;
                    d1_leaf_emit_wall += leaf_start.elapsed();
                }
            }
        }
        Ok(())
    })?;
    d1_materialize_leaf_emitter.flush_resident_batch(dataset)?;
    let next_depth_points = next_depth_runs.iter().map(|run| run.len).sum::<usize>();
    let d1_scheduler_telemetry_end = leaf_emitter.scheduler_telemetry();
    let (
        d1_producer_help_drains,
        d1_producer_help_drain_ms,
        d1_backlog_full_help_drains,
        d1_backlog_full_help_ms,
        d1_producer_help_yields,
        d1_peak_leaf_backlog,
        d1_leaf_backlog_end,
        d1_active_leaf_drainers_end,
    ) = match (d1_scheduler_telemetry_start, d1_scheduler_telemetry_end) {
        (Some(start), Some(end)) => (
            end.producer_help_drains
                .saturating_sub(start.producer_help_drains),
            end.producer_help_drain_ms
                .saturating_sub(start.producer_help_drain_ms),
            end.backlog_full_help_drains
                .saturating_sub(start.backlog_full_help_drains),
            end.backlog_full_help_ms
                .saturating_sub(start.backlog_full_help_ms),
            end.producer_help_yields
                .saturating_sub(start.producer_help_yields),
            end.peak_leaf_backlog
                .saturating_sub(start.peak_leaf_backlog),
            end.leaf_backlog,
            end.active_leaf_drainers,
        ),
        _ => (0, 0, 0, 0, 0, 0, 0, 0),
    };
    tracing::info!(
        "[adsampling/d1-materialize] depth={} runs={} materialized_children={} next_depth_runs={} next_depth_points={} native_next_clusters={} native_next_points={} native_inline_child_clusters={} elapsed_ms={} merge_ms={} materialize_ms={} read_child_ms={} dedup_ms={} write_child_ms={} leaf_emit_ms={} leaf_emit_calls={} leaf_emit_inner_ms={} resident_leaf_batch_flushes={} leaf_throttle_calls={} leaf_throttle_ms={} producer_leaf_backlog_soft_limit={} producer_help_drains={} producer_help_drain_ms={} backlog_full_help_drains={} backlog_full_help_ms={} producer_help_yields={} peak_leaf_backlog_delta={} leaf_backlog_end={} active_leaf_drainers={} retry_recurse_ms={} written_child_runs={}",
        depth,
        d1_work_run_count,
        d1_materialized_children,
        next_depth_runs.len(),
        next_depth_points,
        native_next_clusters.len(),
        native_next_points,
        native_inline_child_clusters,
        materialize_stage_start.elapsed().as_millis(),
        d1_merge_wall.as_millis(),
        d1_materialize_wall.as_millis(),
        d1_read_child_wall.as_millis(),
        d1_dedup_wall.as_millis(),
        d1_write_child_wall.as_millis(),
        d1_leaf_emit_wall.as_millis(),
        d1_materialize_leaf_emitter.emit_calls(),
        d1_materialize_leaf_emitter.emit_ms(),
        d1_materialize_leaf_emitter.resident_batch_flushes(),
        d1_materialize_leaf_emitter.throttle_calls(),
        d1_materialize_leaf_emitter.throttle_ms(),
        d1_producer_leaf_backlog_soft_limit,
        d1_producer_help_drains,
        d1_producer_help_drain_ms,
        d1_backlog_full_help_drains,
        d1_backlog_full_help_ms,
        d1_producer_help_yields,
        d1_peak_leaf_backlog,
        d1_leaf_backlog_end,
        d1_active_leaf_drainers_end,
        d1_retry_recurse_wall.as_millis(),
        d1_written_child_runs,
    );

    if !native_next_clusters.is_empty() {
        let child_context = scan_context.for_depth(depth + 1);
        let native_recurse_start = Instant::now();
        let native_child_count = native_next_clusters.len();
        let mut native_cluster_sizes = native_next_clusters
            .iter()
            .map(|cluster| cluster.points.len())
            .collect::<Vec<_>>();
        native_cluster_sizes.sort_unstable();
        let native_cluster_percentile = |pct: usize| -> usize {
            if native_cluster_sizes.is_empty() {
                return 0;
            }
            let idx = native_cluster_sizes
                .len()
                .saturating_sub(1)
                .saturating_mul(pct)
                / 100;
            native_cluster_sizes[idx]
        };
        let native_cluster_max = native_cluster_sizes.last().copied().unwrap_or(0);
        let native_kernel_leaf_cap = params.kernel_safe_leaf_size();
        let native_oversized_children = native_cluster_sizes
            .iter()
            .filter(|&&size| size > native_kernel_leaf_cap)
            .count();
        let native_scheduler_telemetry_start = leaf_emitter.scheduler_telemetry();
        tracing::info!(
            "[adsampling/d1-native-subtree-recurse-start] source_depth={} replay_depth={} child_clusters={} selected_points={} child_p50={} child_p90={} child_p99={} child_max={} child_gt_kernel_leaf_cap={} kernel_leaf_cap={} external_child_runs=false leaf_drainer_limit={} leaf_drainer_backpressure_limit={}",
            depth,
            depth + 1,
            native_child_count,
            native_next_points,
            native_cluster_percentile(50),
            native_cluster_percentile(90),
            native_cluster_percentile(99),
            native_cluster_max,
            native_oversized_children,
            native_kernel_leaf_cap,
            d1_leaf_drainer_limit.unwrap_or(0),
            d1_leaf_drainer_backpressure_limit.unwrap_or(0),
        );
        let _d1_native_leaf_drainer_limit_guard = d1_leaf_drainer_limit
            .zip(d1_leaf_drainer_backpressure_limit)
            .and_then(|(limit, backpressure_limit)| {
                enter_leaf_drainer_limit_guard(leaf_emitter, limit, backpressure_limit)
            });
        let child_stats = std::thread::scope(|scope| -> AnnResult<PartitionStats> {
            let (progress_stop_tx, progress_stop_rx) = crossbeam_channel::bounded::<()>(1);
            let progress_start_profile = leaf_emitter.leaf_profile_snapshot();
            let progress_start_scheduler = leaf_emitter.scheduler_telemetry();
            let progress_handle = scope.spawn(move || {
                let mut previous_profile = progress_start_profile;
                let mut previous_scheduler = progress_start_scheduler;
                loop {
                    match progress_stop_rx.recv_timeout(Duration::from_secs(60)) {
                        Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                    }
                    let current_profile = leaf_emitter.leaf_profile_snapshot();
                    let current_scheduler = leaf_emitter.scheduler_telemetry();
                    let profile_delta = D1ResidentLeafProfileDelta::from_snapshots(
                        previous_profile.as_ref(),
                        current_profile.as_ref(),
                    );
                    let (
                        leaf_backlog,
                        active_leaf_drainers,
                        producer_help_drains_delta,
                        backlog_full_help_drains_delta,
                    ) = match (previous_scheduler, current_scheduler) {
                        (Some(previous), Some(current)) => (
                            current.leaf_backlog,
                            current.active_leaf_drainers,
                            current
                                .producer_help_drains
                                .saturating_sub(previous.producer_help_drains),
                            current
                                .backlog_full_help_drains
                                .saturating_sub(previous.backlog_full_help_drains),
                        ),
                        _ => (0, 0, 0, 0),
                    };
                    tracing::info!(
                        "[adsampling/d1-native-subtree-recurse-progress] source_depth={} replay_depth={} elapsed_ms={} profile_leaves_delta={} profile_points_delta={} ads_leaves_delta={} ads_rows_delta={} ads_scan_ms_delta={} ads_full_evals_delta={} ads_pruned_evals_delta={} ads_group_evals_delta={} flush_ms_delta={} leaf_backlog={} active_leaf_drainers={} producer_help_drains_delta={} backlog_full_help_drains_delta={}",
                        depth,
                        depth + 1,
                        native_recurse_start.elapsed().as_millis(),
                        profile_delta.leaves,
                        profile_delta.points,
                        profile_delta.ads_leaves,
                        profile_delta.ads_rows,
                        profile_delta.ads_scan.as_millis(),
                        profile_delta.ads_full_evals,
                        profile_delta.ads_pruned_evals,
                        profile_delta.ads_group_evals,
                        profile_delta.flush.as_millis(),
                        leaf_backlog,
                        active_leaf_drainers,
                        producer_help_drains_delta,
                        backlog_full_help_drains_delta,
                    );
                    previous_profile = current_profile;
                    previous_scheduler = current_scheduler;
                }
            });
            let child_stats = parallel_join_clusters(
                dataset,
                native_next_clusters,
                depth + 1,
                parent_n,
                metric,
                params,
                adaptive_c_max,
                min_recurse_size,
                pb,
                None,
                root_fanout_state,
                &child_context,
                leaf_emitter,
            );
            let _ = progress_stop_tx.send(());
            progress_handle.join().map_err(|_| {
                AnnError::log_index_error(
                    "D1 native subtree recurse progress thread panicked".to_string(),
                )
            })?;
            child_stats
        })?;
        let native_scheduler_telemetry_end = leaf_emitter.scheduler_telemetry();
        let (
            native_producer_help_drains,
            native_producer_help_drain_ms,
            native_backlog_full_help_drains,
            native_backlog_full_help_ms,
            native_producer_help_yields,
            native_peak_leaf_backlog_delta,
            native_leaf_backlog_end,
            native_active_leaf_drainers,
            native_leaf_drainer_limit_end,
            native_leaf_cap_large_assignment_hits,
            native_leaf_cap_producer_hits,
            native_leaf_cap_done_hits,
        ) = match (
            native_scheduler_telemetry_start,
            native_scheduler_telemetry_end,
        ) {
            (Some(start), Some(end)) => (
                end.producer_help_drains
                    .saturating_sub(start.producer_help_drains),
                end.producer_help_drain_ms
                    .saturating_sub(start.producer_help_drain_ms),
                end.backlog_full_help_drains
                    .saturating_sub(start.backlog_full_help_drains),
                end.backlog_full_help_ms
                    .saturating_sub(start.backlog_full_help_ms),
                end.producer_help_yields
                    .saturating_sub(start.producer_help_yields),
                end.peak_leaf_backlog
                    .saturating_sub(start.peak_leaf_backlog),
                end.leaf_backlog,
                end.active_leaf_drainers,
                end.leaf_drainer_limit,
                end.leaf_cap_large_assignment_hits
                    .saturating_sub(start.leaf_cap_large_assignment_hits),
                end.leaf_cap_producer_hits
                    .saturating_sub(start.leaf_cap_producer_hits),
                end.leaf_cap_done_hits
                    .saturating_sub(start.leaf_cap_done_hits),
            ),
            _ => (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        };
        tracing::info!(
            "[adsampling/d1-native-subtree-recurse] source_depth={} replay_depth={} child_clusters={} selected_points={} elapsed_ms={} producer_help_drains={} producer_help_drain_ms={} backlog_full_help_drains={} backlog_full_help_ms={} producer_help_yields={} peak_leaf_backlog_delta={} leaf_backlog_end={} active_leaf_drainers={} leaf_drainer_limit_end={} leaf_cap_large_assignment_hits={} leaf_cap_producer_hits={} leaf_cap_done_hits={}",
            depth,
            depth + 1,
            native_child_count,
            native_next_points,
            native_recurse_start.elapsed().as_millis(),
            native_producer_help_drains,
            native_producer_help_drain_ms,
            native_backlog_full_help_drains,
            native_backlog_full_help_ms,
            native_producer_help_yields,
            native_peak_leaf_backlog_delta,
            native_leaf_backlog_end,
            native_active_leaf_drainers,
            native_leaf_drainer_limit_end,
            native_leaf_cap_large_assignment_hits,
            native_leaf_cap_producer_hits,
            native_leaf_cap_done_hits,
        );
        stats.merge_from(child_stats);
    }

    if !next_depth_runs.is_empty() {
        if let Some(path) = params.d2_trace_dump.as_ref() {
            let base_dir = {
                let guard = external_run_store.lock();
                guard.base_dir.clone()
            };
            write_d2_trace_dump(
                path,
                &base_dir,
                io_cfg,
                depth,
                depth + 1,
                &next_depth_runs,
                parent_n,
                dataset.dim().saturating_mul(size_of::<f32>()),
            )?;
            return Err(AnnError::log_index_error(
                D2_TRACE_DUMP_COMPLETE.to_string(),
            ));
        }
        let child_context = scan_context.for_depth(depth + 1);
        tracing::info!(
            "[adsampling/d1-native-subtree-next-depth] source_depth={} replay_depth={} child_runs={} points={} mode=depth_wave_resident",
            depth,
            depth + 1,
            next_depth_runs.len(),
            next_depth_points,
        );
        let child_stats = parallel_join_child_runs_d1_level_scan_ads(
            dataset,
            next_depth_runs,
            depth + 1,
            parent_n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            pb,
            external_run_store,
            root_fanout_state,
            &child_context,
            leaf_emitter,
        )?;
        stats.merge_from(child_stats);
    }

    if !fallback_clusters.is_empty() {
        let fallback_external_run_store = if dataset.is_resident_subset() {
            None
        } else {
            Some(external_run_store)
        };
        let child_stats = parallel_join_clusters(
            dataset,
            fallback_clusters,
            depth,
            parent_n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            pb,
            fallback_external_run_store,
            root_fanout_state,
            &scan_context,
            leaf_emitter,
        )?;
        stats.merge_from(child_stats);
    }

    Ok(stats)
}

fn parallel_join_child_runs_depth_wave(
    dataset: &dyn PointStore,
    child_runs: Vec<ChildRun>,
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
    let expected_leaves =
        (child_runs.iter().map(|run| run.len).sum::<usize>() / adaptive_c_max.max(1)).max(4);
    let mut stats = PartitionStats::new(params.max_depth, expected_leaves);
    if metric != Metric::L2 {
        return parallel_join_child_runs_inline(
            dataset,
            child_runs,
            depth,
            parent_n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            pb,
            external_run_store,
            root_fanout_state,
            assignment_context,
            leaf_emitter,
        );
    }
    let schedule = schedule_depth_wave_child_runs(params, child_runs, depth);

    let wave_count = schedule.wave_runs.len();
    let exact_count = schedule.exact_runs.len();
    let wave_points = schedule.wave_runs.iter().map(|run| run.len).sum::<usize>();
    let exact_points = schedule.exact_runs.iter().map(|run| run.len).sum::<usize>();
    let max_wave_points = schedule
        .wave_runs
        .iter()
        .map(|run| run.len)
        .max()
        .unwrap_or(0);
    let max_exact_points = schedule
        .exact_runs
        .iter()
        .map(|run| run.len)
        .max()
        .unwrap_or(0);
    tracing::info!(
        "[adsampling/d1-sequential-ads-plan] depth={} scan_runs={} exact_runs={} scan_points={} exact_points={} max_scan_points={} max_exact_points={}",
        depth,
        wave_count,
        exact_count,
        wave_points,
        exact_points,
        max_wave_points,
        max_exact_points,
    );

    let wave_context = assignment_context
        .for_depth(depth)
        .with_depth_wave_assignment();

    if !schedule.wave_runs.is_empty() {
        let child_stats = parallel_join_child_runs_d1_level_scan_ads(
            dataset,
            schedule.wave_runs,
            depth,
            parent_n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            pb,
            external_run_store,
            root_fanout_state,
            &wave_context,
            leaf_emitter,
        )?;
        stats.merge_from(child_stats);
    }

    if !schedule.exact_runs.is_empty() {
        let child_stats = parallel_join_child_runs_inline(
            dataset,
            schedule.exact_runs,
            depth,
            parent_n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            pb,
            external_run_store,
            root_fanout_state,
            assignment_context,
            leaf_emitter,
        )?;
        stats.merge_from(child_stats);
    }

    Ok(stats)
}

fn should_use_small_run_wave(params: &ForgeANNParams, child_runs: &[ChildRun]) -> bool {
    params.io_planned_forgeann_enabled()
        && child_runs.len() > 1
        && child_runs
            .iter()
            .any(|run| run.len < ForgeANNParams::IO_PLAN_MIN_RESIDENT_POINTS)
}

pub(crate) fn parallel_join_child_runs_inline(
    dataset: &dyn PointStore,
    mut child_runs: Vec<ChildRun>,
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
    let expected_leaves =
        (child_runs.iter().map(|run| run.len).sum::<usize>() / adaptive_c_max.max(1)).max(4);
    let mut stats = PartitionStats::new(params.max_depth, expected_leaves);

    if child_runs.is_empty() {
        return Ok(stats);
    }

    if should_use_small_run_wave(params, &child_runs) {
        let (small_runs, large_runs): (Vec<_>, Vec<_>) = child_runs
            .into_iter()
            .partition(|run| run.len < ForgeANNParams::IO_PLAN_MIN_RESIDENT_POINTS);
        if !small_runs.is_empty() {
            let wave_points = small_runs.iter().map(|run| run.len).sum::<usize>();
            let (base_dir, io_cfg) = {
                let guard = external_run_store.lock();
                (guard.base_dir.clone(), guard.io_config())
            };
            let (small_points, batch_stats) =
                read_child_runs_batched_from_path(&base_dir, io_cfg, depth - 1, &small_runs)?;
            stats
                .telemetry
                .io_planned_forgeann
                .record_small_run_wave(small_runs.len(), wave_points);
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
            for (child_run, points) in small_runs.into_iter().zip(small_points) {
                debug_assert_eq!(points.len(), child_run.len);
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
            }
        }
        if !large_runs.is_empty() {
            let child_stats = parallel_join_child_runs_inline(
                dataset,
                large_runs,
                depth,
                parent_n,
                metric,
                params,
                adaptive_c_max,
                min_recurse_size,
                pb,
                external_run_store,
                root_fanout_state,
                assignment_context,
                leaf_emitter,
            )?;
            stats.merge_from(child_stats);
        }
        return Ok(stats);
    }

    if child_runs.len() == 1 {
        let child_run = child_runs.pop().unwrap();
        let child_stats = recurse_child_run_maybe_resident(
            dataset,
            child_run,
            depth,
            parent_n,
            metric,
            params,
            adaptive_c_max,
            min_recurse_size,
            pb,
            external_run_store,
            root_fanout_state,
            assignment_context,
            leaf_emitter,
        )?;
        stats.merge_from(child_stats);
        return Ok(stats);
    }

    child_runs.sort_unstable_by_key(|run| run.len);
    let split_at = child_runs.len() / 2;
    let right = child_runs.split_off(split_at);
    let left = child_runs;

    let (left_result, right_result) = rayon::join(
        || {
            parallel_join_child_runs_inline(
                dataset,
                left,
                depth,
                parent_n,
                metric,
                params,
                adaptive_c_max,
                min_recurse_size,
                pb,
                external_run_store,
                root_fanout_state,
                assignment_context,
                leaf_emitter,
            )
        },
        || {
            parallel_join_child_runs_inline(
                dataset,
                right,
                depth,
                parent_n,
                metric,
                params,
                adaptive_c_max,
                min_recurse_size,
                pb,
                external_run_store,
                root_fanout_state,
                assignment_context,
                leaf_emitter,
            )
        },
    );

    stats.merge_from(left_result?);
    stats.merge_from(right_result?);
    Ok(stats)
}
