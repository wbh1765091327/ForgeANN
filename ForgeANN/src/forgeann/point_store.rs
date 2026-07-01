use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::io::Write;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::os::unix::io::RawFd;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use io_uring::{IoUring, opcode, types};
use memmap2::{Advice, Mmap, MmapOptions};
use tracing::{info, warn};

use super::direct_io::{DirectIoConfig, DirectIoFile};
use super::io_runtime::BoundedReadPlan;
use crate::common::{AlignedBoxWithSlice, AnnError, AnnResult, Metric};
use crate::model::InmemDataset;
use crate::utils::load_metadata_from_file;

const FBIN_HEADER_BYTES: u64 = 8;
const STREAM_POINTS_PER_CHUNK: usize = 1024;
const URING_WINDOW_GATHER_MIN_WINDOWS: usize = 2;
const URING_WINDOW_GATHER_QUEUE_DEPTH: usize = 32;
const MMAP_KEEP_PAGE_CACHE_ENV: &str = "FORGEANN_OOM_MMAP_KEEP_PAGE_CACHE";

thread_local! {
    static URING_GATHER_RING: RefCell<Option<IoUring>> = const { RefCell::new(None) };
}

#[cfg(test)]
static URING_GATHER_TEST_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

static URING_GATHER_BATCHES: AtomicU64 = AtomicU64::new(0);
static URING_GATHER_WINDOWS: AtomicU64 = AtomicU64::new(0);
static URING_GATHER_BYTES: AtomicU64 = AtomicU64::new(0);
static URING_GATHER_FALLBACKS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn mmap_keep_page_cache_enabled() -> bool {
    std::env::var(MMAP_KEEP_PAGE_CACHE_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(test)]
pub(crate) fn reset_uring_gather_test_counters() {
    URING_GATHER_TEST_ATTEMPTS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn uring_gather_test_attempts() -> usize {
    URING_GATHER_TEST_ATTEMPTS.load(Ordering::Relaxed)
}

fn record_uring_gather_success(windows: usize, bytes: usize) {
    URING_GATHER_BATCHES.fetch_add(1, Ordering::Relaxed);
    URING_GATHER_WINDOWS.fetch_add(windows as u64, Ordering::Relaxed);
    URING_GATHER_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn record_uring_gather_fallback() {
    URING_GATHER_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn uring_gather_counters() -> (u64, u64, u64, u64) {
    (
        URING_GATHER_BATCHES.load(Ordering::Relaxed),
        URING_GATHER_WINDOWS.load(Ordering::Relaxed),
        URING_GATHER_BYTES.load(Ordering::Relaxed),
        URING_GATHER_FALLBACKS.load(Ordering::Relaxed),
    )
}

/// I/O telemetry for batched point reads.
#[derive(Default, Clone, Debug)]
pub struct PointBatchStats {
    /// Individual `read_point_into` calls that could not be coalesced into a range.
    pub point_calls: u64,
    /// Number of coalesced `read_range_into` calls.
    pub range_calls: u64,
    /// Total rows read via range calls.
    pub range_rows_read: u64,
    /// Approximate bytes read.
    pub bytes_read: u64,
}

impl PointBatchStats {
    pub fn range_hit_ratio(&self) -> f64 {
        let total = self.range_rows_read + self.point_calls;
        if total == 0 {
            return 0.0;
        }
        self.range_rows_read as f64 / total as f64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowedGatherOptions {
    pub max_gap_rows: u32,
    pub max_window_bytes: usize,
    pub alignment_bytes: usize,
    pub sort_ids: bool,
}

impl Default for WindowedGatherOptions {
    fn default() -> Self {
        Self {
            max_gap_rows: 0,
            max_window_bytes: 256 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct WindowedGatherStats {
    pub rows_requested: u64,
    pub windows_submitted: u64,
    pub rows_per_window_sum: u64,
    pub singleton_windows: u64,
    pub range_windows: u64,
    pub range_rows_read: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub alignment_waste_bytes: u64,
    pub overread_rows: u64,
    pub scatter_ops: u64,
}

impl WindowedGatherStats {
    pub fn rows_per_window_avg(&self) -> f64 {
        if self.windows_submitted == 0 {
            0.0
        } else {
            self.rows_per_window_sum as f64 / self.windows_submitted as f64
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.rows_requested += other.rows_requested;
        self.windows_submitted += other.windows_submitted;
        self.rows_per_window_sum += other.rows_per_window_sum;
        self.singleton_windows += other.singleton_windows;
        self.range_windows += other.range_windows;
        self.range_rows_read += other.range_rows_read;
        self.logical_bytes += other.logical_bytes;
        self.physical_bytes += other.physical_bytes;
        self.alignment_waste_bytes += other.alignment_waste_bytes;
        self.overread_rows += other.overread_rows;
        self.scatter_ops += other.scatter_ops;
    }

    pub fn apply_to_point_batch_stats(&self, stats: &mut PointBatchStats) {
        stats.point_calls += self.singleton_windows;
        stats.range_calls += self.range_windows;
        stats.range_rows_read += self.range_rows_read;
        stats.bytes_read += self.physical_bytes;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GatherRow {
    row_id: u32,
    original_pos: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadWindow {
    pub(crate) start_row: u32,
    pub(crate) row_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScatterOp {
    pub(crate) original_pos: usize,
    pub(crate) row_offset_in_window: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GatherPlan {
    pub(crate) windows: Vec<ReadWindow>,
    pub(crate) scatter_by_window: Vec<Vec<ScatterOp>>,
}

pub(crate) fn build_gather_plan(
    ids: &[u32],
    row_bytes: usize,
    options: WindowedGatherOptions,
    stats: &mut WindowedGatherStats,
) -> GatherPlan {
    stats.rows_requested += ids.len() as u64;
    stats.logical_bytes += (ids.len() * row_bytes) as u64;
    if ids.is_empty() {
        return GatherPlan {
            windows: Vec::new(),
            scatter_by_window: Vec::new(),
        };
    }

    let mut rows: Vec<GatherRow> = ids
        .iter()
        .enumerate()
        .map(|(original_pos, &row_id)| GatherRow {
            row_id,
            original_pos,
        })
        .collect();
    if options.sort_ids {
        rows.sort_unstable_by_key(|row| row.row_id);
    }

    let mut windows = Vec::new();
    let mut scatter_by_window = Vec::new();

    let max_window_rows = if row_bytes == 0 {
        ids.len().max(1)
    } else {
        (options.max_window_bytes / row_bytes).max(1)
    };
    let mut sorted_start = 0usize;
    while sorted_start < rows.len() {
        let mut sorted_end = sorted_start + 1;
        let mut prev_row = rows[sorted_start].row_id;
        while sorted_end < rows.len() {
            let next_row = rows[sorted_end].row_id;
            let gap = next_row.saturating_sub(prev_row).saturating_sub(1);
            let proposed_row_count =
                next_row.saturating_sub(rows[sorted_start].row_id) as usize + 1;
            if gap > options.max_gap_rows || proposed_row_count > max_window_rows {
                break;
            }
            prev_row = next_row;
            sorted_end += 1;
        }

        let start_row = rows[sorted_start].row_id;
        let end_row = rows[sorted_end - 1].row_id;
        let row_count = end_row - start_row + 1;
        let requested_rows = sorted_end - sorted_start;
        let mut distinct_requested_rows = 1usize;
        let mut prev_row = rows[sorted_start].row_id;
        for row in &rows[sorted_start + 1..sorted_end] {
            if row.row_id != prev_row {
                distinct_requested_rows += 1;
                prev_row = row.row_id;
            }
        }
        windows.push(ReadWindow {
            start_row,
            row_count,
        });
        stats.windows_submitted += 1;
        stats.rows_per_window_sum += row_count as u64;
        if row_count <= 1 {
            stats.singleton_windows += 1;
        } else {
            stats.range_windows += 1;
            stats.range_rows_read += row_count as u64;
        }
        stats.overread_rows += row_count.saturating_sub(distinct_requested_rows as u32) as u64;

        let logical_bytes = row_count as usize * row_bytes;
        let alignment = options.alignment_bytes.max(1);
        let physical_bytes = if options.sort_ids {
            logical_bytes.div_ceil(alignment) * alignment
        } else {
            logical_bytes
        };
        stats.physical_bytes += physical_bytes as u64;
        stats.alignment_waste_bytes += physical_bytes.saturating_sub(logical_bytes) as u64;

        let mut window_scatter = Vec::with_capacity(requested_rows);
        for row in &rows[sorted_start..sorted_end] {
            window_scatter.push(ScatterOp {
                original_pos: row.original_pos,
                row_offset_in_window: row.row_id - start_row,
            });
            stats.scatter_ops += 1;
        }
        scatter_by_window.push(window_scatter);

        sorted_start = sorted_end;
    }

    GatherPlan {
        windows,
        scatter_by_window,
    }
}

pub(crate) fn scatter_window_rows(
    plan: &GatherPlan,
    window_idx: usize,
    row_width: usize,
    window_slice: &[f32],
    out: &mut [f32],
) {
    for scatter in &plan.scatter_by_window[window_idx] {
        let src_row = scatter.row_offset_in_window as usize;
        let src_start = src_row * row_width;
        let src_end = src_start + row_width;
        let dst_start = scatter.original_pos * row_width;
        let dst_end = dst_start + row_width;
        out[dst_start..dst_end].copy_from_slice(&window_slice[src_start..src_end]);
    }
}

struct UringWindowRead {
    window_idx: usize,
    prefix: usize,
    logical_len: usize,
    scratch: AlignedBoxWithSlice<u8>,
}

fn try_read_windows_with_uring(
    fd: RawFd,
    plan: &GatherPlan,
    row_width: usize,
    row_bytes: usize,
    out: &mut [f32],
    alignment: usize,
) -> io::Result<bool> {
    if plan.windows.len() < URING_WINDOW_GATHER_MIN_WINDOWS {
        return Ok(false);
    }

    #[cfg(test)]
    URING_GATHER_TEST_ATTEMPTS.fetch_add(1, Ordering::Relaxed);

    let alignment = alignment.max(1);
    let qd = URING_WINDOW_GATHER_QUEUE_DEPTH
        .min(plan.windows.len())
        .max(1);
    URING_GATHER_RING.with(|ring_cell| -> io::Result<bool> {
        let mut ring_ref = ring_cell.borrow_mut();
        if ring_ref.is_none() {
            *ring_ref = Some(IoUring::new(URING_WINDOW_GATHER_QUEUE_DEPTH as u32)?);
        }
        let ring = ring_ref
            .as_mut()
            .expect("thread-local io_uring must be initialized");

        let mut next_window = 0usize;
        while next_window < plan.windows.len() {
            let batch_end = (next_window + qd).min(plan.windows.len());
            let mut reads = Vec::with_capacity(batch_end - next_window);
            for window_idx in next_window..batch_end {
                let window = plan.windows[window_idx];
                let logical_offset =
                    FBIN_HEADER_BYTES + (window.start_row as u64).saturating_mul(row_bytes as u64);
                let logical_len = window.row_count as usize * row_bytes;
                let alignment_u64 = alignment as u64;
                let aligned_offset = logical_offset / alignment_u64 * alignment_u64;
                let prefix = (logical_offset - aligned_offset) as usize;
                let aligned_len =
                    prefix.saturating_add(logical_len).div_ceil(alignment) * alignment;
                let scratch = AlignedBoxWithSlice::<u8>::new(aligned_len, alignment)
                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;

                reads.push(UringWindowRead {
                    window_idx,
                    prefix,
                    logical_len,
                    scratch,
                });

                let entry = opcode::Read::new(
                    types::Fd(fd),
                    reads
                        .last_mut()
                        .unwrap()
                        .scratch
                        .as_mut_slice()
                        .as_mut_ptr(),
                    aligned_len as u32,
                )
                .offset(aligned_offset)
                .build()
                .user_data((reads.len() - 1) as u64);

                unsafe {
                    ring.submission().push(&entry).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::WouldBlock,
                            "io_uring submission queue is full",
                        )
                    })?;
                }
            }

            ring.submit_and_wait(reads.len())?;
            let mut completed = 0usize;
            while completed < reads.len() {
                if let Some(cqe) = ring.completion().next() {
                    let slot = cqe.user_data() as usize;
                    let result = cqe.result();
                    if slot >= reads.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("io_uring returned invalid window slot {slot}"),
                        ));
                    }
                    if result < 0 {
                        return Err(io::Error::from_raw_os_error(-result));
                    }
                    let got = result as usize;
                    if got < reads[slot].prefix + reads[slot].logical_len {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!(
                                "io_uring short read: got {got}, expected at least {}",
                                reads[slot].prefix + reads[slot].logical_len
                            ),
                        ));
                    }
                    completed += 1;
                } else {
                    ring.submit_and_wait(1)?;
                }
            }

            for read in &reads {
                let logical = &read.scratch.as_slice()[read.prefix..read.prefix + read.logical_len];
                let window_floats = unsafe {
                    std::slice::from_raw_parts(
                        logical.as_ptr() as *const f32,
                        read.logical_len / std::mem::size_of::<f32>(),
                    )
                };
                scatter_window_rows(plan, read.window_idx, row_width, window_floats, out);
            }

            next_window = batch_end;
        }

        Ok(true)
    })
}

fn scatter_bounded_window_rows(
    plan: &BoundedReadPlan,
    scatter_ranges: &[(usize, usize)],
    window_idx: usize,
    row_width: usize,
    window_slice: &[f32],
    out: &mut [f32],
) {
    let (scatter_start, scatter_end) = scatter_ranges[window_idx];
    for scatter in &plan.scatter[scatter_start..scatter_end] {
        let src_row = scatter.row_offset_in_window as usize;
        let src_start = src_row * row_width;
        let src_end = src_start + row_width;
        let dst_start = scatter.original_pos * row_width;
        let dst_end = dst_start + row_width;
        out[dst_start..dst_end].copy_from_slice(&window_slice[src_start..src_end]);
    }
}

fn bounded_scatter_ranges(plan: &BoundedReadPlan) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(plan.windows.len());
    let mut cursor = 0usize;
    for window_idx in 0..plan.windows.len() {
        let start = cursor;
        while cursor < plan.scatter.len() && plan.scatter[cursor].window_idx == window_idx {
            cursor += 1;
        }
        ranges.push((start, cursor));
    }
    ranges
}

fn try_read_bounded_windows_with_uring(
    fd: RawFd,
    plan: &BoundedReadPlan,
    row_width: usize,
    row_bytes: usize,
    out: &mut [f32],
    alignment: usize,
) -> io::Result<bool> {
    if plan.windows.len() < URING_WINDOW_GATHER_MIN_WINDOWS {
        return Ok(false);
    }

    #[cfg(test)]
    URING_GATHER_TEST_ATTEMPTS.fetch_add(1, Ordering::Relaxed);

    let scatter_ranges = bounded_scatter_ranges(plan);
    let alignment = alignment.max(1);
    let qd = URING_WINDOW_GATHER_QUEUE_DEPTH
        .min(plan.windows.len())
        .max(1);
    URING_GATHER_RING.with(|ring_cell| -> io::Result<bool> {
        let mut ring_ref = ring_cell.borrow_mut();
        if ring_ref.is_none() {
            *ring_ref = Some(IoUring::new(URING_WINDOW_GATHER_QUEUE_DEPTH as u32)?);
        }
        let ring = ring_ref
            .as_mut()
            .expect("thread-local io_uring must be initialized");

        let mut next_window = 0usize;
        while next_window < plan.windows.len() {
            let batch_end = (next_window + qd).min(plan.windows.len());
            let mut reads = Vec::with_capacity(batch_end - next_window);
            for window_idx in next_window..batch_end {
                let window = plan.windows[window_idx];
                let logical_offset =
                    FBIN_HEADER_BYTES + (window.start_row as u64).saturating_mul(row_bytes as u64);
                let logical_len = window.row_count as usize * row_bytes;
                let alignment_u64 = alignment as u64;
                let aligned_offset = logical_offset / alignment_u64 * alignment_u64;
                let prefix = (logical_offset - aligned_offset) as usize;
                let aligned_len =
                    prefix.saturating_add(logical_len).div_ceil(alignment) * alignment;
                let scratch = AlignedBoxWithSlice::<u8>::new(aligned_len, alignment)
                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;

                reads.push(UringWindowRead {
                    window_idx,
                    prefix,
                    logical_len,
                    scratch,
                });

                let entry = opcode::Read::new(
                    types::Fd(fd),
                    reads
                        .last_mut()
                        .unwrap()
                        .scratch
                        .as_mut_slice()
                        .as_mut_ptr(),
                    aligned_len as u32,
                )
                .offset(aligned_offset)
                .build()
                .user_data((reads.len() - 1) as u64);

                unsafe {
                    ring.submission().push(&entry).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::WouldBlock,
                            "io_uring submission queue is full",
                        )
                    })?;
                }
            }

            ring.submit_and_wait(reads.len())?;
            let mut completed = 0usize;
            while completed < reads.len() {
                if let Some(cqe) = ring.completion().next() {
                    let slot = cqe.user_data() as usize;
                    let result = cqe.result();
                    if slot >= reads.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("io_uring returned invalid bounded window slot {slot}"),
                        ));
                    }
                    if result < 0 {
                        return Err(io::Error::from_raw_os_error(-result));
                    }
                    let got = result as usize;
                    if got < reads[slot].prefix + reads[slot].logical_len {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!(
                                "io_uring bounded short read: got {got}, expected at least {}",
                                reads[slot].prefix + reads[slot].logical_len
                            ),
                        ));
                    }
                    completed += 1;
                } else {
                    ring.submit_and_wait(1)?;
                }
            }

            for read in &reads {
                let logical = &read.scratch.as_slice()[read.prefix..read.prefix + read.logical_len];
                let window_floats = unsafe {
                    std::slice::from_raw_parts(
                        logical.as_ptr() as *const f32,
                        read.logical_len / std::mem::size_of::<f32>(),
                    )
                };
                scatter_bounded_window_rows(
                    plan,
                    &scatter_ranges,
                    read.window_idx,
                    row_width,
                    window_floats,
                    out,
                );
            }

            next_window = batch_end;
        }

        Ok(true)
    })
}

pub trait PointStore: Send + Sync {
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()>;
    fn read_range_into(&self, start_pid: u32, count: usize, out: &mut [f32]) -> AnnResult<()>;
    fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()>;

    fn point_id_capacity(&self) -> usize {
        self.len()
    }

    fn read_points(&self, ids: &[u32]) -> AnnResult<Vec<f32>> {
        let mut out = vec![0.0f32; ids.len().saturating_mul(self.dim())];
        self.read_points_into(ids, &mut out)?;
        Ok(out)
    }

    /// Batched read with I/O telemetry.
    ///
    /// Default implementation delegates to `read_points_into` and counts every
    /// ID as an individual point call. `DirectPointStore` overrides this to
    /// expose the actual run-length coalescing statistics.
    fn read_points_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        if !ids.is_empty() {
            stats.point_calls += ids.len() as u64;
            stats.bytes_read += (ids.len() * self.dim() * 4) as u64;
        }
        self.read_points_into(ids, out)
    }

    fn read_points_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        _options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        if !ids.is_empty() {
            stats.rows_requested += ids.len() as u64;
            let row_bytes = self.dim() * std::mem::size_of::<f32>();
            stats.logical_bytes += (ids.len() * row_bytes) as u64;
            stats.physical_bytes += (ids.len() * row_bytes) as u64;
            stats.scatter_ops += ids.len() as u64;
        }
        self.read_points_into(ids, out)
    }

    fn read_points_windowed_into_batch_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        let mut windowed = WindowedGatherStats::default();
        self.read_points_windowed_into_stats(ids, out, options, &mut windowed)?;
        windowed.apply_to_point_batch_stats(stats);
        Ok(())
    }

    #[allow(private_interfaces)]
    fn read_bounded_plan_into_batch_stats(
        &self,
        _plan: &BoundedReadPlan,
        _out: &mut [f32],
        _stats: &mut PointBatchStats,
    ) -> AnnResult<bool> {
        Ok(false)
    }

    fn is_vector_run(&self) -> bool {
        false
    }

    fn is_resident_subset(&self) -> bool {
        false
    }

    fn resident_rows(&self) -> Option<(&[u32], &[f32])> {
        None
    }

    fn prefers_coalesced_window_reads(&self) -> bool {
        false
    }

    fn disables_prefetch_pipeline(&self) -> bool {
        false
    }

    fn source_identity(&self) -> Option<String> {
        None
    }

    fn discard_cached_pages(&self) -> AnnResult<()> {
        Ok(())
    }

    fn get_distance(&self, lhs: u32, rhs: u32, metric: Metric) -> AnnResult<f32> {
        match metric {
            Metric::L2 | Metric::Cosine | Metric::Ip => {
                let mut a = vec![0.0f32; self.dim()];
                let mut b = vec![0.0f32; self.dim()];
                self.read_point_into(lhs, &mut a)?;
                self.read_point_into(rhs, &mut b)?;
                Ok(l2_distance_sq(&a, &b))
            }
        }
    }

    fn calculate_medoid_point_id_with_threads(&self, _num_threads: Option<u32>) -> AnnResult<u32> {
        if self.len() == 0 {
            return Err(AnnError::log_index_error(
                "Cannot compute medoid of empty dataset".to_string(),
            ));
        }

        let dim = self.dim();
        let mut center = vec![0.0f32; dim];
        let mut chunk = vec![0.0f32; STREAM_POINTS_PER_CHUNK.saturating_mul(dim)];
        let mut processed = 0usize;
        while processed < self.len() {
            let chunk_points = (self.len() - processed).min(STREAM_POINTS_PER_CHUNK);
            self.read_range_into(
                processed as u32,
                chunk_points,
                &mut chunk[..chunk_points * dim],
            )?;
            for row in 0..chunk_points {
                let start = row * dim;
                let point = &chunk[start..start + dim];
                for (dst, &value) in center.iter_mut().zip(point.iter()) {
                    *dst += value;
                }
            }
            processed += chunk_points;
        }

        let denom = self.len() as f32;
        for value in &mut center {
            *value /= denom;
        }

        let mut best_id = 0u32;
        let mut best_dist = f32::INFINITY;
        processed = 0;
        while processed < self.len() {
            let chunk_points = (self.len() - processed).min(STREAM_POINTS_PER_CHUNK);
            self.read_range_into(
                processed as u32,
                chunk_points,
                &mut chunk[..chunk_points * dim],
            )?;
            for row in 0..chunk_points {
                let pid = (processed + row) as u32;
                let start = row * dim;
                let point = &chunk[start..start + dim];
                let dist = l2_distance(point, &center);
                if (dist, pid) < (best_dist, best_id) {
                    best_dist = dist;
                    best_id = pid;
                }
            }
            processed += chunk_points;
        }
        Ok(best_id)
    }
}

fn file_source_identity(path: &Path) -> Option<String> {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let metadata = std::fs::metadata(&canonical).ok()?;
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    Some(format!(
        "file:{}:bytes={}:mtime_ns={modified_ns}",
        canonical.display(),
        metadata.len(),
    ))
}

#[derive(Debug)]
pub struct DirectPointStore {
    path: PathBuf,
    file: DirectIoFile,
    num_points: usize,
    dim: usize,
    row_bytes: usize,
}

impl DirectPointStore {
    pub fn open(path: &Path) -> AnnResult<Self> {
        Self::open_with_config(path, DirectIoConfig::disabled())
    }

    pub fn open_with_config(path: &Path, io_cfg: DirectIoConfig) -> AnnResult<Self> {
        let (num_points, dim) = load_metadata_from_file(path)?;
        let file = DirectIoFile::open_read(path, io_cfg)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            num_points,
            dim,
            row_bytes: dim.saturating_mul(std::mem::size_of::<f32>()),
        })
    }

    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[inline]
    pub fn io_config(&self) -> DirectIoConfig {
        self.file.config()
    }

    fn point_offset(&self, pid: u32) -> AnnResult<u64> {
        let idx = pid as usize;
        if idx >= self.num_points {
            return Err(AnnError::log_index_error(format!(
                "Point id {} out of bounds for {} points",
                pid, self.num_points
            )));
        }
        Ok(FBIN_HEADER_BYTES + (idx as u64).saturating_mul(self.row_bytes as u64))
    }

    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> AnnResult<()> {
        while !buf.is_empty() {
            let chunk = buf.len();
            self.file.read_exact_at(buf, offset)?;
            let (_, rest) = buf.split_at_mut(chunk);
            buf = rest;
            offset += chunk as u64;
        }
        Ok(())
    }

    #[inline]
    fn f32_bytes_mut(slice: &mut [f32]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(slice),
            )
        }
    }
}

impl PointStore for DirectPointStore {
    fn len(&self) -> usize {
        self.num_points
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
        if out.len() != self.dim {
            return Err(AnnError::log_index_error(format!(
                "Point buffer width mismatch: got {} expected {}",
                out.len(),
                self.dim
            )));
        }
        let offset = self.point_offset(pid)?;
        self.read_exact_at(Self::f32_bytes_mut(out), offset)
    }

    fn read_range_into(&self, start_pid: u32, count: usize, out: &mut [f32]) -> AnnResult<()> {
        if count == 0 {
            return Ok(());
        }
        if out.len() != count.saturating_mul(self.dim) {
            return Err(AnnError::log_index_error(format!(
                "Range buffer size mismatch: got {} expected {}",
                out.len(),
                count.saturating_mul(self.dim)
            )));
        }
        let start = start_pid as usize;
        let end = start.saturating_add(count);
        if end > self.num_points {
            return Err(AnnError::log_index_error(format!(
                "Range [{}..{}) out of bounds for {} points",
                start, end, self.num_points
            )));
        }
        let offset = FBIN_HEADER_BYTES + (start as u64).saturating_mul(self.row_bytes as u64);
        self.read_exact_at(Self::f32_bytes_mut(out), offset)
    }

    fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(self.dim) {
            return Err(AnnError::log_index_error(format!(
                "Point matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(self.dim)
            )));
        }
        let mut run_start = 0usize;
        while run_start < ids.len() {
            let mut run_end = run_start + 1;
            while run_end < ids.len() && ids[run_end] == ids[run_end - 1].saturating_add(1) {
                run_end += 1;
            }
            let out_start = run_start * self.dim;
            let out_end = run_end * self.dim;
            self.read_range_into(
                ids[run_start],
                run_end - run_start,
                &mut out[out_start..out_end],
            )?;
            run_start = run_end;
        }
        Ok(())
    }

    fn read_points_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(self.dim) {
            return Err(AnnError::log_index_error(format!(
                "Point matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(self.dim)
            )));
        }
        stats.bytes_read += (ids.len() * self.dim * 4) as u64;
        let mut run_start = 0usize;
        while run_start < ids.len() {
            let mut run_end = run_start + 1;
            while run_end < ids.len() && ids[run_end] == ids[run_end - 1].saturating_add(1) {
                run_end += 1;
            }
            let run_len = run_end - run_start;
            let out_start = run_start * self.dim;
            let out_end = run_end * self.dim;
            self.read_range_into(ids[run_start], run_len, &mut out[out_start..out_end])?;
            if run_len > 1 {
                stats.range_calls += 1;
                stats.range_rows_read += run_len as u64;
            } else {
                stats.point_calls += 1;
            }
            run_start = run_end;
        }
        Ok(())
    }

    fn read_points_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(self.dim) {
            return Err(AnnError::log_index_error(format!(
                "Point matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(self.dim)
            )));
        }

        let row_bytes = self.row_bytes;
        let plan = build_gather_plan(ids, row_bytes, *options, stats);
        if self.io_config().enabled {
            match try_read_windows_with_uring(
                self.file.raw_fd(),
                &plan,
                self.dim,
                row_bytes,
                out,
                self.io_config().effective_alignment(),
            ) {
                Ok(true) => {
                    record_uring_gather_success(plan.windows.len(), stats.physical_bytes as usize);
                    return Ok(());
                }
                Ok(false) => {}
                Err(_) => record_uring_gather_fallback(),
            }
        }
        let mut window_buf = vec![
            0.0f32;
            plan.windows
                .iter()
                .map(|window| window.row_count as usize)
                .max()
                .unwrap_or(0)
                * self.dim
        ];
        for (window_idx, window) in plan.windows.iter().enumerate() {
            let rows = window.row_count as usize;
            let elems = rows * self.dim;
            let window_slice = &mut window_buf[..elems];
            self.read_range_into(window.start_row, rows, window_slice)?;
            scatter_window_rows(&plan, window_idx, self.dim, window_slice, out);
        }
        Ok(())
    }

    #[allow(private_interfaces)]
    fn read_bounded_plan_into_batch_stats(
        &self,
        plan: &BoundedReadPlan,
        out: &mut [f32],
        stats: &mut PointBatchStats,
    ) -> AnnResult<bool> {
        if out.is_empty() || plan.windows.is_empty() {
            return Ok(false);
        }
        if out.len() % self.dim != 0 {
            return Err(AnnError::log_index_error(format!(
                "Bounded point matrix size {} is not divisible by dim {}",
                out.len(),
                self.dim
            )));
        }
        if !self.io_config().enabled {
            return Ok(false);
        }
        match try_read_bounded_windows_with_uring(
            self.file.raw_fd(),
            plan,
            self.dim,
            self.row_bytes,
            out,
            self.io_config().effective_alignment(),
        ) {
            Ok(true) => {
                record_uring_gather_success(plan.windows.len(), plan.physical_bytes() as usize);
                for window in &plan.windows {
                    if window.row_count > 1 {
                        stats.range_calls += 1;
                        stats.range_rows_read += window.row_count as u64;
                    } else {
                        stats.point_calls += 1;
                    }
                }
                stats.bytes_read += plan.physical_bytes();
                Ok(true)
            }
            Ok(false) => Ok(false),
            Err(err) => {
                record_uring_gather_fallback();
                Err(AnnError::log_io_error(err))
            }
        }
    }

    fn prefers_coalesced_window_reads(&self) -> bool {
        self.io_config().enabled
    }

    fn source_identity(&self) -> Option<String> {
        file_source_identity(&self.path)
    }
}

#[derive(Debug)]
pub struct MmapPointStore {
    path: PathBuf,
    _file: File,
    mmap: Mmap,
    num_points: usize,
    dim: usize,
    row_bytes: usize,
    keep_page_cache: bool,
}

impl MmapPointStore {
    pub fn open(path: &Path) -> AnnResult<Self> {
        let (num_points, dim) = load_metadata_from_file(path)?;
        let row_bytes = dim.saturating_mul(std::mem::size_of::<f32>());
        let file = File::open(path)?;
        let actual_len = file.metadata()?.len() as u128;
        let expected_len = (FBIN_HEADER_BYTES as u128)
            .saturating_add((num_points as u128).saturating_mul(row_bytes as u128));
        if actual_len < expected_len {
            return Err(AnnError::log_index_error(format!(
                "Mmap point store file {:?} is too small: got {} bytes expected at least {} bytes",
                path, actual_len, expected_len
            )));
        }
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let keep_page_cache = mmap_keep_page_cache_enabled();
        if keep_page_cache {
            info!(
                path = ?path,
                env = MMAP_KEEP_PAGE_CACHE_ENV,
                "mmap point store preserving page cache; MADV_DONTNEED and MADV_RANDOM disabled"
            );
        } else if let Err(err) = mmap.advise(Advice::Random) {
            warn!(
                path = ?path,
                error = ?err,
                "failed to apply MADV_RANDOM to mmap point store; continuing without mmap advice"
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            _file: file,
            mmap,
            num_points,
            dim,
            row_bytes,
            keep_page_cache,
        })
    }

    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn row_byte_range(&self, start_pid: u32, count: usize) -> AnnResult<Range<usize>> {
        let start = start_pid as usize;
        let end = start.saturating_add(count);
        if end > self.num_points {
            return Err(AnnError::log_index_error(format!(
                "Range [{}..{}) out of bounds for {} points",
                start, end, self.num_points
            )));
        }
        let byte_start = (FBIN_HEADER_BYTES as usize)
            .checked_add(start.checked_mul(self.row_bytes).ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "Mmap point store byte offset overflow for pid {}",
                    start_pid
                ))
            })?)
            .ok_or_else(|| {
                AnnError::log_index_error(format!(
                    "Mmap point store byte offset overflow for pid {}",
                    start_pid
                ))
            })?;
        let byte_len = count.checked_mul(self.row_bytes).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Mmap point store byte length overflow for row count {}",
                count
            ))
        })?;
        let byte_end = byte_start.checked_add(byte_len).ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Mmap point store byte range overflow for pid {} count {}",
                start_pid, count
            ))
        })?;
        if byte_end > self.mmap.len() {
            return Err(AnnError::log_index_error(format!(
                "Mmap point store byte range [{}..{}) exceeds mapped length {}",
                byte_start,
                byte_end,
                self.mmap.len()
            )));
        }
        Ok(byte_start..byte_end)
    }

    fn mapped_rows_as_f32(&self, range: &Range<usize>) -> AnnResult<&[f32]> {
        bytemuck::try_cast_slice::<u8, f32>(&self.mmap[range.clone()]).map_err(|err| {
            AnnError::log_index_error(format!("Mmap point store row alignment error: {err:?}"))
        })
    }

    fn discard_mapped_range(&self, range: &Range<usize>) {
        if self.keep_page_cache {
            return;
        }
        let len = range.end.saturating_sub(range.start);
        if len == 0 {
            return;
        }
        if let Err(err) = self.mmap.advise_range(Advice::DontNeed, range.start, len) {
            warn!(
                path = ?self.path,
                start = range.start,
                len,
                error = ?err,
                "failed to discard mmap point-store range after copy"
            );
        }
    }
}

impl PointStore for MmapPointStore {
    fn len(&self) -> usize {
        self.num_points
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
        if out.len() != self.dim {
            return Err(AnnError::log_index_error(format!(
                "Point buffer width mismatch: got {} expected {}",
                out.len(),
                self.dim
            )));
        }
        let range = self.row_byte_range(pid, 1)?;
        out.copy_from_slice(self.mapped_rows_as_f32(&range)?);
        Ok(())
    }

    fn read_range_into(&self, start_pid: u32, count: usize, out: &mut [f32]) -> AnnResult<()> {
        if count == 0 {
            return Ok(());
        }
        if out.len() != count.saturating_mul(self.dim) {
            return Err(AnnError::log_index_error(format!(
                "Range buffer size mismatch: got {} expected {}",
                out.len(),
                count.saturating_mul(self.dim)
            )));
        }
        let range = self.row_byte_range(start_pid, count)?;
        out.copy_from_slice(self.mapped_rows_as_f32(&range)?);
        self.discard_mapped_range(&range);
        Ok(())
    }

    fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(self.dim) {
            return Err(AnnError::log_index_error(format!(
                "Point matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(self.dim)
            )));
        }
        let mut run_start = 0usize;
        while run_start < ids.len() {
            let mut run_end = run_start + 1;
            while run_end < ids.len() && ids[run_end] == ids[run_end - 1].saturating_add(1) {
                run_end += 1;
            }
            let out_start = run_start * self.dim;
            let out_end = run_end * self.dim;
            self.read_range_into(
                ids[run_start],
                run_end - run_start,
                &mut out[out_start..out_end],
            )?;
            run_start = run_end;
        }
        Ok(())
    }

    fn read_points_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(self.dim) {
            return Err(AnnError::log_index_error(format!(
                "Point matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(self.dim)
            )));
        }
        stats.bytes_read += (ids.len() * self.dim * std::mem::size_of::<f32>()) as u64;
        let mut run_start = 0usize;
        while run_start < ids.len() {
            let mut run_end = run_start + 1;
            while run_end < ids.len() && ids[run_end] == ids[run_end - 1].saturating_add(1) {
                run_end += 1;
            }
            let run_len = run_end - run_start;
            let out_start = run_start * self.dim;
            let out_end = run_end * self.dim;
            self.read_range_into(ids[run_start], run_len, &mut out[out_start..out_end])?;
            if run_len > 1 {
                stats.range_calls += 1;
                stats.range_rows_read += run_len as u64;
            } else {
                stats.point_calls += 1;
            }
            run_start = run_end;
        }
        Ok(())
    }

    fn read_points_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(self.dim) {
            return Err(AnnError::log_index_error(format!(
                "Point matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(self.dim)
            )));
        }
        let plan = build_gather_plan(ids, self.row_bytes, *options, stats);
        for (window_idx, window) in plan.windows.iter().enumerate() {
            let range = self.row_byte_range(window.start_row, window.row_count as usize)?;
            let window_slice = self.mapped_rows_as_f32(&range)?;
            scatter_window_rows(&plan, window_idx, self.dim, window_slice, out);
            self.discard_mapped_range(&range);
        }
        Ok(())
    }

    fn prefers_coalesced_window_reads(&self) -> bool {
        true
    }

    fn disables_prefetch_pipeline(&self) -> bool {
        true
    }

    fn source_identity(&self) -> Option<String> {
        file_source_identity(&self.path)
    }

    fn discard_cached_pages(&self) -> AnnResult<()> {
        if self.keep_page_cache {
            return Ok(());
        }
        self.mmap.advise(Advice::DontNeed).map_err(|err| {
            AnnError::log_index_error(format!(
                "Failed to discard mmap point-store cached pages for {:?}: {err}",
                self.path
            ))
        })
    }
}

#[inline]
fn l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[inline]
fn l2_distance_sq(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .sum()
}

pub struct InmemDatasetPointStore<'a> {
    dataset: &'a InmemDataset<f32>,
    num_points: usize,
}

impl<'a> InmemDatasetPointStore<'a> {
    pub fn new(dataset: &'a InmemDataset<f32>, num_points: usize) -> Self {
        Self {
            dataset,
            num_points,
        }
    }
}

impl PointStore for InmemDatasetPointStore<'_> {
    fn len(&self) -> usize {
        self.num_points
    }

    fn dim(&self) -> usize {
        self.dataset.dim
    }

    fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
        if out.len() != self.dataset.dim {
            return Err(AnnError::log_index_error(format!(
                "Point buffer width mismatch: got {} expected {}",
                out.len(),
                self.dataset.dim
            )));
        }
        let vertex = self.dataset.get_vertex(pid)?;
        out.copy_from_slice(vertex.vector());
        Ok(())
    }

    fn read_range_into(&self, start_pid: u32, count: usize, out: &mut [f32]) -> AnnResult<()> {
        if count == 0 {
            return Ok(());
        }
        if out.len() != count.saturating_mul(self.dataset.dim) {
            return Err(AnnError::log_index_error(format!(
                "Range buffer size mismatch: got {} expected {}",
                out.len(),
                count.saturating_mul(self.dataset.dim)
            )));
        }
        for row in 0..count {
            let start = row * self.dataset.dim;
            let end = start + self.dataset.dim;
            self.read_point_into(start_pid + row as u32, &mut out[start..end])?;
        }
        Ok(())
    }

    fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(self.dataset.dim) {
            return Err(AnnError::log_index_error(format!(
                "Point matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(self.dataset.dim)
            )));
        }
        for (row, &pid) in ids.iter().enumerate() {
            let start = row * self.dataset.dim;
            let end = start + self.dataset.dim;
            self.read_point_into(pid, &mut out[start..end])?;
        }
        Ok(())
    }
}

pub struct LimitedPointStore<'a> {
    inner: &'a dyn PointStore,
    num_points: usize,
}

impl<'a> LimitedPointStore<'a> {
    pub fn new(inner: &'a dyn PointStore, num_points: usize) -> Self {
        Self { inner, num_points }
    }

    fn check_pid(&self, pid: u32) -> AnnResult<()> {
        if pid as usize >= self.num_points {
            return Err(AnnError::log_index_error(format!(
                "Point id {} out of bounds for limited store with {} points",
                pid, self.num_points
            )));
        }
        Ok(())
    }
}

impl PointStore for LimitedPointStore<'_> {
    fn len(&self) -> usize {
        self.num_points
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
        self.check_pid(pid)?;
        self.inner.read_point_into(pid, out)
    }

    fn read_range_into(&self, start_pid: u32, count: usize, out: &mut [f32]) -> AnnResult<()> {
        if count == 0 {
            return Ok(());
        }
        let start = start_pid as usize;
        let end = start.saturating_add(count);
        if end > self.num_points {
            return Err(AnnError::log_index_error(format!(
                "Range [{}..{}) out of bounds for limited store with {} points",
                start, end, self.num_points
            )));
        }
        self.inner.read_range_into(start_pid, count, out)
    }

    fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        for &pid in ids {
            self.check_pid(pid)?;
        }
        self.inner.read_points_into(ids, out)
    }

    fn read_points_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        for &pid in ids {
            self.check_pid(pid)?;
        }
        self.inner.read_points_into_stats(ids, out, stats)
    }

    fn read_points_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        for &pid in ids {
            self.check_pid(pid)?;
        }
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
        for &pid in ids {
            self.check_pid(pid)?;
        }
        self.inner
            .read_points_windowed_into_batch_stats(ids, out, options, stats)
    }

    #[allow(private_interfaces)]
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

pub(crate) struct ResidentSubsetPointStore {
    global_ids: Vec<u32>,
    id_to_row: Vec<(u32, usize)>,
    data: Vec<f32>,
    dim: usize,
    sorted_unique_ids: bool,
    point_id_capacity: usize,
}

impl ResidentSubsetPointStore {
    pub(crate) fn new(global_ids: Vec<u32>, dim: usize, data: Vec<f32>) -> AnnResult<Self> {
        if dim == 0 {
            return Err(AnnError::log_index_error(
                "Resident subset point store requires nonzero dimension".to_string(),
            ));
        }
        let expected = global_ids.len().saturating_mul(dim);
        if data.len() != expected {
            return Err(AnnError::log_index_error(format!(
                "Resident subset point matrix size mismatch: got {} expected {}",
                data.len(),
                expected
            )));
        }

        let point_id_capacity = global_ids
            .iter()
            .copied()
            .max()
            .map(|pid| pid as usize + 1)
            .unwrap_or(0);
        let sorted_unique_ids = is_strictly_increasing(&global_ids);
        let id_to_row = if sorted_unique_ids {
            Vec::new()
        } else {
            let mut id_to_row: Vec<(u32, usize)> = global_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(row, global_id)| (global_id, row))
                .collect();
            id_to_row.sort_unstable_by_key(|&(global_id, _)| global_id);
            id_to_row.dedup_by_key(|(global_id, _)| *global_id);
            id_to_row
        };

        Ok(Self {
            global_ids,
            id_to_row,
            data,
            dim,
            sorted_unique_ids,
            point_id_capacity,
        })
    }

    #[cfg(test)]
    pub(crate) fn resident_bytes(&self) -> usize {
        self.data
            .len()
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_add(
                self.global_ids
                    .len()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(
                self.id_to_row
                    .len()
                    .saturating_mul(std::mem::size_of::<(u32, usize)>()),
            )
    }

    fn local_row_for_global_id(&self, pid: u32) -> AnnResult<usize> {
        if self.sorted_unique_ids {
            return self.global_ids.binary_search(&pid).map_err(|_| {
                AnnError::log_index_error(format!(
                    "Point id {} not present in resident subset point store with {} points",
                    pid,
                    self.global_ids.len()
                ))
            });
        }
        self.id_to_row
            .binary_search_by_key(&pid, |&(global_id, _)| global_id)
            .map(|idx| self.id_to_row[idx].1)
            .map_err(|_| {
                AnnError::log_index_error(format!(
                    "Point id {} not present in resident subset point store with {} points",
                    pid,
                    self.global_ids.len()
                ))
            })
    }

    fn copy_row_into(&self, local_row: usize, out: &mut [f32]) {
        let start = local_row * self.dim;
        let end = start + self.dim;
        out.copy_from_slice(&self.data[start..end]);
    }

    fn check_point_buffer(&self, out: &[f32]) -> AnnResult<()> {
        if out.len() != self.dim {
            return Err(AnnError::log_index_error(format!(
                "Point buffer width mismatch: got {} expected {}",
                out.len(),
                self.dim
            )));
        }
        Ok(())
    }

    fn check_matrix_buffer(&self, rows: usize, out: &[f32]) -> AnnResult<()> {
        let expected = rows.saturating_mul(self.dim);
        if out.len() != expected {
            return Err(AnnError::log_index_error(format!(
                "Point matrix size mismatch: got {} expected {}",
                out.len(),
                expected
            )));
        }
        Ok(())
    }

    fn read_sorted_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let mut local_row = 0usize;
        for (row, &pid) in ids.iter().enumerate() {
            while local_row < self.global_ids.len() && self.global_ids[local_row] < pid {
                local_row += 1;
            }
            if self.global_ids.get(local_row).copied() != Some(pid) {
                return Err(AnnError::log_index_error(format!(
                    "Point id {} not present in resident subset point store with {} points",
                    pid,
                    self.global_ids.len()
                )));
            }
            let start = row * self.dim;
            let end = start + self.dim;
            self.copy_row_into(local_row, &mut out[start..end]);
        }
        Ok(())
    }
}

impl PointStore for ResidentSubsetPointStore {
    fn len(&self) -> usize {
        self.global_ids.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn point_id_capacity(&self) -> usize {
        self.point_id_capacity
    }

    fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
        self.check_point_buffer(out)?;
        let local_row = self.local_row_for_global_id(pid)?;
        self.copy_row_into(local_row, out);
        Ok(())
    }

    fn read_range_into(&self, start_pid: u32, count: usize, out: &mut [f32]) -> AnnResult<()> {
        if count == 0 {
            return Ok(());
        }
        self.check_matrix_buffer(count, out)?;
        if self.sorted_unique_ids {
            if let Ok(local_start) = self.global_ids.binary_search(&start_pid) {
                let local_end = local_start.saturating_add(count);
                if local_end <= self.global_ids.len()
                    && self.global_ids[local_end - 1]
                        == start_pid.saturating_add(count.saturating_sub(1) as u32)
                {
                    let start = local_start * self.dim;
                    let end = start + count * self.dim;
                    out.copy_from_slice(&self.data[start..end]);
                    return Ok(());
                }
            }
        }
        for row in 0..count {
            let pid = start_pid.saturating_add(row as u32);
            let start = row * self.dim;
            let end = start + self.dim;
            self.read_point_into(pid, &mut out[start..end])?;
        }
        Ok(())
    }

    fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        self.check_matrix_buffer(ids.len(), out)?;
        if ids.is_empty() {
            return Ok(());
        }
        if ids.len() == self.global_ids.len() && ids == self.global_ids.as_slice() {
            out.copy_from_slice(&self.data);
            return Ok(());
        }
        if self.sorted_unique_ids && is_nondecreasing(ids) {
            return self.read_sorted_points_into(ids, out);
        }
        for (row, &pid) in ids.iter().enumerate() {
            let local_row = self.local_row_for_global_id(pid)?;
            let start = row * self.dim;
            let end = start + self.dim;
            self.copy_row_into(local_row, &mut out[start..end]);
        }
        Ok(())
    }

    fn read_points_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        _stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        self.read_points_into(ids, out)
    }

    fn read_points_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        _options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        if !ids.is_empty() {
            stats.rows_requested += ids.len() as u64;
            stats.scatter_ops += ids.len() as u64;
        }
        self.read_points_into(ids, out)
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

    fn is_resident_subset(&self) -> bool {
        true
    }

    fn resident_rows(&self) -> Option<(&[u32], &[f32])> {
        Some((&self.global_ids, &self.data))
    }
}

fn is_strictly_increasing(ids: &[u32]) -> bool {
    ids.windows(2).all(|window| window[0] < window[1])
}

fn is_nondecreasing(ids: &[u32]) -> bool {
    ids.windows(2).all(|window| window[0] <= window[1])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VectorRunExtent {
    pub(crate) global_start: u32,
    pub(crate) byte_offset: u64,
    pub(crate) len: usize,
}

pub(crate) struct VectorRunPointStore {
    file: File,
    global_ids: Vec<u32>,
    id_to_row: Vec<(u32, usize)>,
    extents: Vec<VectorRunExtent>,
    dim: usize,
    #[cfg(test)]
    depth: usize,
    row_bytes: usize,
}

impl VectorRunPointStore {
    pub(crate) fn materialize(
        source: &dyn PointStore,
        global_ids: &[u32],
        path: &Path,
        _depth: usize,
    ) -> AnnResult<Self> {
        if source.dim() == 0 {
            return Err(AnnError::log_index_error(
                "Vector run point store requires nonzero dimension".to_string(),
            ));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = File::create(path)?;
        let dim = source.dim();
        let row_bytes = dim.saturating_mul(std::mem::size_of::<f32>());
        let mut extents = Vec::new();
        let mut byte_offset = 0u64;
        let mut run_start = 0usize;
        while run_start < global_ids.len() {
            let mut run_end = run_start + 1;
            while run_end < global_ids.len()
                && global_ids[run_end] == global_ids[run_end - 1].saturating_add(1)
            {
                run_end += 1;
            }
            let run_ids = &global_ids[run_start..run_end];
            let mut data = vec![0.0f32; run_ids.len().saturating_mul(dim)];
            source.read_points_into(run_ids, &mut data)?;
            writer.write_all(f32_slice_as_bytes(&data))?;
            extents.push(VectorRunExtent {
                global_start: run_ids[0],
                byte_offset,
                len: run_ids.len(),
            });
            byte_offset =
                byte_offset.saturating_add((data.len() * std::mem::size_of::<f32>()) as u64);
            run_start = run_end;
        }
        writer.flush()?;
        drop(writer);
        let file = File::open(path)?;

        let mut id_to_row: Vec<(u32, usize)> = global_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(row, global_id)| (global_id, row))
            .collect();
        id_to_row.sort_unstable_by_key(|&(global_id, _)| global_id);
        id_to_row.dedup_by_key(|(global_id, _)| *global_id);

        Ok(Self {
            file,
            global_ids: global_ids.to_vec(),
            id_to_row,
            extents,
            dim,
            #[cfg(test)]
            depth: _depth,
            row_bytes,
        })
    }

    #[cfg(test)]
    pub(crate) fn extents(&self) -> &[VectorRunExtent] {
        &self.extents
    }

    #[cfg(test)]
    pub(crate) fn depth(&self) -> usize {
        self.depth
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.global_ids.len().saturating_mul(self.row_bytes)
    }

    fn local_row_for_global_id(&self, pid: u32) -> AnnResult<usize> {
        self.id_to_row
            .binary_search_by_key(&pid, |&(global_id, _)| global_id)
            .map(|idx| self.id_to_row[idx].1)
            .map_err(|_| {
                AnnError::log_index_error(format!(
                    "Point id {} not present in vector run point store with {} points",
                    pid,
                    self.global_ids.len()
                ))
            })
    }

    fn extent_for_local_row(&self, local_row: usize) -> AnnResult<(VectorRunExtent, usize)> {
        let pid = self.global_ids.get(local_row).copied().ok_or_else(|| {
            AnnError::log_index_error(format!(
                "Local vector run row {} out of bounds for {} rows",
                local_row,
                self.global_ids.len()
            ))
        })?;
        self.extents
            .iter()
            .copied()
            .find_map(|extent| {
                let start = extent.global_start;
                let end = start.saturating_add(extent.len as u32);
                (pid >= start && pid < end).then(|| (extent, (pid - start) as usize))
            })
            .ok_or_else(|| {
                AnnError::log_index_error(format!("Point id {} has no vector run extent", pid))
            })
    }

    fn read_local_rows_into(
        &self,
        local_start: usize,
        count: usize,
        out: &mut [f32],
    ) -> AnnResult<()> {
        if count == 0 {
            return Ok(());
        }
        if out.len() != count.saturating_mul(self.dim) {
            return Err(AnnError::log_index_error(format!(
                "Vector run buffer size mismatch: got {} expected {}",
                out.len(),
                count.saturating_mul(self.dim)
            )));
        }
        let (extent, row_in_extent) = self.extent_for_local_row(local_start)?;
        if row_in_extent.saturating_add(count) > extent.len {
            return Err(AnnError::log_index_error(
                "Vector run read crossed extent boundary".to_string(),
            ));
        }
        let offset = extent
            .byte_offset
            .saturating_add((row_in_extent * self.row_bytes) as u64);
        read_file_exact_at(&self.file, f32_slice_as_bytes_mut(out), offset)
    }

    fn check_matrix_buffer(&self, rows: usize, out: &[f32]) -> AnnResult<()> {
        let expected = rows.saturating_mul(self.dim);
        if out.len() != expected {
            return Err(AnnError::log_index_error(format!(
                "Point matrix size mismatch: got {} expected {}",
                out.len(),
                expected
            )));
        }
        Ok(())
    }
}

impl PointStore for VectorRunPointStore {
    fn len(&self) -> usize {
        self.global_ids.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn point_id_capacity(&self) -> usize {
        self.global_ids
            .iter()
            .copied()
            .max()
            .map(|pid| pid as usize + 1)
            .unwrap_or(0)
    }

    fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
        if out.len() != self.dim {
            return Err(AnnError::log_index_error(format!(
                "Point buffer width mismatch: got {} expected {}",
                out.len(),
                self.dim
            )));
        }
        let local_row = self.local_row_for_global_id(pid)?;
        self.read_local_rows_into(local_row, 1, out)
    }

    fn read_range_into(&self, start_pid: u32, count: usize, out: &mut [f32]) -> AnnResult<()> {
        self.check_matrix_buffer(count, out)?;
        if count == 0 {
            return Ok(());
        }
        let local_start = self.local_row_for_global_id(start_pid)?;
        for row in 0..count {
            let expected = start_pid.saturating_add(row as u32);
            let local_row = local_start.saturating_add(row);
            if self.global_ids.get(local_row).copied() != Some(expected) {
                return Err(AnnError::log_index_error(format!(
                    "Vector run range [{}..{}) is not contiguous in the run",
                    start_pid,
                    start_pid.saturating_add(count as u32)
                )));
            }
        }
        self.read_local_rows_into(local_start, count, out)
    }

    fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let mut stats = PointBatchStats::default();
        self.read_points_into_stats(ids, out, &mut stats)
    }

    fn read_points_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        self.check_matrix_buffer(ids.len(), out)?;
        if ids.is_empty() {
            return Ok(());
        }

        let mut rows: Vec<(usize, usize)> = ids
            .iter()
            .copied()
            .enumerate()
            .map(|(out_row, pid)| {
                self.local_row_for_global_id(pid)
                    .map(|local_row| (local_row, out_row))
            })
            .collect::<AnnResult<Vec<_>>>()?;
        rows.sort_unstable_by_key(|&(local_row, out_row)| (local_row, out_row));

        let mut run_start = 0usize;
        while run_start < rows.len() {
            let mut run_end = run_start + 1;
            while run_end < rows.len()
                && rows[run_end].0 == rows[run_end - 1].0.saturating_add(1)
                && self.extent_for_local_row(rows[run_end - 1].0)?.0
                    == self.extent_for_local_row(rows[run_end].0)?.0
            {
                run_end += 1;
            }
            let local_start = rows[run_start].0;
            let run_len = run_end - run_start;
            let mut buffer = vec![0.0f32; run_len.saturating_mul(self.dim)];
            self.read_local_rows_into(local_start, run_len, &mut buffer)?;
            for (offset, &(_, out_row)) in rows[run_start..run_end].iter().enumerate() {
                let src_start = offset * self.dim;
                let dst_start = out_row * self.dim;
                out[dst_start..dst_start + self.dim]
                    .copy_from_slice(&buffer[src_start..src_start + self.dim]);
            }
            stats.range_calls += 1;
            stats.range_rows_read += run_len as u64;
            run_start = run_end;
        }
        stats.bytes_read += (ids.len().saturating_mul(self.row_bytes)) as u64;
        Ok(())
    }

    fn read_points_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        _options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        let mut point_stats = PointBatchStats::default();
        self.read_points_into_stats(ids, out, &mut point_stats)?;
        stats.rows_requested += ids.len() as u64;
        stats.logical_bytes += point_stats.bytes_read;
        stats.physical_bytes += point_stats.bytes_read;
        stats.windows_submitted += point_stats.range_calls;
        stats.range_windows += point_stats.range_calls;
        stats.range_rows_read += point_stats.range_rows_read;
        stats.rows_per_window_sum += point_stats.range_rows_read;
        stats.scatter_ops += ids.len() as u64;
        Ok(())
    }

    fn read_points_windowed_into_batch_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut PointBatchStats,
    ) -> AnnResult<()> {
        let mut windowed = WindowedGatherStats::default();
        self.read_points_windowed_into_stats(ids, out, options, &mut windowed)?;
        windowed.apply_to_point_batch_stats(stats);
        Ok(())
    }

    fn is_vector_run(&self) -> bool {
        true
    }
}

fn f32_slice_as_bytes(slice: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn f32_slice_as_bytes_mut(slice: &mut [f32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, std::mem::size_of_val(slice))
    }
}

fn read_file_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> AnnResult<()> {
    while !buf.is_empty() {
        let read = file.read_at(buf, offset)?;
        if read == 0 {
            return Err(AnnError::log_index_error(
                "Unexpected EOF while reading vector run".to_string(),
            ));
        }
        let (_, rest) = buf.split_at_mut(read);
        buf = rest;
        offset += read as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        DirectPointStore, InmemDatasetPointStore, LimitedPointStore, MmapPointStore,
        PointBatchStats, PointStore, ResidentSubsetPointStore, VectorRunPointStore,
        WindowedGatherOptions, WindowedGatherStats,
    };
    use crate::forgeann::direct_io::DirectIoConfig;
    use crate::model::InmemDataset;
    use crate::utils::save_bin_f32;

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

    fn write_test_fbin(path: &Path, rows: usize, dim: usize) {
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.25 + col as f32 * 0.5))
            .collect();
        save_bin_f32(path, &data, rows, dim, 0).unwrap();
    }

    #[test]
    fn direct_point_store_reads_point_and_ranges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.fbin");
        write_test_fbin(&path, 8, 4);
        let store = DirectPointStore::open(&path).unwrap();

        let mut point = vec![0.0f32; 4];
        store.read_point_into(3, &mut point).unwrap();
        assert_eq!(point, vec![0.75, 1.25, 1.75, 2.25]);

        let ids = vec![1, 2, 3, 6];
        let points = store.read_points(&ids).unwrap();
        assert_eq!(points.len(), ids.len() * 4);
        assert_eq!(&points[0..4], &[0.25, 0.75, 1.25, 1.75]);
        assert_eq!(&points[12..16], &[1.5, 2.0, 2.5, 3.0]);
    }

    #[test]
    fn mmap_point_store_reads_point_ranges_and_ids_like_direct() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmap-tiny.fbin");
        write_test_fbin(&path, 16, 6);
        let direct = DirectPointStore::open(&path).unwrap();
        let mmap = MmapPointStore::open(&path).unwrap();

        assert_eq!(mmap.path(), path.as_path());
        assert_eq!(mmap.len(), direct.len());
        assert_eq!(mmap.dim(), direct.dim());

        let mut direct_point = vec![0.0f32; 6];
        let mut mmap_point = vec![0.0f32; 6];
        direct.read_point_into(7, &mut direct_point).unwrap();
        mmap.read_point_into(7, &mut mmap_point).unwrap();
        assert_eq!(mmap_point, direct_point);

        let mut direct_range = vec![0.0f32; 4 * 6];
        let mut mmap_range = vec![0.0f32; 4 * 6];
        direct.read_range_into(3, 4, &mut direct_range).unwrap();
        mmap.read_range_into(3, 4, &mut mmap_range).unwrap();
        assert_eq!(mmap_range, direct_range);

        let ids = vec![0, 1, 5, 6, 11, 15];
        let mut direct_points = vec![0.0f32; ids.len() * 6];
        let mut mmap_points = vec![0.0f32; ids.len() * 6];
        direct.read_points_into(&ids, &mut direct_points).unwrap();
        mmap.read_points_into(&ids, &mut mmap_points).unwrap();
        assert_eq!(mmap_points, direct_points);
    }

    #[test]
    fn mmap_point_store_preserves_duplicate_rows_and_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmap-duplicates.fbin");
        write_test_fbin(&path, 12, 4);
        let mmap = MmapPointStore::open(&path).unwrap();

        let ids = vec![8, 2, 2, 5, 8, 3];
        let mut out = vec![0.0f32; ids.len() * 4];
        mmap.read_points_into(&ids, &mut out).unwrap();

        for (row, &pid) in ids.iter().enumerate() {
            let mut point = vec![0.0f32; 4];
            mmap.read_point_into(pid, &mut point).unwrap();
            assert_eq!(&out[row * 4..(row + 1) * 4], point.as_slice());
        }
        assert_eq!(&out[4..8], &out[8..12]);
        assert_eq!(&out[0..4], &out[16..20]);
    }

    #[test]
    fn mmap_point_store_windowed_reads_match_direct_for_unsorted_gap_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmap-windowed.fbin");
        write_test_fbin(&path, 32, 5);
        let direct = DirectPointStore::open(&path).unwrap();
        let mmap = MmapPointStore::open(&path).unwrap();

        let ids = vec![20, 3, 4, 4, 9, 11, 10, 20, 0];
        let options = WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 128,
            alignment_bytes: 4096,
            sort_ids: true,
        };
        let mut expected = vec![0.0f32; ids.len() * 5];
        let mut actual = vec![0.0f32; ids.len() * 5];
        let mut stats = WindowedGatherStats::default();
        direct.read_points_into(&ids, &mut expected).unwrap();
        mmap.read_points_windowed_into_stats(&ids, &mut actual, &options, &mut stats)
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(stats.rows_requested, ids.len() as u64);
        assert_eq!(stats.scatter_ops, ids.len() as u64);
        assert!(stats.windows_submitted >= 2);
        assert!(mmap.prefers_coalesced_window_reads());
    }

    #[test]
    fn mmap_point_store_reports_bad_buffers_and_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmap-errors.fbin");
        write_test_fbin(&path, 8, 4);
        let mmap = MmapPointStore::open(&path).unwrap();

        let mut short_point = vec![0.0f32; 3];
        assert!(mmap.read_point_into(0, &mut short_point).is_err());

        let mut point = vec![0.0f32; 4];
        assert!(mmap.read_point_into(8, &mut point).is_err());

        let mut short_range = vec![0.0f32; 7];
        assert!(mmap.read_range_into(0, 2, &mut short_range).is_err());

        let mut range = vec![0.0f32; 2 * 4];
        assert!(mmap.read_range_into(7, 2, &mut range).is_err());

        let mut matrix = vec![0.0f32; 2 * 4];
        assert!(mmap.read_points_into(&[1, 8], &mut matrix).is_err());
    }

    #[test]
    fn limited_point_store_wraps_mmap_point_store_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmap-limited.fbin");
        write_test_fbin(&path, 12, 4);
        let mmap = MmapPointStore::open(&path).unwrap();
        let limited = LimitedPointStore::new(&mmap, 6);

        let mut ok = vec![0.0f32; 4];
        limited.read_point_into(5, &mut ok).unwrap();

        let mut bad = vec![0.0f32; 4];
        assert!(limited.read_point_into(6, &mut bad).is_err());

        let mut range = vec![0.0f32; 2 * 4];
        assert!(limited.read_range_into(5, 2, &mut range).is_err());
    }

    #[test]
    fn inmem_and_direct_point_store_match_reads_and_medoid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.fbin");
        write_test_fbin(&path, 16, 6);

        let inmem = build_test_dataset(16, 6);
        let inmem_store = InmemDatasetPointStore::new(&inmem, 16);
        let direct_store = DirectPointStore::open(&path).unwrap();

        let ids = vec![0, 1, 2, 5, 6, 9, 10, 15];
        let mut inmem_points = vec![0.0f32; ids.len() * 6];
        let mut direct_points = vec![0.0f32; ids.len() * 6];
        inmem_store
            .read_points_into(&ids, &mut inmem_points)
            .unwrap();
        direct_store
            .read_points_into(&ids, &mut direct_points)
            .unwrap();
        assert_eq!(direct_points, inmem_points);

        let inmem_medoid = inmem_store
            .calculate_medoid_point_id_with_threads(Some(1))
            .unwrap();
        let direct_medoid = direct_store
            .calculate_medoid_point_id_with_threads(Some(1))
            .unwrap();
        assert_eq!(direct_medoid, inmem_medoid);
    }

    #[test]
    fn direct_point_store_strict_io_matches_buffered_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("strict-tiny.fbin");
        write_test_fbin(&path, 17, 7);

        let buffered_store = DirectPointStore::open(&path).unwrap();
        let strict_store =
            DirectPointStore::open_with_config(&path, DirectIoConfig::enabled_with_alignment(4096))
                .unwrap();

        assert!(strict_store.io_config().enabled);

        let ids = vec![0, 1, 2, 5, 6, 9, 10, 15, 16];
        let mut buffered_points = vec![0.0f32; ids.len() * 7];
        let mut strict_points = vec![0.0f32; ids.len() * 7];
        buffered_store
            .read_points_into(&ids, &mut buffered_points)
            .unwrap();
        strict_store
            .read_points_into(&ids, &mut strict_points)
            .unwrap();
        assert_eq!(strict_points, buffered_points);

        let mut strict_range = vec![0.0f32; 4 * 7];
        strict_store
            .read_range_into(3, 4, &mut strict_range)
            .unwrap();
        assert_eq!(
            &strict_range[0..7],
            &[0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
        );
    }

    #[test]
    fn read_points_into_stats_coalesces_sorted_ids_and_counts_runs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stats.fbin");
        write_test_fbin(&path, 20, 4);
        let store = DirectPointStore::open(&path).unwrap();

        // Sorted IDs with runs: [1,2,3] [7,8] [15]
        let ids: Vec<u32> = vec![1, 2, 3, 7, 8, 15];
        let mut out_stats = vec![0.0f32; ids.len() * 4];
        let mut stats = super::PointBatchStats::default();
        store
            .read_points_into_stats(&ids, &mut out_stats, &mut stats)
            .unwrap();

        // Verify data matches pointwise reads
        for (i, &pid) in ids.iter().enumerate() {
            let mut point = vec![0.0f32; 4];
            store.read_point_into(pid, &mut point).unwrap();
            assert_eq!(&out_stats[i * 4..(i + 1) * 4], point.as_slice());
        }

        // Verify stats: [1,2,3] = range_calls=1, rows=3; [7,8] = range_calls=1, rows=2; [15] =
        // point_calls=1
        assert_eq!(
            stats.range_calls, 2,
            "expected 2 range calls (runs of 3 and 2)"
        );
        assert_eq!(stats.range_rows_read, 5, "expected 5 rows via range (3+2)");
        assert_eq!(stats.point_calls, 1, "expected 1 point call (singleton 15)");
        assert_eq!(stats.bytes_read, (ids.len() * 4 * 4) as u64);
        assert!((stats.range_hit_ratio() - 5.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn read_points_into_stats_fully_contiguous_gets_single_range_call() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("contiguous.fbin");
        write_test_fbin(&path, 10, 4);
        let store = DirectPointStore::open(&path).unwrap();

        let ids: Vec<u32> = vec![0, 1, 2, 3, 4];
        let mut out = vec![0.0f32; ids.len() * 4];
        let mut stats = super::PointBatchStats::default();
        store
            .read_points_into_stats(&ids, &mut out, &mut stats)
            .unwrap();

        assert_eq!(stats.range_calls, 1);
        assert_eq!(stats.range_rows_read, 5);
        assert_eq!(stats.point_calls, 0);
        assert!((stats.range_hit_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn limited_point_store_preserves_inner_batched_read_stats() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("limited-stats.fbin");
        write_test_fbin(&path, 20, 4);
        let store = DirectPointStore::open(&path).unwrap();
        let limited = LimitedPointStore::new(&store, 20);

        let ids: Vec<u32> = vec![1, 2, 3, 7, 8, 15];
        let mut out = vec![0.0f32; ids.len() * 4];
        let mut stats = super::PointBatchStats::default();
        limited
            .read_points_into_stats(&ids, &mut out, &mut stats)
            .unwrap();

        assert_eq!(stats.range_calls, 2);
        assert_eq!(stats.range_rows_read, 5);
        assert_eq!(stats.point_calls, 1);
        assert_eq!(stats.bytes_read, (ids.len() * 4 * 4) as u64);
        assert!((stats.range_hit_ratio() - 5.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn limited_point_store_preserves_inner_windowed_read_stats() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("limited-windowed.fbin");
        write_test_fbin(&path, 20, 4);
        let store = DirectPointStore::open(&path).unwrap();
        let limited = LimitedPointStore::new(&store, 20);

        let ids: Vec<u32> = vec![1, 3, 4];
        let mut out = vec![0.0f32; ids.len() * 4];
        let mut stats = WindowedGatherStats::default();
        let options = WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 64 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        };

        limited
            .read_points_windowed_into_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        assert_eq!(stats.windows_submitted, 1);
        assert_eq!(stats.overread_rows, 1);
        assert_eq!(stats.scatter_ops, ids.len() as u64);
    }

    #[test]
    fn read_points_windowed_into_stats_preserves_logical_order_for_unsorted_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("windowed-unsorted.fbin");
        write_test_fbin(&path, 20, 4);
        let store = DirectPointStore::open(&path).unwrap();

        let ids: Vec<u32> = vec![8, 2, 3, 7];
        let mut out = vec![0.0f32; ids.len() * 4];
        let mut stats = super::WindowedGatherStats::default();
        let options = super::WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 64 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        };
        store
            .read_points_windowed_into_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        for (i, &pid) in ids.iter().enumerate() {
            let mut point = vec![0.0f32; 4];
            store.read_point_into(pid, &mut point).unwrap();
            assert_eq!(&out[i * 4..(i + 1) * 4], point.as_slice());
        }

        assert_eq!(stats.rows_requested, ids.len() as u64);
        assert_eq!(stats.scatter_ops, ids.len() as u64);
    }

    #[test]
    fn read_points_windowed_into_stats_merges_small_gaps_into_single_window() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("windowed-gap.fbin");
        write_test_fbin(&path, 20, 4);
        let store = DirectPointStore::open(&path).unwrap();

        let ids: Vec<u32> = vec![1, 3, 4];
        let mut out = vec![0.0f32; ids.len() * 4];
        let mut stats = super::WindowedGatherStats::default();
        let options = super::WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 64 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        };
        store
            .read_points_windowed_into_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        assert_eq!(stats.windows_submitted, 1);
        assert_eq!(stats.overread_rows, 1);
        assert_eq!(stats.rows_per_window_avg(), 4.0);
        assert!(stats.physical_bytes >= stats.logical_bytes);
    }

    #[test]
    fn strict_direct_windowed_reads_try_uring_for_multi_window_batches() {
        super::reset_uring_gather_test_counters();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("strict-windowed-uring.fbin");
        write_test_fbin(&path, 64, 8);
        let buffered = DirectPointStore::open(&path).unwrap();
        let strict =
            DirectPointStore::open_with_config(&path, DirectIoConfig::enabled_with_alignment(4096))
                .unwrap();

        let ids: Vec<u32> = vec![12, 2, 3, 30, 31, 7, 50, 51, 52];
        let options = super::WindowedGatherOptions {
            max_gap_rows: 0,
            max_window_bytes: 64,
            alignment_bytes: 4096,
            sort_ids: true,
        };
        let mut expected = vec![0.0f32; ids.len() * 8];
        let mut actual = vec![0.0f32; ids.len() * 8];
        let mut stats = super::WindowedGatherStats::default();

        buffered.read_points_into(&ids, &mut expected).unwrap();
        strict
            .read_points_windowed_into_stats(&ids, &mut actual, &options, &mut stats)
            .unwrap();

        assert_eq!(actual, expected);
        assert!(
            super::uring_gather_test_attempts() > 0,
            "strict direct multi-window reads should attempt the io_uring fast path"
        );
    }

    #[test]
    fn resident_subset_point_store_reads_global_ids_in_order() {
        let global_ids = vec![8, 2, 5, 3];
        let dim = 3;
        let data: Vec<f32> = global_ids
            .iter()
            .flat_map(|&id| (0..dim).map(move |axis| id as f32 * 10.0 + axis as f32))
            .collect();
        let store = ResidentSubsetPointStore::new(global_ids.clone(), dim, data).unwrap();

        let mut point = vec![0.0; dim];
        store.read_point_into(5, &mut point).unwrap();
        assert_eq!(point, vec![50.0, 51.0, 52.0]);

        let ids = vec![3, 8, 2, 5];
        let mut out = vec![0.0; ids.len() * dim];
        store.read_points_into(&ids, &mut out).unwrap();

        assert_eq!(
            out,
            vec![
                30.0, 31.0, 32.0, 80.0, 81.0, 82.0, 20.0, 21.0, 22.0, 50.0, 51.0, 52.0
            ]
        );
        assert_eq!(store.len(), global_ids.len());
        assert_eq!(store.dim(), dim);
    }

    #[test]
    fn resident_subset_point_store_windowed_batch_preserves_order_without_disk_io_stats() {
        let global_ids = vec![40, 10, 30, 20];
        let dim = 2;
        let data: Vec<f32> = global_ids
            .iter()
            .flat_map(|&id| [id as f32, id as f32 + 0.5])
            .collect();
        let store = ResidentSubsetPointStore::new(global_ids, dim, data).unwrap();

        let ids = vec![20, 40, 10, 30];
        let mut out = vec![0.0; ids.len() * dim];
        let options = WindowedGatherOptions {
            max_gap_rows: 10,
            max_window_bytes: 4096,
            alignment_bytes: 4096,
            sort_ids: true,
        };
        let mut stats = PointBatchStats::default();
        store
            .read_points_windowed_into_batch_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        assert_eq!(out, vec![20.0, 20.5, 40.0, 40.5, 10.0, 10.5, 30.0, 30.5]);
        assert_eq!(stats.point_calls, 0);
        assert_eq!(stats.range_calls, 0);
        assert_eq!(stats.range_rows_read, 0);
        assert_eq!(stats.bytes_read, 0);
    }

    #[test]
    fn resident_subset_point_store_uses_sorted_unique_fast_paths() {
        let global_ids = vec![10, 20, 21, 40];
        let dim = 2;
        let data: Vec<f32> = global_ids
            .iter()
            .flat_map(|&id| [id as f32, id as f32 + 0.5])
            .collect();
        let store = ResidentSubsetPointStore::new(global_ids.clone(), dim, data.clone()).unwrap();
        assert!(store.id_to_row.is_empty());

        let mut full = vec![0.0; data.len()];
        store.read_points_into(&global_ids, &mut full).unwrap();
        assert_eq!(full, data);

        let ids = vec![20, 20, 40];
        let mut subset = vec![0.0; ids.len() * dim];
        store.read_points_into(&ids, &mut subset).unwrap();
        assert_eq!(subset, vec![20.0, 20.5, 20.0, 20.5, 40.0, 40.5]);

        let mut range = vec![0.0; 2 * dim];
        store.read_range_into(20, 2, &mut range).unwrap();
        assert_eq!(range, vec![20.0, 20.5, 21.0, 21.5]);
    }

    #[test]
    fn resident_subset_point_store_missing_global_id_errors() {
        let store = ResidentSubsetPointStore::new(vec![1, 3], 2, vec![1.0, 1.5, 3.0, 3.5]).unwrap();
        let mut out = vec![0.0; 2];

        assert!(store.read_point_into(2, &mut out).is_err());
    }

    #[test]
    fn vector_run_point_store_preserves_requested_logical_order_and_reports_runs() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.fbin");
        let run_path = dir.path().join("partition_d02_vectors.bin");
        write_test_fbin(&source_path, 16, 4);
        let source = DirectPointStore::open(&source_path).unwrap();

        let global_ids = vec![3, 4, 5, 9, 10, 12];
        let store = VectorRunPointStore::materialize(&source, &global_ids, &run_path, 2).unwrap();

        let ids = vec![10, 3, 12, 4, 5, 9];
        let mut actual = vec![0.0f32; ids.len() * 4];
        let mut stats = PointBatchStats::default();
        store
            .read_points_into_stats(&ids, &mut actual, &mut stats)
            .unwrap();

        let mut expected = vec![0.0f32; ids.len() * 4];
        source.read_points_into(&ids, &mut expected).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(stats.bytes_read, (ids.len() * 4 * 4) as u64);
        assert_eq!(stats.range_calls, 3);
        assert_eq!(stats.range_rows_read, 6);
        assert_eq!(stats.point_calls, 0);
        assert_eq!(store.depth(), 2);
        assert_eq!(store.extents().len(), 3);
    }
}
