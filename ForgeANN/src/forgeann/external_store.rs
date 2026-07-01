use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::io::{BufWriter, Read, Write};
use std::mem::size_of;
use std::os::unix::io::RawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use io_uring::{IoUring, opcode, types};
use memmap2::{Advice, Mmap, MmapOptions};
use rayon::prelude::*;

use super::direct_io::{DirectIoConfig, DirectIoFile};
use super::hash_prune::{SketchAccessor, SketchStore, generate_hyperplanes};
use super::io_runtime::{BoundedReadPlan, BoundedReadPlanner};
use super::params::ForgeANNParams;
use super::point_store::{
    InmemDatasetPointStore, PointStore, WindowedGatherOptions, WindowedGatherStats,
    build_gather_plan, scatter_window_rows,
};
use crate::common::{AlignedBoxWithSlice, AnnError, AnnResult};
use crate::model::InmemDataset;
use crate::utils::thread_pool::with_rayon_thread_pool;

const HEADER_BYTES: usize = size_of::<u64>() * 2;
const DEFAULT_SKETCH_BUILD_CHUNK_BYTES: usize = 64 * 1024 * 1024;
const MIN_SKETCH_BUILD_CHUNK_ROWS: usize = 1024;
const SKETCH_START_SAMPLE_POINTS: usize = 8192;
const SKETCH_URING_WINDOW_GATHER_MIN_WINDOWS: usize = 2;
const SKETCH_URING_WINDOW_GATHER_QUEUE_DEPTH: usize = 32;

thread_local! {
    static SKETCH_URING_GATHER_RING: RefCell<Option<IoUring>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SketchBuildPlan {
    threads: usize,
    chunk_rows: usize,
    estimated_concurrent_bytes: usize,
    budget_bytes: usize,
    bytes_per_row: usize,
}

impl SketchBuildPlan {
    fn for_params(
        params: &ForgeANNParams,
        width: usize,
        dim: usize,
        requested_threads: u32,
    ) -> Self {
        let requested = requested_sketch_threads(requested_threads);
        let memory_threads = sketch_build_memory_thread_limit(params, dim, width);
        let mut threads = requested.min(memory_threads).max(1);
        let bytes_per_row = sketch_build_bytes_per_row(width, dim);
        let min_chunk_bytes = MIN_SKETCH_BUILD_CHUNK_ROWS.saturating_mul(bytes_per_row);

        let budget_bytes = if params.oom_sketch_cache_bytes == 0 {
            DEFAULT_SKETCH_BUILD_CHUNK_BYTES.saturating_mul(threads)
        } else {
            params.oom_sketch_cache_bytes
        };

        if params.oom_sketch_cache_bytes > 0 {
            let budget_threads = (budget_bytes / min_chunk_bytes.max(1)).max(1);
            threads = threads.min(budget_threads).max(1);
        }

        let per_thread_budget = (budget_bytes / threads.max(1)).max(bytes_per_row);
        let chunk_rows = (per_thread_budget / bytes_per_row).max(MIN_SKETCH_BUILD_CHUNK_ROWS);
        let estimated_concurrent_bytes = chunk_rows
            .saturating_mul(bytes_per_row)
            .saturating_mul(threads);

        Self {
            threads,
            chunk_rows,
            estimated_concurrent_bytes,
            budget_bytes,
            bytes_per_row,
        }
    }
}

#[derive(Clone, Debug)]
struct SketchStartSample {
    pid: u32,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
struct SketchBuildStartSummary {
    dim: usize,
    rows: usize,
    sum: Vec<f64>,
    samples: Vec<SketchStartSample>,
}

impl SketchBuildStartSummary {
    fn empty(dim: usize) -> Self {
        Self {
            dim,
            rows: 0,
            sum: vec![0.0; dim],
            samples: Vec::new(),
        }
    }

    fn from_chunk(
        chunk_start: usize,
        rows: usize,
        dim: usize,
        point_chunk: &[f32],
        sample_stride: usize,
    ) -> Self {
        let mut summary = Self::empty(dim);
        summary.rows = rows;
        let sample_stride = sample_stride.max(1);
        for row in 0..rows {
            let pid = chunk_start + row;
            let start = row * dim;
            let end = start + dim;
            let point = &point_chunk[start..end];
            for (dst, &value) in summary.sum.iter_mut().zip(point.iter()) {
                *dst += value as f64;
            }
            if pid % sample_stride == 0 {
                summary.samples.push(SketchStartSample {
                    pid: pid as u32,
                    values: point.to_vec(),
                });
            }
        }
        summary
    }

    fn merge(&mut self, other: Self) {
        debug_assert_eq!(self.dim, other.dim);
        self.rows = self.rows.saturating_add(other.rows);
        for (dst, value) in self.sum.iter_mut().zip(other.sum.into_iter()) {
            *dst += value;
        }
        self.samples.extend(other.samples);
    }

    fn sample_count(&self) -> usize {
        self.samples.len()
    }

    fn sampled_medoid_start(&self) -> Option<u32> {
        if self.rows == 0 || self.samples.is_empty() {
            return None;
        }
        let denom = self.rows as f64;
        let center: Vec<f64> = self.sum.iter().map(|value| *value / denom).collect();
        let mut best: Option<(f64, u32)> = None;
        for sample in &self.samples {
            let dist = sample
                .values
                .iter()
                .zip(center.iter())
                .map(|(&value, &center_value)| {
                    let delta = value as f64 - center_value;
                    delta * delta
                })
                .sum::<f64>();
            if !dist.is_finite() {
                continue;
            }
            match best {
                Some((best_dist, best_pid))
                    if dist > best_dist || (dist == best_dist && sample.pid >= best_pid) => {}
                _ => best = Some((dist, sample.pid)),
            }
        }
        best.map(|(_, pid)| pid)
    }
}

fn sketch_start_sample_stride(num_points: usize) -> usize {
    if num_points <= SKETCH_START_SAMPLE_POINTS {
        1
    } else {
        num_points.div_ceil(SKETCH_START_SAMPLE_POINTS)
    }
}

#[derive(Debug)]
pub struct DiskSketchStore {
    path: PathBuf,
    backing: SketchBacking,
    rows: usize,
    width: usize,
    sampled_medoid_start: Option<u32>,
}

#[derive(Debug)]
enum SketchBacking {
    Mmap { _file: File, mmap: Mmap },
    Direct { file: DirectIoFile },
}

impl DiskSketchStore {
    pub fn write_from_memory(path: &Path, sketches: &SketchStore) -> AnnResult<Self> {
        Self::write_from_memory_with_config(path, sketches, DirectIoConfig::disabled())
    }

    pub fn write_from_memory_with_config(
        path: &Path,
        sketches: &SketchStore,
        io_cfg: DirectIoConfig,
    ) -> AnnResult<Self> {
        if io_cfg.enabled {
            write_sketch_file_direct(
                path,
                sketches.rows(),
                sketches.width(),
                sketches.data(),
                io_cfg,
            )?;
            return Self::open_with_config(path, io_cfg);
        }

        let mut out = BufWriter::new(File::create(path)?);
        let width = sketches.width();
        let rows = sketches.rows();

        out.write_all(&(rows as u64).to_le_bytes())?;
        out.write_all(&(width as u64).to_le_bytes())?;
        for value in sketches.data() {
            out.write_all(&value.to_le_bytes())?;
        }
        out.flush()?;

        Self::open_with_config(path, io_cfg)
    }

    pub fn build_from_dataset(
        path: &Path,
        dataset: &InmemDataset<f32>,
        num_points: usize,
        params: &ForgeANNParams,
        num_threads: u32,
    ) -> AnnResult<Self> {
        let store = InmemDatasetPointStore::new(dataset, num_points);
        Self::build_from_point_store(path, &store, num_points, params, num_threads)
    }

    pub fn build_from_point_store(
        path: &Path,
        dataset: &dyn PointStore,
        num_points: usize,
        params: &ForgeANNParams,
        requested_threads: u32,
    ) -> AnnResult<Self> {
        let io_cfg = if params.strict_oom_io_enabled() {
            DirectIoConfig::enabled_with_alignment(4096)
        } else {
            DirectIoConfig::disabled()
        };
        let dim = dataset.dim();
        let width = params.m_hash_bits;
        let hyperplanes = generate_hyperplanes(dim, width, params.random_seed);
        let build_plan = SketchBuildPlan::for_params(params, width, dim, requested_threads);
        let chunk_rows = build_plan.chunk_rows;
        let sketch_threads = build_plan.threads;
        let start_sample_stride = sketch_start_sample_stride(num_points);
        let mut start_summary = SketchBuildStartSummary::empty(dim);
        tracing::info!(
            "[disk-sketch-store/build-plan] rows={} dim={} width={} threads={} chunk_rows={} bytes_per_row={} estimated_concurrent_bytes={} sketch_cache_budget_bytes={} strict_io={}",
            num_points,
            dim,
            width,
            build_plan.threads,
            build_plan.chunk_rows,
            build_plan.bytes_per_row,
            build_plan.estimated_concurrent_bytes,
            build_plan.budget_bytes,
            io_cfg.enabled,
        );
        if sketch_threads <= 1 {
            let expected_bytes = HEADER_BYTES.saturating_add(
                num_points
                    .saturating_mul(width)
                    .saturating_mul(size_of::<f32>()),
            );
            let direct_writer = if io_cfg.enabled {
                let writer = DirectIoFile::create_write(path, io_cfg)?;
                writer.write_all_at(&(num_points as u64).to_le_bytes(), 0)?;
                writer.write_all_at(&(width as u64).to_le_bytes(), size_of::<u64>() as u64)?;
                Some(writer)
            } else {
                None
            };
            let mut out = if io_cfg.enabled {
                None
            } else {
                let mut writer = BufWriter::new(File::create(path)?);
                writer.write_all(&(num_points as u64).to_le_bytes())?;
                writer.write_all(&(width as u64).to_le_bytes())?;
                Some(writer)
            };
            for chunk_start in (0..num_points).step_by(chunk_rows) {
                let rows = (num_points - chunk_start).min(chunk_rows);
                let mut point_chunk = vec![0.0f32; rows.saturating_mul(dim)];
                let mut chunk_data = vec![0.0f32; rows.saturating_mul(width)];
                dataset.read_range_into(chunk_start as u32, rows, &mut point_chunk)?;
                start_summary.merge(SketchBuildStartSummary::from_chunk(
                    chunk_start,
                    rows,
                    dim,
                    &point_chunk,
                    start_sample_stride,
                ));

                for (offset, row) in chunk_data.chunks_exact_mut(width).enumerate() {
                    let point = &point_chunk[offset * dim..(offset + 1) * dim];
                    fill_sketch_row_from_slice(point, &hyperplanes, row)?;
                }

                if let Some(writer) = out.as_mut() {
                    for value in &chunk_data {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                } else if let Some(writer) = direct_writer.as_ref() {
                    let byte_offset = HEADER_BYTES as u64
                        + (chunk_start as u64)
                            .saturating_mul(width as u64)
                            .saturating_mul(size_of::<f32>() as u64);
                    writer.write_all_at(f32_slice_as_bytes(&chunk_data), byte_offset)?;
                } else {
                    unreachable!("sketch writer must be initialized");
                }
            }
            if let Some(mut writer) = out {
                writer.flush()?;
            }
            if let Some(writer) = direct_writer {
                writer.set_len(expected_bytes as u64)?;
                writer.sync_all()?;
            }
        } else {
            let chunk_starts: Vec<usize> = (0..num_points).step_by(chunk_rows).collect();
            let hyperplanes = Arc::new(hyperplanes);

            if io_cfg.enabled {
                let writer = DirectIoFile::create_write(path, io_cfg)?;
                writer.write_all_at(&(num_points as u64).to_le_bytes(), 0)?;
                writer.write_all_at(&(width as u64).to_le_bytes(), size_of::<u64>() as u64)?;
                let expected_bytes = HEADER_BYTES.saturating_add(
                    num_points
                        .saturating_mul(width)
                        .saturating_mul(size_of::<f32>()),
                );

                let chunk_summaries = with_rayon_thread_pool(sketch_threads as u32, || {
                    chunk_starts
                        .par_iter()
                        .map(|&chunk_start| -> AnnResult<SketchBuildStartSummary> {
                            let rows = (num_points - chunk_start).min(chunk_rows);
                            let mut point_chunk = vec![0.0f32; rows.saturating_mul(dim)];
                            let mut chunk_data = vec![0.0f32; rows.saturating_mul(width)];
                            dataset.read_range_into(chunk_start as u32, rows, &mut point_chunk)?;
                            let summary = SketchBuildStartSummary::from_chunk(
                                chunk_start,
                                rows,
                                dim,
                                &point_chunk,
                                start_sample_stride,
                            );

                            for (offset, row) in chunk_data.chunks_exact_mut(width).enumerate() {
                                let point = &point_chunk[offset * dim..(offset + 1) * dim];
                                fill_sketch_row_from_slice(point, hyperplanes.as_slice(), row)?;
                            }

                            let byte_offset = HEADER_BYTES as u64
                                + (chunk_start as u64)
                                    .saturating_mul(width as u64)
                                    .saturating_mul(size_of::<f32>() as u64);
                            writer.write_all_at(f32_slice_as_bytes(&chunk_data), byte_offset)?;
                            Ok(summary)
                        })
                        .collect::<AnnResult<Vec<_>>>()
                })??;
                for summary in chunk_summaries {
                    start_summary.merge(summary);
                }
                writer.set_len(expected_bytes as u64)?;
                writer.sync_all()?;
            } else {
                let chunk_results = with_rayon_thread_pool(sketch_threads as u32, || {
                    chunk_starts
                        .par_iter()
                        .map(|&chunk_start| -> AnnResult<(
                            usize,
                            Vec<f32>,
                            SketchBuildStartSummary,
                        )> {
                            let rows = (num_points - chunk_start).min(chunk_rows);
                            let mut point_chunk = vec![0.0f32; rows.saturating_mul(dim)];
                            let mut chunk_data = vec![0.0f32; rows.saturating_mul(width)];
                            dataset.read_range_into(chunk_start as u32, rows, &mut point_chunk)?;
                            let summary = SketchBuildStartSummary::from_chunk(
                                chunk_start,
                                rows,
                                dim,
                                &point_chunk,
                                start_sample_stride,
                            );

                            for (offset, row) in chunk_data.chunks_exact_mut(width).enumerate() {
                                let point = &point_chunk[offset * dim..(offset + 1) * dim];
                                fill_sketch_row_from_slice(point, hyperplanes.as_slice(), row)?;
                            }

                            Ok((chunk_start, chunk_data, summary))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })??;

                let mut chunk_results = chunk_results;
                chunk_results.sort_unstable_by_key(|(chunk_start, _, _)| *chunk_start);
                let mut writer = BufWriter::new(File::create(path)?);
                writer.write_all(&(num_points as u64).to_le_bytes())?;
                writer.write_all(&(width as u64).to_le_bytes())?;
                for (_, chunk_data, summary) in chunk_results {
                    start_summary.merge(summary);
                    for value in &chunk_data {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                writer.flush()?;
            }
        }

        let sampled_medoid_start = start_summary.sampled_medoid_start();
        tracing::info!(
            "[disk-sketch-store/start-candidate] rows={} dim={} sampled_points={} sample_stride={} sampled_medoid_start={:?}",
            start_summary.rows,
            dim,
            start_summary.sample_count(),
            start_sample_stride,
            sampled_medoid_start,
        );
        let mut store = Self::open_with_config(path, io_cfg)?;
        store.sampled_medoid_start = sampled_medoid_start;
        Ok(store)
    }

    pub fn sampled_medoid_start(&self) -> Option<u32> {
        self.sampled_medoid_start
    }

    pub fn open(path: &Path) -> AnnResult<Self> {
        Self::open_with_config(path, DirectIoConfig::disabled())
    }

    pub fn open_with_config(path: &Path, io_cfg: DirectIoConfig) -> AnnResult<Self> {
        let mut file = File::open(path)?;
        let mut header = [0u8; HEADER_BYTES];
        file.read_exact(&mut header)?;
        let rows = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let width = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;

        let file_bytes = file.metadata()?.len() as usize;
        let expected_bytes = HEADER_BYTES
            .checked_add(
                rows.checked_mul(width)
                    .and_then(|values| values.checked_mul(size_of::<f32>()))
                    .ok_or_else(|| {
                        AnnError::log_index_error("Sketch file size overflow".to_string())
                    })?,
            )
            .ok_or_else(|| AnnError::log_index_error("Sketch file size overflow".to_string()))?;

        if file_bytes != expected_bytes {
            return Err(AnnError::log_index_error(format!(
                "Sketch file size mismatch: expected {expected_bytes}, found {file_bytes}"
            )));
        }

        let backing = if io_cfg.enabled {
            SketchBacking::Direct {
                file: DirectIoFile::open_read(path, io_cfg)?,
            }
        } else {
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let _ = mmap.advise(Advice::Sequential);
            SketchBacking::Mmap { _file: file, mmap }
        };

        Ok(Self {
            path: path.to_path_buf(),
            backing,
            rows,
            width,
            sampled_medoid_start: None,
        })
    }

    fn row_range(&self, idx: usize) -> AnnResult<(usize, usize)> {
        if idx >= self.rows {
            return Err(AnnError::log_index_error(format!(
                "Sketch row {idx} is out of bounds for {} rows",
                self.rows
            )));
        }

        let row_bytes = self
            .width
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| AnnError::log_index_error("Sketch row size overflow".to_string()))?;
        let start = HEADER_BYTES
            .checked_add(idx.checked_mul(row_bytes).ok_or_else(|| {
                AnnError::log_index_error("Sketch row offset overflow".to_string())
            })?)
            .ok_or_else(|| AnnError::log_index_error("Sketch row offset overflow".to_string()))?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| AnnError::log_index_error("Sketch row end overflow".to_string()))?;

        Ok((start, end))
    }

    fn read_sketch_range_into(&self, start: usize, count: usize, out: &mut [f32]) -> AnnResult<()> {
        let row_bytes = self.width * size_of::<f32>();
        let byte_offset = HEADER_BYTES + start * row_bytes;
        let byte_len = count * row_bytes;
        match &self.backing {
            SketchBacking::Mmap { mmap, .. } => {
                let bytes = &mmap[byte_offset..byte_offset + byte_len];
                for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(size_of::<f32>())) {
                    *slot = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            SketchBacking::Direct { file } => {
                let mut bytes = vec![0u8; byte_len];
                file.read_exact_at(&mut bytes, byte_offset as u64)?;
                for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(size_of::<f32>())) {
                    *slot = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
        }
        Ok(())
    }

    pub fn row(&self, idx: usize) -> AnnResult<Vec<f32>> {
        let mut row = vec![0.0f32; self.width];
        self.row_copy_into(idx, &mut row)?;
        Ok(row)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    pub fn file_bytes(&self) -> usize {
        std::fs::metadata(&self.path)
            .map(|metadata| metadata.len() as usize)
            .unwrap_or(0)
    }

    #[cfg(test)]
    fn uses_mmap(&self) -> bool {
        matches!(self.backing, SketchBacking::Mmap { .. })
    }
}

pub(crate) struct ResidentPrefixSketchAccessor<'a> {
    prefix: SketchStore,
    backing: &'a DiskSketchStore,
}

impl<'a> ResidentPrefixSketchAccessor<'a> {
    pub(crate) fn from_disk_prefix(
        backing: &'a DiskSketchStore,
        prefix_rows: usize,
    ) -> AnnResult<Self> {
        if prefix_rows > backing.rows {
            return Err(AnnError::log_index_error(format!(
                "Resident sketch prefix rows out of bounds: prefix_rows={} rows={}",
                prefix_rows, backing.rows
            )));
        }

        let width = backing.width;
        let mut data = vec![0.0f32; prefix_rows.saturating_mul(width)];
        if prefix_rows > 0 {
            backing.read_sketch_range_into(0, prefix_rows, &mut data)?;
        }

        Ok(Self {
            prefix: SketchStore::from_row_major(data, width),
            backing,
        })
    }

    #[inline]
    fn prefix_rows(&self) -> usize {
        self.prefix.rows()
    }

    fn scatter_subset_rows(
        subset_positions: &[usize],
        subset_out: &[f32],
        row_width: usize,
        out: &mut [f32],
    ) {
        for (subset_idx, &out_row) in subset_positions.iter().enumerate() {
            let src_start = subset_idx * row_width;
            let src_end = src_start + row_width;
            let dst_start = out_row * row_width;
            let dst_end = dst_start + row_width;
            out[dst_start..dst_end].copy_from_slice(&subset_out[src_start..src_end]);
        }
    }
}

fn record_bounded_sketch_plan_stats(
    plan: &BoundedReadPlan,
    row_bytes: usize,
    stats: &mut WindowedGatherStats,
) {
    stats.rows_requested += plan.requested_rows() as u64;
    stats.logical_bytes += plan.logical_bytes();
    stats.physical_bytes += plan.physical_bytes();
    stats.windows_submitted += plan.windows.len() as u64;
    stats.scatter_ops += plan.scatter.len() as u64;
    stats.overread_rows += plan.physical_rows().saturating_sub(plan.requested_rows()) as u64;
    let row_span_bytes: u64 = plan
        .windows
        .iter()
        .map(|window| window.row_count as u64 * row_bytes as u64)
        .sum();
    stats.alignment_waste_bytes += plan.physical_bytes().saturating_sub(row_span_bytes);
    for window in &plan.windows {
        stats.rows_per_window_sum += window.row_count as u64;
        if window.row_count <= 1 {
            stats.singleton_windows += 1;
        } else {
            stats.range_windows += 1;
            stats.range_rows_read += window.row_count as u64;
        }
    }
}

fn bounded_sketch_scatter_ranges(plan: &BoundedReadPlan) -> Vec<(usize, usize)> {
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

fn scatter_bounded_sketch_window(
    plan: &BoundedReadPlan,
    scatter_start: usize,
    scatter_end: usize,
    row_width: usize,
    window_slice: &[f32],
    out: &mut [f32],
) {
    for scatter in &plan.scatter[scatter_start..scatter_end] {
        let src_row = scatter.row_offset_in_window as usize;
        let src_start = src_row * row_width;
        let src_end = src_start + row_width;
        let dst_start = scatter.original_pos * row_width;
        let dst_end = dst_start + row_width;
        out[dst_start..dst_end].copy_from_slice(&window_slice[src_start..src_end]);
    }
}

fn keep_resident_prefix_logical_stats_only(stats: &mut WindowedGatherStats) {
    stats.windows_submitted = 0;
    stats.rows_per_window_sum = 0;
    stats.singleton_windows = 0;
    stats.range_windows = 0;
    stats.range_rows_read = 0;
    stats.physical_bytes = 0;
    stats.alignment_waste_bytes = 0;
    stats.overread_rows = 0;
}

struct SketchUringWindowRead {
    window_idx: usize,
    prefix: usize,
    logical_len: usize,
    scratch: AlignedBoxWithSlice<u8>,
}

fn try_read_bounded_sketch_windows_with_uring(
    fd: RawFd,
    plan: &BoundedReadPlan,
    row_width: usize,
    row_bytes: usize,
    out: &mut [f32],
    alignment: usize,
) -> io::Result<bool> {
    if plan.windows.len() < SKETCH_URING_WINDOW_GATHER_MIN_WINDOWS {
        return Ok(false);
    }

    let scatter_ranges = bounded_sketch_scatter_ranges(plan);
    let alignment = alignment.max(1);
    let qd = SKETCH_URING_WINDOW_GATHER_QUEUE_DEPTH
        .min(plan.windows.len())
        .max(1);
    SKETCH_URING_GATHER_RING.with(|ring_cell| -> io::Result<bool> {
        let mut ring_ref = ring_cell.borrow_mut();
        if ring_ref.is_none() {
            *ring_ref = Some(IoUring::new(SKETCH_URING_WINDOW_GATHER_QUEUE_DEPTH as u32)?);
        }
        let ring = ring_ref
            .as_mut()
            .expect("thread-local sketch io_uring must be initialized");

        let mut next_window = 0usize;
        while next_window < plan.windows.len() {
            let batch_end = (next_window + qd).min(plan.windows.len());
            let mut reads = Vec::with_capacity(batch_end - next_window);
            for window_idx in next_window..batch_end {
                let window = plan.windows[window_idx];
                let logical_offset = HEADER_BYTES as u64
                    + (window.start_row as u64).saturating_mul(row_bytes as u64);
                let logical_len = window.row_count as usize * row_bytes;
                let alignment_u64 = alignment as u64;
                let aligned_offset = logical_offset / alignment_u64 * alignment_u64;
                let prefix = (logical_offset - aligned_offset) as usize;
                let aligned_len =
                    prefix.saturating_add(logical_len).div_ceil(alignment) * alignment;
                let scratch = AlignedBoxWithSlice::<u8>::new(aligned_len, alignment)
                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;

                reads.push(SketchUringWindowRead {
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
                            "sketch io_uring submission queue is full",
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
                            format!("io_uring returned invalid sketch window slot {slot}"),
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
                                "io_uring sketch short read: got {got}, expected at least {}",
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
                        read.logical_len / size_of::<f32>(),
                    )
                };
                let (scatter_start, scatter_end) = scatter_ranges[read.window_idx];
                scatter_bounded_sketch_window(
                    plan,
                    scatter_start,
                    scatter_end,
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

impl SketchAccessor for ResidentPrefixSketchAccessor<'_> {
    fn width(&self) -> usize {
        self.backing.width()
    }

    fn rows(&self) -> usize {
        self.backing.rows()
    }

    fn row_copy_into(&self, idx: usize, dst: &mut [f32]) -> AnnResult<()> {
        if idx < self.prefix_rows() {
            self.prefix.row_copy_into(idx, dst)
        } else {
            self.backing.row_copy_into(idx, dst)
        }
    }

    fn read_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let width = self.width();
        if out.len() != ids.len().saturating_mul(width) {
            return Err(AnnError::log_index_error(format!(
                "Resident prefix sketch matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(width)
            )));
        }

        for (row, &id) in ids.iter().enumerate() {
            let start = row * width;
            let end = start + width;
            self.row_copy_into(id as usize, &mut out[start..end])?;
        }
        Ok(())
    }

    fn read_rows_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        let width = self.width();
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(width) {
            return Err(AnnError::log_index_error(format!(
                "Resident prefix sketch matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(width)
            )));
        }

        let prefix_rows = self.prefix_rows() as u32;
        if prefix_rows == 0 {
            return self
                .backing
                .read_rows_windowed_into_stats(ids, out, options, stats);
        }

        let mut prefix_ids = Vec::new();
        let mut prefix_positions = Vec::new();
        let mut tail_ids = Vec::new();
        let mut tail_positions = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            if id < prefix_rows {
                prefix_ids.push(id);
                prefix_positions.push(pos);
            } else {
                tail_ids.push(id);
                tail_positions.push(pos);
            }
        }

        if !prefix_ids.is_empty() {
            let mut prefix_out = vec![0.0f32; prefix_ids.len() * width];
            let mut prefix_stats = WindowedGatherStats::default();
            self.prefix.read_rows_windowed_into_stats(
                &prefix_ids,
                &mut prefix_out,
                options,
                &mut prefix_stats,
            )?;
            keep_resident_prefix_logical_stats_only(&mut prefix_stats);
            Self::scatter_subset_rows(&prefix_positions, &prefix_out, width, out);
            stats.merge(&prefix_stats);
        }

        if !tail_ids.is_empty() {
            let mut tail_out = vec![0.0f32; tail_ids.len() * width];
            let mut tail_stats = WindowedGatherStats::default();
            self.backing.read_rows_windowed_into_stats(
                &tail_ids,
                &mut tail_out,
                options,
                &mut tail_stats,
            )?;
            Self::scatter_subset_rows(&tail_positions, &tail_out, width, out);
            stats.merge(&tail_stats);
        }

        Ok(())
    }

    fn read_rows_bounded_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        max_window_bytes: usize,
        max_read_amplification: f64,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        let width = self.width();
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(width) {
            return Err(AnnError::log_index_error(format!(
                "Resident prefix sketch matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(width)
            )));
        }

        let prefix_rows = self.prefix_rows() as u32;
        if prefix_rows == 0 {
            return self.backing.read_rows_bounded_into_stats(
                ids,
                out,
                max_window_bytes,
                max_read_amplification,
                stats,
            );
        }

        let mut prefix_ids = Vec::new();
        let mut prefix_positions = Vec::new();
        let mut tail_ids = Vec::new();
        let mut tail_positions = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            if id < prefix_rows {
                prefix_ids.push(id);
                prefix_positions.push(pos);
            } else {
                tail_ids.push(id);
                tail_positions.push(pos);
            }
        }

        if !prefix_ids.is_empty() {
            let mut prefix_out = vec![0.0f32; prefix_ids.len() * width];
            let mut prefix_stats = WindowedGatherStats::default();
            self.prefix.read_rows_bounded_into_stats(
                &prefix_ids,
                &mut prefix_out,
                max_window_bytes,
                max_read_amplification,
                &mut prefix_stats,
            )?;
            keep_resident_prefix_logical_stats_only(&mut prefix_stats);
            Self::scatter_subset_rows(&prefix_positions, &prefix_out, width, out);
            stats.merge(&prefix_stats);
        }

        if !tail_ids.is_empty() {
            let mut tail_out = vec![0.0f32; tail_ids.len() * width];
            let mut tail_stats = WindowedGatherStats::default();
            self.backing.read_rows_bounded_into_stats(
                &tail_ids,
                &mut tail_out,
                max_window_bytes,
                max_read_amplification,
                &mut tail_stats,
            )?;
            Self::scatter_subset_rows(&tail_positions, &tail_out, width, out);
            stats.merge(&tail_stats);
        }

        Ok(())
    }

    fn resident_bytes(&self) -> usize {
        size_of::<Self>() + self.prefix.resident_bytes()
    }
}

fn sketch_build_bytes_per_row(width: usize, dim: usize) -> usize {
    width
        .saturating_mul(size_of::<f32>())
        .saturating_add(dim.saturating_mul(size_of::<f32>()))
        .max(1)
}

fn requested_sketch_threads(requested_threads: u32) -> usize {
    let requested = if requested_threads == 0 {
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
    } else {
        requested_threads as usize
    };
    requested.max(1)
}

fn sketch_build_memory_thread_limit(params: &ForgeANNParams, dim: usize, width: usize) -> usize {
    let budget_bytes = params.effective_oom_memory_budget_bytes();
    let per_worker_bytes = dim
        .saturating_mul(width)
        .saturating_mul(size_of::<f32>())
        .saturating_add(dim.saturating_mul(size_of::<f32>()))
        .saturating_add(4 * 1024 * 1024)
        .max(1);
    (budget_bytes / per_worker_bytes).max(1)
}

fn fill_sketch_row_from_slice(
    point: &[f32],
    hyperplanes: &[f32],
    out_row: &mut [f32],
) -> AnnResult<()> {
    let dim = point.len();
    if hyperplanes.len() != out_row.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Hyperplane shape mismatch: hyperplanes={} out_row={} dim={}",
            hyperplanes.len(),
            out_row.len(),
            dim
        )));
    }

    for (h_idx, h) in hyperplanes.chunks_exact(dim).enumerate() {
        let mut acc = 0.0f32;
        for (a, b) in point.iter().zip(h.iter()) {
            acc += *a * *b;
        }
        out_row[h_idx] = acc;
    }
    Ok(())
}

fn f32_slice_as_bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

impl SketchAccessor for DiskSketchStore {
    fn width(&self) -> usize {
        self.width
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn row_copy_into(&self, idx: usize, dst: &mut [f32]) -> AnnResult<()> {
        if dst.len() != self.width {
            return Err(AnnError::log_index_error(format!(
                "Sketch row copy width mismatch: dst={} expected={}",
                dst.len(),
                self.width
            )));
        }

        let (start, end) = self.row_range(idx)?;
        match &self.backing {
            SketchBacking::Mmap { mmap, .. } => {
                let bytes = mmap.get(start..end).ok_or_else(|| {
                    AnnError::log_index_error(format!(
                        "Sketch row {idx} byte range is out of bounds for mapped file"
                    ))
                })?;
                for (slot, chunk) in dst.iter_mut().zip(bytes.chunks_exact(size_of::<f32>())) {
                    *slot = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            SketchBacking::Direct { file } => {
                let mut bytes = vec![0u8; end - start];
                file.read_exact_at(&mut bytes, start as u64)?;
                for (slot, chunk) in dst.iter_mut().zip(bytes.chunks_exact(size_of::<f32>())) {
                    *slot = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
        }
        Ok(())
    }

    fn read_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let w = self.width();
        if ids.is_empty() {
            return Ok(());
        }
        // Coalesce contiguous runs of consecutive IDs into range reads
        let mut run_start = 0usize;
        while run_start < ids.len() {
            let mut run_end = run_start + 1;
            while run_end < ids.len() && ids[run_end] == ids[run_end - 1] + 1 {
                run_end += 1;
            }
            let run_len = run_end - run_start;
            let start_idx = ids[run_start] as usize;
            self.read_sketch_range_into(start_idx, run_len, &mut out[run_start * w..run_end * w])?;
            run_start = run_end;
        }
        Ok(())
    }

    fn read_rows_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        let width = self.width();
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(width) {
            return Err(AnnError::log_index_error(format!(
                "Sketch matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(width)
            )));
        }

        let row_bytes = width * size_of::<f32>();
        let plan = build_gather_plan(ids, row_bytes, *options, stats);
        let mut window_buf = vec![
            0.0f32;
            plan.windows
                .iter()
                .map(|window| window.row_count as usize)
                .max()
                .unwrap_or(0)
                * width
        ];
        for (window_idx, window) in plan.windows.iter().enumerate() {
            let rows = window.row_count as usize;
            let elems = rows * width;
            let window_slice = &mut window_buf[..elems];
            self.read_sketch_range_into(window.start_row as usize, rows, window_slice)?;
            scatter_window_rows(&plan, window_idx, width, window_slice, out);
        }
        Ok(())
    }

    fn read_rows_bounded_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        max_window_bytes: usize,
        max_read_amplification: f64,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        let width = self.width();
        if ids.is_empty() {
            return Ok(());
        }
        if out.len() != ids.len().saturating_mul(width) {
            return Err(AnnError::log_index_error(format!(
                "Sketch matrix size mismatch: got {} expected {}",
                out.len(),
                ids.len().saturating_mul(width)
            )));
        }

        let row_bytes = width * size_of::<f32>();
        let planner = BoundedReadPlanner {
            row_bytes,
            max_window_bytes,
            max_read_amplification,
        };
        let plan = match &self.backing {
            SketchBacking::Direct { file } => planner.plan_with_aligned_read_cost(
                ids,
                file.config().effective_alignment(),
                HEADER_BYTES as u64,
            ),
            SketchBacking::Mmap { .. } => planner.plan(ids),
        };
        record_bounded_sketch_plan_stats(&plan, row_bytes, stats);

        if let SketchBacking::Direct { file } = &self.backing {
            if try_read_bounded_sketch_windows_with_uring(
                file.raw_fd(),
                &plan,
                width,
                row_bytes,
                out,
                file.config().effective_alignment(),
            )
            .map_err(AnnError::log_io_error)?
            {
                return Ok(());
            }
        }

        let mut window_buf = vec![
            0.0f32;
            plan.windows
                .iter()
                .map(|window| window.row_count as usize)
                .max()
                .unwrap_or(0)
                * width
        ];
        let mut scatter_cursor = 0usize;
        for (window_idx, window) in plan.windows.iter().enumerate() {
            let scatter_start = scatter_cursor;
            while scatter_cursor < plan.scatter.len()
                && plan.scatter[scatter_cursor].window_idx == window_idx
            {
                scatter_cursor += 1;
            }
            let rows = window.row_count as usize;
            let elems = rows * width;
            let window_slice = &mut window_buf[..elems];
            self.read_sketch_range_into(window.start_row as usize, rows, window_slice)?;
            scatter_bounded_sketch_window(
                &plan,
                scatter_start,
                scatter_cursor,
                width,
                window_slice,
                out,
            );
        }
        Ok(())
    }

    fn resident_bytes(&self) -> usize {
        size_of::<Self>()
            + match &self.backing {
                SketchBacking::Mmap { .. } => 0,
                SketchBacking::Direct { .. } => 0,
            }
    }
}

fn write_sketch_file_direct(
    path: &Path,
    rows: usize,
    width: usize,
    data: &[f32],
    io_cfg: DirectIoConfig,
) -> AnnResult<()> {
    if data.len() != rows.saturating_mul(width) {
        return Err(AnnError::log_index_error(format!(
            "Sketch payload size mismatch: rows={} width={} values={}",
            rows,
            width,
            data.len()
        )));
    }

    let file = DirectIoFile::create_write(path, io_cfg)?;
    let mut bytes = Vec::with_capacity(HEADER_BYTES + data.len().saturating_mul(size_of::<f32>()));
    bytes.extend_from_slice(&(rows as u64).to_le_bytes());
    bytes.extend_from_slice(&(width as u64).to_le_bytes());
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    file.write_all_at(&bytes, 0)?;
    file.set_len(bytes.len() as u64)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use tempfile::tempdir;

    use super::{DiskSketchStore, ResidentPrefixSketchAccessor};
    use crate::forgeann::ForgeANNParams;
    use crate::forgeann::direct_io::{
        DirectIoConfig, direct_io_test_write_calls, reset_direct_io_test_counters,
    };
    use crate::forgeann::hash_prune::{SketchAccessor, compute_sketches};
    use crate::forgeann::point_store::{
        DirectPointStore, PointStore, WindowedGatherOptions, WindowedGatherStats,
    };
    use crate::model::InmemDataset;
    use crate::utils::save_bin_f32;

    fn build_test_dataset(num_points: usize, dim: usize) -> InmemDataset<f32> {
        let data: Vec<f32> = (0..num_points)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.25 + col as f32 * 0.5))
            .collect();
        build_test_dataset_from_data(num_points, dim, &data)
    }

    fn build_test_dataset_from_data(
        num_points: usize,
        dim: usize,
        data: &[f32],
    ) -> InmemDataset<f32> {
        assert_eq!(data.len(), num_points * dim);
        let mut dataset = InmemDataset::new(num_points, 1.0, dim).unwrap();
        dataset.data.copy_from_slice(data);
        dataset
    }

    struct CountingRangePointStore {
        data: Vec<f32>,
        rows: usize,
        dim: usize,
        point_reads: Arc<AtomicUsize>,
        range_reads: Arc<AtomicUsize>,
    }

    impl CountingRangePointStore {
        fn new(
            rows: usize,
            dim: usize,
            data: Vec<f32>,
        ) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            let point_reads = Arc::new(AtomicUsize::new(0));
            let range_reads = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    data,
                    rows,
                    dim,
                    point_reads: Arc::clone(&point_reads),
                    range_reads: Arc::clone(&range_reads),
                },
                point_reads,
                range_reads,
            )
        }
    }

    struct BlockingRangePointStore {
        data: Vec<f32>,
        rows: usize,
        dim: usize,
        inflight_reads: Arc<AtomicUsize>,
        max_inflight_reads: Arc<AtomicUsize>,
    }

    impl BlockingRangePointStore {
        fn new(rows: usize, dim: usize, data: Vec<f32>) -> (Self, Arc<AtomicUsize>) {
            let inflight_reads = Arc::new(AtomicUsize::new(0));
            let max_inflight_reads = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    data,
                    rows,
                    dim,
                    inflight_reads,
                    max_inflight_reads: Arc::clone(&max_inflight_reads),
                },
                max_inflight_reads,
            )
        }
    }

    impl PointStore for BlockingRangePointStore {
        fn len(&self) -> usize {
            self.rows
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn read_point_into(&self, pid: u32, out: &mut [f32]) -> crate::common::AnnResult<()> {
            let point = pid as usize;
            let start = point * self.dim;
            let end = start + self.dim;
            out.copy_from_slice(&self.data[start..end]);
            Ok(())
        }

        fn read_range_into(
            &self,
            start_pid: u32,
            count: usize,
            out: &mut [f32],
        ) -> crate::common::AnnResult<()> {
            let inflight = self.inflight_reads.fetch_add(1, Ordering::AcqRel) + 1;
            let mut observed = self.max_inflight_reads.load(Ordering::Acquire);
            while inflight > observed {
                match self.max_inflight_reads.compare_exchange_weak(
                    observed,
                    inflight,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(next) => observed = next,
                }
            }

            thread::sleep(Duration::from_millis(25));
            let start = start_pid as usize * self.dim;
            let end = start + count * self.dim;
            out.copy_from_slice(&self.data[start..end]);
            self.inflight_reads.fetch_sub(1, Ordering::AcqRel);
            Ok(())
        }

        fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> crate::common::AnnResult<()> {
            for (row, &pid) in ids.iter().enumerate() {
                let start = row * self.dim;
                let end = start + self.dim;
                self.read_point_into(pid, &mut out[start..end])?;
            }
            Ok(())
        }
    }

    impl PointStore for CountingRangePointStore {
        fn len(&self) -> usize {
            self.rows
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn read_point_into(&self, pid: u32, out: &mut [f32]) -> crate::common::AnnResult<()> {
            self.point_reads.fetch_add(1, Ordering::Relaxed);
            let point = pid as usize;
            let start = point * self.dim;
            let end = start + self.dim;
            out.copy_from_slice(&self.data[start..end]);
            Ok(())
        }

        fn read_range_into(
            &self,
            start_pid: u32,
            count: usize,
            out: &mut [f32],
        ) -> crate::common::AnnResult<()> {
            self.range_reads.fetch_add(1, Ordering::Relaxed);
            let start = start_pid as usize * self.dim;
            let end = start + count * self.dim;
            out.copy_from_slice(&self.data[start..end]);
            Ok(())
        }

        fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> crate::common::AnnResult<()> {
            for (row, &pid) in ids.iter().enumerate() {
                let start = row * self.dim;
                let end = start + self.dim;
                self.read_point_into(pid, &mut out[start..end])?;
            }
            Ok(())
        }
    }

    #[test]
    fn disk_sketch_store_round_trips_rows_from_dataset() {
        let dataset = build_test_dataset(16, 8);
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 6;
        let sketches = compute_sketches(&dataset, 16, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches.bin");

        let store = DiskSketchStore::write_from_memory(&path, &sketches).unwrap();

        assert_eq!(store.width(), params.m_hash_bits);
        for point in 0..16 {
            let mut row = vec![0.0f32; params.m_hash_bits];
            store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, sketches.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_reports_file_backed_bytes() {
        let dataset = build_test_dataset(8, 4);
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 5;
        let sketches = compute_sketches(&dataset, 8, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches.bin");

        let store = DiskSketchStore::write_from_memory(&path, &sketches).unwrap();

        assert_eq!(store.path(), PathBuf::from(&path));
        assert!(store.file_bytes() >= 8 * 5 * std::mem::size_of::<f32>());
    }

    #[test]
    fn disk_sketch_store_reads_rows_without_full_clone() {
        let dataset = build_test_dataset(12, 6);
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 7;
        let expected = compute_sketches(&dataset, 12, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-direct.bin");

        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 12, &params, 1).unwrap();

        assert_eq!(store.rows(), 12);
        assert_eq!(store.width(), 7);
        assert!(
            store.resident_bytes() < store.file_bytes(),
            "disk sketch store should avoid cloning the full sketch matrix into heap memory"
        );

        for point in 0..12 {
            let mut row = vec![0.0f32; params.m_hash_bits];
            store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_builds_from_direct_point_store() {
        let rows = 12usize;
        let dim = 6usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.25 + col as f32 * 0.5))
            .collect();
        let dir = tempdir().unwrap();
        let dataset_path = dir.path().join("tiny.fbin");
        save_bin_f32(&dataset_path, &data, rows, dim, 0).unwrap();

        let store = DirectPointStore::open(&dataset_path).unwrap();
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 7;

        let sketch_path = dir.path().join("sketches-direct-store.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 1)
                .unwrap();

        let expected_dataset = build_test_dataset(rows, dim);
        let expected = compute_sketches(&expected_dataset, rows, &params, 1).unwrap();
        for point in 0..rows {
            let mut row = vec![0.0f32; params.m_hash_bits];
            disk_store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_parallel_build_matches_reference_sketches() {
        let rows = 257usize;
        let dim = 32usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.03125 + col as f32 * 0.125))
            .collect();
        let dir = tempdir().unwrap();
        let dataset_path = dir.path().join("parallel-tiny.fbin");
        save_bin_f32(&dataset_path, &data, rows, dim, 0).unwrap();

        let store = DirectPointStore::open(&dataset_path).unwrap();
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 11;
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 64 * 1024 * 1024;

        let sketch_path = dir.path().join("sketches-parallel-direct-store.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 4)
                .unwrap();

        let expected_dataset = build_test_dataset_from_data(rows, dim, &data);
        let expected = compute_sketches(&expected_dataset, rows, &params, 4).unwrap();
        for point in 0..rows {
            let mut row = vec![0.0f32; params.m_hash_bits];
            disk_store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_prefers_range_reads_when_building_from_point_store() {
        let rows = 129usize;
        let dim = 24usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.125 + col as f32 * 0.0625))
            .collect();
        let (store, point_reads, range_reads) =
            CountingRangePointStore::new(rows, dim, data.clone());
        let dir = tempdir().unwrap();
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 9;
        params.oom_memory_budget_bytes = 256 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 8 * 1024 * 1024;

        let sketch_path = dir.path().join("sketches-range-first.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 4)
                .unwrap();

        assert!(range_reads.load(Ordering::Relaxed) > 0);
        assert_eq!(point_reads.load(Ordering::Relaxed), 0);

        let expected_dataset = build_test_dataset_from_data(rows, dim, &data);
        let expected = compute_sketches(&expected_dataset, rows, &params, 4).unwrap();
        for point in 0..rows {
            let mut row = vec![0.0f32; params.m_hash_bits];
            disk_store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_derives_sampled_medoid_start_from_build_pass() {
        let rows = 9usize;
        let dim = 1usize;
        let data: Vec<f32> = (0..rows).map(|row| row as f32).collect();
        let (store, point_reads, range_reads) =
            CountingRangePointStore::new(rows, dim, data.clone());
        let dir = tempdir().unwrap();
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 3;
        params.oom_memory_budget_bytes = 64 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 64 * 1024 * 1024;

        let sketch_path = dir.path().join("sketches-with-start.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 1)
                .unwrap();

        assert_eq!(disk_store.sampled_medoid_start(), Some(4));
        assert_eq!(point_reads.load(Ordering::Relaxed), 0);
        assert_eq!(
            range_reads.load(Ordering::Relaxed),
            1,
            "start selection should reuse the sketch build range read, not add medoid scans"
        );
    }

    #[test]
    fn disk_sketch_store_can_overlap_range_reads_during_parallel_build() {
        let rows = 65_536usize;
        let dim = 64usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.015625 + col as f32 * 0.03125))
            .collect();
        let (store, max_inflight_reads) = BlockingRangePointStore::new(rows, dim, data);
        let dir = tempdir().unwrap();
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 11;
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 2
            * super::MIN_SKETCH_BUILD_CHUNK_ROWS
            * (dim + params.m_hash_bits)
            * std::mem::size_of::<f32>();

        let sketch_path = dir.path().join("sketches-overlap.bin");
        let _disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 4)
                .unwrap();

        assert!(
            max_inflight_reads.load(Ordering::Relaxed) > 1,
            "expected sketch build to overlap multiple range reads when parallel workers are available"
        );
    }

    #[test]
    fn sketch_build_plan_treats_oom_sketch_cache_as_global_parallel_budget() {
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 32 * 1024 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 4 * 1024 * 1024 * 1024;
        params.m_hash_bits = 12;

        let plan = super::SketchBuildPlan::for_params(&params, params.m_hash_bits, 768, 42);
        let min_chunk_bytes = super::MIN_SKETCH_BUILD_CHUNK_ROWS
            .saturating_mul((768 + params.m_hash_bits).saturating_mul(std::mem::size_of::<f32>()));

        assert_eq!(plan.threads, 42);
        assert!(
            plan.chunk_rows < 100_000,
            "chunk rows should be sized per concurrent worker, not as a 4GiB per-worker chunk"
        );
        assert!(
            plan.estimated_concurrent_bytes
                <= params
                    .oom_sketch_cache_bytes
                    .saturating_add(min_chunk_bytes),
            "estimated concurrent sketch scratch {} must fit within the global budget {} plus one minimum chunk {}",
            plan.estimated_concurrent_bytes,
            params.oom_sketch_cache_bytes,
            min_chunk_bytes
        );
    }

    #[test]
    fn disk_sketch_store_strict_single_thread_build_writes_chunks_directly() {
        let rows = 4096usize;
        let dim = 64usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.0625 + col as f32 * 0.015625))
            .collect();
        let dir = tempdir().unwrap();
        let dataset_path = dir.path().join("single-thread-strict.fbin");
        save_bin_f32(&dataset_path, &data, rows, dim, 0).unwrap();

        let store = DirectPointStore::open_with_config(
            &dataset_path,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 11;
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 256 * 1024;

        reset_direct_io_test_counters();
        let sketch_path = dir.path().join("sketches-single-thread-strict.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 4)
                .unwrap();

        assert!(!disk_store.uses_mmap());
        assert!(
            direct_io_test_write_calls() > 1,
            "strict single-thread sketch build should write each chunk directly instead of buffering the full sketch file"
        );
    }

    #[test]
    fn disk_sketch_store_parallel_strict_io_matches_reference_sketches() {
        let rows = 4096usize;
        let dim = 64usize;
        let data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.0625 + col as f32 * 0.015625))
            .collect();
        let dir = tempdir().unwrap();
        let dataset_path = dir.path().join("parallel-strict.fbin");
        save_bin_f32(&dataset_path, &data, rows, dim, 0).unwrap();

        let store = DirectPointStore::open_with_config(
            &dataset_path,
            DirectIoConfig::enabled_with_alignment(4096),
        )
        .unwrap();
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 11;
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.oom_sketch_cache_bytes = 256 * 1024;

        let sketch_path = dir.path().join("sketches-parallel-strict.bin");
        let disk_store =
            DiskSketchStore::build_from_point_store(&sketch_path, &store, rows, &params, 4)
                .unwrap();
        assert!(
            !disk_store.uses_mmap(),
            "strict sketch path must stay off mmap"
        );

        let expected_dataset = build_test_dataset_from_data(rows, dim, &data);
        let expected = compute_sketches(&expected_dataset, rows, &params, 4).unwrap();
        for point in [0usize, 7, 255, 1024, rows - 1] {
            let mut row = vec![0.0f32; params.m_hash_bits];
            disk_store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }
    }

    #[test]
    fn disk_sketch_store_strict_io_avoids_mmap_and_round_trips_rows() {
        let dataset = build_test_dataset(19, 9);
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 5;
        params.oom_enable = true;
        let expected = compute_sketches(&dataset, 19, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-strict.bin");

        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 19, &params, 1).unwrap();
        assert!(
            !store.uses_mmap(),
            "strict sketch path must not reopen the file via mmap"
        );

        for point in 0..19 {
            let mut row = vec![0.0f32; params.m_hash_bits];
            store.row_copy_into(point, &mut row).unwrap();
            assert_eq!(row, expected.row(point));
        }

        let reopened =
            DiskSketchStore::open_with_config(&path, DirectIoConfig::enabled_with_alignment(4096))
                .unwrap();
        assert!(!reopened.uses_mmap());
        let mut row = vec![0.0f32; params.m_hash_bits];
        reopened.row_copy_into(7, &mut row).unwrap();
        assert_eq!(row, expected.row(7));
    }

    #[test]
    fn disk_sketch_store_windowed_reads_preserve_logical_order_for_unsorted_ids() {
        let dataset = build_test_dataset(16, 8);
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 6;
        let expected = compute_sketches(&dataset, 16, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-windowed-unsorted.bin");
        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 16, &params, 1).unwrap();

        let ids = vec![9_u32, 2, 10, 3];
        let mut out = vec![0.0f32; ids.len() * params.m_hash_bits];
        let mut stats = WindowedGatherStats::default();
        let options = WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 64 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        };

        store
            .read_rows_windowed_into_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        assert_eq!(stats.windows_submitted, 2);
        for (row, &pid) in ids.iter().enumerate() {
            let start = row * params.m_hash_bits;
            let end = start + params.m_hash_bits;
            assert_eq!(&out[start..end], expected.row(pid as usize));
        }
    }

    #[test]
    fn disk_sketch_store_windowed_reads_merge_small_gaps_into_single_window() {
        let dataset = build_test_dataset(16, 8);
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 6;

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-windowed-gap.bin");
        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 16, &params, 1).unwrap();

        let ids = vec![4_u32, 6];
        let mut out = vec![0.0f32; ids.len() * params.m_hash_bits];
        let mut stats = WindowedGatherStats::default();
        let options = WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 64 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        };

        store
            .read_rows_windowed_into_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        assert_eq!(stats.windows_submitted, 1);
        assert_eq!(stats.overread_rows, 1);
    }

    #[test]
    fn disk_sketch_store_direct_bounded_reads_merge_rows_by_aligned_block() {
        let dataset = build_test_dataset(192, 8);
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 6;
        params.oom_enable = true;
        let expected = compute_sketches(&dataset, 192, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-direct-bounded-aligned.bin");
        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 192, &params, 1).unwrap();
        assert!(!store.uses_mmap());

        let ids = vec![120_u32, 2, 80];
        let mut out = vec![0.0f32; ids.len() * params.m_hash_bits];
        let mut stats = WindowedGatherStats::default();
        store
            .read_rows_bounded_into_stats(&ids, &mut out, 4096, 1.5, &mut stats)
            .unwrap();

        assert_eq!(stats.windows_submitted, 1);
        assert_eq!(stats.physical_bytes, 4096);
        assert!(stats.alignment_waste_bytes > 0);
        for (row, &pid) in ids.iter().enumerate() {
            let start = row * params.m_hash_bits;
            let end = start + params.m_hash_bits;
            assert_eq!(&out[start..end], expected.row(pid as usize));
        }
    }

    #[test]
    fn resident_prefix_sketch_accessor_preserves_logical_order_for_mixed_cached_and_tail_ids() {
        let dataset = build_test_dataset(16, 8);
        let mut params = ForgeANNParams::default();
        params.m_hash_bits = 6;
        let expected = compute_sketches(&dataset, 16, &params, 1).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("sketches-prefix-cache.bin");
        let store = DiskSketchStore::build_from_dataset(&path, &dataset, 16, &params, 1).unwrap();
        let cached = ResidentPrefixSketchAccessor::from_disk_prefix(&store, 8).unwrap();

        let ids = vec![9_u32, 2, 10, 3, 7, 12, 0];
        let mut out = vec![0.0f32; ids.len() * params.m_hash_bits];
        let mut stats = WindowedGatherStats::default();
        let options = WindowedGatherOptions {
            max_gap_rows: 1,
            max_window_bytes: 64 * 1024,
            alignment_bytes: 4096,
            sort_ids: true,
        };

        cached
            .read_rows_windowed_into_stats(&ids, &mut out, &options, &mut stats)
            .unwrap();

        for (row, &pid) in ids.iter().enumerate() {
            let start = row * params.m_hash_bits;
            let end = start + params.m_hash_bits;
            assert_eq!(&out[start..end], expected.row(pid as usize));
        }
    }
}
