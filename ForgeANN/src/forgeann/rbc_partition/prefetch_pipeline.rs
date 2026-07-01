#[cfg(test)]
use super::child_run_io::default_rbc_windowed_options;
use super::*;

pub(crate) struct PrefetchedPointBatch {
    pub ids: Vec<u32>,
    pub data: Vec<f32>,
    pub len: usize,
    pub io_stats: PointBatchStats,
    pub load: Duration,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub producer_wait: Duration,
    pub budget_wait: Duration,
    pub prefetch_permit_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefetchPipelineKind {
    Off,
    Local,
    Strict,
}

impl Default for PrefetchPipelineKind {
    fn default() -> Self {
        Self::Off
    }
}

impl PrefetchPipelineKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Local => "local",
            Self::Strict => "strict",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StrictPrefetchPipelineConfig {
    pub enabled: bool,
    pub queue_depth: usize,
    pub budget_bytes: usize,
    pub max_window_bytes: usize,
    pub max_gap_rows: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StrictPrefetchPipelineProfile {
    pub pipeline: PrefetchPipelineKind,
    pub queue_depth: usize,
    pub prefetch_budget_bytes: usize,
    pub prefetch_used_peak_bytes: usize,
    pub batches: usize,
    pub io_wall: Duration,
    pub consumer_wait: Duration,
    pub producer_wait: Duration,
    pub budget_wait: Duration,
    pub fallback_budget_exhausted: usize,
    pub range_reads: u64,
    pub point_reads: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
}

impl StrictPrefetchPipelineProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        if other.pipeline != PrefetchPipelineKind::Off {
            self.pipeline = other.pipeline;
        }
        self.queue_depth = self.queue_depth.max(other.queue_depth);
        self.prefetch_budget_bytes = self.prefetch_budget_bytes.max(other.prefetch_budget_bytes);
        self.prefetch_used_peak_bytes = self
            .prefetch_used_peak_bytes
            .max(other.prefetch_used_peak_bytes);
        self.batches += other.batches;
        self.io_wall += other.io_wall;
        self.consumer_wait += other.consumer_wait;
        self.producer_wait += other.producer_wait;
        self.budget_wait += other.budget_wait;
        self.fallback_budget_exhausted += other.fallback_budget_exhausted;
        self.range_reads += other.range_reads;
        self.point_reads += other.point_reads;
        self.logical_bytes += other.logical_bytes;
        self.physical_bytes += other.physical_bytes;
    }

    fn record_batch(&mut self, batch: &PrefetchedPointBatch) {
        self.batches += 1;
        self.io_wall += batch.load;
        self.producer_wait += batch.producer_wait;
        self.budget_wait += batch.budget_wait;
        self.range_reads += batch.io_stats.range_calls;
        self.point_reads += batch.io_stats.point_calls;
        self.logical_bytes += batch.logical_bytes;
        self.physical_bytes += batch.physical_bytes;
    }

    pub(crate) fn read_amplification(&self) -> f64 {
        if self.logical_bytes == 0 {
            0.0
        } else {
            self.physical_bytes as f64 / self.logical_bytes as f64
        }
    }

    pub(crate) fn avg_read_size_bytes(&self) -> u64 {
        let reads = self.range_reads + self.point_reads;
        if reads == 0 {
            0
        } else {
            self.physical_bytes / reads
        }
    }
}

#[derive(Debug)]
pub(crate) struct StrictPrefetchGate {
    pub capacity_bytes: usize,
    pub used_bytes: AtomicUsize,
    pub peak_bytes: AtomicUsize,
}

impl StrictPrefetchGate {
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            used_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
        }
    }

    fn acquire_blocking(&self, bytes: usize) -> Option<Duration> {
        if bytes == 0 {
            return Some(Duration::ZERO);
        }
        if bytes > self.capacity_bytes {
            return None;
        }
        let start = Instant::now();
        loop {
            let mut current = self.used_bytes.load(Ordering::Acquire);
            while current.saturating_add(bytes) <= self.capacity_bytes {
                let next = current + bytes;
                match self.used_bytes.compare_exchange_weak(
                    current,
                    next,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.record_peak(next);
                        return Some(start.elapsed());
                    }
                    Err(observed) => current = observed,
                }
            }
            std::thread::yield_now();
        }
    }

    fn release(&self, bytes: usize) {
        if bytes > 0 {
            self.used_bytes.fetch_sub(bytes, Ordering::Release);
        }
    }

    fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Acquire)
    }

    fn record_peak(&self, value: usize) {
        let mut current = self.peak_bytes.load(Ordering::Relaxed);
        while value > current {
            match self.peak_bytes.compare_exchange_weak(
                current,
                value,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }
}

/// 一个简单的栈上 Top-K 容器，用于避免堆分配。
/// 假设 K 不会非常大（例如 <= 32）。
pub(crate) struct StackTopK {
    // 存储 (dist, id)
    buf: [(f32, usize); 32],
    len: usize,
    k: usize,
}

impl StackTopK {
    pub(crate) fn new(k: usize) -> Self {
        let k = k.min(32);
        Self {
            buf: [(f32::MAX, 0); 32],
            len: 0,
            k,
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, dist: f32, id: usize) {
        if self.len < self.k {
            self.buf[self.len] = (dist, id);
            self.len += 1;
            let mut i = self.len - 1;
            while i > 0 {
                if self.buf[i].0 < self.buf[i - 1].0 {
                    self.buf.swap(i, i - 1);
                    i -= 1;
                } else {
                    break;
                }
            }
        } else if dist < self.buf[self.k - 1].0 {
            self.buf[self.k - 1] = (dist, id);
            let mut i = self.k - 1;
            while i > 0 {
                if self.buf[i].0 < self.buf[i - 1].0 {
                    self.buf.swap(i, i - 1);
                    i -= 1;
                } else {
                    break;
                }
            }
        }
    }

    #[inline]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &(f32, usize)> {
        self.buf[0..self.len].iter()
    }
}

pub(crate) fn choose_prefetch_batch_points_for_budget(
    point_tile_size: usize,
    dim: usize,
    num_workers: usize,
    memory_budget_bytes: Option<usize>,
) -> usize {
    pub(crate) const DEFAULT_PREFETCH_BYTES: usize = 64 * 1024 * 1024;
    pub(crate) const MIN_PREFETCH_BYTES: usize = 8 * 1024 * 1024;
    pub(crate) const MAX_PREFETCH_BYTES: usize = 1024 * 1024 * 1024;

    let row_bytes = dim.saturating_mul(size_of::<f32>()).max(1);
    let target_bytes = memory_budget_bytes
        .filter(|&budget| budget > 0)
        .map(|budget| (budget / 32).clamp(MIN_PREFETCH_BYTES, MAX_PREFETCH_BYTES))
        .unwrap_or(DEFAULT_PREFETCH_BYTES)
        .clamp(MIN_PREFETCH_BYTES, MAX_PREFETCH_BYTES);
    let max_points_by_bytes = (target_bytes / row_bytes).max(point_tile_size);
    let worker_scaled_points = point_tile_size
        .saturating_mul(num_workers.max(1))
        .max(point_tile_size);

    worker_scaled_points
        .min(max_points_by_bytes)
        .max(point_tile_size)
}

pub(crate) fn choose_spool_prefetch_workers(
    num_workers: usize,
    memory_budget_bytes: Option<usize>,
) -> usize {
    let num_workers = num_workers.max(1);
    if memory_budget_bytes.is_some() {
        num_workers
    } else {
        num_workers.saturating_mul(2).max(1)
    }
}

pub(crate) fn choose_compute_block_points_for_batch(
    point_tile_size: usize,
    batch_points: usize,
    num_workers: usize,
) -> usize {
    pub(crate) const MIN_COMPUTE_TILE: usize = 256;

    let point_tile_size = point_tile_size.max(1);
    let batch_points = batch_points.max(1);
    let num_workers = num_workers.max(1);
    let max_blocks_by_min_tile = batch_points.div_ceil(MIN_COMPUTE_TILE).max(1);
    let target_blocks = num_workers.min(max_blocks_by_min_tile);
    let target_tile = batch_points.div_ceil(target_blocks);

    target_tile.clamp(MIN_COMPUTE_TILE, point_tile_size)
}

pub(crate) fn choose_prefetch_queue_depth_for_budget(
    batch_points: usize,
    dim: usize,
    memory_budget_bytes: Option<usize>,
) -> usize {
    pub(crate) const MAX_PREFETCH_QUEUE_DEPTH: usize = 8;

    let Some(memory_budget_bytes) = memory_budget_bytes.filter(|&budget| budget > 0) else {
        return 1;
    };
    let batch_bytes = batch_points
        .max(1)
        .saturating_mul(dim.max(1))
        .saturating_mul(size_of::<f32>())
        .max(1);
    // Reserve a larger fraction of the available OOM budget for the queue so the prefetch
    // pipeline can stay ahead of compute instead of oscillating at depth 4.
    let queue_budget = memory_budget_bytes / 4;
    (queue_budget / batch_bytes).clamp(1, MAX_PREFETCH_QUEUE_DEPTH)
}

pub(crate) fn should_use_strict_prefetch_for_gemm_assignment(
    params: &ForgeANNParams,
    points: usize,
    leaders: usize,
    dim: usize,
    memory_budget_bytes: Option<usize>,
    num_workers: usize,
) -> bool {
    if !params.strict_oom_prefetch_pipeline_enabled() || points == 0 || leaders == 0 || dim == 0 {
        return false;
    }

    let (point_tile_size, _) =
        choose_gemm_tile_sizes_with_override_and_budget(leaders, dim, memory_budget_bytes);
    let batch_points = choose_prefetch_batch_points_for_budget(
        point_tile_size,
        dim,
        num_workers.max(1),
        memory_budget_bytes,
    );

    points > batch_points
}

pub(crate) fn prefetch_point_batch_windowed(
    dataset: &dyn PointStore,
    ids: &[u32],
    options: &WindowedGatherOptions,
) -> AnnResult<PrefetchedPointBatch> {
    let load_start = Instant::now();
    let mut io_stats = PointBatchStats::default();
    let mut data = vec![0.0f32; ids.len().saturating_mul(dataset.dim())];
    dataset.read_points_windowed_into_batch_stats(ids, &mut data, options, &mut io_stats)?;
    let logical_bytes = ids
        .len()
        .saturating_mul(dataset.dim())
        .saturating_mul(size_of::<f32>()) as u64;
    let physical_bytes = io_stats.bytes_read;
    Ok(PrefetchedPointBatch {
        ids: ids.to_vec(),
        data,
        len: ids.len(),
        io_stats,
        load: load_start.elapsed(),
        logical_bytes,
        physical_bytes,
        producer_wait: Duration::ZERO,
        budget_wait: Duration::ZERO,
        prefetch_permit_bytes: 0,
    })
}

pub(crate) fn strict_prefetch_windowed_options(
    dataset: &dyn PointStore,
    fallback: WindowedGatherOptions,
    config: StrictPrefetchPipelineConfig,
) -> WindowedGatherOptions {
    let row_bytes = dataset.dim().saturating_mul(size_of::<f32>()).max(1);
    let configured_gap_rows = config.max_gap_rows.min(u32::MAX as usize) as u32;
    WindowedGatherOptions {
        max_gap_rows: configured_gap_rows,
        max_window_bytes: config.max_window_bytes.max(row_bytes),
        alignment_bytes: fallback.alignment_bytes,
        sort_ids: fallback.sort_ids,
    }
}

pub(crate) fn for_each_prefetched_point_batch_profiled<F>(
    dataset: &dyn PointStore,
    ids: &[u32],
    batch_points: usize,
    queue_depth: usize,
    options: WindowedGatherOptions,
    strict_config: Option<StrictPrefetchPipelineConfig>,
    strict_gate: Option<&StrictPrefetchGate>,
    profile: Option<&mut StrictPrefetchPipelineProfile>,
    mut f: F,
) -> AnnResult<()>
where
    F: FnMut(PrefetchedPointBatch) -> AnnResult<()>,
{
    if ids.is_empty() {
        return Ok(());
    }

    if dataset.disables_prefetch_pipeline() {
        let mut local_profile = StrictPrefetchPipelineProfile {
            pipeline: PrefetchPipelineKind::Off,
            ..StrictPrefetchPipelineProfile::default()
        };
        let batch_points = batch_points.max(1);
        for chunk in ids.chunks(batch_points) {
            let batch = prefetch_point_batch_windowed(dataset, chunk, &options)?;
            local_profile.record_batch(&batch);
            f(batch)?;
        }
        if let Some(profile) = profile {
            profile.merge(local_profile);
        }
        return Ok(());
    }

    let strict_config = strict_config.filter(|config| config.enabled);
    let row_bytes = dataset.dim().saturating_mul(size_of::<f32>()).max(1);
    let mut batch_points = batch_points.max(1);
    if let Some(config) = strict_config.filter(|config| config.budget_bytes > 0) {
        batch_points = batch_points.min((config.budget_bytes / row_bytes).max(1));
    }
    let mut local_profile = StrictPrefetchPipelineProfile {
        pipeline: if strict_config.is_some() {
            PrefetchPipelineKind::Strict
        } else {
            PrefetchPipelineKind::Local
        },
        queue_depth: queue_depth.max(1),
        ..StrictPrefetchPipelineProfile::default()
    };
    let local_gate = (strict_gate.is_none())
        .then(|| strict_config.filter(|config| config.budget_bytes > 0))
        .flatten()
        .map(|config| StrictPrefetchGate::new(config.budget_bytes));
    let gate_ref = strict_gate.or(local_gate.as_ref());
    if let Some(config) = strict_config {
        local_profile.queue_depth = config.queue_depth.max(1);
        local_profile.prefetch_budget_bytes = config.budget_bytes;
        if config.budget_bytes > 0 && config.budget_bytes < row_bytes {
            local_profile.fallback_budget_exhausted += 1;
        }
    }
    let effective_queue_depth = local_profile.queue_depth.max(1);
    let read_options = strict_config
        .map(|config| strict_prefetch_windowed_options(dataset, options, config))
        .unwrap_or(options);

    let (tx, rx) = mpsc::sync_channel::<AnnResult<PrefetchedPointBatch>>(effective_queue_depth);
    std::thread::scope(|scope| -> AnnResult<()> {
        let producer_gate_ref = gate_ref;
        let consumer_gate_ref = gate_ref;
        scope.spawn(move || {
            for chunk in ids.chunks(batch_points) {
                let batch_bytes = chunk
                    .len()
                    .saturating_mul(dataset.dim())
                    .saturating_mul(size_of::<f32>());
                let mut budget_wait = Duration::ZERO;
                let mut permit_bytes = 0usize;
                if let Some(gate) = producer_gate_ref {
                    match gate.acquire_blocking(batch_bytes) {
                        Some(wait) => {
                            budget_wait = wait;
                            permit_bytes = batch_bytes;
                        }
                        None => {
                            permit_bytes = 0;
                        }
                    }
                }

                let mut batch = prefetch_point_batch_windowed(dataset, chunk, &read_options);
                if let Ok(batch) = &mut batch {
                    batch.budget_wait = budget_wait;
                    batch.prefetch_permit_bytes = permit_bytes;
                } else if let Some(gate) = producer_gate_ref {
                    gate.release(permit_bytes);
                    permit_bytes = 0;
                }
                let send_start = Instant::now();
                let mut batch = batch;
                loop {
                    if let Ok(batch) = &mut batch {
                        batch.producer_wait = send_start.elapsed();
                    }
                    match tx.try_send(batch) {
                        Ok(()) => break,
                        Err(mpsc::TrySendError::Full(returned)) => {
                            batch = returned;
                            std::thread::yield_now();
                        }
                        Err(mpsc::TrySendError::Disconnected(returned)) => {
                            if let Some(gate) = producer_gate_ref {
                                match returned {
                                    Ok(batch) => gate.release(batch.prefetch_permit_bytes),
                                    Err(_) => gate.release(permit_bytes),
                                }
                            }
                            return;
                        }
                    }
                }
            }
        });

        for _ in ids.chunks(batch_points) {
            let recv_start = Instant::now();
            let batch = rx.recv().map_err(|err| {
                crate::common::AnnError::log_index_error(format!(
                    "Prefetch worker terminated before delivering RBC point batch: {err}"
                ))
            })??;
            local_profile.consumer_wait += recv_start.elapsed();
            local_profile.record_batch(&batch);
            let permit_bytes = batch.prefetch_permit_bytes;
            let result = f(batch);
            if let Some(gate) = consumer_gate_ref {
                gate.release(permit_bytes);
            }
            result?;
        }
        Ok(())
    })?;

    if let Some(gate) = gate_ref {
        local_profile.prefetch_used_peak_bytes = gate.peak_bytes();
    }
    if strict_config.is_some() && gate_ref.is_none() {
        local_profile.fallback_budget_exhausted = 1;
    }
    if let Some(profile) = profile {
        profile.merge(local_profile);
    }
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn for_each_prefetched_point_batch<F>(
    dataset: &dyn PointStore,
    ids: &[u32],
    batch_points: usize,
    queue_depth: usize,
    options: WindowedGatherOptions,
    f: F,
) -> AnnResult<()>
where
    F: FnMut(PrefetchedPointBatch) -> AnnResult<()>,
{
    for_each_prefetched_point_batch_profiled(
        dataset,
        ids,
        batch_points,
        queue_depth,
        options,
        None,
        None,
        None,
        f,
    )
}

#[cfg(test)]
mod strict_prefetch_pipeline_tests {
    use std::io::Write;
    use std::mem::size_of;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    use tempfile::tempdir;

    use super::PointStore;
    use crate::common::{AnnResult, Metric};
    use crate::forgeann::point_store::{DirectPointStore, PointBatchStats, WindowedGatherOptions};

    pub fn write_test_fbin(path: &std::path::Path, rows: usize, dim: usize) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(rows as u32).to_le_bytes()).unwrap();
        file.write_all(&(dim as u32).to_le_bytes()).unwrap();
        for row in 0..rows {
            for col in 0..dim {
                let value = row as f32 * 1000.0 + col as f32;
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    pub fn strict_prefetch_profile_records_waits_and_budget() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("prefetch-rbc-strict-profile.fbin");
        write_test_fbin(&path, 32, 8);
        let store = DirectPointStore::open(&path).unwrap();
        let ids = vec![8_u32, 2, 3, 7, 15, 16];
        let config = super::StrictPrefetchPipelineConfig {
            enabled: true,
            queue_depth: 3,
            budget_bytes: 2 * store.dim() * std::mem::size_of::<f32>(),
            max_window_bytes: 64 * 1024,
            max_gap_rows: 2,
        };
        let mut profile = super::StrictPrefetchPipelineProfile::default();
        let mut seen_batches = Vec::new();

        super::for_each_prefetched_point_batch_profiled(
            &store,
            &ids,
            2,
            1,
            super::default_rbc_windowed_options(&store, 2),
            Some(config),
            None,
            Some(&mut profile),
            |batch| {
                seen_batches.push(batch.ids.clone());
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(seen_batches, vec![vec![8, 2], vec![3, 7], vec![15, 16]]);
        assert_eq!(profile.pipeline, super::PrefetchPipelineKind::Strict);
        assert_eq!(profile.queue_depth, 3);
        assert_eq!(profile.batches, 3);
        assert!(profile.io_wall > Duration::ZERO);
        assert!(profile.consumer_wait > Duration::ZERO);
        assert!(profile.prefetch_used_peak_bytes <= config.budget_bytes);
        assert!(
            profile.logical_bytes >= (ids.len() * store.dim() * std::mem::size_of::<f32>()) as u64
        );
        assert!(profile.physical_bytes >= profile.logical_bytes);
    }

    #[test]
    pub fn prefetch_disabled_store_forces_pipeline_off() {
        struct NoPrefetchStore {
            rows: usize,
            dim: usize,
        }

        impl PointStore for NoPrefetchStore {
            fn len(&self) -> usize {
                self.rows
            }

            fn dim(&self) -> usize {
                self.dim
            }

            fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
                for (axis, value) in out.iter_mut().enumerate() {
                    *value = pid as f32 * 10.0 + axis as f32;
                }
                Ok(())
            }

            fn read_range_into(
                &self,
                start_pid: u32,
                count: usize,
                out: &mut [f32],
            ) -> AnnResult<()> {
                for row in 0..count {
                    let offset = row * self.dim;
                    self.read_point_into(
                        start_pid + row as u32,
                        &mut out[offset..offset + self.dim],
                    )?;
                }
                Ok(())
            }

            fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
                for (row, &pid) in ids.iter().enumerate() {
                    let offset = row * self.dim;
                    self.read_point_into(pid, &mut out[offset..offset + self.dim])?;
                }
                Ok(())
            }

            fn disables_prefetch_pipeline(&self) -> bool {
                true
            }
        }

        let store = NoPrefetchStore { rows: 32, dim: 4 };
        let ids = vec![7_u32, 1, 2, 12, 13];
        let config = super::StrictPrefetchPipelineConfig {
            enabled: true,
            queue_depth: 4,
            budget_bytes: 1024,
            max_window_bytes: 1024,
            max_gap_rows: 2,
        };
        let mut profile = super::StrictPrefetchPipelineProfile::default();
        let mut seen = Vec::new();

        super::for_each_prefetched_point_batch_profiled(
            &store,
            &ids,
            2,
            8,
            super::default_rbc_windowed_options(&store, 2),
            Some(config),
            None,
            Some(&mut profile),
            |batch| {
                seen.push(batch.ids.clone());
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(seen, vec![vec![7, 1], vec![2, 12], vec![13]]);
        assert_eq!(profile.pipeline, super::PrefetchPipelineKind::Off);
        assert_eq!(profile.queue_depth, 0);
        assert_eq!(profile.prefetch_budget_bytes, 0);
        assert_eq!(profile.batches, 3);
    }

    #[test]
    pub fn forced_oversized_leaf_split_uses_strict_prefetch_options() {
        pub struct RecordingPointStore {
            pub rows: usize,
            pub dim: usize,
            pub observed_large_batch_gaps: StdMutex<Vec<u32>>,
        }

        impl PointStore for RecordingPointStore {
            fn len(&self) -> usize {
                self.rows
            }

            fn dim(&self) -> usize {
                self.dim
            }

            fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
                for (axis, value) in out.iter_mut().enumerate() {
                    *value = pid as f32 * 0.01 + axis as f32;
                }
                Ok(())
            }

            fn read_range_into(
                &self,
                start_pid: u32,
                count: usize,
                out: &mut [f32],
            ) -> AnnResult<()> {
                for row in 0..count {
                    let pid = start_pid + row as u32;
                    let offset = row * self.dim;
                    self.read_point_into(pid, &mut out[offset..offset + self.dim])?;
                }
                Ok(())
            }

            fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
                for (row, &pid) in ids.iter().enumerate() {
                    let offset = row * self.dim;
                    self.read_point_into(pid, &mut out[offset..offset + self.dim])?;
                }
                Ok(())
            }

            fn read_points_windowed_into_batch_stats(
                &self,
                ids: &[u32],
                out: &mut [f32],
                options: &WindowedGatherOptions,
                stats: &mut PointBatchStats,
            ) -> AnnResult<()> {
                if ids.len() > 100 {
                    self.observed_large_batch_gaps
                        .lock()
                        .unwrap()
                        .push(options.max_gap_rows);
                }
                stats.point_calls += ids.len() as u64;
                stats.bytes_read += ids
                    .len()
                    .saturating_mul(self.dim)
                    .saturating_mul(size_of::<f32>()) as u64;
                self.read_points_into(ids, out)
            }

            fn prefers_coalesced_window_reads(&self) -> bool {
                true
            }
        }

        let store = RecordingPointStore {
            rows: 8_101,
            dim: 4,
            observed_large_batch_gaps: StdMutex::new(Vec::new()),
        };
        let mut params = super::ForgeANNParams::default();
        params.oom_enable = true;
        params.c_min = 1;
        params.max_leaders = 2;
        params.fanout_top = 1;

        let leaf = (0..store.rows as u32).collect::<Vec<_>>();
        let split = super::forced_split_oversized_leaf(
            &store,
            Metric::L2,
            &params,
            leaf,
            0,
            super::FORCED_LEAF_HARD_CAP,
            17,
        )
        .unwrap();
        let observed = store.observed_large_batch_gaps.lock().unwrap().clone();

        assert!(!split.is_empty());
        assert!(!observed.is_empty());
        assert!(
            observed.iter().all(|&gap| gap == 4),
            "forced leaf split point batches should honor strict gap rows: {observed:?}"
        );
    }
}

pub(crate) fn split_seeded_clusters_balanced(
    mut clusters: Vec<SeededCluster>,
) -> (Vec<SeededCluster>, Vec<SeededCluster>) {
    clusters.sort_unstable_by(|left, right| right.points.len().cmp(&left.points.len()));

    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut left_points = 0usize;
    let mut right_points = 0usize;

    for cluster in clusters {
        if left_points <= right_points {
            left_points += cluster.points.len();
            left.push(cluster);
        } else {
            right_points += cluster.points.len();
            right.push(cluster);
        }
    }

    if left.is_empty() && !right.is_empty() {
        left.push(right.pop().unwrap());
    } else if right.is_empty() && !left.is_empty() {
        right.push(left.pop().unwrap());
    }

    (left, right)
}
