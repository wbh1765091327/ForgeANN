use std::collections::BTreeMap;
use std::mem::size_of;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crossbeam::channel;

use super::io_runtime::{
    BoundedReadPlanner as PointPipelineReadPlanner, VectorWindowCache, VectorWindowCacheStats,
    VectorWindowKey,
};
use super::params::ForgeANNParams;
use super::point_store::{PointBatchStats, PointStore, ResidentSubsetPointStore};
use crate::common::{AnnError, AnnResult};

const DEFAULT_PIPELINE_IO_THREADS: usize = 2;
const DEFAULT_PIPELINE_QUEUE_DEPTH: usize = 4;
const DEFAULT_PIPELINE_MAX_WINDOW_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_PIPELINE_MAX_READ_AMPLIFICATION: f64 = 2.0;
const DEFAULT_PIPELINE_MIN_LEAF_POINTS: usize = 1024;

#[derive(Clone, Debug)]
pub(crate) struct PointPipelineConfig {
    pub(crate) enabled: bool,
    pub(crate) io_threads: usize,
    pub(crate) queue_depth: usize,
    pub(crate) budget_bytes: usize,
    pub(crate) max_window_bytes: usize,
    pub(crate) max_read_amplification: f64,
    pub(crate) min_leaf_points: usize,
    pub(crate) window_cache_bytes: usize,
}

impl Default for PointPipelineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            io_threads: DEFAULT_PIPELINE_IO_THREADS,
            queue_depth: DEFAULT_PIPELINE_QUEUE_DEPTH,
            budget_bytes: 0,
            max_window_bytes: DEFAULT_PIPELINE_MAX_WINDOW_BYTES,
            max_read_amplification: DEFAULT_PIPELINE_MAX_READ_AMPLIFICATION,
            min_leaf_points: DEFAULT_PIPELINE_MIN_LEAF_POINTS,
            window_cache_bytes: 0,
        }
    }
}

impl PointPipelineConfig {
    pub(crate) fn from_params(params: &ForgeANNParams) -> Self {
        let io_planned_actual_prefetch = params.io_planned_forgeann_enabled();
        let mut config = Self {
            enabled: params.oom_enable || io_planned_actual_prefetch,
            io_threads: ForgeANNParams::OOM_POINT_PIPELINE_IO_THREADS,
            queue_depth: ForgeANNParams::OOM_POINT_PIPELINE_QUEUE_DEPTH,
            budget_bytes: ForgeANNParams::OOM_POINT_PIPELINE_BUDGET_BYTES,
            max_window_bytes: ForgeANNParams::OOM_POINT_PIPELINE_MAX_WINDOW_BYTES,
            max_read_amplification: ForgeANNParams::OOM_POINT_PIPELINE_MAX_READ_AMPLIFICATION,
            min_leaf_points: ForgeANNParams::OOM_POINT_PIPELINE_MIN_LEAF_POINTS,
            window_cache_bytes: ForgeANNParams::IO_PLAN_WINDOW_CACHE_BYTES,
        };
        if !params.oom_enable && io_planned_actual_prefetch {
            config.budget_bytes =
                derive_io_planned_prefetch_budget_bytes(params.effective_oom_memory_budget_bytes());
        }
        config
    }
}

pub(crate) fn derive_io_planned_prefetch_budget_bytes(oom_budget_bytes: usize) -> usize {
    const MIN_DERIVED_BUDGET: usize = 64 * 1024 * 1024;
    const MAX_DERIVED_BUDGET: usize = 8 * 1024 * 1024 * 1024;

    (oom_budget_bytes.saturating_mul(15) / 100).clamp(MIN_DERIVED_BUDGET, MAX_DERIVED_BUDGET)
}

#[derive(Debug)]
struct BudgetState {
    used: usize,
    peak: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PointPipelineBudget {
    capacity: usize,
    state: Arc<(Mutex<BudgetState>, Condvar)>,
}

impl PointPipelineBudget {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            state: Arc::new((Mutex::new(BudgetState { used: 0, peak: 0 }), Condvar::new())),
        }
    }

    pub(crate) fn peak_bytes(&self) -> usize {
        self.state
            .0
            .lock()
            .expect("point pipeline budget poisoned")
            .peak
    }

    fn acquire(&self, bytes: usize, wait: &mut Duration) -> AnnResult<PointPipelinePermit> {
        let bytes = bytes.max(1);
        if bytes > self.capacity {
            return Err(AnnError::log_index_error(format!(
                "Point pipeline batch requires {bytes} bytes, exceeding budget {}",
                self.capacity
            )));
        }

        let start = Instant::now();
        let (lock, cv) = &*self.state;
        let mut state = lock.lock().expect("point pipeline budget poisoned");
        while state.used.saturating_add(bytes) > self.capacity {
            state = cv.wait(state).expect("point pipeline budget poisoned");
        }
        *wait += start.elapsed();
        state.used += bytes;
        state.peak = state.peak.max(state.used);
        Ok(PointPipelinePermit {
            budget: self.clone(),
            bytes,
        })
    }

    fn release(&self, bytes: usize) {
        let (lock, cv) = &*self.state;
        let mut state = lock.lock().expect("point pipeline budget poisoned");
        state.used = state.used.saturating_sub(bytes);
        cv.notify_one();
    }
}

#[derive(Debug)]
pub(crate) struct PointPipelinePermit {
    budget: PointPipelineBudget,
    bytes: usize,
}

impl Drop for PointPipelinePermit {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PointPipelineStats {
    pub(crate) batches: usize,
    pub(crate) points: usize,
    pub(crate) requested_rows: usize,
    pub(crate) physical_rows: usize,
    pub(crate) planned_windows: usize,
    pub(crate) logical_bytes: u64,
    pub(crate) physical_bytes: u64,
    pub(crate) direct_read_calls_avoided: usize,
    pub(crate) read_wall: Duration,
    pub(crate) consumer_wait: Duration,
    pub(crate) producer_wait: Duration,
    pub(crate) budget_wait: Duration,
    pub(crate) ready_queue_depth_peak: usize,
    pub(crate) permit_peak_bytes: usize,
    pub(crate) window_cache: VectorWindowCacheStats,
}

impl PointPipelineStats {
    pub(crate) fn merge(&mut self, other: Self) {
        self.batches += other.batches;
        self.points += other.points;
        self.requested_rows += other.requested_rows;
        self.physical_rows += other.physical_rows;
        self.planned_windows += other.planned_windows;
        self.logical_bytes += other.logical_bytes;
        self.physical_bytes += other.physical_bytes;
        self.direct_read_calls_avoided += other.direct_read_calls_avoided;
        self.read_wall += other.read_wall;
        self.consumer_wait += other.consumer_wait;
        self.producer_wait += other.producer_wait;
        self.budget_wait += other.budget_wait;
        self.ready_queue_depth_peak = self
            .ready_queue_depth_peak
            .max(other.ready_queue_depth_peak);
        self.permit_peak_bytes = self.permit_peak_bytes.max(other.permit_peak_bytes);
        self.window_cache.merge(&other.window_cache);
    }

    pub(crate) fn avg_read_size_bytes(&self) -> f64 {
        if self.planned_windows == 0 {
            0.0
        } else {
            self.physical_bytes as f64 / self.planned_windows as f64
        }
    }

    pub(crate) fn read_amplification(&self) -> f64 {
        if self.logical_bytes == 0 {
            0.0
        } else {
            self.physical_bytes as f64 / self.logical_bytes as f64
        }
    }

    pub(crate) fn avg_rows_per_window(&self) -> f64 {
        if self.planned_windows == 0 {
            0.0
        } else {
            self.physical_rows as f64 / self.planned_windows as f64
        }
    }
}

#[derive(Debug)]
pub(crate) struct PipelinedPointBatch {
    pub(crate) data: Vec<f32>,
    pub(crate) io_stats: PointBatchStats,
    pub(crate) load: Duration,
    pub(crate) pipeline_stats: PointPipelineStats,
    _permit: PointPipelinePermit,
}

pub(crate) struct HydratedResidentSubset {
    pub(crate) store: ResidentSubsetPointStore,
    pub(crate) pipeline_stats: PointPipelineStats,
    pub(crate) io_stats: PointBatchStats,
    pub(crate) load: Duration,
}

#[derive(Debug)]
struct ReadyBatch {
    seq: usize,
    batch: PipelinedPointBatch,
}

fn read_points_planned_into_batch_stats(
    dataset: &dyn PointStore,
    ids: &[u32],
    out: &mut [f32],
    config: &PointPipelineConfig,
    stats: &mut PointBatchStats,
    window_cache: Option<&Arc<Mutex<VectorWindowCache>>>,
) -> AnnResult<PointPipelineStats> {
    let dim = dataset.dim();
    if out.len() != ids.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Point pipeline output size mismatch: got {} expected {}",
            out.len(),
            ids.len().saturating_mul(dim)
        )));
    }
    if dataset.is_resident_subset() {
        dataset.read_points_into(ids, out)?;
        let logical_bytes = ids
            .len()
            .saturating_mul(dim)
            .saturating_mul(size_of::<f32>()) as u64;
        return Ok(PointPipelineStats {
            requested_rows: ids.len(),
            physical_rows: ids.len(),
            logical_bytes,
            physical_bytes: 0,
            ..PointPipelineStats::default()
        });
    }

    let planner = PointPipelineReadPlanner {
        row_bytes: dim.saturating_mul(std::mem::size_of::<f32>()).max(1),
        max_window_bytes: config.max_window_bytes.max(1),
        max_read_amplification: config.max_read_amplification,
    };
    let plan = planner.plan(ids);
    let mut pipeline_stats = PointPipelineStats {
        requested_rows: plan.requested_rows(),
        physical_rows: plan.physical_rows(),
        planned_windows: plan.windows.len(),
        logical_bytes: plan.logical_bytes(),
        physical_bytes: plan.physical_bytes(),
        direct_read_calls_avoided: plan.requested_rows().saturating_sub(plan.windows.len()),
        ..PointPipelineStats::default()
    };
    if window_cache.is_none() && dataset.read_bounded_plan_into_batch_stats(&plan, out, stats)? {
        return Ok(pipeline_stats);
    }
    let mut window_data = Vec::new();
    let mut scatter_cursor = 0usize;
    for (window_idx, window) in plan.windows.iter().enumerate() {
        let rows = window.row_count as usize;
        let scatter_start = scatter_cursor;
        while scatter_cursor < plan.scatter.len()
            && plan.scatter[scatter_cursor].window_idx == window_idx
        {
            scatter_cursor += 1;
        }
        let window_scatter = &plan.scatter[scatter_start..scatter_cursor];
        let key = VectorWindowKey {
            start_pid: window.start_row,
            row_count: window.row_count,
        };
        let cached = window_cache.and_then(|cache| {
            let mut cache = cache.lock().expect("vector window cache poisoned");
            let hit = cache.get(&key);
            if hit.is_some() {
                pipeline_stats.window_cache.hits += 1;
                pipeline_stats.window_cache.saved_direct_read_calls += 1;
                pipeline_stats.window_cache.logical_bytes =
                    pipeline_stats.window_cache.logical_bytes.saturating_add(
                        rows.saturating_mul(dim).saturating_mul(size_of::<f32>()) as u64,
                    );
            } else {
                pipeline_stats.window_cache.misses += 1;
            }
            hit
        });
        if let Some(cached) = cached {
            for scatter in window_scatter {
                let src_start = scatter.row_offset_in_window as usize * dim;
                let src_end = src_start + dim;
                let dst_start = scatter.original_pos * dim;
                let dst_end = dst_start + dim;
                out[dst_start..dst_end].copy_from_slice(&cached[src_start..src_end]);
            }
            continue;
        }

        window_data.clear();
        window_data.resize(rows.saturating_mul(dim), 0.0f32);
        dataset.read_range_into(window.start_row, rows, &mut window_data)?;
        if let Some(cache) = window_cache {
            let data: Arc<[f32]> = Arc::from(window_data.clone().into_boxed_slice());
            let mut cache = cache.lock().expect("vector window cache poisoned");
            let before = cache.stats().clone();
            cache.insert(key, data);
            let after = cache.stats().clone();
            pipeline_stats.window_cache.inserts += after.inserts.saturating_sub(before.inserts);
            pipeline_stats.window_cache.evictions +=
                after.evictions.saturating_sub(before.evictions);
            pipeline_stats.window_cache.bytes_inserted = pipeline_stats
                .window_cache
                .bytes_inserted
                .saturating_add(after.bytes_inserted.saturating_sub(before.bytes_inserted));
            pipeline_stats.window_cache.bytes_evicted = pipeline_stats
                .window_cache
                .bytes_evicted
                .saturating_add(after.bytes_evicted.saturating_sub(before.bytes_evicted));
            pipeline_stats.window_cache.used_peak_bytes = pipeline_stats
                .window_cache
                .used_peak_bytes
                .max(cache.used_bytes());
            pipeline_stats.window_cache.physical_bytes = pipeline_stats
                .window_cache
                .physical_bytes
                .saturating_add(rows.saturating_mul(dim).saturating_mul(size_of::<f32>()) as u64);
        }
        for scatter in window_scatter {
            let src_start = scatter.row_offset_in_window as usize * dim;
            let src_end = src_start + dim;
            let dst_start = scatter.original_pos * dim;
            let dst_end = dst_start + dim;
            out[dst_start..dst_end].copy_from_slice(&window_data[src_start..src_end]);
        }
        if rows > 1 {
            stats.range_calls += 1;
            stats.range_rows_read += rows as u64;
        } else {
            stats.point_calls += 1;
        }
    }
    debug_assert_eq!(scatter_cursor, plan.scatter.len());
    stats.bytes_read += plan.physical_bytes();

    Ok(pipeline_stats)
}

fn read_one_batch(
    dataset: &dyn PointStore,
    ids: Vec<u32>,
    config: &PointPipelineConfig,
    budget: &PointPipelineBudget,
    window_cache: Option<&Arc<Mutex<VectorWindowCache>>>,
) -> AnnResult<PipelinedPointBatch> {
    let dim = dataset.dim();
    let bytes = ids
        .len()
        .saturating_mul(dim)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    let mut budget_wait = Duration::ZERO;
    let permit = budget.acquire(bytes, &mut budget_wait)?;
    let load_start = Instant::now();
    let mut data = vec![0.0f32; ids.len().saturating_mul(dim)];
    let mut io_stats = PointBatchStats::default();
    let mut pipeline_stats = read_points_planned_into_batch_stats(
        dataset,
        &ids,
        &mut data,
        config,
        &mut io_stats,
        window_cache,
    )?;
    let load = load_start.elapsed();
    pipeline_stats.batches = 1;
    pipeline_stats.points = ids.len();
    pipeline_stats.read_wall = load;
    pipeline_stats.budget_wait = budget_wait;

    Ok(PipelinedPointBatch {
        data,
        io_stats,
        load,
        pipeline_stats,
        _permit: permit,
    })
}

pub(crate) fn for_each_pipelined_point_batch<F>(
    dataset: &dyn PointStore,
    ids: &[u32],
    batch_points: usize,
    config: &PointPipelineConfig,
    mut consume: F,
) -> AnnResult<PointPipelineStats>
where
    F: FnMut(PipelinedPointBatch) -> AnnResult<()>,
{
    if ids.is_empty() {
        return Ok(PointPipelineStats::default());
    }

    let dim = dataset.dim();
    let row_bytes = dim.saturating_mul(std::mem::size_of::<f32>()).max(1);
    let queue_depth = config.queue_depth.max(1);
    let io_threads = config.io_threads.max(1);
    let budget_bytes = config
        .budget_bytes
        .max(row_bytes.saturating_mul(io_threads));
    let max_points_by_budget = (budget_bytes / row_bytes / io_threads).max(1);
    let effective_batch_points = batch_points.max(1).min(max_points_by_budget);
    let chunks: Vec<Vec<u32>> = ids
        .chunks(effective_batch_points)
        .map(|chunk| chunk.to_vec())
        .collect();
    let batch_count = chunks.len();
    let budget = PointPipelineBudget::new(budget_bytes);
    let window_cache = (config.window_cache_bytes > 0).then(|| {
        Arc::new(Mutex::new(VectorWindowCache::new(
            config.window_cache_bytes,
        )))
    });
    let (job_tx, job_rx) = channel::bounded::<(usize, Vec<u32>)>(queue_depth);
    let (ready_tx, ready_rx) = channel::bounded::<AnnResult<ReadyBatch>>(queue_depth);
    let producer_wait = Arc::new(Mutex::new(Duration::ZERO));

    std::thread::scope(|scope| {
        for _ in 0..io_threads {
            let job_rx = job_rx.clone();
            let ready_tx = ready_tx.clone();
            let budget = budget.clone();
            let window_cache = window_cache.clone();
            let producer_wait_for_worker = Arc::clone(&producer_wait);
            scope.spawn(move || {
                for (seq, batch_ids) in job_rx {
                    let result =
                        read_one_batch(dataset, batch_ids, config, &budget, window_cache.as_ref())
                            .map(|batch| ReadyBatch { seq, batch });
                    let send_start = Instant::now();
                    if ready_tx.send(result).is_err() {
                        return;
                    }
                    if let Ok(mut wait) = producer_wait_for_worker.lock() {
                        *wait += send_start.elapsed();
                    }
                }
            });
        }
        drop(ready_tx);

        let producer_wait_for_thread = Arc::clone(&producer_wait);
        scope.spawn(move || {
            for (seq, chunk) in chunks.into_iter().enumerate() {
                let send_start = Instant::now();
                if job_tx.send((seq, chunk)).is_err() {
                    return;
                }
                if let Ok(mut wait) = producer_wait_for_thread.lock() {
                    *wait += send_start.elapsed();
                }
            }
        });

        let mut stats = PointPipelineStats::default();
        let mut pending = BTreeMap::new();
        for expected_seq in 0..batch_count {
            while !pending.contains_key(&expected_seq) {
                let recv_start = Instant::now();
                let ready = ready_rx.recv().map_err(|err| {
                    AnnError::log_index_error(format!(
                        "Point pipeline I/O workers ended before ready batch: {err}"
                    ))
                })??;
                stats.consumer_wait += recv_start.elapsed();
                let observed_ready_depth = (ready_rx.len() + 1).min(queue_depth);
                stats.ready_queue_depth_peak =
                    stats.ready_queue_depth_peak.max(observed_ready_depth);
                pending.insert(ready.seq, ready.batch);
            }
            let batch = pending
                .remove(&expected_seq)
                .expect("expected ready batch is present");
            stats.merge(batch.pipeline_stats.clone());
            consume(batch)?;
        }

        stats.producer_wait = producer_wait.lock().map(|wait| *wait).unwrap_or_default();
        stats.permit_peak_bytes = budget.peak_bytes();
        Ok(stats)
    })
}

pub(crate) fn hydrate_resident_subset_with_stats(
    dataset: &dyn PointStore,
    ids: &[u32],
    config: &PointPipelineConfig,
) -> AnnResult<HydratedResidentSubset> {
    let dim = dataset.dim();
    let mut data = Vec::with_capacity(ids.len().saturating_mul(dim));
    let mut effective = config.clone();
    effective.enabled = true;
    if effective.budget_bytes == 0 {
        effective.budget_bytes = ids
            .len()
            .saturating_mul(dim)
            .saturating_mul(std::mem::size_of::<f32>())
            .max(1);
    }
    let mut io_stats = PointBatchStats::default();
    let mut load = Duration::ZERO;
    let pipeline_stats =
        for_each_pipelined_point_batch(dataset, ids, ids.len().max(1), &effective, |batch| {
            io_stats.point_calls += batch.io_stats.point_calls;
            io_stats.range_calls += batch.io_stats.range_calls;
            io_stats.range_rows_read += batch.io_stats.range_rows_read;
            io_stats.bytes_read += batch.io_stats.bytes_read;
            load += batch.load;
            data.extend_from_slice(&batch.data);
            Ok(())
        })?;
    let global_ids = ids.to_vec();
    let store = ResidentSubsetPointStore::new(global_ids, dim, data)?;
    Ok(HydratedResidentSubset {
        store,
        pipeline_stats,
        io_stats,
        load,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::mem::size_of;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{PointPipelineConfig, PointPipelineReadPlanner, for_each_pipelined_point_batch};
    use crate::common::Metric;
    use crate::forgeann::direct_io::DirectIoConfig;
    use crate::forgeann::io_runtime::{VectorWindowCache, VectorWindowKey};
    use crate::forgeann::point_store::{
        DirectPointStore, InmemDatasetPointStore, PointStore, reset_uring_gather_test_counters,
        uring_gather_test_attempts,
    };
    use crate::model::InmemDataset;

    fn build_test_store(rows: usize, dim: usize) -> InmemDataset<f32> {
        let mut dataset = InmemDataset::new(rows, 1.0, dim).expect("dataset");
        for row in 0..rows {
            for col in 0..dim {
                dataset.data[row * dim + col] = row as f32 + col as f32 * 0.25;
            }
        }
        dataset
    }

    fn write_test_fbin(path: &std::path::Path, rows: usize, dim: usize) {
        let mut file = std::fs::File::create(path).expect("create fbin");
        file.write_all(&(rows as u32).to_le_bytes()).unwrap();
        file.write_all(&(dim as u32).to_le_bytes()).unwrap();
        for row in 0..rows {
            for col in 0..dim {
                let value = row as f32 + col as f32 * 0.25;
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }

    struct CountingPointStore<'a> {
        inner: InmemDatasetPointStore<'a>,
        range_calls: AtomicUsize,
    }

    impl<'a> CountingPointStore<'a> {
        fn new(dataset: &'a InmemDataset<f32>, len: usize) -> Self {
            Self {
                inner: InmemDatasetPointStore::new(dataset, len),
                range_calls: AtomicUsize::new(0),
            }
        }
    }

    impl PointStore for CountingPointStore<'_> {
        fn len(&self) -> usize {
            self.inner.len()
        }

        fn dim(&self) -> usize {
            self.inner.dim()
        }

        fn read_point_into(&self, pid: u32, out: &mut [f32]) -> crate::common::AnnResult<()> {
            self.inner.read_point_into(pid, out)
        }

        fn read_range_into(
            &self,
            start_pid: u32,
            count: usize,
            out: &mut [f32],
        ) -> crate::common::AnnResult<()> {
            self.range_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.read_range_into(start_pid, count, out)
        }

        fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> crate::common::AnnResult<()> {
            self.inner.read_points_into(ids, out)
        }

        fn get_distance(
            &self,
            lhs: u32,
            rhs: u32,
            metric: Metric,
        ) -> crate::common::AnnResult<f32> {
            self.inner.get_distance(lhs, rhs, metric)
        }
    }

    #[test]
    fn ready_batches_preserve_input_point_order() {
        let dataset = build_test_store(16, 3);
        let store = InmemDatasetPointStore::new(&dataset, 16);
        let ids = vec![8_u32, 2, 3, 7, 15, 1];
        let mut seen = Vec::new();
        let mut config = PointPipelineConfig::default();
        config.enabled = true;
        config.io_threads = 2;
        config.queue_depth = 2;
        config.budget_bytes = 1024 * 1024;

        let mut cursor = 0usize;
        let stats = for_each_pipelined_point_batch(&store, &ids, 2, &config, |batch| {
            let rows = batch.data.len() / store.dim();
            let batch_ids = &ids[cursor..cursor + rows];
            cursor += rows;
            seen.extend_from_slice(batch_ids);
            for (row, &pid) in batch_ids.iter().enumerate() {
                let mut expected = vec![0.0f32; store.dim()];
                store.read_point_into(pid, &mut expected)?;
                assert_eq!(
                    &batch.data[row * store.dim()..(row + 1) * store.dim()],
                    expected.as_slice()
                );
            }
            Ok(())
        })
        .expect("pipeline batches");

        assert_eq!(seen, ids);
        assert_eq!(stats.batches, 3);
        assert!(stats.ready_queue_depth_peak <= config.queue_depth);
    }

    #[test]
    fn batch_sizing_reserves_budget_for_each_io_worker() {
        let dataset = build_test_store(16, 2);
        let store = InmemDatasetPointStore::new(&dataset, 16);
        let ids: Vec<u32> = (0..8).collect();
        let row_bytes = store.dim() * size_of::<f32>();
        let mut config = PointPipelineConfig::default();
        config.enabled = true;
        config.io_threads = 4;
        config.queue_depth = 4;
        config.budget_bytes = row_bytes * config.io_threads;

        let stats =
            for_each_pipelined_point_batch(&store, &ids, ids.len(), &config, |_batch| Ok(()))
                .expect("pipeline batches");

        assert_eq!(stats.batches, ids.len());
        assert!(stats.permit_peak_bytes <= config.budget_bytes);
    }

    #[test]
    fn bounded_amplification_planner_keeps_sparse_ids_under_configured_ratio() {
        let planner = PointPipelineReadPlanner {
            row_bytes: 128,
            max_window_bytes: 64 * 1024,
            max_read_amplification: 2.0,
        };
        let ids = vec![0_u32, 1, 100, 101, 220, 221];
        let plan = planner.plan(&ids);

        assert!(plan.read_amplification() <= 2.0);
        assert!(plan.windows.len() >= 3);
    }

    #[test]
    fn dense_ids_coalesce_into_larger_windows_than_fixed_gap_four() {
        let planner = PointPipelineReadPlanner {
            row_bytes: 128,
            max_window_bytes: 64 * 1024,
            max_read_amplification: 2.0,
        };
        let ids: Vec<u32> = (0_u32..10).chain(15..25).collect();
        let plan = planner.plan(&ids);

        assert_eq!(plan.windows.len(), 1);
        assert_eq!(plan.windows[0].row_count, 25);
        assert!(plan.read_amplification() <= 2.0);
    }

    #[test]
    fn leaf_hydrate_plan_creates_resident_subset_with_identical_vectors() {
        let dataset = build_test_store(16, 4);
        let store = InmemDatasetPointStore::new(&dataset, 16);
        let ids = vec![9_u32, 2, 7, 3];
        let config = PointPipelineConfig {
            enabled: true,
            budget_bytes: 1024 * 1024,
            ..PointPipelineConfig::default()
        };

        let resident = super::hydrate_resident_subset_with_stats(&store, &ids, &config)
            .expect("hydrate")
            .store;
        let mut actual = vec![0.0f32; ids.len() * store.dim()];
        let mut expected = vec![0.0f32; ids.len() * store.dim()];
        resident.read_points_into(&ids, &mut actual).unwrap();
        store.read_points_into(&ids, &mut expected).unwrap();

        assert_eq!(actual, expected);
        assert!(resident.resident_bytes() >= ids.len() * store.dim() * size_of::<f32>());
    }

    #[test]
    fn vector_window_cache_accounts_hits_and_respects_hard_budget() {
        let mut cache = VectorWindowCache::new(4 * size_of::<f32>());
        let first_key = VectorWindowKey {
            start_pid: 10,
            row_count: 2,
        };
        let second_key = VectorWindowKey {
            start_pid: 20,
            row_count: 2,
        };

        cache.insert(first_key, vec![1.0, 2.0].into());
        assert_eq!(cache.used_bytes(), 2 * size_of::<f32>());
        assert!(cache.get(&first_key).is_some());
        assert_eq!(cache.stats().hits, 1);

        cache.insert(second_key, vec![3.0, 4.0].into());
        assert_eq!(cache.used_bytes(), 4 * size_of::<f32>());
        cache.insert(
            VectorWindowKey {
                start_pid: 30,
                row_count: 2,
            },
            vec![5.0, 6.0].into(),
        );

        assert!(cache.used_bytes() <= 4 * size_of::<f32>());
        assert!(cache.stats().evictions >= 1);
    }

    #[test]
    fn window_cache_reuses_physical_windows_across_batches() {
        let dataset = build_test_store(64, 4);
        let store = CountingPointStore::new(&dataset, 64);
        let ids = vec![10_u32, 11, 12, 13, 10, 11, 12, 13];
        let mut config = PointPipelineConfig::default();
        config.enabled = true;
        config.io_threads = 1;
        config.queue_depth = 1;
        config.budget_bytes = 1024 * 1024;
        config.max_window_bytes = 1024 * 1024;
        config.max_read_amplification = 1.5;
        config.window_cache_bytes = 1024 * 1024;

        let stats =
            for_each_pipelined_point_batch(&store, &ids, 4, &config, |_batch| Ok(())).unwrap();

        assert_eq!(stats.planned_windows, 2);
        assert_eq!(store.range_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stats.window_cache.hits, 1);
        assert_eq!(stats.window_cache.misses, 1);
        assert_eq!(stats.window_cache.saved_direct_read_calls, 1);
        assert!(stats.window_cache.used_peak_bytes >= 4 * 4 * size_of::<f32>());
    }

    #[test]
    fn strict_direct_pipeline_uses_bounded_uring_fast_path() {
        reset_uring_gather_test_counters();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("strict-pipeline-bounded.fbin");
        write_test_fbin(&path, 128, 8);
        let buffered = DirectPointStore::open(&path).unwrap();
        let strict =
            DirectPointStore::open_with_config(&path, DirectIoConfig::enabled_with_alignment(4096))
                .unwrap();
        let ids = vec![90_u32, 2, 3, 40, 41, 42, 70, 71, 110, 111];
        let mut expected = vec![0.0f32; ids.len() * strict.dim()];
        buffered.read_points_into(&ids, &mut expected).unwrap();

        let mut actual = Vec::new();
        let mut config = PointPipelineConfig::default();
        config.enabled = true;
        config.io_threads = 1;
        config.queue_depth = 1;
        config.budget_bytes = 1024 * 1024;
        config.max_window_bytes = 2 * strict.dim() * size_of::<f32>();
        config.max_read_amplification = 1.5;
        config.window_cache_bytes = 0;

        let stats = for_each_pipelined_point_batch(&strict, &ids, ids.len(), &config, |batch| {
            actual.extend_from_slice(&batch.data);
            Ok(())
        })
        .unwrap();

        assert_eq!(actual, expected);
        assert!(stats.planned_windows >= 2);
        assert!(
            uring_gather_test_attempts() > 0,
            "strict direct point pipeline should attempt bounded io_uring reads"
        );
    }
}
