use std::mem::size_of;

use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;
use rayon::prelude::*;

use super::params::ForgeANNParams;
use super::point_store::{PointStore, WindowedGatherOptions, WindowedGatherStats};
use crate::common::{AnnError, AnnResult};
use crate::model::InmemDataset;
use crate::utils::thread_pool::with_rayon_thread_pool;

pub trait SketchAccessor: Sync {
    fn width(&self) -> usize;
    fn rows(&self) -> usize;
    fn row_copy_into(&self, idx: usize, dst: &mut [f32]) -> AnnResult<()>;
    fn resident_bytes(&self) -> usize;

    /// Batched row read — default implementation calls `row_copy_into` per ID.
    /// Implementations should override to coalesce contiguous runs.
    fn read_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let w = self.width();
        for (i, &id) in ids.iter().enumerate() {
            self.row_copy_into(id as usize, &mut out[i * w..(i + 1) * w])?;
        }
        Ok(())
    }

    fn read_rows_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        _options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        if !ids.is_empty() {
            let row_bytes = self.width() * size_of::<f32>();
            stats.rows_requested += ids.len() as u64;
            stats.logical_bytes += (ids.len() * row_bytes) as u64;
            stats.physical_bytes += (ids.len() * row_bytes) as u64;
            stats.scatter_ops += ids.len() as u64;
            stats.singleton_windows += ids.len() as u64;
            stats.windows_submitted += ids.len() as u64;
            stats.rows_per_window_sum += ids.len() as u64;
        }
        self.read_rows_into(ids, out)
    }

    fn read_rows_bounded_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        max_window_bytes: usize,
        _max_read_amplification: f64,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        let row_bytes = self.width().saturating_mul(size_of::<f32>()).max(1);
        let options = WindowedGatherOptions {
            max_gap_rows: 0,
            max_window_bytes: max_window_bytes.max(row_bytes),
            alignment_bytes: row_bytes,
            sort_ids: true,
        };
        self.read_rows_windowed_into_stats(ids, out, &options, stats)
    }
}

/// 行优先存储的 sketch 矩阵，避免为每个点分配一个独立的小 Vec。
#[derive(Debug, Clone)]
pub struct SketchStore {
    data: Vec<f32>,
    width: usize,
}

impl SketchStore {
    #[inline]
    pub fn from_row_major(data: Vec<f32>, width: usize) -> Self {
        Self { data, width }
    }

    #[inline]
    pub fn row(&self, idx: usize) -> &[f32] {
        let start = idx * self.width;
        let end = start + self.width;
        &self.data[start..end]
    }

    #[inline]
    pub fn estimate_bytes(&self) -> usize {
        size_of::<Self>() + self.data.capacity().saturating_mul(size_of::<f32>())
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn rows(&self) -> usize {
        if self.width == 0 {
            0
        } else {
            self.data.len() / self.width
        }
    }

    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[cfg(test)]
    pub(crate) fn from_test_data(data: Vec<f32>, width: usize) -> Self {
        Self::from_row_major(data, width)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ResidentSubsetSketchAccessor {
    global_ids: Vec<u32>,
    id_to_row: Vec<(u32, usize)>,
    data: Vec<f32>,
    width: usize,
    rows: usize,
    sorted_unique_ids: bool,
}

impl ResidentSubsetSketchAccessor {
    pub(crate) fn new(
        global_ids: Vec<u32>,
        rows: usize,
        width: usize,
        data: Vec<f32>,
    ) -> AnnResult<Self> {
        if width == 0 {
            return Err(AnnError::log_index_error(
                "Resident subset sketch accessor requires nonzero width".to_string(),
            ));
        }
        let expected = global_ids.len().saturating_mul(width);
        if data.len() != expected {
            return Err(AnnError::log_index_error(format!(
                "Resident subset sketch matrix size mismatch: got {} expected {}",
                data.len(),
                expected
            )));
        }

        let sorted_unique_ids = is_strictly_increasing_u32(&global_ids);
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
            width,
            rows,
            sorted_unique_ids,
        })
    }

    fn local_row_for_global_id(&self, id: u32) -> AnnResult<usize> {
        if self.sorted_unique_ids {
            return self.global_ids.binary_search(&id).map_err(|_| {
                AnnError::log_index_error(format!(
                    "Sketch row {} not present in resident subset with {} rows",
                    id,
                    self.global_ids.len()
                ))
            });
        }
        self.id_to_row
            .binary_search_by_key(&id, |&(global_id, _)| global_id)
            .map(|idx| self.id_to_row[idx].1)
            .map_err(|_| {
                AnnError::log_index_error(format!(
                    "Sketch row {} not present in resident subset with {} rows",
                    id,
                    self.global_ids.len()
                ))
            })
    }

    fn copy_local_row_into(&self, local_row: usize, out: &mut [f32]) {
        let start = local_row * self.width;
        let end = start + self.width;
        out.copy_from_slice(&self.data[start..end]);
    }

    fn read_sorted_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let mut local_row = 0usize;
        for (row, &id) in ids.iter().enumerate() {
            while local_row < self.global_ids.len() && self.global_ids[local_row] < id {
                local_row += 1;
            }
            if self.global_ids.get(local_row).copied() != Some(id) {
                return Err(AnnError::log_index_error(format!(
                    "Sketch row {} not present in resident subset with {} rows",
                    id,
                    self.global_ids.len()
                )));
            }
            let start = row * self.width;
            let end = start + self.width;
            self.copy_local_row_into(local_row, &mut out[start..end]);
        }
        Ok(())
    }
}

impl SketchAccessor for ResidentSubsetSketchAccessor {
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
        let local_row = self.local_row_for_global_id(idx as u32)?;
        self.copy_local_row_into(local_row, dst);
        Ok(())
    }

    fn read_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let expected = ids.len().saturating_mul(self.width);
        if out.len() != expected {
            return Err(AnnError::log_index_error(format!(
                "Resident subset sketch matrix size mismatch: got {} expected {}",
                out.len(),
                expected
            )));
        }
        if ids.is_empty() {
            return Ok(());
        }
        if ids.len() == self.global_ids.len() && ids == self.global_ids.as_slice() {
            out.copy_from_slice(&self.data);
            return Ok(());
        }
        if self.sorted_unique_ids && is_nondecreasing_u32(ids) {
            return self.read_sorted_rows_into(ids, out);
        }
        for (row, &id) in ids.iter().enumerate() {
            let local_row = self.local_row_for_global_id(id)?;
            let start = row * self.width;
            let end = start + self.width;
            self.copy_local_row_into(local_row, &mut out[start..end]);
        }
        Ok(())
    }

    fn read_rows_windowed_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        _options: &WindowedGatherOptions,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        if !ids.is_empty() {
            let row_bytes = self.width.saturating_mul(size_of::<f32>());
            stats.rows_requested += ids.len() as u64;
            stats.logical_bytes += ids.len().saturating_mul(row_bytes) as u64;
            stats.scatter_ops += ids.len() as u64;
            stats.windows_submitted += 1;
            stats.rows_per_window_sum += ids.len() as u64;
        }
        self.read_rows_into(ids, out)
    }

    fn read_rows_bounded_into_stats(
        &self,
        ids: &[u32],
        out: &mut [f32],
        _max_window_bytes: usize,
        _max_read_amplification: f64,
        stats: &mut WindowedGatherStats,
    ) -> AnnResult<()> {
        if !ids.is_empty() {
            let row_bytes = self.width.saturating_mul(size_of::<f32>());
            stats.rows_requested += ids.len() as u64;
            stats.logical_bytes += ids.len().saturating_mul(row_bytes) as u64;
            stats.scatter_ops += ids.len() as u64;
            stats.windows_submitted += 1;
            stats.rows_per_window_sum += ids.len() as u64;
        }
        self.read_rows_into(ids, out)
    }

    fn resident_bytes(&self) -> usize {
        size_of::<Self>()
            .saturating_add(self.global_ids.capacity().saturating_mul(size_of::<u32>()))
            .saturating_add(
                self.id_to_row
                    .capacity()
                    .saturating_mul(size_of::<(u32, usize)>()),
            )
            .saturating_add(self.data.capacity().saturating_mul(size_of::<f32>()))
    }
}

fn is_strictly_increasing_u32(ids: &[u32]) -> bool {
    ids.windows(2).all(|window| window[0] < window[1])
}

fn is_nondecreasing_u32(ids: &[u32]) -> bool {
    ids.windows(2).all(|window| window[0] <= window[1])
}

impl SketchAccessor for SketchStore {
    #[inline]
    fn width(&self) -> usize {
        self.width()
    }

    #[inline]
    fn rows(&self) -> usize {
        self.rows()
    }

    fn row_copy_into(&self, idx: usize, dst: &mut [f32]) -> AnnResult<()> {
        if dst.len() != self.width {
            return Err(AnnError::log_index_error(format!(
                "Sketch row copy width mismatch: dst={} expected={}",
                dst.len(),
                self.width
            )));
        }
        let row = self.row(idx);
        dst.copy_from_slice(row);
        Ok(())
    }

    fn read_rows_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
        let w = self.width();
        for (i, &id) in ids.iter().enumerate() {
            out[i * w..(i + 1) * w].copy_from_slice(self.row(id as usize));
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
        if ids.is_empty() {
            return Ok(());
        }
        let row_bytes = self.width * size_of::<f32>();
        let plan = super::point_store::build_gather_plan(ids, row_bytes, *options, stats);
        let mut window_buf = vec![
            0.0f32;
            plan.windows
                .iter()
                .map(|window| window.row_count as usize)
                .max()
                .unwrap_or(0)
                * self.width
        ];
        for (window_idx, window) in plan.windows.iter().enumerate() {
            let rows = window.row_count as usize;
            let elems = rows * self.width;
            let window_slice = &mut window_buf[..elems];
            let start = window.start_row as usize * self.width;
            let end = start + elems;
            window_slice.copy_from_slice(&self.data[start..end]);
            super::point_store::scatter_window_rows(
                &plan,
                window_idx,
                self.width,
                window_slice,
                out,
            );
        }
        Ok(())
    }

    #[inline]
    fn resident_bytes(&self) -> usize {
        self.estimate_bytes()
    }
}

/// 简单的 bfloat16 实现：截断 float32 的高 16 位。
#[derive(Clone, Copy, Debug, Default)]
pub struct BFloat16(pub u16);

impl BFloat16 {
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        Self((v.to_bits() >> 16) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

/// HashPrune 水库中的单个槽位。
///
/// 每个槽占 8 字节：4B point id + 2B hash + 2B bf16 距离。
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub target: u32,
    pub hash: u16,
    pub dist: BFloat16,
}

impl Slot {
    #[inline]
    pub fn new(target: u32, hash: u16, dist: f32) -> Self {
        Self {
            target,
            hash,
            dist: BFloat16::from_f32(dist),
        }
    }
}

/// 单个点对应的 HashPrune 水库。
///
/// 槽按 hash 有序；容量固定为 l_max。
#[derive(Debug)]
pub struct HashPruneReservoir {
    l_max: usize,
    slots: Vec<Slot>,
    /// 当前最远槽的下标（仅在 slots 已满时有效）。
    farthest_idx: usize,
    /// 当前最远槽的距离缓存（仅在 slots 已满时有效）。
    farthest_dist: f32,
}

impl HashPruneReservoir {
    pub fn new(l_max: usize) -> Self {
        assert!(l_max > 0, "l_max must be > 0");
        Self {
            l_max,
            slots: Vec::with_capacity(l_max),
            farthest_idx: 0,
            farthest_dist: 0.0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.slots.clear();
        self.farthest_idx = 0;
        self.farthest_dist = 0.0;
    }

    #[inline]
    pub fn capacity_hint(&self) -> usize {
        self.l_max
    }

    /// 重新扫描所有槽，更新 farthest 缓存。O(l_max)，仅在替换后调用。
    #[inline]
    fn recompute_farthest(&mut self) {
        let mut idx = 0usize;
        let mut dist = self.slots[0].dist.to_f32();
        for (i, slot) in self.slots.iter().enumerate().skip(1) {
            let d = slot.dist.to_f32();
            if d > dist {
                dist = d;
                idx = i;
            }
        }
        self.farthest_idx = idx;
        self.farthest_dist = dist;
    }

    #[inline]
    fn recompute_farthest_if_nonempty(&mut self) {
        if self.slots.is_empty() {
            self.farthest_idx = 0;
            self.farthest_dist = 0.0;
        } else {
            self.recompute_farthest();
        }
    }

    #[inline]
    fn lower_bound_hash(&self, hash: u16) -> usize {
        self.slots.partition_point(|s| s.hash < hash)
    }

    #[inline]
    fn upper_bound_hash(&self, hash: u16) -> usize {
        self.slots.partition_point(|s| s.hash <= hash)
    }

    #[inline]
    fn insert_slot_sorted(&mut self, slot: Slot) {
        let insert_pos = self.upper_bound_hash(slot.hash);
        let dist = slot.dist.to_f32();
        self.slots.insert(insert_pos, slot);
        if self.slots.len() == 1 {
            self.farthest_idx = 0;
            self.farthest_dist = dist;
        } else if dist >= self.farthest_dist {
            self.recompute_farthest();
        } else if insert_pos <= self.farthest_idx {
            self.farthest_idx += 1;
        }
    }

    #[inline]
    fn replace_slot_sorted(&mut self, pos: usize, slot: Slot) {
        self.slots.remove(pos);
        let insert_pos = self.upper_bound_hash(slot.hash);
        self.slots.insert(insert_pos, slot);
        self.recompute_farthest_if_nonempty();
    }

    #[inline]
    fn target_position(&self, target: u32) -> Option<usize> {
        self.slots.iter().position(|slot| slot.target == target)
    }

    /// 向水库插入一个候选 (target, hash, dist)。
    ///
    /// - 若同一 target 已存在，则只保留更近距离；
    /// - 若水库未满，则直接插入，不因 hash 冲突删除已有边；
    /// - 若水库已满且 hash 冲突，则替换同 hash 中最远且更远的槽；
    /// - 若水库已满且无 hash 冲突，则若更近则淘汰当前全局最远点。
    pub fn insert(&mut self, target: u32, hash: u16, dist: f32) {
        if let Some(pos) = self.target_position(target) {
            if dist < self.slots[pos].dist.to_f32() {
                self.replace_slot_sorted(pos, Slot::new(target, hash, dist));
            }
            return;
        }

        if self.slots.len() < self.l_max {
            self.insert_slot_sorted(Slot::new(target, hash, dist));
            return;
        }

        let hash_start = self.lower_bound_hash(hash);
        if hash_start < self.slots.len() && self.slots[hash_start].hash == hash {
            let hash_end = self.upper_bound_hash(hash);
            let mut farthest_same_hash_idx = hash_start;
            let mut farthest_same_hash_dist = self.slots[hash_start].dist.to_f32();
            for idx in hash_start + 1..hash_end {
                let existing_dist = self.slots[idx].dist.to_f32();
                if existing_dist > farthest_same_hash_dist {
                    farthest_same_hash_dist = existing_dist;
                    farthest_same_hash_idx = idx;
                }
            }

            if dist < farthest_same_hash_dist {
                self.replace_slot_sorted(farthest_same_hash_idx, Slot::new(target, hash, dist));
            }
            return;
        }

        if dist < self.farthest_dist {
            self.replace_slot_sorted(self.farthest_idx, Slot::new(target, hash, dist));
        }
    }

    /// 取出当前水库中的邻居 ID。
    pub fn into_neighbors(self) -> Vec<u32> {
        self.slots.into_iter().map(|s| s.target).collect()
    }

    pub fn into_slots(self) -> Vec<Slot> {
        self.slots
    }

    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    pub fn drain_neighbors(&mut self) -> Vec<u32> {
        let slots = std::mem::take(&mut self.slots);
        self.farthest_idx = 0;
        self.farthest_dist = 0.0;
        slots.into_iter().map(|s| s.target).collect()
    }

    pub fn drain_slots(&mut self) -> Vec<Slot> {
        let slots = std::mem::take(&mut self.slots);
        self.farthest_idx = 0;
        self.farthest_dist = 0.0;
        slots
    }
}

/// 为每个向量预计算 m 维 sketch：sketch[v][i] = v · H_i。
/// 其中 H_i 是随机超平面。
pub fn compute_sketches(
    dataset: &InmemDataset<f32>,
    num_points: usize,
    params: &ForgeANNParams,
    num_threads: u32,
) -> AnnResult<SketchStore> {
    let store = super::point_store::InmemDatasetPointStore::new(dataset, num_points);
    compute_sketches_from_store(&store, num_points, params, num_threads)
}

pub fn compute_sketches_from_store(
    dataset: &dyn PointStore,
    num_points: usize,
    params: &ForgeANNParams,
    num_threads: u32,
) -> AnnResult<SketchStore> {
    let dim = dataset.dim();
    let m = params.m_hash_bits;
    let hyperplanes = generate_hyperplanes(dim, m, params.random_seed);

    let mut sketch_data = vec![0.0f32; num_points.saturating_mul(m)];

    // 对每个点计算 sketch，并直接写入目标缓冲区。
    let build_row = |i: usize, out_row: &mut [f32]| -> AnnResult<()> {
        fill_sketch_row(dataset, i, &hyperplanes, out_row)
    };

    if num_threads == 1 {
        for (i, row) in sketch_data.chunks_exact_mut(m).enumerate() {
            build_row(i, row)?;
        }
    } else if num_threads == 0 {
        sketch_data
            .par_chunks_exact_mut(m)
            .enumerate()
            .try_for_each(|(i, row)| build_row(i, row))?;
    } else {
        with_rayon_thread_pool(num_threads, || {
            sketch_data
                .par_chunks_exact_mut(m)
                .enumerate()
                .try_for_each(|(i, row)| build_row(i, row))
        })??;
    }

    Ok(SketchStore {
        data: sketch_data,
        width: m,
    })
}

pub fn generate_hyperplanes(dim: usize, m: usize, random_seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(random_seed ^ 0x5a5a_5a5a_1234_5678);
    let dist = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let mut hyperplanes: Vec<f32> = Vec::with_capacity(m.saturating_mul(dim));
    for _ in 0..m {
        for _ in 0..dim {
            hyperplanes.push(dist.sample(&mut rng));
        }
    }
    hyperplanes
}

pub fn fill_sketch_row(
    dataset: &dyn PointStore,
    point_idx: usize,
    hyperplanes: &[f32],
    out_row: &mut [f32],
) -> AnnResult<()> {
    let dim = dataset.dim();
    if hyperplanes.len() != out_row.len().saturating_mul(dim) {
        return Err(AnnError::log_index_error(format!(
            "Hyperplane shape mismatch: hyperplanes={} out_row={} dim={}",
            hyperplanes.len(),
            out_row.len(),
            dim
        )));
    }

    let mut point = vec![0.0f32; dim];
    dataset.read_point_into(point_idx as u32, &mut point)?;
    for (h_idx, h) in hyperplanes.chunks_exact(dim).enumerate() {
        let mut acc = 0.0f32;
        for (a, b) in point.iter().zip(h.iter()) {
            acc += *a * *b;
        }
        out_row[h_idx] = acc;
    }
    Ok(())
}

/// 根据个性化残差 sketch 计算哈希 h_p(c)。
///
/// 按照论文公式：使用 sign(sketch(c)[i] - sketch(p)[i]) 作为第 i 位。
pub fn compute_hash(sketch_p: &[f32], sketch_c: &[f32], m_bits: usize) -> u16 {
    assert!(m_bits <= 16, "m_bits must be <= 16");
    let mut h: u16 = 0;
    for i in 0..m_bits {
        let bit = if sketch_c[i] - sketch_p[i] >= 0.0 {
            1u16
        } else {
            0u16
        };
        h |= bit << i;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::{HashPruneReservoir, ResidentSubsetSketchAccessor, SketchAccessor};
    use crate::forgeann::point_store::{WindowedGatherOptions, WindowedGatherStats};

    fn targets(reservoir: &HashPruneReservoir) -> Vec<u32> {
        reservoir.slots().iter().map(|slot| slot.target).collect()
    }

    #[test]
    fn reservoir_keeps_hash_collisions_until_capacity() {
        let mut reservoir = HashPruneReservoir::new(4);

        reservoir.insert(10, 7, 10.0);
        reservoir.insert(11, 7, 9.0);
        reservoir.insert(12, 7, 8.0);

        assert_eq!(targets(&reservoir), vec![10, 11, 12]);
        assert_eq!(reservoir.slots().len(), 3);

        reservoir.insert(13, 7, 7.0);
        assert_eq!(targets(&reservoir), vec![10, 11, 12, 13]);

        reservoir.insert(14, 7, 6.0);
        let kept = targets(&reservoir);
        assert_eq!(kept.len(), 4);
        assert!(kept.contains(&14));
        assert!(!kept.contains(&10));
    }

    #[test]
    fn reservoir_deduplicates_same_target_without_growing_degree() {
        let mut reservoir = HashPruneReservoir::new(4);

        reservoir.insert(10, 7, 10.0);
        reservoir.insert(10, 7, 8.0);

        assert_eq!(targets(&reservoir), vec![10]);
        assert_eq!(reservoir.slots().len(), 1);
        assert_eq!(reservoir.slots()[0].dist.to_f32(), 8.0);
    }

    #[test]
    fn full_reservoir_without_hash_collision_replaces_global_farthest() {
        let mut reservoir = HashPruneReservoir::new(3);

        reservoir.insert(10, 1, 10.0);
        reservoir.insert(11, 2, 9.0);
        reservoir.insert(12, 3, 8.0);
        reservoir.insert(13, 4, 7.0);

        let kept = targets(&reservoir);
        assert_eq!(kept.len(), 3);
        assert!(kept.contains(&13));
        assert!(!kept.contains(&10));
    }

    #[test]
    fn resident_subset_sketch_accessor_reads_by_global_id() {
        let accessor = ResidentSubsetSketchAccessor::new(
            vec![2, 5, 9],
            16,
            2,
            vec![20.0, 21.0, 50.0, 51.0, 90.0, 91.0],
        )
        .unwrap();

        let mut row = vec![0.0; 2];
        accessor.row_copy_into(5, &mut row).unwrap();
        assert_eq!(row, vec![50.0, 51.0]);

        let mut rows = vec![0.0; 6];
        accessor.read_rows_into(&[9, 2, 5], &mut rows).unwrap();
        assert_eq!(rows, vec![90.0, 91.0, 20.0, 21.0, 50.0, 51.0]);

        let mut stats = WindowedGatherStats::default();
        accessor
            .read_rows_windowed_into_stats(
                &[2, 9],
                &mut rows[..4],
                &WindowedGatherOptions::default(),
                &mut stats,
            )
            .unwrap();
        assert_eq!(&rows[..4], &[20.0, 21.0, 90.0, 91.0]);
        assert_eq!(stats.rows_requested, 2);
        assert_eq!(stats.logical_bytes, 16);
        assert_eq!(stats.physical_bytes, 0);
    }
}
