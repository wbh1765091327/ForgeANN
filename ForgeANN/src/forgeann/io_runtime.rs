use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const DEFAULT_MAX_READ_AMPLIFICATION: f64 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadWindow {
    pub(crate) start_row: u32,
    pub(crate) row_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadScatter {
    pub(crate) window_idx: usize,
    pub(crate) row_offset_in_window: u32,
    pub(crate) original_pos: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BoundedReadPlan {
    pub(crate) windows: Vec<ReadWindow>,
    pub(crate) scatter: Vec<ReadScatter>,
    requested_rows: usize,
    physical_rows: usize,
    physical_bytes: u64,
    row_bytes: usize,
}

impl BoundedReadPlan {
    pub(crate) fn requested_rows(&self) -> usize {
        self.requested_rows
    }

    pub(crate) fn physical_rows(&self) -> usize {
        self.physical_rows
    }

    pub(crate) fn logical_bytes(&self) -> u64 {
        self.requested_rows.saturating_mul(self.row_bytes) as u64
    }

    pub(crate) fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }

    #[allow(dead_code)]
    pub(crate) fn read_amplification(&self) -> f64 {
        let logical = self.logical_bytes();
        if logical == 0 {
            0.0
        } else {
            self.physical_bytes() as f64 / logical as f64
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BoundedReadPlanner {
    pub(crate) row_bytes: usize,
    pub(crate) max_window_bytes: usize,
    pub(crate) max_read_amplification: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedRow {
    row_id: u32,
    original_pos: usize,
}

impl BoundedReadPlanner {
    pub(crate) fn plan(&self, ids: &[u32]) -> BoundedReadPlan {
        let row_bytes = self.row_bytes.max(1);
        let max_window_rows = (self.max_window_bytes.max(row_bytes) / row_bytes).max(1);
        let max_amp =
            if self.max_read_amplification.is_finite() && self.max_read_amplification >= 1.0 {
                self.max_read_amplification
            } else {
                DEFAULT_MAX_READ_AMPLIFICATION
            };

        let mut rows: Vec<PlannedRow> = ids
            .iter()
            .enumerate()
            .map(|(original_pos, &row_id)| PlannedRow {
                row_id,
                original_pos,
            })
            .collect();
        rows.sort_unstable_by_key(|row| row.row_id);

        let mut plan = BoundedReadPlan {
            row_bytes,
            requested_rows: ids.len(),
            ..BoundedReadPlan::default()
        };
        let mut start = 0usize;
        while start < rows.len() {
            let mut end = start + 1;
            while end < rows.len() {
                let start_row = rows[start].row_id;
                let end_row = rows[end].row_id;
                let physical_rows = end_row.saturating_sub(start_row) as usize + 1;
                let requested_rows = end - start + 1;
                let amplification = physical_rows as f64 / requested_rows as f64;
                if physical_rows > max_window_rows || amplification > max_amp {
                    break;
                }
                end += 1;
            }

            let window_idx = plan.windows.len();
            let start_row = rows[start].row_id;
            let end_row = rows[end - 1].row_id;
            let row_count = end_row.saturating_sub(start_row) + 1;
            plan.windows.push(ReadWindow {
                start_row,
                row_count,
            });
            plan.physical_rows += row_count as usize;
            plan.physical_bytes += row_count as u64 * row_bytes as u64;
            for row in &rows[start..end] {
                plan.scatter.push(ReadScatter {
                    window_idx,
                    row_offset_in_window: row.row_id - start_row,
                    original_pos: row.original_pos,
                });
            }
            start = end;
        }

        plan
    }

    pub(crate) fn plan_with_aligned_read_cost(
        &self,
        ids: &[u32],
        alignment_bytes: usize,
        header_bytes: u64,
    ) -> BoundedReadPlan {
        let row_bytes = self.row_bytes.max(1);
        let alignment_bytes = alignment_bytes.max(1);
        let max_window_bytes = self.max_window_bytes.max(row_bytes).max(alignment_bytes);
        let max_amp =
            if self.max_read_amplification.is_finite() && self.max_read_amplification >= 1.0 {
                self.max_read_amplification
            } else {
                DEFAULT_MAX_READ_AMPLIFICATION
            };

        let mut rows: Vec<PlannedRow> = ids
            .iter()
            .enumerate()
            .map(|(original_pos, &row_id)| PlannedRow {
                row_id,
                original_pos,
            })
            .collect();
        rows.sort_unstable_by_key(|row| row.row_id);

        let mut plan = BoundedReadPlan {
            row_bytes,
            requested_rows: ids.len(),
            ..BoundedReadPlan::default()
        };
        let mut start = 0usize;
        while start < rows.len() {
            let start_row = rows[start].row_id;
            let mut end = start + 1;
            let mut touched_blocks =
                aligned_blocks_for_row(start_row, row_bytes, alignment_bytes, header_bytes);
            let mut touched_block_count = touched_blocks.end - touched_blocks.start;
            let mut best_end = end;
            let mut best_physical_bytes = aligned_read_bytes_for_rows(
                start_row,
                start_row,
                row_bytes,
                alignment_bytes,
                header_bytes,
            );

            while end < rows.len() {
                let end_row = rows[end].row_id;
                let physical_bytes = aligned_read_bytes_for_rows(
                    start_row,
                    end_row,
                    row_bytes,
                    alignment_bytes,
                    header_bytes,
                );
                if physical_bytes > max_window_bytes as u64 {
                    break;
                }

                let next_blocks =
                    aligned_blocks_for_row(end_row, row_bytes, alignment_bytes, header_bytes);
                let extra_blocks = next_blocks
                    .end
                    .saturating_sub(touched_blocks.end.max(next_blocks.start));
                let candidate_touched_block_count =
                    touched_block_count.saturating_add(extra_blocks);
                let mandatory_bytes =
                    candidate_touched_block_count.saturating_mul(alignment_bytes as u64);
                let amplification = if mandatory_bytes == 0 {
                    1.0
                } else {
                    physical_bytes as f64 / mandatory_bytes as f64
                };
                if amplification > max_amp {
                    break;
                }

                touched_blocks.end = touched_blocks.end.max(next_blocks.end);
                touched_block_count = candidate_touched_block_count;
                end += 1;
                best_end = end;
                best_physical_bytes = physical_bytes;
            }

            let window_idx = plan.windows.len();
            let end_row = rows[best_end - 1].row_id;
            let row_count = end_row.saturating_sub(start_row) + 1;
            plan.windows.push(ReadWindow {
                start_row,
                row_count,
            });
            plan.physical_rows += row_count as usize;
            plan.physical_bytes += best_physical_bytes;
            for row in &rows[start..best_end] {
                plan.scatter.push(ReadScatter {
                    window_idx,
                    row_offset_in_window: row.row_id - start_row,
                    original_pos: row.original_pos,
                });
            }
            start = best_end;
        }

        plan
    }
}

#[derive(Clone, Copy, Debug)]
struct AlignedBlockRange {
    start: u64,
    end: u64,
}

fn aligned_blocks_for_row(
    row_id: u32,
    row_bytes: usize,
    alignment_bytes: usize,
    header_bytes: u64,
) -> AlignedBlockRange {
    let row_start = header_bytes + (row_id as u64).saturating_mul(row_bytes as u64);
    let row_end = row_start.saturating_add(row_bytes as u64);
    let alignment = alignment_bytes as u64;
    AlignedBlockRange {
        start: row_start / alignment,
        end: row_end.div_ceil(alignment),
    }
}

fn aligned_read_bytes_for_rows(
    start_row: u32,
    end_row: u32,
    row_bytes: usize,
    alignment_bytes: usize,
    header_bytes: u64,
) -> u64 {
    let alignment = alignment_bytes as u64;
    let logical_offset = header_bytes + (start_row as u64).saturating_mul(row_bytes as u64);
    let row_count = end_row.saturating_sub(start_row) as u64 + 1;
    let logical_end = logical_offset.saturating_add(row_count.saturating_mul(row_bytes as u64));
    let aligned_offset = logical_offset / alignment * alignment;
    let aligned_end = logical_end.div_ceil(alignment) * alignment;
    aligned_end.saturating_sub(aligned_offset)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct VectorWindowKey {
    pub(crate) start_pid: u32,
    pub(crate) row_count: u32,
}

impl Hash for VectorWindowKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start_pid.hash(state);
        self.row_count.hash(state);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct VectorWindowCacheStats {
    pub(crate) hits: usize,
    pub(crate) misses: usize,
    pub(crate) inserts: usize,
    pub(crate) evictions: usize,
    pub(crate) used_peak_bytes: usize,
    pub(crate) bytes_inserted: usize,
    pub(crate) bytes_evicted: usize,
    pub(crate) saved_direct_read_calls: usize,
    pub(crate) logical_bytes: u64,
    pub(crate) physical_bytes: u64,
}

impl VectorWindowCacheStats {
    pub(crate) fn merge(&mut self, other: &Self) {
        self.hits += other.hits;
        self.misses += other.misses;
        self.inserts += other.inserts;
        self.evictions += other.evictions;
        self.used_peak_bytes = self.used_peak_bytes.max(other.used_peak_bytes);
        self.bytes_inserted = self.bytes_inserted.saturating_add(other.bytes_inserted);
        self.bytes_evicted = self.bytes_evicted.saturating_add(other.bytes_evicted);
        self.saved_direct_read_calls += other.saved_direct_read_calls;
        self.logical_bytes = self.logical_bytes.saturating_add(other.logical_bytes);
        self.physical_bytes = self.physical_bytes.saturating_add(other.physical_bytes);
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct VectorWindowEntry {
    data: Arc<[f32]>,
    bytes: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct VectorWindowCache {
    budget_bytes: usize,
    used_bytes: usize,
    entries: HashMap<VectorWindowKey, VectorWindowEntry>,
    order: VecDeque<VectorWindowKey>,
    stats: VectorWindowCacheStats,
}

impl VectorWindowCache {
    #[allow(dead_code)]
    pub(crate) fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
            stats: VectorWindowCacheStats::default(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    #[allow(dead_code)]
    pub(crate) fn stats(&self) -> &VectorWindowCacheStats {
        &self.stats
    }

    #[allow(dead_code)]
    pub(crate) fn get(&mut self, key: &VectorWindowKey) -> Option<Arc<[f32]>> {
        match self.entries.get(key) {
            Some(entry) => {
                self.stats.hits += 1;
                self.stats.saved_direct_read_calls += 1;
                self.order.retain(|candidate| candidate != key);
                self.order.push_back(*key);
                Some(Arc::clone(&entry.data))
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn insert(&mut self, key: VectorWindowKey, data: Arc<[f32]>) {
        let bytes = data.len().saturating_mul(std::mem::size_of::<f32>());
        if bytes == 0 || bytes > self.budget_bytes {
            return;
        }

        if let Some(previous) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(previous.bytes);
            self.order.retain(|candidate| candidate != &key);
        }

        self.entries.insert(key, VectorWindowEntry { data, bytes });
        self.order.push_back(key);
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        self.stats.inserts += 1;
        self.stats.bytes_inserted = self.stats.bytes_inserted.saturating_add(bytes);
        self.stats.used_peak_bytes = self.stats.used_peak_bytes.max(self.used_bytes);
        self.evict_to_budget();
        self.stats.used_peak_bytes = self.stats.used_peak_bytes.max(self.used_bytes);
    }

    fn evict_to_budget(&mut self) {
        while self.used_bytes > self.budget_bytes {
            let Some(victim) = self.order.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&victim) {
                self.used_bytes = self.used_bytes.saturating_sub(entry.bytes);
                self.stats.evictions += 1;
                self.stats.bytes_evicted = self.stats.bytes_evicted.saturating_add(entry.bytes);
            }
        }
    }
}
