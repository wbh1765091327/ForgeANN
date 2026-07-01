use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::leaf_build::{PENDING_EDGE_DIRECT, PENDING_EDGE_MIRROR, PendingEdge, PendingEdgeSink};
use crate::common::{AnnError, AnnResult, Metric};

const RUNTIME_OVERLAY_MAX_SLOTS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpineOverlayPruneOptions {
    pub max_degree: usize,
    pub metric: Metric,
    pub overlay_budget: usize,
    pub overlay_spine_fraction: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpineOverlayPruneStats {
    pub raw_edges: usize,
    pub merged_candidates: usize,
    pub overfull_rows: usize,
    pub final_edges: usize,
    pub direct_ballots: usize,
    pub mirror_candidates: usize,
    pub rows_with_overlay: usize,
    pub overlay_edges_added: usize,
    pub base_refill_edges: usize,
    pub reduce_wall: Duration,
}

#[derive(Clone, Debug, PartialEq)]
struct SpineOverlayCandidate {
    dst: u32,
    dist2: f32,
    family_support: u16,
    reciprocal_support: u16,
    rank_mass: u32,
    best_rank: u8,
    distance_rank: usize,
}

pub(crate) trait SpineOverlayEdgeRecorder: Sync {
    fn write_edges(&self, view_family: u32, edges: &[PendingEdge]) -> AnnResult<()>;
}

pub(crate) struct SpineOverlayTaggedEdgeSink<'a> {
    recorder: &'a dyn SpineOverlayEdgeRecorder,
    view_family: u32,
    inner: &'a dyn PendingEdgeSink,
}

impl<'a> SpineOverlayTaggedEdgeSink<'a> {
    pub(crate) fn new(
        recorder: &'a dyn SpineOverlayEdgeRecorder,
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

impl PendingEdgeSink for SpineOverlayTaggedEdgeSink<'_> {
    fn flush_pending_edges(&self, edges: &mut Vec<PendingEdge>) -> AnnResult<()> {
        self.recorder.write_edges(self.view_family, edges)?;
        self.inner.flush_pending_edges(edges)
    }
}

#[derive(Debug)]
pub struct RuntimeSpineOverlayAccumulator {
    rows: Vec<Mutex<RuntimeSpineOverlayRow>>,
    capacity: usize,
    records: AtomicUsize,
    direct_records: AtomicUsize,
    mirror_records: AtomicUsize,
}

#[derive(Debug)]
struct RuntimeSpineOverlayRow {
    len: u8,
    slots: [RuntimeOverlayCandidate; RUNTIME_OVERLAY_MAX_SLOTS],
}

#[derive(Clone, Copy, Debug)]
struct RuntimeOverlayUpdate {
    dst: u32,
    dist2: f32,
    family: u32,
    local_rank: u8,
    direct: bool,
    reciprocal: bool,
}

#[derive(Clone, Copy, Debug)]
struct RuntimeOverlayCandidate {
    dst: u32,
    dist2: f32,
    family_support: u16,
    reciprocal_support: u16,
    rank_mass: u32,
    best_rank: u8,
    family_bits: u64,
    reciprocal_bits: u64,
}

impl RuntimeSpineOverlayAccumulator {
    pub fn create(num_points: usize, options: SpineOverlayPruneOptions) -> AnnResult<Self> {
        validate_spine_overlay_options(&options)?;
        let capacity = runtime_overlay_capacity(options);
        let rows = (0..num_points)
            .map(|_| Mutex::new(RuntimeSpineOverlayRow::default()))
            .collect();
        Ok(Self {
            rows,
            capacity,
            records: AtomicUsize::new(0),
            direct_records: AtomicUsize::new(0),
            mirror_records: AtomicUsize::new(0),
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl SpineOverlayEdgeRecorder for RuntimeSpineOverlayAccumulator {
    fn write_edges(&self, view_family: u32, edges: &[PendingEdge]) -> AnnResult<()> {
        if edges.is_empty() {
            return Ok(());
        }

        let mut updates = Vec::with_capacity(edges.len());
        let mut direct = 0usize;
        let mut mirror = 0usize;
        for edge in edges {
            if edge.flags & PENDING_EDGE_DIRECT != 0 {
                direct += 1;
                if edge.p < self.rows.len() {
                    updates.push((
                        edge.p,
                        RuntimeOverlayUpdate {
                            dst: edge.c,
                            dist2: edge.dist,
                            family: view_family,
                            local_rank: edge.local_rank.max(1),
                            direct: true,
                            reciprocal: false,
                        },
                    ));
                }

                let reciprocal_row = edge.c as usize;
                if reciprocal_row < self.rows.len() {
                    let dst = u32::try_from(edge.p).map_err(|_| {
                        AnnError::log_index_error(format!(
                            "SpineOverlay runtime accumulator supports only u32 source ids, got {}",
                            edge.p
                        ))
                    })?;
                    updates.push((
                        reciprocal_row,
                        RuntimeOverlayUpdate {
                            dst,
                            dist2: edge.dist,
                            family: view_family,
                            local_rank: edge.local_rank.max(1),
                            direct: false,
                            reciprocal: true,
                        },
                    ));
                }
            }
            if edge.flags & PENDING_EDGE_MIRROR != 0 {
                mirror += 1;
            }
        }

        updates.sort_unstable_by_key(|(row, update)| (*row, update.dst));
        let mut idx = 0usize;
        while idx < updates.len() {
            let row_id = updates[idx].0;
            let mut row = self.rows[row_id].lock();
            while idx < updates.len() && updates[idx].0 == row_id {
                row.insert(updates[idx].1, self.capacity);
                idx += 1;
            }
        }

        self.records.fetch_add(edges.len(), AtomicOrdering::Relaxed);
        self.direct_records
            .fetch_add(direct, AtomicOrdering::Relaxed);
        self.mirror_records
            .fetch_add(mirror, AtomicOrdering::Relaxed);
        Ok(())
    }
}

impl Default for RuntimeSpineOverlayRow {
    fn default() -> Self {
        Self {
            len: 0,
            slots: [RuntimeOverlayCandidate::empty(); RUNTIME_OVERLAY_MAX_SLOTS],
        }
    }
}

impl RuntimeSpineOverlayRow {
    fn insert(&mut self, update: RuntimeOverlayUpdate, capacity: usize) {
        let capacity = capacity.min(RUNTIME_OVERLAY_MAX_SLOTS);
        if capacity == 0 {
            return;
        }
        let len = self.len as usize;
        if let Some(candidate) = self
            .slots
            .iter_mut()
            .take(len)
            .find(|candidate| candidate.dst == update.dst)
        {
            candidate.apply(update);
            return;
        }

        let candidate = RuntimeOverlayCandidate::new(update);
        if len < capacity {
            self.slots[len] = candidate;
            self.len = self.len.saturating_add(1);
            return;
        }

        let Some((worst_idx, worst)) = self
            .slots
            .iter()
            .take(len)
            .enumerate()
            .max_by(|(_, left), (_, right)| runtime_retention_cmp(left, right))
        else {
            return;
        };
        if runtime_retention_cmp(&candidate, worst).is_lt() {
            self.slots[worst_idx] = candidate;
        }
    }
}

impl RuntimeOverlayCandidate {
    const fn empty() -> Self {
        Self {
            dst: u32::MAX,
            dist2: f32::INFINITY,
            family_support: 0,
            reciprocal_support: 0,
            rank_mass: 0,
            best_rank: u8::MAX,
            family_bits: 0,
            reciprocal_bits: 0,
        }
    }

    fn new(update: RuntimeOverlayUpdate) -> Self {
        let mut candidate = Self::empty();
        candidate.dst = update.dst;
        candidate.apply(update);
        candidate
    }

    fn apply(&mut self, update: RuntimeOverlayUpdate) {
        self.dist2 = self.dist2.min(update.dist2);
        let bit = family_bit(update.family);
        if update.direct && self.family_bits & bit == 0 {
            self.family_bits |= bit;
            self.family_support = self.family_support.saturating_add(1);
            let rank = update.local_rank.max(1);
            self.rank_mass = self.rank_mass.saturating_add(rank_weight(rank));
            if rank < self.best_rank {
                self.best_rank = rank;
            }
        }
        if update.reciprocal && self.reciprocal_bits & bit == 0 {
            self.reciprocal_bits |= bit;
            self.reciprocal_support = self.reciprocal_support.saturating_add(1);
        }
    }

    fn into_candidate(self) -> SpineOverlayCandidate {
        SpineOverlayCandidate {
            dst: self.dst,
            dist2: self.dist2,
            family_support: self.family_support,
            reciprocal_support: self.reciprocal_support,
            rank_mass: self.rank_mass,
            best_rank: self.best_rank,
            distance_rank: 0,
        }
    }
}

pub fn validate_spine_overlay_options(options: &SpineOverlayPruneOptions) -> AnnResult<()> {
    if options.metric != Metric::L2 {
        return Err(AnnError::log_index_config_error(
            "spine_overlay_prune_enable".to_string(),
            "SpineOverlayPrune currently supports only L2".to_string(),
        ));
    }
    if options.max_degree == 0 {
        return Err(AnnError::log_index_config_error(
            "max_degree".to_string(),
            "SpineOverlayPrune requires positive max-degree".to_string(),
        ));
    }
    if options.overlay_budget > options.max_degree {
        return Err(AnnError::log_index_config_error(
            "spine_overlay_budget".to_string(),
            "spine-overlay-budget must be <= max-degree".to_string(),
        ));
    }
    if !options.overlay_spine_fraction.is_finite()
        || options.overlay_spine_fraction < 0.0
        || options.overlay_spine_fraction > 1.0
    {
        return Err(AnnError::log_index_config_error(
            "spine_overlay_spine_fraction".to_string(),
            "spine-overlay-spine-fraction must be in [0, 1]".to_string(),
        ));
    }
    Ok(())
}

pub fn build_spine_overlay_prune_graph_from_runtime_accumulator<F>(
    accumulator: RuntimeSpineOverlayAccumulator,
    base_rows: &[Vec<u32>],
    num_points: usize,
    options: SpineOverlayPruneOptions,
    mut write_row: F,
) -> AnnResult<SpineOverlayPruneStats>
where
    F: FnMut(u32, &[u32]) -> AnnResult<()>,
{
    validate_spine_overlay_options(&options)?;
    if base_rows.len() != num_points {
        return Err(AnnError::log_index_error(format!(
            "SpineOverlayPrune base row count {} does not match num_points {}",
            base_rows.len(),
            num_points
        )));
    }
    if accumulator.rows.len() != num_points {
        return Err(AnnError::log_index_error(format!(
            "SpineOverlayPrune runtime row count {} does not match num_points {}",
            accumulator.rows.len(),
            num_points
        )));
    }

    let reduce_start = Instant::now();
    let mut stats = SpineOverlayPruneStats {
        raw_edges: accumulator.records.load(AtomicOrdering::Relaxed),
        direct_ballots: accumulator.direct_records.load(AtomicOrdering::Relaxed),
        mirror_candidates: accumulator.mirror_records.load(AtomicOrdering::Relaxed),
        ..SpineOverlayPruneStats::default()
    };

    for (vid, row_mutex) in accumulator.rows.into_iter().enumerate() {
        let row_state = row_mutex.into_inner();
        let mut row = row_state.slots[..row_state.len as usize].to_vec();
        row.sort_by(|a, b| {
            a.dist2
                .partial_cmp(&b.dist2)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.dst.cmp(&b.dst))
        });
        let mut candidates = row
            .into_iter()
            .map(RuntimeOverlayCandidate::into_candidate)
            .collect::<Vec<_>>();
        for (rank, candidate) in candidates.iter_mut().enumerate() {
            candidate.distance_rank = rank;
        }
        stats.merged_candidates = stats.merged_candidates.saturating_add(candidates.len());
        if base_rows[vid].len() > options.max_degree {
            stats.overfull_rows = stats.overfull_rows.saturating_add(1);
        }

        let overlay = select_runtime_spine_overlay_candidates(&candidates, options.max_degree);
        let (selected, overlay_added, base_refill) =
            select_spine_overlay_row_with_stats(&base_rows[vid], &overlay, options);
        if overlay_added > 0 {
            stats.rows_with_overlay = stats.rows_with_overlay.saturating_add(1);
        }
        stats.overlay_edges_added = stats.overlay_edges_added.saturating_add(overlay_added);
        stats.base_refill_edges = stats.base_refill_edges.saturating_add(base_refill);
        stats.final_edges = stats.final_edges.saturating_add(selected.len());
        write_row(vid as u32, &selected)?;
    }
    stats.reduce_wall = reduce_start.elapsed();
    Ok(stats)
}

fn rank_weight(rank: u8) -> u32 {
    match rank {
        1 => 6,
        2 => 3,
        3 => 2,
        _ => 1,
    }
}

fn runtime_overlay_capacity(options: SpineOverlayPruneOptions) -> usize {
    options
        .overlay_budget
        .saturating_mul(8)
        .max(RUNTIME_OVERLAY_MAX_SLOTS)
        .min(RUNTIME_OVERLAY_MAX_SLOTS)
        .min(options.max_degree.max(1))
}

fn family_bit(family: u32) -> u64 {
    let hash = family
        .wrapping_mul(0x9E37_79B1)
        .rotate_left(13)
        .wrapping_mul(0x85EB_CA6B);
    1_u64 << (hash & 63)
}

fn runtime_retention_cmp(a: &RuntimeOverlayCandidate, b: &RuntimeOverlayCandidate) -> Ordering {
    b.family_support
        .cmp(&a.family_support)
        .then_with(|| b.rank_mass.cmp(&a.rank_mass))
        .then_with(|| a.best_rank.cmp(&b.best_rank))
        .then_with(|| b.reciprocal_support.cmp(&a.reciprocal_support))
        .then_with(|| a.dist2.partial_cmp(&b.dist2).unwrap_or(Ordering::Equal))
        .then_with(|| a.dst.cmp(&b.dst))
}

fn select_runtime_spine_overlay_candidates(
    row: &[SpineOverlayCandidate],
    max_degree: usize,
) -> Vec<u32> {
    let mut ranked = row.to_vec();
    ranked.sort_by(|a, b| {
        stability_cmp(a, b)
            .then_with(|| a.distance_rank.cmp(&b.distance_rank))
            .then_with(|| a.dst.cmp(&b.dst))
    });
    ranked
        .into_iter()
        .take(max_degree)
        .map(|candidate| candidate.dst)
        .collect()
}

#[cfg(test)]
pub fn select_spine_overlay_row(
    base_row: &[u32],
    overlay_row: &[u32],
    options: SpineOverlayPruneOptions,
) -> Vec<u32> {
    select_spine_overlay_row_with_stats(base_row, overlay_row, options).0
}

fn select_spine_overlay_row_with_stats(
    base_row: &[u32],
    overlay_row: &[u32],
    options: SpineOverlayPruneOptions,
) -> (Vec<u32>, usize, usize) {
    let max_degree = options.max_degree;
    let overlay_budget = options.overlay_budget.min(max_degree);
    let base_keep = max_degree.saturating_sub(overlay_budget);
    let mut selected = Vec::with_capacity(max_degree.min(base_row.len() + overlay_budget));

    for &dst in base_row.iter().take(base_keep) {
        push_unique_id(dst, &mut selected);
    }

    let mut overlay_added = 0usize;
    for &dst in overlay_row {
        if selected.len() >= max_degree || overlay_added >= overlay_budget {
            break;
        }
        if push_unique_id(dst, &mut selected) {
            overlay_added += 1;
        }
    }

    let mut base_refill = 0usize;
    for &dst in base_row.iter().skip(base_keep) {
        if selected.len() >= max_degree {
            break;
        }
        if push_unique_id(dst, &mut selected) {
            base_refill += 1;
        }
    }

    (selected, overlay_added, base_refill)
}

fn push_unique_id(dst: u32, selected: &mut Vec<u32>) -> bool {
    if selected.contains(&dst) {
        return false;
    }
    selected.push(dst);
    true
}

fn stability_cmp(a: &SpineOverlayCandidate, b: &SpineOverlayCandidate) -> Ordering {
    b.family_support
        .cmp(&a.family_support)
        .then_with(|| b.reciprocal_support.cmp(&a.reciprocal_support))
        .then_with(|| b.rank_mass.cmp(&a.rank_mass))
        .then_with(|| a.best_rank.cmp(&b.best_rank))
        .then_with(|| a.dist2.partial_cmp(&b.dist2).unwrap_or(Ordering::Equal))
        .then_with(|| a.dst.cmp(&b.dst))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> SpineOverlayPruneOptions {
        SpineOverlayPruneOptions {
            max_degree: 5,
            metric: Metric::L2,
            overlay_budget: 2,
            overlay_spine_fraction: 0.8,
        }
    }

    #[test]
    fn spine_overlay_preserves_base_prefix_then_adds_overlay_tail() {
        let selected = select_spine_overlay_row(&[1, 2, 3, 4, 5], &[9, 2, 8, 7], options());
        assert_eq!(selected, vec![1, 2, 3, 9, 8]);
    }

    #[test]
    fn spine_overlay_refills_from_base_tail_when_overlay_duplicates() {
        let selected = select_spine_overlay_row(&[1, 2, 3, 4, 5], &[1, 2, 9], options());
        assert_eq!(selected, vec![1, 2, 3, 9, 4]);
    }

    #[test]
    fn spine_overlay_budget_zero_returns_base_prefix() {
        let selected = select_spine_overlay_row(
            &[1, 2, 3, 4],
            &[9, 8],
            SpineOverlayPruneOptions {
                max_degree: 3,
                metric: Metric::L2,
                overlay_budget: 0,
                overlay_spine_fraction: 0.8,
            },
        );
        assert_eq!(selected, vec![1, 2, 3]);
    }

    #[test]
    fn runtime_accumulator_counts_direct_and_reciprocal_updates() {
        let accumulator = RuntimeSpineOverlayAccumulator::create(
            4,
            SpineOverlayPruneOptions {
                max_degree: 3,
                metric: Metric::L2,
                overlay_budget: 1,
                overlay_spine_fraction: 0.8,
            },
        )
        .unwrap();
        accumulator
            .write_edges(
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
                        p: 1,
                        c: 0,
                        hash: 0,
                        dist: 1.0,
                        mandatory: false,
                        local_rank: 1,
                        flags: PENDING_EDGE_MIRROR,
                    },
                ],
            )
            .unwrap();
        let base_rows = vec![vec![2, 3, 1], vec![2, 3, 0], vec![], vec![]];
        let mut rows = Vec::new();
        let stats = build_spine_overlay_prune_graph_from_runtime_accumulator(
            accumulator,
            &base_rows,
            4,
            SpineOverlayPruneOptions {
                max_degree: 3,
                metric: Metric::L2,
                overlay_budget: 1,
                overlay_spine_fraction: 0.8,
            },
            |vid, row| {
                rows.push((vid, row.to_vec()));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(stats.direct_ballots, 1);
        assert_eq!(stats.mirror_candidates, 1);
        assert_eq!(stats.overlay_edges_added, 2);
        assert_eq!(rows[0].1, vec![2, 3, 1]);
        assert_eq!(rows[1].1, vec![2, 3, 0]);
    }
}
