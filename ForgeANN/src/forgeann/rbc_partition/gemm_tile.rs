use super::*;

pub(crate) fn choose_gemm_tile_sizes_for_budget(
    num_leaders: usize,
    dim: usize,
    memory_budget_bytes: Option<usize>,
) -> (usize, usize) {
    const MAX_POINT_TILE: usize = 8192;
    const MIN_POINT_TILE: usize = 256;
    const MAX_LEADER_TILE: usize = 512;
    const TARGET_BYTES: usize = 64 * 1024 * 1024;

    let num_leaders = num_leaders.max(1);
    let leader_tile = num_leaders.min(MAX_LEADER_TILE);
    let target_bytes = memory_budget_bytes
        .filter(|&budget| budget > 0)
        .map(|budget| {
            const OOM_TARGET_BYTES_CAP: usize = 32 * 1024 * 1024;
            let reserve = budget / 256;
            reserve
                .clamp(8 * 1024 * 1024, OOM_TARGET_BYTES_CAP)
                .max(8 * 1024 * 1024)
        })
        .unwrap_or(TARGET_BYTES);
    let bytes_per_point = dim
        .saturating_mul(size_of::<f32>())
        .saturating_mul(2)
        .saturating_add(leader_tile.saturating_mul(size_of::<f32>()));
    let point_tile = if bytes_per_point == 0 {
        MAX_POINT_TILE
    } else {
        (target_bytes / bytes_per_point).clamp(MIN_POINT_TILE, MAX_POINT_TILE)
    };

    (point_tile, leader_tile)
}

pub(crate) fn parse_gemm_tile_override_value(value: &str) -> Option<(usize, usize)> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    let (point_tile, leader_tile) = trimmed.split_once('x')?;
    let point_tile = point_tile.trim().parse::<usize>().ok()?;
    let leader_tile = leader_tile.trim().parse::<usize>().ok()?;
    if point_tile == 0 || leader_tile == 0 {
        return None;
    }

    Some((point_tile, leader_tile))
}

#[inline]
fn choose_gemm_tile_sizes_with_override_value(
    override_value: Option<&str>,
    num_leaders: usize,
    dim: usize,
    memory_budget_bytes: Option<usize>,
) -> (usize, usize) {
    if let Some((point_tile, leader_tile)) = override_value.and_then(parse_gemm_tile_override_value)
    {
        let point_tile = point_tile.clamp(256, 8192);
        let leader_tile = leader_tile.min(num_leaders.max(1)).min(512).max(1);
        return (point_tile, leader_tile);
    }

    choose_gemm_tile_sizes_for_budget(num_leaders, dim, memory_budget_bytes)
}

#[inline]
pub(crate) fn choose_gemm_tile_sizes_with_override_and_budget(
    num_leaders: usize,
    dim: usize,
    memory_budget_bytes: Option<usize>,
) -> (usize, usize) {
    let override_value = std::env::var("FORGEANN_GEMM_TILE").ok();
    choose_gemm_tile_sizes_with_override_value(
        override_value.as_deref(),
        num_leaders,
        dim,
        memory_budget_bytes,
    )
}

#[inline]
pub(crate) fn update_block_topk_from_gram_tile(
    topk_rows: &mut [StackTopK],
    x_norms: &[f32],
    l_norms: &[f32],
    gram_tile: ArrayView2<'_, f32>,
    leader_offset: usize,
) {
    for (row_idx, topk) in topk_rows.iter_mut().enumerate() {
        let x_norm = x_norms[row_idx];
        for (local_leader_idx, &gram) in gram_tile.row(row_idx).iter().enumerate() {
            let leader_idx = leader_offset + local_leader_idx;
            let dist2 = (x_norm + l_norms[leader_idx] - 2.0 * gram).max(0.0);
            topk.push(dist2, leader_idx);
        }
    }
}

#[inline]
pub(crate) fn update_block_best_from_gram_tile(
    best_distances: &mut [f32],
    best_leaders: &mut [usize],
    x_norms: &[f32],
    l_norms: &[f32],
    gram_tile: ArrayView2<'_, f32>,
    leader_offset: usize,
) {
    for row_idx in 0..best_distances.len() {
        let x_norm = x_norms[row_idx];
        let row = gram_tile.row(row_idx);
        for (local_leader_idx, &gram) in row.iter().enumerate() {
            let leader_idx = leader_offset + local_leader_idx;
            let dist2 = (x_norm + l_norms[leader_idx] - 2.0 * gram).max(0.0);
            if dist2 < best_distances[row_idx] {
                best_distances[row_idx] = dist2;
                best_leaders[row_idx] = leader_idx;
            }
        }
    }
}
