pub(crate) fn sample_set_bottomk(ids: &[u32], k: usize, seed: u64) -> Vec<u32> {
    sample_positions_bottomk(ids, k, seed)
        .into_iter()
        .map(|idx| ids[idx])
        .collect()
}

pub(crate) fn sample_positions_bottomk(ids: &[u32], k: usize, seed: u64) -> Vec<usize> {
    if k == 0 || ids.is_empty() {
        return Vec::new();
    }

    let mut unique: Vec<(u32, usize)> = ids
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, id)| (id, idx))
        .collect();
    unique.sort_unstable_by_key(|&(id, idx)| (id, idx));
    unique.dedup_by_key(|(id, _)| *id);

    if k >= unique.len() {
        return unique.into_iter().map(|(_, idx)| idx).collect();
    }

    let mut scored: Vec<(u64, u32, usize)> = unique
        .into_iter()
        .map(|(id, idx)| (stable_sample_score(id, seed), id, idx))
        .collect();
    scored.select_nth_unstable_by(k, |left, right| (left.0, left.1).cmp(&(right.0, right.1)));
    scored.truncate(k);
    scored.sort_unstable_by_key(|&(score, id, _)| (score, id));
    scored.into_iter().map(|(_, _, idx)| idx).collect()
}

fn stable_sample_score(point_id: u32, seed: u64) -> u64 {
    splitmix64(seed ^ (point_id as u64).wrapping_mul(0x9e3779b97f4a7c15))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::sample_set_bottomk;

    #[test]
    fn bottomk_sampling_is_order_invariant_and_seed_deterministic() {
        let ids = vec![42, 7, 9, 100, 3, 81, 12, 5];
        let mut reversed = ids.clone();
        reversed.reverse();

        let sample = sample_set_bottomk(&ids, 4, 17);
        let reversed_sample = sample_set_bottomk(&reversed, 4, 17);
        assert_eq!(sample, reversed_sample);
        assert_eq!(sample, sample_set_bottomk(&ids, 4, 17));
        assert_ne!(sample, sample_set_bottomk(&ids, 4, 18));
    }

    #[test]
    fn bottomk_sampling_returns_all_ids_when_k_covers_set() {
        let ids = vec![8, 3, 8, 2, 1];
        assert_eq!(sample_set_bottomk(&ids, 8, 1), vec![1, 2, 3, 8]);
        assert!(sample_set_bottomk(&ids, 0, 1).is_empty());
    }
}
