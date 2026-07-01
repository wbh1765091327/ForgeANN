use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use crate::common::{AnnError, AnnResult, Metric};
use crate::model::graph::AdjacencyList;
use crate::model::{InmemDataset, Neighbor};

pub(crate) const FINAL_PRUNE_ALPHA: f32 = 1.2;
pub(crate) const FINAL_PRUNE_MAX_OCCLUSION_SIZE: usize = 750;
pub(crate) const FINAL_PRUNE_SATURATE: bool = true;

pub(crate) fn final_robust_prune_rows(
    dataset: &InmemDataset<f32>,
    graph_rows: &mut [Vec<u32>],
    max_degree: usize,
    metric: Metric,
    num_threads: u32,
) -> AnnResult<()> {
    let prune_one = |(vid, row): (usize, &mut Vec<u32>)| -> AnnResult<()> {
        if row.is_empty() {
            return Ok(());
        }

        let vid_u32 = vid as u32;
        let query = dataset.get_vertex(vid_u32)?;
        let mut pool = Vec::with_capacity(row.len());
        for nid in row.iter().copied() {
            let vertex = dataset.get_vertex(nid)?;
            pool.push(Neighbor::new(nid, query.compare(&vertex, metric)));
        }

        let pruned = robust_prune_candidates(
            dataset,
            vid_u32,
            &mut pool,
            max_degree,
            FINAL_PRUNE_MAX_OCCLUSION_SIZE,
            FINAL_PRUNE_ALPHA,
            FINAL_PRUNE_SATURATE,
            metric,
        )?;
        *row = pruned.edges;
        Ok(())
    };

    if num_threads == 1 {
        for item in graph_rows.iter_mut().enumerate() {
            prune_one(item)?;
        }
        return Ok(());
    }

    let mut run_parallel = || -> AnnResult<()> {
        graph_rows
            .par_iter_mut()
            .enumerate()
            .map(prune_one)
            .collect::<AnnResult<Vec<_>>>()
            .map(|_| ())
    };

    if num_threads == 0 {
        run_parallel()
    } else {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .map_err(|err| {
                AnnError::log_index_error(format!(
                    "Failed to create final prune Rayon thread pool: {err}"
                ))
            })?;
        pool.install(run_parallel)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn robust_prune_candidates(
    dataset: &InmemDataset<f32>,
    location: u32,
    pool: &mut Vec<Neighbor>,
    max_degree: usize,
    max_candidate_size: usize,
    alpha: f32,
    saturate_graph: bool,
    metric: Metric,
) -> AnnResult<AdjacencyList> {
    let mut result = AdjacencyList::for_range(max_degree);
    if pool.is_empty() {
        return Ok(result);
    }

    pool.sort_unstable();
    if pool.len() > max_candidate_size {
        pool.truncate(max_candidate_size);
    }

    let mut occlude_factor = vec![0.0; pool.len()];
    let mut cur_alpha = 1.0;
    while cur_alpha <= alpha && result.len() < max_degree {
        for i in 0..pool.len() {
            if result.len() >= max_degree {
                break;
            }
            if occlude_factor[i] > cur_alpha {
                continue;
            }

            occlude_factor[i] = f32::MAX;
            let neighbor = pool[i];
            if neighbor.id != location {
                result.push(neighbor.id);
            }

            for j in i + 1..pool.len() {
                if occlude_factor[j] > alpha {
                    continue;
                }
                let neighbor2 = pool[j];
                let djk = dataset.get_distance(neighbor2.id, neighbor.id, metric)?;
                match metric {
                    Metric::L2 | Metric::Cosine => {
                        occlude_factor[j] = if djk == 0.0 {
                            f32::MAX
                        } else {
                            occlude_factor[j].max(neighbor2.distance / djk)
                        };
                    }
                    _ => {
                        return Err(AnnError::log_index_config_error(
                            "metric".to_string(),
                            "ForgeANN final prune supports only L2/Cosine".to_string(),
                        ));
                    }
                }
            }
        }
        cur_alpha *= 1.2;
    }

    if saturate_graph && alpha > 1.0 {
        for neighbor in pool.iter() {
            if result.len() >= max_degree {
                break;
            }
            if neighbor.id != location && !result.contains(&neighbor.id) {
                result.push(neighbor.id);
            }
        }
    }

    if result.len() > max_degree {
        return Err(AnnError::log_index_error(format!(
            "final prune produced {} neighbors over max degree {}",
            result.len(),
            max_degree
        )));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::InmemDataset;

    fn line_dataset(rows: usize) -> InmemDataset<f32> {
        let dim = 16;
        let mut dataset = InmemDataset::<f32>::new(rows, 1.0, dim).unwrap();
        for row in 0..rows {
            dataset.data[row * dim] = row as f32;
        }
        dataset.num_active_pts = rows;
        dataset
    }

    #[test]
    fn final_prune_removes_self_loop_and_saturates_from_remaining_candidates() {
        let dataset = line_dataset(5);
        let mut pool = vec![
            Neighbor::new(2, 0.0),
            Neighbor::new(1, 1.0),
            Neighbor::new(3, 1.0),
            Neighbor::new(4, 2.0),
        ];

        let result =
            robust_prune_candidates(&dataset, 2, &mut pool, 3, 750, 1.2, true, Metric::L2).unwrap();

        assert_eq!(result.as_slice(), &[1, 3, 4]);
    }

    #[test]
    fn final_prune_honors_max_candidate_size_before_saturating() {
        let dataset = line_dataset(5);
        let mut pool = vec![
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 2.0),
            Neighbor::new(3, 3.0),
            Neighbor::new(4, 4.0),
        ];

        let result =
            robust_prune_candidates(&dataset, 0, &mut pool, 4, 2, 1.2, true, Metric::L2).unwrap();

        assert!(result.as_slice().iter().all(|id| *id <= 2));
        assert!(result.len() <= 2);
    }

    #[test]
    fn final_prune_deduplicates_candidate_ids() {
        let dataset = line_dataset(3);
        let mut pool = vec![Neighbor::new(1, 1.0), Neighbor::new(1, 1.0)];

        let result =
            robust_prune_candidates(&dataset, 0, &mut pool, 4, 750, 1.2, true, Metric::L2).unwrap();

        assert_eq!(result.as_slice(), &[1]);
    }

    #[test]
    fn final_prune_without_saturate_keeps_occlusion_result_only() {
        let dim = 16;
        let mut dataset = InmemDataset::<f32>::new(4, 1.0, dim).unwrap();
        dataset.data[dim] = 1.0;
        dataset.data[2 * dim] = 1.0;
        dataset.data[3 * dim] = 3.0;
        dataset.num_active_pts = 4;
        let mut pool = vec![
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 1.0),
            Neighbor::new(3, 9.0),
        ];

        let result =
            robust_prune_candidates(&dataset, 0, &mut pool, 4, 750, 1.2, false, Metric::L2)
                .unwrap();

        assert_eq!(result.as_slice(), &[1]);
    }
}
