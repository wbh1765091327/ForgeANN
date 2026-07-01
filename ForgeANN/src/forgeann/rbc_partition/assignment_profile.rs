#[cfg(test)]
use super::child_run_io::{
    AssignmentSpoolSegment, D1LevelScanHeapEntry, D1LevelScanRunBatchInput, D1LevelScanRunWork,
    build_d1_level_scan_compute_tasks, clamp_d1_level_scan_pipeline_inflight,
    compute_clusters_gemm_budgeted, compute_clusters_gemm_budgeted_with_context,
    compute_clusters_gemm_to_spool_budgeted_with_context, d1_ads_compute_chunk_points,
    d1_level_scan_batch_points, d1_level_scan_memory_budget_bytes, default_rbc_windowed_options,
    drain_ordered_d1_level_scan_batches,
    materialize_merged_children_from_spool_segments_with_inline_limit, next_d1_level_scan_batch,
    parallel_join_child_runs, parallel_join_child_runs_inline, read_assign_record,
    read_child_run_chain, schedule_depth_wave_child_runs,
    should_use_depth_wave_child_run_scheduler, write_assign_record, write_child_run,
};
#[cfg(test)]
use super::prefetch_pipeline::strict_prefetch_windowed_options;
use super::*;

pub const ASSIGNMENT_DEPTH_BUCKETS: usize = 5;
pub const ASSIGNMENT_LEADER_BUCKETS: usize = 6;
pub const ASSIGNMENT_FALLBACK_REASONS: usize = 15;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct AssignmentDecisionBucketProfile {
    pub(crate) nodes: usize,
    pub(crate) points: usize,
    pub(crate) work: u128,
    pub(crate) max_points: usize,
    pub(crate) max_leaders: usize,
    pub(crate) max_work: u128,
    pub(crate) gemm_nodes: usize,
    pub(crate) gemm_points: usize,
    pub(crate) gemm_wall: Duration,
    pub(crate) adsampling_nodes: usize,
    pub(crate) adsampling_points: usize,
    pub(crate) adsampling_wall: Duration,
    pub(crate) adsampling_recall_sum: f64,
    pub(crate) adsampling_recall_samples: usize,
    pub(crate) adsampling_mismatches: usize,
}

impl AssignmentDecisionBucketProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        self.nodes += other.nodes;
        self.points += other.points;
        self.work += other.work;
        self.max_points = self.max_points.max(other.max_points);
        self.max_leaders = self.max_leaders.max(other.max_leaders);
        self.max_work = self.max_work.max(other.max_work);
        self.gemm_nodes += other.gemm_nodes;
        self.gemm_points += other.gemm_points;
        self.gemm_wall += other.gemm_wall;
        self.adsampling_nodes += other.adsampling_nodes;
        self.adsampling_points += other.adsampling_points;
        self.adsampling_wall += other.adsampling_wall;
        self.adsampling_recall_sum += other.adsampling_recall_sum;
        self.adsampling_recall_samples += other.adsampling_recall_samples;
        self.adsampling_mismatches += other.adsampling_mismatches;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AssignmentDecisionProfile {
    pub(crate) depth_leader_buckets:
        [[AssignmentDecisionBucketProfile; ASSIGNMENT_LEADER_BUCKETS]; ASSIGNMENT_DEPTH_BUCKETS],
    pub(crate) fallback_reasons: [usize; ASSIGNMENT_FALLBACK_REASONS],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AdSamplingSchedulerStats {
    pub(crate) ads_large_tasks: usize,
    pub(crate) ads_chunks_total: usize,
    pub(crate) ads_waves: usize,
    pub(crate) ads_exact_fallback_small: usize,
    pub(crate) ads_exact_fallback_nested: usize,
    pub(crate) ads_parallelism_collapse_count: usize,
}

impl AdSamplingSchedulerStats {
    pub(crate) fn merge(&mut self, other: Self) {
        self.ads_large_tasks += other.ads_large_tasks;
        self.ads_chunks_total += other.ads_chunks_total;
        self.ads_waves += other.ads_waves;
        self.ads_exact_fallback_small += other.ads_exact_fallback_small;
        self.ads_exact_fallback_nested += other.ads_exact_fallback_nested;
        self.ads_parallelism_collapse_count += other.ads_parallelism_collapse_count;
    }
}

#[derive(Debug, Default)]
pub(crate) struct AdSamplingSchedulerRuntime {
    pub depth_ads_disabled: AtomicBool,
    pub ads_large_tasks: AtomicUsize,
    pub ads_chunks_total: AtomicUsize,
    pub ads_waves: AtomicUsize,
    pub ads_exact_fallback_small: AtomicUsize,
    pub ads_exact_fallback_nested: AtomicUsize,
    pub ads_parallelism_collapse_count: AtomicUsize,
}

impl AdSamplingSchedulerRuntime {
    pub(crate) fn snapshot(&self) -> AdSamplingSchedulerStats {
        AdSamplingSchedulerStats {
            ads_large_tasks: self.ads_large_tasks.load(Ordering::Relaxed),
            ads_chunks_total: self.ads_chunks_total.load(Ordering::Relaxed),
            ads_waves: self.ads_waves.load(Ordering::Relaxed),
            ads_exact_fallback_small: self.ads_exact_fallback_small.load(Ordering::Relaxed),
            ads_exact_fallback_nested: self.ads_exact_fallback_nested.load(Ordering::Relaxed),
            ads_parallelism_collapse_count: self
                .ads_parallelism_collapse_count
                .load(Ordering::Relaxed),
        }
    }

    pub(crate) fn record_exact_fallback_reason(&self, reason: Option<&str>) {
        match reason {
            Some("scheduler-small") => {
                self.ads_exact_fallback_small
                    .fetch_add(1, Ordering::Relaxed);
            }
            Some("scheduler-nested-large") => {
                self.ads_exact_fallback_nested
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    pub(crate) fn record_ads_profile(&self, params: &ForgeANNParams, profile: &AdSamplingProfile) {
        let class = classify_adsampling_task(params, profile.points, profile.leaders);
        if class.large_ads || class.huge_ads {
            self.ads_large_tasks.fetch_add(1, Ordering::Relaxed);
        }
        self.ads_chunks_total
            .fetch_add(profile.chunks, Ordering::Relaxed);
        if profile.scheduler_mode == "depth-wave" {
            self.ads_waves.fetch_add(1, Ordering::Relaxed);
        }
        if class.large_ads
            && profile.effective_parallelism < ForgeANNParams::ADS_MIN_EFFECTIVE_PARALLELISM
        {
            self.ads_parallelism_collapse_count
                .fetch_add(1, Ordering::Relaxed);
            if ForgeANNParams::ADS_FALLBACK_EXACT_ON_COLLAPSE {
                self.depth_ads_disabled.store(true, Ordering::Release);
            }
        }
    }

    pub(crate) fn depth_ads_disabled(&self) -> bool {
        self.depth_ads_disabled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AssignmentDecisionRecord {
    pub(crate) depth: usize,
    pub(crate) points: usize,
    pub(crate) leaders: usize,
    pub(crate) fanout: usize,
    pub(crate) wall: Duration,
    pub(crate) adsampling: bool,
    pub(crate) fallback_reason: Option<usize>,
    pub(crate) recall_at_fanout: f64,
    pub(crate) mismatches: usize,
}

impl AssignmentDecisionProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        for depth in 0..ASSIGNMENT_DEPTH_BUCKETS {
            for leader_bucket in 0..ASSIGNMENT_LEADER_BUCKETS {
                self.depth_leader_buckets[depth][leader_bucket]
                    .merge(other.depth_leader_buckets[depth][leader_bucket]);
            }
        }
        for (dst, src) in self.fallback_reasons.iter_mut().zip(other.fallback_reasons) {
            *dst += src;
        }
    }

    pub(crate) fn record(&mut self, record: AssignmentDecisionRecord) {
        if record.adsampling {
            self.record_adsampling(
                record.depth,
                record.points,
                record.leaders,
                record.fanout,
                record.wall,
                record.recall_at_fanout,
                record.mismatches,
            );
        } else {
            self.record_gemm(
                record.depth,
                record.points,
                record.leaders,
                record.fanout,
                record.wall,
                record.fallback_reason.map(assignment_fallback_reason_label),
            );
        }
    }

    pub(crate) fn record_gemm(
        &mut self,
        depth: usize,
        points: usize,
        leaders: usize,
        _fanout: usize,
        wall: Duration,
        fallback_reason: Option<&str>,
    ) {
        let bucket = self.bucket_mut(depth, leaders);
        bucket.nodes += 1;
        bucket.points += points;
        let work = assignment_work(points, leaders);
        bucket.work += work;
        bucket.max_points = bucket.max_points.max(points);
        bucket.max_leaders = bucket.max_leaders.max(leaders);
        bucket.max_work = bucket.max_work.max(work);
        bucket.gemm_nodes += 1;
        bucket.gemm_points += points;
        bucket.gemm_wall += wall;
        if let Some(reason) = fallback_reason {
            self.fallback_reasons[assignment_fallback_reason_index(reason)] += 1;
        }
    }

    pub(crate) fn record_adsampling(
        &mut self,
        depth: usize,
        points: usize,
        leaders: usize,
        _fanout: usize,
        wall: Duration,
        recall_at_fanout: f64,
        mismatches: usize,
    ) {
        let bucket = self.bucket_mut(depth, leaders);
        bucket.nodes += 1;
        bucket.points += points;
        let work = assignment_work(points, leaders);
        bucket.work += work;
        bucket.max_points = bucket.max_points.max(points);
        bucket.max_leaders = bucket.max_leaders.max(leaders);
        bucket.max_work = bucket.max_work.max(work);
        bucket.adsampling_nodes += 1;
        bucket.adsampling_points += points;
        bucket.adsampling_wall += wall;
        bucket.adsampling_recall_sum += recall_at_fanout;
        bucket.adsampling_recall_samples += 1;
        bucket.adsampling_mismatches += mismatches;
    }

    fn bucket_mut(&mut self, depth: usize, leaders: usize) -> &mut AssignmentDecisionBucketProfile {
        let depth_bucket = assignment_depth_bucket_index(depth);
        let leader_bucket = assignment_leader_bucket_index(leaders);
        &mut self.depth_leader_buckets[depth_bucket][leader_bucket]
    }
}

#[inline]
pub(crate) fn assignment_work(points: usize, leaders: usize) -> u128 {
    (points as u128).saturating_mul(leaders as u128)
}

pub(crate) fn assignment_depth_bucket_index(depth: usize) -> usize {
    depth.min(ASSIGNMENT_DEPTH_BUCKETS - 1)
}

pub fn assignment_depth_bucket_label(bucket: usize) -> &'static str {
    match bucket {
        0 => "0",
        1 => "1",
        2 => "2",
        3 => "3",
        4 => ">=4",
        _ => "unknown",
    }
}

pub(crate) fn assignment_leader_bucket_index(leaders: usize) -> usize {
    match leaders {
        0..=255 => 0,
        256..=511 => 1,
        512..=1023 => 2,
        1024..=2047 => 3,
        2048..=4095 => 4,
        _ => 5,
    }
}

pub fn assignment_leader_bucket_label(bucket: usize) -> &'static str {
    match bucket {
        0 => "<256",
        1 => "[256,512)",
        2 => "[512,1024)",
        3 => "[1024,2048)",
        4 => "[2048,4096)",
        5 => ">=4096",
        _ => "unknown",
    }
}

pub fn assignment_fallback_reason_label(index: usize) -> &'static str {
    match index {
        0 => "disabled",
        1 => "root-depth",
        2 => "max-depth",
        3 => "empty-points",
        4 => "fanout",
        5 => "leaders-le-fanout",
        6 => "min-points",
        7 => "min-leaders",
        8 => "min-work",
        9 => "group-dims",
        10 => "seed-exact",
        11 => "scheduler-nested-large",
        12 => "scheduler-small",
        13 => "scheduler-collapse",
        14 => "other",
        _ => "other",
    }
}

pub fn assignment_fallback_reason_index(reason: &str) -> usize {
    match reason {
        "disabled" => 0,
        "root-depth" => 1,
        "max-depth" => 2,
        "empty-points" => 3,
        "fanout" => 4,
        "leaders-le-fanout" => 5,
        "min-points" => 6,
        "min-leaders" => 7,
        "min-work" => 8,
        "group-dims" => 9,
        "seed-exact" => 10,
        "scheduler-nested-large" => 11,
        "scheduler-small" => 12,
        "scheduler-collapse" => 13,
        _ => 14,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SeededCluster {
    pub points: Vec<u32>,
    pub seed: u64,
}

#[derive(Clone, Debug, Default)]
pub struct GemmProfile {
    pub(crate) total_wall: Duration,
    pub(crate) build_x: Duration,
    pub(crate) gemm: Duration,
    pub(crate) topk: Duration,
    pub(crate) merge: Duration,
    pub(crate) spool_write: Duration,
    pub(crate) flush: Duration,
    pub(crate) blocks: usize,
    pub(crate) assignment_decision: Option<AssignmentDecisionRecord>,
    pub(crate) prefetch: StrictPrefetchPipelineProfile,
    pub(crate) point_pipeline: PointPipelineStats,
}

impl GemmProfile {
    pub(crate) fn merge(&mut self, other: GemmProfile) {
        self.total_wall += other.total_wall;
        self.build_x += other.build_x;
        self.gemm += other.gemm;
        self.topk += other.topk;
        self.merge += other.merge;
        self.spool_write += other.spool_write;
        self.flush += other.flush;
        self.blocks += other.blocks;
        self.prefetch.merge(other.prefetch);
        self.point_pipeline.merge(other.point_pipeline);
        if self.assignment_decision.is_none() {
            self.assignment_decision = other.assignment_decision;
        }
    }
}

#[cfg(test)]
mod assignment_profile_tests {
    use std::collections::BTreeSet;
    use std::io::{Seek, Write};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use tempfile::tempdir;

    use crate::common::AnnResult;
    use crate::forgeann::ForgeANNParams;
    use crate::forgeann::point_store::{InmemDatasetPointStore, LimitedPointStore, PointStore};
    use crate::model::InmemDataset;

    #[test]
    pub fn assignment_decision_profile_records_depth_leader_buckets_and_fallbacks() {
        let mut profile = super::AssignmentDecisionProfile::default();

        profile.record_gemm(
            1,
            134_093,
            751,
            2,
            Duration::from_millis(42),
            Some("min-leaders"),
        );
        profile.record_adsampling(1, 134_093, 751, 2, Duration::from_millis(31), 0.9825, 17);

        let depth_bucket = super::assignment_depth_bucket_index(1);
        let leader_bucket = super::assignment_leader_bucket_index(751);
        let bucket = profile.depth_leader_buckets[depth_bucket][leader_bucket];

        assert_eq!(bucket.nodes, 2);
        assert_eq!(bucket.points, 268_186);
        assert_eq!(bucket.work, 201_407_686);
        assert_eq!(bucket.max_points, 134_093);
        assert_eq!(bucket.max_leaders, 751);
        assert_eq!(bucket.gemm_nodes, 1);
        assert_eq!(bucket.gemm_wall, Duration::from_millis(42));
        assert_eq!(bucket.adsampling_nodes, 1);
        assert_eq!(bucket.adsampling_wall, Duration::from_millis(31));
        assert_eq!(bucket.adsampling_recall_samples, 1);
        assert_eq!(bucket.adsampling_mismatches, 17);
        assert_eq!(
            profile.fallback_reasons[super::assignment_fallback_reason_index("min-leaders")],
            1
        );
    }

    #[test]
    pub fn depth_wave_scheduler_uses_ads_for_small_depth_tasks() {
        let params = ForgeANNParams::default();

        let context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None);

        assert!(context.should_use_adsampling_assignment(65_000, 525, 2));
        assert_eq!(
            context.assignment_gemm_fallback_reason(65_000, 525, 2),
            None
        );
    }

    #[test]
    pub fn depth_wave_non_large_assignment_uses_ads_without_point_pipeline() {
        let rows = 2_048usize;
        let dim = 8usize;
        let mut dataset = InmemDataset::new(rows, 1.0, dim).unwrap();
        for point in 0..rows {
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = ((point * 19 + axis * 11) % 127) as f32;
            }
        }
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let cur = (0..rows as u32).collect::<Vec<_>>();
        let leaders = (0..96u32).step_by(3).collect::<Vec<_>>();
        let params = ForgeANNParams::default();

        let context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None)
                .with_depth_wave_assignment();
        assert!(context.should_use_adsampling_assignment(rows, leaders.len(), 2));

        let (_clusters, profile) = super::compute_clusters_gemm_budgeted_with_context(
            &store,
            &cur,
            &leaders,
            2,
            None,
            Some(&context),
        )
        .unwrap();

        let decision = profile.assignment_decision.expect("assignment decision");
        assert!(decision.adsampling);
        assert_eq!(profile.point_pipeline.batches, 0);
        assert!(profile.prefetch.batches > 0);
        let stats = context.ads_scheduler_stats();
        assert!(stats.ads_chunks_total > 1);
    }

    #[test]
    pub fn depth_wave_child_run_schedule_preserves_manifest_order() {
        let mut params = ForgeANNParams::default();
        params.max_leaders = 5_000;
        params.psamp_fraction = 0.0057;

        let child_runs = vec![
            super::ChildRun {
                extents: Vec::new(),
                len: 4_096,
                seed: 1,
            },
            super::ChildRun {
                extents: Vec::new(),
                len: 500_000,
                seed: 2,
            },
            super::ChildRun {
                extents: Vec::new(),
                len: 1_000_000,
                seed: 3,
            },
            super::ChildRun {
                extents: Vec::new(),
                len: 64_000,
                seed: 4,
            },
            super::ChildRun {
                extents: Vec::new(),
                len: 400_000,
                seed: 5,
            },
        ];

        let schedule = super::schedule_depth_wave_child_runs(&params, child_runs, 1);
        assert_eq!(
            schedule
                .wave_runs
                .iter()
                .map(|run| run.len)
                .collect::<Vec<_>>(),
            vec![4_096, 500_000, 1_000_000, 64_000, 400_000]
        );
        assert!(schedule.exact_runs.is_empty());
    }

    #[test]
    pub fn depth_wave_child_run_scheduler_executes_d1_level_scan_ads() {
        let rows = 96usize;
        let dim = 6usize;
        let mut dataset = InmemDataset::new(rows, 1.0, dim).unwrap();
        for point in 0..rows {
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = ((point * 31 + axis * 17) % 251) as f32;
            }
        }
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let dir = tempdir().unwrap();
        let external_store = Arc::new(super::Mutex::new(super::ExternalRunStore::new(
            dir.path(),
            super::DirectIoConfig::disabled(),
        )));
        let child_a: Vec<u32> = (0..48_u32).collect();
        let child_b: Vec<u32> = (48..96_u32).collect();
        let child_runs = {
            let mut guard = external_store.lock();
            let extent_a = super::write_child_run(&mut guard, 0, &child_a).unwrap();
            let extent_b = super::write_child_run(&mut guard, 0, &child_b).unwrap();
            vec![
                super::ChildRun {
                    extents: vec![extent_a],
                    len: child_a.len(),
                    seed: 11,
                },
                super::ChildRun {
                    extents: vec![extent_b],
                    len: child_b.len(),
                    seed: 22,
                },
            ]
        };

        let mut params = ForgeANNParams::default();
        params.c_min = 2;
        params.c_max = 8;
        params.max_depth = 2;
        params.max_leaders = 8;
        params.psamp_fraction = 0.25;
        params.fanout_top = 1;
        params.fanout_second = 1;
        params.adsampling_group_dims = 2;
        let adaptive_c_max = params.adaptive_c_max(rows);
        let min_recurse_size = adaptive_c_max.saturating_mul(2);
        let context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None);
        let (tx, rx) = crossbeam::channel::unbounded();
        let stats = super::parallel_join_child_runs(
            &store,
            child_runs,
            1,
            rows,
            crate::common::Metric::L2,
            &params,
            adaptive_c_max,
            min_recurse_size,
            &super::ProgressBar::hidden(),
            &external_store,
            &super::RootFanoutState::fixed(params.fanout_top),
            &context,
            &tx,
        )
        .unwrap();
        drop(tx);

        assert!(context.ads_scheduler_stats().ads_chunks_total > 0);
        assert!(stats.gemm_profile.prefetch.batches > 0);
        assert_eq!(stats.telemetry.point_pipeline.batches, 0);
        let emitted_points = rx.into_iter().map(|leaf| leaf.len()).sum::<usize>();
        assert_eq!(emitted_points, rows);
    }

    #[test]
    pub fn materialize_from_segmented_spool_uses_only_selected_segments() {
        let temp = tempdir().unwrap();
        let mut store =
            super::ExternalRunStore::new(temp.path(), super::DirectIoConfig::disabled());
        let cur = vec![10_u32, 11, 12, 13];
        let fanout = 1usize;
        let spool = tempfile::NamedTempFile::new_in(temp.path()).unwrap();
        let mut segments = Vec::new();
        {
            let mut writer = std::io::BufWriter::new(spool.reopen().unwrap());
            let first_offset = writer.stream_position().unwrap();
            super::write_assign_record(&mut writer, &[0], fanout).unwrap();
            super::write_assign_record(&mut writer, &[1], fanout).unwrap();
            let first_len = writer.stream_position().unwrap() - first_offset;
            segments.push(super::AssignmentSpoolSegment {
                offset: first_offset,
                len: first_len,
            });

            super::write_assign_record(&mut writer, &[0], fanout).unwrap();
            super::write_assign_record(&mut writer, &[0], fanout).unwrap();

            let second_offset = writer.stream_position().unwrap();
            super::write_assign_record(&mut writer, &[1], fanout).unwrap();
            super::write_assign_record(&mut writer, &[0], fanout).unwrap();
            let second_len = writer.stream_position().unwrap() - second_offset;
            segments.push(super::AssignmentSpoolSegment {
                offset: second_offset,
                len: second_len,
            });
            writer.flush().unwrap();
        }

        let merge_groups = vec![
            super::MergedRawGroup {
                raw_children: vec![0],
                raw_len: 2,
            },
            super::MergedRawGroup {
                raw_children: vec![1],
                raw_len: 2,
            },
        ];
        let materialized =
            super::materialize_merged_children_from_spool_segments_with_inline_limit(
                &cur,
                0,
                fanout,
                &merge_groups,
                &spool,
                &segments,
                &mut store,
                Some(4),
            )
            .unwrap();

        let actual: Vec<BTreeSet<u32>> = materialized
            .into_iter()
            .map(|child| {
                child
                    .points
                    .unwrap_or_else(|| {
                        super::read_child_run_chain(&store, 0, &child.extents).unwrap()
                    })
                    .into_iter()
                    .collect()
            })
            .collect();

        assert_eq!(
            actual,
            vec![BTreeSet::from([10_u32, 13]), BTreeSet::from([11_u32, 12])]
        );
    }

    #[test]
    pub fn d1_level_scan_pipeline_inflight_respects_request_and_budget() {
        let one_mib_batch_points = 1024usize;
        let dim = 256usize;

        assert_eq!(
            super::clamp_d1_level_scan_pipeline_inflight(
                4,
                one_mib_batch_points,
                dim,
                Some(64 * 1024 * 1024),
            ),
            4
        );
        assert_eq!(
            super::clamp_d1_level_scan_pipeline_inflight(
                4,
                one_mib_batch_points,
                dim,
                Some(2 * 1024 * 1024),
            ),
            1
        );
        assert_eq!(
            super::clamp_d1_level_scan_pipeline_inflight(0, one_mib_batch_points, dim, None),
            1
        );
    }

    #[test]
    pub fn d1_level_scan_heap_merge_deduplicates_points_and_preserves_occurrences() {
        let runs = vec![
            super::D1LevelScanRunWork {
                points: vec![1, 3, 7],
                seed: 1,
                raw_counts: Vec::new(),
                assignment_segments: Vec::new(),
                leaders: 0,
                fanout: 1,
                max_leaders: 0,
            },
            super::D1LevelScanRunWork {
                points: vec![2, 3, 5, 7],
                seed: 2,
                raw_counts: Vec::new(),
                assignment_segments: Vec::new(),
                leaders: 0,
                fanout: 1,
                max_leaders: 0,
            },
        ];
        let mut heap = super::BinaryHeap::new();
        for (run_idx, run) in runs.iter().enumerate() {
            heap.push(super::D1LevelScanHeapEntry {
                point: run.points[0],
                run_idx,
                local_idx: 0,
            });
        }

        let first = super::next_d1_level_scan_batch(&mut heap, &runs, 3).unwrap();
        let second = super::next_d1_level_scan_batch(&mut heap, &runs, 3).unwrap();

        assert_eq!(first.point_ids, vec![1, 2, 3]);
        assert_eq!(
            first
                .occurrences_by_point
                .iter()
                .map(|occurrences| {
                    occurrences
                        .iter()
                        .map(|occurrence| (occurrence.run_idx, occurrence.local_idx))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
            vec![vec![(0, 0)], vec![(1, 0)], vec![(0, 1), (1, 1)]]
        );
        assert_eq!(second.point_ids, vec![5, 7]);
        assert_eq!(
            second
                .occurrences_by_point
                .iter()
                .map(|occurrences| {
                    occurrences
                        .iter()
                        .map(|occurrence| (occurrence.run_idx, occurrence.local_idx))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
            vec![vec![(1, 2)], vec![(0, 2), (1, 3)]]
        );
    }

    #[test]
    pub fn d1_ads_compute_chunk_points_targets_many_tasks_when_enough_points_exist() {
        let workers = 4usize;
        let chunks_per_worker = 8usize;
        let points = workers * chunks_per_worker * 512;
        let chunk_points = super::d1_ads_compute_chunk_points(points, workers, chunks_per_worker);

        assert!((256..=8192).contains(&chunk_points));
        assert!(points.div_ceil(chunk_points) >= workers * chunks_per_worker);
    }

    #[test]
    pub fn d1_level_scan_compute_tasks_split_large_run_group() {
        let input = super::D1LevelScanRunBatchInput {
            point_ids: (0..4096_u32).collect(),
            source_offsets: (0..4096).collect(),
            local_indices: (10_000..14_096).collect(),
        };
        let grouped = vec![None, Some(input)];

        let tasks = super::build_d1_level_scan_compute_tasks(&grouped, 1024);

        assert_eq!(tasks.len(), 4);
        assert_eq!(
            tasks
                .iter()
                .map(|task| {
                    (
                        task.run_idx,
                        task.point_start,
                        task.point_end,
                        task.run_point_start,
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                (1, 0, 1024, 10_000),
                (1, 1024, 2048, 11_024),
                (1, 2048, 3072, 12_048),
                (1, 3072, 4096, 13_072),
            ]
        );
    }

    #[test]
    pub fn d1_level_scan_ordered_drain_waits_for_missing_batch_then_flushes() {
        let mut pending = super::BTreeMap::new();
        let mut next_apply = 0usize;
        let mut applied = Vec::new();

        pending.insert(1, "one");
        let drained =
            super::drain_ordered_d1_level_scan_batches(&mut pending, &mut next_apply, |value| {
                applied.push(value);
                Ok(())
            })
            .unwrap();
        assert_eq!(drained, 0);
        assert!(applied.is_empty());
        assert_eq!(next_apply, 0);

        pending.insert(0, "zero");
        let drained =
            super::drain_ordered_d1_level_scan_batches(&mut pending, &mut next_apply, |value| {
                applied.push(value);
                Ok(())
            })
            .unwrap();

        assert_eq!(drained, 2);
        assert_eq!(applied, vec!["zero", "one"]);
        assert_eq!(next_apply, 2);
    }

    #[test]
    pub fn d1_level_scan_batch_points_defaults_to_balanced_large_window_when_budget_allows() {
        let params = ForgeANNParams::default();
        assert_eq!(
            super::d1_level_scan_batch_points(&params, 768, Some(32 * 1024 * 1024 * 1024)),
            262_144
        );
    }

    #[test]
    pub fn d1_level_scan_memory_budget_prefers_oom_budget_over_resident_task_budget() {
        let full_budget = 32 * 1024 * 1024 * 1024;
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        params.oom_memory_budget_bytes = full_budget;

        let budget = super::d1_level_scan_memory_budget_bytes(&params).unwrap();
        assert_eq!(budget, full_budget);
        assert_eq!(
            super::d1_level_scan_batch_points(&params, 768, Some(budget)),
            262_144
        );
    }

    #[test]
    pub fn prefetch_queue_depth_uses_oom_budget_for_ahead_batches() {
        let point_tile = 4_096;
        let dim = 768;
        let workers = 42;
        let batch_points = super::choose_prefetch_batch_points_for_budget(
            point_tile,
            dim,
            workers,
            Some(16 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            super::choose_prefetch_queue_depth_for_budget(batch_points, dim, None),
            1
        );
        assert_eq!(
            super::choose_prefetch_queue_depth_for_budget(
                batch_points,
                dim,
                Some(16 * 1024 * 1024 * 1024),
            ),
            8
        );
        assert_eq!(
            super::choose_prefetch_queue_depth_for_budget(
                batch_points.saturating_mul(64),
                dim,
                Some(16 * 1024 * 1024 * 1024),
            ),
            1
        );
    }

    #[test]
    pub fn strict_prefetch_gemm_scheduler_skips_single_batch_small_nodes() {
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        let budget = Some(32 * 1024 * 1024 * 1024usize);
        let workers = 42;

        assert!(!super::should_use_strict_prefetch_for_gemm_assignment(
            &params, 65_000, 525, 768, budget, workers,
        ));
        assert!(super::should_use_strict_prefetch_for_gemm_assignment(
            &params, 1_000_000, 1_991, 768, budget, workers,
        ));
        assert!(!super::should_use_strict_prefetch_for_gemm_assignment(
            &params, 1_000_000, 1_991, 768, budget, workers,
        ));
    }

    #[test]
    pub fn default_rbc_windowed_options_coalesce_strict_direct_gaps() {
        pub struct DirectLikePointStore {
            pub coalesced: bool,
        }

        impl PointStore for DirectLikePointStore {
            fn len(&self) -> usize {
                0
            }

            fn dim(&self) -> usize {
                768
            }

            fn read_point_into(&self, _pid: u32, _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_range_into(
                &self,
                _start_pid: u32,
                _count: usize,
                _out: &mut [f32],
            ) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_points_into(&self, _ids: &[u32], _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn prefers_coalesced_window_reads(&self) -> bool {
                self.coalesced
            }
        }

        let buffered = DirectLikePointStore { coalesced: false };
        let strict_direct = DirectLikePointStore { coalesced: true };

        let buffered_options = super::default_rbc_windowed_options(&buffered, 16_384);
        let strict_options = super::default_rbc_windowed_options(&strict_direct, 16_384);

        assert_eq!(buffered_options.max_gap_rows, 1);
        assert!(
            strict_options.max_gap_rows > buffered_options.max_gap_rows,
            "direct GEMM/exact fallback reads should still coalesce small gaps"
        );
        assert!(strict_options.max_gap_rows <= 8);
        assert!(strict_options.max_window_bytes >= buffered_options.max_window_bytes);
    }

    #[test]
    pub fn default_rbc_windowed_options_caps_direct_gap_rows_for_sparse_batches() {
        pub struct DirectLikePointStore;

        impl PointStore for DirectLikePointStore {
            fn len(&self) -> usize {
                0
            }

            fn dim(&self) -> usize {
                768
            }

            fn read_point_into(&self, _pid: u32, _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_range_into(
                &self,
                _start_pid: u32,
                _count: usize,
                _out: &mut [f32],
            ) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_points_into(&self, _ids: &[u32], _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn prefers_coalesced_window_reads(&self) -> bool {
                true
            }
        }

        let store = DirectLikePointStore;
        let options = super::default_rbc_windowed_options(&store, 16_384);

        assert!(
            options.max_gap_rows <= 8,
            "default local RBC point batches must not merge sparse ids into huge direct-read windows"
        );
        assert!(options.max_window_bytes >= 4 * 1024 * 1024);
    }

    #[test]
    pub fn limited_point_store_preserves_direct_coalescing_preference() {
        pub struct DirectLikePointStore;

        impl PointStore for DirectLikePointStore {
            fn len(&self) -> usize {
                1024
            }

            fn dim(&self) -> usize {
                768
            }

            fn read_point_into(&self, _pid: u32, _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_range_into(
                &self,
                _start_pid: u32,
                _count: usize,
                _out: &mut [f32],
            ) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_points_into(&self, _ids: &[u32], _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn prefers_coalesced_window_reads(&self) -> bool {
                true
            }
        }

        let inner = DirectLikePointStore;
        let limited = LimitedPointStore::new(&inner, inner.len());
        let options = super::default_rbc_windowed_options(&limited, 16_384);

        assert!(
            options.max_gap_rows > 1,
            "LimitedPointStore must not hide direct-I/O coalescing from RBC gather planning"
        );
        assert!(options.max_gap_rows <= 8);
    }

    #[test]
    pub fn strict_prefetch_windowed_options_honor_configured_gap_limit() {
        pub struct DirectLikePointStore;

        impl PointStore for DirectLikePointStore {
            fn len(&self) -> usize {
                0
            }

            fn dim(&self) -> usize {
                768
            }

            fn read_point_into(&self, _pid: u32, _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_range_into(
                &self,
                _start_pid: u32,
                _count: usize,
                _out: &mut [f32],
            ) -> AnnResult<()> {
                panic!("not used")
            }

            fn read_points_into(&self, _ids: &[u32], _out: &mut [f32]) -> AnnResult<()> {
                panic!("not used")
            }

            fn prefers_coalesced_window_reads(&self) -> bool {
                true
            }
        }

        let store = DirectLikePointStore;
        let fallback = super::default_rbc_windowed_options(&store, 16_384);
        let strict = super::strict_prefetch_windowed_options(
            &store,
            fallback,
            super::StrictPrefetchPipelineConfig {
                enabled: true,
                queue_depth: 3,
                budget_bytes: 1024 * 1024 * 1024,
                max_window_bytes: 16 * 1024 * 1024,
                max_gap_rows: 4,
            },
        );

        assert!(fallback.max_gap_rows <= 8);
        assert_eq!(strict.max_gap_rows, 4);
        assert!(strict.max_window_bytes >= fallback.max_window_bytes);
        assert!(strict.max_window_bytes >= 16 * 1024 * 1024);
    }

    #[test]
    pub fn depth_wave_child_run_scheduler_can_start_from_scheduler_scope_worker() {
        let params = ForgeANNParams::default();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let use_wave =
            pool.install(|| super::should_use_depth_wave_child_run_scheduler(&params, 1));

        assert!(use_wave);
    }

    #[test]
    pub fn depth_wave_context_records_prefetched_ads_assignment() {
        let rows = 1_536usize;
        let dim = 8usize;
        let mut dataset = InmemDataset::new(rows, 1.0, dim).unwrap();
        for point in 0..rows {
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = ((point * 11 + axis * 5) % 101) as f32;
            }
        }
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let cur = (0..rows as u32).collect::<Vec<_>>();
        let leaders = (0..96u32).step_by(3).collect::<Vec<_>>();
        let params = ForgeANNParams::default();

        let context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None)
                .with_depth_wave_assignment();
        let (_clusters, profile) = super::compute_clusters_gemm_budgeted_with_context(
            &store,
            &cur,
            &leaders,
            2,
            None,
            Some(&context),
        )
        .unwrap();

        let decision = profile.assignment_decision.expect("assignment decision");
        assert!(decision.adsampling);
        assert_eq!(profile.point_pipeline.batches, 0);
        assert!(profile.prefetch.batches > 0);
        let stats = context.ads_scheduler_stats();
        assert_eq!(stats.ads_waves, 1);
        assert!(stats.ads_chunks_total > 0);
    }

    #[test]
    pub fn depth_wave_large_assignment_uses_gemm_style_prefetch_not_point_pipeline() {
        let rows = 4_096usize;
        let dim = 8usize;
        let mut dataset = InmemDataset::new(rows, 1.0, dim).unwrap();
        for point in 0..rows {
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = ((point * 23 + axis * 7) % 113) as f32;
            }
        }
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let cur = (0..rows as u32).collect::<Vec<_>>();
        let leaders = (0..384u32).step_by(3).collect::<Vec<_>>();
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;

        let context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None)
                .with_depth_wave_assignment();
        assert!(context.should_use_adsampling_assignment(rows, leaders.len(), 2));

        let (_clusters, profile) = super::compute_clusters_gemm_budgeted_with_context(
            &store,
            &cur,
            &leaders,
            2,
            Some(256 * 1024 * 1024),
            Some(&context),
        )
        .unwrap();

        let decision = profile.assignment_decision.expect("assignment decision");
        assert!(decision.adsampling);
        assert_eq!(profile.point_pipeline.batches, 0);
        assert!(profile.prefetch.batches > 0);
        assert!(context.ads_scheduler_stats().ads_chunks_total > 1);
    }

    #[test]
    pub fn depth_wave_forces_ads_even_after_parallelism_collapse() {
        let params = ForgeANNParams::default();

        let context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None)
                .with_depth_wave_assignment();
        context
            .scheduler_runtime
            .depth_ads_disabled
            .store(true, Ordering::Release);

        assert!(context.should_use_adsampling_assignment(4_096, 128, 2));
        assert_eq!(context.assignment_gemm_fallback_reason(4_096, 128, 2), None);
    }

    #[test]
    pub fn depth_wave_context_is_not_short_circuited_inside_rayon_worker() {
        let rows = 1_536usize;
        let dim = 8usize;
        let mut dataset = InmemDataset::new(rows, 1.0, dim).unwrap();
        for point in 0..rows {
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = ((point * 17 + axis * 3) % 103) as f32;
            }
        }
        let store = InmemDatasetPointStore::new(&dataset, rows);
        let cur = (0..rows as u32).collect::<Vec<_>>();
        let leaders = (0..96u32).step_by(3).collect::<Vec<_>>();
        let params = ForgeANNParams::default();

        let context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None)
                .with_depth_wave_assignment();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let (_clusters, profile) = pool
            .install(|| {
                super::compute_clusters_gemm_budgeted_with_context(
                    &store,
                    &cur,
                    &leaders,
                    2,
                    None,
                    Some(&context),
                )
            })
            .unwrap();

        let decision = profile.assignment_decision.expect("assignment decision");
        assert!(decision.adsampling);
        assert_eq!(
            decision
                .fallback_reason
                .map(super::assignment_fallback_reason_label),
            None
        );
        let stats = context.ads_scheduler_stats();
        assert_eq!(stats.ads_waves, 1);
    }
}

#[cfg(test)]
mod io_planned_recursion_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crossbeam::channel::{Receiver, unbounded};
    use indicatif::ProgressBar;
    use parking_lot::Mutex;
    use tempfile::tempdir;

    use super::{ExternalRunStore, RootFanoutState};
    use crate::common::{AnnResult, Metric};
    use crate::forgeann::ForgeANNParams;
    use crate::forgeann::direct_io::DirectIoConfig;
    use crate::forgeann::point_store::{InmemDatasetPointStore, PointStore};
    use crate::model::InmemDataset;

    pub fn build_test_dataset(num_points: usize, dim: usize) -> InmemDataset<f32> {
        let mut dataset = InmemDataset::new(num_points, 1.0, dim).unwrap();
        for point in 0..num_points {
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = point as f32 * 0.25 + axis as f32;
            }
        }
        dataset
    }

    pub fn collect_sorted_leaves(receiver: Receiver<Vec<u32>>) -> Vec<Vec<u32>> {
        let mut leaves: Vec<Vec<u32>> = receiver
            .iter()
            .map(|mut leaf| {
                leaf.sort_unstable();
                leaf
            })
            .collect();
        leaves.sort_unstable();
        leaves
    }

    pub fn base_params() -> ForgeANNParams {
        let mut params = ForgeANNParams::default();
        params.oom_enable = true;
        params.oom_memory_budget_bytes = 512 * 1024 * 1024;
        params.c_min = 8;
        params.c_max = 12;
        params.max_depth = 4;
        params.max_leaders = 16;
        params.fanout_top = 4;
        params.fanout_second = 2;
        params
    }

    pub fn child_points(num_points: usize) -> Vec<u32> {
        (0..num_points as u32)
            .filter(|point| point % 3 != 1)
            .collect()
    }

    pub struct CountingPointStore<'a> {
        pub inner: InmemDatasetPointStore<'a>,
        pub range_calls: AtomicUsize,
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

        fn read_point_into(&self, pid: u32, out: &mut [f32]) -> AnnResult<()> {
            self.inner.read_point_into(pid, out)
        }

        fn read_range_into(&self, start_pid: u32, count: usize, out: &mut [f32]) -> AnnResult<()> {
            self.range_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.read_range_into(start_pid, count, out)
        }

        fn read_points_into(&self, ids: &[u32], out: &mut [f32]) -> AnnResult<()> {
            self.inner.read_points_into(ids, out)
        }

        fn get_distance(&self, lhs: u32, rhs: u32, metric: Metric) -> AnnResult<f32> {
            self.inner.get_distance(lhs, rhs, metric)
        }
    }

    #[test]
    pub fn child_run_extent_batcher_coalesces_adjacent_extents_and_preserves_order() {
        let dir = tempdir().unwrap();
        let mut store = ExternalRunStore::new(dir.path(), DirectIoConfig::disabled());
        let extent_a = store.append_points(1, &[1, 2, 3, 4]).unwrap();
        let extent_b = store.append_points(1, &[10, 11]).unwrap();
        let extent_c = store.append_points(1, &[20, 21, 22]).unwrap();
        store.finalize().unwrap();

        let runs = vec![
            super::ChildRun {
                extents: vec![extent_c],
                len: extent_c.len,
                seed: 3,
            },
            super::ChildRun {
                extents: vec![extent_a, extent_b],
                len: extent_a.len + extent_b.len,
                seed: 1,
            },
        ];

        let (points, stats) = super::read_child_runs_batched_from_path(
            dir.path(),
            DirectIoConfig::disabled(),
            1,
            &runs,
        )
        .unwrap();

        assert_eq!(points, vec![vec![20, 21, 22], vec![1, 2, 3, 4, 10, 11]]);
        assert_eq!(stats.logical_extents, 3);
        assert!(
            stats.coalesced_reads < stats.logical_extents,
            "adjacent child extents should share physical reads"
        );
        assert!(stats.header_read_savings >= 1);
    }

    #[test]
    pub fn small_run_wave_reads_shared_vector_window_once_for_siblings() {
        let num_points = 64;
        let dim = 4;
        let dataset = build_test_dataset(num_points, dim);
        let point_store = CountingPointStore::new(&dataset, num_points);
        let mut params = base_params();
        params.max_depth = 1;
        let adaptive_c_max = params.adaptive_c_max(8);
        let min_recurse_size = adaptive_c_max.saturating_mul(2);

        let external_dir = tempdir().unwrap();
        let external_store = Arc::new(Mutex::new(ExternalRunStore::new(
            external_dir.path(),
            DirectIoConfig::disabled(),
        )));
        let child_runs = {
            let mut guard = external_store.lock();
            let left = super::write_child_run(&mut guard, 0, &[10, 11, 12, 13]).unwrap();
            let right = super::write_child_run(&mut guard, 0, &[14, 15, 16, 17]).unwrap();
            vec![
                super::ChildRun {
                    extents: vec![left],
                    len: left.len,
                    seed: 11,
                },
                super::ChildRun {
                    extents: vec![right],
                    len: right.len,
                    seed: 12,
                },
            ]
        };
        let assignment_context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None);
        let (tx, rx) = unbounded();

        let stats = super::parallel_join_child_runs_inline(
            &point_store,
            child_runs,
            1,
            8,
            Metric::L2,
            &params,
            adaptive_c_max,
            min_recurse_size,
            &ProgressBar::hidden(),
            &external_store,
            &RootFanoutState::fixed(params.fanout_top),
            &assignment_context,
            &tx,
        )
        .unwrap();
        drop(tx);

        let leaves = collect_sorted_leaves(rx);
        assert_eq!(leaves, vec![vec![10, 11, 12, 13], vec![14, 15, 16, 17]]);
        assert_eq!(point_store.range_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stats.telemetry.io_planned_forgeann.small_run_waves, 1);
        assert_eq!(stats.telemetry.point_pipeline.planned_windows, 1);
        assert_eq!(
            stats
                .telemetry
                .io_planned_forgeann
                .io_pain_by_depth
                .get(&1)
                .map(|depth| depth.point_gather_windows),
            Some(1)
        );
    }

    #[test]
    pub fn adsampling_root_and_depth_contexts_match_full_gemm() {
        let num_points = 48;
        let dim = 8;
        let dataset = build_test_dataset(num_points, dim);
        let point_store = InmemDatasetPointStore::new(&dataset, num_points);
        let cur: Vec<u32> = (0..num_points as u32).collect();
        let leaders: Vec<u32> = (0..8_u32).collect();
        let mut params = base_params();
        params.adsampling_rotation_sidecar = Some(std::path::PathBuf::from("unused.fbin"));
        params.adsampling_epsilon = 100.0;
        params.adsampling_group_dims = 2;

        let (baseline, _) =
            super::compute_clusters_gemm_budgeted(&point_store, &cur, &leaders, 2, None).unwrap();
        let root_context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(0, &params, None);
        let (root_ads, root_profile) = super::compute_clusters_gemm_budgeted_with_context(
            &point_store,
            &cur,
            &leaders,
            2,
            None,
            Some(&root_context),
        )
        .unwrap();
        assert_eq!(root_ads, baseline);
        assert_eq!(root_profile.blocks, 1);

        let depth_one_context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None);
        let (depth_one, depth_one_profile) = super::compute_clusters_gemm_budgeted_with_context(
            &point_store,
            &cur,
            &leaders,
            2,
            None,
            Some(&depth_one_context),
        )
        .unwrap();
        assert_eq!(depth_one, baseline);
        assert_eq!(depth_one_profile.gemm, std::time::Duration::ZERO);
        assert!(
            depth_one_profile
                .assignment_decision
                .expect("depth assignment decision")
                .adsampling
        );
    }

    #[test]
    pub fn depth_adsampling_uses_original_dataset_when_gate_passes() {
        pub struct PanicPointStore {
            pub dim: usize,
        }

        impl crate::forgeann::point_store::PointStore for PanicPointStore {
            fn len(&self) -> usize {
                10_000
            }

            fn dim(&self) -> usize {
                self.dim
            }

            fn read_point_into(&self, _pid: u32, _out: &mut [f32]) -> crate::common::AnnResult<()> {
                panic!("depth ADSampling must not read the root ADS sidecar")
            }

            fn read_range_into(
                &self,
                _start_pid: u32,
                _count: usize,
                _out: &mut [f32],
            ) -> crate::common::AnnResult<()> {
                panic!("depth ADSampling must not read the root ADS sidecar")
            }

            fn read_points_into(
                &self,
                _ids: &[u32],
                _out: &mut [f32],
            ) -> crate::common::AnnResult<()> {
                panic!("depth ADSampling must not read the root ADS sidecar")
            }
        }

        let num_points = 48;
        let dim = 8;
        let dataset = build_test_dataset(num_points, dim);
        let point_store = InmemDatasetPointStore::new(&dataset, num_points);
        let sidecar = PanicPointStore { dim };
        let cur: Vec<u32> = (0..num_points as u32).collect();
        let leaders: Vec<u32> = (0..8_u32).collect();
        let mut params = base_params();
        params.adsampling_epsilon = 100.0;
        params.adsampling_group_dims = 2;

        let (baseline, _) =
            super::compute_clusters_gemm_budgeted(&point_store, &cur, &leaders, 2, None).unwrap();
        let depth_context = super::AssignmentContext::new_for_assignment_with_adsampling_dataset(
            1,
            &params,
            Some(&sidecar),
        );
        let (depth_ads, profile) = super::compute_clusters_gemm_budgeted_with_context(
            &point_store,
            &cur,
            &leaders,
            2,
            None,
            Some(&depth_context),
        )
        .unwrap();

        assert_eq!(depth_ads, baseline);
        assert_eq!(profile.blocks, 1);
        assert_eq!(profile.gemm, std::time::Duration::ZERO);
    }

    #[test]
    pub fn depth_adsampling_spool_and_non_spool_paths_share_gate() {
        let num_points = 48;
        let dim = 8;
        let dataset = build_test_dataset(num_points, dim);
        let point_store = InmemDatasetPointStore::new(&dataset, num_points);
        let cur: Vec<u32> = (0..num_points as u32).collect();
        let leaders: Vec<u32> = (0..8_u32).collect();
        let mut params = base_params();
        params.adsampling_epsilon = 100.0;
        params.adsampling_group_dims = 2;
        let context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None);

        let (clusters, profile) = super::compute_clusters_gemm_budgeted_with_context(
            &point_store,
            &cur,
            &leaders,
            2,
            None,
            Some(&context),
        )
        .unwrap();
        assert_eq!(profile.blocks, 1);
        assert_eq!(profile.gemm, std::time::Duration::ZERO);

        let dir = tempfile::tempdir().unwrap();
        let spool = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
        let mut writer = std::io::BufWriter::new(spool.reopen().unwrap());
        let mut raw_counts = vec![0usize; leaders.len()];
        let spool_profile = super::compute_clusters_gemm_to_spool_budgeted_with_context(
            &point_store,
            &cur,
            &leaders,
            2,
            None,
            &mut writer,
            &mut raw_counts,
            Some(&context),
        )
        .unwrap();
        assert_eq!(spool_profile.blocks, 1);
        assert_eq!(spool_profile.gemm, std::time::Duration::ZERO);

        let mut spooled = vec![Vec::new(); leaders.len()];
        let mut reader = std::io::BufReader::new(spool.reopen().unwrap());
        let mut assign = vec![0u16; 2];
        for &point in &cur {
            let len = super::read_assign_record(&mut reader, &mut assign).unwrap();
            for &leader_idx in &assign[..len] {
                spooled[leader_idx as usize].push(point);
            }
        }

        assert_eq!(spooled, clusters);
    }

    #[test]
    pub fn io_planned_actual_vector_run_matches_external_child_recursion() {
        let num_points = 96;
        let dim = 6;
        let dataset = build_test_dataset(num_points, dim);
        let point_store = InmemDatasetPointStore::new(&dataset, num_points);
        let child_points = child_points(num_points);
        let mut params = base_params();
        // Set fanout_second=1 so vector-run depth guard allows fanout=1 at depth=1.
        params.fanout_second = 1;
        let adaptive_c_max = params.adaptive_c_max(child_points.len());
        let min_recurse_size = adaptive_c_max.saturating_mul(2);
        let external_dir = tempdir().unwrap();
        let external_store = Arc::new(Mutex::new(ExternalRunStore::new(
            external_dir.path(),
            DirectIoConfig::disabled(),
        )));
        let child_run = {
            let mut guard = external_store.lock();
            let extent = super::write_child_run(&mut guard, 0, &child_points).unwrap();
            super::ChildRun {
                extents: vec![extent],
                len: child_points.len(),
                seed: 0x5eed,
            }
        };

        let (external_tx, external_rx) = unbounded();
        let external_assignment_context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None);
        super::recurse_child_run_maybe_resident(
            &point_store,
            child_run.clone(),
            1,
            child_points.len(),
            Metric::L2,
            &params,
            adaptive_c_max,
            min_recurse_size,
            &ProgressBar::hidden(),
            &external_store,
            &RootFanoutState::fixed(params.fanout_top),
            &external_assignment_context,
            &external_tx,
        )
        .unwrap();
        drop(external_tx);
        let (actual_tx, actual_rx) = unbounded();
        let actual_assignment_context =
            super::AssignmentContext::new_for_assignment_with_adsampling_dataset(1, &params, None);
        let stats = super::recurse_child_run_maybe_resident(
            &point_store,
            child_run,
            1,
            child_points.len(),
            Metric::L2,
            &params,
            adaptive_c_max,
            min_recurse_size,
            &ProgressBar::hidden(),
            &external_store,
            &RootFanoutState::fixed(params.fanout_top),
            &actual_assignment_context,
            &actual_tx,
        )
        .unwrap();
        drop(actual_tx);

        assert_eq!(
            collect_sorted_leaves(actual_rx),
            collect_sorted_leaves(external_rx)
        );
        assert!(stats.telemetry.io_planned_forgeann.executed_vector_run >= 1);
    }

    #[test]
    pub fn strict_prefetch_profile_contributes_io_pain_by_depth() {
        let mut stats = super::PartitionStats::new(4, 8);
        stats.telemetry.io_planned_forgeann.enabled = true;
        let mut profile = super::GemmProfile {
            assignment_decision: Some(super::AssignmentDecisionRecord {
                depth: 3,
                points: 12_000,
                leaders: 64,
                fanout: 1,
                wall: std::time::Duration::from_millis(200),
                adsampling: false,
                fallback_reason: None,
                recall_at_fanout: 0.0,
                mismatches: 0,
            }),
            ..super::GemmProfile::default()
        };
        profile.prefetch.pipeline = super::PrefetchPipelineKind::Strict;
        profile.prefetch.batches = 2;
        profile.prefetch.range_reads = 1_024;
        profile.prefetch.point_reads = 512;
        profile.prefetch.logical_bytes = 12_000 * 768 * size_of::<f32>() as u64;
        profile.prefetch.physical_bytes = profile.prefetch.logical_bytes * 3 / 2;
        profile.prefetch.io_wall = std::time::Duration::from_millis(140);
        profile.prefetch.consumer_wait = std::time::Duration::from_millis(120);
        profile.prefetch.producer_wait = std::time::Duration::from_millis(7);
        profile.prefetch.prefetch_used_peak_bytes = 64 * 1024 * 1024;

        stats.record_gemm_profile(profile);

        let depth = stats
            .telemetry
            .io_planned_forgeann
            .io_pain_by_depth
            .get(&3)
            .expect("strict prefetch profile should populate depth I/O pain");
        assert_eq!(depth.point_gather_batches, 2);
        assert_eq!(depth.point_gather_windows, 1_536);
        assert_eq!(depth.logical_bytes, 12_000 * 768 * size_of::<f32>() as u64);
        assert_eq!(depth.physical_bytes, depth.logical_bytes * 3 / 2);
        assert_eq!(depth.consumer_wait_ms, 120.0);
        assert_eq!(depth.prefetch_used_peak_bytes, 64 * 1024 * 1024);
    }
}
