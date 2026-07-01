use std::path::PathBuf;

use crate::common::Metric;

#[derive(Clone, Debug)]
pub struct ForgeANNParams {
    pub m_hash_bits: usize,
    pub l_max: usize,
    pub c_min: usize,
    pub c_max: usize,
    pub psamp_fraction: f64,
    pub max_leaders: usize,
    pub max_depth: usize,
    pub min_shrink_ratio: f32,
    pub fanout_top: usize,
    pub fanout_second: usize,
    pub leaf_knn: usize,
    pub max_full_matrix_leaf_size: usize,
    pub random_seed: u64,
    pub metric: Metric,
    pub oom_enable: bool,
    pub oom_memory_budget_bytes: usize,
    pub oom_sketch_cache_bytes: usize,
    pub oom_spill_cache_bytes: usize,
    pub oom_resident_reservoir_cap_bytes: Option<usize>,
    pub oom_temp_dir: PathBuf,
    pub oom_sketch_temp_dir: PathBuf,
    pub oom_spill_temp_dir: PathBuf,
    pub oom_partition_temp_dir: PathBuf,
    pub oom_vector_temp_dir: PathBuf,
    pub oom_keep_artifacts: bool,
    pub oom_profile_json: Option<PathBuf>,
    pub d2_trace_dump: Option<PathBuf>,
    pub adsampling_epsilon: f32,
    pub adsampling_group_dims: usize,
    pub adsampling_rotation_sidecar: Option<PathBuf>,
    pub leaf_ads_tiling_enable: bool,
    pub leaf_ads_cpu_budget: usize,
    pub leaf_ads_target_tile_ms: u64,
    pub leaf_ads_min_tile_rows: usize,
    pub leaf_ads_max_tile_rows: usize,
    pub leaf_ads_split_threshold: usize,
    pub leaf_ads_wavefront_pairmask_enable: bool,
    pub leaf_batch_drain_enable: bool,
    pub leaf_batch_drain_max_leaves: usize,
    pub leaf_batch_drain_max_points: usize,
    pub leaf_batch_drain_min_backlog: usize,
    pub leaf_ads_work_graph_enable: bool,
    pub leaf_ads_work_graph_quantum_ms: u64,
    pub leaf_ads_work_graph_target_queue_ms: u64,
    pub leaf_ads_work_graph_min_tile_rows: usize,
    pub leaf_ads_work_graph_max_tile_rows: usize,
    pub leaf_ads_work_graph_split_threshold: usize,
    pub final_prune_enable: bool,
    pub graph_start_hint: Option<u32>,
    pub spine_overlay_prune_enable: bool,
    pub spine_overlay_budget: usize,
    pub spine_overlay_spine_fraction: f32,
    pub view_lune_prune_enable: bool,
    pub view_lune_oracle_json: Option<PathBuf>,
    pub view_lune_candidate_width_multiplier: f32,
    pub view_lune_max_witnesses_per_victim: usize,
    pub view_lune_nearest_core: usize,
}

impl ForgeANNParams {
    pub const ADS_CHUNKS_PER_WORKER: usize = 8;
    pub const ADS_DEPTH_MAX_DEPTH: usize = 2;
    pub const ADS_DEPTH_SEED_EXACT_M: usize = 8;
    pub const ADS_FALLBACK_EXACT_ON_COLLAPSE: bool = true;
    pub const ADS_HUGE_MIN_WORK: usize = 1_000_000_000;
    pub const ADS_LARGE_MIN_LEADERS: usize = 1_500;
    pub const ADS_LARGE_MIN_POINTS: usize = 350_000;
    pub const ADS_LARGE_MIN_WORK: usize = 512_000_000;
    pub const ADS_MIN_EFFECTIVE_PARALLELISM: f64 = 8.0;
    pub const ADS_ROOT_SEED_EXACT_M: usize = 8;
    pub const ADS_SPARSE_FULL64_THRESHOLD: u32 = 8;
    pub const ADS_SPARSE_GROUP16_THRESHOLD: u32 = 4;
    pub const ADS_WAVE_DEPTHS: usize = 1;
    pub const DEFAULT_OOM_MEMORY_BUDGET_BYTES: usize = 12 * 1024 * 1024 * 1024;
    pub const IO_PLAN_MAX_AVG_READ_SIZE_BYTES: usize = 16 * 1024;
    pub const IO_PLAN_MIN_CONSUMER_WAIT_RATIO: f64 = 0.5;
    pub const IO_PLAN_MIN_DEPTH: usize = 2;
    pub const IO_PLAN_MIN_RESIDENT_POINTS: usize = 4096;
    pub const IO_PLAN_MIN_SAVED_WAIT_MS: f64 = 1.0;
    pub const IO_PLAN_MIN_SAVED_WAIT_PER_GIB_MS: f64 = 100.0;
    pub const IO_PLAN_RESIDENT_BUDGET_RATIO: f64 = 0.25;
    pub const IO_PLAN_WINDOW_CACHE_BYTES: usize = 1536 * 1024 * 1024;
    pub const LEAF_ADSAMPLING_MIN_SIZE: usize = 513;
    pub const LEAF_ADSAMPLING_SEED_EXACT_M: usize = 8;
    pub const LEAF_ADS_DEFAULT_MAX_TILE_ROWS: usize = 1024;
    pub const LEAF_ADS_DEFAULT_MIN_TILE_ROWS: usize = 64;
    pub const LEAF_ADS_DEFAULT_SPLIT_THRESHOLD: usize = 1024;
    pub const LEAF_ADS_DEFAULT_TARGET_TILE_MS: u64 = 8;
    pub const LEAF_ADS_WORK_GRAPH_DEFAULT_MAX_TILE_ROWS: usize = 2048;
    pub const LEAF_ADS_WORK_GRAPH_DEFAULT_MIN_TILE_ROWS: usize = 256;
    pub const LEAF_ADS_WORK_GRAPH_DEFAULT_QUANTUM_MS: u64 = 16;
    pub const LEAF_ADS_WORK_GRAPH_DEFAULT_SPLIT_THRESHOLD: usize = 1024;
    pub const LEAF_ADS_WORK_GRAPH_DEFAULT_TARGET_QUEUE_MS: u64 = 200;
    pub const LEAF_BATCH_DRAIN_DEFAULT_MAX_LEAVES: usize = 16;
    pub const LEAF_BATCH_DRAIN_DEFAULT_MAX_POINTS: usize = 16_384;
    pub const LEAF_BATCH_DRAIN_DEFAULT_MIN_BACKLOG: usize = 2;
    pub const MIN_OOM_MEMORY_BUDGET_BYTES: usize = 512 * 1024 * 1024;
    pub const OOM_POINT_PIPELINE_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;
    pub const OOM_POINT_PIPELINE_IO_THREADS: usize = 4;
    pub const OOM_POINT_PIPELINE_MAX_READ_AMPLIFICATION: f64 = 1.5;
    pub const OOM_POINT_PIPELINE_MAX_WINDOW_BYTES: usize = 16 * 1024 * 1024;
    pub const OOM_POINT_PIPELINE_MIN_LEAF_POINTS: usize = 1024;
    pub const OOM_POINT_PIPELINE_QUEUE_DEPTH: usize = 32;
    pub const OOM_STRICT_PREFETCH_BUDGET_BYTES: usize = 6 * 1024 * 1024 * 1024;
    pub const OOM_STRICT_PREFETCH_MAX_GAP_ROWS: usize = 4;
    pub const OOM_STRICT_PREFETCH_MAX_WINDOW_BYTES: usize = 16 * 1024 * 1024;
    pub const OOM_STRICT_PREFETCH_QUEUE_DEPTH: usize = 3;

    pub fn production_sota_oom() -> Self {
        let mut params = Self::default();
        params.m_hash_bits = 12;
        params.l_max = 64;
        params.c_min = 256;
        params.c_max = 1280;
        params.psamp_fraction = 0.0057;
        params.max_leaders = 5000;
        params.fanout_top = 8;
        params.fanout_second = 3;
        params.leaf_knn = 2;
        params.random_seed = 42;
        params.metric = Metric::L2;
        params.oom_enable = true;
        params.adsampling_epsilon = 1.0;
        params.adsampling_group_dims = 64;
        params.leaf_ads_wavefront_pairmask_enable = true;
        params.final_prune_enable = false;
        params
    }

    #[inline]
    pub fn fanout_for_depth(&self, depth: usize) -> usize {
        match depth {
            0 => self.fanout_top,
            1 => self.fanout_second,
            _ => 1,
        }
    }

    #[inline]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    #[inline]
    pub fn kernel_safe_leaf_size(&self) -> usize {
        self.max_full_matrix_leaf_size.max(1)
    }

    #[inline]
    pub fn root_adsampling_enabled(&self) -> bool {
        true
    }

    #[inline]
    pub fn adsampling_enabled(&self) -> bool {
        self.root_adsampling_enabled()
    }

    #[inline]
    pub fn adsampling_rotation_available(&self) -> bool {
        self.adsampling_rotation_sidecar.is_some()
    }

    #[inline]
    pub fn adaptive_psamp_fraction(&self, cluster_size: usize, depth: usize) -> f64 {
        let mut frac = self.psamp_fraction;

        if cluster_size >= 20_000_000 {
            frac *= 0.05;
        } else if cluster_size >= 10_000_000 {
            frac *= 0.1;
        } else if cluster_size >= 5_000_000 {
            frac *= 0.2;
        } else if cluster_size >= 1_000_000 {
            frac *= 0.4;
        } else if cluster_size >= 100_000 {
            frac *= 0.7;
        }

        if depth > 4 {
            frac *= 0.35;
        } else if depth > 2 {
            frac *= 0.5;
        } else if depth > 0 {
            frac *= 0.8;
        }

        frac.clamp(0.00005, 0.1)
    }

    #[inline]
    pub fn adaptive_c_max(&self, total_size: usize) -> usize {
        if total_size >= 5_000_000 {
            self.c_max.saturating_mul(2)
        } else {
            self.c_max
        }
    }

    #[inline]
    pub fn adaptive_fanout(&self, _cluster_size: usize, depth: usize) -> usize {
        self.fanout_for_depth(depth)
    }

    #[inline]
    pub fn effective_oom_memory_budget_bytes(&self) -> usize {
        if self.oom_memory_budget_bytes == 0 {
            Self::DEFAULT_OOM_MEMORY_BUDGET_BYTES
        } else {
            self.oom_memory_budget_bytes
                .max(Self::MIN_OOM_MEMORY_BUDGET_BYTES)
        }
    }

    #[inline]
    pub fn strict_oom_io_enabled(&self) -> bool {
        self.oom_enable
    }

    #[inline]
    pub fn strict_oom_prefetch_pipeline_enabled(&self) -> bool {
        self.strict_oom_io_enabled()
    }

    #[inline]
    pub fn io_planned_forgeann_enabled(&self) -> bool {
        self.oom_enable
    }

    #[inline]
    pub fn io_plan_temp_budget_bytes(&self) -> usize {
        self.effective_oom_memory_budget_bytes() * 15 / 100
    }

    #[inline]
    pub fn effective_oom_strict_prefetch_budget_bytes(&self) -> usize {
        Self::OOM_STRICT_PREFETCH_BUDGET_BYTES
    }
}

impl Default for ForgeANNParams {
    fn default() -> Self {
        Self {
            m_hash_bits: 12,
            l_max: 64,
            c_min: 256,
            c_max: 2048,
            psamp_fraction: 0.001,
            max_leaders: 1000,
            max_depth: 10,
            min_shrink_ratio: 1.5,
            fanout_top: 10,
            fanout_second: 3,
            leaf_knn: 2,
            max_full_matrix_leaf_size: 3500,
            random_seed: 42,
            metric: Metric::L2,
            oom_enable: false,
            oom_memory_budget_bytes: 0,
            oom_sketch_cache_bytes: 0,
            oom_spill_cache_bytes: 0,
            oom_resident_reservoir_cap_bytes: None,
            oom_temp_dir: PathBuf::new(),
            oom_sketch_temp_dir: PathBuf::new(),
            oom_spill_temp_dir: PathBuf::new(),
            oom_partition_temp_dir: PathBuf::new(),
            oom_vector_temp_dir: PathBuf::new(),
            oom_keep_artifacts: false,
            oom_profile_json: None,
            d2_trace_dump: None,
            adsampling_epsilon: 1.75,
            adsampling_group_dims: 32,
            adsampling_rotation_sidecar: None,
            leaf_ads_tiling_enable: false,
            leaf_ads_cpu_budget: 0,
            leaf_ads_target_tile_ms: Self::LEAF_ADS_DEFAULT_TARGET_TILE_MS,
            leaf_ads_min_tile_rows: Self::LEAF_ADS_DEFAULT_MIN_TILE_ROWS,
            leaf_ads_max_tile_rows: Self::LEAF_ADS_DEFAULT_MAX_TILE_ROWS,
            leaf_ads_split_threshold: Self::LEAF_ADS_DEFAULT_SPLIT_THRESHOLD,
            leaf_ads_wavefront_pairmask_enable: true,
            leaf_batch_drain_enable: true,
            leaf_batch_drain_max_leaves: Self::LEAF_BATCH_DRAIN_DEFAULT_MAX_LEAVES,
            leaf_batch_drain_max_points: Self::LEAF_BATCH_DRAIN_DEFAULT_MAX_POINTS,
            leaf_batch_drain_min_backlog: Self::LEAF_BATCH_DRAIN_DEFAULT_MIN_BACKLOG,
            leaf_ads_work_graph_enable: false,
            leaf_ads_work_graph_quantum_ms: Self::LEAF_ADS_WORK_GRAPH_DEFAULT_QUANTUM_MS,
            leaf_ads_work_graph_target_queue_ms: Self::LEAF_ADS_WORK_GRAPH_DEFAULT_TARGET_QUEUE_MS,
            leaf_ads_work_graph_min_tile_rows: Self::LEAF_ADS_WORK_GRAPH_DEFAULT_MIN_TILE_ROWS,
            leaf_ads_work_graph_max_tile_rows: Self::LEAF_ADS_WORK_GRAPH_DEFAULT_MAX_TILE_ROWS,
            leaf_ads_work_graph_split_threshold: Self::LEAF_ADS_WORK_GRAPH_DEFAULT_SPLIT_THRESHOLD,
            final_prune_enable: false,
            graph_start_hint: None,
            spine_overlay_prune_enable: false,
            spine_overlay_budget: 2,
            spine_overlay_spine_fraction: 0.80,
            view_lune_prune_enable: false,
            view_lune_oracle_json: None,
            view_lune_candidate_width_multiplier: 2.0,
            view_lune_max_witnesses_per_victim: 4,
            view_lune_nearest_core: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::ForgeANNParams;
    use crate::common::Metric;

    #[test]
    fn adaptive_psamp_uses_fixed_value_with_mild_depth_decay() {
        let mut params = ForgeANNParams::default();
        params.psamp_fraction = 0.001;

        assert_eq!(params.adaptive_psamp_fraction(10_000, 0), 0.001);
        assert_eq!(params.adaptive_psamp_fraction(10_000_000, 0), 0.0001);
        assert_eq!(params.adaptive_psamp_fraction(10_000_000, 3), 0.00005);
    }

    #[test]
    fn adaptive_fanout_matches_depth_schedule() {
        let mut params = ForgeANNParams::default();
        params.fanout_top = 8;
        params.fanout_second = 3;

        assert_eq!(params.adaptive_fanout(10_000, 0), 8);
        assert_eq!(params.adaptive_fanout(2_000_000, 1), 3);
        assert_eq!(params.adaptive_fanout(20_000_000, 4), 1);
    }

    #[test]
    fn production_sota_oom_matches_validated_wiki_profile() {
        let params = ForgeANNParams::production_sota_oom();

        assert_eq!(params.metric, Metric::L2);
        assert_eq!(params.m_hash_bits, 12);
        assert_eq!(params.l_max, 64);
        assert_eq!(params.c_min, 256);
        assert_eq!(params.c_max, 1280);
        assert_eq!(params.psamp_fraction, 0.0057);
        assert_eq!(params.fanout_top, 8);
        assert_eq!(params.fanout_second, 3);
        assert_eq!(params.leaf_knn, 2);
        assert_eq!(params.max_leaders, 5000);
        assert!(params.oom_enable);
        assert!(params.strict_oom_io_enabled());
        assert!(params.strict_oom_prefetch_pipeline_enabled());
        assert_eq!(ForgeANNParams::OOM_STRICT_PREFETCH_QUEUE_DEPTH, 3);
        assert_eq!(
            params.effective_oom_strict_prefetch_budget_bytes(),
            6 * 1024 * 1024 * 1024
        );
        assert_eq!(
            ForgeANNParams::OOM_STRICT_PREFETCH_MAX_WINDOW_BYTES,
            16 * 1024 * 1024
        );
        assert_eq!(ForgeANNParams::OOM_STRICT_PREFETCH_MAX_GAP_ROWS, 4);
        assert_eq!(ForgeANNParams::OOM_POINT_PIPELINE_IO_THREADS, 4);
        assert_eq!(ForgeANNParams::OOM_POINT_PIPELINE_QUEUE_DEPTH, 32);
        assert_eq!(
            ForgeANNParams::OOM_POINT_PIPELINE_BUDGET_BYTES,
            2 * 1024 * 1024 * 1024
        );
        assert_eq!(
            ForgeANNParams::OOM_POINT_PIPELINE_MAX_READ_AMPLIFICATION,
            1.5
        );
        assert!(params.io_planned_forgeann_enabled());
        assert_eq!(ForgeANNParams::IO_PLAN_MIN_DEPTH, 2);
        assert_eq!(ForgeANNParams::IO_PLAN_RESIDENT_BUDGET_RATIO, 0.25);
        assert_eq!(
            ForgeANNParams::IO_PLAN_WINDOW_CACHE_BYTES,
            1536 * 1024 * 1024
        );
        assert_eq!(ForgeANNParams::IO_PLAN_MIN_RESIDENT_POINTS, 4096);
        assert_eq!(ForgeANNParams::IO_PLAN_MIN_SAVED_WAIT_MS, 1.0);
        assert_eq!(ForgeANNParams::IO_PLAN_MIN_SAVED_WAIT_PER_GIB_MS, 100.0);
        assert!(params.root_adsampling_enabled());
        assert_eq!(params.adsampling_epsilon, 1.0);
        assert_eq!(params.adsampling_group_dims, 64);
        assert!(!params.leaf_ads_tiling_enable);
        assert_eq!(params.leaf_ads_cpu_budget, 0);
        assert_eq!(
            params.leaf_ads_target_tile_ms,
            ForgeANNParams::LEAF_ADS_DEFAULT_TARGET_TILE_MS
        );
        assert_eq!(
            params.leaf_ads_min_tile_rows,
            ForgeANNParams::LEAF_ADS_DEFAULT_MIN_TILE_ROWS
        );
        assert_eq!(
            params.leaf_ads_max_tile_rows,
            ForgeANNParams::LEAF_ADS_DEFAULT_MAX_TILE_ROWS
        );
        assert_eq!(
            params.leaf_ads_split_threshold,
            ForgeANNParams::LEAF_ADS_DEFAULT_SPLIT_THRESHOLD
        );
        assert!(params.leaf_ads_wavefront_pairmask_enable);
        assert!(!params.final_prune_enable);
        assert!(!params.leaf_ads_work_graph_enable);
        assert_eq!(
            params.leaf_ads_work_graph_quantum_ms,
            ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_QUANTUM_MS
        );
        assert_eq!(
            params.leaf_ads_work_graph_target_queue_ms,
            ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_TARGET_QUEUE_MS
        );
        assert_eq!(
            params.leaf_ads_work_graph_min_tile_rows,
            ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_MIN_TILE_ROWS
        );
        assert_eq!(
            params.leaf_ads_work_graph_max_tile_rows,
            ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_MAX_TILE_ROWS
        );
        assert_eq!(
            params.leaf_ads_work_graph_split_threshold,
            ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_SPLIT_THRESHOLD
        );
        assert_eq!(ForgeANNParams::ADS_ROOT_SEED_EXACT_M, 8);
        assert_eq!(ForgeANNParams::ADS_DEPTH_MAX_DEPTH, 2);
        assert_eq!(ForgeANNParams::ADS_DEPTH_SEED_EXACT_M, 8);
        assert_eq!(ForgeANNParams::ADS_WAVE_DEPTHS, 1);
        assert_eq!(ForgeANNParams::ADS_CHUNKS_PER_WORKER, 8);
        assert!(ForgeANNParams::ADS_FALLBACK_EXACT_ON_COLLAPSE);
    }

    #[test]
    fn default_params_are_non_oom_library_defaults() {
        let params = ForgeANNParams::default();

        assert!(!params.oom_enable);
        assert_eq!(params.oom_memory_budget_bytes, 0);
        assert_eq!(params.oom_temp_dir, PathBuf::new());
        assert_eq!(params.oom_sketch_temp_dir, PathBuf::new());
        assert_eq!(params.oom_spill_temp_dir, PathBuf::new());
        assert_eq!(params.oom_partition_temp_dir, PathBuf::new());
        assert_eq!(params.oom_vector_temp_dir, PathBuf::new());
        assert_eq!(params.oom_profile_json, None);
        assert!(params.leaf_ads_wavefront_pairmask_enable);
        assert!(!params.leaf_ads_tiling_enable);
        assert!(!params.leaf_ads_work_graph_enable);
    }

    #[test]
    fn effective_oom_memory_budget_uses_default_when_unset() {
        let params = ForgeANNParams::default();
        assert_eq!(
            params.effective_oom_memory_budget_bytes(),
            ForgeANNParams::DEFAULT_OOM_MEMORY_BUDGET_BYTES
        );
    }
}
