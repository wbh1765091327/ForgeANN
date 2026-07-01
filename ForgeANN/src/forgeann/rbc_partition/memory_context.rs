use super::*;

pub(crate) struct DepthWaveChildRunSchedule {
    pub wave_runs: Vec<ChildRun>,
    pub exact_runs: Vec<ChildRun>,
}

pub(crate) fn io_plan_config_for_params(params: &ForgeANNParams) -> IoPlanConfig {
    if !params.io_planned_forgeann_enabled() {
        return IoPlanConfig::disabled();
    }
    IoPlanConfig {
        enabled: true,
        dry_run: false,
        allow_full_build: true,
        min_depth: ForgeANNParams::IO_PLAN_MIN_DEPTH,
        vector_run_enable: true,
        resident_enable: true,
        prefetch_streaming_enable: true,
        small_run_wave_enable: true,
        child_run_batching_enable: true,
        temp_budget_bytes: params.io_plan_temp_budget_bytes(),
        window_cache_bytes: ForgeANNParams::IO_PLAN_WINDOW_CACHE_BYTES,
        min_resident_points: ForgeANNParams::IO_PLAN_MIN_RESIDENT_POINTS,
        max_resident_points: 0,
        max_avg_read_size_bytes: ForgeANNParams::IO_PLAN_MAX_AVG_READ_SIZE_BYTES,
        min_consumer_wait_ratio: ForgeANNParams::IO_PLAN_MIN_CONSUMER_WAIT_RATIO,
        min_saved_wait_ms: ForgeANNParams::IO_PLAN_MIN_SAVED_WAIT_MS,
        min_saved_wait_per_gib_ms: ForgeANNParams::IO_PLAN_MIN_SAVED_WAIT_PER_GIB_MS,
        verify: false,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MergedRawGroup {
    pub raw_children: Vec<u16>,
    pub raw_len: usize,
}

#[derive(Debug)]
pub(crate) struct MaterializedMergedChild {
    pub extents: Vec<RunExtent>,
    pub points: Option<Vec<u32>>,
}

#[derive(Debug)]
pub(crate) struct BufferedMergedChild {
    pub extents: Vec<RunExtent>,
    pub buffer: Vec<u32>,
    pub inline: bool,
}

pub(crate) struct ExternalPartitionAttempt {
    pub assign_spool: NamedTempFile,
    pub fanout: usize,
    pub merge_groups: Vec<MergedRawGroup>,
}

#[derive(Debug)]
pub(crate) struct RootFanoutState {
    pub profile: Mutex<RootFanoutProfile>,
    pub forced_root_leaders: Option<Vec<u32>>,
}

impl RootFanoutState {
    pub(crate) fn fixed(fanout: usize) -> Self {
        Self {
            profile: Mutex::new(RootFanoutProfile::fixed(fanout)),
            forced_root_leaders: None,
        }
    }

    pub(crate) fn fixed_for_root(fanout: usize, total_points: usize, root_leaders: &[u32]) -> Self {
        let mut profile = RootFanoutProfile::fixed(fanout);
        profile.total_points = total_points;
        profile.root_leader_hash = root_leader_fingerprint(root_leaders);
        update_projected_root_fanout_bytes(&mut profile);
        Self {
            profile: Mutex::new(profile),
            forced_root_leaders: Some(root_leaders.to_vec()),
        }
    }

    pub(crate) fn profile(&self) -> RootFanoutProfile {
        self.profile.lock().clone()
    }
}

pub(crate) fn select_root_leaders(
    indices: &[u32],
    params: &ForgeANNParams,
    rng: &mut (impl Rng + Send),
) -> Vec<u32> {
    if indices.is_empty() {
        return Vec::new();
    }

    // Keep leader selection deterministic given rng seed and avoid reading point vectors here.
    // Root experimental selection policies (radius/risk/reachable) were removed; this is a
    // fixed-only sampling heuristic.
    let max_leaders = params.max_leaders.max(1).min(indices.len());
    let mut num_leaders = ((params.psamp_fraction * indices.len() as f64).round() as usize).max(1);
    num_leaders = num_leaders.min(max_leaders);

    let sample_seed: u64 = rng.random();
    sample_set_bottomk(indices, num_leaders, sample_seed)
}

pub(crate) fn root_leader_fingerprint(leaders: &[u32]) -> u64 {
    let mut hasher = Sha256::new();
    for &leader in leaders {
        hasher.update(leader.to_le_bytes());
    }
    let digest = hasher.finalize();
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("sha256 output must have 32 bytes"),
    )
}

fn update_projected_root_fanout_bytes(profile: &mut RootFanoutProfile) {
    let projected_assignments: usize = profile
        .kept_hist
        .iter()
        .enumerate()
        .map(|(fanout, count)| fanout.saturating_mul(*count))
        .sum();
    profile.projected_assignment_bytes = projected_assignments.saturating_mul(size_of::<u32>());
    profile.projected_partition_d00_bytes_pre_dedup = profile.projected_assignment_bytes;
}

pub(crate) fn initialize_root_fanout_state(
    _dataset: &dyn PointStore,
    indices: &[u32],
    leaders: &[u32],
    params: &ForgeANNParams,
    _metric: Metric,
    _rng: &mut (impl Rng + Send),
) -> AnnResult<RootFanoutState> {
    let fixed_fanout = params.fanout_top.max(1).min(leaders.len().max(1)).min(32);
    if leaders.is_empty() {
        return Ok(RootFanoutState::fixed(fixed_fanout));
    }
    Ok(RootFanoutState::fixed_for_root(
        fixed_fanout,
        indices.len(),
        leaders,
    ))
}
