use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use clap::{ArgAction, Parser};
use forgeann::common::{AnnError, AnnResult, Metric};
use forgeann::forgeann::ForgeANNParams;
use forgeann::index::{ForgeAnnGraphBuildConfig, build_forgeann_graph_oom_to_file};
use forgeann::utils::{Timer, file_exists, load_metadata_from_file};

const FINAL_PRUNE_ALPHA: f32 = 1.2;
const D2_TRACE_DUMP_COMPLETE: &str = "forgeann_d2_trace_dump_complete";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ProcSample {
    thread_count: Option<usize>,
    rss_kib: Option<usize>,
    running_ns: Option<u64>,
}

#[derive(Debug)]
struct PhaseTracker {
    enabled: bool,
    stop_flag: Arc<AtomicBool>,
    current_phase: Arc<Mutex<String>>,
    sampler: Option<JoinHandle<Vec<ProcPhaseSample>>>,
    phase_name: String,
    phase_start: Instant,
    phase_cpu_start_ns: Option<u64>,
    completed_phases: Vec<CompletedPhase>,
}

#[derive(Clone, Debug)]
struct ProcPhaseSample {
    phase_name: String,
    wall_offset: Duration,
    running_ns: Option<u64>,
    thread_count: Option<usize>,
    rss_kib: Option<usize>,
}

#[derive(Clone, Debug)]
struct CompletedPhase {
    name: String,
    wall_time: Duration,
    cpu_time: Duration,
    peak_threads: usize,
    peak_rss_kib: Option<usize>,
    max_cpu_cores: f64,
    sample_count: usize,
}

impl PhaseTracker {
    fn from_env() -> Self {
        let enabled = phase_monitoring_enabled();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let current_phase = Arc::new(Mutex::new("startup".to_string()));
        let sampler = enabled
            .then(|| spawn_phase_sampler(Arc::clone(&stop_flag), Arc::clone(&current_phase)));

        Self {
            enabled,
            stop_flag,
            current_phase,
            sampler,
            phase_name: "startup".to_string(),
            phase_start: Instant::now(),
            phase_cpu_start_ns: current_running_ns(),
            completed_phases: Vec::new(),
        }
    }

    fn start_phase(&mut self, name: &str) {
        self.finish_current_phase();
        self.phase_name = name.to_string();
        *self.current_phase.lock().unwrap() = self.phase_name.clone();
        self.phase_start = Instant::now();
        self.phase_cpu_start_ns = current_running_ns();
    }

    fn finish(mut self) {
        self.finish_current_phase();
        if !self.enabled {
            return;
        }

        self.stop_flag.store(true, Ordering::Relaxed);
        let mut samples = self
            .sampler
            .take()
            .and_then(|handle| handle.join().ok())
            .unwrap_or_default();
        if let Some(sample) = read_proc_sample() {
            samples.push(ProcPhaseSample {
                phase_name: self.phase_name.clone(),
                wall_offset: self.phase_start.elapsed(),
                running_ns: sample.running_ns,
                thread_count: sample.thread_count,
                rss_kib: sample.rss_kib,
            });
        }

        annotate_completed_phases(&mut self.completed_phases, &samples);
        print_phase_monitoring_report(&self.completed_phases, &samples);
    }

    fn finish_current_phase(&mut self) {
        if !self.enabled && self.completed_phases.is_empty() {
            return;
        }

        let wall_time = self.phase_start.elapsed();
        let cpu_end_ns = current_running_ns();
        let cpu_time = match (self.phase_cpu_start_ns, cpu_end_ns) {
            (Some(start), Some(end)) if end >= start => Duration::from_nanos(end - start),
            _ => Duration::ZERO,
        };

        self.completed_phases.push(CompletedPhase {
            name: self.phase_name.clone(),
            wall_time,
            cpu_time,
            peak_threads: 0,
            peak_rss_kib: None,
            max_cpu_cores: 0.0,
            sample_count: 0,
        });
    }
}

fn phase_monitoring_enabled() -> bool {
    std::env::var("FORGEANN_PHASE_MONITOR")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn spawn_phase_sampler(
    stop_flag: Arc<AtomicBool>,
    current_phase: Arc<Mutex<String>>,
) -> JoinHandle<Vec<ProcPhaseSample>> {
    let start = Instant::now();
    const SAMPLE_INTERVAL: Duration = Duration::from_millis(200);

    thread::spawn(move || {
        let mut samples = Vec::new();
        while !stop_flag.load(Ordering::Relaxed) {
            let phase_name = current_phase.lock().unwrap().clone();
            if let Some(sample) = read_proc_sample() {
                samples.push(ProcPhaseSample {
                    phase_name,
                    wall_offset: start.elapsed(),
                    running_ns: sample.running_ns,
                    thread_count: sample.thread_count,
                    rss_kib: sample.rss_kib,
                });
            }
            thread::sleep(SAMPLE_INTERVAL);
        }
        samples
    })
}

fn read_proc_sample() -> Option<ProcSample> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let mut snapshot = parse_status_snapshot(&status)?;
    snapshot.running_ns = current_running_ns();
    Some(snapshot)
}

fn current_running_ns() -> Option<u64> {
    let schedstat = fs::read_to_string("/proc/self/schedstat").ok()?;
    parse_schedstat_running_ns(&schedstat)
}

fn parse_status_snapshot(status: &str) -> Option<ProcSample> {
    let mut thread_count = None;
    let mut rss_kib = None;

    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Threads:") {
            thread_count = rest.trim().parse::<usize>().ok();
        } else if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss_kib = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<usize>().ok());
        }
    }

    if thread_count.is_none() && rss_kib.is_none() {
        None
    } else {
        Some(ProcSample {
            thread_count,
            rss_kib,
            running_ns: None,
        })
    }
}

fn parse_schedstat_running_ns(schedstat: &str) -> Option<u64> {
    schedstat
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u64>().ok())
}

fn validate_args(args: &Args) -> AnnResult<()> {
    if !args.data_type.eq_ignore_ascii_case("float") {
        return Err(AnnError::log_index_config_error(
            "data_type".to_string(),
            "production ForgeANN build currently supports only --data-type float".to_string(),
        ));
    }
    if !args.dist_fn.eq_ignore_ascii_case("l2") {
        return Err(AnnError::log_index_config_error(
            "dist_fn".to_string(),
            "production ForgeANN ADSampling path currently requires --dist-fn l2".to_string(),
        ));
    }
    if args.m_hash_bits == 0 || args.m_hash_bits > 16 {
        return Err(AnnError::log_index_config_error(
            "m_hash_bits".to_string(),
            "m_hash_bits must be in [1, 16]".to_string(),
        ));
    }

    if args.leaf_size_min == 0 || args.leaf_size_min > args.leaf_size_max {
        return Err(AnnError::log_index_config_error(
            "leaf_size_min".to_string(),
            "leaf_size_min must be > 0 and <= leaf_size_max".to_string(),
        ));
    }
    if args.leaf_knn == 0 {
        return Err(AnnError::log_index_config_error(
            "leaf_knn".to_string(),
            "leaf_knn must be > 0".to_string(),
        ));
    }

    if args.fanout_top == 0 || args.fanout_second == 0 {
        return Err(AnnError::log_index_config_error(
            "fanout".to_string(),
            "fanout_top and fanout_second must be > 0".to_string(),
        ));
    }
    if args.max_leaders == 0 {
        return Err(AnnError::log_index_config_error(
            "max_leaders".to_string(),
            "max_leaders must be > 0".to_string(),
        ));
    }
    if !args.psamp_fraction.is_finite() || args.psamp_fraction <= 0.0 {
        return Err(AnnError::log_index_config_error(
            "psamp_fraction".to_string(),
            "psamp_fraction must be finite and > 0".to_string(),
        ));
    }
    if !args.oom_memory_budget_gb.is_finite() || args.oom_memory_budget_gb < 0.0 {
        return Err(AnnError::log_index_config_error(
            "oom_memory_budget_gb".to_string(),
            "oom-memory-budget-gb must be finite and >= 0".to_string(),
        ));
    }
    if !args.oom_sketch_cache_gb.is_finite() || args.oom_sketch_cache_gb < 0.0 {
        return Err(AnnError::log_index_config_error(
            "oom_sketch_cache_gb".to_string(),
            "oom-sketch-cache-gb must be finite and >= 0".to_string(),
        ));
    }
    if !args.oom_spill_cache_gb.is_finite() || args.oom_spill_cache_gb < 0.0 {
        return Err(AnnError::log_index_config_error(
            "oom_spill_cache_gb".to_string(),
            "oom-spill-cache-gb must be finite and >= 0".to_string(),
        ));
    }
    if let Some(cap_gb) = args.oom_resident_reservoir_cap_gb {
        if !cap_gb.is_finite() || cap_gb < 0.0 {
            return Err(AnnError::log_index_config_error(
                "oom_resident_reservoir_cap_gb".to_string(),
                "oom-resident-reservoir-cap-gb must be finite and >= 0".to_string(),
            ));
        }
    }
    if args.leaf_ads_target_tile_ms == 0 {
        return Err(AnnError::log_index_config_error(
            "leaf_ads_target_tile_ms".to_string(),
            "leaf-ads-target-tile-ms must be > 0".to_string(),
        ));
    }
    if args.leaf_ads_min_tile_rows == 0
        || args.leaf_ads_max_tile_rows == 0
        || args.leaf_ads_min_tile_rows > args.leaf_ads_max_tile_rows
    {
        return Err(AnnError::log_index_config_error(
            "leaf_ads_tile_rows".to_string(),
            "leaf-ads-min-tile-rows and leaf-ads-max-tile-rows must be > 0 and min <= max"
                .to_string(),
        ));
    }
    if args.leaf_ads_split_threshold < ForgeANNParams::LEAF_ADSAMPLING_MIN_SIZE {
        return Err(AnnError::log_index_config_error(
            "leaf_ads_split_threshold".to_string(),
            format!(
                "leaf-ads-split-threshold must be >= {}",
                ForgeANNParams::LEAF_ADSAMPLING_MIN_SIZE
            ),
        ));
    }
    if args.leaf_ads_wavefront_pairmask_enable && args.leaf_ads_wavefront_pairmask_disable {
        return Err(AnnError::log_index_config_error(
            "leaf_ads_wavefront_pairmask".to_string(),
            "leaf-ads-wavefront-pairmask-enable and leaf-ads-wavefront-pairmask-disable are mutually exclusive"
                .to_string(),
        ));
    }
    let leaf_ads_executor_flags = usize::from(args.leaf_ads_tiling_enable)
        + usize::from(args.leaf_ads_wavefront_pairmask_enable)
        + usize::from(args.leaf_ads_work_graph_enable);
    if leaf_ads_executor_flags > 1 {
        return Err(AnnError::log_index_config_error(
            "leaf_ads_executor".to_string(),
            "leaf ADS experimental executors are mutually exclusive".to_string(),
        ));
    }
    if args.leaf_batch_drain_enable && args.leaf_batch_drain_disable {
        return Err(AnnError::log_index_config_error(
            "leaf_batch_drain".to_string(),
            "leaf-batch-drain-enable and leaf-batch-drain-disable are mutually exclusive"
                .to_string(),
        ));
    }
    if args.final_prune && args.spine_overlay_prune_enable {
        return Err(AnnError::log_index_config_error(
            "final_prune".to_string(),
            "final-prune and spine-overlay-prune-enable are mutually exclusive".to_string(),
        ));
    }
    if args.final_prune && (args.view_lune_prune_enable || args.view_lune_oracle_json.is_some()) {
        return Err(AnnError::log_index_config_error(
            "final_prune".to_string(),
            "final-prune and view-lune-prune/oracle are mutually exclusive".to_string(),
        ));
    }
    if args.spine_overlay_prune_enable
        && (args.view_lune_prune_enable || args.view_lune_oracle_json.is_some())
    {
        return Err(AnnError::log_index_config_error(
            "view_lune_prune_enable".to_string(),
            "ViewLunePrune and SpineOverlayPrune are mutually exclusive".to_string(),
        ));
    }
    if args.leaf_ads_work_graph_quantum_ms == 0 {
        return Err(AnnError::log_index_config_error(
            "leaf_ads_work_graph_quantum_ms".to_string(),
            "leaf-ads-work-graph-quantum-ms must be > 0".to_string(),
        ));
    }
    if args.leaf_ads_work_graph_target_queue_ms == 0 {
        return Err(AnnError::log_index_config_error(
            "leaf_ads_work_graph_target_queue_ms".to_string(),
            "leaf-ads-work-graph-target-queue-ms must be > 0".to_string(),
        ));
    }
    if args.leaf_ads_work_graph_min_tile_rows == 0
        || args.leaf_ads_work_graph_max_tile_rows == 0
        || args.leaf_ads_work_graph_min_tile_rows > args.leaf_ads_work_graph_max_tile_rows
    {
        return Err(AnnError::log_index_config_error(
            "leaf_ads_work_graph_tile_rows".to_string(),
            "leaf-ads-work-graph-min-tile-rows and leaf-ads-work-graph-max-tile-rows must be > 0 and min <= max"
                .to_string(),
        ));
    }
    if args.leaf_ads_work_graph_split_threshold < ForgeANNParams::LEAF_ADSAMPLING_MIN_SIZE {
        return Err(AnnError::log_index_config_error(
            "leaf_ads_work_graph_split_threshold".to_string(),
            format!(
                "leaf-ads-work-graph-split-threshold must be >= {}",
                ForgeANNParams::LEAF_ADSAMPLING_MIN_SIZE
            ),
        ));
    }
    if args.spine_overlay_prune_enable {
        if !args.dist_fn.eq_ignore_ascii_case("l2") {
            return Err(AnnError::log_index_config_error(
                "spine_overlay_prune_enable".to_string(),
                "SpineOverlayPrune currently supports only --dist-fn l2".to_string(),
            ));
        }
        if args.spine_overlay_budget > args.max_degree as usize {
            return Err(AnnError::log_index_config_error(
                "spine_overlay_budget".to_string(),
                "spine-overlay-budget must be <= max-degree".to_string(),
            ));
        }
        if !args.spine_overlay_spine_fraction.is_finite()
            || args.spine_overlay_spine_fraction < 0.0
            || args.spine_overlay_spine_fraction > 1.0
        {
            return Err(AnnError::log_index_config_error(
                "spine_overlay_spine_fraction".to_string(),
                "spine-overlay-spine-fraction must be in [0, 1]".to_string(),
            ));
        }
    }
    if args.view_lune_prune_enable || args.view_lune_oracle_json.is_some() {
        if !args.dist_fn.eq_ignore_ascii_case("l2") {
            return Err(AnnError::log_index_config_error(
                "view_lune_prune_enable".to_string(),
                "ViewLunePrune currently supports only --dist-fn l2".to_string(),
            ));
        }
        if !args.view_lune_candidate_width_multiplier.is_finite()
            || args.view_lune_candidate_width_multiplier <= 0.0
        {
            return Err(AnnError::log_index_config_error(
                "view_lune_candidate_width_multiplier".to_string(),
                "view-lune-candidate-width-multiplier must be finite and > 0".to_string(),
            ));
        }
        if args.view_lune_max_witnesses_per_victim == 0 {
            return Err(AnnError::log_index_config_error(
                "view_lune_max_witnesses_per_victim".to_string(),
                "view-lune-max-witnesses-per-victim must be > 0".to_string(),
            ));
        }
        if args.view_lune_nearest_core > args.max_degree as usize {
            return Err(AnnError::log_index_config_error(
                "view_lune_nearest_core".to_string(),
                "view-lune-nearest-core must be <= max-degree".to_string(),
            ));
        }
    }

    Ok(())
}

fn leaf_ads_wavefront_pairmask_enabled(args: &Args) -> bool {
    if args.leaf_ads_tiling_enable || args.leaf_ads_work_graph_enable {
        return false;
    }
    !args.leaf_ads_wavefront_pairmask_disable
}

fn annotate_completed_phases(phases: &mut [CompletedPhase], samples: &[ProcPhaseSample]) {
    for phase in phases {
        let mut peak_threads = 0usize;
        let mut peak_rss_kib: Option<usize> = None;
        let mut sample_count = 0usize;
        let mut max_cpu_cores = 0.0f64;
        let mut previous: Option<&ProcPhaseSample> = None;

        for sample in samples
            .iter()
            .filter(|sample| sample.phase_name == phase.name)
        {
            sample_count += 1;
            if let Some(thread_count) = sample.thread_count {
                peak_threads = peak_threads.max(thread_count);
            }
            if let Some(rss_kib) = sample.rss_kib {
                peak_rss_kib = Some(match peak_rss_kib {
                    Some(peak) => peak.max(rss_kib),
                    None => rss_kib,
                });
            }

            if let Some(prev) = previous {
                if let Some(cpu_cores) = sample_cpu_cores(prev, sample) {
                    max_cpu_cores = max_cpu_cores.max(cpu_cores);
                }
            }
            previous = Some(sample);
        }

        phase.peak_threads = peak_threads;
        phase.peak_rss_kib = peak_rss_kib;
        phase.max_cpu_cores = max_cpu_cores.max(phase_cpu_cores_used(phase));
        phase.sample_count = sample_count;
    }
}

fn sample_cpu_cores(previous: &ProcPhaseSample, current: &ProcPhaseSample) -> Option<f64> {
    let prev_running_ns = previous.running_ns?;
    let current_running_ns = current.running_ns?;
    if current_running_ns < prev_running_ns {
        return None;
    }

    let wall_delta = current.wall_offset.checked_sub(previous.wall_offset)?;
    if wall_delta.is_zero() {
        return None;
    }

    Some((current_running_ns - prev_running_ns) as f64 / wall_delta.as_secs_f64() / 1_000_000_000.0)
}

fn print_phase_monitoring_report(phases: &[CompletedPhase], samples: &[ProcPhaseSample]) {
    println!("=== ForgeANN Phase Monitor ===");
    for phase in phases {
        println!(
            "  phase={} wall={:.3}s cpu={:.3}s avg_cpu_cores={:.2} peak_cpu_cores={:.2} peak_threads={} peak_rss_gib={} samples={}",
            phase.name,
            phase.wall_time.as_secs_f64(),
            phase.cpu_time.as_secs_f64(),
            phase_cpu_cores_used(phase),
            phase.max_cpu_cores,
            phase.peak_threads,
            format_optional_gib(phase.peak_rss_kib),
            phase.sample_count,
        );
    }

    if !samples.is_empty() {
        println!("  sampled timeline:");
        for sample in samples {
            println!(
                "    t={:.1}s phase={} threads={} rss_gib={} cpu_running_s={}",
                sample.wall_offset.as_secs_f64(),
                sample.phase_name,
                sample
                    .thread_count
                    .map(|count| count.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                format_optional_gib(sample.rss_kib),
                sample
                    .running_ns
                    .map(|value| format!("{:.3}", value as f64 / 1_000_000_000.0))
                    .unwrap_or_else(|| "-".to_string()),
            );
        }
    }

    println!("===========================");
}

fn phase_cpu_cores_used(phase: &CompletedPhase) -> f64 {
    if phase.wall_time.is_zero() {
        0.0
    } else {
        phase.cpu_time.as_secs_f64() / phase.wall_time.as_secs_f64()
    }
}

fn format_optional_gib(rss_kib: Option<usize>) -> String {
    rss_kib
        .map(|value| format!("{:.2}", value as f64 / 1024.0 / 1024.0))
        .unwrap_or_else(|| "-".to_string())
}

fn gib_to_bytes(value_gib: f64) -> AnnResult<usize> {
    if !value_gib.is_finite() || value_gib < 0.0 {
        return Err(AnnError::log_index_config_error(
            "oom_budget".to_string(),
            "OOM memory values must be finite and >= 0".to_string(),
        ));
    }

    let bytes = value_gib * 1024.0 * 1024.0 * 1024.0;
    if bytes > usize::MAX as f64 {
        return Err(AnnError::log_index_config_error(
            "oom_budget".to_string(),
            "OOM memory value is too large".to_string(),
        ));
    }

    Ok(bytes.round() as usize)
}

/// 使用 ForgeANN 构图并生成 `_mem.index` 的简易驱动程序。
#[allow(clippy::too_many_arguments)]
fn build_forgeann_index(
    metric: Metric,
    data_path: &str,
    r: u32,
    index_path_prefix: &str,
    num_threads: u32,
    overwrite: bool,
    forgeann_params: &ForgeANNParams,
) -> AnnResult<()> {
    let mut phase_tracker = PhaseTracker::from_env();

    let (data_num, data_dim) = load_metadata_from_file(data_path).unwrap();

    let config = ForgeAnnGraphBuildConfig::new(
        metric,
        data_dim,
        data_dim.div_ceil(32) * 32,
        data_num,
        r as usize,
        num_threads,
    )?;

    let data_path_buf = PathBuf::from(data_path);
    let timer = Timer::new();
    let index_path_prefix_buf = PathBuf::from(index_path_prefix);
    std::fs::create_dir_all(&index_path_prefix_buf)?;
    let inmem_index_path = index_path_prefix_buf.join("_mem.index");
    if !overwrite && file_exists(&inmem_index_path) {
        println!(
            "ForgeANN: in-memory graph {:?} already exists, use --overwrite to rebuild.",
            inmem_index_path
        );
        return Ok(());
    }
    println!(
        "ForgeANN: building graph for {} points (dim={}) with R={} m_hash_bits={}...",
        data_num, data_dim, r, forgeann_params.m_hash_bits
    );

    phase_tracker.start_phase("build_graph");
    let result = (|| -> AnnResult<()> {
        build_forgeann_graph_oom_to_file(
            &data_path_buf,
            &config,
            forgeann_params,
            &inmem_index_path,
        )?;

        let diff = timer.elapsed();
        println!(
            "ForgeANN: graph build done in {} seconds",
            diff.as_secs_f64()
        );
        println!(
            "ForgeANN: OOM graph written directly to {:?}",
            inmem_index_path
        );
        Ok(())
    })();
    phase_tracker.finish();

    result
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Data type; production ForgeANN currently supports float only
    #[clap(long)]
    data_type: String,

    /// Distance function; production ADSampling currently supports l2 only
    #[clap(long)]
    dist_fn: String,

    /// Input data file in bin format
    #[clap(long)]
    data_path: String,

    /// Path prefix for saving index file components
    #[clap(long)]
    index_path_prefix: String,

    /// Maximum graph degree (R), 也是 ForgeANN 的 l_max
    #[clap(long, short = 'R', default_value_t = 64)]
    max_degree: u32,

    /// Number of threads used for building index (defaults to num of CPU logic cores)
    #[clap(long, short = 'T', default_value_t = 0)]
    num_threads: u32,

    /// Overwrite existing index files
    #[clap(long)]
    overwrite: bool,

    /// HashPrune 残差哈希位数 m，必须 <= 16
    #[clap(long, default_value_t = 12)]
    m_hash_bits: usize,

    /// 叶子最小大小 C_min
    #[clap(long, default_value_t = 256)]
    leaf_size_min: usize,

    /// 叶子最大大小 C_max
    #[clap(long, default_value_t = 1280)]
    leaf_size_max: usize,

    /// RBC 采样 leader 比例 P_samp
    #[clap(long, default_value_t = 0.0057)]
    psamp_fraction: f64,

    /// 顶层 fanout
    #[clap(long, default_value_t = 8)]
    fanout_top: usize,

    /// 第二层 fanout
    #[clap(long, default_value_t = 3)]
    fanout_second: usize,

    /// Full-dimensional rotated fbin sidecar for root ADSampling assignment
    #[clap(long)]
    adsampling_rotation_sidecar: Option<PathBuf>,

    /// 叶内 k-NN 候选个数
    #[clap(long, default_value_t = 2)]
    leaf_knn: usize,

    /// 单层最多 leader 数
    #[clap(long, default_value_t = 5000)]
    max_leaders: usize,

    /// Reuse a known graph entry point and skip direct medoid start selection
    #[clap(long)]
    graph_start: Option<u32>,

    /// 随机种子
    #[clap(long, default_value_t = 42)]
    seed: u64,

    /// OOM production 模式总内存预算（GiB）
    #[clap(long, default_value_t = 0.0)]
    oom_memory_budget_gb: f64,

    /// OOM production 模式 sketch cache 预算（GiB）
    #[clap(long, default_value_t = 0.0)]
    oom_sketch_cache_gb: f64,

    /// OOM production 模式 spill cache 预算（GiB）
    #[clap(long, default_value_t = 0.0)]
    oom_spill_cache_gb: f64,

    /// Optional cap for resident hash-prune reservoirs in GiB; 0 disables resident reservoirs
    #[clap(long)]
    oom_resident_reservoir_cap_gb: Option<f64>,

    /// OOM production 模式临时目录
    #[clap(long, default_value = "")]
    oom_temp_dir: String,

    /// Optional OOM sketch artifact directory; defaults to --oom-temp-dir
    #[clap(long, default_value = "")]
    oom_sketch_temp_dir: String,

    /// Optional OOM spill/candidate artifact directory; defaults to --oom-temp-dir
    #[clap(long, default_value = "")]
    oom_spill_temp_dir: String,

    /// Optional OOM partition child-run artifact directory; defaults to --oom-temp-dir
    #[clap(long, default_value = "")]
    oom_partition_temp_dir: String,

    /// Optional OOM vector-run artifact directory; defaults to partition artifact directory
    #[clap(long, default_value = "")]
    oom_vector_temp_dir: String,

    /// 保留 OOM production 中间产物
    #[clap(long)]
    oom_keep_artifacts: bool,

    /// 将 OOM production profiling 额外写到 JSON 文件
    #[clap(long)]
    oom_profile_json: Option<PathBuf>,

    /// Enable experimental CPU-budgeted tiled leaf ADS operator executor
    #[clap(long)]
    leaf_ads_tiling_enable: bool,

    /// Explicitly select the production wavefront pair-mask leaf ADS executor; enabled by default
    #[clap(long)]
    leaf_ads_wavefront_pairmask_enable: bool,

    /// Disable the default wavefront pair-mask leaf ADS executor and use row-wise leaf ADS unless another executor is selected
    #[clap(long)]
    leaf_ads_wavefront_pairmask_disable: bool,

    /// Enable experimental batched leaf drain resident point hydration
    #[clap(long)]
    leaf_batch_drain_enable: bool,

    /// Disable batched leaf drain resident point hydration
    #[clap(long)]
    leaf_batch_drain_disable: bool,

    /// Maximum leaves to hydrate together in one batched leaf drain
    #[clap(long, default_value_t = ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MAX_LEAVES)]
    leaf_batch_drain_max_leaves: usize,

    /// Maximum total leaf points to hydrate together in one batched leaf drain
    #[clap(long, default_value_t = ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MAX_POINTS)]
    leaf_batch_drain_max_points: usize,

    /// Minimum queued leaf backlog required before forming batched leaf drains
    #[clap(long, default_value_t = ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MIN_BACKLOG)]
    leaf_batch_drain_min_backlog: usize,

    /// Leaf ADS tiled executor CPU budget; 0 derives from --num-threads
    #[clap(long, default_value_t = 0)]
    leaf_ads_cpu_budget: usize,

    /// Target wall time per leaf ADS source-row tile
    #[clap(long, default_value_t = ForgeANNParams::LEAF_ADS_DEFAULT_TARGET_TILE_MS)]
    leaf_ads_target_tile_ms: u64,

    /// Minimum rows per leaf ADS source-row tile
    #[clap(long, default_value_t = ForgeANNParams::LEAF_ADS_DEFAULT_MIN_TILE_ROWS)]
    leaf_ads_min_tile_rows: usize,

    /// Maximum rows per leaf ADS source-row tile
    #[clap(long, default_value_t = ForgeANNParams::LEAF_ADS_DEFAULT_MAX_TILE_ROWS)]
    leaf_ads_max_tile_rows: usize,

    /// Minimum leaf size that uses the tiled leaf ADS executor
    #[clap(long, default_value_t = ForgeANNParams::LEAF_ADS_DEFAULT_SPLIT_THRESHOLD)]
    leaf_ads_split_threshold: usize,

    /// Enable experimental credit-bounded flat leaf ADS work graph
    #[clap(long)]
    leaf_ads_work_graph_enable: bool,

    /// Target compute quantum for one flat ADS handle
    #[clap(long, default_value_t = ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_QUANTUM_MS)]
    leaf_ads_work_graph_quantum_ms: u64,

    /// Target queued downstream ADS/leaf work per worker
    #[clap(long, default_value_t = ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_TARGET_QUEUE_MS)]
    leaf_ads_work_graph_target_queue_ms: u64,

    /// Minimum rows claimed by one flat ADS handle
    #[clap(long, default_value_t = ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_MIN_TILE_ROWS)]
    leaf_ads_work_graph_min_tile_rows: usize,

    /// Maximum rows claimed by one flat ADS handle
    #[clap(long, default_value_t = ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_MAX_TILE_ROWS)]
    leaf_ads_work_graph_max_tile_rows: usize,

    /// Minimum leaf size that uses the flat ADS work graph
    #[clap(long, default_value_t = ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_SPLIT_THRESHOLD)]
    leaf_ads_work_graph_split_threshold: usize,

    /// Enable default-off SpineOverlay runtime bounded final graph path
    #[clap(long)]
    spine_overlay_prune_enable: bool,

    /// Enable final RobustPrune pass; default SOTA path writes no-final-prune graph directly
    #[clap(long, default_value_t = false, action = ArgAction::Set)]
    final_prune: bool,

    /// Number of final graph slots reserved for SpineOverlay replacement edges
    #[clap(long, default_value_t = 2)]
    spine_overlay_budget: usize,

    /// Protected base-row fraction kept before overlay insertion
    #[clap(long, default_value_t = 0.80)]
    spine_overlay_spine_fraction: f32,

    /// Enable default-off ViewLune metadata-only final graph executor
    #[clap(long)]
    view_lune_prune_enable: bool,

    /// Write ViewLune oracle report without requiring the executor to change output
    #[clap(long)]
    view_lune_oracle_json: Option<PathBuf>,

    /// Candidate row width multiplier for ViewLune, clamped internally to [R, 4R]
    #[clap(long, default_value_t = 2.0)]
    view_lune_candidate_width_multiplier: f32,

    /// Maximum retained witness pivots for each source/victim pair
    #[clap(long, default_value_t = 4)]
    view_lune_max_witnesses_per_victim: usize,

    /// Distance-nearest prefix protected from ViewLune witness pruning
    #[clap(long, default_value_t = 0)]
    view_lune_nearest_core: usize,

    /// Dump child-run trace after D1 materialization and exit
    #[clap(long)]
    oom_d2_trace_dump: Option<PathBuf>,
}

fn build_production_params(args: &Args, metric: Metric) -> AnnResult<ForgeANNParams> {
    let mut params = ForgeANNParams::production_sota_oom();
    params.metric = metric;
    params.l_max = args.max_degree as usize;
    params.m_hash_bits = args.m_hash_bits;
    params.c_min = args.leaf_size_min;
    params.c_max = args.leaf_size_max;
    params.psamp_fraction = args.psamp_fraction;
    params.fanout_top = args.fanout_top;
    params.fanout_second = args.fanout_second;
    params.leaf_knn = args.leaf_knn;
    params.max_leaders = args.max_leaders;
    params.graph_start_hint = args.graph_start;
    params.random_seed = args.seed;
    params.adsampling_rotation_sidecar = args.adsampling_rotation_sidecar.clone();
    params.oom_memory_budget_bytes = gib_to_bytes(args.oom_memory_budget_gb)?;
    params.oom_sketch_cache_bytes = gib_to_bytes(args.oom_sketch_cache_gb)?;
    params.oom_spill_cache_bytes = gib_to_bytes(args.oom_spill_cache_gb)?;
    params.oom_resident_reservoir_cap_bytes = args
        .oom_resident_reservoir_cap_gb
        .map(gib_to_bytes)
        .transpose()?;
    params.oom_temp_dir = PathBuf::from(&args.oom_temp_dir);
    params.oom_sketch_temp_dir = PathBuf::from(&args.oom_sketch_temp_dir);
    params.oom_spill_temp_dir = PathBuf::from(&args.oom_spill_temp_dir);
    params.oom_partition_temp_dir = PathBuf::from(&args.oom_partition_temp_dir);
    params.oom_vector_temp_dir = PathBuf::from(&args.oom_vector_temp_dir);
    params.oom_keep_artifacts = args.oom_keep_artifacts;
    params.oom_profile_json = args.oom_profile_json.clone();

    params.d2_trace_dump = args.oom_d2_trace_dump.clone();
    params.leaf_ads_tiling_enable = args.leaf_ads_tiling_enable;
    params.leaf_ads_wavefront_pairmask_enable = leaf_ads_wavefront_pairmask_enabled(args);
    params.leaf_batch_drain_enable = !args.leaf_batch_drain_disable;
    params.leaf_batch_drain_max_leaves = args.leaf_batch_drain_max_leaves;
    params.leaf_batch_drain_max_points = args.leaf_batch_drain_max_points;
    params.leaf_batch_drain_min_backlog = args.leaf_batch_drain_min_backlog;
    params.leaf_ads_cpu_budget = args.leaf_ads_cpu_budget;
    params.leaf_ads_target_tile_ms = args.leaf_ads_target_tile_ms;
    params.leaf_ads_min_tile_rows = args.leaf_ads_min_tile_rows;
    params.leaf_ads_max_tile_rows = args.leaf_ads_max_tile_rows;
    params.leaf_ads_split_threshold = args.leaf_ads_split_threshold;
    params.leaf_ads_work_graph_enable = args.leaf_ads_work_graph_enable;
    params.leaf_ads_work_graph_quantum_ms = args.leaf_ads_work_graph_quantum_ms;
    params.leaf_ads_work_graph_target_queue_ms = args.leaf_ads_work_graph_target_queue_ms;
    params.leaf_ads_work_graph_min_tile_rows = args.leaf_ads_work_graph_min_tile_rows;
    params.leaf_ads_work_graph_max_tile_rows = args.leaf_ads_work_graph_max_tile_rows;
    params.leaf_ads_work_graph_split_threshold = args.leaf_ads_work_graph_split_threshold;
    params.final_prune_enable = args.final_prune;
    params.spine_overlay_prune_enable = args.spine_overlay_prune_enable;
    params.spine_overlay_budget = args.spine_overlay_budget;
    params.spine_overlay_spine_fraction = args.spine_overlay_spine_fraction;
    params.view_lune_prune_enable = args.view_lune_prune_enable;
    params.view_lune_oracle_json = args.view_lune_oracle_json.clone();
    params.view_lune_candidate_width_multiplier = args.view_lune_candidate_width_multiplier;
    params.view_lune_max_witnesses_per_victim = args.view_lune_max_witnesses_per_victim;
    params.view_lune_nearest_core = args.view_lune_nearest_core;

    Ok(params)
}

fn main() -> AnnResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
        )
        .init();

    let args = Args::parse();

    validate_args(&args)?;

    let metric = Metric::L2;
    let num_threads = if args.num_threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(1)
    } else {
        args.num_threads
    };

    println!(
        "Starting ForgeANN production SOTA index build with R: {}  final_prune={} final_prune_alpha: {}  #threads: {} fanout=[{},{}] leaf_knn={}",
        args.max_degree,
        args.final_prune,
        FINAL_PRUNE_ALPHA,
        num_threads,
        args.fanout_top,
        args.fanout_second,
        args.leaf_knn
    );

    let forgeann_params = build_production_params(&args, metric)?;

    let err = match args.data_type.as_str() {
        "float" => build_forgeann_index(
            metric,
            &args.data_path,
            args.max_degree,
            &args.index_path_prefix,
            num_threads,
            args.overwrite,
            &forgeann_params,
        ),
        _ => {
            println!("Unsupported type. Use one of int8, uint8, float or f16.");
            return Err(AnnError::log_index_config_error(
                "data_type".to_string(),
                "Invalid data type".to_string(),
            ));
        }
    };

    match err {
        Ok(_) => {
            println!("ForgeANN: index build completed successfully");
            Ok(())
        }
        Err(AnnError::Index { err }) if err == D2_TRACE_DUMP_COMPLETE => {
            println!("ForgeANN: D2 trace dump completed successfully");
            Ok(())
        }
        Err(err) => {
            println!("ForgeANN: error: {:?}", err);
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Duration;

    use clap::Parser;
    use forgeann::common::Metric;
    use forgeann::index::load_mem_graph;

    use super::{
        Args, CompletedPhase, build_forgeann_index, build_production_params,
        parse_schedstat_running_ns, parse_status_snapshot, phase_cpu_cores_used, validate_args,
    };

    fn base_args() -> [&'static str; 9] {
        [
            "build_forgeann_index",
            "--data-type",
            "float",
            "--dist-fn",
            "l2",
            "--data-path",
            "/tmp/data.fbin",
            "--index-path-prefix",
            "/tmp/index",
        ]
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("forgeann-{label}-{}-{}", std::process::id(), nanos));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_test_fbin(path: &std::path::Path, rows: usize, dim: usize) {
        let mut file = File::create(path).unwrap();
        file.write_all(&(rows as u32).to_le_bytes()).unwrap();
        file.write_all(&(dim as u32).to_le_bytes()).unwrap();
        for row in 0..rows {
            for axis in 0..dim {
                let value = row as f32 * 0.25 + axis as f32 * 0.5;
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        file.flush().unwrap();
    }

    #[test]
    fn parse_status_snapshot_reads_threads_and_rss_kib() {
        let status = "\
Name:\tbuild_forgeann_index\n\
State:\tR (running)\n\
Threads:\t42\n\
VmRSS:\t  123456 kB\n";

        let snapshot = parse_status_snapshot(status).expect("status snapshot");
        assert_eq!(snapshot.thread_count, Some(42));
        assert_eq!(snapshot.rss_kib, Some(123_456));
    }

    #[test]
    fn parse_schedstat_running_ns_reads_first_field() {
        let schedstat = "123456789 987654321 42\n";
        assert_eq!(parse_schedstat_running_ns(schedstat), Some(123_456_789));
    }

    #[test]
    fn completed_phase_cpu_cores_used_uses_wall_and_cpu_deltas() {
        let phase = CompletedPhase {
            name: "build_graph".to_string(),
            wall_time: Duration::from_secs(2),
            cpu_time: Duration::from_secs(6),
            peak_threads: 14,
            peak_rss_kib: Some(2_048_000),
            max_cpu_cores: 3.5,
            sample_count: 4,
        };

        assert!((phase_cpu_cores_used(&phase) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn args_default_to_production_sota_profile() {
        let args = Args::parse_from(base_args());

        assert_eq!(args.max_degree, 64);
        assert_eq!(args.m_hash_bits, 12);
        assert_eq!(args.leaf_size_min, 256);
        assert_eq!(args.leaf_size_max, 1280);
        assert_eq!(args.psamp_fraction, 0.0057);
        assert_eq!(args.fanout_top, 8);
        assert_eq!(args.fanout_second, 3);
        assert_eq!(args.leaf_knn, 2);
        assert_eq!(args.max_leaders, 5000);
        assert!(args.oom_resident_reservoir_cap_gb.is_none());
        assert!(!args.leaf_ads_tiling_enable);
        assert!(!args.leaf_ads_wavefront_pairmask_enable);
        assert!(!args.leaf_ads_wavefront_pairmask_disable);
        assert!(!args.leaf_batch_drain_enable);
        assert!(!args.leaf_batch_drain_disable);
        assert_eq!(
            args.leaf_batch_drain_max_leaves,
            super::ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MAX_LEAVES
        );
        assert_eq!(
            args.leaf_batch_drain_max_points,
            super::ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MAX_POINTS
        );
        assert_eq!(
            args.leaf_batch_drain_min_backlog,
            super::ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MIN_BACKLOG
        );
        assert_eq!(args.leaf_ads_cpu_budget, 0);
        assert_eq!(
            args.leaf_ads_target_tile_ms,
            super::ForgeANNParams::LEAF_ADS_DEFAULT_TARGET_TILE_MS
        );
        assert_eq!(
            args.leaf_ads_min_tile_rows,
            super::ForgeANNParams::LEAF_ADS_DEFAULT_MIN_TILE_ROWS
        );
        assert_eq!(
            args.leaf_ads_max_tile_rows,
            super::ForgeANNParams::LEAF_ADS_DEFAULT_MAX_TILE_ROWS
        );
        assert_eq!(
            args.leaf_ads_split_threshold,
            super::ForgeANNParams::LEAF_ADS_DEFAULT_SPLIT_THRESHOLD
        );
        assert!(!args.leaf_ads_work_graph_enable);
        assert_eq!(
            args.leaf_ads_work_graph_quantum_ms,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_QUANTUM_MS
        );
        assert_eq!(
            args.leaf_ads_work_graph_target_queue_ms,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_TARGET_QUEUE_MS
        );
        assert_eq!(
            args.leaf_ads_work_graph_min_tile_rows,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_MIN_TILE_ROWS
        );
        assert_eq!(
            args.leaf_ads_work_graph_max_tile_rows,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_MAX_TILE_ROWS
        );
        assert_eq!(
            args.leaf_ads_work_graph_split_threshold,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_SPLIT_THRESHOLD
        );
        assert!(!args.final_prune);
        assert!(!args.spine_overlay_prune_enable);
        assert_eq!(args.spine_overlay_budget, 2);
        assert_eq!(args.spine_overlay_spine_fraction, 0.80);
        assert!(!args.view_lune_prune_enable);
        assert!(args.view_lune_oracle_json.is_none());
        assert_eq!(args.view_lune_candidate_width_multiplier, 2.0);
        assert_eq!(args.view_lune_max_witnesses_per_victim, 4);
        assert_eq!(args.view_lune_nearest_core, 0);
    }

    #[test]
    fn production_params_enable_only_sota_oom_path() {
        let args = Args::parse_from(base_args());
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(params.oom_enable);
        assert!(params.strict_oom_io_enabled());
        assert!(params.strict_oom_prefetch_pipeline_enabled());
        assert!(params.root_adsampling_enabled());
        assert_eq!(params.fanout_top, 8);
        assert_eq!(params.fanout_second, 3);
        assert_eq!(params.leaf_knn, 2);
        assert_eq!(params.max_leaders, 5000);
        assert_eq!(params.adsampling_epsilon, 1.0);
        assert_eq!(params.adsampling_group_dims, 64);
        assert!(params.oom_resident_reservoir_cap_bytes.is_none());
        assert!(!params.leaf_ads_tiling_enable);
        assert!(params.leaf_ads_wavefront_pairmask_enable);
        assert_eq!(params.leaf_ads_cpu_budget, 0);
        assert_eq!(
            params.leaf_ads_target_tile_ms,
            super::ForgeANNParams::LEAF_ADS_DEFAULT_TARGET_TILE_MS
        );
        assert_eq!(
            params.leaf_ads_min_tile_rows,
            super::ForgeANNParams::LEAF_ADS_DEFAULT_MIN_TILE_ROWS
        );
        assert_eq!(
            params.leaf_ads_max_tile_rows,
            super::ForgeANNParams::LEAF_ADS_DEFAULT_MAX_TILE_ROWS
        );
        assert_eq!(
            params.leaf_ads_split_threshold,
            super::ForgeANNParams::LEAF_ADS_DEFAULT_SPLIT_THRESHOLD
        );
        assert!(params.leaf_batch_drain_enable);
        assert_eq!(
            params.leaf_batch_drain_max_leaves,
            super::ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MAX_LEAVES
        );
        assert_eq!(
            params.leaf_batch_drain_max_points,
            super::ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MAX_POINTS
        );
        assert_eq!(
            params.leaf_batch_drain_min_backlog,
            super::ForgeANNParams::LEAF_BATCH_DRAIN_DEFAULT_MIN_BACKLOG
        );
        assert!(!params.leaf_ads_work_graph_enable);
        assert_eq!(
            params.leaf_ads_work_graph_quantum_ms,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_QUANTUM_MS
        );
        assert_eq!(
            params.leaf_ads_work_graph_target_queue_ms,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_TARGET_QUEUE_MS
        );
        assert_eq!(
            params.leaf_ads_work_graph_min_tile_rows,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_MIN_TILE_ROWS
        );
        assert_eq!(
            params.leaf_ads_work_graph_max_tile_rows,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_MAX_TILE_ROWS
        );
        assert_eq!(
            params.leaf_ads_work_graph_split_threshold,
            super::ForgeANNParams::LEAF_ADS_WORK_GRAPH_DEFAULT_SPLIT_THRESHOLD
        );
        assert!(!params.final_prune_enable);
        assert!(!params.spine_overlay_prune_enable);
        assert_eq!(params.spine_overlay_budget, 2);
        assert_eq!(params.spine_overlay_spine_fraction, 0.80);
        assert!(!params.view_lune_prune_enable);
        assert!(params.view_lune_oracle_json.is_none());
        assert_eq!(params.view_lune_candidate_width_multiplier, 2.0);
        assert_eq!(params.view_lune_max_witnesses_per_victim, 4);
        assert_eq!(params.view_lune_nearest_core, 0);
    }

    #[test]
    fn final_prune_flag_is_explicitly_configurable() {
        let args = Args::parse_from(base_args().into_iter().chain(["--final-prune=true"]));
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(params.final_prune_enable);

        let args = Args::parse_from(base_args().into_iter().chain(["--final-prune=false"]));
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(!params.final_prune_enable);
    }

    #[test]
    fn spine_overlay_prune_flags_are_forwarded_to_params() {
        let args = Args::parse_from(base_args().into_iter().chain([
            "--spine-overlay-prune-enable",
            "--spine-overlay-budget",
            "4",
            "--spine-overlay-spine-fraction",
            "0.9",
        ]));
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(params.spine_overlay_prune_enable);
        assert_eq!(params.spine_overlay_budget, 4);
        assert_eq!(params.spine_overlay_spine_fraction, 0.9);
    }

    #[test]
    fn view_lune_prune_flags_are_forwarded_to_params() {
        let args = Args::parse_from(base_args().into_iter().chain([
            "--view-lune-prune-enable",
            "--view-lune-oracle-json",
            "/tmp/view_lune.json",
            "--view-lune-candidate-width-multiplier",
            "1.5",
            "--view-lune-max-witnesses-per-victim",
            "6",
            "--view-lune-nearest-core",
            "4",
        ]));
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(params.view_lune_prune_enable);
        assert_eq!(
            params.view_lune_oracle_json,
            Some(PathBuf::from("/tmp/view_lune.json"))
        );
        assert_eq!(params.view_lune_candidate_width_multiplier, 1.5);
        assert_eq!(params.view_lune_max_witnesses_per_victim, 6);
        assert_eq!(params.view_lune_nearest_core, 4);
    }

    #[test]
    fn leaf_ads_tiling_flags_enable_experimental_operator_executor() {
        let args = Args::parse_from(base_args().into_iter().chain([
            "--leaf-ads-tiling-enable",
            "--leaf-ads-cpu-budget",
            "12",
            "--leaf-ads-target-tile-ms",
            "5",
            "--leaf-ads-min-tile-rows",
            "128",
            "--leaf-ads-max-tile-rows",
            "512",
            "--leaf-ads-split-threshold",
            "2048",
        ]));
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(params.leaf_ads_tiling_enable);
        assert!(!params.leaf_ads_wavefront_pairmask_enable);
        assert_eq!(params.leaf_ads_cpu_budget, 12);
        assert_eq!(params.leaf_ads_target_tile_ms, 5);
        assert_eq!(params.leaf_ads_min_tile_rows, 128);
        assert_eq!(params.leaf_ads_max_tile_rows, 512);
        assert_eq!(params.leaf_ads_split_threshold, 2048);
    }

    #[test]
    fn leaf_batch_drain_defaults_on_and_can_be_disabled() {
        let args = Args::parse_from(base_args().into_iter().chain([
            "--leaf-batch-drain-disable",
            "--leaf-batch-drain-max-leaves",
            "12",
            "--leaf-batch-drain-max-points",
            "8192",
            "--leaf-batch-drain-min-backlog",
            "4",
        ]));
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(!params.leaf_batch_drain_enable);
        assert_eq!(params.leaf_batch_drain_max_leaves, 12);
        assert_eq!(params.leaf_batch_drain_max_points, 8192);
        assert_eq!(params.leaf_batch_drain_min_backlog, 4);
    }

    #[test]
    fn leaf_batch_drain_enable_flag_is_accepted_for_compatibility() {
        let args = Args::parse_from(base_args().into_iter().chain(["--leaf-batch-drain-enable"]));
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(params.leaf_batch_drain_enable);
    }

    #[test]
    fn leaf_batch_drain_enable_and_disable_conflict() {
        let args = Args::parse_from(
            base_args()
                .into_iter()
                .chain(["--leaf-batch-drain-enable", "--leaf-batch-drain-disable"]),
        );

        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn leaf_ads_wavefront_pairmask_disable_uses_rowwise_ads() {
        let args = Args::parse_from(
            base_args()
                .into_iter()
                .chain(["--leaf-ads-wavefront-pairmask-disable"]),
        );
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(!params.leaf_ads_wavefront_pairmask_enable);
        assert!(!params.leaf_ads_tiling_enable);
        assert!(!params.leaf_ads_work_graph_enable);
    }

    #[test]
    fn leaf_ads_wavefront_pairmask_flag_selects_production_executor() {
        let args = Args::parse_from(
            base_args()
                .into_iter()
                .chain(["--leaf-ads-wavefront-pairmask-enable"]),
        );
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(params.leaf_ads_wavefront_pairmask_enable);
        assert!(!params.leaf_ads_tiling_enable);
        assert!(!params.leaf_ads_work_graph_enable);
    }

    #[test]
    fn leaf_ads_work_graph_flags_enable_flat_operator_executor() {
        let args = Args::parse_from(base_args().into_iter().chain([
            "--leaf-ads-work-graph-enable",
            "--leaf-ads-work-graph-quantum-ms",
            "11",
            "--leaf-ads-work-graph-target-queue-ms",
            "150",
            "--leaf-ads-work-graph-min-tile-rows",
            "256",
            "--leaf-ads-work-graph-max-tile-rows",
            "1024",
            "--leaf-ads-work-graph-split-threshold",
            "2048",
        ]));
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert!(params.leaf_ads_work_graph_enable);
        assert_eq!(params.leaf_ads_work_graph_quantum_ms, 11);
        assert_eq!(params.leaf_ads_work_graph_target_queue_ms, 150);
        assert_eq!(params.leaf_ads_work_graph_min_tile_rows, 256);
        assert_eq!(params.leaf_ads_work_graph_max_tile_rows, 1024);
        assert_eq!(params.leaf_ads_work_graph_split_threshold, 2048);
        assert!(!params.leaf_ads_tiling_enable);
        assert!(!params.leaf_ads_wavefront_pairmask_enable);
    }

    #[test]
    fn resident_reservoir_cap_flag_sets_explicit_byte_cap() {
        let args = Args::parse_from(
            base_args()
                .into_iter()
                .chain(["--oom-resident-reservoir-cap-gb", "1"]),
        );
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert_eq!(
            params.oom_resident_reservoir_cap_bytes,
            Some(1024 * 1024 * 1024)
        );
    }

    #[test]
    fn oom_split_temp_dirs_are_forwarded_to_params() {
        let args = Args::parse_from(base_args().into_iter().chain([
            "--oom-temp-dir",
            "/tmp/oom-root",
            "--oom-sketch-temp-dir",
            "/tmp/oom-sketch",
            "--oom-spill-temp-dir",
            "/tmp/oom-spill",
            "--oom-partition-temp-dir",
            "/tmp/oom-partition",
            "--oom-vector-temp-dir",
            "/tmp/oom-vector",
        ]));
        validate_args(&args).unwrap();

        let params = build_production_params(&args, Metric::L2).unwrap();
        assert_eq!(params.oom_temp_dir, PathBuf::from("/tmp/oom-root"));
        assert_eq!(params.oom_sketch_temp_dir, PathBuf::from("/tmp/oom-sketch"));
        assert_eq!(params.oom_spill_temp_dir, PathBuf::from("/tmp/oom-spill"));
        assert_eq!(
            params.oom_partition_temp_dir,
            PathBuf::from("/tmp/oom-partition")
        );
        assert_eq!(params.oom_vector_temp_dir, PathBuf::from("/tmp/oom-vector"));
    }

    #[test]
    fn removed_legacy_oom_flags_are_rejected_by_clap() {
        for flag in [
            "--oom-resident-subtree-enable",
            "--oom-resident-replay-batch-plan-only",
            "--oom-resident-replay-batch-cache-epoch-report",
            "--reuse-countdown-nextuse-enable",
            "--reuse-countdown-nextuse-disable",
            "--reuse-countdown-nextuse-admit-far",
        ] {
            let args = Args::try_parse_from(base_args().into_iter().chain([flag]));
            assert!(args.is_err(), "{flag} should be removed");
        }
        for flag in [
            "--oom-resident-subtree-strategy",
            "--oom-resident-replay-batch-target-fill",
            "--oom-resident-replay-batch-start-batch",
            "--oom-resident-replay-batch-max-batches",
            "--oom-resident-subtree-budget-gib",
            "--oom-resident-subtree-budget-frac",
            "--oom-resident-subtree-gain-threshold",
            "--oom-resident-subtree-min-duplicate-ratio",
            "--oom-resident-subtree-min-saved-rows",
            "--oom-resident-subtree-min-unique-points",
            "--reuse-countdown-cache-budget-gb",
            "--reuse-countdown-cache-report-json",
            "--reuse-countdown-max-inflight",
            "--reuse-countdown-access-trace",
            "--reuse-countdown-access-trace-limit",
            "--reuse-countdown-nextuse-window-runs",
            "--reuse-countdown-nextuse-eviction-sample",
            "--reuse-countdown-nextuse-policy",
        ] {
            let args = Args::try_parse_from(base_args().into_iter().chain([flag, "1"]));
            assert!(args.is_err(), "{flag} should be removed");
        }
    }

    #[test]
    fn invalid_leaf_ads_tiling_flags_are_rejected() {
        for argv in [
            vec!["--leaf-ads-target-tile-ms", "0"],
            vec!["--leaf-ads-min-tile-rows", "0"],
            vec!["--leaf-ads-max-tile-rows", "0"],
            vec![
                "--leaf-ads-min-tile-rows",
                "512",
                "--leaf-ads-max-tile-rows",
                "128",
            ],
            vec!["--leaf-ads-split-threshold", "63"],
        ] {
            let args = Args::parse_from(base_args().into_iter().chain(argv));
            assert!(validate_args(&args).is_err());
        }
    }

    #[test]
    fn invalid_leaf_ads_work_graph_flags_are_rejected() {
        for argv in [
            vec!["--leaf-ads-work-graph-quantum-ms", "0"],
            vec!["--leaf-ads-work-graph-target-queue-ms", "0"],
            vec!["--leaf-ads-work-graph-min-tile-rows", "0"],
            vec!["--leaf-ads-work-graph-max-tile-rows", "0"],
            vec![
                "--leaf-ads-work-graph-min-tile-rows",
                "1024",
                "--leaf-ads-work-graph-max-tile-rows",
                "256",
            ],
            vec!["--leaf-ads-work-graph-split-threshold", "63"],
            vec!["--leaf-ads-tiling-enable", "--leaf-ads-work-graph-enable"],
            vec![
                "--leaf-ads-wavefront-pairmask-enable",
                "--leaf-ads-tiling-enable",
            ],
            vec![
                "--leaf-ads-wavefront-pairmask-enable",
                "--leaf-ads-work-graph-enable",
            ],
            vec![
                "--leaf-ads-wavefront-pairmask-enable",
                "--leaf-ads-wavefront-pairmask-disable",
            ],
        ] {
            let args = Args::parse_from(base_args().into_iter().chain(argv));
            assert!(validate_args(&args).is_err());
        }
    }

    #[test]
    fn invalid_resident_reservoir_cap_flags_are_rejected() {
        for value in ["-1", "nan", "inf"] {
            let args = Args::try_parse_from(
                base_args()
                    .into_iter()
                    .chain(["--oom-resident-reservoir-cap-gb", value]),
            );
            if let Ok(args) = args {
                assert!(
                    validate_args(&args).is_err(),
                    "--oom-resident-reservoir-cap-gb={value} should fail"
                );
            }
        }
    }

    #[test]
    fn invalid_spine_overlay_prune_flags_are_rejected() {
        for argv in [
            vec![
                "--spine-overlay-prune-enable",
                "--max-degree",
                "4",
                "--spine-overlay-budget",
                "5",
            ],
            vec![
                "--spine-overlay-prune-enable",
                "--spine-overlay-spine-fraction=-0.1",
            ],
            vec![
                "--spine-overlay-prune-enable",
                "--spine-overlay-spine-fraction",
                "1.1",
            ],
        ] {
            let args = Args::parse_from(base_args().into_iter().chain(argv));
            assert!(validate_args(&args).is_err());
        }
    }

    #[test]
    fn invalid_view_lune_prune_flags_are_rejected() {
        for argv in [
            vec!["--view-lune-prune-enable", "--final-prune=true"],
            vec!["--view-lune-prune-enable", "--spine-overlay-prune-enable"],
            vec![
                "--view-lune-prune-enable",
                "--view-lune-candidate-width-multiplier",
                "0",
            ],
            vec![
                "--view-lune-prune-enable",
                "--view-lune-max-witnesses-per-victim",
                "0",
            ],
            vec![
                "--view-lune-prune-enable",
                "--max-degree",
                "4",
                "--view-lune-nearest-core",
                "5",
            ],
        ] {
            let args = Args::parse_from(base_args().into_iter().chain(argv));
            assert!(validate_args(&args).is_err());
        }
    }

    #[test]
    fn removed_experimental_flags_are_rejected_by_clap() {
        for flag in [
            "--oom-enable",
            "--plain-pipnn-oom",
            "--oom-point-store-backend",
            "--oom-strict-io",
            "--oom-point-pipeline",
            "--io-plan-window-cache-gb",
            "--root-assignment-strategy",
            "--root-stop-after-d00",
            "--level-scan-replay-d1-checkpoint-dir",
            "--adsampling-depth-breakdown-root-runs",
            "--ads-scheduler",
            "--forgeann-backend",
            "--auto-params",
            "--estimate-only",
            "--lbuild",
        ] {
            let err = Args::try_parse_from(base_args().into_iter().chain([flag])).unwrap_err();
            assert!(
                err.to_string().contains("unexpected argument"),
                "{flag} should be rejected"
            );
        }
    }

    #[test]
    fn production_oom_build_writes_loadable_mem_index() {
        let dir = unique_temp_dir("forgeann-production-oom-build");
        let data_path = dir.join("tiny.fbin");
        let index_dir = dir.join("index");
        std::fs::create_dir_all(&index_dir).unwrap();
        write_test_fbin(&data_path, 48, 8);

        let mut params = super::ForgeANNParams::production_sota_oom();
        params.oom_temp_dir = dir.join("oom-artifacts");
        params.c_min = 8;
        params.c_max = 16;
        params.max_leaders = 8;
        params.psamp_fraction = 1.0;
        params.fanout_top = 2;
        params.fanout_second = 2;

        build_forgeann_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            index_dir.to_str().unwrap(),
            1,
            true,
            &params,
        )
        .unwrap();

        let graph_path = index_dir.join("_mem.index");
        assert!(graph_path.exists());

        load_mem_graph(&graph_path, 48)
            .expect("production OOM build should emit a loadable _mem.index");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn production_oom_build_writes_loadable_mem_index_with_spine_overlay_prune() {
        let dir = unique_temp_dir("forgeann-production-spine-overlay-prune-build");
        let data_path = dir.join("tiny.fbin");
        let index_dir = dir.join("index");
        std::fs::create_dir_all(&index_dir).unwrap();
        write_test_fbin(&data_path, 48, 8);

        let mut params = super::ForgeANNParams::production_sota_oom();
        params.oom_temp_dir = dir.join("oom-artifacts");
        params.c_min = 8;
        params.c_max = 16;
        params.max_leaders = 8;
        params.psamp_fraction = 1.0;
        params.fanout_top = 2;
        params.fanout_second = 2;
        params.spine_overlay_prune_enable = true;
        params.spine_overlay_budget = 2;
        params.spine_overlay_spine_fraction = 0.80;

        build_forgeann_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            index_dir.to_str().unwrap(),
            1,
            true,
            &params,
        )
        .unwrap();

        let graph_path = index_dir.join("_mem.index");
        assert!(graph_path.exists());

        load_mem_graph(&graph_path, 48)
            .expect("SpineOverlayPrune OOM build should emit a loadable _mem.index");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn production_oom_build_writes_loadable_mem_index_with_view_lune_prune() {
        let dir = unique_temp_dir("forgeann-production-view-lune-prune-build");
        let data_path = dir.join("tiny.fbin");
        let index_dir = dir.join("index");
        let oracle_path = dir.join("view_lune_oracle.json");
        std::fs::create_dir_all(&index_dir).unwrap();
        write_test_fbin(&data_path, 48, 8);

        let mut params = super::ForgeANNParams::production_sota_oom();
        params.oom_temp_dir = dir.join("oom-artifacts");
        params.c_min = 8;
        params.c_max = 16;
        params.max_leaders = 8;
        params.psamp_fraction = 1.0;
        params.fanout_top = 2;
        params.fanout_second = 2;
        params.view_lune_prune_enable = true;
        params.view_lune_oracle_json = Some(oracle_path.clone());
        params.view_lune_candidate_width_multiplier = 2.0;
        params.view_lune_max_witnesses_per_victim = 4;

        build_forgeann_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            index_dir.to_str().unwrap(),
            1,
            true,
            &params,
        )
        .unwrap();

        let graph_path = index_dir.join("_mem.index");
        assert!(graph_path.exists());
        assert!(oracle_path.exists());
        let oracle_json = std::fs::read_to_string(&oracle_path).unwrap();
        assert!(oracle_json.contains("view_lune_prune_oracle_v1"));
        assert!(oracle_json.contains("\"final_vector_io_count\": 0"));

        load_mem_graph(&graph_path, 48)
            .expect("ViewLunePrune OOM build should emit a loadable _mem.index");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn production_oom_build_view_lune_spill_matches_memory_path() {
        let dir = unique_temp_dir("forgeann-production-view-lune-spill-build");
        let data_path = dir.join("tiny.fbin");
        let spill_index_dir = dir.join("spill-index");
        let memory_index_dir = dir.join("memory-index");
        let oracle_path = dir.join("view_lune_oracle.json");
        std::fs::create_dir_all(&spill_index_dir).unwrap();
        std::fs::create_dir_all(&memory_index_dir).unwrap();
        write_test_fbin(&data_path, 48, 8);

        let mut params = super::ForgeANNParams::production_sota_oom();
        params.oom_temp_dir = dir.join("oom-artifacts");
        params.c_min = 8;
        params.c_max = 16;
        params.max_leaders = 8;
        params.psamp_fraction = 1.0;
        params.fanout_top = 2;
        params.fanout_second = 2;
        params.view_lune_prune_enable = true;
        params.view_lune_candidate_width_multiplier = 2.0;
        params.view_lune_max_witnesses_per_victim = 4;

        build_forgeann_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            spill_index_dir.to_str().unwrap(),
            1,
            true,
            &params,
        )
        .unwrap();

        let mut memory_params = params.clone();
        memory_params.oom_temp_dir = dir.join("oom-artifacts-memory");
        memory_params.view_lune_oracle_json = Some(oracle_path);
        build_forgeann_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            memory_index_dir.to_str().unwrap(),
            1,
            true,
            &memory_params,
        )
        .unwrap();

        let spill_graph = std::fs::read(spill_index_dir.join("_mem.index")).unwrap();
        let memory_graph = std::fs::read(memory_index_dir.join("_mem.index")).unwrap();
        assert_eq!(spill_graph, memory_graph);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn production_oom_build_view_lune_oracle_only_preserves_loadable_graph() {
        let dir = unique_temp_dir("forgeann-production-view-lune-oracle-build");
        let data_path = dir.join("tiny.fbin");
        let index_dir = dir.join("index");
        let oracle_path = dir.join("view_lune_oracle.json");
        std::fs::create_dir_all(&index_dir).unwrap();
        write_test_fbin(&data_path, 48, 8);

        let mut params = super::ForgeANNParams::production_sota_oom();
        params.oom_temp_dir = dir.join("oom-artifacts");
        params.c_min = 8;
        params.c_max = 16;
        params.max_leaders = 8;
        params.psamp_fraction = 1.0;
        params.fanout_top = 2;
        params.fanout_second = 2;
        params.view_lune_oracle_json = Some(oracle_path.clone());

        build_forgeann_index(
            Metric::L2,
            data_path.to_str().unwrap(),
            8,
            index_dir.to_str().unwrap(),
            1,
            true,
            &params,
        )
        .unwrap();

        let graph_path = index_dir.join("_mem.index");
        assert!(graph_path.exists());
        assert!(oracle_path.exists());
        load_mem_graph(&graph_path, 48)
            .expect("ViewLune oracle-only OOM build should emit a loadable _mem.index");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
