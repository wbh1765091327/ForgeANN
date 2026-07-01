use std::fs;
use std::path::{Path, PathBuf};

use forgeann::common::Metric;
use forgeann::forgeann::ForgeANNParams;
use forgeann::index::{ForgeAnnGraphBuildConfig, build_forgeann_graph_oom_to_file, load_mem_graph};
use forgeann::utils::{load_metadata_from_file, save_bin_f32};

fn unique_temp_dir(label: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.push(format!("forgeann-{label}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_test_fbin(path: &Path, rows: usize, dim: usize) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let data: Vec<f32> = (0..rows)
        .flat_map(|row| (0..dim).map(move |col| row as f32 * 0.25 + col as f32 * 0.5))
        .collect();
    save_bin_f32(path, &data, rows, dim, 0).unwrap();
}

fn test_config(data_path: &Path, num_threads: u32) -> ForgeAnnGraphBuildConfig {
    let (data_num, data_dim) = load_metadata_from_file(data_path).unwrap();
    ForgeAnnGraphBuildConfig::new(
        Metric::L2,
        data_dim,
        data_dim.div_ceil(32) * 32,
        data_num,
        16,
        num_threads,
    )
    .unwrap()
}

#[test]
fn forgeann_oom_to_file_production_round_trips_with_final_robust_prune() {
    let dir = unique_temp_dir("forgeann-oom-file");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    write_test_fbin(&data_path, 48, 8);

    let config = test_config(&data_path, 1);
    let mut params = ForgeANNParams::default();
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_forgeann_oom");
    params.c_min = 2;
    params.c_max = 8;
    params.max_depth = 6;
    params.max_leaders = 32;
    params.fanout_top = 4;
    params.fanout_second = 2;

    build_forgeann_graph_oom_to_file(&data_path, &config, &params, &graph_path).unwrap();

    assert!(graph_path.exists(), "expected _mem.index output");

    let loaded = load_mem_graph(&graph_path, 48).unwrap();
    assert_eq!(loaded.neighbors.len(), 48);
    for (vid, neighbors) in loaded.neighbors.iter().enumerate() {
        assert!(
            !neighbors.is_empty(),
            "expected node {vid} to have neighbors in production OOM graph"
        );
        assert!(
            neighbors.len() <= 16,
            "expected node {vid} to respect write_range cap"
        );
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn forgeann_oom_to_file_allows_strict_io_with_final_robust_prune() {
    let dir = unique_temp_dir("forgeann-oom-file-strict");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    write_test_fbin(&data_path, 40, 8);

    let config = test_config(&data_path, 1);
    let mut params = ForgeANNParams::default();
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_temp_dir = dir.join("_forgeann_oom");
    params.c_min = 2;
    params.c_max = 8;
    params.max_depth = 6;
    params.max_leaders = 24;
    params.fanout_top = 4;
    params.fanout_second = 2;

    build_forgeann_graph_oom_to_file(&data_path, &config, &params, &graph_path).unwrap();

    assert!(graph_path.exists(), "expected _mem.index output");

    let loaded = load_mem_graph(&graph_path, 40).unwrap();
    assert_eq!(loaded.neighbors.len(), 40);
    for neighbors in &loaded.neighbors {
        assert!(!neighbors.is_empty());
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn forgeann_oom_to_file_records_direct_point_store_profile() {
    let dir = unique_temp_dir("forgeann-oom-file-profile");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    let profile_path = dir.join("oom_profile.json");
    write_test_fbin(&data_path, 48, 8);

    let config = test_config(&data_path, 1);
    let mut params = ForgeANNParams::default();
    params.oom_enable = true;
    params.oom_keep_artifacts = false;
    params.oom_profile_json = Some(profile_path.clone());
    params.oom_temp_dir = dir.join("_forgeann_oom");
    params.c_min = 2;
    params.c_max = 8;
    params.max_depth = 6;
    params.max_leaders = 32;
    params.fanout_top = 4;
    params.fanout_second = 2;

    build_forgeann_graph_oom_to_file(&data_path, &config, &params, &graph_path).unwrap();

    assert!(graph_path.exists(), "expected _mem.index output");
    let profile_text = fs::read_to_string(&profile_path).unwrap();
    let profile_json: serde_json::Value = serde_json::from_str(&profile_text).unwrap();
    assert_eq!(profile_json["point_store_backend"], "direct");
    assert!(profile_json.get("resident_subtree").is_none());

    let loaded = load_mem_graph(&graph_path, 48).unwrap();
    assert_eq!(loaded.neighbors.len(), 48);
    for (vid, neighbors) in loaded.neighbors.iter().enumerate() {
        assert!(
            !neighbors.is_empty(),
            "expected node {vid} to have neighbors in production OOM graph"
        );
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn forgeann_oom_to_file_allows_final_robust_prune_with_strict_io_flag() {
    let dir = unique_temp_dir("forgeann-oom-file-strict-final-robust-prune");
    let data_path = dir.join("tiny.fbin");
    let graph_path = dir.join("_mem.index");
    write_test_fbin(&data_path, 24, 8);

    let config = test_config(&data_path, 1);
    let mut params = ForgeANNParams::default();
    params.oom_enable = true;
    params.oom_temp_dir = dir.join("_forgeann_oom");

    build_forgeann_graph_oom_to_file(&data_path, &config, &params, &graph_path).unwrap();

    assert!(graph_path.exists(), "expected _mem.index output");
    let loaded = load_mem_graph(&graph_path, 24).unwrap();
    assert_eq!(loaded.neighbors.len(), 24);

    fs::remove_dir_all(&dir).unwrap();
}
