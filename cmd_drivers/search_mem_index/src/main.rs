use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read as _};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::Parser;
use forgeann::utils::file_util::load_metadata_from_file;
use memmap2::{Advice, Mmap, MmapOptions};

/// Search a _mem.index (Vamana/ForgeANN adjacency-list format) in pure memory mode.
#[derive(Parser)]
struct Args {
    /// Path prefix where _mem.index lives
    #[clap(long)]
    index_path_prefix: PathBuf,

    /// Path to base data .fbin file (raw f32 vectors)
    #[clap(long)]
    data_path: PathBuf,

    /// Query vectors in .fbin format
    #[clap(long)]
    query_file: PathBuf,

    /// Result output .fbin
    #[clap(long)]
    result_path: PathBuf,

    /// Ground truth file
    #[clap(long)]
    gt_file: PathBuf,

    /// Recall@K
    #[clap(long, default_value_t = 10)]
    recall_at: usize,

    /// Search list size (L)
    #[clap(long, default_value_t = 100)]
    search_list_size: usize,

    /// Comma-separated search list sizes to run after loading the graph once
    #[clap(long, value_delimiter = ',')]
    search_list_sizes: Vec<usize>,

    /// Number of threads
    #[clap(long, short = 'T', default_value_t = 1)]
    num_threads: usize,

    /// Search only the first N queries from the query file
    #[clap(long)]
    max_queries: Option<usize>,

    /// Optional DiskANN medoids file used to select a query-specific entry point
    #[clap(long)]
    entry_medoids_file: Option<PathBuf>,

    /// Optional DiskANN centroids fbin used with --entry-medoids-file
    #[clap(long)]
    entry_centroids_file: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // 1. Load base data
    let data = MmapF32Data::open(&args.data_path)?;
    let num_points = data.len();
    let dim = data.dim();
    eprintln!("Base data: {num_points} points, dim={dim}");

    // 2. Load graph
    let graph_path = args.index_path_prefix.join("_mem.index");
    let (start_node, graph) = load_graph(&graph_path, num_points)?;
    eprintln!(
        "Graph: {} nodes, start={}, max_degree={}",
        graph.len(),
        start_node,
        graph.iter().map(|v| v.len()).max().unwrap_or(0)
    );

    // 3. Load queries
    let (num_queries, query_dim) = load_metadata_from_file(&args.query_file)?;
    assert_eq!(query_dim, dim, "query dim {query_dim} != data dim {dim}");
    let queries = load_fbin_f32(&args.query_file)?;
    let query_count = effective_query_count(num_queries, args.max_queries);
    if query_count == 0 {
        anyhow::bail!("max-queries selected zero queries");
    }
    eprintln!("Queries: {query_count} selected from {num_queries}");

    let query_starts = match (&args.entry_medoids_file, &args.entry_centroids_file) {
        (Some(medoids_file), Some(centroids_file)) => {
            let selector = EntrySelector::open(medoids_file, centroids_file, dim)?;
            let starts = (0..query_count)
                .map(|qi| {
                    let query = &queries[qi * dim..(qi + 1) * dim];
                    selector.select(query)
                })
                .collect::<Vec<_>>();
            eprintln!(
                "Entry selector: {} medoids from {:?}",
                selector.medoids.len(),
                medoids_file
            );
            starts
        }
        (None, None) => vec![start_node; query_count],
        _ => {
            anyhow::bail!("--entry-medoids-file and --entry-centroids-file must be passed together")
        }
    };

    // 4. Search
    let k = args.recall_at;
    let search_list_sizes = if args.search_list_sizes.is_empty() {
        vec![args.search_list_size]
    } else {
        args.search_list_sizes.clone()
    };
    if search_list_sizes.iter().any(|&l| l == 0) {
        anyhow::bail!("search list size must be positive");
    }

    let gt = load_gt(&args.gt_file, query_count, args.recall_at)?;
    let multiple_l = search_list_sizes.len() > 1;
    for &l in &search_list_sizes {
        let t0 = Instant::now();
        let (all_results, query_latencies) = if args.num_threads <= 1 {
            let mut query_latencies = Vec::with_capacity(query_count);
            let mut visited = VisitTracker::new(graph.len());
            let all_results = (0..query_count)
                .map(|qi| {
                    let query = &queries[qi * dim..(qi + 1) * dim];
                    let t_query = Instant::now();
                    let result =
                        search_one(query, &data, &graph, query_starts[qi], k, l, &mut visited);
                    query_latencies.push(t_query.elapsed());
                    result
                })
                .collect::<Vec<_>>();
            (all_results, query_latencies)
        } else {
            parallel_search(
                &queries,
                &data,
                dim,
                &graph,
                &query_starts,
                k,
                l,
                query_count,
                args.num_threads,
            )
        };
        let elapsed = t0.elapsed();
        eprintln!(
            "Search L={l}: {query_count} queries in {:.2}s ({:.1} QPS)",
            elapsed.as_secs_f64(),
            query_count as f64 / elapsed.as_secs_f64()
        );
        if let Some(first_query) = query_latencies.first() {
            let mut sorted = query_latencies.clone();
            sorted.sort_unstable();
            let p50 = sorted[sorted.len() / 2];
            let p99 = sorted[((sorted.len() as f64 * 0.99).round() as usize).min(sorted.len() - 1)];
            eprintln!(
                "Latency L={l}: first_query={:.3}ms p50={:.3}ms p99={:.3}ms",
                duration_ms(*first_query),
                duration_ms(p50),
                duration_ms(p99)
            );
        }

        let recall = compute_recall(&all_results, &gt, args.recall_at);
        eprintln!("Recall@{} L={}: {:.4}", args.recall_at, l, recall);

        let result_path = result_path_for_l(&args.result_path, l, multiple_l);
        save_result_fbin(&result_path, &all_results, k)?;
        eprintln!("Results saved to {:?}", result_path);
    }

    Ok(())
}

fn parallel_search(
    queries: &[f32],
    data: &MmapF32Data,
    dim: usize,
    graph: &[Vec<u32>],
    starts: &[u32],
    k: usize,
    l: usize,
    num_queries: usize,
    num_threads: usize,
) -> (Vec<Vec<u32>>, Vec<Duration>) {
    let results: std::sync::Mutex<Vec<Option<(Vec<u32>, Duration)>>> =
        std::sync::Mutex::new(vec![None; num_queries]);

    std::thread::scope(|s| {
        let chunk = (num_queries + num_threads - 1) / num_threads;
        for tid in 0..num_threads {
            let lo = tid * chunk;
            let hi = (lo + chunk).min(num_queries);
            if lo >= hi {
                continue;
            }
            let results = &results;
            s.spawn(move || {
                let mut visited = VisitTracker::new(graph.len());
                for qi in lo..hi {
                    let query = &queries[qi * dim..(qi + 1) * dim];
                    let t_query = Instant::now();
                    let result = search_one(query, data, graph, starts[qi], k, l, &mut visited);
                    results.lock().unwrap()[qi] = Some((result, t_query.elapsed()));
                }
            });
        }
    });

    let pairs = results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|o| o.unwrap())
        .collect::<Vec<_>>();
    let mut all_results = Vec::with_capacity(pairs.len());
    let mut query_latencies = Vec::with_capacity(pairs.len());
    for (result, latency) in pairs {
        all_results.push(result);
        query_latencies.push(latency);
    }
    (all_results, query_latencies)
}

fn effective_query_count(total_queries: usize, max_queries: Option<usize>) -> usize {
    max_queries.map_or(total_queries, |limit| limit.min(total_queries))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn compute_recall(all_results: &[Vec<u32>], gt: &[u32], recall_at: usize) -> f64 {
    let mut total_recall = 0.0f64;
    for qi in 0..all_results.len() {
        let gt_set: HashSet<u32> = gt[qi * recall_at..(qi + 1) * recall_at]
            .iter()
            .copied()
            .collect();
        let hits = all_results[qi]
            .iter()
            .filter(|&&id| gt_set.contains(&id))
            .count();
        total_recall += hits as f64 / recall_at as f64;
    }
    total_recall / all_results.len() as f64
}

fn result_path_for_l(base: &PathBuf, l: usize, multiple_l: bool) -> PathBuf {
    if !multiple_l {
        return base.clone();
    }
    let parent = base.parent();
    let stem = base
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| "results".into());
    let extension = base
        .extension()
        .map(|s| format!(".{}", s.to_string_lossy()))
        .unwrap_or_default();
    let file_name = format!("{stem}.L{l}{extension}");
    match parent {
        Some(parent) => parent.join(file_name),
        None => PathBuf::from(file_name),
    }
}

fn load_fbin_f32(path: &PathBuf) -> anyhow::Result<Vec<f32>> {
    let (n, d) = load_metadata_from_file(path)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.read_u32::<LittleEndian>()?; // n
    reader.read_u32::<LittleEndian>()?; // d
    let mut data = vec![0.0f32; n * d];
    reader.read_exact(bytemuck::cast_slice_mut(&mut data))?;
    Ok(data)
}

struct EntrySelector {
    medoids: Vec<u32>,
    centroids: Vec<f32>,
    dim: usize,
}

impl EntrySelector {
    fn open(medoids_file: &PathBuf, centroids_file: &PathBuf, dim: usize) -> anyhow::Result<Self> {
        let (num_medoids, medoid_cols) = load_metadata_from_file(medoids_file)?;
        if medoid_cols != 1 {
            anyhow::bail!(
                "medoids file {:?} has cols={medoid_cols}, expected 1",
                medoids_file
            );
        }
        let mut reader = BufReader::new(File::open(medoids_file)?);
        reader.read_u32::<LittleEndian>()?;
        reader.read_u32::<LittleEndian>()?;
        let mut medoids = vec![0u32; num_medoids];
        for medoid in &mut medoids {
            *medoid = reader.read_u32::<LittleEndian>()?;
        }

        let (num_centroids, centroid_dim) = load_metadata_from_file(centroids_file)?;
        if centroid_dim != dim {
            anyhow::bail!(
                "centroids file {:?} has dim={centroid_dim}, expected query dim {dim}",
                centroids_file
            );
        }
        if num_centroids != medoids.len() {
            anyhow::bail!(
                "centroids count {} does not match medoids count {}",
                num_centroids,
                medoids.len()
            );
        }
        let centroids = load_fbin_f32(centroids_file)?;
        Ok(Self {
            medoids,
            centroids,
            dim,
        })
    }

    fn select(&self, query: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut best_dist = f32::INFINITY;
        for centroid_id in 0..self.medoids.len() {
            let centroid = &self.centroids[centroid_id * self.dim..(centroid_id + 1) * self.dim];
            let dist = l2_distance(query, centroid);
            if dist < best_dist {
                best_dist = dist;
                best = centroid_id;
            }
        }
        self.medoids[best]
    }
}

struct MmapF32Data {
    _file: File,
    mmap: Mmap,
    rows: usize,
    dim: usize,
}

impl MmapF32Data {
    fn open(path: &PathBuf) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        if mmap.len() < 8 {
            anyhow::bail!("fbin file {:?} is too small", path);
        }
        let rows = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let expected_len = 8usize
            .checked_add(
                rows.checked_mul(dim)
                    .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| anyhow::anyhow!("fbin dimensions overflow for {:?}", path))?,
            )
            .ok_or_else(|| anyhow::anyhow!("fbin byte length overflow for {:?}", path))?;
        if mmap.len() < expected_len {
            anyhow::bail!(
                "fbin file {:?} is truncated: got {} bytes, expected at least {}",
                path,
                mmap.len(),
                expected_len
            );
        }
        let _ = mmap.advise(Advice::Random);
        Ok(Self {
            _file: file,
            mmap,
            rows,
            dim,
        })
    }

    fn len(&self) -> usize {
        self.rows
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn row(&self, row: usize) -> &[f32] {
        let start = 8 + row * self.dim * std::mem::size_of::<f32>();
        let end = start + self.dim * std::mem::size_of::<f32>();
        bytemuck::cast_slice(&self.mmap[start..end])
    }
}

fn load_graph(path: &PathBuf, expected_num_points: usize) -> anyhow::Result<(u32, Vec<Vec<u32>>)> {
    let mut reader = BufReader::new(File::open(path)?);
    let _total_size = reader.read_u64::<LittleEndian>()?;
    let _max_degree = reader.read_u32::<LittleEndian>()?;
    let start = reader.read_u32::<LittleEndian>()?;
    let _frozen_pts = reader.read_u64::<LittleEndian>()?;

    let mut graph = Vec::with_capacity(expected_num_points);
    for _ in 0..expected_num_points {
        let degree = reader.read_u32::<LittleEndian>()?;
        let mut neighbors = vec![0u32; degree as usize];
        for nbr in &mut neighbors {
            *nbr = reader.read_u32::<LittleEndian>()?;
        }
        graph.push(neighbors);
    }
    Ok((start, graph))
}

fn load_gt(path: &PathBuf, num_queries: usize, recall_at: usize) -> anyhow::Result<Vec<u32>> {
    let (n, d) = load_metadata_from_file(path)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.read_u32::<LittleEndian>()?; // n
    reader.read_u32::<LittleEndian>()?; // d
    let total = n * d;
    let mut all = vec![0u32; total];
    for id in &mut all {
        *id = reader.read_u32::<LittleEndian>()?;
    }
    let nq = num_queries.min(n);
    let mut gt = vec![0u32; nq * recall_at];
    for qi in 0..nq {
        for ki in 0..recall_at.min(d) {
            gt[qi * recall_at + ki] = all[qi * d + ki];
        }
    }
    Ok(gt)
}

struct VisitTracker {
    marks: Vec<u16>,
    generation: u16,
}

impl VisitTracker {
    fn new(size: usize) -> Self {
        Self {
            marks: vec![0; size],
            generation: 0,
        }
    }

    fn reset(&mut self) {
        if self.generation == u16::MAX {
            self.marks.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    fn mark(&mut self, idx: usize) {
        self.marks[idx] = self.generation;
    }

    fn mark_if_new(&mut self, idx: usize) -> bool {
        if self.marks[idx] == self.generation {
            false
        } else {
            self.marks[idx] = self.generation;
            true
        }
    }
}

/// Greedy BFS search on the graph (standard Vamana/DiskANN algorithm).
fn search_one(
    query: &[f32],
    data: &MmapF32Data,
    graph: &[Vec<u32>],
    start: u32,
    k: usize,
    l: usize,
    visited: &mut VisitTracker,
) -> Vec<u32> {
    let l_eff = l.max(k + 1);
    // Max-heap of (-dist, id) => gives us the farthest element on top
    // We use (Reverse(dist), id) in a max-heap => smallest -dist (i.e. farthest) on top
    // Actually: BinaryHeap is max-heap. With Reverse<OrderedFloat>, smaller floats pop first.
    // We want to pop the farthest when trimming, so use Reverse for the candidates.

    visited.reset();
    visited.mark(start as usize);

    let d0 = l2_distance(query, data.row(start as usize));

    // Candidates: min-heap by distance (closest first to expand)
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::with_capacity(l_eff);
    candidates.push(Reverse((OrderedF32(d0), start)));

    // Best results: max-heap by distance (farthest on top, for trimming)
    let mut best: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(l_eff);
    best.push((OrderedF32(d0), start));

    while let Some(Reverse((cd, cid))) = candidates.pop() {
        // Prune: if closest candidate is farther than L-th best, stop
        if best.len() >= l_eff {
            if let Some(&(OrderedF32(worst), _)) = best.peek() {
                if cd.0 > worst {
                    break;
                }
            }
        }

        let neighbors = &graph[cid as usize];
        for &nbr in neighbors {
            let ni = nbr as usize;
            if ni >= graph.len() || !visited.mark_if_new(ni) {
                continue;
            }
            let dist = l2_distance(query, data.row(ni));

            candidates.push(Reverse((OrderedF32(dist), nbr)));
            best.push((OrderedF32(dist), nbr));

            while best.len() > l_eff {
                best.pop();
            }
        }
    }

    // Sort by distance, take top-k
    let mut sorted: Vec<(OrderedF32, u32)> = best.into_iter().collect();
    sorted.sort_by(|a, b| {
        a.0.0
            .partial_cmp(&b.0.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.into_iter().take(k).map(|(_, id)| id).collect()
}

#[derive(Clone, Copy, Debug)]
struct OrderedF32(f32);

impl PartialEq for OrderedF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

fn save_result_fbin(path: &PathBuf, results: &[Vec<u32>], k: usize) -> anyhow::Result<()> {
    let n = results.len() as u32;
    let mut writer = std::io::BufWriter::new(File::create(path)?);
    writer.write_u32::<LittleEndian>(n)?;
    writer.write_u32::<LittleEndian>(k as u32)?;
    for result in results {
        for i in 0..k {
            writer.write_u32::<LittleEndian>(result.get(i).copied().unwrap_or(0))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::effective_query_count;

    #[test]
    fn max_queries_caps_available_queries() {
        assert_eq!(effective_query_count(10, None), 10);
        assert_eq!(effective_query_count(10, Some(50)), 10);
        assert_eq!(effective_query_count(10, Some(5)), 5);
        assert_eq!(effective_query_count(10, Some(0)), 0);
    }
}
