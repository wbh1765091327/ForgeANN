# PiPNN 实现与优化说明

本文档从工程实现角度，系统说明当前 ForgeANN 中 `PiPNN` 的具体实现方式、执行流程、调度器设计、参数推荐逻辑，以及已经落地并验证过的优化。

文档以当前代码为准，主要对应以下模块：

- `ForgeANN/src/forgeann/mod.rs`
- `ForgeANN/src/forgeann/params.rs`
- `ForgeANN/src/forgeann/rbc_partition/`
- `ForgeANN/src/forgeann/scheduler.rs`
- `ForgeANN/src/forgeann/leaf_build.rs`
- `cmd_drivers/build_forgeann_index/src/main.rs`

## 1. 总体目标

PiPNN 的目标不是直接像 Vamana/HNSW 那样做全局 beam-search 式构图，而是采用“两阶段”思路：

1. 先用 RBC 递归分区，把全体点切成许多中小叶子。
2. 再在每个叶子内做精确或准精确的局部 kNN，并通过 HashPrune reservoir 汇总成全局邻接表。

在当前实现里，它的核心设计点是：

- RBC 分区和叶子构建并行执行，而不是先全量分区再统一构图。
- 分区和叶构建共用一个 Rayon 线程池，`--num-threads` 是整个 PiPNN 构建阶段的硬上限。
- 对 L2 距离优先走 GEMM 路径，把大规模距离计算转成矩阵乘，尽可能让 CPU 做高吞吐计算。

这套实现可以概括成：

`compute sketches -> init reservoirs -> RBC streaming partition -> leaf scheduling/build -> materialize graph`

## 2. 顶层构图流程

顶层入口是 `ForgeANN/src/forgeann/mod.rs` 中的 `build_forgeann_graph`。

当前执行顺序如下：

1. 校验参数和线程数。
2. 预计算所有点的 sketch。
3. 为每个点初始化一个 `HashPruneReservoir`。
4. 创建单个受限 Rayon 线程池。
5. 在这个统一线程池里运行：
   - producer：RBC 递归分区
   - consumers/drainers：叶子构建
6. RBC 和叶子全部完成后，把 reservoir 转成最终邻接表并回调写入 index。

关键点：

- sketch 只计算一次，保存在 `SketchStore` 中。
- reservoir 也是一次性初始化，后续叶子任务只做插入。
- `BlasThreadEnv::set_single_threaded()` 会在进入 PiPNN 主阶段前把底层 BLAS/OpenMP 钉成单线程，避免 GEMM 内部再开线程造成 oversubscription。
- `run_scheduler_scope` 使用同一个 Rayon pool 同时承载 producer 和 leaf workers，因此 `num_threads` 不是“RBC 线程数”，而是整个 PiPNN 阶段的总线程预算。

这里直接看一段顶层 orchestrate 代码会更直观：

```rust
let sketches = compute_sketches(dataset, num_points, params, num_threads)?;

let mut reservoirs: Vec<Mutex<HashPruneReservoir>> = Vec::with_capacity(num_points);
for _ in 0..num_points {
    reservoirs.push(Mutex::new(HashPruneReservoir::new(params.l_max)));
}

let scheduler_budget =
    SchedulerBudget::for_worker_count(effective_threads, params.kernel_safe_leaf_size());

BlasThreadEnv::set_single_threaded();

let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(effective_threads)
    .build()
    .map_err(|e| AnnError::log_index_error(format!("Failed to create Rayon pool: {}", e)))?;

let (scheduler_profile, scheduler_telemetry) = run_scheduler_scope(
    &pool,
    scheduler_budget,
    leaf_context,
    &pb,
    |emitter| {
        let mut rng = StdRng::seed_from_u64(params.random_seed);
        rbc_partition_streaming(
            dataset,
            &all_indices,
            metric,
            params,
            effective_threads as u32,
            &mut rng,
            emitter,
        )
    },
)?;
```

这段代码把 PiPNN 的执行骨架说得非常清楚：

- sketch 先一次性算完
- reservoir 先一次性分配完
- 统一创建一个 Rayon pool
- RBC 不返回“所有叶子列表”，而是通过 `emitter` 一边递归一边把叶子流给 scheduler
- leaf build 和 producer 全程共用这一个线程池

## 3. CLI 与参数映射

命令行入口是 `cmd_drivers/build_forgeann_index/src/main.rs` 中的 `build_forgeann_index`。

主要参数与 `ForgeANNParams` 的映射关系：

- `-R/--max-degree` -> `l_max`
- `--m-hash-bits` -> `m_hash_bits`
- `--leaf-size-min` -> `c_min`
- `--leaf-size-max` -> `c_max`
- `--psamp-fraction` -> `psamp_fraction`
- `--fanout-top` -> `fanout_top`
- `--fanout-second` -> `fanout_second`
- `--leaf-knn` -> `leaf_knn`
- `--max-leaders` -> `max_leaders`
- `--final-prune` -> `final_prune`
- `--final-prune-alpha` -> `final_prune_alpha`
- `--seed` -> `random_seed`
- `--dist-fn` -> `metric`

如果传了 `--auto-params`，则先调用 `ForgeANN/src/forgeann/params.rs` 中的 `ForgeANNParams::recommend` 自动生成一组参数，再覆盖用户显式指定的非推荐项，如：

- `m_hash_bits`
- `final_prune`
- `final_prune_alpha`
- `random_seed`
- `metric`

Wiki-scale production runs use `cmd_drivers/build_forgeann_index/src/main.rs`. Current defaults are:

- `AUTO_PARAMS=1`
- `R=64`
- `Lbuild=100`
- `T=42`
- `m_hash_bits=12`
- `seed=42`

脚本还会在前台运行时通过 `tee` 同步写日志，便于后续提取 profiling 结果。

## 4. 参数推荐逻辑

参数推荐逻辑在 `ForgeANN/src/forgeann/params.rs` 的 `ForgeANNParams::recommend`。

核心思想不是拍脑袋，而是先估一个“PiPNN 可以承受的总距离计算预算”，再反推各参数。

### 4.1 预算模型

代码里把 Vamana 的等效距离计算量近似为：

`D_vamana = n * L * R`

然后假设 GEMM 路径对 beam-search 式逐点计算有一个保守的 `10x` 效率优势：

`D_budget = D_vamana * 10`

再把预算按 40/60 分给：

- RBC 分区：40%
- 叶内构图：60%

### 4.2 自动推荐出的关键参数

自动推荐会反推：

- `c_max`
- `max_leaders`
- `psamp_fraction`
- `fanout_top`
- `fanout_second`
- `leaf_knn`

对 wiki35m 这类超大规模数据，当前推荐逻辑有两个重要特征：

1. `fanout_top` 固定为 `10`
2. `fanout_second` 对 `n >= 20M` 会优先选 `2`，即 `10x2`

这不是口头描述，而是代码里明确这样写的：

```rust
let fanout_top = 10_usize;
let fanout_second = if n >= 20_000_000 { 2_usize } else { 3_usize };
let f_total = (fanout_top * fanout_second) as f64;

let adaptive_multiplier = if n >= 5_000_000 { 2.0 } else { 1.0 };
let c_max_raw = 2.0 * d_leaf_budget / (n as f64 * f_total * adaptive_multiplier);
let mut c_max = (c_max_raw as usize).clamp(512, 2048);
if n >= 20_000_000 && fanout_second == 2 {
    // Keep wiki35m-class 10x2 runs aligned with the historical f10s2 baseline
    // so fanout changes do not silently inflate leaf size from 1280 -> 1920.
    c_max = c_max.min(1280);
}
```

同时，为了避免 `10x2` 自动推荐把叶子隐式放大得过头，代码还专门限制了 wiki35m 级别数据集上的 `c_max` 上限，使其与历史 `10x2` 基线对齐。

完整的预算反推部分也值得直接看：

```rust
let l_vamana = r.max(100) as f64;
let d_vamana = n as f64 * l_vamana * r as f64;

let gemm_advantage = 10.0_f64;
let d_budget = d_vamana * gemm_advantage;

let d_rbc_budget = d_budget * 0.4;
let d_leaf_budget = d_budget * 0.6;

let max_leaders_raw = d_rbc_budget / (n as f64 * 2.0);
let max_leaders = (max_leaders_raw as usize).clamp(500, 10000);

let psamp = if n >= 20_000_000 {
    max_leaders as f64 / (n as f64 * 0.05)
} else if n >= 10_000_000 {
    max_leaders as f64 / (n as f64 * 0.1)
} else if n >= 5_000_000 {
    max_leaders as f64 / (n as f64 * 0.2)
} else if n >= 1_000_000 {
    max_leaders as f64 / (n as f64 * 0.4)
} else {
    max_leaders as f64 / (n as f64 * 0.7)
};
let psamp = psamp.clamp(0.001, 0.1);
```

### 4.3 自适应参数

除了静态推荐值，运行时还有 3 个重要自适应参数：

- `adaptive_psamp_fraction`
- `adaptive_c_max`
- `adaptive_fanout`

关键代码如下：

```rust
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

pub fn adaptive_c_max(&self, total_size: usize) -> usize {
    if total_size >= 20_000_000 {
        self.c_max.saturating_mul(2)
    } else if total_size >= 10_000_000 {
        self.c_max.saturating_mul(2)
    } else if total_size >= 5_000_000 {
        self.c_max.saturating_mul(2)
    } else {
        self.c_max
    }
}

pub fn adaptive_fanout(&self, _cluster_size: usize, depth: usize) -> usize {
    self.fanout_for_depth(depth)
}
```

当前规则：

- 大数据集会降低 `psamp_fraction`，避免顶层 leaders 爆炸。
- 大数据集会把 `c_max` 放大一倍，减少顶层分区压力。
- `fanout` 仅依赖深度：`depth=0` 用 `fanout_top`，`depth=1` 用 `fanout_second`，其余层固定为 `1`。

## 5. RBC 分区实现

RBC 分区入口是 `ForgeANN/src/forgeann/rbc_partition/` 中的 `rbc_partition_streaming`。

它的职责不是一次性返回所有叶子，而是边递归边把叶子通过 `LeafEmitter` 发给调度器。

### 5.1 基本流程

RBC 的整体流程如下：

1. 以全量点集作为根节点进入递归。
2. 对当前簇做若干早停判断：
   - `max_depth`
   - `min_shrink_ratio`
   - `n <= adaptive_c_max`
   - `depth >= 3 && n <= 2 * adaptive_c_max`
   - `depth >= 5 && n < 3 * adaptive_c_max`
3. 如果不早停，则按当前 `psamp_fraction` 和 `max_leaders` 采样 leaders。
4. 对每个点计算与 leaders 的距离并归到 1 个或多个 cluster。
5. 做 cluster merge。
6. 必要时对 merged clusters 做 dedup。
7. 小 cluster 直接发叶子，大 cluster 继续递归。

对应核心逻辑在 `ForgeANN/src/forgeann/rbc_partition/` 中的 `rbc_recurse_parallel`。

### 5.2 递归并行策略

当前版本不再用一个无界线程爆炸式的递归模型，而是：

- 每次分裂后把待递归的大簇包装成 `SeededCluster`
- 用 `ForgeANN/src/forgeann/rbc_partition/` 中的 `parallel_join_clusters` 递归地二分
- 每层只通过 `rayon::join` 把左右两半并行

这样设计的目的有两个：

1. 保持线程数由外层 Rayon pool 严格控制
2. 避免某些不平衡 cluster 序列导致的极端递归形状和线程风暴

另外，拆分左右子树时不是简单按 cluster 数一刀切，而是通过 `split_seeded_clusters_balanced` 按总点数尽量均衡划分，让左右子树工作量更接近。

### 5.3 Cluster assignment

RBC 最重的步骤是 cluster assignment，即“当前簇内的每个点应该落到哪些 leaders 下”。

当前实现分两条路径：

- `metric == L2 && n >= 256`：走 GEMM 路径
- 其他情况：走标量距离路径

GEMM 路径入口是 `ForgeANN/src/forgeann/rbc_partition/` 中的 `compute_clusters_gemm`。

它的实现方式是：

1. 先把当前所有 leaders 拉成一个 `l_mat`
2. 再把当前点集按 `point_tile_size` 分块
3. 每个 point tile 再按 `leader_tile_size` 流式扫 leader
4. 用 `general_mat_mul` 计算 Gram matrix
5. 把 Gram 转成 L2 距离
6. 更新 top-k 或 best leader
7. 把 block 内结果 materialize 成 cluster buffers

### 5.4 GEMM 分块策略

默认 tile 选择逻辑直接如下：

```rust
fn choose_gemm_tile_sizes(num_leaders: usize, dim: usize) -> (usize, usize) {
    const MAX_POINT_TILE: usize = 8192;
    const MIN_POINT_TILE: usize = 256;
    const MAX_LEADER_TILE: usize = 512;
    const TARGET_BYTES: usize = 64 * 1024 * 1024;

    let num_leaders = num_leaders.max(1);
    let leader_tile = num_leaders.min(MAX_LEADER_TILE);
    let bytes_per_point = dim
        .saturating_mul(size_of::<f32>())
        .saturating_mul(2)
        .saturating_add(leader_tile.saturating_mul(size_of::<f32>()));
    let point_tile = if bytes_per_point == 0 {
        MAX_POINT_TILE
    } else {
        (TARGET_BYTES / bytes_per_point).clamp(MIN_POINT_TILE, MAX_POINT_TILE)
    };

    (point_tile, leader_tile)
}
```

- `point_tile_size` 上限 `8192`
- `leader_tile_size` 上限 `512`
- 目标工作集大小约 `64 MiB`

同时支持通过环境变量 `FORGEANN_GEMM_TILE=POINTxLEADER` 覆盖。

这主要用于受控 A/B 实验，而不是默认长期依赖。

### 5.5 `fanout=1` 特化

当前实现对 `fanout=1` 做了专门 fast path。

原因很简单：

- 当 `fanout=1` 时，不需要维护每行的 top-k 集合
- 只需要维护每个点的“当前最优 leader”

GEMM 路径里真正走的是下面这段逻辑，而不是通用 `StackTopK`：

```rust
if local_fanout == 1 {
    let mut best_distances = vec![f32::INFINITY; block_n];
    let mut best_leaders = vec![0usize; block_n];

    for leader_offset in (0..nl).step_by(leader_tile_size) {
        let leader_end = (leader_offset + leader_tile_size).min(nl);
        let leader_view = l_mat.slice(ndarray::s![leader_offset..leader_end, ..]);
        let tile_cols = leader_end - leader_offset;
        let mut gram_tile = borrow_gram_tile(&mut scratch.gram_data, block_n, tile_cols);
        ndarray::linalg::general_mat_mul(1.0, &x_mat, &leader_view.t(), 0.0, &mut gram_tile);
        update_block_best_from_gram_tile(
            &mut best_distances,
            &mut best_leaders,
            &scratch.x_norms[..block_n],
            &l_norms,
            gram_tile.view(),
            leader_offset,
        );
    }

    scratch.local_clusters.resize_with(nl, Vec::new);
    for v in scratch.local_clusters[..nl].iter_mut() {
        v.clear();
    }
    for (i, &best_lid) in best_leaders.iter().enumerate() {
        scratch.local_clusters[best_lid].push(block[i]);
    }

    merge_cluster_buffers_in_place(&mut state.clusters, &mut scratch.local_clusters);
}
```

在非 GEMM 标量路径里，也有对 `fanout=1` 的专门逻辑，不再创建 `StackTopK`。

这是一项已经落地并被 wiki35m 实测验证过的优化，目的是减少在绝大多数递归层中对 top-k 容器的维护成本。

### 5.6 分区统计与 profiling

RBC 维护了比较完整的 `PartitionStats`，包括：

- 各深度处理簇数和叶子数
- 各种叶子终止原因
- pre/post merge cluster 数
- assignments before/after dedup
- dedup runs / skipped
- 各阶段耗时
- GEMM profile
- oversized leaf 事件统计

这些信息最终由 `log_final_summary` 打印到日志中，用来分析：

- `RBC Duration`
- `Cluster assign`
- `GEMM profile`
- `Oversized leaf events`
- `Leaf size distribution`

## 6. Oversized leaf 保护与 hard cap

当前实现里，RBC 叶子发射不是“无脑直接 emit”。

有一个强制硬上限：

- `FORCED_LEAF_HARD_CAP = 8000`

对应逻辑直接如下：

```rust
if leaf.len() > FORCED_LEAF_HARD_CAP {
    stats.forced_leaf_splits += 1;
    let split_seed = forced_leaf_split_seed(leaf, depth, reason);
    let split_leaves = forced_split_oversized_leaf(
        dataset,
        metric,
        params,
        std::mem::take(leaf),
        depth,
        FORCED_LEAF_HARD_CAP,
        split_seed,
    )?;

    for mut split_leaf in split_leaves {
        emit_leaf_with_safety(
            dataset,
            metric,
            params,
            &mut split_leaf,
            depth,
            reason,
            max_leaf_size,
            stats,
            leaf_emitter,
        )?;
    }
    return Ok(());
}
```

它的目的不是调整正常叶子大小，而是防止极端坏 case：

- 某些大叶子因为递归早停/分区效果不足直接落到 leaf build
- 造成 blockwise 叶子成本过高
- 甚至引起极端内存或长尾 wall-time 问题

当前策略是：

- 正常情况下仍按 `adaptive_c_max` 和递归规则自然结束
- 只有发射叶子前发现叶子过大时，才进入强制分裂保护

在 wiki35m 上，这个 `8k hard cap` 已经证明是有效的：

- 它把 `10K+` 叶子全部清掉了
- 同时只引入很少的 forced split 次数

## 7. 统一 Scheduler 设计

这部分是当前实现最重要的工程设计之一。

入口是 `ForgeANN/src/forgeann/scheduler.rs` 中的 `run_scheduler_scope`。

### 7.1 设计目标

统一 scheduler 的目标是：

- `--num-threads` 必须是硬上限
- RBC producer 和 leaf build 共用一个线程池
- 避免过去那种 producer、consumer 各自开线程导致的 oversubscription
- 在 producer 活跃时，不让 leaf 构建把所有 worker 抢光
- 在 producer 结束后，叶子阶段又能尽可能把池子用满

### 7.2 关键数据结构

调度器核心结构是 `ForgeANN/src/forgeann/scheduler.rs` 中的 `ForgeannScheduler`。

其中几个最关键的字段：

- `leaf_backlog: ArrayQueue<Vec<u32>>`
- `outstanding_leaf_tasks`
- `inflight_leaf_tasks`
- `active_leaf_drainers`
- `producer_active`
- `leaf_profile`
- `SchedulerTelemetry`

整体上它是一个：

- producer 单向发叶子
- bounded queue 缓冲
- 多个 leaf drainer 消费
- producer 在 backlog 满时主动帮忙 drain

的模式。

对应的数据结构直接看字段最清楚：

```rust
pub struct ForgeannScheduler<'a> {
    budget: SchedulerBudget,
    context: LeafTaskContext<'a>,
    progress: &'a ProgressBar,
    leaf_backlog: ArrayQueue<Vec<u32>>,
    outstanding_leaf_tasks: AtomicUsize,
    inflight_leaf_tasks: AtomicUsize,
    peak_inflight_leaf_tasks: AtomicUsize,
    peak_leaf_backlog: AtomicUsize,
    active_leaf_drainers: AtomicUsize,
    producer_active: AtomicBool,
    inline_leaf_fallbacks: AtomicUsize,
    producer_help_drains: AtomicUsize,
    producer_help_yields: AtomicUsize,
    large_leaf_count: AtomicUsize,
    large_leaf_block_tasks: AtomicUsize,
    first_error: Mutex<Option<String>>,
    leaf_profile: Mutex<LeafProfile>,
}
```

### 7.3 为什么使用 bounded backlog

当前 backlog 容量和 large-leaf 阈值是这样算的：

```rust
pub fn for_worker_count(worker_count: usize, kernel_safe_leaf_size: usize) -> Self {
    let worker_count = worker_count.max(1);
    let leaf_backlog_capacity = worker_count.saturating_mul(8).max(1);
    let large_leaf_min_size = cmp::max(4096, kernel_safe_leaf_size.saturating_mul(2));

    Self {
        worker_count,
        leaf_backlog_capacity,
        large_leaf_min_size,
        leaf_block_rows: Self::DEFAULT_LEAF_BLOCK_ROWS,
    }
}
```

- `leaf_backlog_capacity = worker_count * 8`

这不是为了追求“排队越长越好”，而是为了：

- 给 producer 留一点前推空间
- 但又不能让 leaf 无限排队、吞掉过多内存

所以这是一个明确受控的背压模型。

### 7.4 producer-active 阶段的 worker 分配

producer 活跃期限制 leaf worker 数的关键逻辑如下：

```rust
fn active_producer_leaf_worker_cap(&self) -> usize {
    if self.budget.worker_count <= 1 {
        return 1;
    }

    // Reserve roughly two thirds of the pool for the RBC producer/recursion path while it is
    // still generating work. This keeps early and mid-phase wall time focused on the critical
    // path without statically idling threads when no leaf backlog exists.
    self.budget.worker_count.div_ceil(3).max(1)
}

fn spawned_leaf_worker_limit_for_backlog(&self, backlog: usize) -> usize {
    if backlog == 0 {
        return 0;
    }

    let cap = if self.producer_is_active() {
        self.active_producer_leaf_worker_cap()
    } else {
        self.max_leaf_worker_capacity()
    };

    backlog.min(cap)
}
```

当前策略是：

- 当 producer 还在活跃生成叶子时，只允许大约三分之一线程池容量用于 leaf drainer
- 剩余线程留给 RBC 递归

这个策略的动机是：

- 构建 wall-time 的关键路径在 producer/RBC
- 如果一开始就把太多线程让给 leaf，会拖慢最关键的 RBC 前推速度

当 producer 结束后，调度器会放开限制，允许 leaf 使用几乎全部剩余 worker capacity。

### 7.5 producer helping

当 backlog 满时，当前实现不会单纯依赖 `rayon::yield_now()`。

而是先尝试：

- 由 producer 自己 inline 地从 backlog 中拿一个 leaf 做掉

这条逻辑不是概念，而是 `emit_leaf` 里直接这么做：

```rust
loop {
    match self.scheduler.try_enqueue_leaf(leaf) {
        Ok(()) => {
            self.scheduler.maybe_spawn_leaf_drainer(self.scope);
            return Ok(());
        }
        Err(next_leaf) => {
            leaf = next_leaf;
            self.scheduler.maybe_spawn_leaf_drainer(self.scope);
            if self.scheduler.try_help_from_backlog(true)? {
                continue;
            }

            if rayon::yield_now().is_none() {
                self.scheduler
                    .producer_help_yields
                    .fetch_add(1, Ordering::Relaxed);
                std::thread::yield_now();
            }
        }
    }
}
```

其中 `try_help_from_backlog` 本身非常直接：

```rust
fn try_help_from_backlog(&self, count_drain: bool) -> AnnResult<bool> {
    let Some(leaf) = self.try_take_queued_leaf() else {
        return Ok(false);
    };
    if count_drain {
        self.producer_help_drains.fetch_add(1, Ordering::Relaxed);
    }
    self.process_taken_leaf(leaf)?;
    Ok(true)
}
```

这样做的原因是：

- 仅靠 `yield_now` 可能让当前 worker 执行更多嵌套 Rayon job
- 之前这类路径曾触发过深递归/stack overflow 风险
- inline drain 一个 leaf 能直接释放 backlog 压力，更稳定

日志里的 `producer_help_drains`、`producer_help_yields` 就是这条路径的观测指标。

### 7.6 large leaf 处理

当叶子足够大时，scheduler 会走专门的大叶子并行路径，而不是用 thread-local scratch 的普通单叶流程。

判断和分派逻辑本身如下：

```rust
let mut profile = if leaf.len() >= self.budget.large_leaf_min_size {
    self.large_leaf_count.fetch_add(1, Ordering::Relaxed);
    self.large_leaf_block_tasks
        .fetch_add(leaf.len().div_ceil(self.budget.leaf_block_rows), Ordering::Relaxed);
    process_leaf_parallel_large_profiled(
        self.context.dataset,
        self.context.metric,
        self.context.sketches,
        &leaf,
        self.context.params,
        self.context.reservoirs,
        self.budget.leaf_block_rows,
    )?
} else {
    with_thread_local_leaf_scratch(
        self.context.params.kernel_safe_leaf_size(),
        self.context.dataset.dim,
        self.context.params.leaf_knn,
        |scratch| {
            process_leaf_profiled(
                self.context.dataset,
                self.context.metric,
                self.context.sketches,
                &leaf,
                self.context.params,
                self.context.reservoirs,
                scratch,
            )
        },
    )?
};
```

- `max(4096, kernel_safe_leaf_size * 2)`

这样可以避免大叶子在单 worker 上跑太久，形成 leaf stage 的长尾。

## 8. Leaf 构建实现

叶子构建在 `ForgeANN/src/forgeann/leaf_build.rs`。

### 8.1 输出形式

每个叶子不会直接生成最终邻接表，而是往每个点自己的 `HashPruneReservoir` 中插入候选边。

这意味着叶子之间可以天然并行：

- 不需要合并一个巨大全局 edge list
- 只需要对每个目标点的 reservoir 加锁插入

### 8.2 普通叶子路径

普通叶子由 `ForgeANN/src/forgeann/leaf_build.rs` 中的 `process_leaf_profiled` 完成。

L2 下的实现流程：

1. 把叶子中的向量装到 `scratch.x`
2. 计算每行范数
3. 用 GEMM 算出 `X * X^T`
4. 转成平方距离矩阵
5. 对每行做 top-k
6. 用 `sketch_i` / `sketch_j` 计算 hash
7. 暂存在 `PendingEdge`
8. 批量 flush 到 reservoir

这里的 hash 不是拿来做 partition，而是用于 HashPrune reservoir 中的残差哈希插入。

对应的核心代码可以直接看这一段：

```rust
scratch.dmat.clear();
scratch.dmat.resize(leaf_size * leaf_size, 0.0f32);

if metric == Metric::L2 {
    let dim = dataset.dim;
    if scratch.x.nrows() < leaf_size || scratch.x.ncols() < dim {
        scratch.x = Array2::<f32>::zeros((leaf_size.max(scratch.x.nrows()), dim));
    }

    let mut x_view = scratch.x.slice_mut(ndarray::s![0..leaf_size, 0..dim]);

    for (i_local, &i_global) in leaf.iter().enumerate() {
        let vertex = dataset.get_vertex(i_global)?;
        let vec = vertex.vector();
        let mut row = x_view.row_mut(i_local);
        row.assign(&ndarray::ArrayView1::from(vec));
    }

    let norms: Vec<f32> = x_view.map_axis(Axis(1), |row| row.dot(&row)).to_vec();

    let mut dmat_view =
        ArrayViewMut2::from_shape((leaf_size, leaf_size), &mut scratch.dmat).unwrap();
    ndarray::linalg::general_mat_mul(1.0, &x_view, &x_view.t(), 0.0, &mut dmat_view);

    for i in 0..leaf_size {
        let offset = i * leaf_size;
        let norm_i = norms[i];
        for j in 0..leaf_size {
            let gram_ij = scratch.dmat[offset + j];
            let mut dist2 = norm_i + norms[j] - 2.0 * gram_ij;
            if dist2 < 0.0 {
                dist2 = 0.0;
            }
            scratch.dmat[offset + j] = dist2;
        }
    }
}
```

### 8.3 大叶子路径

当叶子超过 `max_full_matrix_leaf_size` 时，会走 blockwise 路径：

- `ForgeANN/src/forgeann/leaf_build.rs` 中的 `process_leaf_blockwise`

而在 scheduler 判定为 large leaf 时，会走更激进的并行 block 路径：

- `ForgeANN/src/forgeann/leaf_build.rs` 中的 `process_leaf_parallel_large_profiled`

其思路是：

1. 一次把整个叶子的向量矩阵 `x` 装入内存
2. 按 `block_rows` 把行切块
3. 每个 block 与全叶矩阵做 GEMM
4. 每个 block 独立产出 top-k 与 pending edges
5. wave-by-wave 并行执行，避免 block scratch 占用过大

对应还有一个 `choose_parallel_block_wave_width`，用于限制同时并行 block 数。

核心执行代码如下：

```rust
let mut x = Array2::<f32>::zeros((leaf_size, dim));

{
    let mut x_view = x.slice_mut(ndarray::s![0..leaf_size, 0..dim]);
    for (i_local, &i_global) in leaf.iter().enumerate() {
        let vertex = dataset.get_vertex(i_global)?;
        let vec = vertex.vector();
        let mut row = x_view.row_mut(i_local);
        row.assign(&ndarray::ArrayView1::from(vec));
    }
}

let x_view = x.view();
let norms: Vec<f32> = if metric == Metric::L2 {
    x_view.map_axis(Axis(1), |row| row.dot(&row)).to_vec()
} else {
    Vec::new()
};

let block_starts: Vec<usize> = (0..leaf_size).step_by(block_rows).collect();
let wave_width = choose_parallel_block_wave_width(
    leaf_size,
    block_rows,
    params.leaf_knn,
    block_starts.len(),
);

for wave in block_starts.chunks(wave_width) {
    let mut results: Vec<_> = wave
        .par_iter()
        .map(|&block_start| {
            process_leaf_parallel_large_block(
                dataset,
                metric,
                sketches,
                leaf,
                params,
                &x_view,
                &norms,
                block_start,
                block_rows,
                k,
            )
        })
        .collect::<AnnResult<Vec<_>>>()?;
    results.sort_unstable_by_key(|result| result.block_start);

    for mut result in results {
        flush_pending_edges(&mut result.edges, reservoirs);
        profile.merge(result.profile);
    }
}
```

### 8.4 thread-local scratch

普通叶子路径使用 thread-local scratch，关键代码如下：

```rust
fn with_thread_local_leaf_scratch<R, F>(
    c_max: usize,
    dim: usize,
    leaf_knn: usize,
    f: F,
) -> AnnResult<R>
where
    F: FnOnce(&mut LeafScratch) -> AnnResult<R>,
{
    TLS_LEAF_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let recreate = match slot.as_ref() {
            Some(scratch) => scratch.x.ncols() != dim || scratch.x.nrows() < c_max,
            None => true,
        };
        if recreate {
            *slot = Some(LeafScratch::new(c_max, dim, leaf_knn));
        }
        f(slot.as_mut().expect("thread-local scratch initialized"))
    })
}
```

这样可以避免：

- 每个叶子都重复分配 `dmat`
- 每个叶子都重建 `x`
- 每个叶子都重复分配 edge buffer

这是当前叶子阶段稳定性能的一个重要基础。

## 9. 当前 Profiling 指标怎么解读

现在日志里最关键的几组指标如下。

### 9.1 构图总阶段

- `PiPNN compute_sketches done`
- `PiPNN init reservoirs done`
- `PiPNN streaming (RBC+Leaf) done`
- `PiPNN build graph done`
- `PiPNN build_graph_pipnn done`

### 9.2 RBC 阶段

- `RBC Duration`
- `Cluster assign`
- `GEMM profile: build_x / gemm / topk / merge / blocks`
- `Oversized leaf events`
- `Leaf size distribution`

### 9.3 Scheduler / leaf 阶段

- `producer_wall`
- `total_wall`
- `leaf_parallelism`
- `peak_inflight_leaf_tasks`
- `peak_leaf_backlog`
- `producer_help_drains`
- `producer_help_yields`
- `large_leaf_count`
- `large_leaf_block_tasks`
- `leaf profile: load / distance / topk / hash / flush`

这些指标的直观含义：

- `producer_wall ≈ total_wall`：说明总 wall-time 主要被 RBC/producer 卡住
- `leaf_parallelism`：叶子阶段并行效率的近似指标
- `large_leaf_count` / `large_leaf_block_tasks`：大叶子是否仍然在拖尾
- `GEMM profile.blocks`：RBC assignment 被切成了多少 block

## 10. 这段时间做过的关键优化

下面只记录已经真正落地并在代码中存在的优化，不写纯设想。

### 10.1 自动参数推荐偏向 wiki35m 选择 `10x2`

对应提交脉络：

- `5870bc1 add pipnn auto parameter recommendation`
- `f3bdc4f feat: prefer 10x2 auto params for wiki35m-scale datasets`

当前效果：

- 对 `n >= 20M` 的大规模数据，`auto_params` 会优先给 `fanout_top=10, fanout_second=2`
- 同时约束 `c_max`，避免隐式变大破坏历史 `10x2` 运行形态

### 10.2 RBC / leaf 使用单一线程池，线程数变成真正硬上限

对应提交脉络：

- `74681b0 pipnn: add scheduler instrumentation and thread-cap fixes`
- `2a9c2fe pipnn: stabilize producer scheduling and enable profiling`

目标：

- 避免历史版本里 producer/consumer 各自开线程造成 oversubscription
- 避免线程数失控冲到不可控水平

当前状态：

- `--num-threads` 是整个 PiPNN 阶段的硬上限
- BLAS/OpenMP 在 PiPNN 主阶段被钉成单线程

### 10.3 scheduler 引入 bounded backlog + producer helping

优化点：

- bounded `ArrayQueue`
- backlog 满时 producer inline drain 一个 leaf
- producer 活跃时保留一部分 worker 给 RBC
- producer 结束后放开 leaf worker cap

解决的问题：

- 线程打满但 wall-time 反而退化
- backlog 无界增长
- 依赖 `yield_now` 带来的不稳定执行形态
- 某些情况下的 stack overflow 风险

### 10.4 叶子 thread-local scratch 复用

这项优化的目的是减少叶子构建中的高频堆分配和矩阵缓冲重建。

当前 thread-local scratch 复用：

- `dmat`
- `block_dmat`
- `x`
- `edges`
- `row_topk`

### 10.5 大叶子 blockwise / parallel 路径

当前实现中：

- 普通大叶子会走 blockwise 精确 top-k
- 更大的叶子会在 scheduler 中进入 parallel-large-leaf 路径

这比早期把所有叶子都视作同一种 kernel 的做法更稳，也减少了极端大叶子对 tail latency 的影响。

### 10.6 `fanout=1` RBC fast path

对应提交：

- `fad71ca pipnn: speed up wiki35m RBC partitioning`

主要改动：

- GEMM 路径对 `fanout=1` 不再维护 `StackTopK`
- 标量路径对 `fanout=1` 也直接走 best leader 逻辑

由于日志显示 `Fanout per attempt (avg)` 大约只有 `1.1`，这项优化覆盖面非常广。

### 10.7 平衡化的递归拆分

当前 `parallel_join_clusters` 使用按点数尽量均衡的二分拆分，而不是简单按 cluster 数一刀切。

这样做的收益：

- 左右子树工作量更平衡
- 降低“一个分支很快结束、另一个分支很重”的尾部不均衡
- 更有利于 Rayon `join` 的实际并行效果

### 10.8 oversized leaf 8k hard cap

对应提交：

- `b15147f pipnn: cap oversized emitted leaves at 8k`

这项优化的作用不是提高平均性能，而是稳定坏尾部：

- 消灭 `10K+` 极端叶子
- 降低 `large_leaf_block_tasks`
- 限制最坏情况内存和 wall-time

在 wiki35m 上，这项优化已经实测有效。

## 11. 近期实验的关键信息

当前主参考结果是：

- baseline：`wiki35m_pipnn_20260323_142633.log`
- `4096x256` tile 实验：`wiki35m_pipnn_20260323_154057.log`
- `4096x384` tile 实验：`wiki35m_pipnn_20260323_161955.log`

基线结果大致是：

- `RBC Duration: 35m57s`
- `PiPNN streaming (RBC+Leaf): 2159.18s`
- `PiPNN build_graph_pipnn: 2195.80s`
- `leaf_parallelism: 24.21`
- `large_leaf_count: 62`
- `large_leaf_block_tasks: 483`
- 搜索：`QPS≈5065.9`, `R@10≈96.06`

tile A/B 的结论是：

- `4096x256` 比 baseline 更慢
- `4096x384` 只比 baseline 略快，大约 `1s` 级别

所以单纯继续调 `leader tile` 不是后续主攻方向。

## 12. 当前实现的关键限制

这部分非常重要，避免后续误判。

### 12.1 RBC 分区仍然使用原始高维向量

虽然 PiPNN 一开始会预计算 sketch，但当前 sketch 主要用于 leaf/hash pruning。

RBC partition 的 cluster assignment 仍然是在原始向量上完成的：

- leaders 从 dataset 取原始向量
- point block 也从 dataset 取原始向量
- GEMM 直接在原始 `dim` 上做

这意味着：

- 当前最重的 RBC assignment 还没有吃到 sketch 维度压缩的红利
- 从第一性原理看，这仍是未来最大潜在优化点之一

### 12.2 `fanout≈1` 下仍然存在大量 cluster materialization

虽然 `fanout=1` 已经绕开了 `StackTopK`，但当前实现仍然需要：

- 维护 `best_leaders`
- 写入 `local_clusters`
- merge 到最终 `clusters`

这部分仍然存在较重的内存写放大和 `Vec<Vec<u32>>` 管理成本。

### 12.3 自动参数估时仍偏乐观

当前 `estimate_build_time` 只是一个预算模型，不是精确仿真器。

在历史 wiki35m 实验里，自动预估通常明显乐观，不能直接拿来当真实 wall-time。

## 13. 如何读这套实现

如果要快速理解代码，建议按这个顺序读：

1. `cmd_drivers/build_forgeann_index/src/main.rs`
2. `ForgeANN/src/forgeann/mod.rs`
3. `ForgeANN/src/forgeann/params.rs`
4. `ForgeANN/src/forgeann/rbc_partition/`
5. `ForgeANN/src/forgeann/scheduler.rs`
6. `ForgeANN/src/forgeann/leaf_build.rs`
7. `ForgeANN/src/forgeann/hash_prune.rs`

这样读的原因是：

- 先理解顶层 orchestration
- 再理解参数来源
- 再看最重的 RBC
- 最后看 scheduler 和 leaf 的细节

## 14. 一句话总结

当前 PiPNN 实现的核心是：

- 用 RBC 递归流式地产生叶子
- 用一个统一的、线程数受控的 scheduler 在同一线程池里并行做叶子构图
- 用 GEMM 加速 RBC assignment 和叶内距离计算
- 用 reservoir 汇总最终邻接表

而这段时间的优化主线，可以概括为：

- 参数上让 wiki35m 更稳定地选择 `10x2`
- 调度上把线程上限真正收住
- 在不破坏质量的前提下消除极端大叶子
- 对 `fanout=1` 和大叶子路径做专门快路径
- 增加 profiling 和 phase monitoring，让后续优化基于证据而不是猜测
