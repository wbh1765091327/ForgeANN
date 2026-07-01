# ForgeANN — 图构建引擎

ForgeANN 是 ForgeANN 的核心图构建模块，实现了一种基于 RBC (Radius-Based Clustering) 分区的近邻图构建算法，支持全内存和 OOM (Out-of-Memory) 两种运行模式。

## 目录

- [架构概览](#架构概览)
- [构建流水线](#构建流水线)
- [模块导读](#模块导读)
- [OOM 模式](#oom-模式)
- [关键数据结构](#关键数据结构)

## 架构概览

```
build_forgeann_graph()
  │
  ├─ In-Memory 路径:
  │   compute_sketches → init reservoirs → run_scheduler_scope(RBC + leaf)
  │                                              ↓
  │                                        rbc_partition_streaming
  │                                              ↓
  │                                        process_leaf (per leaf)
  │                                              ↓
  │                                        HashPrune → reservoir emit
  │
  └─ OOM 路径:
      D0 ADS → D1 ADS → D1 resident-subtree → D1 materialize
         ↓
      native subtree child processing
         ↓
      bounded async leaf drain → direct point-store graph write → final RobustPrune
```

## 构建流水线

### Phase 1: Sketch 计算
`hash_prune.rs` — 为每个向量预计算 m 维随机投影 sketch，用于后续 HashPrune 剪枝。所有 sketch 在 RBC 开始前一次性算完。

### Phase 2: RBC 分区
`rbc_partition/` — 递归地将点集分成若干叶子簇。从根节点开始，每层用 GEMM 计算点到 leader 的距离，将点分配到最近的 leader 子集，直到叶子大小 ≤ `kernel_safe_leaf_size()`。

### Phase 3: 叶子构建
`leaf_build.rs` — 对每个叶子簇，在叶子内执行 k-NN 搜索，生成候选邻居边。小叶子用精确搜索，大叶子用分块 GEMM + TopK。

### Phase 4: 剪枝 & 输出
`hash_prune.rs` (HashPruneReservoir) — 每个点维护一个大小为 `l_max` 的水库，收到候选边后用 HashPrune 算法保留最优邻居。构图结束时从水库输出最终邻接表。

## 模块导读

### 入口 & 编排

| 文件 | 行数 | 职责 |
|------|------|------|
| `mod.rs` | 2845 | **主入口**。`build_forgeann_graph()` 和 `build_forgeann_graph_oom()` 的实现，内存预算规划，Rayon 线程池管理，阶段间编排。 |
| `params.rs` | 994 | **参数定义**。`ForgeANNParams` 结构体，包含所有算法超参（fanout、leaf size、hash bits、OOM budget 等）。 |

### 核心算法

| 文件 | 行数 | 职责 |
|------|------|------|
| `rbc_partition/` | ~24000 | **RBC 分区子系统**。递归分区、GEMM 距离计算、D1 resident-subtree、D1 materialize 和 I/O 规划。核心文件是 `child_run_io.rs`。 |
| `leaf_build.rs` | 1554 | **叶子构建**。对叶子簇执行 k-NN：小叶子精确搜索，大叶子分块 GEMM + TopK。输出候选边到 `PendingEdgeSink`。 |
| `hash_prune.rs` | 475 | **HashPrune 剪枝**。`SketchAccessor` trait、`HashPruneReservoir` 水库。基于 sketch 的近邻剪枝算法。 |
| `scheduler.rs` | 2176 | **任务调度器**。`ForgeannScheduler` 在单个 Rayon FIFO 线程池上协调 RBC 分区和叶子构建的并发执行，控制 backlog 和内存使用。 |
| `adsampling.rs` | 3188 | **自适应采样**。AdSampling 根分配策略：用旋转 + 分组量化替代全量 GEMM，加速根节点分配。 |

### 数据存储 & I/O

| 文件 | 行数 | 职责 |
|------|------|------|
| `point_store.rs` | 1941 | **向量存储抽象**。`PointStore` trait 统一内存/mmap/DirectIO 读取。实现包括 `InmemDatasetPointStore`、`DirectPointStore`、`ResidentSubsetPointStore` 等。 |
| `point_pipeline.rs` | 878 | **异步点预取管线**。OOM 模式下在叶子处理前异步预取向量到内存，减少叶子阶段的 I/O 等待。 |
| `external_store.rs` | 1560 | **外存 sketch 存储**。`DiskSketchStore` 将 sketch 持久化到磁盘，`ResidentPrefixSketchAccessor` 管理常驻内存前缀。 |
| `direct_io.rs` | 287 | **Direct I/O 封装**。绕过 page cache 的直接磁盘读写，用于大数据集减少内存压力。 |
| `io_runtime.rs` | 268 | **I/O 运行时**。向量窗口缓存 `VectorWindowCache` 和有界读取规划器。 |

### OOM 专用

| 文件 | 行数 | 职责 |
|------|------|------|
| `spill_hashprune.rs` | 2612 | **外存候选边 spill**。`SpillEdgeSink` / `InMemorySpillEdgeSink` 将超出内存预算的候选边写入磁盘分片，构建完成后 reduce 归并。 |
| `io_planned_forgeann.rs` | 1707 | **I/O 规划器**。根据内存预算和存储状态（resident / vector-run / raw-ids）为每个节点选择最优 I/O 策略。 |

### 辅助模块

| 文件 | 行数 | 职责 |
|------|------|------|
| `sampling.rs` | 89 | **采样工具**。`sample_set_bottomk` 等采样辅助函数。 |

## OOM 模式

当数据集超出可用内存时，`build_forgeann_graph_oom()` 启用生产 OOM 路径：

1. **D0/D1 ADSampling**: 根层和 D1 使用 ADSampling assignment，减少全量 GEMM 代价
2. **D1 resident-subtree**: 原生消费满足预算的 D1 子树，按组 hydrate/build，并复用 resident point/sketch 数据
3. **D1 materialize/native child processing**: 对剩余 child runs 走当前原生 child/leaf 消费路径
4. **有界 leaf drain**: leaf 构建异步 drain，但保持生产默认 backlog 上限
5. **Direct point-store graph write**: 最终图直接写 `_mem.index`
6. **Final RobustPrune**: 生产 driver 默认开启最终 RobustPrune；该阶段直接基于 mmap 数据集和 `InMemoryGraph` 执行，不声称遵循 strict I/O

内存预算分配（`OomMemoryPlan`）:
- 45% 水库 (reservoir)
- 15% scratch 缓冲区
- 剩余预算作为 I/O/replay headroom

## 关键数据结构

- **`ForgeANNParams`** — 所有算法超参，入口必读
- **`PointStore`** trait — 向量读取的统一接口
- **`HashPruneReservoir`** — 每个点一个，存储 TopK 邻居候选
- **`SketchAccessor`** trait — sketch 读取接口
- **`ForgeannScheduler`** — 协调 RBC 和 leaf 的并发调度器
- **`OomMemoryPlan`** — OOM 模式下的内存预算划分

## 阅读建议

1. 先读 `params.rs` 了解所有参数含义
2. 再读 `mod.rs` 中的 `build_forgeann_graph_with_store()` 理解完整 in-memory 流程
3. 然后读 `rbc_partition/mod.rs` 了解分区逻辑的入口
4. 最后按需深入 OOM 相关模块（`spill_hashprune.rs`、`io_planned_forgeann.rs`）
