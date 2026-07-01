# PIPNN 参数选择实用指南

## 快速开始

### 使用自动推荐（推荐）

PIPNN 提供了基于理论模型的自动参数推荐功能：

```rust
use forgeann::forgeann::ForgeANNParams;

// 自动推荐参数
let params = ForgeANNParams::default();

// 使用参数构建 ForgeANN/PiPNN 图
build_forgeann_graph(dataset, num_points, num_threads, &params, on_node_done)?;
```

### 手动配置

如果需要精细控制，可以手动设置参数：

```rust
let params = ForgeANNParams {
    // Fanout 参数
    fanout_top: 10,
    fanout_second: 3,

    // 叶子大小
    c_min: 256,
    c_max: 2048,

    // Leader 采样
    psamp_fraction: 0.001,
    max_leaders: 1000,

    // HashPrune
    m_hash_bits: 12,
    l_max: 64,

    // 其他参数使用默认值
    ..Default::default()
};
```

## 核心参数详解

### 1. Fanout 参数

#### 1.1 fanout_top（顶层 fanout）

**含义**：在第 0 层（顶层）分区时，每个点被复制到多少个子簇中。

**影响**：
- ↑ 召回率提升（点在更多簇中出现）
- ↑ 构图时间增加（更多距离计算）
- ↑ 内存占用增加（更多叶子副本）

**推荐值**：

| 数据规模 (n) | 推荐值 | 说明 |
|-------------|-------|------|
| < 1M | 8 | 小规模数据，簇数少，较小 fanout 即可 |
| 1M - 10M | 10 | 默认值，适合大多数场景 |
| 10M - 100M | 12-15 | 大规模数据，需要更多复制以保证召回率 |
| ≥ 100M | 15-20 | 超大规模，显著增加 fanout |

**调优建议**：
- 如果召回率不足，优先增加 `fanout_top`
- 如果构图时间过长，考虑减小 `fanout_top`
- 通常 `fanout_top` 应该是 `fanout_second` 的 2-4 倍

#### 1.2 fanout_second（第二层 fanout）

**含义**：在第 1 层分区时，每个点被复制到多少个子簇中。

**影响**：
- 与 `fanout_top` 类似，但影响较小（因为作用在更小的簇上）
- 总复制因子 = `fanout_top × fanout_second`

**推荐值**：

| 数据规模 (n) | 推荐值 | 说明 |
|-------------|-------|------|
| < 1M | 2 | 小规模数据 |
| 1M - 10M | 3 | 默认值 |
| 10M - 100M | 3-4 | 大规模数据 |
| ≥ 100M | 4-5 | 超大规模 |

**调优建议**：
- `fanout_second` 对成本的影响比 `fanout_top` 更大（因为作用在更多的簇上）
- 如果内存或时间受限，优先减小 `fanout_second`
- 对于高维数据（d > 512），可适当增加 `fanout_second`

#### 1.3 总复制因子 (F_total)

**计算**：`F_total = fanout_top × fanout_second`

**推荐范围**：

| 数据规模 (n) | F_total 范围 | 典型值 |
|-------------|-------------|-------|
| < 1M | 12-24 | 16 |
| 1M - 10M | 20-40 | 30 |
| 10M - 100M | 36-60 | 48 |
| ≥ 100M | 60-120 | 80 |

**经验公式**：

```
F_total ≈ 10 × log₁₀(n)
```

例如：
- n = 1M → F_total ≈ 60（实际使用 30 即可）
- n = 10M → F_total ≈ 70（实际使用 30-48）
- n = 100M → F_total ≈ 80（实际使用 60-80）

### 2. 叶子大小参数

#### 2.1 c_max（最大叶子大小）

**含义**：RBC 分区的目标叶子大小上限。

**影响**：
- ↑ 叶内距离计算增加（O(c²)）
- ↑ 叶子数量减少（减少 RBC 开销）
- ↑ 召回率可能提升（叶内更全面）

**推荐值**：

| 数据规模 (n) | 推荐值 | 说明 |
|-------------|-------|------|
| < 1M | 1024-2048 | 默认 2048 |
| 1M - 10M | 2048 | 默认值 |
| 10M - 100M | 2048-4096 | 大规模数据，增大叶子以减少叶子数 |
| ≥ 100M | 4096 | 超大规模，显著增大叶子 |

**注意**：
- `c_max` 受 `max_full_matrix_leaf_size` 限制（默认 3500）
- 超过该限制会切换到 blockwise 模式（性能略降）
- 对于超大规模数据，`adaptive_c_max` 会自动 ×2

**调优建议**：
- 如果 RBC 时间过长（叶子数太多），增加 `c_max`
- 如果叶内 GEMM 时间过长，减小 `c_max`
- 通常 `c_max` 应该在 1024-4096 范围内

#### 2.2 c_min（最小叶子大小）

**含义**：RBC 分区的目标叶子大小下限。

**影响**：
- 防止产生过小的叶子（浪费 overhead）
- 通常设为 `c_max / 8` 左右

**推荐值**：`256`（默认值，通常不需要修改）

### 3. Leader 采样参数

#### 3.1 psamp_fraction（采样比例）

**含义**：在 RBC 分区时，从当前簇中采样多少比例的点作为 leaders。

**影响**：
- ↑ Leader 数量增加
- ↑ RBC 距离计算增加
- ↑ 簇划分更精细

**推荐值**：

| 数据规模 (n) | 推荐值 | 说明 |
|-------------|-------|------|
| < 1M | 0.005-0.01 | 较大采样比例 |
| 1M - 10M | 0.001 | 默认值 |
| 10M - 100M | 0.0005-0.001 | 减小采样比例 |
| ≥ 100M | 0.0001-0.0005 | 显著减小采样比例 |

**注意**：
- `adaptive_psamp_fraction` 会根据簇大小和深度自动调整
- 实际采样比例 = `psamp_fraction × scale_factor`
- 对于 n ≥ 20M，`scale_factor = 0.05`

**调优建议**：
- 如果 RBC 时间过长，减小 `psamp_fraction`
- 如果召回率不足，增加 `psamp_fraction`
- 通常不需要手动调整，使用自动推荐即可

#### 3.2 max_leaders（最大 leader 数）

**含义**：每层最多采样多少个 leaders（上限）。

**影响**：
- 限制 RBC 距离计算的上界
- 防止在大簇中采样过多 leaders

**推荐值**：

| 数据规模 (n) | 推荐值 | 说明 |
|-------------|-------|------|
| < 1M | 500-1000 | 默认 1000 |
| 1M - 10M | 1000-2000 | 默认 1000 |
| 10M - 100M | 2000-5000 | 增加上限 |
| ≥ 100M | 5000-10000 | 显著增加上限 |

**调优建议**：
- 如果顶层 RBC 时间过长，减小 `max_leaders`
- 如果召回率不足且 `psamp_fraction` 已较大，增加 `max_leaders`
- 通常 `max_leaders` 应该在 500-10000 范围内

### 4. HashPrune 参数

#### 4.1 l_max（水库容量）

**含义**：每个点的 HashPrune 水库容量，对应目标最大出度。

**影响**：
- ↑ 图的平均出度增加
- ↑ 内存占用增加（主要瓶颈）
- ↑ 搜索时的候选点增加

**推荐值**：

| 目标召回率 | 推荐值 | 说明 |
|-----------|-------|------|
| 90% | 32 | 低出度，快速搜索 |
| 95% | 64 | 默认值，平衡 |
| 98% | 96-128 | 高召回率 |
| 99%+ | 128-256 | 极高召回率 |

**注意**：
- `l_max` 是内存占用的主要因素：`M ≈ 576n × (l_max/64)`
- 对于 n=35M, l_max=64：`M ≈ 20 GB`
- 对于 n=35M, l_max=128：`M ≈ 40 GB`

**调优建议**：
- 如果内存不足，减小 `l_max`
- 如果召回率不足，增加 `l_max`
- 通常 `l_max` 应该在 32-128 范围内

#### 4.2 m_hash_bits（哈希位数）

**含义**：HashPrune 使用的残差哈希位数（超平面个数）。

**影响**：
- ↑ 哈希冲突减少
- ↑ Sketch 计算和存储增加
- 通常 12 位已足够（冲突率 < 0.1%）

**推荐值**：`12`（默认值，通常不需要修改）

**调优建议**：
- 如果 `l_max` 很大（> 128），可考虑增加到 14-16
- 如果内存极度受限，可减小到 10
- 必须 ≤ 16（u16 限制）

### 5. 其他参数

#### 5.1 leaf_knn（叶内 k-NN）

**含义**：在每个叶子内部，为每个点选择多少个最近邻。

**影响**：
- ↑ 每个点的候选边增加
- ↑ HashPrune 水库更快填满
- 通常 2-3 即可（HashPrune 会去重）

**推荐值**：`2`（默认值，通常不需要修改）

#### 5.2 max_depth（最大深度）

**含义**：RBC 递归的最大深度，超过该深度直接叶化。

**影响**：
- 防止过深递归（栈溢出）
- 通常不会达到（RBC 会提前停止）

**推荐值**：`10`（默认值，通常不需要修改）

#### 5.3 min_shrink_ratio（最小收缩率）

**含义**：父子簇规模的最小收缩率阈值，低于该阈值时提前叶化。

**影响**：
- ↑ RBC 更激进地早停
- ↓ 叶子数量减少
- ↓ 召回率可能略降

**推荐值**：`1.5`（默认值）

**调优建议**：
- 如果 RBC 时间过长，增加到 2.0-3.0
- 如果召回率不足，减小到 1.2-1.3

#### 5.4 final_prune（最终精剪）

**含义**：是否在构图完成后进行 RobustPrune 精剪。

**影响**：
- ↑ 图质量提升（去除冗余边）
- ↑ 构图时间增加（约 10-20%）

**推荐值**：`true`（默认值，建议启用）

#### 5.5 final_prune_alpha（精剪参数）

**含义**：RobustPrune 的 alpha 参数，控制精剪强度。

**影响**：
- ↑ 精剪更激进（保留更少边）
- ↓ 召回率可能略降

**推荐值**：`1.2`（默认值）

## 参数组合示例

### 示例 1：小规模数据（1M 点）

```rust
let params = ForgeANNParams {
    fanout_top: 8,
    fanout_second: 2,
    c_max: 2048,
    psamp_fraction: 0.005,
    max_leaders: 500,
    l_max: 64,
    ..Default::default()
};
```

**预期性能**：
- 构图时间：~20 秒（8 线程）
- 内存峰值：~1 GB
- 召回率：~96%

### 示例 2：中等规模数据（10M 点）

```rust
let params = ForgeANNParams {
    fanout_top: 10,
    fanout_second: 3,
    c_max: 2048,
    psamp_fraction: 0.001,
    max_leaders: 1000,
    l_max: 64,
    ..Default::default()
};
```

**预期性能**：
- 构图时间：~300 秒（42 线程）
- 内存峰值：~7 GB
- 召回率：~95%

### 示例 3：大规模数据（35M 点）

```rust
let params = ForgeANNParams {
    fanout_top: 12,
    fanout_second: 4,
    c_max: 2048,  // adaptive_c_max 会自动 ×2 = 4096
    psamp_fraction: 0.0005,
    max_leaders: 5000,
    l_max: 64,
    ..Default::default()
};
```

**预期性能**：
- 构图时间：~1800 秒（42 线程）
- 内存峰值：~22 GB
- 召回率：~96%

### 示例 4：超大规模数据（100M 点）

```rust
let params = ForgeANNParams {
    fanout_top: 15,
    fanout_second: 4,
    c_max: 4096,  // adaptive_c_max 会自动 ×2 = 8192
    psamp_fraction: 0.0002,
    max_leaders: 10000,
    l_max: 64,
    min_shrink_ratio: 2.0,  // 更激进的早停
    ..Default::default()
};
```

**预期性能**：
- 构图时间：~6000 秒（42 线程）
- 内存峰值：~60 GB
- 召回率：~95%

## 调优流程

### 1. 初始配置

使用自动推荐：

```rust
let params = ForgeANNParams::recommend(n, d, R, num_threads);
```

### 2. 构图并评估

```bash
cargo run -r -p build_forgeann_index -- \
    --data-path data.bin \
    --index-path index.bin \
    --auto-params  # 使用自动推荐
```

观察日志输出：
- RBC 时间占比
- 叶内 GEMM 时间占比
- 内存峰值
- 平均出度

### 3. 评估召回率

```bash
cargo run -r -p search_velo_index -- \
    --index-path index.bin \
    --query-path queries.bin \
    --ground-truth gt.bin \
    --k 10
```

### 4. 根据结果调整

**情况 1：召回率不足（< 95%）**

优先级：
1. 增加 `fanout_top`（+2）
2. 增加 `fanout_second`（+1）
3. 增加 `l_max`（×1.5）
4. 减小 `min_shrink_ratio`（-0.2）

**情况 2：构图时间过长**

优先级：
1. 减小 `fanout_second`（-1）
2. 增加 `c_max`（×1.5）
3. 减小 `max_leaders`（×0.7）
4. 增加 `min_shrink_ratio`（+0.3）

**情况 3：内存不足**

优先级：
1. 减小 `l_max`（×0.75）
2. 减小 `c_max`（×0.75）
3. 减小 `fanout_second`（-1）

**情况 4：RBC 时间占比过高（> 50%）**

优先级：
1. 增加 `c_max`（×1.5）
2. 减小 `max_leaders`（×0.7）
3. 增加 `min_shrink_ratio`（+0.3）

**情况 5：叶内 GEMM 时间占比过高（> 70%）**

优先级：
1. 减小 `c_max`（×0.75）
2. 减小 `fanout_second`（-1）

### 5. 迭代优化

重复步骤 2-4，直到达到目标性能。

## 常见问题

### Q1：为什么自动推荐的参数构图时间比预期长？

**可能原因**：
1. CPU 性能低于假设（50 GFLOPS/core）
2. 内存带宽瓶颈
3. BLAS 库未优化（使用 OpenBLAS 或 MKL）

**解决方案**：
- 检查 BLAS 库：`ldd target/release/build_forgeann_index | grep blas`
- 使用 `perf` 分析瓶颈：`perf record -g cargo run -r ...`

### Q2：为什么召回率低于预期？

**可能原因**：
1. 数据分布不均匀（聚类严重）
2. Fanout 不足
3. `l_max` 过小

**解决方案**：
- 增加 `fanout_top` 和 `fanout_second`
- 增加 `l_max`
- 检查数据分布：计算点间距离的方差

### Q3：为什么内存占用远超预期？

**可能原因**：
1. `l_max` 过大
2. 叶子 scratch 空间过大（`c_max` 过大）
3. 内存碎片

**解决方案**：
- 减小 `l_max`
- 减小 `c_max`
- 使用 `jemalloc`：在 `Cargo.toml` 中添加 `jemallocator`

### Q4：如何在有限内存下构建大规模索引？

**策略**：
1. 减小 `l_max`（例如 32）
2. 使用流式构图（已默认启用）
3. 分批构图：将数据分成多个子集，分别构图后合并

### Q5：如何选择线程数？

**推荐**：
- 构图：使用所有物理核心（不含超线程）
- 搜索：使用 1-2 个线程（避免锁竞争）

**示例**：
```bash
# 查看物理核心数
lscpu | grep "Core(s) per socket"

# 构图时使用所有物理核心
cargo run -r -p build_forgeann_index -- --num-threads 42

# 搜索时使用单线程
cargo run -r -p search_velo_index -- --num-threads 1
```

## 参考资料

- [PIPNN Fanout 理论分析](pipnn_fanout_theory.md)
- [PIPNN Fanout 数学推导](pipnn_fanout_math.md)
- [PIPNN 论文](https://arxiv.org/abs/...)（待发布）

---

**文档版本**：1.0
**最后更新**：2026-03-15
**作者**：ForgeANN Team
