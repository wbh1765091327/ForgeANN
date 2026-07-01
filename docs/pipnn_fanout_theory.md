# PIPNN Fanout 理论分析：层级复制策略

## 1. 问题定义

在 PIPNN (Partition-based Index with Progressive Nearest Neighbors) 算法中，**fanout** 是指在分层分区过程中，每个点被复制到多少个子分区中。这是一个关键的超参数，直接影响：

- **召回率 (Recall)**：fanout 越大，点在多个分区中出现，搜索时更容易找到真实近邻
- **构图成本 (Build Cost)**：fanout 越大，需要处理的点对数量增加，构图时间和内存消耗上升
- **搜索效率 (Search Efficiency)**：fanout 影响图的连通性和搜索路径长度

本文档从理论角度分析如何根据数据规模和分布选择最优的 fanout 策略。

## 2. PIPNN 分层结构

PIPNN 使用 RBC (Radius-Based Clustering) 递归分区构建层级结构：

```
Level 0 (Top):     全部 n 个点
                   ↓ (fanout_top = F₀)
Level 1:           k₀ 个簇，每簇约 n/k₀ 个点
                   ↓ (fanout_second = F₁)
Level 2:           k₁ 个簇，每簇约 n/(k₀·k₁) 个点
                   ↓ (fanout = 1)
Level 3+:          继续递归直到叶子大小 ∈ [c_min, c_max]
```

### 2.1 当前实现的 Fanout 策略

根据 `params.rs:45-53`：

```rust
pub fn fanout_for_depth(&self, depth: usize) -> usize {
    match depth {
        0 => self.fanout_top,      // 默认 10
        1 => self.fanout_second,   // 默认 3
        _ => 1,                     // depth ≥ 2 时不再复制
    }
}
```

**关键观察**：
- 仅在前两层进行点复制
- 总复制因子：`F_total = fanout_top × fanout_second = 10 × 3 = 30`
- 每个点最多出现在 30 个叶子分区中

## 3. 理论模型

### 3.1 距离计算成本分析

PIPNN 的总距离计算量可分解为：

**D_total = D_rbc + D_leaf**

其中：
- **D_rbc**：RBC 分区过程中的距离计算
- **D_leaf**：叶子内部的全对全距离计算

#### 3.1.1 RBC 分区成本

在第 `d` 层，设：
- `n_d`：该层的点数
- `k_d`：采样的 leader 数量
- `F_d`：该层的 fanout

则该层的距离计算量：

```
D_rbc(d) = n_d × k_d + n_d × F_d × k_{d+1}
           ↑            ↑
           找最近的     为每个选中的
           k_d 个       leader 找
           leaders      k_{d+1} 个
                       sub-leaders
```

对于两层 fanout 策略：

```
D_rbc = n × k₀ + n × F₀ × k₁
```

其中：
- `k₀ = min(n × psamp₀, max_leaders)`
- `k₁ = min((n/k₀) × psamp₁, max_leaders)`

#### 3.1.2 叶子内部成本

每个点平均出现在 `F_total = F₀ × F₁` 个叶子中，每个叶子平均大小为 `c_avg`：

```
D_leaf = n × F_total × c_avg / 2
```

除以 2 是因为全对全距离计算中每对点只计算一次。

### 3.2 召回率分析

#### 3.2.1 近邻丢失概率

假设点 `p` 的真实 k-近邻为 `{q₁, q₂, ..., qₖ}`。在分层分区中，`p` 和 `qᵢ` 能够在同一叶子中相遇的概率取决于：

1. **第 0 层分配**：`p` 被分配到 `F₀` 个簇，`qᵢ` 被分配到 `F₀` 个簇
   - 相遇概率：`P₀ = F₀ / k₀`（假设簇分配均匀）

2. **第 1 层分配**：在已相遇的簇中，再次相遇的概率
   - 相遇概率：`P₁ = F₁ / k₁`

3. **总相遇概率**：
   ```
   P_meet = 1 - (1 - P₀ × P₁)^{F₀}
          ≈ F₀ × F₁ / (k₀ × k₁)  (当 P₀ × P₁ << 1)
   ```

#### 3.2.2 Fanout 与召回率的关系

**关键洞察**：召回率与 `F_total = F₀ × F₁` 近似线性相关（在合理范围内）：

```
Recall ≈ min(1, α × F_total / √n)
```

其中 `α` 是数据分布相关的常数。

**推导依据**：
- 簇数量通常与 `√n` 成正比（保持簇大小稳定）
- 点复制越多，近邻共现概率越高
- 但存在饱和效应：当 `F_total` 足够大时，继续增加收益递减

### 3.3 最优 Fanout 选择

#### 3.3.1 约束优化问题

给定：
- 数据规模 `n`
- 向量维度 `d`
- 目标出度 `R`（l_max）
- 目标召回率 `r_target`

目标：最小化构图时间，同时满足召回率约束

```
minimize:   T_build = (D_rbc + D_leaf) × 2d / (FLOPS × threads)
subject to: Recall(F₀, F₁) ≥ r_target
            F₀, F₁ ∈ ℕ⁺
```

#### 3.3.2 预算分配策略

当前实现（`params.rs:115-200`）采用**预算分配法**：

1. **确定总预算**：
   ```
   D_budget = D_vamana × gemm_advantage
            = n × L × R × 10
   ```
   其中 `L ≈ R` 是 Vamana 的搜索列表长度，`gemm_advantage = 10` 是 GEMM 相对 beam search 的效率优势。

2. **分配预算**：
   - RBC：40% → `D_rbc_budget = 0.4 × D_budget`
   - Leaf：60% → `D_leaf_budget = 0.6 × D_budget`

3. **反推参数**：
   - 从 `D_leaf_budget` 反推 `c_max`
   - 从 `D_rbc_budget` 反推 `max_leaders`
   - 从 `max_leaders` 反推 `psamp_fraction`

## 4. 不同数据规模的 Fanout 策略

### 4.1 小规模数据 (n < 1M)

**特征**：
- 簇数量少（k₀ < 1000）
- 相遇概率高
- 内存压力小

**推荐策略**：
```
fanout_top = 8-12
fanout_second = 2-4
F_total = 16-48
```

**理由**：
- 较小的 fanout 已能保证高召回率
- 避免过度复制导致的冗余计算

### 4.2 中等规模数据 (1M ≤ n < 10M)

**特征**：
- 簇数量中等（k₀ ≈ 1000-5000）
- 需要平衡召回率和成本
- 内存开始成为瓶颈

**推荐策略**：
```
fanout_top = 10
fanout_second = 3
F_total = 30
```

**理由**：
- 这是当前默认值，经验上效果良好
- `F_total = 30` 在 10M 规模下能达到 95%+ 召回率

### 4.3 大规模数据 (10M ≤ n < 100M)

**特征**：
- 簇数量大（k₀ ≈ 5000-10000）
- 相遇概率降低
- 内存和计算成本显著

**推荐策略**：
```
fanout_top = 12-15
fanout_second = 3-4
F_total = 36-60
```

**理由**：
- 需要增加 `fanout_top` 以补偿簇数量增加
- 保持 `fanout_second` 较小以控制总成本
- 可考虑引入第三层 fanout（depth=2 时 fanout=2）

### 4.4 超大规模数据 (n ≥ 100M)

**特征**：
- 簇数量极大（k₀ ≈ 10000+）
- 相遇概率很低
- 必须严格控制内存

**推荐策略**：
```
fanout_top = 15-20
fanout_second = 4-5
fanout_depth2 = 2
F_total = 120-200
```

**理由**：
- 需要三层 fanout 以保证召回率
- 增加 `c_max`（叶子大小）以减少叶子数量
- 使用更激进的 RBC early-stop（`min_shrink_ratio`）

## 5. 数据分布的影响

### 5.1 均匀分布 (Uniform)

**特征**：
- 点在空间中均匀分布
- 簇边界清晰
- 近邻通常在同一簇内

**Fanout 调整**：
- 可使用较小的 fanout（F_total = 20-30）
- 重点优化叶子大小 `c_max`

### 5.2 聚类分布 (Clustered)

**特征**：
- 点形成明显的簇
- 簇内密集，簇间稀疏
- 近邻可能跨簇

**Fanout 调整**：
- 需要较大的 fanout（F_total = 30-50）
- 增加 `fanout_top` 以覆盖多个簇
- 可适当减小 `c_max`

### 5.3 长尾分布 (Long-tail)

**特征**：
- 少数热点区域密集
- 大部分区域稀疏
- 近邻分布不均

**Fanout 调整**：
- 使用自适应 fanout：
  ```rust
  fanout = base_fanout × (1 + log(local_density))
  ```
- 对稀疏区域增加 fanout
- 对密集区域减少 fanout

### 5.4 高维数据 (d > 512)

**特征**：
- 维度灾难：点间距离趋于均匀
- 簇结构不明显
- 近邻搜索困难

**Fanout 调整**：
- 显著增加 fanout（F_total = 50-100）
- 减小 `c_max` 以提高叶内精度
- 考虑使用降维技术（PCA/Random Projection）

## 6. 实现建议

### 6.1 自适应 Fanout 策略

建议在 `ForgeANNParams` 中添加自适应 fanout 方法：

```rust
pub fn adaptive_fanout_for_depth(
    &self,
    depth: usize,
    cluster_size: usize,
    total_size: usize,
) -> usize {
    let base_fanout = self.fanout_for_depth(depth);

    // 根据数据规模调整
    let scale_factor = if total_size >= 100_000_000 {
        1.5
    } else if total_size >= 10_000_000 {
        1.2
    } else {
        1.0
    };

    // 根据簇大小调整
    let cluster_factor = if cluster_size < 1000 {
        0.8  // 小簇减少 fanout
    } else if cluster_size > 100_000 {
        1.3  // 大簇增加 fanout
    } else {
        1.0
    };

    (base_fanout as f64 * scale_factor * cluster_factor)
        .round() as usize
        .max(1)
}
```

### 6.2 三层 Fanout 支持

对于超大规模数据，建议支持三层 fanout：

```rust
pub struct ForgeANNParams {
    // ...
    pub fanout_top: usize,      // depth 0
    pub fanout_second: usize,   // depth 1
    pub fanout_third: usize,    // depth 2 (新增)
    // ...
}

pub fn fanout_for_depth(&self, depth: usize) -> usize {
    match depth {
        0 => self.fanout_top,
        1 => self.fanout_second,
        2 => self.fanout_third,
        _ => 1,
    }
}
```

### 6.3 动态 Fanout 调整

在构图过程中，根据实时统计动态调整 fanout：

```rust
pub struct FanoutAdjuster {
    target_recall: f64,
    current_recall_estimate: f64,
    adjustment_rate: f64,
}

impl FanoutAdjuster {
    pub fn adjust(&mut self, depth: usize, base_fanout: usize) -> usize {
        let recall_gap = self.target_recall - self.current_recall_estimate;
        let adjustment = (recall_gap * self.adjustment_rate * base_fanout as f64)
            .round() as isize;

        (base_fanout as isize + adjustment)
            .max(1) as usize
    }
}
```

## 7. 实验验证

### 7.1 推荐实验设置

对于给定数据集，建议进行以下实验：

1. **Fanout 扫描**：
   - 固定其他参数
   - 变化 `fanout_top ∈ {5, 8, 10, 12, 15, 20}`
   - 变化 `fanout_second ∈ {2, 3, 4, 5}`
   - 测量召回率和构图时间

2. **规模测试**：
   - 在不同数据规模下（1M, 10M, 35M, 100M）
   - 使用推荐的 fanout 策略
   - 验证召回率 ≥ 95%

3. **分布测试**：
   - 在不同分布数据上（uniform, clustered, long-tail）
   - 比较固定 fanout vs 自适应 fanout
   - 测量召回率和效率

### 7.2 评估指标

- **召回率 (Recall@k)**：`|真实 k-NN ∩ 返回 k-NN| / k`
- **构图时间 (Build Time)**：总构图时间（秒）
- **内存峰值 (Peak Memory)**：构图过程中的最大内存占用
- **平均出度 (Avg Degree)**：图中每个点的平均邻居数
- **QPS (Queries Per Second)**：搜索吞吐量

## 8. 总结

### 8.1 核心原则

1. **Fanout 与召回率正相关**：增加 fanout 可提高召回率，但收益递减
2. **Fanout 与成本正相关**：fanout 增加导致距离计算量和内存占用增加
3. **分层策略优于单层**：两层 fanout（top + second）比单层更高效
4. **数据规模决定基础 fanout**：大规模数据需要更大的 fanout
5. **数据分布影响最优 fanout**：聚类数据和高维数据需要更大的 fanout

### 8.2 快速参考表

| 数据规模 (n) | fanout_top | fanout_second | fanout_third | F_total | 适用场景 |
|-------------|-----------|--------------|-------------|---------|---------|
| < 1M        | 8         | 2            | 1           | 16      | 小规模实验 |
| 1M-10M      | 10        | 3            | 1           | 30      | 中等规模（默认）|
| 10M-100M    | 12        | 4            | 1           | 48      | 大规模 |
| ≥ 100M      | 15        | 4            | 2           | 120     | 超大规模 |

### 8.3 未来方向

1. **学习型 Fanout**：使用机器学习预测最优 fanout
2. **分区感知 Fanout**：根据每个分区的特征动态调整
3. **多目标优化**：同时优化召回率、构图时间和内存占用
4. **在线调整**：在搜索阶段根据查询反馈调整 fanout

---

**文档版本**：1.0
**最后更新**：2026-03-15
**作者**：ForgeANN Team
