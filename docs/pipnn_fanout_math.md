# PIPNN Fanout 数学推导与实验设计

## 1. 数学模型

### 1.1 符号定义

| 符号 | 含义 |
|-----|------|
| `n` | 数据集大小（点数）|
| `d` | 向量维度 |
| `R` | 目标最大出度（l_max）|
| `F_i` | 第 `i` 层的 fanout |
| `k_i` | 第 `i` 层的 leader 数量 |
| `c_avg` | 平均叶子大小 |
| `c_max` | 最大叶子大小 |
| `D` | 距离计算次数 |
| `T` | 构图时间（秒）|
| `M` | 内存占用（字节）|
| `r` | 召回率 |

### 1.2 距离计算量精确模型

#### 1.2.1 RBC 分区阶段

**第 0 层（顶层）**：

```
n₀ = n
k₀ = min(n × psamp₀ × scale₀, max_leaders)
D₀ = n × k₀
```

其中 `scale₀` 是自适应缩放因子（见 `params.rs:69-93`）。

**第 1 层**：

每个第 0 层的簇被进一步分区：

```
n₁ = n / k₀  (平均每个簇的大小)
k₁ = min(n₁ × psamp₁ × scale₁, max_leaders)
D₁ = n × F₀ × k₁
```

注意：
- 每个点被复制到 `F₀` 个簇中
- 每个簇内独立采样 `k₁` 个 leaders
- 总距离计算 = 点数 × 复制因子 × leaders

**第 2 层及以后**：

当 `fanout = 1` 时，不再有点复制：

```
D₂₊ = n × F₀ × F₁ × k₂
```

其中 `k₂` 是第 2 层的平均 leader 数（通常很小或为 0，因为很快达到叶子）。

**总 RBC 成本**：

```
D_rbc = D₀ + D₁ + D₂₊
      ≈ n × k₀ + n × F₀ × k₁
      ≈ n × (k₀ + F₀ × k₁)
```

#### 1.2.2 叶子处理阶段

**叶子数量**：

```
N_leaves = k₀ × k₁ × ... × k_depth
```

对于两层 fanout：

```
N_leaves ≈ k₀ × k₁
```

**每个点的叶子出现次数**：

```
N_replicas = F₀ × F₁ × ... × F_depth
```

对于两层 fanout：

```
N_replicas = F₀ × F₁
```

**平均叶子大小**：

```
c_avg = n × N_replicas / N_leaves
      = n × (F₀ × F₁) / (k₀ × k₁)
```

**叶内距离计算**：

每个叶子内部进行全对全距离计算（对称矩阵，只计算上三角）：

```
D_leaf_single = c_avg × (c_avg - 1) / 2
```

总叶内距离计算：

```
D_leaf = N_leaves × D_leaf_single
       = k₀ × k₁ × [n × F₀ × F₁ / (k₀ × k₁)]²/ 2
       = n² × (F₀ × F₁)² / (2 × k₀ × k₁)
```

**简化形式**：

当 `c_avg << n` 时（通常成立）：

```
D_leaf ≈ n × N_replicas × c_avg / 2
       = n × F₀ × F₁ × c_avg / 2
```

#### 1.2.3 总距离计算量

```
D_total = D_rbc + D_leaf
        = n × (k₀ + F₀ × k₁) + n × F₀ × F₁ × c_avg / 2
```

### 1.3 构图时间模型

#### 1.3.1 FLOP 计算

每次距离计算（L2 距离）需要的 FLOP：

```
FLOP_per_dist = 2d  (d 次乘法 + d 次加法)
```

总 FLOP：

```
FLOP_total = D_total × 2d
```

#### 1.3.2 GEMM 加速

叶内距离计算使用 GEMM（矩阵乘法）：

```
X ∈ ℝ^{c×d}  (叶内点的向量矩阵)
G = X × X^T   (Gram 矩阵)
D[i,j] = ||x_i||² + ||x_j||² - 2G[i,j]
```

GEMM FLOP：

```
FLOP_gemm = 2 × c² × d
```

相比逐对计算的加速比：

```
speedup = (c² × 2d) / (2 × c² × d) = 1
```

**注意**：理论上 FLOP 相同，但 GEMM 的实际加速来自：
1. **向量化**：AVX2/AVX512 SIMD 指令
2. **缓存优化**：矩阵分块，提高缓存命中率
3. **指令级并行**：CPU 流水线优化

实际加速比（经验值）：

```
speedup_actual ≈ 5-15x  (取决于 c 和 d)
```

#### 1.3.3 时间估算

**单线程时间**：

```
T_single = FLOP_total / FLOPS_per_core
```

其中 `FLOPS_per_core` 是单核峰值吞吐量：
- AVX2 f32：~50 GFLOPS
- AVX512 f32：~100 GFLOPS

**多线程时间**：

```
T_build = T_single / (threads × efficiency)
```

其中 `efficiency ∈ [0.7, 0.9]` 是并行效率（考虑锁竞争、负载不均等）。

**加上 overhead**：

```
T_total = T_build × (1 + overhead)
```

其中 `overhead ≈ 0.15`（内存分配、sketch 计算、图构建等）。

### 1.4 内存占用模型

#### 1.4.1 主要内存组件

**Sketch 存储**：

```
M_sketch = n × m × sizeof(f32)
         = n × 12 × 4 bytes  (默认 m=12)
         = 48n bytes
```

**HashPrune Reservoirs**：

```
M_reservoir = n × (struct_overhead + l_max × slot_size)
            = n × (64 + 64 × 8) bytes  (默认 l_max=64)
            = 576n bytes
```

其中 `slot_size = 8 bytes`（u32 + u16 + bf16）。

**Leaf Scratch（每个 worker）**：

```
M_scratch_per_worker = c_max² × sizeof(f32) + c_max × d × sizeof(f32)
                     = c_max² × 4 + c_max × d × 4
```

对于 `c_max=2048, d=768`：

```
M_scratch_per_worker ≈ 16 MB + 6 MB = 22 MB
```

**总内存（峰值）**：

```
M_peak = M_sketch + M_reservoir + M_scratch × threads
       = 48n + 576n + 22 MB × threads
       = 624n + 22 MB × threads
```

对于 `n=35M, threads=42`：

```
M_peak ≈ 21.8 GB + 0.9 GB ≈ 22.7 GB
```

#### 1.4.2 内存优化

**流式处理**：

当前实现使用流式 RBC（`rbc_partition_streaming`），避免同时存储所有叶子：

```
M_streaming = M_sketch + M_reservoir + M_scratch × threads + M_channel
```

其中 `M_channel` 是 channel 缓冲区（通常 < 100 MB）。

### 1.5 召回率模型

#### 1.5.1 概率分析

假设：
1. 簇分配近似均匀
2. 点的近邻在空间中随机分布（最坏情况）

**第 0 层相遇概率**：

点 `p` 被分配到 `F₀` 个簇（从 `k₀` 个簇中选择）。
点 `q` 被分配到 `F₀` 个簇。

相遇概率（至少在一个簇中相遇）：

```
P₀ = 1 - [(k₀ - F₀)! × k₀! / ((k₀ - F₀)! × (k₀ - F₀)!)] / [k₀! / (F₀! × (k₀ - F₀)!)]²
   ≈ 1 - (1 - F₀/k₀)^{F₀}  (近似)
   ≈ F₀²/k₀  (当 F₀ << k₀)
```

**第 1 层相遇概率**：

给定已在第 0 层相遇，在第 1 层再次相遇的概率：

```
P₁ ≈ F₁²/k₁
```

**总相遇概率**：

```
P_meet = P₀ × P₁
       ≈ (F₀²/k₀) × (F₁²/k₁)
       = (F₀ × F₁)² / (k₀ × k₁)
```

**召回率估计**：

假设每个点有 `K` 个真实近邻，每个近邻独立相遇：

```
Recall ≈ P_meet
       = (F₀ × F₁)² / (k₀ × k₁)
```

**实际召回率**：

由于：
1. 近邻通常在空间上聚集（不是随机分布）
2. HashPrune 机制提供额外的去重和过滤
3. 叶内 k-NN 选择最近的邻居

实际召回率通常高于理论值：

```
Recall_actual ≈ min(1, β × P_meet)
```

其中 `β ∈ [2, 5]` 是数据分布相关的增益因子。

#### 1.5.2 召回率与 Fanout 的关系

从上述模型可得：

```
Recall ∝ (F₀ × F₁)²
```

即召回率与 `F_total = F₀ × F₁` 的平方成正比（在小 fanout 范围内）。

但当 `F_total` 增大时，存在饱和效应：

```
Recall ≈ 1 - exp(-α × F_total²)
```

其中 `α` 取决于数据规模和分布。

### 1.6 最优 Fanout 推导

#### 1.6.1 优化目标

给定召回率约束 `r_target`，最小化构图时间：

```
minimize:   T_build(F₀, F₁)
subject to: Recall(F₀, F₁) ≥ r_target
            F₀, F₁ ∈ ℕ⁺
```

#### 1.6.2 拉格朗日方法

构造拉格朗日函数：

```
L(F₀, F₁, λ) = T_build(F₀, F₁) + λ × (r_target - Recall(F₀, F₁))
```

求偏导：

```
∂L/∂F₀ = ∂T_build/∂F₀ - λ × ∂Recall/∂F₀ = 0
∂L/∂F₁ = ∂T_build/∂F₁ - λ × ∂Recall/∂F₁ = 0
```

**时间对 Fanout 的导数**：

```
∂T_build/∂F₀ ≈ n × k₁ × 2d / FLOPS + n × F₁ × c_avg × d / FLOPS
∂T_build/∂F₁ ≈ n × F₀ × c_avg × d / FLOPS
```

**召回率对 Fanout 的导数**：

```
∂Recall/∂F₀ ≈ 2 × F₀ × F₁² / (k₀ × k₁)
∂Recall/∂F₁ ≈ 2 × F₀² × F₁ / (k₀ × k₁)
```

**最优条件**：

```
(∂T_build/∂F₀) / (∂Recall/∂F₀) = (∂T_build/∂F₁) / (∂Recall/∂F₁)
```

简化后：

```
(k₁ + F₁ × c_avg) / (2 × F₀ × F₁²) = (F₀ × c_avg) / (2 × F₀² × F₁)
```

进一步简化：

```
k₁ / F₁² + c_avg / F₁ = c_avg / F₀
```

**近似解**：

当 `k₁ << c_avg × F₁` 时：

```
F₀ ≈ F₁
```

即最优策略是让两层 fanout 相等。

但实际中，由于：
1. 第 0 层的 leader 数量 `k₀` 通常远大于 `k₁`
2. 第 0 层的距离计算成本更高（全局范围）

因此实践中：

```
F₀ > F₁
```

经验值：`F₀ ≈ 2-4 × F₁`。

## 2. 实验设计

### 2.1 实验目标

1. **验证理论模型**：实际构图时间和召回率是否符合理论预测
2. **确定最优 Fanout**：在不同数据规模和分布下的最优 fanout 组合
3. **评估自适应策略**：自适应 fanout 相比固定 fanout 的优势

### 2.2 数据集

#### 2.2.1 合成数据集

**均匀分布**：

```python
import numpy as np

def generate_uniform(n, d):
    return np.random.uniform(-1, 1, (n, d)).astype(np.float32)
```

**聚类分布**：

```python
def generate_clustered(n, d, n_clusters=100):
    centers = np.random.randn(n_clusters, d).astype(np.float32)
    labels = np.random.randint(0, n_clusters, n)
    data = centers[labels] + np.random.randn(n, d).astype(np.float32) * 0.1
    return data
```

**长尾分布**：

```python
def generate_longtail(n, d, alpha=2.0):
    # Zipf 分布控制簇大小
    cluster_sizes = np.random.zipf(alpha, 1000)
    cluster_sizes = cluster_sizes / cluster_sizes.sum() * n
    # ...
```

#### 2.2.2 真实数据集

| 数据集 | 规模 (n) | 维度 (d) | 分布特征 |
|-------|---------|---------|---------|
| SIFT1M | 1M | 128 | 图像特征，聚类 |
| GIST1M | 1M | 960 | 图像特征，高维 |
| Deep1B | 1B | 96 | 深度学习特征 |
| Text2Image1B | 1B | 200 | 多模态特征 |

### 2.3 实验变量

#### 2.3.1 自变量

1. **Fanout 组合**：
   - `fanout_top ∈ {5, 8, 10, 12, 15, 20}`
   - `fanout_second ∈ {1, 2, 3, 4, 5}`

2. **数据规模**：
   - `n ∈ {1M, 5M, 10M, 35M, 100M}`

3. **数据分布**：
   - Uniform, Clustered, Long-tail

4. **其他参数**：
   - `c_max ∈ {1024, 2048, 4096}`
   - `max_leaders ∈ {500, 1000, 5000, 10000}`

#### 2.3.2 因变量

1. **构图时间** (T_build)：秒
2. **召回率** (Recall@10, Recall@100)：百分比
3. **内存峰值** (M_peak)：GB
4. **平均出度** (Avg_degree)：个
5. **搜索 QPS**：queries/second

### 2.4 实验流程

#### 2.4.1 单点实验

对于每个参数组合 `(F₀, F₁, n, dist)`：

1. **生成/加载数据**
2. **构图**：
   ```bash
   cargo run -r -p build_forgeann_index -- \
       --data-path data.bin \
       --index-path index.bin \
       --fanout-top $F0 \
       --fanout-second $F1 \
       --c-max $CMAX \
       --max-leaders $LEADERS
   ```
3. **记录指标**：
   - 解析日志中的构图时间
   - 记录内存峰值（从 `/proc/self/status`）
4. **评估召回率**：
   ```bash
   cargo run -r -p search_velo_index -- \
       --index-path index.bin \
       --query-path queries.bin \
       --ground-truth gt.bin \
       --k 10
   ```
5. **保存结果**：
   ```json
   {
     "fanout_top": 10,
     "fanout_second": 3,
     "n": 10000000,
     "dist": "uniform",
     "build_time": 123.45,
     "recall_10": 0.956,
     "recall_100": 0.982,
     "memory_peak_gb": 6.2,
     "avg_degree": 62.3,
     "qps": 1234
   }
   ```

#### 2.4.2 批量实验

使用脚本自动化：

```python
import subprocess
import json
import itertools

# 参数网格
fanout_tops = [5, 8, 10, 12, 15, 20]
fanout_seconds = [1, 2, 3, 4, 5]
scales = [1_000_000, 10_000_000, 35_000_000]
dists = ["uniform", "clustered"]

results = []

for f0, f1, n, dist in itertools.product(
    fanout_tops, fanout_seconds, scales, dists
):
    print(f"Running: F0={f0}, F1={f1}, n={n}, dist={dist}")

    # 生成数据
    subprocess.run([
        "python", "generate_data.py",
        "--n", str(n),
        "--d", "768",
        "--dist", dist,
        "--output", f"data_{n}_{dist}.bin"
    ])

    # 构图
    result = subprocess.run([
        "cargo", "run", "-r", "-p", "build_forgeann_index", "--",
        "--data-path", f"data_{n}_{dist}.bin",
        "--index-path", f"index_{f0}_{f1}.bin",
        "--fanout-top", str(f0),
        "--fanout-second", str(f1)
    ], capture_output=True, text=True)

    # 解析结果
    build_time = parse_build_time(result.stdout)
    memory_peak = parse_memory_peak(result.stdout)

    # 评估召回率
    recall = evaluate_recall(f"index_{f0}_{f1}.bin")

    results.append({
        "fanout_top": f0,
        "fanout_second": f1,
        "n": n,
        "dist": dist,
        "build_time": build_time,
        "recall_10": recall,
        "memory_peak_gb": memory_peak
    })

# 保存结果
with open("experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### 2.5 结果分析

#### 2.5.1 可视化

**1. Fanout vs 召回率**：

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_json("experiment_results.json")

for n in df['n'].unique():
    subset = df[df['n'] == n]
    subset['f_total'] = subset['fanout_top'] * subset['fanout_second']

    plt.figure(figsize=(10, 6))
    plt.scatter(subset['f_total'], subset['recall_10'], alpha=0.6)
    plt.xlabel('F_total (fanout_top × fanout_second)')
    plt.ylabel('Recall@10')
    plt.title(f'Fanout vs Recall (n={n})')
    plt.grid(True)
    plt.savefig(f'fanout_vs_recall_n{n}.png')
```

**2. Fanout vs 构图时间**：

```python
plt.figure(figsize=(10, 6))
plt.scatter(subset['f_total'], subset['build_time'], alpha=0.6)
plt.xlabel('F_total')
plt.ylabel('Build Time (s)')
plt.title(f'Fanout vs Build Time (n={n})')
plt.grid(True)
plt.savefig(f'fanout_vs_time_n{n}.png')
```

**3. Pareto 前沿**：

```python
# 找到 Pareto 最优点（召回率高、时间短）
pareto = []
for i, row in subset.iterrows():
    dominated = False
    for j, other in subset.iterrows():
        if i != j:
            if (other['recall_10'] >= row['recall_10'] and
                other['build_time'] <= row['build_time'] and
                (other['recall_10'] > row['recall_10'] or
                 other['build_time'] < row['build_time'])):
                dominated = True
                break
    if not dominated:
        pareto.append(row)

pareto_df = pd.DataFrame(pareto)
plt.figure(figsize=(10, 6))
plt.scatter(subset['build_time'], subset['recall_10'],
            alpha=0.3, label='All')
plt.scatter(pareto_df['build_time'], pareto_df['recall_10'],
            color='red', s=100, label='Pareto Optimal')
plt.xlabel('Build Time (s)')
plt.ylabel('Recall@10')
plt.legend()
plt.title(f'Pareto Frontier (n={n})')
plt.savefig(f'pareto_n{n}.png')
```

#### 2.5.2 统计分析

**1. 回归分析**：

拟合召回率模型：

```python
from sklearn.linear_model import LinearRegression

X = df[['fanout_top', 'fanout_second', 'n']].values
y = df['recall_10'].values

# 对数变换
X_log = np.log(X)
y_logit = np.log(y / (1 - y))  # logit 变换

model = LinearRegression()
model.fit(X_log, y_logit)

print("Coefficients:", model.coef_)
print("R²:", model.score(X_log, y_logit))
```

**2. 方差分析 (ANOVA)**：

检验不同因素的显著性：

```python
from scipy import stats

# 检验 fanout_top 的影响
groups = [df[df['fanout_top'] == f]['recall_10'].values
          for f in fanout_tops]
f_stat, p_value = stats.f_oneway(*groups)
print(f"fanout_top: F={f_stat:.2f}, p={p_value:.4f}")
```

### 2.6 预期结果

#### 2.6.1 理论预测

基于前述模型，预期：

1. **召回率**：
   - `Recall ≈ 0.5 + 0.015 × F_total`（对于 n=10M）
   - 饱和点：`F_total ≈ 40-50`

2. **构图时间**：
   - `T_build ≈ 0.001 × n × F_total`（秒，42 线程）
   - 对于 n=10M, F_total=30：`T ≈ 300s`

3. **内存峰值**：
   - `M_peak ≈ 600n + 1GB`
   - 对于 n=10M：`M ≈ 7GB`

#### 2.6.2 最优 Fanout

预期最优组合（召回率 ≥ 95%）：

| n | 最优 F₀ | 最优 F₁ | F_total | 预期召回率 | 预期时间 |
|---|--------|--------|---------|-----------|---------|
| 1M | 8 | 2 | 16 | 96% | 16s |
| 10M | 10 | 3 | 30 | 95% | 300s |
| 35M | 12 | 4 | 48 | 96% | 1680s |
| 100M | 15 | 4 | 60 | 95% | 6000s |

## 3. 实现清单

### 3.1 代码修改

1. **添加 fanout_third 参数**：
   - 修改 `ForgeANNParams` 结构体
   - 更新 `fanout_for_depth` 方法

2. **实现自适应 fanout**：
   - 添加 `adaptive_fanout_for_depth` 方法
   - 在 `rbc_partition_streaming` 中调用

3. **添加实验工具**：
   - `cmd_drivers/experiment_fanout/`：批量实验脚本
   - `scripts/analyze_fanout.py`：结果分析脚本

### 3.2 实验脚本

1. **数据生成**：`scripts/generate_synthetic_data.py`
2. **批量构图**：`scripts/run_fanout_sweep.sh`
3. **结果分析**：`scripts/analyze_results.py`
4. **可视化**：`scripts/plot_results.py`

### 3.3 文档

1. **理论文档**：`docs/pipnn_fanout_theory.md`（已完成）
2. **数学推导**：`docs/pipnn_fanout_math.md`（本文档）
3. **实验报告**：`docs/pipnn_fanout_experiments.md`（待完成）

---

**文档版本**：1.0
**最后更新**：2026-03-15
**作者**：ForgeANN Team
