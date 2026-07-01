# ForgeANN Out-of-Memory 向量图构建技术文档

本文面向研究人员，主线不是泛泛介绍 ANN，而是回答一个更具体的问题：

**在主存受限、数据规模很大时，ForgeANN 如何把向量图索引的构建过程改造成可执行、可扩展、尽量以顺序 I/O 为主的 out-of-memory 流水线。**

文档基于当前仓库中的实际实现整理，重点覆盖：

- 为什么 ForgeANN 当前选择 PiPNN 作为 OOM 构图主线；
- 代码中真正执行的构图流程是什么；
- sketch、分区、叶子构图、spill、reduce、图写出是如何串起来的；
- `_mem.index` 和 `_disk.index` 之间如何衔接；
- 当前已经观察到的构建结果、质量结果与工程边界。

文中尽量避免过多符号推导，但会在必要处解释设计背后的算法动机。

---

## 1. 问题背景：为什么“大规模向量图构建”会先撞上内存墙

高质量 ANN 图索引通常依赖两类能力：

- 让真正的近邻在构图时“相遇”；
- 在候选足够丰富的前提下，对每个点保留一小批质量高、方向多样的边。

如果数据集能完整放进内存，很多图构建算法都能工作，包括 HNSW、Vamana、DiskANN 这一类“边搜索边建图”的方法，也包括 PiPNN 这种“先分区、后局部构图”的方法。

但一旦数据规模上升到数千万点、每点数百维，问题就会从“浮点计算够不够快”转向“状态如何组织”：

- 基向量是否必须整体常驻堆内存；
- 候选边是否必须一直留在全局内存结构里；
- 最终图是否必须先完整物化到内存，再统一落盘；
- 构图过程中访问模式是顺序为主，还是由搜索路径驱动的大量随机访问。

对 OOM 构建来说，这些问题通常比单次距离计算本身更关键。

在 ForgeANN 当前代码里，核心判断很明确：

- 如果继续沿用 search-based incremental build，候选发现会强依赖当前图结构，外存场景下天然更容易退化成随机访问；
- 如果保留 PiPNN 的 partition-first 思路，则候选发现可以主要发生在叶子内部，距离计算可以块化，候选归并也更容易改造成外部化 reduce。

因此，ForgeANN 当前的 OOM 主线并不是把 Vamana 或 DiskANN 直接“落盘化”，而是围绕 PiPNN 的结构去做外存化。

---

## 2. 为什么 PiPNN 适合做 OOM 构图

### 2.1 PiPNN 的核心思想

PiPNN 的基本流程可以概括成三段：

1. 先做层次化分区，把全体点拆成许多局部叶子；
2. 在每个叶子内部做局部稠密候选发现；
3. 把来自多个叶子的候选边统一约简成最终邻接。

这和 Vamana / DiskANN 那种“对每个点做一次基于当前图的搜索，再把点插入图中”的路线本质不同。

PiPNN 的候选发现并不依赖当前图，而依赖分区结构本身。这个差异在 OOM 场景里非常重要，因为它意味着：

- 候选发现可以批处理；
- 叶子内部距离计算可以变成 GEMM 或分块 dense compute；
- 候选边可以先写成外部流，再做统一 reduce；
- 最终图不必在构图全过程中持续参与候选发现。

### 2.2 ForgeANN 看重的不是“PiPNN 理论很优雅”，而是三个工程性质

当前实现真正利用的是 PiPNN 的三个工程优势。

第一，**partition-first**。  
候选发现先由分区暴露，而不是由已有图上的 beam search 暴露。这使得构图更像规则化批处理，而不像随机游走。

第二，**leaf-local dense compute**。  
叶子内部的距离计算天然适合 L2 场景下的矩阵乘法，CPU 上可以直接走 GEMM/blocked GEMM 路线。

第三，**HashPrune 的顺序无关性可以支持 spill + reduce**。  
在 ForgeANN 的实现里，叶子产生的候选不会必须立即更新一个全局图结构，而是可以先写成候选 run，再按点归并到外部 HashPrune reducer 中。这是 OOM 化最关键的一步。

因此，PiPNN 在 ForgeANN 中并不是“另一个构图算法选项”，而是当前最容易被改造成外部化流水线的构图骨架。

---

## 3. ForgeANN 当前 OOM 构图主线的总览

当前代码中的主入口有两层。

第一层是命令行驱动。  
`build_forgeann_index` 负责解析参数、加载数据元信息、构造 `ForgeANNParams`，并最终生成 `_mem.index`。

第二层是索引内部入口。  
OOM 路径会调用一个“轻量化图构建到文件”的接口，核心行为是：

- 先只加载数据，不走默认的增量图构建流程；
- 在 PiPNN OOM 模式下直接把图写成 `_mem.index`；
- 后续如需要磁盘索引，再由 `build_disk_index` 从 `_mem.index` 构建 `_disk.index`。

从实现角度，当前流水线可以概括成：

1. 以 memory-mapped 方式打开 base vectors；
2. 直接从数据集生成并持久化 sketch 文件；
3. 运行 RBC 流式分区，并在递归过程中不断发射叶子；
4. 对每个叶子做局部候选发现；
5. 把候选边写入 spill run；
6. 以 shard 为单位做外部 HashPrune reduce；
7. 直接顺序写出 `_mem.index`；
8. 如需磁盘查询，再把 `_mem.index` 转成 `_disk.index`。

这条链路的关键点在于：

- `_mem.index` 仍然是当前系统的标准图制品；
- OOM 构建不是引入一套平行索引格式，而是换掉构图执行模型；
- 这样查询侧和磁盘布局侧就能复用现有实现。

---

## 4. 参数层：ForgeANN 如何控制 PiPNN 的 OOM 构图行为

### 4.1 参数对象的职责

当前 PiPNN 的主要超参数集中在 `ForgeANNParams` 中。它既包含算法参数，也包含 OOM 运行参数。

主要算法参数包括：

- `m_hash_bits`：HashPrune 使用的 sketch 位数；
- `l_max`：每点候选水库上限，也对应目标度数上界；
- `c_min` / `c_max`：叶子大小范围；
- `psamp_fraction`：RBC leader 采样比例；
- `max_leaders`：每层最多使用多少 leaders；
- `max_depth`：递归深度上限；
- `fanout_top` / `fanout_second`：顶层和次顶层的 overlap fanout；
- `leaf_knn`：叶子内部每点保留多少局部邻居候选；
- `final_prune` 与 `final_prune_alpha`：是否对输出图再做一遍最终剪枝。

OOM 相关参数包括：

- `oom_enable`：是否启用 OOM 路径；
- `oom_memory_budget_bytes`：当前暴露给用户的总预算参数；
- `oom_sketch_cache_bytes`：sketch 相关预算；
- `oom_spill_cache_bytes`：spill 缓冲预算；
- `oom_temp_dir`：外部中间文件目录；
- `oom_keep_artifacts`：是否保留中间产物；
- `oom_profile_json`：是否把 OOM profile 写成 JSON。

需要特别说明的一点是：

**当前这些“预算”参数更接近内部阶段控制参数，而不是进程 RSS 的严格硬上限。**

也就是说，ForgeANN 目前已经把主要中间状态做了外部化，但还没有形成真正严格的全局 RSS budget scheduler。

### 4.2 自动参数推荐的思路

`ForgeANNParams::recommend()` 不是简单给出一组经验值，而是做了一个构建预算反推。

当前实现的基本想法是：

- 用 `n × L × R` 近似 Vamana 一类方法的距离计算规模；
- 假设 GEMM 路径在吞吐上有一个保守的优势倍数；
- 把预算分配给 RBC 分区和叶内构图；
- 反推出 `c_max`、`max_leaders`、`psamp_fraction` 和 fanout。

对于 wiki35m 这类规模的数据，当前推荐逻辑明显偏向：

- 顶层 `fanout_top = 10`
- 第二层 `fanout_second = 2`

也就是常见的 `10x2` 配置，而不是更激进的 `10x3`。这在代码和历史实验文档里都被明确强化过，目的是在图质量和构建成本之间取得更稳定的平衡。

---

## 5. 数据访问层：OOM 构图从哪里真正开始

OOM 构图的第一步不是 spill，也不是 reduce，而是**停止在一开始就为整个数据集分配完整堆缓冲区**。

### 5.1 InmemDataset 的两种装载方式

当前 `InmemDataset` 同时支持两种模式：

- 常规模式：分配对齐堆内存，再把数据从文件拷入；
- memory-mapped 模式：不预分配完整 backing buffer，而是直接 mmap 数据文件。

OOM 路径会显式调用“只加载数据但避免预分配全量缓冲”的入口。其行为是：

1. 创建一个面向 memory-mapped 访问的数据集对象；
2. 对输入数据文件执行 `mmap`；
3. 后续按点读取向量时，直接从映射区域取对应切片。

这样做有两个直接效果：

- 构图不会在开始阶段就把全部 base vectors 复制进 Rust 堆对象；
- 数据集对象仍然保持与原有 PiPNN 代码兼容的 `get_vertex()` 接口，上层叶子构图逻辑不需要重写。

### 5.2 这一步仍然不是“严格 RSS 封顶”

需要区分两件事：

- “不在用户态堆上保留整份 base vectors 副本”；
- “进程 RSS 永远不大于某个给定预算”。

当前 ForgeANN 已经实现了前者，但还没有完全做到后者。原因是：

- memory map 仍然可能把热点页拉入页缓存；
- sketch、叶子 scratch、spill buffer、reduce reservoir slab 等仍然会占用一部分内存；
- Linux 页缓存与进程 RSS 的交互并不是一个简单的“只读不算内存”模型。

所以更准确的说法是：

**当前实现已经把数据访问层改造成了 mmap-first 的 OOM 友好路径，但系统级 strict-RSS 预算器还不是最终完成态。**

---

## 6. Sketch 阶段：为什么先把 sketch 外部化

PiPNN 的 HashPrune 需要 sketch。原始内存版路径会先为全部点计算 sketch，并把 sketch 矩阵留在内存中。

这在大数据集上很快会变成一块长期常驻的全局状态，因此 OOM 路径做了一个直接但关键的替换：

- sketch 不再先在内存中整体计算并保留；
- 而是直接从数据集逐点生成到磁盘文件；
- 然后重新打开为一个统一的只读 sketch store。

### 6.1 DiskSketchStore 的行为

当前 sketch 外部化组件的行为非常直白：

1. 先写一个文件头，记录行数和宽度；
2. 对每个点计算一行 sketch；
3. 把所有 sketch 行顺序写入磁盘；
4. 再通过 mmap 打开该文件，提供 `row_copy_into()` 之类的访问接口。

这意味着对上层而言：

- sketch 仍然看起来像一个“按行可读”的矩阵；
- 但它不再需要整块常驻堆内存。

### 6.2 为什么这一步值得做

研究上可以把 sketch 看作一种辅助编码；工程上它其实是 OOM 成败的分水岭之一。

因为如果不先处理 sketch，就会出现一种典型情况：

- 虽然候选边已经 spill 了；
- 最终图也不常驻了；
- 但全量 sketch 矩阵仍然在内存里。

这样 OOM 化只能算做了一半。

ForgeANN 现在的做法，是把 sketch 和基向量都改成“按需读取”的对象，让真正需要常驻的只剩下局部 scratch 和当前正在处理的状态。

---

## 7. RBC 流式分区：OOM PiPNN 的真正骨架

### 7.1 分区并不是先完整生成一棵树

当前 RBC 路径并不是“先构建完整分区树，再统一遍历树叶”，而是流式递归：

- 进入一个 cluster；
- 判断是否应停止递归；
- 如果停止，就直接发射叶子；
- 如果继续，就选 leaders、做 assignment、形成子簇；
- 再递归处理这些子簇。

这件事的工程意义很大，因为它避免了在中间阶段持有一整个巨大的树结构或全量叶子列表。

### 7.2 ForgeANN 的 RBC 终止规则

当前代码中，叶子生成不是单纯只看 `c_max`。会综合以下条件：

- 到达最大深度；
- cluster 大小已经落入可接受范围；
- 深层递归收缩比不再明显；
- cluster 已经很小，继续递归的收益不大；
- 分区失败或得到空簇，需要 fallback 叶化。

这使当前实现不是一个“死板的固定深度切分器”，而是一个更偏工程稳态的 RBC 递归器。

### 7.3 overlap fanout 在哪里发挥作用

PiPNN 图质量的关键之一，是顶层和次顶层允许一个点进入多个 child cluster。

当前实现按深度调节 fanout：

- 根层使用 `fanout_top`；
- 第二层使用 `fanout_second`；
- 更深层通常退化为 `1`。

这背后的直觉是：

- 真正有价值的 overlap 主要发生在靠近根部的几层；
- 深层如果仍然维持高 fanout，代价会快速膨胀；
- 顶层和次顶层带来的 candidate coverage 往往已经足够重要。

这也是为什么 `10x2` 这类配置会成为当前大规模数据集的主力选择。

### 7.4 为什么 RBC 在大规模上还能继续跑

代码中有两项很关键的工程设计。

第一，**cluster assignment 在 L2 下会切到 GEMM-based blocked assignment**。  
也就是说，大 cluster 面对很多 leaders 时，不会简单做逐点逐 leader 的标量比较，而会把点块和 leader 块组织成矩阵乘法。

第二，**上层 child runs 可以在早期深度外部化**。  
当前实现允许在前几层把 child cluster 的点 ID 写成 run 文件，再按需读回递归，而不是所有 child 都必须保存在内存向量里。

这两点结合起来，使 RBC 在大数据上仍然有机会保持：

- 受控的活跃内存；
- 规则化的计算块；
- 不必把所有子簇对象同时留在堆里。

### 7.5 大叶子的保护机制

当前代码里还有一个很实用的保护：对过大的叶子会强制做安全分裂，而不是任由超大叶子直接进入后续叶内全矩阵路径。

这件事的价值并不在于数学优雅，而在于它避免了一个非常现实的问题：

- 局部叶子如果失控变得太大；
- 后续叶内 scratch 和两两距离计算会突然变成构图过程中的内存和耗时尖峰。

因此，RBC 阶段本身已经内含了一个“不要让下游叶子阶段崩掉”的工程保护层。

---

## 8. 调度器：为什么 RBC 和叶子构图能同时推进

### 8.1 单线程池承载 producer 和 leaf workers

当前 PiPNN 主构图不是“一个线程池做分区，另一个线程池做叶子”，而是统一调度：

- RBC 递归是 producer；
- 叶子构图是 consumer；
- 两者共用同一个受限 Rayon pool。

这件事有两个直接好处：

- `--num-threads` 真正成为整个构图阶段的线程上限，而不是多个线程池叠加后的理论值；
- 不会出现 BLAS/GEMM 线程、RBC 线程、叶子线程彼此叠加导致的严重 oversubscription。

### 8.2 为什么要强制把 BLAS/OpenMP 设成单线程

PiPNN 的很多计算最终会落到 GEMM 上。如果不控制底层 BLAS/OpenMP 线程数，就可能出现：

- Rayon 已经开了 `N` 个 worker；
- 每个 worker 内部又让 BLAS 再开 `M` 个线程；
- 最终实际 runnable threads 远超过机器核数。

当前实现因此在主阶段开始前显式把常见 BLAS/OpenMP 环境变量钉到单线程。这种处理对研究人员来说看似“只是实现细节”，但它直接决定了大规模构图是否会因为线程过量而性能反而变差。

---

## 9. 叶子构图：局部 dense compute 如何变成候选边

### 9.1 小叶子与大叶子的两条路径

当前叶子构图对不同规模叶子采用两条不同路径。

对于不太大的叶子：

- 直接在叶子内部构造局部矩阵；
- L2 情况下走 `X * X^T` 的 Gram 路径；
- 再还原成距离矩阵；
- 对每一行选出 `leaf_knn` 个局部近邻。

对于更大的叶子：

- 改用 blockwise 的精确 top-k 路径；
- 避免一次性分配完整 `|leaf| × |leaf|` 距离矩阵。

这个切换点由 `max_full_matrix_leaf_size` 控制。

### 9.2 L2 场景为什么适合 GEMM

对向量图构建来说，最昂贵的操作之一是成批距离计算。L2 下可以利用恒等式：

- 距离平方 = 向量范数和减去两倍点积。

所以叶子内部不必逐对显式算距离，而可以先算 Gram 矩阵，再恢复距离。这就是当前代码中大量使用 `general_mat_mul` 和块状 Gram tile 的原因。

从系统角度看，这一步的重要性在于：

- 计算模式规则；
- 内存访问连续；
- 更接近高吞吐的 dense linear algebra，而不是图搜索中的随机访问。

### 9.3 候选边是如何产生的

当前叶子阶段对每个点做的事情可以简化为：

1. 找出叶子内部 top-`k` 近邻；
2. 对每个候选 `(p, c)` 计算 `p -> c` 的 sketch hash；
3. 同时也生成反向 `c -> p` 候选；
4. 先放入一个叶子局部的边缓冲；
5. 当缓冲足够大时，批量 flush 到下游 sink。

这里有两个关键点。

第一，ForgeANN 当前保留的是 **bi-directed local candidate generation**。  
也就是局部邻居不是只保留单向边，而是会把双向候选都送入后续 HashPrune。

第二，边不是立刻写入全局图。  
这一点是 OOM 化的根基。如果每条边一产生就要立即更新全局结构，外存流水线就很难成立。

---

## 10. Spill：为什么候选边先写成外部 run

### 10.1 Spill 的角色

在内存版 PiPNN 中，候选边通常会进入每个点对应的常驻 reservoir。  
在 OOM 路径中，这样做会把“每点一个 reservoir”的状态长期留在内存里，因此当前实现改成：

- 叶子先把候选边写入 spill sink；
- spill sink 有一个有界缓冲；
- 达到阈值后，把候选写成一个 run 文件。

### 10.2 Run 文件里存的是什么

当前 spill record 的逻辑内容大致是：

- 源点 `p`；
- 目标点 `c`；
- 哈希值 `hash`；
- 距离 `dist`。

这些记录在 flush 前会按源点排序，然后顺序写出。因此每个 run 文件本质上是一个按源点近似有序的候选流。

### 10.3 这样做的真正收益

Spill 的价值不是“反正能落盘”，而是把候选处理从：

- 在线更新全局状态

改成：

- 先生成外部候选流；
- 再离线归并和约简。

这使得后续 reducer 可以只关注当前 shard，而不是同时为所有点保持活跃状态。

---

## 11. External HashPrune Reduce：OOM 构图真正成立的地方

### 11.1 reducer 的输入不是一张图，而是一组 run 文件

当前外部 reducer 的输入是 spill 阶段产生的多个 run 文件。  
这些 run 文件会通过多路归并按点推进。

具体行为是：

1. 打开所有 run 文件；
2. 对每个文件读出当前最小点号的候选；
3. 通过小顶堆或等价结构按源点顺序归并；
4. 把同一 shard 内同一点的候选送入该点的 HashPrune reservoir；
5. 当某个点的输入结束时，立刻导出其邻接。

### 11.2 为什么 reduce 可以按 shard 做

因为当前实现只要求：

- 同一源点的候选最终都能流经它自己的 reservoir；
- reservoir 的最终结果只依赖候选集合，而不依赖它们的到达顺序。

这样一来，reducer 就不需要：

- 为全部点同时维护全局状态；
- 也不需要等到所有 run 全部读完，才开始生成最终图。

它可以只为当前 shard 分配一块 reservoir slab，处理完这个 shard 后清空复用。

这也是当前 OOM 路径能把最终图逐点输出到 `_mem.index` 的基础。

### 11.3 多 run 文件过多时怎么办

当前实现已经考虑到打开 run 文件数量过多的问题。

如果 run 文件数超过打开文件预算，reducer 会先进行分轮 compaction：

- 把一批 run 文件合并成一个新的 run；
- 再继续下一轮；
- 直到活跃文件数落到可接受范围，再做最终 reduce。

这说明当前系统已经意识到 OOM 不只是“内存不够”，还包括：

- 文件句柄预算；
- 归并 fan-in；
- spill 文件组织方式。

但这一步仍然不是最终最优状态。它更像一个已经可工作的外存 reducer，而不是严格最优的 source-shard segment spill 设计。

---

## 12. 直接写 `_mem.index`：为什么最终图不必先放进 `final_graph`

### 12.1 当前 OOM 路径的两种终态

当前 PiPNN 图构建对终态输出有两种情况。

如果 `final_prune = false`：

- OOM 构图可以直接边 reduce 边写 `_mem.index`；
- 不需要先填满 `final_graph`。

如果 `final_prune = true`：

- 系统会先把 PiPNN 输出放回 `final_graph`；
- 运行一次 per-node final prune；
- 再统一写出 `_mem.index`。

因此，只有第一种情况才是严格意义上的“完全流式终态输出”。

### 12.2 DirectMemGraphWriter 的作用

当前有一个专门的直接图写出器，负责把邻接顺序写成 `_mem.index`。

它的工作方式非常简单：

1. 先写一个占位头部；
2. 每个点写出自己的度数和邻接列表；
3. 构建结束时回填：
   - 文件总大小；
   - 观察到的最大度数；
   - 搜索起点；
   - frozen point 数量。

这一步的重要性在于：

- `_mem.index` 仍然保持与现有加载器兼容；
- 图构建阶段不需要额外引入新的终态文件格式；
- OOM 路径能直接和后续 `build_disk_index` 接起来。

---

## 13. 从 `_mem.index` 到 `_disk.index`：ForgeANN 如何接上磁盘检索

### 13.1 整体关系

ForgeANN 当前的 OOM 图构建并不直接生成磁盘查询格式，而是先生成 `_mem.index`，然后由单独的磁盘布局构建器把它转成 `_disk.index`。

所以完整链路是：

1. `build_forgeann_index` 生成 `_mem.index`；
2. `build_disk_index` 读取现有 `_mem.index`；
3. 把图和向量组织成页式磁盘布局；
4. `search_disk_index` 再对 `_disk.index` 执行搜索。

### 13.2 当前磁盘布局的重点变化

仓库近期一个很重要的变化，是默认磁盘布局已经切换到：

- `_disk.index` 默认写**全精度向量记录**；
- 不再默认生成 4-bit RabitQ ext-code 作为主数据布局的一部分；
- 仍然保留 1-bit RabitQ sidecar，用作兼容量化辅助数据。

从工程上看，这个变化有两个意义。

第一，它把“图构建正确”和“额外量化压缩”解耦了。  
也就是说，现在默认磁盘布局更像是：

- 先把图和 full-precision vector page 正确组织好；
- 再把 1-bit sidecar 作为附加信息，而不是强制依赖 4-bit 码本路径。

第二，它让 `search_disk_index` 的默认读取路径也更直接。  
当前默认是走全精度 `MmapIndex` 路径，而不是默认假定磁盘页中一定含有 4-bit RabitQ 扩展块。

### 13.3 当前 `_disk.index` 的记录组织

当前磁盘布局构建器的核心行为可以概括为：

- 读取 `_mem.index` 里的图邻接；
- 逐点从 base data 文件中读出原始向量；
- 把向量对齐到页式布局需要的 padded 维度；
- 为每个点写一个固定 record：
  - 全精度向量；
  - 邻居数量；
  - 固定上限长度的邻接列表。

同时还会生成：

- `id -> page` 的映射文件；
- RabitQ sidecar 数据文件。

因此，当前 ForgeANN 磁盘索引的生成过程，本质上是把“已经构好的图”重新编码成一个页友好的磁盘表示，而不是在磁盘布局阶段重新构图。

---

## 14. 查询侧与磁盘布局的关系

当前 `search_disk_index` 使用的默认读取对象是 `MmapIndex`。

在全精度路径下，它会：

- 读取 `_disk.index` 元数据；
- 理解固定 record 大小和每页能放多少条记录；
- 通过页缓存和 `MmCache` 管理磁盘页；
- 直接按全精度向量记录进行搜索。

这意味着对于研究人员来说，可以把 ForgeANN 当前系统理解成两段：

- 前半段：PiPNN OOM 构图，目标是生成高质量 `_mem.index`；
- 后半段：磁盘布局和磁盘搜索，目标是把这张图变成适合页缓存和 NVMe 的查询格式。

这种分层很有价值，因为它让“构图算法”和“磁盘页组织策略”可以分别演进。

---

## 15. 当前已观察到的结果

这一节只总结当前仓库内已有文档和实现说明中能追溯的结果，不把无法复核的临时口头观察混进正式技术文档。

### 15.1 Gist1m 上的端到端 OOM 结果

现有技术报告里已经给出了一个完整、可查询的端到端对比：

- PiPNN OOM 先生成 `_mem.index`；
- 再生成 `_disk.index`；
- 最后和官方 DiskANN 做构建及搜索对比。

报告记录的核心结论是：

- PiPNN OOM 图构建到 `_mem.index` 约 2 分 16 秒；
- 磁盘布局阶段约 11 分 33 秒；
- 端到端得到可查询磁盘索引约 13 分 49 秒；
- 图构建阶段峰值 RSS 约 7.25 GiB；
- 在 Gist1m 的 `L=50` 和 `L=100` 上，PiPNN OOM 获得了更高的 Recall@10 和更高的 QPS。

这组结果至少说明一件事：

**当前这条 OOM 构图路径不是“能跑通测试”的原型，而是一条已经走通过真实数据集、真实查询流程的完整工程链路。**

### 15.2 wiki35m 上的近期构图结果

仓库中另有一份基准结果整理，记录了多组 wiki35m 配置的构图与后续 pipeline 表现。

这些实验展示出几个稳定趋势。

第一，`fanout_top` 从 10 降到 9、8 时，构图时间会下降。  
第二，随着 fanout 下降，RBC 产生的叶子总点数、叶子数量和 oversized leaf 事件也会下降。  
第三，构图更快不等于最终效果更好。已经完成完整 pipeline 的结果显示：

- `f10s2` 比 `f9s2` 的召回和延迟更稳；
- `auto-params` 基线在完整 pipeline 上仍然是最稳妥的一组。

这说明 ForgeANN 当前对大规模数据集的主要经验是：

- overlap 的确有价值；
- 但 overlap 不是越大越好，也不是越小越省时就越划算；
- `10x2` 和自动参数推荐的稳定性，是当前代码和实验共同支持的结论。

### 15.3 “OOM”在当前实现里的准确含义

从现有报告和代码状态看，当前 ForgeANN 的 OOM 更准确地表示：

- 主要中间状态已经外部化；
- 不要求全量 sketch、全量候选、全量最终图同时常驻内存；
- 构图可以在大数据集上顺序推进并落盘。

它**不等价于**：

- 进程 RSS 永远被严格卡死在某个小预算之下。

这是当前系统需要清楚区分的一个边界。

---

## 16. 当前实现的主要优点

把代码和结果综合起来，ForgeANN 当前 OOM 构图主线有五个明显优点。

### 16.1 它保留了 PiPNN 最适合外存化的部分

也就是：

- 分区先行；
- 叶内局部 dense compute；
- HashPrune 统一约简。

这使得系统避免走到 search-based graph build 的随机访问瓶颈上。

### 16.2 它已经形成了清晰的外存流水线

这条流水线不是停留在概念层，而是已经具备：

- memory-mapped dataset；
- disk-backed sketch；
- spill run；
- external reduce；
- direct `_mem.index` writer。

### 16.3 它和现有索引生态兼容

OOM 构图输出的不是新格式，而是现有 `_mem.index`。  
这使得后续磁盘布局和查询路径都可以复用，不需要重新发明一套完整的查询系统。

### 16.4 它已经具备研究迭代的良好插点

未来想继续研究：

- 更严格的 memory budget；
- 更少的小文件 spill 组织；
- 更强的 backbone / bridge edge；
- strict-RSS 执行模型；

都可以在当前骨架上继续推进，而不是推倒重来。

### 16.5 它已经在真实数据上证明了“不是空想”

Gist1m 的端到端结果说明：

- 这条路径已经能跑完；
- 最终索引可查；
- 在当前实验口径下质量和吞吐都具备竞争力。

---

## 17. 当前实现的局限与研究空间

### 17.1 预算还不是严格的全局 RSS 控制

当前虽然暴露了 OOM memory budget 参数，但并没有一个统一的 budget scheduler 去精确约束：

- point tile；
- leader panel；
- spill 缓冲；
- reduce reservoir slab；
- 活跃叶子 scratch；
- 页缓存引起的驻留。

因此，当前系统更像“外部化主状态的 OOM builder”，而不是“严格内存上界 builder”。

### 17.2 spill 仍然可以更接近 source-shard 组织

当前 spill 方案已经能工作，但它仍然主要是：

- 生成一批 run 文件；
- 再做归并与 compaction。

后续如果进一步走向 strict-RSS 和更低文件数，理想方向会是：

- 在边产生时直接按 source shard 组织 spill；
- reducer 直接消费 shard segments；
- 尽量减少全局 compaction 轮数。

### 17.3 partition child 的外部化还不算最终形态

RBC 现在已经支持上层 child run 外部化，这很关键。  
但如果继续追求文件数下降和更加稳定的 OOM 行为，未来更自然的方向会是：

- append-only run store；
- extent 管理；
- 不再让 child-run 过度依赖 file-per-child 语义。

### 17.4 final prune 的严格外存化还有空间

当前 `final_prune = false` 时可以保持完全流式图写出；  
一旦开启 `final_prune`，系统仍然需要回到 `final_graph` 这一内存结构。

这意味着：

- 当前已经有可用的 OOM 主线；
- 但“完全外存化 final prune”仍是一个值得继续做的研究问题。

### 17.5 图质量增强仍可继续研究

从现有文档来看，当前系统主要优先保证的是：

- 构图可执行；
- 图质量恢复到稳定可用；
- 端到端能连接到磁盘查询。

而诸如：

- 更强的长程导航边；
- 更稳定的 full-scale recall；
- 更少候选下的同等质量；

仍然是后续可以继续优化的方向。

---

## 18. 对研究人员的总结

如果只用一句话概括 ForgeANN 当前的 OOM 向量图构建路线，可以说：

**它不是把传统 search-based 图构建强行“搬到磁盘上”，而是围绕 PiPNN 的 partition-first 结构，把 sketch、候选和最终图这三类原本全局常驻的状态，逐步改造成可流式、可外部化、可分阶段归并的构建流水线。**

具体来说：

- 基向量通过 memory map 按需读取；
- sketch 直接持久化成磁盘矩阵；
- RBC 递归流式产生叶子；
- 叶子内部用 dense compute 暴露局部候选；
- 候选边先 spill 成外部 run；
- HashPrune 再按 shard 做外部 reduce；
- 最终邻接直接写成 `_mem.index`；
- 磁盘布局阶段再把 `_mem.index` 重编码成 `_disk.index`。

这条路线的价值不只是“能省一点内存”，而是它把原本很难在 OOM 场景下稳定执行的向量图构建，改造成了一条更接近顺序 I/O 和局部 dense compute 的系统流程。

从当前仓库中的结果看，它已经具备三层意义：

- 在实现上，它已经是完整可运行的；
- 在系统上，它已经是外存友好的；
- 在实验上，它已经不是纸面设计，而是拿到了真实数据上的端到端结果。

如果后续继续推进，最值得研究的方向不是“是否继续做 OOM”，而是：

- 如何把当前 partial externalization 进一步推进到 strict-RSS；
- 如何把 spill 和 partition run 从“能工作”优化到“更少文件、更稳上界、更高吞吐”；
- 如何在不牺牲 OOM 可执行性的前提下继续提升图质量。

这也是 ForgeANN 当前这条 OOM 构图主线的真正研究价值所在。
