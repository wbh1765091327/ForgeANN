# ForgeANN Agent Guide

## Project Scope

ForgeANN is a Rust ANN graph-construction and search project. The current
large-scale production driver is `build_forgeann_index`, with ADSampling
partitioning, native D1 resident-subtree execution, strict OOM point I/O,
bounded async leaf draining, leaf wavefront pair-mask ADS, and direct
point-store graph output.

Primary areas:

- `ForgeANN/src/forgeann/`: partitioning, OOM child-run I/O,
  resident-subtree execution, leaf scheduling.
- `ForgeANN/src/index/inmem_index/`: in-memory and OOM index build entry points.
- `cmd_drivers/build_forgeann_index/`: production ForgeANN graph-build driver.
- `cmd_drivers/search_mem_index/`: quality validation against `_mem.index`.
- `ForgeANN/tests/` and local benchmark/run logs: correctness and performance
  validation.

## Build And Test

Use the nightly toolchain from `rust-toolchain.toml`.

```bash
cargo build -r -p build_forgeann_index
cargo build -r -p search_mem_index
```

For focused validation, prefer:

```bash
cargo test resident_subtree
cargo test leaf_ads_wavefront_pairmask
cargo test adsampling
cargo test -p build_forgeann_index
```

Use `rg`/`rg --files` for repo search. Keep generated logs, run directories,
and benchmark artifacts outside tracked source unless the user asks for a
committed artifact.

## Current SOTA Full-Build Path

The expected from-scratch Wiki-scale path is:

```text
D0 ADS
D1 ADS
D1 native resident-subtree
D1 materialize
D1 native subtree recurse
bounded async leaf drain
direct point-store graph write
```

Legacy D1 checkpoint replay, reuse-countdown D2 replay, diagnostics-only flows,
and partial-run shortcuts are not part of the SOTA comparison path.

Expected log evidence:

```text
[adsampling/assign] depth=0
[adsampling/d1-sequential-ads-plan] depth=1
[adsampling/assign] depth=1
[adsampling/d1-resident-subtree-plan] depth=1
[adsampling/d1-resident-subtree] depth=1
[adsampling/d1-materialize] depth=1
[adsampling/d1-native-subtree-recurse] source_depth=1
[forgeann/scheduler-post-producer-leaf-drain]
ForgeANN lightweight OOM graph build done via direct point store
ForgeANN: graph build done
Exit status: 0
```

## SOTA Runtime Defaults

Use these defaults for current large-scale quality runs unless the user
explicitly asks for an ablation:

- `fanout_top=10`
- `fanout_second=3`
- `leaf_knn=3`
- `max_degree=64`
- `m_hash_bits=12`
- `leaf_size_min=256`
- `leaf_size_max=1280`
- `psamp_fraction=0.05`
- `max_leader_ratio=0.00015`
- leaf wavefront pair-mask ADS enabled
- leaf batch drain enabled
- D0/D1/leaf ADS epsilon and group dims matched across stages

Use PiPNN's leader sampling semantics, but do not hard-code the paper's small
leader cap for Wiki-scale and larger production runs. `PIPNN_P_SAMP=0.05`
controls candidate leader sampling. `MAX_LEADER_RATIO=0.00015` is the current
scalable hard-cap rule; change it only as a named ablation and record the D0
child-size distribution. A fixed `PIPNN_LEADER_HARD_CAP=1000` is useful as a
paper-reference ablation, but it produced root-like child distributions on
large MSMARCO-scale data.

```bash
PIPNN_P_SAMP=${PIPNN_P_SAMP:-0.05}
MAX_LEADER_RATIO=${MAX_LEADER_RATIO:-0.00015}
N=$(python3 - "$DATA_PATH" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    print(struct.unpack("<I", f.read(4))[0])
PY
)
MAX_LEADERS=$(python3 - "$N" "$MAX_LEADER_RATIO" <<'PY'
import math, sys
n = int(sys.argv[1])
ratio = float(sys.argv[2])
print(max(1, math.ceil(n * ratio)))
PY
)
```

Record `PIPNN_P_SAMP`, `MAX_LEADER_RATIO`, computed `MAX_LEADERS`, and the
observed D0 child-size distribution in the run directory.

For repeated experiments on the same dataset, reuse the known graph entry point
instead of paying the direct medoid scan every run. Keep it in a dataset-local
sidecar or run metadata and pass it explicitly:

```bash
GRAPH_START=${GRAPH_START:-}
GRAPH_START_ARG=()
if [ -n "$GRAPH_START" ]; then
  GRAPH_START_ARG=(--graph-start "$GRAPH_START")
fi
```

## ADS And Resident-Subtree Environment

Use stage-specific ADS knobs instead of global hidden disable switches:

```bash
export FORGEANN_D0_ADS_EPSILON=1.75
export FORGEANN_D1_ADS_EPSILON=1.75
export FORGEANN_LEAF_ADS_EPSILON=1.75
export FORGEANN_D0_ADS_GROUP_DIMS=64
export FORGEANN_D1_ADS_GROUP_DIMS=64
export FORGEANN_LEAF_ADS_GROUP_DIMS=64
```

Native D1 resident-subtree is production SOTA. Defaults are already enabled, but
set these explicitly for reproducible full-build runs:

```bash
export FORGEANN_D1_RESIDENT_SUBTREE_ENABLE=1
export FORGEANN_D1_RESIDENT_SUBTREE_GROUP_PIPELINE_SLOTS=2
export FORGEANN_D1_RESIDENT_SUBTREE_IO_THREADS=8
export FORGEANN_D1_RESIDENT_SUBTREE_MAX_RESIDENT_RATIO=0.20
export FORGEANN_D1_RESIDENT_SUBTREE_MIN_BUFFER_RATIO=0.70
```

Leaf batch drain is on by default. Use CLI flags for the production path:

```text
--leaf-ads-wavefront-pairmask-enable
--leaf-batch-drain-enable
```

Do not use removed or hidden switches such as `reuse-countdown-*`,
`FORGEANN_ASSIGNMENT_ADS_DISABLE`, or `FORGEANN_LEAF_ADS_DISABLE`.

## Final Prune

The current SOTA target is no final prune. `--final-prune=false` is the default
production path and writes `_mem.index` directly. Use `--final-prune=true` only
for named ablations of the older final RobustPrune path.

For no-prune runs, record binary help evidence for `--final-prune` and confirm
the build log contains:

```text
ForgeANN production OOM final RobustPrune disabled
```

The log must not contain `ForgeANN production final RobustPrune done`.

## Recommended Wiki-Scale Command Template

Use NVMe paths for data, index, OOM temp, logs, and command artifacts.

```bash
DATA_PATH=/nvme2/wiki/wiki_base.fbin
RUN_DIR=/nvme2/wiki/forgeann_10x3_lk3_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/index" "$RUN_DIR/oom"

PIPNN_P_SAMP=${PIPNN_P_SAMP:-0.05}
MAX_LEADER_RATIO=${MAX_LEADER_RATIO:-0.00015}
GRAPH_START=${GRAPH_START:-}
GRAPH_START_ARG=()
if [ -n "$GRAPH_START" ]; then
  GRAPH_START_ARG=(--graph-start "$GRAPH_START")
fi
N=$(python3 - "$DATA_PATH" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    print(struct.unpack("<I", f.read(4))[0])
PY
)
MAX_LEADERS=$(python3 - "$N" "$MAX_LEADER_RATIO" <<'PY'
import math, sys
n = int(sys.argv[1])
ratio = float(sys.argv[2])
print(max(1, math.ceil(n * ratio)))
PY
)

cat > "$RUN_DIR/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

/usr/bin/time -v env \\
  RUST_BACKTRACE=1 \\
  RUST_LOG=info \\
  FORGEANN_D0_ADS_EPSILON=1.75 \\
  FORGEANN_D1_ADS_EPSILON=1.75 \\
  FORGEANN_LEAF_ADS_EPSILON=1.75 \\
  FORGEANN_D0_ADS_GROUP_DIMS=64 \\
  FORGEANN_D1_ADS_GROUP_DIMS=64 \\
  FORGEANN_LEAF_ADS_GROUP_DIMS=64 \\
  FORGEANN_D1_RESIDENT_SUBTREE_ENABLE=1 \\
  FORGEANN_D1_RESIDENT_SUBTREE_GROUP_PIPELINE_SLOTS=2 \\
  FORGEANN_D1_RESIDENT_SUBTREE_IO_THREADS=8 \\
  FORGEANN_D1_RESIDENT_SUBTREE_MAX_RESIDENT_RATIO=0.20 \\
  FORGEANN_D1_RESIDENT_SUBTREE_MIN_BUFFER_RATIO=0.70 \\
  target/release/build_forgeann_index \\
    --data-type float \\
    --dist-fn l2 \\
    --data-path "$DATA_PATH" \\
    --index-path-prefix "$RUN_DIR/index" \\
    --overwrite \\
    --max-degree 64 \\
    --num-threads 32 \\
    --fanout-top 10 \\
    --fanout-second 3 \\
    --leaf-knn 3 \\
    --psamp-fraction "$PIPNN_P_SAMP" \\
    --max-leaders "$MAX_LEADERS" \\
    "\${GRAPH_START_ARG[@]}" \\
    --adsampling-rotation-sidecar "$DATA_PATH" \\
    --oom-memory-budget-gb 32 \\
    --oom-sketch-cache-gb 4 \\
    --oom-spill-cache-gb 4 \\
    --oom-temp-dir "$RUN_DIR/oom" \\
    --oom-keep-artifacts \\
    --oom-profile-json "$RUN_DIR/logs/oom_profile.json" \\
    --final-prune=false \\
    --leaf-ads-wavefront-pairmask-enable \\
    --leaf-batch-drain-enable \\
  > "$RUN_DIR/logs/build.log" 2>&1
EOF

bash "$RUN_DIR/command.sh"
```

For every serious run, keep:

```text
command.sh
logs/build.log
logs/oom_profile.json
index/_mem.index
```

## Stage Metrics To Record

For each full-build run, collect:

- D0 ADS `total_ms`, `epsilon`, `group_dims`, `recall_at_fanout`.
- D1 ADS `total_ms`, `epsilon`, `group_dims`, `recall_at_fanout`.
- D1 resident-subtree `hydrate_ms`, `build_ms`, grouped read amp, resident peak
  bytes, group pipeline slots, ready queue peak.
- D1 materialize `elapsed_ms`, `next_depth_runs`, `leaf_backlog_end`.
- Leaf ADS rows, full/pruned/group evals, wavefront pair-mask counters.
- Post-producer leaf drain `drain_ms`.
- `/usr/bin/time -v` wall time and max RSS.
- Final `_mem.index` size and `Exit status`.
- Quality from `search_mem_index` for the agreed `L` values.

## Engineering Practice

- Compare only true from-scratch full builds against true from-scratch full
  builds.
- Do not compare checkpoint replay, diagnostics-only flows, or partial outputs
  against full-build wall time.
- Preserve current SOTA defaults unless running a named ablation.
- Keep hot graph-construction paths allocation-conscious and concurrency-aware.
  Avoid per-point locks, global hot-path queues, and unnecessary hash maps in
  tight loops.
- Resident subtree and child processing should hydrate event-local/resident
  buffers. Partitioning and leaf build should consume resident/local data, not
  repeatedly query storage per point.
- When reporting results, state the exact path, command flags, env vars, run
  directory, wall time, stage times, RSS, final index size, and quality.
