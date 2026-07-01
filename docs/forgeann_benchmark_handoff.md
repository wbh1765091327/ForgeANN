# ForgeANN Benchmark Handoff

This document is for a benchmark agent that needs to run ForgeANN graph
construction across datasets of different sizes without inheriting all of the
historical experiment-specific flags. It describes the current production
full-build path, how to choose parameters, what to log, and how to tell whether
a run is valid.

The current large-scale production path is:

```text
D0 ADS
D1 ADS
D1 resident-subtree
D1 materialize / native subtree child processing
bounded async leaf drain
final direct point store graph write
```

Do not call D1 checkpoint replay a full build. Checkpoint replay skips D0/D1
work and is useful for diagnosis, prefix validation, and regression checks, but
it is not comparable to from-scratch full-build wall time.

## Driver

Build the release binary first:

```bash
cargo build -r -p build_forgeann_index
```

Use:

```bash
target/release/build_forgeann_index
```

The input should usually be an fbin file with float vectors. Unless a dedicated
rotation sidecar exists, use the same data file for
`--adsampling-rotation-sidecar`.

## Core Defaults

These are the default choices for serious ForgeANN full-build benchmarks:

```text
distance:               l2
max_degree:             64
lbuild:                 100
m_hash_bits:            12
root assignment:        adsampling
top fanout:             6
second fanout:          2
leaf_knn:               2
ADS epsilon:            1.00
ADS group dims:         64
ADS depth max depth:    2
ADS root seed exact m:  8
ADS depth seed exact m: 8
ADS scheduler:          depth-wave
ADS wave depths:        1
ADS chunks per worker:  8
```

Keep `--fanout-top 6` and `--fanout-second 2` for mainline comparisons. Change
them only in a specific fanout study.

## Parameter Selection

Let:

```text
N            number of points
D            vector dimension
row_bytes    D * 4
data_gb      N * D * 4 / 2^30
RAM_GB       physical memory available to the benchmark process
T            worker threads, usually physical cores or the target CPU budget
```

### Thread Count

Use the intended CPU budget as `--num-threads`. For dedicated machines, use the
physical core count. Avoid oversubscribing BLAS/OpenMP runtimes:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_DYNAMIC=0
```

### Leaf Size

Use leaf sizes based on scale:

```text
N < 1M:       --leaf-size-min 64   --leaf-size-max 512
1M <= N < 10M --leaf-size-min 128  --leaf-size-max 1024
N >= 10M:     --leaf-size-min 256  --leaf-size-max 1280
```

For production large OOM runs, prefer `256..1280`. Smaller leaves can inflate
leaf count and edge-write overhead; much larger leaves can increase scratch RSS
and exact leaf build time.

### Leaders And Sampling

Choose a root leader cap that keeps D0 children large enough for D1/D2 to matter
but not so large that D0 bookkeeping dominates:

```text
target_root_leaders = clamp(N / 7000, 64, 5000)
max_leaders         = target_root_leaders
```

Use these starting `--psamp-fraction` values:

```text
N < 1M:        max(0.02, min(0.25, 4 * max_leaders / N))
1M <= N < 10M: 0.0100
N >= 10M:      0.0057
```

For very small smoke tests, it is fine if the sample fraction is high. For
large runs, keep `--max-leaders 5000` and `--psamp-fraction 0.0057` unless a
root fanout experiment explicitly changes it.

### OOM Memory Budget

ForgeANN OOM budget is an internal budget, not a hard RSS limit. RSS also
includes thread stacks, allocator retained pages, resident subtree buffers, sketch
state, and temporary exact-leaf scratch. Still, choose
the OOM budget as the main knob:

```text
oom_memory_budget_gb = clamp(0.30 * data_gb, 8, 0.55 * RAM_GB)
```

For small smoke tests, `1..4 GiB` is enough. For Wiki35M-like 768-dim data,
`32 GiB` is the current reference budget.

### Cache And I/O Budgets

Start from the OOM memory budget:

```text
strict_prefetch_budget_gb       = min(6.0, max(1.0, 0.20 * oom_memory_budget_gb))
sketch_cache_gb                 = min(4.0, max(0.5, 0.125 * oom_memory_budget_gb))
spill_cache_gb                  = min(4.0, max(0.5, 0.125 * oom_memory_budget_gb))
point_pipeline_budget_gb        = min(2.0, max(0.5, 0.0625 * oom_memory_budget_gb))
io_plan_window_cache_gb         = min(1.5, max(0.25, 0.05 * oom_memory_budget_gb))
```

Notes:

- Keep the resident-subtree memory cap at the production default unless an
  experiment explicitly compares cap pressure, grouped read amplification, RSS,
  and final graph wall time together.

### I/O Flags

Use strict OOM I/O for large benchmarks:

```text
--oom-enable
--oom-strict-io
--oom-strict-prefetch-pipeline
--oom-strict-prefetch-queue-depth 3
--oom-strict-prefetch-max-window-mb 16
--oom-strict-prefetch-max-gap-rows 4
--oom-point-pipeline
--oom-point-pipeline-io-threads 4
--oom-point-pipeline-queue-depth 32
--oom-point-pipeline-max-read-amplification 1.5
```

For very small smoke tests, OOM mode is still useful to exercise the same code
path, but use smaller budgets.

## Full-Build Command Template

Set these variables first:

```bash
DATA_PATH=/path/to/base.fbin
ROTATION_SIDECAR="$DATA_PATH"
RUN_ROOT=/nvme2/runs
RUN_NAME=forgeann_fullbuild_$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RUN_ROOT/$RUN_NAME"

THREADS=42
OOM_BUDGET_GB=32
SKETCH_CACHE_GB=4
SPILL_CACHE_GB=4
STRICT_PREFETCH_GB=6
POINT_PIPELINE_GB=2
WINDOW_CACHE_GB=1.5

LEAF_MIN=256
LEAF_MAX=1280
PSAMP=0.0057
MAX_LEADERS=5000
```

Then run:

```bash
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/index" "$RUN_DIR/oom"

cat > "$RUN_DIR/command.sh" <<EOF
/usr/bin/time -v env \\
  RUST_BACKTRACE=1 \\
  RUST_LOG=info \\
  OPENBLAS_NUM_THREADS=1 \\
  OMP_NUM_THREADS=1 \\
  MKL_NUM_THREADS=1 \\
  BLIS_NUM_THREADS=1 \\
  GOTO_NUM_THREADS=1 \\
  MKL_DYNAMIC=0 \\
  target/release/build_forgeann_index \\
    --data-type float \\
    --dist-fn l2 \\
    --data-path "$DATA_PATH" \\
    --index-path-prefix "$RUN_DIR/index" \\
    --overwrite \\
    --max-degree 64 \\
    --lbuild 100 \\
    --num-threads "$THREADS" \\
    --m-hash-bits 12 \\
    --seed 42 \\
    --leaf-size-min "$LEAF_MIN" \\
    --leaf-size-max "$LEAF_MAX" \\
    --psamp-fraction "$PSAMP" \\
    --fanout-top 6 \\
    --fanout-second 2 \\
    --leaf-knn 2 \\
    --max-leaders "$MAX_LEADERS" \\
    --forgeann-backend cpu \\
    --root-assignment-strategy adsampling \\
    --adsampling-epsilon 1.00 \\
    --adsampling-group-dims 64 \\
    --adsampling-rotation-sidecar "$ROTATION_SIDECAR" \\
    --adsampling-root-seed-exact-m 8 \\
    --adsampling-depth-enable \\
    --adsampling-depth-min-leaders 1 \\
    --adsampling-depth-min-points 1 \\
    --adsampling-depth-min-work 1 \\
    --adsampling-depth-max-depth 2 \\
    --adsampling-depth-seed-exact-m 8 \\
    --ads-scheduler depth-wave \\
    --ads-wave-depths 1 \\
    --ads-chunks-per-worker 8 \\
    --ads-fallback-exact-on-collapse \\
    --oom-enable \\
    --oom-strict-io \\
    --oom-strict-prefetch-pipeline \\
    --oom-strict-prefetch-queue-depth 3 \\
    --oom-strict-prefetch-budget-gb "$STRICT_PREFETCH_GB" \\
    --oom-strict-prefetch-max-window-mb 16 \\
    --oom-strict-prefetch-max-gap-rows 4 \\
    --oom-memory-budget-gb "$OOM_BUDGET_GB" \\
    --oom-sketch-cache-gb "$SKETCH_CACHE_GB" \\
    --oom-spill-cache-gb "$SPILL_CACHE_GB" \\
    --oom-temp-dir "$RUN_DIR/oom" \\
    --oom-keep-artifacts \\
    --oom-profile-json "$RUN_DIR/logs/oom_profile.json" \\
    --oom-point-pipeline \\
    --oom-point-pipeline-io-threads 4 \\
    --oom-point-pipeline-queue-depth 32 \\
    --oom-point-pipeline-budget-gb "$POINT_PIPELINE_GB" \\
    --oom-point-pipeline-max-read-amplification 1.5 \\
    --io-plan-min-depth 2 \\
    --io-plan-resident-budget-ratio 0.25 \\
    --io-plan-prefetch-streaming-enable \\
    --io-plan-small-run-wave-enable \\
    --io-plan-child-run-batching-enable \\
    --io-plan-window-cache-gb "$WINDOW_CACHE_GB" \\
    --io-plan-min-resident-points 4096 \\
    --io-plan-min-saved-wait-ms 1 \\
    --io-plan-min-saved-wait-per-gb-ms 100 \\
  > "$RUN_DIR/logs/build.log" 2>&1
EOF

bash "$RUN_DIR/command.sh"
```

Keep `command.sh`, `build.log`, and `oom_profile.json` for every serious run.

## Expected Log Evidence

A valid full build should contain these lines:

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

Invalid or incomplete runs:

- `_mem.index` is missing or zero bytes.
- The process exits with a nonzero status.
- The log only shows `level-scan/d1-checkpoint-replay` and skips D0/D1.
- The run is killed during or after RBC summary before final graph write.

## Metrics To Record

For each serious run, record:

```text
run_dir
git commit
command.sh
wall time from /usr/bin/time
Maximum resident set size
final _mem.index size
D0 ADS total_ms
D1 ADS total_ms
D1 materialize elapsed_ms, next_depth_runs, leaf_backlog_end
D1 resident-subtree hydrate_ms, build_ms, grouped read amp
D1 resident-subtree resident_peak_bytes
D1 native subtree leaf_backlog and active_leaf_drainers
post-producer leaf drain_ms
external reduce ms
oom_profile.json memory.rss_hwm_kb
```

Useful extraction commands:

```bash
rg -n \
  "adsampling/assign|adsampling/d1-resident-subtree|adsampling/d1-materialize|adsampling/d1-native-subtree-recurse|scheduler-post-producer-leaf-drain|Maximum resident|Exit status|ForgeANN: graph build done" \
  "$RUN_DIR/logs/build.log"

python3 - <<'PY' "$RUN_DIR/logs/oom_profile.json"
import json, sys
p = json.load(open(sys.argv[1]))
print("stage_ms", p.get("stage_ms"))
print("rss_hwm_kb", p.get("memory", {}).get("rss_hwm_kb"))
print("rss_by_stage", p.get("memory", {}).get("rss_by_stage"))
PY
```

## Validation Ladder

Use this order before spending a full large run:

```text
1. Release build.
2. Tiny OOM smoke test on a small fbin.
3. Checkpoint or prefix replay only for implementation sanity.
4. True full build on the target dataset.
```

Checkpoint replay is good for verifying that a code path runs, but do not use
its wall time as a full-build baseline.

## Current Wiki35M Reference

The current mainline Wiki35M full-build command uses the large-scale defaults:

```text
N = 35,167,920
D = 768
threads = 42
leaf size = 256..1280
top fanout = 6
second fanout = 2
max_leaders = 5000
psamp_fraction = 0.0057
oom_memory_budget_gb = 32
```

Recent pre-RSS-fix reference:

```text
graph build: about 2157s
wall: about 35:59
RSS HWM: about 53.9 GiB
```

Current full builds should be re-baselined around native D1 resident-subtree
hydrate/build timing, grouped read amplification, full wall time, and RSS
together.

## Cleanup Policy

For final benchmark records, keep:

```text
logs/build.log
logs/oom_profile.json
command.sh
index/_mem.index
```

Keep OOM temp artifacts only while debugging. Once a run is accepted, it is
safe to delete large `oom/` artifacts if logs and final index are preserved.
