# ViewLunePrune Handoff

Branch: `codex/view-lune-prune`

This branch adds `ViewLunePrune`, a default-off ForgeANN OOM pruning path that
uses resident/leaf-local lune witnesses to perform metadata-only final graph
selection. The implementation keeps the existing no-final, SpineOverlay, and
final RobustPrune paths available for comparison.

## What ViewLunePrune Does

ViewLunePrune compiles local lune witnesses during leaf build:

```text
source x, earlier candidate z, later candidate y
emit witness when d2(z, y) < d2(x, y) / 1.2
```

The final reducer scans bounded source candidates by `(source_dist2, dst)`.
A victim candidate is skipped only when one of its retained pivots has already
been selected for that source row. The final row is then refilled by distance
order to degree `R`. This preserves zero final vector I/O: pruning uses leaf
metadata and already-computed resident-local distances only.

## Public Flags

Default production behavior is unchanged unless these flags are set:

```text
--view-lune-prune-enable
--view-lune-oracle-json <path>
--view-lune-candidate-width-multiplier <f32>  # default 2.0, clamped [R,4R]
--view-lune-max-witnesses-per-victim <usize> # default 4
--view-lune-nearest-core <usize>             # default 0
```

Validation:

- L2 only.
- Mutually exclusive with final RobustPrune and SpineOverlay execution.
- Oracle-only mode writes the base no-final graph and report without changing
  graph output.

## Main Code Paths

- `ForgeANN/src/forgeann/view_lune_prune.rs`
  - Runtime accumulator, candidate reservoir, retained witness reservoir,
    reducer, oracle JSON report, and unit tests.
- `ForgeANN/src/forgeann/leaf_build.rs`
  - `PendingLuneWitness` and leaf witness emission from exact/full-matrix and
    ADS leaf paths.
- `ForgeANN/src/forgeann/scheduler.rs`
  - Tags leaves with stable route family and wraps the active leaf sink.
- `ForgeANN/src/forgeann/mod.rs`
  - Adds the ViewLune metadata-only OOM streaming path and recorder plumbing.
- `ForgeANN/src/index/inmem_index/forgeann_oom_build.rs`
  - Dispatches ViewLune OOM graph build, direct final graph writer, and oracle
    report handling.
- `cmd_drivers/build_forgeann_index/src/main.rs`
  - CLI flags, parameter forwarding, and flag validation tests.

## Engineering Optimizations Already Applied

The first working version had good search quality but too much build overhead.
This branch includes several rounds of overhead reduction:

- Executor-only path no longer stores the full no-final base graph unless an
  oracle report needs base overlap metrics.
- Final ViewLune rows stream directly into `DirectMemGraphWriter`; no
  all-row `Vec<Vec<u32>>` materialization.
- Candidate entries are reduced to `{dst:u32, dist2:f32}`.
- Retained witnesses are reduced to `{pivot:u32, rank_key:u32}`.
- Default retained witness storage is inline-4, avoiding millions of tiny heap
  vectors; heap fallback remains for `max_witnesses_per_victim > 4`.
- Retained witnesses dedup by pivot, because final pruning only needs to know
  whether a selected pivot exists.
- Candidate insertions keep rows ordered by destination and use binary search.
- Full rows cache the current worst candidate to avoid repeated O(R) scans for
  rejected candidates.
- Leaf flush now uses in-place recorder fast paths, avoiding temporary update
  vectors for candidate/witness records.
- Distinct witness family accounting uses cheap hot-path append and exact
  sort/dedup during snapshot.

## Validation Already Run

Focused tests/builds on this branch:

```bash
cargo test view_lune_prune --lib
cargo test -p build_forgeann_index view_lune
cargo test -p build_forgeann_index
cargo test spine_overlay --lib
cargo test final_prune --lib
cargo build -r -p build_forgeann_index
```

All passed before handoff.

## Prior Wiki5M Quality Results

Run root:

```text
/mnt/nvme0n1p1/enzyme/view_lune_wiki5m_eval_20260627_104153
/mnt/nvme0n1p1/enzyme/view_lune_wiki5m_eval_latest
```

Settings:

```text
Wiki5M, fanout=(10,3), leaf_knn=3, R=48
ViewLune executor enabled, oracle JSON enabled for the earlier run
Native DiskANN disk-search layout used for end-to-end disk comparison
```

Native DiskANN search summary:

```text
/mnt/nvme0n1p1/enzyme/view_lune_wiki5m_eval_latest/native_disk_search_repeats_summary.csv
```

Observed averages:

| Graph | L100 Recall | L100 QPS | L200 Recall | L200 QPS |
| --- | ---: | ---: | ---: | ---: |
| Forge no-final | 97.60 | 1745 | 98.30 | 1020 |
| Forge RobustPrune | 97.70 | 1802 | 98.40 | 1022 |
| ViewLunePrune | 97.90 | 1803 | 98.60 | 1021 |
| DiskANN official | 98.20 | 1868 | 98.40 | 1040 |

Interpretation: the search signal is positive. ViewLune improved recall over
no-final and RobustPrune while keeping QPS roughly stable in native DiskANN
disk search.

## Previous Build-Cost Problem

The earlier pre-optimization ViewLune build was not production-ready:

```text
no-final build wall:   about 8m19s
ViewLune build wall:   about 11m44s
ViewLune max RSS:      about 29.7GB
no-final max RSS:      about 23.0GB
```

That run included oracle reporting and was taken before the accumulator and
flush-path optimizations listed above. Re-measure with executor-only before
drawing the next conclusion.

## New-Machine Measurement To Run Next

Use an idle machine and NVMe paths. Do not run this while another large ANN
job is active; wall/RSS will be polluted.

Recommended executor-only build:

```bash
DATA_PATH=/mnt/nvme0n1p1/enzyme/wiki5m_data/wiki5m.fbin
RUN_DIR=/mnt/nvme0n1p1/enzyme/view_lune_wiki5m_executor_only_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/index" "$RUN_DIR/oom"

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

/usr/bin/time -v env \
  RUST_BACKTRACE=1 \
  RUST_LOG=info \
  FORGEANN_D0_ADS_EPSILON=1.75 \
  FORGEANN_D1_ADS_EPSILON=1.75 \
  FORGEANN_LEAF_ADS_EPSILON=1.75 \
  FORGEANN_D0_ADS_GROUP_DIMS=64 \
  FORGEANN_D1_ADS_GROUP_DIMS=64 \
  FORGEANN_LEAF_ADS_GROUP_DIMS=64 \
  FORGEANN_D1_RESIDENT_SUBTREE_ENABLE=1 \
  FORGEANN_D1_RESIDENT_SUBTREE_GROUP_PIPELINE_SLOTS=2 \
  FORGEANN_D1_RESIDENT_SUBTREE_IO_THREADS=8 \
  FORGEANN_D1_RESIDENT_SUBTREE_MAX_RESIDENT_RATIO=0.20 \
  FORGEANN_D1_RESIDENT_SUBTREE_MIN_BUFFER_RATIO=0.70 \
  target/release/build_forgeann_index \
    --data-type float \
    --dist-fn l2 \
    --data-path "$DATA_PATH" \
    --index-path-prefix "$RUN_DIR/index" \
    --overwrite \
    --max-degree 48 \
    --num-threads 32 \
    --fanout-top 10 \
    --fanout-second 3 \
    --leaf-knn 3 \
    --psamp-fraction "$PIPNN_P_SAMP" \
    --max-leaders "$MAX_LEADERS" \
    --graph-start 466692 \
    --adsampling-rotation-sidecar "$DATA_PATH" \
    --oom-memory-budget-gb 32 \
    --oom-sketch-cache-gb 4 \
    --oom-spill-cache-gb 4 \
    --oom-temp-dir "$RUN_DIR/oom" \
    --oom-keep-artifacts \
    --oom-profile-json "$RUN_DIR/logs/oom_profile.json" \
    --final-prune=false \
    --view-lune-prune-enable \
    --leaf-ads-wavefront-pairmask-enable \
    --leaf-batch-drain-enable \
  > "$RUN_DIR/logs/build.log" 2>&1
```

Important: do not pass `--view-lune-oracle-json` for the overhead measurement.
Oracle mode intentionally stores extra base-row comparison state.

Compare against the existing no-final baseline or rebuild a fresh no-final
baseline on the same machine. The gate is still:

```text
ViewLune wall <= no-final wall * 1.10
ViewLune RSS  <= no-final RSS  * 1.10
Recall/QPS should stay at least comparable to the prior native DiskANN result
```

## Known Current Blocker

At handoff time this machine was not suitable for a fair Wiki5M overhead run:

```text
process: neighbors
CPU:     about 30 cores
RSS:     about 152GB
load:    about 30+
```

That is why this branch includes code-level optimization and tests, but not a
fresh post-optimization Wiki5M build measurement.

