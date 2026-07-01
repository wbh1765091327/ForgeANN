# ForgeANN

ForgeANN is a Rust ANN graph-construction project centered on the production
ForgeANN OOM builder, with PiPNN/RBC partitioning retained as the core
algorithmic backbone. The main workflow is:

```text
base.fbin
  -> build_forgeann_index
  -> _mem.index
  -> search_mem_index for recall/quality checks
```

## Build

```bash
cargo build -r -p build_forgeann_index -p search_mem_index
```

For focused validation:

```bash
cargo check --workspace
cargo test -p build_forgeann_index
cargo test -p forgeann --test forgeann_oom_build_test
```

## Build A ForgeANN Graph

```bash
DATA_PATH=/path/to/base.fbin
INDEX_PREFIX=/path/to/index_dir
OOM_TMP=/path/to/tmp_dir

./target/release/build_forgeann_index \
  --data-type float \
  --dist-fn l2 \
  --data-path "$DATA_PATH" \
  --index-path-prefix "$INDEX_PREFIX" \
  --overwrite \
  --max-degree 64 \
  --num-threads 32 \
  --fanout-top 8 \
  --fanout-second 3 \
  --leaf-knn 2 \
  --oom-memory-budget-gb 16 \
  --oom-sketch-cache-gb 4 \
  --oom-spill-cache-gb 4 \
  --oom-temp-dir "$OOM_TMP" \
  --oom-keep-artifacts
```

The production default path is:

```text
D0 ADS
D1 ADS
D1 native resident-subtree
D1 materialize
D1 native subtree recurse
bounded async leaf drain
direct _mem.index graph write
no final RobustPrune by default
```

The current Wiki-scale quality profile uses `fanout_top=10`,
`fanout_second=3`, `leaf_knn=3`, `max_degree=64`, ADSampling, native D1
resident-subtree execution, leaf batch drain, and `--final-prune=false`.

## Check Graph Quality

```bash
QUERY_PATH=/path/to/query.fbin
GT_PATH=/path/to/gt_base_100
RESULT_PATH=/path/to/results.fbin

./target/release/search_mem_index \
  --index-path-prefix "$INDEX_PREFIX" \
  --data-path "$DATA_PATH" \
  --query-file "$QUERY_PATH" \
  --result-path "$RESULT_PATH" \
  --gt-file "$GT_PATH" \
  --recall-at 10 \
  --search-list-size 100 \
  --num-threads 12
```

`search_mem_index` is retained as a lightweight recall evaluator for `_mem.index`. Disk-index build/search, legacy DiskANN builders, and microbench-only drivers have been removed from the workspace.
