use std::mem::size_of;
use std::sync::atomic::AtomicUsize;

// ---------------------------------------------------------------------------
// Leaf / partition result enums
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub(crate) enum LeafReason {
    NaturalSize,
    MaxDepth,
    ShrinkRatio,
    SmallDeep,
    MinRecurse,
    PartitionFallback,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PartitionResult {
    SuccessFirst,
    SuccessRetry,
    FailedEmpty,
    FailedNoSplit,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RbcPhase {
    CurDedup,
    ClusterAssign,
    MergeClusters,
    MergedDedup,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub(crate) const FORCED_LEAF_HARD_CAP: usize = 8_000;
pub(crate) const FORCED_LEAF_SPLIT_RETRY_LIMIT: usize = 6;
pub(crate) const EXTERNAL_ASSIGN_DEPTH_LIMIT: usize = 1;
pub(crate) const EXTERNAL_CHILD_DEPTH_LIMIT: usize = 3;
pub(crate) const MIN_EXTERNAL_ASSIGN_POINTS_NON_ROOT: usize = 32 * 1024;
pub(crate) const RUN_EXTENT_HEADER_BYTES: usize = size_of::<u64>();
pub(crate) const MATERIALIZE_CHILD_BUFFER_POINTS: usize = 4096;
pub(crate) const MATERIALIZE_BATCH_POINTS: usize = 16_384;
pub(crate) static VECTOR_RUN_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// Run / child-run structs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub(crate) struct RunExtent {
    pub byte_offset: u64,
    pub len: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ChildRun {
    pub extents: Vec<RunExtent>,
    pub len: usize,
    pub seed: u64,
}
