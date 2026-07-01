use std::fmt;
use std::str::FromStr;

mod aligned_allocator;
pub use aligned_allocator::AlignedBoxWithSlice;

mod ann_result;
pub use ann_result::*;

/// Distance metric
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Metric {
    /// Squared Euclidean (L2-Squared)
    L2,

    /// Cosine similarity
    /// TODO: T should be float for Cosine distance
    Cosine,

    /// Inner Product similarity
    Ip,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseMetricError;

impl fmt::Display for ParseMetricError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid metric")
    }
}

impl FromStr for Metric {
    type Err = ParseMetricError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "l2" => Ok(Metric::L2),
            "cosine" => Ok(Metric::Cosine),
            "ip" => Ok(Metric::Ip),
            _ => Err(ParseMetricError),
        }
    }
}
