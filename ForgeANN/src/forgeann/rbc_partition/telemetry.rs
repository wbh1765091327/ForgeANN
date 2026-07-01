use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct RootFanoutProfile {
    pub(crate) policy: String,
    pub(crate) selection_semantics: String,
    pub(crate) enabled: bool,
    pub(crate) fixed_fanout: usize,
    pub(crate) total_points: usize,
    pub(crate) kept_hist: Vec<usize>,
    pub(crate) avg_fanout: f64,
    pub(crate) selector_ms: u128,
    pub(crate) projected_assignment_bytes: usize,
    pub(crate) projected_partition_d00_bytes_pre_dedup: usize,
    pub(crate) root_leader_hash: u64,
}

impl Default for RootFanoutProfile {
    fn default() -> Self {
        Self {
            policy: "fixed".to_string(),
            selection_semantics: "fixed_root_fanout_v1".to_string(),
            enabled: false,
            fixed_fanout: 0,
            total_points: 0,
            kept_hist: Vec::new(),
            avg_fanout: 0.0,
            selector_ms: 0,
            projected_assignment_bytes: 0,
            projected_partition_d00_bytes_pre_dedup: 0,
            root_leader_hash: 0,
        }
    }
}

impl RootFanoutProfile {
    pub(crate) fn fixed(fanout: usize) -> Self {
        Self {
            enabled: true,
            fixed_fanout: fanout,
            kept_hist: vec![0; fanout.max(1) + 1],
            ..Self::default()
        }
    }

    pub(crate) fn observed(&self) -> bool {
        self.fixed_fanout > 0
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RbcPartitionTelemetry {
    pub(crate) root_fanout: RootFanoutProfile,
    pub(crate) assignment_decisions: Box<AssignmentDecisionProfile>,
    pub(crate) ads_scheduler: AdSamplingSchedulerStats,
    pub(crate) io_planned_forgeann: IoPlannerStats,
    pub(crate) point_pipeline: PointPipelineStats,
}

impl RbcPartitionTelemetry {
    pub(crate) fn merge(&mut self, other: Self) {
        self.assignment_decisions.merge(*other.assignment_decisions);
        self.ads_scheduler.merge(other.ads_scheduler);
        self.io_planned_forgeann.merge(other.io_planned_forgeann);
        self.point_pipeline.merge(other.point_pipeline);
        if other.root_fanout.observed() {
            self.root_fanout = other.root_fanout;
        }
    }
}

fn finite_json(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

pub(crate) fn root_fanout_profile_json(profile: &RootFanoutProfile) -> String {
    let value = json!({
        "root_fanout": {
            "policy": profile.policy,
            "selection_semantics": profile.selection_semantics,
            "enabled": profile.enabled,
            "fixed_fanout": profile.fixed_fanout,
            "total_points": profile.total_points,
            "kept_hist": profile.kept_hist,
            "avg_fanout": finite_json(profile.avg_fanout),
            "selector_ms": profile.selector_ms,
            "projected_assignment_bytes": profile.projected_assignment_bytes,
            "projected_partition_d00_bytes_pre_dedup": profile.projected_partition_d00_bytes_pre_dedup,
            "root_leader_hash": profile.root_leader_hash,
        }
    });
    serde_json::to_string_pretty(&value).expect("root fanout profile JSON is serializable") + "\n"
}
