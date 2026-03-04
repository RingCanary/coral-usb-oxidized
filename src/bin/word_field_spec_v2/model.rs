use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictMode {
    Endpoint,
    Best,
    Strict,
    ThreePoint,
}

impl PredictMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "endpoint" => Ok(Self::Endpoint),
            "best" => Ok(Self::Best),
            "strict" => Ok(Self::Strict),
            "threepoint" => Ok(Self::ThreePoint),
            _ => Err(format!(
                "invalid --predict-mode '{}', expected endpoint|best|strict|threepoint",
                value
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Endpoint => "endpoint",
            Self::Best => "best",
            Self::Strict => "strict",
            Self::ThreePoint => "threepoint",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MonotonicityClass {
    Const,
    MonotoneUp,
    MonotoneDown,
    MidpointPulse,
    NonMonotone,
    Insufficient,
}

impl MonotonicityClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Const => "const",
            Self::MonotoneUp => "monotone_up",
            Self::MonotoneDown => "monotone_down",
            Self::MidpointPulse => "midpoint_pulse",
            Self::NonMonotone => "non_monotone",
            Self::Insufficient => "insufficient",
        }
    }

    pub fn is_monotone(self) -> bool {
        matches!(self, Self::Const | Self::MonotoneUp | Self::MonotoneDown)
    }

    pub fn tier(self) -> PatchTier {
        match self {
            Self::Const | Self::MonotoneUp | Self::MonotoneDown => PatchTier::SafeCore,
            Self::MidpointPulse | Self::NonMonotone => PatchTier::DiscreteFlags,
            Self::Insufficient => PatchTier::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PatchTier {
    SafeCore,
    DiscreteFlags,
    Unknown,
}

impl PatchTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SafeCore => "safe_core",
            Self::DiscreteFlags => "discrete_flags",
            Self::Unknown => "unknown",
        }
    }
}

pub fn classify_three_point(lo: u64, mid: u64, hi: u64) -> MonotonicityClass {
    if lo == mid && mid == hi {
        return MonotonicityClass::Const;
    }
    if lo == hi && lo != mid {
        return MonotonicityClass::MidpointPulse;
    }
    if lo <= mid && mid <= hi {
        return MonotonicityClass::MonotoneUp;
    }
    if lo >= mid && mid >= hi {
        return MonotonicityClass::MonotoneDown;
    }
    MonotonicityClass::NonMonotone
}

pub fn classify_word_monotonicity(
    low: u64,
    mid: Option<u64>,
    high: u64,
    lane_bytes: usize,
) -> MonotonicityClass {
    let Some(mid_val) = mid else {
        return MonotonicityClass::Insufficient;
    };

    let mut classes = Vec::with_capacity(lane_bytes);
    for idx in 0..lane_bytes {
        let shift = idx * 8;
        let lo_b = (low >> shift) & 0xff;
        let mid_b = (mid_val >> shift) & 0xff;
        let hi_b = (high >> shift) & 0xff;
        classes.push(classify_three_point(lo_b, mid_b, hi_b));
    }

    if classes
        .iter()
        .any(|c| matches!(c, MonotonicityClass::Insufficient))
    {
        return MonotonicityClass::Insufficient;
    }
    if classes
        .iter()
        .any(|c| matches!(c, MonotonicityClass::NonMonotone))
    {
        return MonotonicityClass::NonMonotone;
    }
    if classes
        .iter()
        .any(|c| matches!(c, MonotonicityClass::MidpointPulse))
    {
        return MonotonicityClass::MidpointPulse;
    }

    let any_up = classes
        .iter()
        .any(|c| matches!(c, MonotonicityClass::MonotoneUp));
    let any_down = classes
        .iter()
        .any(|c| matches!(c, MonotonicityClass::MonotoneDown));

    if any_up && any_down {
        return MonotonicityClass::NonMonotone;
    }
    if any_up {
        return MonotonicityClass::MonotoneUp;
    }
    if any_down {
        return MonotonicityClass::MonotoneDown;
    }
    MonotonicityClass::Const
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct AnalysisReport {
    #[serde(default)]
    pub lane16: LaneReport,
    #[serde(default)]
    pub lane32: LaneReport,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct LaneReport {
    #[serde(default)]
    pub top_groups: Vec<GroupReport>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct GroupReport {
    #[serde(default)]
    pub stride_residue: Option<i64>,
    #[serde(default)]
    pub offsets: Vec<usize>,
    #[serde(default)]
    pub values_by_dim: Vec<GroupValueSeries>,
    #[serde(default)]
    pub per_offset_fits: Vec<PerOffsetFit>,
    #[serde(default)]
    pub best_formula: FormulaFit,
    #[serde(default)]
    pub bitfield_fits: Vec<BitfieldFit>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct GroupValueSeries {
    pub dim: i64,
    #[serde(default)]
    pub values: Vec<i64>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct PerOffsetFit {
    pub offset: usize,
    #[serde(default)]
    pub values_by_dim: Vec<ScalarValueByDim>,
    #[serde(default)]
    pub best_formula: FormulaFit,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct ScalarValueByDim {
    pub dim: i64,
    pub value: i64,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct FormulaFit {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub top_candidates: Vec<FormulaFit>,
    #[serde(default)]
    pub exact_ratio: f64,
    #[serde(default)]
    pub mae: f64,
    #[serde(default)]
    pub complexity: i64,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct BitfieldFit {
    #[serde(default)]
    pub bit_range: Vec<usize>,
    #[serde(default)]
    pub best_formula: FormulaFit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LaneKey {
    Lane16,
    Lane32,
}

impl LaneKey {
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "lane16" => Some(Self::Lane16),
            "lane32" => Some(Self::Lane32),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lane16 => "lane16",
            Self::Lane32 => "lane32",
        }
    }

    pub fn lane_bytes(self) -> usize {
        match self {
            Self::Lane16 => 2,
            Self::Lane32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WordContext {
    pub offset: usize,
    pub lane: LaneKey,
    pub lane_bytes: usize,
    pub residue: Option<i64>,
    pub low_val: u64,
    pub mid_val: Option<u64>,
    pub high_val: u64,
    pub base_word: u64,
    pub target_word: u64,
    pub mono_class: MonotonicityClass,
    pub best_formula: FormulaFit,
    pub group_bitfield_fits: Vec<BitfieldFit>,
}

#[derive(Debug, Clone)]
pub struct RuleChoice {
    pub lane: LaneKey,
    pub residue: Option<i64>,
    pub offset: Option<usize>,
    pub policy: Option<PredictMode>,
    pub model: Option<String>,
    pub div: Option<i64>,
    pub domain: Option<String>,
    pub bits: Option<u8>,
    pub bit_lo: Option<u8>,
    pub bit_hi: Option<u8>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuleSignature {
    pub lane: LaneKey,
    pub policy: Option<PredictMode>,
    pub model: Option<String>,
    pub div: Option<i64>,
    pub domain: Option<String>,
    pub bits: Option<u8>,
    pub bit_lo: Option<u8>,
    pub bit_hi: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct CandidateEval {
    pub rule: RuleChoice,
    pub mismatch_bytes: usize,
}

#[derive(Debug, Serialize, Clone)]
pub struct FieldSpecOutput {
    pub offset_rules: Vec<FieldSpecRuleOut>,
    pub residue_rules: Vec<FieldSpecRuleOut>,
}

#[derive(Debug, Serialize, Clone)]
pub struct FieldSpecRuleOut {
    pub lane: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residue: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub div: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bits: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bit_range: Option<[u8; 2]>,
}

#[derive(Debug, Serialize, Clone)]
pub struct MismatchSummary {
    pub mismatch_vs_target: usize,
    pub mismatch_ratio_vs_target: f64,
    pub mismatch_preview: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PatchSimulation {
    pub summary: MismatchSummary,
    pub patched: Vec<u8>,
    pub changed_from_base: Vec<usize>,
    pub byte_tiers: std::collections::HashMap<usize, PatchTier>,
}

#[derive(Debug, Serialize, Clone)]
pub struct OffsetNote {
    pub offset: usize,
    pub lane: String,
    pub residue: Option<i64>,
    pub mono_class: String,
    pub patch_tier: String,
    pub baseline_word_mismatch_bytes: usize,
    pub selected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_rule: Option<FieldSpecRuleOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_mismatch_bytes: Option<usize>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ReportOut {
    pub analysis_json: String,
    pub base_exec: String,
    pub target_exec: String,
    pub chunk_index: usize,
    pub predict_mode: String,
    pub lane_priority: String,
    pub dims: DimsOut,
    pub assigned_word_offsets: usize,
    pub baseline: MismatchSummary,
    pub with_v2_spec: MismatchSummary,
    pub v2_changed_byte_count: usize,
    pub safe_core_byte_count: usize,
    pub discrete_flags_byte_count: usize,
    pub unknown_byte_count: usize,
    pub residue_rule_count: usize,
    pub offset_rule_count: usize,
    pub per_offset_notes: Vec<OffsetNote>,
    pub out_spec: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_patchspec: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_patchspec_safe: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_patchspec_discrete: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct DimsOut {
    pub low: i64,
    pub high: i64,
    pub target: i64,
    pub tile_size: i64,
}
