use coral_usb_oxidized::extract_instruction_chunk_from_serialized_executable;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PredictMode {
    Endpoint,
    Best,
}

impl PredictMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "endpoint" => Ok(Self::Endpoint),
            "best" => Ok(Self::Best),
            _ => Err(format!(
                "invalid --predict-mode '{}', expected endpoint|best",
                value
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Endpoint => "endpoint",
            Self::Best => "best",
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    analysis_json: String,
    base_exec: String,
    target_exec: String,
    chunk_index: usize,
    low_dim: i64,
    high_dim: i64,
    target_dim: i64,
    tile_size: i64,
    predict_mode: PredictMode,
    lane_priority: String,
    out_spec: String,
    out_report: Option<String>,
    out_patchspec: Option<String>,
}

fn parse_i64_flag(value: &str, flag: &str) -> Result<i64, String> {
    value
        .parse::<i64>()
        .map_err(|e| format!("{} invalid integer '{}': {}", flag, value, e))
}

fn parse_usize_flag(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|e| format!("{} invalid integer '{}': {}", flag, value, e))
}

fn next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    *idx += 1;
    if *idx >= args.len() {
        return Err(format!("{} requires a value", flag));
    }
    Ok(args[*idx].clone())
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        return Err(
            "missing args: --analysis-json --base-exec --target-exec --low-dim --high-dim --target-dim --out-spec"
                .to_string(),
        );
    }

    let mut analysis_json: Option<String> = None;
    let mut base_exec: Option<String> = None;
    let mut target_exec: Option<String> = None;
    let mut chunk_index: usize = 0;
    let mut low_dim: Option<i64> = None;
    let mut high_dim: Option<i64> = None;
    let mut target_dim: Option<i64> = None;
    let mut tile_size: i64 = 64;
    let mut predict_mode = PredictMode::Endpoint;
    let mut lane_priority = "lane32,lane16".to_string();
    let mut out_spec: Option<String> = None;
    let mut out_report: Option<String> = None;
    let mut out_patchspec: Option<String> = None;

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--analysis-json" => {
                analysis_json = Some(next_arg(&args, &mut idx, "--analysis-json")?);
            }
            "--base-exec" => {
                base_exec = Some(next_arg(&args, &mut idx, "--base-exec")?);
            }
            "--target-exec" => {
                target_exec = Some(next_arg(&args, &mut idx, "--target-exec")?);
            }
            "--chunk-index" => {
                chunk_index = parse_usize_flag(
                    &next_arg(&args, &mut idx, "--chunk-index")?,
                    "--chunk-index",
                )?;
            }
            "--low-dim" => {
                low_dim = Some(parse_i64_flag(
                    &next_arg(&args, &mut idx, "--low-dim")?,
                    "--low-dim",
                )?);
            }
            "--high-dim" => {
                high_dim = Some(parse_i64_flag(
                    &next_arg(&args, &mut idx, "--high-dim")?,
                    "--high-dim",
                )?);
            }
            "--target-dim" => {
                target_dim = Some(parse_i64_flag(
                    &next_arg(&args, &mut idx, "--target-dim")?,
                    "--target-dim",
                )?);
            }
            "--tile-size" => {
                tile_size =
                    parse_i64_flag(&next_arg(&args, &mut idx, "--tile-size")?, "--tile-size")?;
            }
            "--predict-mode" => {
                predict_mode = PredictMode::parse(&next_arg(&args, &mut idx, "--predict-mode")?)?;
            }
            "--lane-priority" => {
                lane_priority = next_arg(&args, &mut idx, "--lane-priority")?;
            }
            "--out-spec" => {
                out_spec = Some(next_arg(&args, &mut idx, "--out-spec")?);
            }
            "--out-report" => {
                out_report = Some(next_arg(&args, &mut idx, "--out-report")?);
            }
            "--out-patchspec" => {
                out_patchspec = Some(next_arg(&args, &mut idx, "--out-patchspec")?);
            }
            other => {
                return Err(format!("unknown flag: {}", other));
            }
        }
        idx += 1;
    }

    let Some(analysis_json) = analysis_json else {
        return Err("missing --analysis-json".to_string());
    };
    let Some(base_exec) = base_exec else {
        return Err("missing --base-exec".to_string());
    };
    let Some(target_exec) = target_exec else {
        return Err("missing --target-exec".to_string());
    };
    let Some(low_dim) = low_dim else {
        return Err("missing --low-dim".to_string());
    };
    let Some(high_dim) = high_dim else {
        return Err("missing --high-dim".to_string());
    };
    let Some(target_dim) = target_dim else {
        return Err("missing --target-dim".to_string());
    };
    let Some(out_spec) = out_spec else {
        return Err("missing --out-spec".to_string());
    };

    if tile_size <= 0 {
        return Err("--tile-size must be > 0".to_string());
    }

    Ok(Config {
        analysis_json,
        base_exec,
        target_exec,
        chunk_index,
        low_dim,
        high_dim,
        target_dim,
        tile_size,
        predict_mode,
        lane_priority,
        out_spec,
        out_report,
        out_patchspec,
    })
}

#[derive(Debug, Deserialize, Default, Clone)]
struct AnalysisReport {
    #[serde(default)]
    lane16: LaneReport,
    #[serde(default)]
    lane32: LaneReport,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct LaneReport {
    #[serde(default)]
    top_groups: Vec<GroupReport>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct GroupReport {
    #[serde(default)]
    stride_residue: Option<i64>,
    #[serde(default)]
    offsets: Vec<usize>,
    #[serde(default)]
    values_by_dim: Vec<GroupValueSeries>,
    #[serde(default)]
    per_offset_fits: Vec<PerOffsetFit>,
    #[serde(default)]
    best_formula: FormulaFit,
    #[serde(default)]
    bitfield_fits: Vec<BitfieldFit>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct GroupValueSeries {
    dim: i64,
    #[serde(default)]
    values: Vec<i64>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct PerOffsetFit {
    offset: usize,
    #[serde(default)]
    values_by_dim: Vec<ScalarValueByDim>,
    #[serde(default)]
    best_formula: FormulaFit,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct ScalarValueByDim {
    dim: i64,
    value: i64,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct FormulaFit {
    #[serde(default)]
    model: String,
    #[serde(default)]
    params: Map<String, Value>,
    #[serde(default)]
    top_candidates: Vec<FormulaFit>,
    #[serde(default)]
    exact_ratio: f64,
    #[serde(default)]
    mae: f64,
    #[serde(default)]
    complexity: i64,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct BitfieldFit {
    #[serde(default)]
    bit_range: Vec<usize>,
    #[serde(default)]
    best_formula: FormulaFit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum LaneKey {
    Lane16,
    Lane32,
}

impl LaneKey {
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "lane16" => Some(Self::Lane16),
            "lane32" => Some(Self::Lane32),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Lane16 => "lane16",
            Self::Lane32 => "lane32",
        }
    }

    fn lane_bytes(self) -> usize {
        match self {
            Self::Lane16 => 2,
            Self::Lane32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
struct WordContext {
    offset: usize,
    lane: LaneKey,
    lane_bytes: usize,
    residue: Option<i64>,
    low_val: u64,
    high_val: u64,
    base_word: u64,
    target_word: u64,
    best_formula: FormulaFit,
    group_bitfield_fits: Vec<BitfieldFit>,
}

#[derive(Debug, Clone)]
struct RuleChoice {
    lane: LaneKey,
    residue: Option<i64>,
    offset: Option<usize>,
    policy: Option<PredictMode>,
    model: Option<String>,
    div: Option<i64>,
    domain: Option<String>,
    bits: Option<u8>,
    bit_lo: Option<u8>,
    bit_hi: Option<u8>,
    source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RuleSignature {
    lane: LaneKey,
    policy: Option<PredictMode>,
    model: Option<String>,
    div: Option<i64>,
    domain: Option<String>,
    bits: Option<u8>,
    bit_lo: Option<u8>,
    bit_hi: Option<u8>,
}

#[derive(Debug, Clone)]
struct CandidateEval {
    rule: RuleChoice,
    mismatch_bytes: usize,
}

#[derive(Debug, Serialize, Clone)]
struct FieldSpecOutput {
    offset_rules: Vec<FieldSpecRuleOut>,
    residue_rules: Vec<FieldSpecRuleOut>,
}

#[derive(Debug, Serialize, Clone)]
struct FieldSpecRuleOut {
    lane: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    residue: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    div: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bits: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bit_range: Option<[u8; 2]>,
}

#[derive(Debug, Serialize, Clone)]
struct MismatchSummary {
    mismatch_vs_target: usize,
    mismatch_ratio_vs_target: f64,
    mismatch_preview: Vec<usize>,
}

#[derive(Debug, Clone)]
struct PatchSimulation {
    summary: MismatchSummary,
    patched: Vec<u8>,
    changed_from_base: Vec<usize>,
}

#[derive(Debug, Serialize, Clone)]
struct OffsetNote {
    offset: usize,
    lane: String,
    residue: Option<i64>,
    baseline_word_mismatch_bytes: usize,
    selected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_rule: Option<FieldSpecRuleOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_mismatch_bytes: Option<usize>,
}

#[derive(Debug, Serialize, Clone)]
struct ReportOut {
    analysis_json: String,
    base_exec: String,
    target_exec: String,
    chunk_index: usize,
    predict_mode: String,
    lane_priority: String,
    dims: DimsOut,
    assigned_word_offsets: usize,
    baseline: MismatchSummary,
    with_v2_spec: MismatchSummary,
    v2_changed_byte_count: usize,
    residue_rule_count: usize,
    offset_rule_count: usize,
    per_offset_notes: Vec<OffsetNote>,
    out_spec: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    out_patchspec: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct DimsOut {
    low: i64,
    high: i64,
    target: i64,
    tile_size: i64,
}

fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn value_to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_u64().map(|x| x as i64)),
        Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
}

fn param_f64(params: &Map<String, Value>, key: &str, default: f64) -> f64 {
    params.get(key).and_then(value_to_f64).unwrap_or(default)
}

fn param_i64(params: &Map<String, Value>, key: &str, default: i64) -> i64 {
    params.get(key).and_then(value_to_i64).unwrap_or(default)
}

fn model_complexity(model: &str) -> i64 {
    match model {
        "const" => 0,
        "tile-linear" => 1,
        "tile2div-linear" => 2,
        "tile-quadratic" => 2,
        _ => 9,
    }
}

fn formula_effective_complexity(formula: &FormulaFit) -> i64 {
    if formula.complexity != 0 {
        formula.complexity
    } else {
        model_complexity(&formula.model)
    }
}

fn is_exact_formula(formula: &FormulaFit) -> bool {
    (formula.exact_ratio - 1.0).abs() <= 1e-9 && formula.mae.abs() <= 1e-9
}

fn pick_best_candidate(
    best: &FormulaFit,
    model_override: Option<&str>,
    div_override: Option<i64>,
) -> (String, Map<String, Value>) {
    let mut cands: Vec<&FormulaFit> = Vec::with_capacity(1 + best.top_candidates.len());
    cands.push(best);
    for cand in &best.top_candidates {
        cands.push(cand);
    }

    let mut filtered: Vec<&FormulaFit> = Vec::new();
    for cand in cands {
        let model = cand.model.as_str();
        if let Some(mo) = model_override {
            if model != mo {
                continue;
            }
        }
        if let Some(div) = div_override {
            if model != "tile2div-linear" {
                continue;
            }
            if param_i64(&cand.params, "div", -1) != div {
                continue;
            }
        }
        filtered.push(cand);
    }

    if filtered.is_empty() {
        return (best.model.clone(), best.params.clone());
    }

    filtered.sort_by(|a, b| {
        let exact_a = a.exact_ratio;
        let exact_b = b.exact_ratio;
        let mae_a = a.mae;
        let mae_b = b.mae;
        let cplx_a = formula_effective_complexity(a);
        let cplx_b = formula_effective_complexity(b);

        exact_b
            .partial_cmp(&exact_a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| mae_a.partial_cmp(&mae_b).unwrap_or(Ordering::Equal))
            .then_with(|| cplx_a.cmp(&cplx_b))
            .then_with(|| a.model.cmp(&b.model))
    });

    let chosen = filtered[0];
    (chosen.model.clone(), chosen.params.clone())
}

fn pick_formula(best: &FormulaFit, mode: PredictMode) -> (String, Map<String, Value>) {
    if matches!(mode, PredictMode::Best) {
        return (best.model.clone(), best.params.clone());
    }

    let mut exact: Vec<&FormulaFit> = best
        .top_candidates
        .iter()
        .filter(|cand| is_exact_formula(cand))
        .collect();

    if !exact.is_empty() {
        exact.sort_by(|a, b| {
            let rank_model = |m: &str| -> i64 {
                match m {
                    "tile2div-linear" => 0,
                    "tile-linear" => 1,
                    _ => 2,
                }
            };

            let a_rank = rank_model(&a.model);
            let b_rank = rank_model(&b.model);

            let a_a = param_f64(&a.params, "a", 0.0);
            let a_b = param_f64(&a.params, "b", 0.0);
            let b_a = param_f64(&b.params, "a", 0.0);
            let b_b = param_f64(&b.params, "b", 0.0);

            let a_int_err = (a_a - a_a.round()).abs() + (a_b - a_b.round()).abs();
            let b_int_err = (b_a - b_a.round()).abs() + (b_b - b_b.round()).abs();

            let a_div = param_i64(&a.params, "div", 1);
            let b_div = param_i64(&b.params, "div", 1);

            a_rank
                .cmp(&b_rank)
                .then_with(|| a_int_err.partial_cmp(&b_int_err).unwrap_or(Ordering::Equal))
                .then_with(|| a_a.abs().partial_cmp(&b_a.abs()).unwrap_or(Ordering::Equal))
                .then_with(|| a_div.cmp(&b_div))
                .then_with(|| a.model.cmp(&b.model))
        });

        let chosen = exact[0];
        return (chosen.model.clone(), chosen.params.clone());
    }

    (best.model.clone(), best.params.clone())
}

fn parse_lane_priority(value: &str) -> Result<Vec<LaneKey>, String> {
    let mut out = Vec::new();
    for token in value.split(',') {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        let Some(lane) = LaneKey::from_token(t) else {
            return Err(format!(
                "invalid lane priority token '{}' (expected lane32,lane16)",
                t
            ));
        };
        out.push(lane);
    }
    if out.is_empty() {
        return Err("empty --lane-priority".to_string());
    }
    Ok(out)
}

fn lane_report<'a>(analysis: &'a AnalysisReport, lane: LaneKey) -> &'a LaneReport {
    match lane {
        LaneKey::Lane16 => &analysis.lane16,
        LaneKey::Lane32 => &analysis.lane32,
    }
}

fn read_word_le(bytes: &[u8], offset: usize, lane_bytes: usize) -> u64 {
    let mut out = 0u64;
    for idx in 0..lane_bytes {
        out |= (bytes[offset + idx] as u64) << (idx * 8);
    }
    out
}

fn values_map_scalar(rows: &[ScalarValueByDim]) -> HashMap<i64, u64> {
    let mut out = HashMap::new();
    for row in rows {
        out.insert(row.dim, row.value as u64);
    }
    out
}

fn is_formula_empty(formula: &FormulaFit) -> bool {
    formula.model.is_empty()
        && formula.params.is_empty()
        && formula.top_candidates.is_empty()
        && formula.exact_ratio == 0.0
        && formula.mae == 0.0
        && formula.complexity == 0
}

fn build_contexts(
    analysis: &AnalysisReport,
    base_blob: &[u8],
    target_blob: &[u8],
    low_dim: i64,
    high_dim: i64,
    lanes: &[LaneKey],
) -> Vec<WordContext> {
    let mut contexts = Vec::new();
    let mut assigned_offsets: HashSet<usize> = HashSet::new();

    for lane in lanes {
        let lane_bytes = lane.lane_bytes();
        let lane_obj = lane_report(analysis, *lane);

        for group in &lane_obj.top_groups {
            let residue = group.stride_residue;

            if !group.per_offset_fits.is_empty() {
                for per in &group.per_offset_fits {
                    let off = per.offset;
                    if assigned_offsets.contains(&off) {
                        continue;
                    }
                    if off + lane_bytes > base_blob.len() || off + lane_bytes > target_blob.len() {
                        continue;
                    }

                    let vals = values_map_scalar(&per.values_by_dim);
                    let Some(low_val) = vals.get(&low_dim).copied() else {
                        continue;
                    };
                    let Some(high_val) = vals.get(&high_dim).copied() else {
                        continue;
                    };

                    let best_formula = if is_formula_empty(&per.best_formula) {
                        group.best_formula.clone()
                    } else {
                        per.best_formula.clone()
                    };

                    contexts.push(WordContext {
                        offset: off,
                        lane: *lane,
                        lane_bytes,
                        residue,
                        low_val,
                        high_val,
                        base_word: read_word_le(base_blob, off, lane_bytes),
                        target_word: read_word_le(target_blob, off, lane_bytes),
                        best_formula,
                        group_bitfield_fits: group.bitfield_fits.clone(),
                    });
                    assigned_offsets.insert(off);
                }
                continue;
            }

            if group.offsets.is_empty() || group.values_by_dim.is_empty() {
                continue;
            }

            let mut dim_to_values: HashMap<i64, &Vec<i64>> = HashMap::new();
            for row in &group.values_by_dim {
                dim_to_values.insert(row.dim, &row.values);
            }
            let Some(low_series) = dim_to_values.get(&low_dim) else {
                continue;
            };
            let Some(high_series) = dim_to_values.get(&high_dim) else {
                continue;
            };

            for (idx, off) in group.offsets.iter().copied().enumerate() {
                if assigned_offsets.contains(&off) {
                    continue;
                }
                if off + lane_bytes > base_blob.len() || off + lane_bytes > target_blob.len() {
                    continue;
                }
                if idx >= low_series.len() || idx >= high_series.len() {
                    continue;
                }

                contexts.push(WordContext {
                    offset: off,
                    lane: *lane,
                    lane_bytes,
                    residue,
                    low_val: low_series[idx] as u64,
                    high_val: high_series[idx] as u64,
                    base_word: read_word_le(base_blob, off, lane_bytes),
                    target_word: read_word_le(target_blob, off, lane_bytes),
                    best_formula: group.best_formula.clone(),
                    group_bitfield_fits: group.bitfield_fits.clone(),
                });
                assigned_offsets.insert(off);
            }
        }
    }

    contexts
}

fn tiles(dim: i64, tile_size: i64) -> i64 {
    dim / tile_size
}

fn bits_mask(bits: u8) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        ((1u128 << bits) - 1) as u64
    }
}

fn to_signed(v: u64, bits: u8) -> i64 {
    let mask = bits_mask(bits);
    let vv = v & mask;
    let sign = 1u64 << (bits - 1);
    if (vv & sign) != 0 {
        (vv as i128 - (1i128 << bits)) as i64
    } else {
        vv as i64
    }
}

fn from_signed(v: i64, bits: u8) -> u64 {
    let mask = bits_mask(bits) as i128;
    ((v as i128) & mask) as u64
}

fn interp_domain(low: u64, high: u64, xl: f64, xh: f64, xt: f64, bits: u8, domain: &str) -> i128 {
    if (xh - xl).abs() <= f64::EPSILON {
        return low as i128;
    }

    let frac = (xt - xl) / (xh - xl);

    match domain {
        "u" => {
            let y = low as f64 + ((high as f64 - low as f64) * frac);
            y.round() as i128
        }
        "s" => {
            let l = to_signed(low, bits);
            let h = to_signed(high, bits);
            let y = l as f64 + ((h as f64 - l as f64) * frac);
            from_signed(y.round() as i64, bits) as i128
        }
        "mod" => {
            let ring = 1i128 << bits;
            let half = ring / 2;
            let low_i = (low as i128).rem_euclid(ring);
            let high_i = (high as i128).rem_euclid(ring);
            let delta = (high_i - low_i + half).rem_euclid(ring) - half;
            let y = (low_i as f64 + (delta as f64 * frac)).round() as i128;
            y.rem_euclid(ring)
        }
        _ => {
            let y = low as f64 + ((high as f64 - low as f64) * frac);
            y.round() as i128
        }
    }
}

fn predict_value(
    model: &str,
    params: &Map<String, Value>,
    low_dim: i64,
    low_val: u64,
    high_dim: i64,
    high_val: u64,
    target_dim: i64,
    tile_size: i64,
    mode: PredictMode,
    domain: &str,
    bits: u8,
) -> i128 {
    if matches!(mode, PredictMode::Best) {
        let tt = tiles(target_dim, tile_size) as f64;
        match model {
            "const" => {
                let c = param_f64(params, "c", low_val as f64);
                return c.round() as i128;
            }
            "tile-linear" => {
                let a = param_f64(params, "a", 0.0);
                let b = param_f64(params, "b", low_val as f64);
                return (a * tt + b).round() as i128;
            }
            "tile-quadratic" => {
                let a = param_f64(params, "a", 0.0);
                let b = param_f64(params, "b", 0.0);
                let c = param_f64(params, "c", low_val as f64);
                return (a * tt * tt + b * tt + c).round() as i128;
            }
            "tile2div-linear" => {
                let div = param_i64(params, "div", 1).max(1);
                let x =
                    ((tiles(target_dim, tile_size) * tiles(target_dim, tile_size)) / div) as f64;
                let a = param_f64(params, "a", 0.0);
                let b = param_f64(params, "b", low_val as f64);
                return (a * x + b).round() as i128;
            }
            _ => {}
        }
    }

    let tl = tiles(low_dim, tile_size);
    let th = tiles(high_dim, tile_size);
    let tt = tiles(target_dim, tile_size);

    match model {
        "const" => low_val as i128,
        "tile-linear" => interp_domain(
            low_val, high_val, tl as f64, th as f64, tt as f64, bits, domain,
        ),
        "tile2div-linear" => {
            let div = param_i64(params, "div", 1).max(1);
            let xl = ((tl * tl) / div) as f64;
            let xh = ((th * th) / div) as f64;
            let xt = ((tt * tt) / div) as f64;
            interp_domain(low_val, high_val, xl, xh, xt, bits, domain)
        }
        "tile-quadratic" => interp_domain(
            low_val, high_val, tl as f64, th as f64, tt as f64, bits, domain,
        ),
        _ => low_val as i128,
    }
}

fn collect_endpoint_models(formula: &FormulaFit) -> Vec<(String, Option<i64>)> {
    let mut out = Vec::new();
    let mut seen: HashSet<(String, Option<i64>)> = HashSet::new();

    let mut push = |model: &str, div: Option<i64>| {
        let key = (model.to_string(), div);
        if seen.insert(key.clone()) {
            out.push(key);
        }
    };

    let mut all: Vec<&FormulaFit> = Vec::with_capacity(1 + formula.top_candidates.len());
    all.push(formula);
    for cand in &formula.top_candidates {
        all.push(cand);
    }

    for cand in all {
        match cand.model.as_str() {
            "tile2div-linear" => push("tile2div-linear", Some(param_i64(&cand.params, "div", 1))),
            "tile-linear" => push("tile-linear", None),
            "const" => push("const", None),
            "tile-quadratic" => push("tile-linear", None),
            _ => {}
        }
    }

    let (picked_model, picked_params) = pick_formula(formula, PredictMode::Endpoint);
    match picked_model.as_str() {
        "tile2div-linear" => push(
            "tile2div-linear",
            Some(param_i64(&picked_params, "div", 1).max(1)),
        ),
        "tile-linear" => push("tile-linear", None),
        "const" => push("const", None),
        "tile-quadratic" => push("tile-linear", None),
        _ => {}
    }

    out
}

fn predict_word(
    ctx: &WordContext,
    global_mode: PredictMode,
    low_dim: i64,
    high_dim: i64,
    target_dim: i64,
    tile_size: i64,
    rule: Option<&RuleChoice>,
) -> (u64, String, Map<String, Value>, String, u8) {
    let mut mode_use = global_mode;
    if let Some(r) = rule {
        if let Some(p) = r.policy {
            mode_use = p;
        }
    }

    let (model, params) = if matches!(mode_use, PredictMode::Best) {
        pick_best_candidate(
            &ctx.best_formula,
            rule.and_then(|r| r.model.as_deref()),
            rule.and_then(|r| r.div),
        )
    } else if let Some(r) = rule {
        if let Some(m) = &r.model {
            let mut p = Map::new();
            if let Some(div) = r.div {
                p.insert("div".to_string(), Value::from(div));
            }
            (m.clone(), p)
        } else {
            pick_formula(&ctx.best_formula, mode_use)
        }
    } else {
        pick_formula(&ctx.best_formula, mode_use)
    };

    let lane_bits = (ctx.lane_bytes * 8) as u8;
    let bit_lo = rule.and_then(|r| r.bit_lo);
    let bit_hi = rule.and_then(|r| r.bit_hi);
    let has_bit_range = bit_lo.is_some() && bit_hi.is_some();

    let (low_eval, high_eval, width, mut bits) = if has_bit_range {
        let lo = bit_lo.unwrap();
        let hi = bit_hi.unwrap();
        let width = hi - lo + 1;
        let mask = bits_mask(width);
        (
            (ctx.low_val >> lo) & mask,
            (ctx.high_val >> lo) & mask,
            width,
            width,
        )
    } else {
        (ctx.low_val, ctx.high_val, lane_bits, lane_bits)
    };

    if let Some(r) = rule {
        if let Some(rb) = r.bits {
            bits = rb;
        }
    }

    let domain = rule
        .and_then(|r| r.domain.as_ref())
        .map(|x| x.as_str())
        .unwrap_or("u")
        .to_string();

    let pred_field = predict_value(
        &model, &params, low_dim, low_eval, high_dim, high_eval, target_dim, tile_size, mode_use,
        &domain, bits,
    );

    let field_mod = 1i128 << width;
    let pred_field_norm = pred_field.rem_euclid(field_mod) as u64;

    let pred_word = if has_bit_range {
        let lo = bit_lo.unwrap();
        let mask = bits_mask(width);
        (ctx.base_word & !(mask << lo)) | ((pred_field_norm & mask) << lo)
    } else {
        pred_field_norm
    };

    let lane_mod = 1u128 << lane_bits;
    let pred_word_norm = (pred_word as u128 % lane_mod) as u64;

    (pred_word_norm, model, params, domain, bits)
}

fn mismatch_bytes_word(pred_word: u64, target_word: u64, lane_bytes: usize) -> usize {
    let mut out = 0usize;
    for idx in 0..lane_bytes {
        let pb = ((pred_word >> (idx * 8)) & 0xff) as u8;
        let tb = ((target_word >> (idx * 8)) & 0xff) as u8;
        if pb != tb {
            out += 1;
        }
    }
    out
}

fn lane_bit_options(lane_bits: u8) -> Vec<u8> {
    let seeds: &[u8] = if lane_bits <= 16 {
        &[8, 12, 16]
    } else {
        &[8, 12, 16, 20, 24, 28, 32]
    };

    let mut out: Vec<u8> = seeds.iter().copied().filter(|x| *x <= lane_bits).collect();
    if !out.contains(&lane_bits) {
        out.push(lane_bits);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn formula_div(formula: &FormulaFit) -> Option<i64> {
    if formula.model == "tile2div-linear" {
        Some(param_i64(&formula.params, "div", 1))
    } else {
        None
    }
}

fn collect_bit_ranges(ctx: &WordContext, baseline_pred: u64) -> Vec<(u8, u8, String, FormulaFit)> {
    let mut out: Vec<(u8, u8, String, FormulaFit)> = Vec::new();

    for bf in &ctx.group_bitfield_fits {
        if bf.bit_range.len() != 2 {
            continue;
        }
        let lo = bf.bit_range[0] as u8;
        let hi = bf.bit_range[1] as u8;
        if lo > hi || hi >= (ctx.lane_bytes * 8) as u8 {
            continue;
        }
        out.push((
            lo,
            hi,
            "group_bitfield".to_string(),
            bf.best_formula.clone(),
        ));
    }

    let mut mism_bytes = Vec::new();
    for idx in 0..ctx.lane_bytes {
        let pb = ((baseline_pred >> (idx * 8)) & 0xff) as u8;
        let tb = ((ctx.target_word >> (idx * 8)) & 0xff) as u8;
        if pb != tb {
            mism_bytes.push(idx as u8);
        }
    }

    for b in &mism_bytes {
        let lo = b * 8;
        let hi = lo + 7;
        out.push((
            lo,
            hi,
            "mismatch_byte".to_string(),
            ctx.best_formula.clone(),
        ));
    }

    if !mism_bytes.is_empty() {
        let lo = mism_bytes.iter().copied().min().unwrap() * 8;
        let hi = mism_bytes.iter().copied().max().unwrap() * 8 + 7;
        out.push((
            lo,
            hi,
            "mismatch_span".to_string(),
            ctx.best_formula.clone(),
        ));
    }

    let mut uniq = Vec::new();
    let mut seen: HashSet<(u8, u8, String, String, Option<i64>)> = HashSet::new();
    for (lo, hi, src, formula) in out {
        let key = (
            lo,
            hi,
            src.clone(),
            formula.model.clone(),
            formula_div(&formula),
        );
        if seen.insert(key) {
            uniq.push((lo, hi, src, formula));
        }
    }

    uniq
}

fn rule_signature(rule: &RuleChoice) -> RuleSignature {
    RuleSignature {
        lane: rule.lane,
        policy: rule.policy,
        model: rule.model.clone(),
        div: rule.div,
        domain: rule.domain.clone(),
        bits: rule.bits,
        bit_lo: rule.bit_lo,
        bit_hi: rule.bit_hi,
    }
}

fn evaluate_rule_on_ctx(
    ctx: &WordContext,
    global_mode: PredictMode,
    low_dim: i64,
    high_dim: i64,
    target_dim: i64,
    tile_size: i64,
    rule: RuleChoice,
) -> CandidateEval {
    let (pred_word, _model, _params, _domain, _bits) = predict_word(
        ctx,
        global_mode,
        low_dim,
        high_dim,
        target_dim,
        tile_size,
        Some(&rule),
    );
    CandidateEval {
        mismatch_bytes: mismatch_bytes_word(pred_word, ctx.target_word, ctx.lane_bytes),
        rule,
    }
}

fn candidate_rank(candidate: &CandidateEval, lane_bits: u8) -> (usize, usize, i64, u8, u8) {
    let bit_width = if let (Some(lo), Some(hi)) = (candidate.rule.bit_lo, candidate.rule.bit_hi) {
        hi - lo + 1
    } else {
        lane_bits
    };
    (
        candidate.mismatch_bytes,
        if candidate.rule.bit_lo.is_some() {
            0
        } else {
            1
        },
        model_complexity(candidate.rule.model.as_deref().unwrap_or("")),
        bit_width,
        candidate.rule.bits.unwrap_or(lane_bits),
    )
}

fn propose_offset_rule(
    ctx: &WordContext,
    global_mode: PredictMode,
    low_dim: i64,
    high_dim: i64,
    target_dim: i64,
    tile_size: i64,
) -> (usize, Option<CandidateEval>) {
    let (baseline_pred, _m, _p, _d, _b) = predict_word(
        ctx,
        global_mode,
        low_dim,
        high_dim,
        target_dim,
        tile_size,
        None,
    );
    let baseline_mismatch = mismatch_bytes_word(baseline_pred, ctx.target_word, ctx.lane_bytes);
    if baseline_mismatch == 0 {
        return (0, None);
    }

    let lane_bits = (ctx.lane_bytes * 8) as u8;
    let mut evals: HashMap<RuleSignature, CandidateEval> = HashMap::new();

    for (model, div) in collect_endpoint_models(&ctx.best_formula) {
        for domain in ["u", "s", "mod"] {
            for bits in lane_bit_options(lane_bits) {
                let rule = RuleChoice {
                    lane: ctx.lane,
                    residue: ctx.residue,
                    offset: Some(ctx.offset),
                    policy: Some(PredictMode::Endpoint),
                    model: Some(model.clone()),
                    div,
                    domain: Some(domain.to_string()),
                    bits: Some(bits),
                    bit_lo: None,
                    bit_hi: None,
                    source: "full_word".to_string(),
                };
                let ev = evaluate_rule_on_ctx(
                    ctx,
                    global_mode,
                    low_dim,
                    high_dim,
                    target_dim,
                    tile_size,
                    rule,
                );
                if ev.mismatch_bytes >= baseline_mismatch {
                    continue;
                }
                let sig = rule_signature(&ev.rule);
                let keep = match evals.get(&sig) {
                    Some(prev) => ev.mismatch_bytes < prev.mismatch_bytes,
                    None => true,
                };
                if keep {
                    evals.insert(sig, ev);
                }
            }
        }
    }

    for (lo, hi, source, formula) in collect_bit_ranges(ctx, baseline_pred) {
        let width = hi - lo + 1;

        let mut bit_opts = lane_bit_options(width);
        if !bit_opts.contains(&width) {
            bit_opts.push(width);
        }
        bit_opts.sort_unstable();
        bit_opts.dedup();

        for (model, div) in collect_endpoint_models(&formula) {
            for domain in ["u", "s", "mod"] {
                for bits in &bit_opts {
                    if *bits > width {
                        continue;
                    }

                    let rule = RuleChoice {
                        lane: ctx.lane,
                        residue: ctx.residue,
                        offset: Some(ctx.offset),
                        policy: Some(PredictMode::Endpoint),
                        model: Some(model.clone()),
                        div,
                        domain: Some(domain.to_string()),
                        bits: Some(*bits),
                        bit_lo: Some(lo),
                        bit_hi: Some(hi),
                        source: source.clone(),
                    };
                    let ev = evaluate_rule_on_ctx(
                        ctx,
                        global_mode,
                        low_dim,
                        high_dim,
                        target_dim,
                        tile_size,
                        rule,
                    );
                    if ev.mismatch_bytes >= baseline_mismatch {
                        continue;
                    }
                    let sig = rule_signature(&ev.rule);
                    let keep = match evals.get(&sig) {
                        Some(prev) => ev.mismatch_bytes < prev.mismatch_bytes,
                        None => true,
                    };
                    if keep {
                        evals.insert(sig, ev);
                    }
                }
            }
        }
    }

    if evals.is_empty() {
        return (baseline_mismatch, None);
    }

    let mut candidates: Vec<CandidateEval> = evals.into_values().collect();
    candidates.sort_by(|a, b| {
        candidate_rank(a, lane_bits)
            .cmp(&candidate_rank(b, lane_bits))
            .then_with(|| a.rule.source.cmp(&b.rule.source))
    });

    (baseline_mismatch, candidates.into_iter().next())
}

fn write_word_le(bytes: &mut [u8], offset: usize, lane_bytes: usize, value: u64) {
    for idx in 0..lane_bytes {
        bytes[offset + idx] = ((value >> (idx * 8)) & 0xff) as u8;
    }
}

fn simulate_patch(
    contexts: &[WordContext],
    base_blob: &[u8],
    target_blob: &[u8],
    global_mode: PredictMode,
    low_dim: i64,
    high_dim: i64,
    target_dim: i64,
    tile_size: i64,
    offset_rules: &HashMap<usize, RuleChoice>,
    residue_rules: &HashMap<(LaneKey, i64), RuleChoice>,
) -> PatchSimulation {
    let mut patched = base_blob.to_vec();

    for ctx in contexts {
        let rule_ref = offset_rules.get(&ctx.offset).or_else(|| {
            ctx.residue
                .and_then(|res| residue_rules.get(&(ctx.lane, res)))
        });
        let (pred_word, _m, _p, _d, _b) = predict_word(
            ctx,
            global_mode,
            low_dim,
            high_dim,
            target_dim,
            tile_size,
            rule_ref,
        );
        write_word_le(&mut patched, ctx.offset, ctx.lane_bytes, pred_word);
    }

    let mut mism = Vec::new();
    let mut changed_from_base = Vec::new();
    for idx in 0..target_blob.len() {
        if patched[idx] != target_blob[idx] {
            mism.push(idx);
        }
        if patched[idx] != base_blob[idx] {
            changed_from_base.push(idx);
        }
    }

    PatchSimulation {
        summary: MismatchSummary {
            mismatch_vs_target: mism.len(),
            mismatch_ratio_vs_target: mism.len() as f64 / target_blob.len() as f64,
            mismatch_preview: mism.into_iter().take(200).collect(),
        },
        patched,
        changed_from_base,
    }
}

fn rule_to_output(
    rule: &RuleChoice,
    include_offset: bool,
    include_residue: bool,
) -> FieldSpecRuleOut {
    FieldSpecRuleOut {
        lane: rule.lane.as_str().to_string(),
        residue: if include_residue { rule.residue } else { None },
        offset: if include_offset { rule.offset } else { None },
        policy: rule.policy.map(|p| p.as_str().to_string()),
        model: rule.model.clone(),
        div: rule.div,
        domain: rule.domain.clone(),
        bits: rule.bits,
        bit_range: if let (Some(lo), Some(hi)) = (rule.bit_lo, rule.bit_hi) {
            Some([lo, hi])
        } else {
            None
        },
    }
}

fn ensure_parent_dir(path: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let analysis_blob = fs::read(&config.analysis_json)?;
    let analysis: AnalysisReport = serde_json::from_slice(&analysis_blob)?;

    let base_exec_blob = fs::read(&config.base_exec)?;
    let target_exec_blob = fs::read(&config.target_exec)?;

    let base_blob =
        extract_instruction_chunk_from_serialized_executable(&base_exec_blob, config.chunk_index)?;
    let target_blob = extract_instruction_chunk_from_serialized_executable(
        &target_exec_blob,
        config.chunk_index,
    )?;

    if base_blob.len() != target_blob.len() {
        return Err(format!(
            "base/target chunk size mismatch: base={} target={}",
            base_blob.len(),
            target_blob.len()
        )
        .into());
    }

    let lanes = parse_lane_priority(&config.lane_priority)?;

    let contexts = build_contexts(
        &analysis,
        &base_blob,
        &target_blob,
        config.low_dim,
        config.high_dim,
        &lanes,
    );

    if contexts.is_empty() {
        return Err("no assignable word offsets found".into());
    }

    let baseline = simulate_patch(
        &contexts,
        &base_blob,
        &target_blob,
        config.predict_mode,
        config.low_dim,
        config.high_dim,
        config.target_dim,
        config.tile_size,
        &HashMap::new(),
        &HashMap::new(),
    );

    let mut chosen_offset_rules: HashMap<usize, RuleChoice> = HashMap::new();
    let mut notes: Vec<OffsetNote> = Vec::new();

    for ctx in &contexts {
        let (baseline_word_mismatch, best) = propose_offset_rule(
            ctx,
            config.predict_mode,
            config.low_dim,
            config.high_dim,
            config.target_dim,
            config.tile_size,
        );

        if let Some(best_eval) = best {
            let rule_out = rule_to_output(&best_eval.rule, true, true);
            chosen_offset_rules.insert(ctx.offset, best_eval.rule.clone());
            notes.push(OffsetNote {
                offset: ctx.offset,
                lane: ctx.lane.as_str().to_string(),
                residue: ctx.residue,
                baseline_word_mismatch_bytes: baseline_word_mismatch,
                selected: true,
                selected_rule: Some(rule_out),
                selected_mismatch_bytes: Some(best_eval.mismatch_bytes),
            });
        } else {
            notes.push(OffsetNote {
                offset: ctx.offset,
                lane: ctx.lane.as_str().to_string(),
                residue: ctx.residue,
                baseline_word_mismatch_bytes: baseline_word_mismatch,
                selected: false,
                selected_rule: None,
                selected_mismatch_bytes: None,
            });
        }
    }

    notes.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.lane.cmp(&b.lane)));

    let mut residue_to_offsets: BTreeMap<(LaneKey, i64), Vec<usize>> = BTreeMap::new();
    for ctx in &contexts {
        if let Some(residue) = ctx.residue {
            residue_to_offsets
                .entry((ctx.lane, residue))
                .or_default()
                .push(ctx.offset);
        }
    }

    let mut residue_rules: HashMap<(LaneKey, i64), RuleChoice> = HashMap::new();

    for ((lane, residue), offs) in residue_to_offsets {
        if offs.is_empty() {
            continue;
        }

        let mut signatures: Vec<RuleSignature> = Vec::new();
        let mut chosen_for_all = true;
        let mut sample_rule: Option<RuleChoice> = None;

        for off in &offs {
            let Some(rule) = chosen_offset_rules.get(off) else {
                chosen_for_all = false;
                break;
            };
            signatures.push(rule_signature(rule));
            if sample_rule.is_none() {
                sample_rule = Some(rule.clone());
            }
        }

        if !chosen_for_all || signatures.is_empty() {
            continue;
        }

        let first = signatures[0].clone();
        if !signatures.iter().all(|sig| sig == &first) {
            continue;
        }

        let Some(mut sample) = sample_rule else {
            continue;
        };
        sample.offset = None;
        sample.residue = Some(residue);

        residue_rules.insert((lane, residue), sample);
        for off in offs {
            chosen_offset_rules.remove(&off);
        }
    }

    let improved = simulate_patch(
        &contexts,
        &base_blob,
        &target_blob,
        config.predict_mode,
        config.low_dim,
        config.high_dim,
        config.target_dim,
        config.tile_size,
        &chosen_offset_rules,
        &residue_rules,
    );

    let mut offset_rule_rows: Vec<(usize, FieldSpecRuleOut)> = chosen_offset_rules
        .iter()
        .map(|(off, rule)| (*off, rule_to_output(rule, true, false)))
        .collect();
    offset_rule_rows.sort_by_key(|(off, _)| *off);

    let mut residue_rule_rows: Vec<((LaneKey, i64), FieldSpecRuleOut)> = residue_rules
        .iter()
        .map(|(k, rule)| (*k, rule_to_output(rule, false, true)))
        .collect();
    residue_rule_rows.sort_by(|a, b| a.0.cmp(&b.0));

    let spec = FieldSpecOutput {
        offset_rules: offset_rule_rows.into_iter().map(|(_, row)| row).collect(),
        residue_rules: residue_rule_rows.into_iter().map(|(_, row)| row).collect(),
    };

    ensure_parent_dir(&config.out_spec)?;
    fs::write(
        &config.out_spec,
        serde_json::to_string_pretty(&spec)? + "\n",
    )?;

    if let Some(out_patchspec) = &config.out_patchspec {
        ensure_parent_dir(out_patchspec)?;
        let mut lines = vec![
            "# emitted by word_field_spec_v2.rs".to_string(),
            format!(
                "# mode={} target_dim={} low_dim={} high_dim={}",
                config.predict_mode.as_str(),
                config.target_dim,
                config.low_dim,
                config.high_dim
            ),
            format!("# lane_priority={}", config.lane_priority),
            String::new(),
        ];
        for off in &improved.changed_from_base {
            lines.push(format!(
                "{} {} 0x{:02x}",
                base_blob.len(),
                off,
                improved.patched[*off]
            ));
        }
        fs::write(out_patchspec, lines.join("\n") + "\n")?;
    }

    let report = ReportOut {
        analysis_json: config.analysis_json.clone(),
        base_exec: config.base_exec.clone(),
        target_exec: config.target_exec.clone(),
        chunk_index: config.chunk_index,
        predict_mode: config.predict_mode.as_str().to_string(),
        lane_priority: config.lane_priority.clone(),
        dims: DimsOut {
            low: config.low_dim,
            high: config.high_dim,
            target: config.target_dim,
            tile_size: config.tile_size,
        },
        assigned_word_offsets: contexts.len(),
        baseline: baseline.summary.clone(),
        with_v2_spec: improved.summary.clone(),
        v2_changed_byte_count: improved.changed_from_base.len(),
        residue_rule_count: spec.residue_rules.len(),
        offset_rule_count: spec.offset_rules.len(),
        per_offset_notes: notes,
        out_spec: config.out_spec.clone(),
        out_patchspec: config.out_patchspec.clone(),
    };

    if let Some(out_report) = &config.out_report {
        ensure_parent_dir(out_report)?;
        fs::write(out_report, serde_json::to_string_pretty(&report)? + "\n")?;
    }

    println!("Wrote spec: {}", config.out_spec);
    if let Some(out_patchspec) = &config.out_patchspec {
        println!("Wrote patchspec: {}", out_patchspec);
    }
    println!(
        "baseline_mismatch={} v2_mismatch={} residue_rules={} offset_rules={} changed_bytes={}",
        report.baseline.mismatch_vs_target,
        report.with_v2_spec.mismatch_vs_target,
        report.residue_rule_count,
        report.offset_rule_count,
        report.v2_changed_byte_count,
    );

    Ok(())
}

fn main() {
    let config = match parse_args() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("error: {}", err);
            std::process::exit(2);
        }
    };

    if let Err(err) = run(config) {
        eprintln!("error: {}", err);
        std::process::exit(1);
    }
}
