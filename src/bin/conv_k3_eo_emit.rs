use coral_usb_oxidized::extract_instruction_chunk_from_serialized_executable;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct FamilySpec {
    schema_version: u32,
    family_id: String,
    #[serde(default)]
    family_mode: Option<String>,
    #[serde(default)]
    same_product: Option<usize>,
    #[serde(default)]
    fixed_height: Option<usize>,
    anchor_height: usize,
    anchor_width: usize,
    eo_payload_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: String,
    bias: bool,
    asset_root: String,
    regimes: Vec<RegimeSpec>,
}

#[derive(Debug, Deserialize)]
struct RegimeSpec {
    name: String,
    channels: usize,
    anchor_compiled_model: String,
    anchor_uncompiled_model: String,
    anchor_metadata: String,
    #[serde(default)]
    anchor_executable: Option<String>,
    #[serde(default)]
    field_runtime: Option<FieldRuntimeSpec>,
    targets: Vec<TargetSpec>,
}

#[derive(Debug, Deserialize)]
struct FieldRuntimeSpec {
    analysis_json: String,
    low_dim: i64,
    high_dim: i64,
    #[serde(default)]
    mid_dim: Option<i64>,
    #[serde(default = "default_predict_mode")]
    predict_mode: String,
    #[serde(default = "default_tile_size")]
    tile_size: i64,
    #[serde(default)]
    chunk_index: usize,
    #[serde(default = "default_lane_priority")]
    lane_priority: String,
}

#[derive(Debug, Deserialize)]
struct TargetSpec {
    height: usize,
    width: usize,
    target_model: String,
    target_compiled_model: String,
    target_metadata: String,
    source_kind: String,
    #[serde(default)]
    rules: Vec<[usize; 2]>,
    #[serde(default)]
    field_spec_json: Option<String>,
    #[serde(default)]
    lookup_rules: Vec<[usize; 2]>,
}

#[derive(Debug, Serialize)]
struct EmitReport {
    family_id: String,
    asset_root: String,
    regime_name: String,
    channels: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    family_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    same_product: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixed_height: Option<usize>,
    anchor_height: usize,
    anchor_width: usize,
    target_height: usize,
    target_width: usize,
    eo_payload_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: String,
    bias: bool,
    source_kind: String,
    rule_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    field_rule_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lookup_rule_count: Option<usize>,
    anchor_compiled_model: String,
    anchor_uncompiled_model: String,
    anchor_metadata: String,
    target_model: String,
    target_compiled_model: String,
    target_metadata: String,
}

#[derive(Debug, Deserialize, Default)]
struct AnalysisReport {
    #[serde(default)]
    lane16: LaneAnalysis,
    #[serde(default)]
    lane32: LaneAnalysis,
}

#[derive(Debug, Deserialize, Default)]
struct LaneAnalysis {
    #[serde(default)]
    top_groups: Vec<GroupAnalysis>,
}

#[derive(Debug, Deserialize, Default)]
struct GroupAnalysis {
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
}

#[derive(Debug, Deserialize, Default)]
struct GroupValueSeries {
    dim: i64,
    #[serde(default)]
    values: Vec<i64>,
}

#[derive(Debug, Deserialize, Default)]
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
}

#[derive(Debug, Deserialize, Default)]
struct FieldSpecOutput {
    #[serde(default)]
    offset_rules: Vec<FieldSpecRule>,
    #[serde(default)]
    residue_rules: Vec<FieldSpecRule>,
}

#[derive(Debug, Deserialize, Clone)]
struct FieldSpecRule {
    lane: String,
    #[serde(default)]
    residue: Option<i64>,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    policy: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    div: Option<i64>,
    #[serde(default)]
    domain: Option<String>,
    #[serde(default)]
    bits: Option<u8>,
    #[serde(default)]
    bit_range: Option<[u8; 2]>,
}

#[derive(Debug)]
struct PerOffsetRow {
    offset: usize,
    stride_residue: Option<i64>,
    values_by_dim: Vec<ScalarValueByDim>,
    best_formula: FormulaFit,
}

#[derive(Debug, Clone, Copy)]
enum LanePick {
    Lane16,
    Lane32,
}

impl LanePick {
    fn key(self) -> &'static str {
        match self {
            Self::Lane16 => "lane16",
            Self::Lane32 => "lane32",
        }
    }

    fn bytes(self) -> usize {
        match self {
            Self::Lane16 => 2,
            Self::Lane32 => 4,
        }
    }
}

fn default_tile_size() -> i64 {
    64
}

fn default_predict_mode() -> String {
    "threepoint".to_string()
}

fn default_lane_priority() -> String {
    "lane32,lane16".to_string()
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --family-spec PATH --channels N --target-height N [--target-width N] [--out-patchspec PATH] [--out-report PATH]"
    );
}

fn next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    *idx += 1;
    if *idx >= args.len() {
        return Err(format!("{flag} requires a value"));
    }
    Ok(args[*idx].clone())
}

fn resolve_relative(base: &Path, rel: &str) -> String {
    base.join(rel).display().to_string()
}

fn parse_lane_priority(value: &str) -> Result<Vec<LanePick>, Box<dyn Error>> {
    let mut out = Vec::new();
    for token in value.split(',') {
        match token.trim() {
            "" => {}
            "lane16" => out.push(LanePick::Lane16),
            "lane32" => out.push(LanePick::Lane32),
            other => {
                return Err(format!("invalid lane priority token: {other}").into());
            }
        }
    }
    if out.is_empty() {
        return Err("empty lane priority".into());
    }
    Ok(out)
}

fn load_base_chunk(path: &Path, chunk_index: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let blob = fs::read(path)?;
    Ok(extract_instruction_chunk_from_serialized_executable(&blob, chunk_index)?)
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

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut m = [[0.0f64; 4]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = a[r][c];
        }
        m[r][3] = b[r];
    }
    for col in 0..3 {
        let mut pivot = col;
        for r in (col + 1)..3 {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        if m[pivot][col].abs() <= 1e-12 {
            return None;
        }
        if pivot != col {
            m.swap(col, pivot);
        }
        let p = m[col][col];
        for j in col..4 {
            m[col][j] /= p;
        }
        for r in 0..3 {
            if r == col {
                continue;
            }
            let factor = m[r][col];
            if factor.abs() <= 1e-12 {
                continue;
            }
            for j in col..4 {
                m[r][j] -= factor * m[col][j];
            }
        }
    }
    Some([m[0][3], m[1][3], m[2][3]])
}

fn fit_three_point_predict(
    lo_x: f64,
    lo_y: f64,
    mid_x: f64,
    mid_y: f64,
    hi_x: f64,
    hi_y: f64,
    target_x: f64,
) -> i128 {
    if (lo_y - mid_y).abs() <= 1e-12 && (mid_y - hi_y).abs() <= 1e-12 {
        return lo_y.round() as i128;
    }
    if (hi_x - lo_x).abs() > 1e-12 {
        let mid_lin = lo_y + ((hi_y - lo_y) * ((mid_x - lo_x) / (hi_x - lo_x)));
        if (mid_lin - mid_y).abs() <= 1e-9 {
            let yt = lo_y + ((hi_y - lo_y) * ((target_x - lo_x) / (hi_x - lo_x)));
            return yt.round() as i128;
        }
    }
    let mat = [
        [lo_x * lo_x, lo_x, 1.0],
        [mid_x * mid_x, mid_x, 1.0],
        [hi_x * hi_x, hi_x, 1.0],
    ];
    if let Some(sol) = solve_3x3(mat, [lo_y, mid_y, hi_y]) {
        return (sol[0] * target_x * target_x + sol[1] * target_x + sol[2]).round() as i128;
    }
    if (hi_x - lo_x).abs() > 1e-12 {
        let yt = lo_y + ((hi_y - lo_y) * ((target_x - lo_x) / (hi_x - lo_x)));
        yt.round() as i128
    } else {
        lo_y.round() as i128
    }
}

fn decode_domain_value(v: u64, bits: u8, domain: &str) -> f64 {
    match domain {
        "s" => to_signed(v, bits) as f64,
        _ => v as f64,
    }
}

fn encode_domain_value(v: i128, bits: u8, domain: &str) -> i128 {
    match domain {
        "s" => from_signed(v as i64, bits) as i128,
        "mod" => {
            let ring = 1i128 << bits;
            v.rem_euclid(ring)
        }
        _ => v,
    }
}

fn param_i64(params: &Map<String, Value>, key: &str, default: i64) -> i64 {
    params
        .get(key)
        .and_then(|v| match v {
            Value::Number(n) => n.as_i64().or_else(|| n.as_u64().map(|x| x as i64)),
            Value::String(s) => s.parse::<i64>().ok(),
            _ => None,
        })
        .unwrap_or(default)
}

fn param_f64(params: &Map<String, Value>, key: &str, default: f64) -> f64 {
    params
        .get(key)
        .and_then(|v| match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.parse::<f64>().ok(),
            _ => None,
        })
        .unwrap_or(default)
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

    let mut filtered = Vec::new();
    for cand in cands {
        if let Some(model) = model_override {
            if cand.model != model {
                continue;
            }
        }
        if let Some(div) = div_override {
            if cand.model != "tile2div-linear" || param_i64(&cand.params, "div", -1) != div {
                continue;
            }
        }
        filtered.push(cand);
    }
    if filtered.is_empty() {
        return (best.model.clone(), best.params.clone());
    }

    filtered.sort_by(|a, b| {
        b.exact_ratio
            .partial_cmp(&a.exact_ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.mae.partial_cmp(&b.mae).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.model.cmp(&b.model))
    });
    (filtered[0].model.clone(), filtered[0].params.clone())
}

fn pick_formula(best: &FormulaFit, mode: &str) -> (String, Map<String, Value>) {
    if mode == "best" {
        return (best.model.clone(), best.params.clone());
    }
    let mut exact: Vec<&FormulaFit> = best
        .top_candidates
        .iter()
        .filter(|cand| (cand.exact_ratio - 1.0).abs() <= 1e-9 && cand.mae.abs() <= 1e-9)
        .collect();
    if exact.is_empty() {
        return (best.model.clone(), best.params.clone());
    }
    exact.sort_by(|a, b| {
        let rank_model = |m: &str| match m {
            "tile2div-linear" => 0,
            "tile-linear" => 1,
            _ => 2,
        };
        let a_a = param_f64(&a.params, "a", 0.0);
        let a_b = param_f64(&a.params, "b", 0.0);
        let b_a = param_f64(&b.params, "a", 0.0);
        let b_b = param_f64(&b.params, "b", 0.0);
        let a_int_err = (a_a - a_a.round()).abs() + (a_b - a_b.round()).abs();
        let b_int_err = (b_a - b_a.round()).abs() + (b_b - b_b.round()).abs();
        rank_model(&a.model)
            .cmp(&rank_model(&b.model))
            .then_with(|| a_int_err.partial_cmp(&b_int_err).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| param_i64(&a.params, "div", 1).cmp(&param_i64(&b.params, "div", 1)))
    });
    (exact[0].model.clone(), exact[0].params.clone())
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
    mode: &str,
    domain: &str,
    bits: u8,
) -> i128 {
    if mode == "best" {
        let tt = tiles(target_dim, tile_size) as f64;
        match model {
            "const" => return param_f64(params, "c", low_val as f64).round() as i128,
            "tile-linear" => {
                return (param_f64(params, "a", 0.0) * tt + param_f64(params, "b", low_val as f64))
                    .round() as i128
            }
            "tile-quadratic" => {
                let a = param_f64(params, "a", 0.0);
                let b = param_f64(params, "b", 0.0);
                let c = param_f64(params, "c", low_val as f64);
                return (a * tt * tt + b * tt + c).round() as i128;
            }
            "tile2div-linear" => {
                let div = param_i64(params, "div", 1).max(1);
                let x = ((tiles(target_dim, tile_size) * tiles(target_dim, tile_size)) / div) as f64;
                return (param_f64(params, "a", 0.0) * x + param_f64(params, "b", low_val as f64))
                    .round() as i128;
            }
            _ => {}
        }
    }
    let tl = tiles(low_dim, tile_size);
    let th = tiles(high_dim, tile_size);
    let tt = tiles(target_dim, tile_size);
    match model {
        "const" => low_val as i128,
        "tile-linear" | "tile-quadratic" => {
            interp_domain(low_val, high_val, tl as f64, th as f64, tt as f64, bits, domain)
        }
        "tile2div-linear" => {
            let div = param_i64(params, "div", 1).max(1);
            interp_domain(
                low_val,
                high_val,
                ((tl * tl) / div) as f64,
                ((th * th) / div) as f64,
                ((tt * tt) / div) as f64,
                bits,
                domain,
            )
        }
        _ => low_val as i128,
    }
}

fn read_word_le(bytes: &[u8], offset: usize, lane_bytes: usize) -> u64 {
    let mut out = 0u64;
    for idx in 0..lane_bytes {
        out |= (bytes[offset + idx] as u64) << (idx * 8);
    }
    out
}

fn write_word_le(bytes: &mut [u8], offset: usize, lane_bytes: usize, value: u64) {
    for idx in 0..lane_bytes {
        bytes[offset + idx] = ((value >> (idx * 8)) & 0xff) as u8;
    }
}

fn iter_per_offsets(lane: &LaneAnalysis) -> Vec<PerOffsetRow> {
    let mut out = Vec::new();
    for group in &lane.top_groups {
        if !group.per_offset_fits.is_empty() {
            for per in &group.per_offset_fits {
                out.push(PerOffsetRow {
                    offset: per.offset,
                    stride_residue: group.stride_residue,
                    values_by_dim: per.values_by_dim.clone(),
                    best_formula: per.best_formula.clone(),
                });
            }
            continue;
        }
        if group.offsets.is_empty() || group.values_by_dim.is_empty() {
            continue;
        }
        let mut dim_to_values = HashMap::new();
        for row in &group.values_by_dim {
            dim_to_values.insert(row.dim, row.values.clone());
        }
        for (idx, off) in group.offsets.iter().copied().enumerate() {
            let mut vals = Vec::new();
            for (dim, series) in &dim_to_values {
                if idx < series.len() {
                    vals.push(ScalarValueByDim {
                        dim: *dim,
                        value: series[idx],
                    });
                }
            }
            if !vals.is_empty() {
                out.push(PerOffsetRow {
                    offset: off,
                    stride_residue: group.stride_residue,
                    values_by_dim: vals,
                    best_formula: group.best_formula.clone(),
                });
            }
        }
    }
    out
}

fn values_map(values_by_dim: &[ScalarValueByDim]) -> HashMap<i64, i64> {
    let mut out = HashMap::new();
    for row in values_by_dim {
        out.insert(row.dim, row.value);
    }
    out
}

fn match_rule<'a>(
    field_spec: &'a FieldSpecOutput,
    lane: &str,
    offset: usize,
    residue: Option<i64>,
) -> Option<&'a FieldSpecRule> {
    for rule in &field_spec.offset_rules {
        if rule.lane == lane && rule.offset == Some(offset) {
            return Some(rule);
        }
    }
    if let Some(residue) = residue {
        for rule in &field_spec.residue_rules {
            if rule.lane == lane && rule.residue == Some(residue) {
                return Some(rule);
            }
        }
    }
    None
}

fn emit_from_field_spec(
    analysis: &AnalysisReport,
    field_spec: &FieldSpecOutput,
    base_chunk: &[u8],
    runtime: &FieldRuntimeSpec,
    target_dim: i64,
) -> Result<Vec<[usize; 2]>, Box<dyn Error>> {
    let mut patched = base_chunk.to_vec();
    let mut assigned_offsets: HashSet<usize> = HashSet::new();
    for lane in parse_lane_priority(&runtime.lane_priority)? {
        let lane_obj = match lane {
            LanePick::Lane16 => &analysis.lane16,
            LanePick::Lane32 => &analysis.lane32,
        };
        for per in iter_per_offsets(lane_obj) {
            let off = per.offset;
            if assigned_offsets.contains(&off) || off + lane.bytes() > base_chunk.len() {
                continue;
            }
            let vals = values_map(&per.values_by_dim);
            let Some(low_val) = vals.get(&runtime.low_dim).copied() else {
                continue;
            };
            let Some(high_val) = vals.get(&runtime.high_dim).copied() else {
                continue;
            };
            let override_rule = match_rule(field_spec, lane.key(), off, per.stride_residue);

            let mut mode_use = runtime.predict_mode.clone();
            if let Some(rule) = override_rule {
                if let Some(policy) = &rule.policy {
                    mode_use = policy.clone();
                }
            }

            let best = &per.best_formula;
            let (model, params) = if mode_use == "best" {
                pick_best_candidate(
                    best,
                    override_rule.and_then(|r| r.model.as_deref()),
                    override_rule.and_then(|r| r.div),
                )
            } else if let Some(rule) = override_rule {
                if let Some(model) = &rule.model {
                    let mut params = Map::new();
                    if let Some(div) = rule.div {
                        params.insert("div".to_string(), Value::from(div));
                    }
                    (model.clone(), params)
                } else {
                    pick_formula(best, &mode_use)
                }
            } else {
                pick_formula(best, &mode_use)
            };

            let lane_bits = (lane.bytes() * 8) as u8;
            let bit_range = override_rule.and_then(|r| r.bit_range);
            let (low_eval, high_eval, mid_eval, field_width, mut bits) = if let Some([lo, hi]) = bit_range
            {
                let width = hi - lo + 1;
                let mask = bits_mask(width);
                (
                    ((low_val as u64) >> lo) & mask,
                    ((high_val as u64) >> lo) & mask,
                    runtime
                        .mid_dim
                        .and_then(|mid_dim| vals.get(&mid_dim).copied())
                        .map(|v| ((v as u64) >> lo) & mask),
                    width,
                    width,
                )
            } else {
                (
                    low_val as u64,
                    high_val as u64,
                    runtime.mid_dim.and_then(|mid_dim| vals.get(&mid_dim).copied()).map(|v| v as u64),
                    lane_bits,
                    lane_bits,
                )
            };
            if let Some(bits_override) = override_rule.and_then(|r| r.bits) {
                bits = bits_override;
            }
            let domain = override_rule
                .and_then(|r| r.domain.as_deref())
                .unwrap_or("u")
                .to_string();

            let pred_word = if mode_use == "threepoint" {
                if let (Some(mid_dim), Some(mid_value)) = (runtime.mid_dim, mid_eval) {
                    let pred = encode_domain_value(
                        fit_three_point_predict(
                            tiles(runtime.low_dim, runtime.tile_size) as f64,
                            decode_domain_value(low_eval, bits, &domain),
                            tiles(mid_dim, runtime.tile_size) as f64,
                            decode_domain_value(mid_value, bits, &domain),
                            tiles(runtime.high_dim, runtime.tile_size) as f64,
                            decode_domain_value(high_eval, bits, &domain),
                            tiles(target_dim, runtime.tile_size) as f64,
                        ),
                        bits,
                        &domain,
                    );
                    let pred_field_norm = pred.rem_euclid(1i128 << field_width) as u64;
                    if let Some([lo, _hi]) = bit_range {
                        let base_word = read_word_le(base_chunk, off, lane.bytes());
                        let field_mask = bits_mask(field_width);
                        (base_word & !(field_mask << lo)) | ((pred_field_norm & field_mask) << lo)
                    } else {
                        pred_field_norm
                    }
                } else {
                    let pred_field = predict_value(
                        &model,
                        &params,
                        runtime.low_dim,
                        low_eval,
                        runtime.high_dim,
                        high_eval,
                        target_dim,
                        runtime.tile_size,
                        "best",
                        &domain,
                        bits,
                    );
                    let pred_field_norm = pred_field.rem_euclid(1i128 << field_width) as u64;
                    if let Some([lo, _hi]) = bit_range {
                        let base_word = read_word_le(base_chunk, off, lane.bytes());
                        let field_mask = bits_mask(field_width);
                        (base_word & !(field_mask << lo)) | ((pred_field_norm & field_mask) << lo)
                    } else {
                        pred_field_norm
                    }
                }
            } else {
                let pred_field = predict_value(
                    &model,
                    &params,
                    runtime.low_dim,
                    low_eval,
                    runtime.high_dim,
                    high_eval,
                    target_dim,
                    runtime.tile_size,
                    &mode_use,
                    &domain,
                    bits,
                );
                let pred_field_norm = pred_field.rem_euclid(1i128 << field_width) as u64;
                if let Some([lo, _hi]) = bit_range {
                    let base_word = read_word_le(base_chunk, off, lane.bytes());
                    let field_mask = bits_mask(field_width);
                    (base_word & !(field_mask << lo)) | ((pred_field_norm & field_mask) << lo)
                } else {
                    pred_field_norm
                }
            };
            write_word_le(
                &mut patched,
                off,
                lane.bytes(),
                ((pred_word as u128) % (1u128 << lane_bits)) as u64,
            );
            assigned_offsets.insert(off);
        }
    }
    let mut out = Vec::new();
    for idx in 0..base_chunk.len() {
        if patched[idx] != base_chunk[idx] {
            out.push([idx, patched[idx] as usize]);
        }
    }
    Ok(out)
}

fn write_patchspec(path: &Path, payload_len: usize, rules: &[[usize; 2]]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if rules.is_empty() {
        fs::write(path, "# no EO patch required for this target\n")?;
        return Ok(());
    }
    let mut lines = Vec::with_capacity(rules.len() + 1);
    lines.push("# emitted by conv_k3_eo_emit".to_string());
    for [offset, value] in rules {
        lines.push(format!("{payload_len} {offset} {value}"));
    }
    fs::write(path, lines.join("\n") + "\n")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "conv_k3_eo_emit".to_string());
    if args.len() == 1 {
        usage(&program);
        return Err("missing required args".into());
    }

    let mut family_spec_path: Option<PathBuf> = None;
    let mut channels: Option<usize> = None;
    let mut target_height: Option<usize> = None;
    let mut target_width: Option<usize> = None;
    let mut out_patchspec: Option<PathBuf> = None;
    let mut out_report: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                return Ok(());
            }
            "--family-spec" => {
                family_spec_path = Some(PathBuf::from(next_arg(&args, &mut i, "--family-spec")?))
            }
            "--channels" => {
                channels = Some(
                    next_arg(&args, &mut i, "--channels")?
                        .parse::<usize>()
                        .map_err(|e| format!("--channels invalid integer: {e}"))?,
                )
            }
            "--target-height" => {
                target_height = Some(
                    next_arg(&args, &mut i, "--target-height")?
                        .parse::<usize>()
                        .map_err(|e| format!("--target-height invalid integer: {e}"))?,
                )
            }
            "--target-width" => {
                target_width = Some(
                    next_arg(&args, &mut i, "--target-width")?
                        .parse::<usize>()
                        .map_err(|e| format!("--target-width invalid integer: {e}"))?,
                )
            }
            "--out-patchspec" => {
                out_patchspec = Some(PathBuf::from(next_arg(&args, &mut i, "--out-patchspec")?))
            }
            "--out-report" => {
                out_report = Some(PathBuf::from(next_arg(&args, &mut i, "--out-report")?))
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    let family_spec_path = family_spec_path.ok_or("missing --family-spec")?;
    let channels = channels.ok_or("missing --channels")?;
    let target_height = target_height.ok_or("missing --target-height")?;
    let spec_dir = family_spec_path
        .parent()
        .ok_or("family spec path must have a parent directory")?;
    let spec: FamilySpec = serde_json::from_slice(&fs::read(&family_spec_path)?)?;
    if spec.schema_version != 1 && spec.schema_version != 2 {
        return Err(format!("unsupported schema_version {}", spec.schema_version).into());
    }

    let regime = spec
        .regimes
        .iter()
        .find(|r| r.channels == channels)
        .ok_or_else(|| format!("unsupported channels={} in {}", channels, family_spec_path.display()))?;

    let matching_targets: Vec<&TargetSpec> = regime
        .targets
        .iter()
        .filter(|t| t.height == target_height && target_width.map(|w| w == t.width).unwrap_or(true))
        .collect();
    let target = match matching_targets.as_slice() {
        [] => {
            return Err(format!(
                "unsupported target height={}{} for regime {}",
                target_height,
                target_width
                    .map(|w| format!(" width={}", w))
                    .unwrap_or_default(),
                regime.name
            )
            .into())
        }
        [single] => *single,
        _ => {
            return Err(format!(
                "target height={} is ambiguous for regime {}; pass --target-width",
                target_height, regime.name
            )
            .into())
        }
    };

    let mut emitted_rules = if spec.schema_version == 1 {
        if let Some(same_product) = spec.same_product {
            if target.height
                .checked_mul(target.width)
                .ok_or("target area overflow")?
                != same_product
            {
                return Err(format!(
                    "target area mismatch for {}: {}x{} != same_product {}",
                    regime.name, target.height, target.width, same_product
                )
                .into());
            }
        }
        target.rules.clone()
    } else {
        let family_mode = spec.family_mode.as_deref().unwrap_or("fixed_height_band");
        if family_mode != "fixed_height_band" {
            return Err(format!("unsupported schema_version=2 family_mode={family_mode}").into());
        }
        if spec.fixed_height != Some(target.height) {
            return Err(format!(
                "target height {} does not match fixed_height {:?}",
                target.height, spec.fixed_height
            )
            .into());
        }
        let runtime = regime
            .field_runtime
            .as_ref()
            .ok_or("schema_version=2 regime missing field_runtime")?;
        if target.source_kind == "noop" {
            Vec::new()
        } else {
        let field_spec_rel = target
            .field_spec_json
            .as_ref()
            .ok_or("schema_version=2 target missing field_spec_json")?;
        let anchor_exec_rel = regime
            .anchor_executable
            .as_ref()
            .ok_or("schema_version=2 regime missing anchor_executable")?;
        let analysis: AnalysisReport =
            serde_json::from_slice(&fs::read(spec_dir.join(&runtime.analysis_json))?)?;
        let field_spec: FieldSpecOutput =
            serde_json::from_slice(&fs::read(spec_dir.join(field_spec_rel))?)?;
        let base_chunk = load_base_chunk(&spec_dir.join(anchor_exec_rel), runtime.chunk_index)?;
        let mut rules = emit_from_field_spec(&analysis, &field_spec, &base_chunk, runtime, target.width as i64)?;
        for rule in &target.lookup_rules {
            rules.push(*rule);
        }
        rules.sort_by_key(|row| row[0]);
        rules
        }
    };

    emitted_rules.sort_by_key(|row| row[0]);

    if let Some(path) = out_patchspec.as_ref() {
        write_patchspec(path, spec.eo_payload_len, &emitted_rules)?;
    }

    let (field_rule_count, lookup_rule_count) = if spec.schema_version == 2 {
        let lookup = target.lookup_rules.len();
        (Some(emitted_rules.len().saturating_sub(lookup)), Some(lookup))
    } else {
        (None, None)
    };

    let report = EmitReport {
        family_id: spec.family_id,
        asset_root: spec.asset_root,
        regime_name: regime.name.clone(),
        channels,
        family_mode: spec.family_mode,
        same_product: spec.same_product,
        fixed_height: spec.fixed_height,
        anchor_height: spec.anchor_height,
        anchor_width: spec.anchor_width,
        target_height: target.height,
        target_width: target.width,
        eo_payload_len: spec.eo_payload_len,
        kernel_size: spec.kernel_size,
        stride: spec.stride,
        padding: spec.padding,
        bias: spec.bias,
        source_kind: target.source_kind.clone(),
        rule_count: emitted_rules.len(),
        field_rule_count,
        lookup_rule_count,
        anchor_compiled_model: resolve_relative(spec_dir, &regime.anchor_compiled_model),
        anchor_uncompiled_model: resolve_relative(spec_dir, &regime.anchor_uncompiled_model),
        anchor_metadata: resolve_relative(spec_dir, &regime.anchor_metadata),
        target_model: resolve_relative(spec_dir, &target.target_model),
        target_compiled_model: resolve_relative(spec_dir, &target.target_compiled_model),
        target_metadata: resolve_relative(spec_dir, &target.target_metadata),
    };

    if let Some(path) = out_report.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }

    println!(
        "family_id={} regime={} channels={} target={}x{} rule_count={} source={}",
        report.family_id,
        report.regime_name,
        report.channels,
        report.target_height,
        report.target_width,
        report.rule_count,
        report.source_kind
    );
    if report.rule_count == 0 {
        println!("note=no_eo_patch_needed");
    }

    Ok(())
}
