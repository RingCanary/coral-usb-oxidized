use crate::fit::{
    bits_mask, decode_domain_value, encode_domain_value, fit_three_point_predict, interp_domain,
    tiles,
};
use crate::model::{FormulaFit, PredictMode, RuleChoice, WordContext};
use crate::util::{param_f64, param_i64};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::HashSet;

pub fn model_complexity(model: &str) -> i64 {
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

pub fn pick_best_candidate(
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
        if let Some(mo) = model_override {
            if cand.model != mo {
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
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.mae.partial_cmp(&b.mae).unwrap_or(Ordering::Equal))
            .then_with(|| formula_effective_complexity(a).cmp(&formula_effective_complexity(b)))
            .then_with(|| a.model.cmp(&b.model))
    });

    (filtered[0].model.clone(), filtered[0].params.clone())
}

pub fn pick_formula(best: &FormulaFit, mode: PredictMode) -> (String, Map<String, Value>) {
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
            let a_a = param_f64(&a.params, "a", 0.0);
            let a_b = param_f64(&a.params, "b", 0.0);
            let b_a = param_f64(&b.params, "a", 0.0);
            let b_b = param_f64(&b.params, "b", 0.0);
            let a_int_err = (a_a - a_a.round()).abs() + (a_b - a_b.round()).abs();
            let b_int_err = (b_a - b_a.round()).abs() + (b_b - b_b.round()).abs();
            rank_model(&a.model)
                .cmp(&rank_model(&b.model))
                .then_with(|| a_int_err.partial_cmp(&b_int_err).unwrap_or(Ordering::Equal))
                .then_with(|| a_a.abs().partial_cmp(&b_a.abs()).unwrap_or(Ordering::Equal))
                .then_with(|| param_i64(&a.params, "div", 1).cmp(&param_i64(&b.params, "div", 1)))
                .then_with(|| a.model.cmp(&b.model))
        });
        return (exact[0].model.clone(), exact[0].params.clone());
    }

    (best.model.clone(), best.params.clone())
}

pub fn predict_value(
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
                let x =
                    ((tiles(target_dim, tile_size) * tiles(target_dim, tile_size)) / div) as f64;
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
        "tile-linear" | "tile-quadratic" => interp_domain(
            low_val, high_val, tl as f64, th as f64, tt as f64, bits, domain,
        ),
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

pub fn collect_endpoint_models(formula: &FormulaFit) -> Vec<(String, Option<i64>)> {
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

pub fn predict_word(
    ctx: &WordContext,
    global_mode: PredictMode,
    low_dim: i64,
    high_dim: i64,
    mid_dim: Option<i64>,
    target_dim: i64,
    tile_size: i64,
    rule: Option<&RuleChoice>,
) -> (u64, String, Map<String, Value>, String, u8) {
    if matches!(global_mode, PredictMode::Strict) && !ctx.mono_class.is_monotone() {
        return (
            ctx.base_word,
            "strict-skip".to_string(),
            Map::new(),
            "u".to_string(),
            (ctx.lane_bytes * 8) as u8,
        );
    }

    let mut mode_use = rule.and_then(|r| r.policy).unwrap_or(global_mode);
    if matches!(mode_use, PredictMode::Strict) {
        mode_use = PredictMode::Endpoint;
    }

    let lane_bits = (ctx.lane_bytes * 8) as u8;
    let bit_lo = rule.and_then(|r| r.bit_lo);
    let bit_hi = rule.and_then(|r| r.bit_hi);
    let has_bit_range = bit_lo.is_some() && bit_hi.is_some();

    let (low_eval, high_eval, mid_eval, width, mut bits) = if has_bit_range {
        let lo = bit_lo.unwrap();
        let width = bit_hi.unwrap() - lo + 1;
        let mask = bits_mask(width);
        (
            (ctx.low_val >> lo) & mask,
            (ctx.high_val >> lo) & mask,
            ctx.mid_val.map(|v| (v >> lo) & mask),
            width,
            width,
        )
    } else {
        (ctx.low_val, ctx.high_val, ctx.mid_val, lane_bits, lane_bits)
    };
    if let Some(rb) = rule.and_then(|r| r.bits) {
        bits = rb;
    }

    let domain = rule
        .and_then(|r| r.domain.as_ref())
        .map(|x| x.as_str())
        .unwrap_or("u")
        .to_string();

    if matches!(mode_use, PredictMode::ThreePoint) {
        if let (Some(mid_d), Some(mid_v)) = (mid_dim, mid_eval) {
            let (tl, tm, th, tt) = (
                tiles(low_dim, tile_size) as f64,
                tiles(mid_d, tile_size) as f64,
                tiles(high_dim, tile_size) as f64,
                tiles(target_dim, tile_size) as f64,
            );
            let lo_dec = decode_domain_value(low_eval, bits, &domain);
            let mid_dec = decode_domain_value(mid_v, bits, &domain);
            let hi_dec = decode_domain_value(high_eval, bits, &domain);
            let linear_mid = if (th - tl).abs() > 1e-12 {
                lo_dec + ((hi_dec - lo_dec) * ((tm - tl) / (th - tl)))
            } else {
                lo_dec
            };
            let mode_name =
                if (lo_dec - mid_dec).abs() <= 1e-12 && (mid_dec - hi_dec).abs() <= 1e-12 {
                    "const"
                } else if (linear_mid - mid_dec).abs() <= 1e-9 {
                    "tile-linear"
                } else {
                    "tile-quadratic"
                };
            let pred_field = encode_domain_value(
                fit_three_point_predict(tl, lo_dec, tm, mid_dec, th, hi_dec, tt),
                bits,
                &domain,
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
            return (
                ((pred_word as u128) % lane_mod) as u64,
                mode_name.to_string(),
                Map::new(),
                domain,
                bits,
            );
        }
        mode_use = PredictMode::Best;
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

    let pred_field = predict_value(
        &model, &params, low_dim, low_eval, high_dim, high_eval, target_dim, tile_size, mode_use,
        &domain, bits,
    );
    let pred_field_norm = pred_field.rem_euclid(1i128 << width) as u64;
    let pred_word = if has_bit_range {
        let lo = bit_lo.unwrap();
        let mask = bits_mask(width);
        (ctx.base_word & !(mask << lo)) | ((pred_field_norm & mask) << lo)
    } else {
        pred_field_norm
    };

    (
        ((pred_word as u128) % (1u128 << lane_bits)) as u64,
        model,
        params,
        domain,
        bits,
    )
}
