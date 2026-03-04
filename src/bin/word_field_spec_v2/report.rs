use crate::args::Config;
use crate::model::*;
use crate::predict::{collect_endpoint_models, model_complexity, predict_word};
use coral_usb_oxidized::extract_instruction_chunk_from_serialized_executable;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::Path;

pub fn parse_lane_priority(value: &str) -> Result<Vec<LaneKey>, String> {
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

fn lane_report(analysis: &AnalysisReport, lane: LaneKey) -> &LaneReport {
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
    mid_dim: Option<i64>,
    mid_blob: Option<&[u8]>,
    lanes: &[LaneKey],
) -> Vec<WordContext> {
    let mut contexts = Vec::new();
    let mut assigned_offsets: HashSet<usize> = HashSet::new();

    for lane in lanes {
        let lane_bytes = lane.lane_bytes();
        for group in &lane_report(analysis, *lane).top_groups {
            let residue = group.stride_residue;
            if !group.per_offset_fits.is_empty() {
                for per in &group.per_offset_fits {
                    let off = per.offset;
                    if assigned_offsets.contains(&off)
                        || off + lane_bytes > base_blob.len()
                        || off + lane_bytes > target_blob.len()
                    {
                        continue;
                    }

                    let vals = values_map_scalar(&per.values_by_dim);
                    let (Some(low_val), Some(high_val)) =
                        (vals.get(&low_dim).copied(), vals.get(&high_dim).copied())
                    else {
                        continue;
                    };
                    let best_formula = if is_formula_empty(&per.best_formula) {
                        group.best_formula.clone()
                    } else {
                        per.best_formula.clone()
                    };
                    let mid_val = if let Some(md) = mid_dim {
                        vals.get(&md).copied().or_else(|| {
                            mid_blob.and_then(|blob| {
                                (off + lane_bytes <= blob.len())
                                    .then(|| read_word_le(blob, off, lane_bytes))
                            })
                        })
                    } else {
                        None
                    };

                    contexts.push(WordContext {
                        offset: off,
                        lane: *lane,
                        lane_bytes,
                        residue,
                        low_val,
                        mid_val,
                        high_val,
                        base_word: read_word_le(base_blob, off, lane_bytes),
                        target_word: read_word_le(target_blob, off, lane_bytes),
                        mono_class: classify_word_monotonicity(
                            low_val, mid_val, high_val, lane_bytes,
                        ),
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
            let (Some(low_series), Some(high_series)) =
                (dim_to_values.get(&low_dim), dim_to_values.get(&high_dim))
            else {
                continue;
            };

            for (idx, off) in group.offsets.iter().copied().enumerate() {
                if assigned_offsets.contains(&off)
                    || off + lane_bytes > base_blob.len()
                    || off + lane_bytes > target_blob.len()
                    || idx >= low_series.len()
                    || idx >= high_series.len()
                {
                    continue;
                }

                let low_val = low_series[idx] as u64;
                let high_val = high_series[idx] as u64;
                let mid_val = if let Some(md) = mid_dim {
                    dim_to_values
                        .get(&md)
                        .and_then(|series| series.get(idx).copied())
                        .map(|x| x as u64)
                        .or_else(|| {
                            mid_blob.and_then(|blob| {
                                (off + lane_bytes <= blob.len())
                                    .then(|| read_word_le(blob, off, lane_bytes))
                            })
                        })
                } else {
                    None
                };

                contexts.push(WordContext {
                    offset: off,
                    lane: *lane,
                    lane_bytes,
                    residue,
                    low_val,
                    mid_val,
                    high_val,
                    base_word: read_word_le(base_blob, off, lane_bytes),
                    target_word: read_word_le(target_blob, off, lane_bytes),
                    mono_class: classify_word_monotonicity(low_val, mid_val, high_val, lane_bytes),
                    best_formula: group.best_formula.clone(),
                    group_bitfield_fits: group.bitfield_fits.clone(),
                });
                assigned_offsets.insert(off);
            }
        }
    }

    contexts
}

fn mismatch_bytes_word(pred_word: u64, target_word: u64, lane_bytes: usize) -> usize {
    let mut out = 0usize;
    for idx in 0..lane_bytes {
        if ((pred_word >> (idx * 8)) & 0xff) != ((target_word >> (idx * 8)) & 0xff) {
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

fn formula_div(formula: &FormulaFit) -> Option<i64> {
    if formula.model == "tile2div-linear" {
        formula
            .params
            .get("div")
            .and_then(crate::util::value_to_i64)
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
        let (lo, hi) = (bf.bit_range[0] as u8, bf.bit_range[1] as u8);
        if lo <= hi && hi < (ctx.lane_bytes * 8) as u8 {
            out.push((
                lo,
                hi,
                "group_bitfield".to_string(),
                bf.best_formula.clone(),
            ));
        }
    }

    let mut mism_bytes = Vec::new();
    for idx in 0..ctx.lane_bytes {
        if ((baseline_pred >> (idx * 8)) & 0xff) != ((ctx.target_word >> (idx * 8)) & 0xff) {
            mism_bytes.push(idx as u8);
        }
    }
    for b in &mism_bytes {
        let lo = b * 8;
        out.push((
            lo,
            lo + 7,
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

fn evaluate_rule_on_ctx(
    ctx: &WordContext,
    global_mode: PredictMode,
    low_dim: i64,
    high_dim: i64,
    mid_dim: Option<i64>,
    target_dim: i64,
    tile_size: i64,
    rule: RuleChoice,
) -> CandidateEval {
    let (pred_word, _, _, _, _) = predict_word(
        ctx,
        global_mode,
        low_dim,
        high_dim,
        mid_dim,
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
    mid_dim: Option<i64>,
    target_dim: i64,
    tile_size: i64,
) -> (usize, Option<CandidateEval>) {
    if matches!(global_mode, PredictMode::Strict) && !ctx.mono_class.is_monotone() {
        return (0, None);
    }

    let (baseline_pred, _, _, _, _) = predict_word(
        ctx,
        global_mode,
        low_dim,
        high_dim,
        mid_dim,
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
                    mid_dim,
                    target_dim,
                    tile_size,
                    rule,
                );
                if ev.mismatch_bytes < baseline_mismatch {
                    let sig = rule_signature(&ev.rule);
                    let keep = evals
                        .get(&sig)
                        .map(|prev| ev.mismatch_bytes < prev.mismatch_bytes)
                        .unwrap_or(true);
                    if keep {
                        evals.insert(sig, ev);
                    }
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
                        mid_dim,
                        target_dim,
                        tile_size,
                        rule,
                    );
                    if ev.mismatch_bytes < baseline_mismatch {
                        let sig = rule_signature(&ev.rule);
                        let keep = evals
                            .get(&sig)
                            .map(|prev| ev.mismatch_bytes < prev.mismatch_bytes)
                            .unwrap_or(true);
                        if keep {
                            evals.insert(sig, ev);
                        }
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
    mid_dim: Option<i64>,
    target_dim: i64,
    tile_size: i64,
    offset_rules: &HashMap<usize, RuleChoice>,
    residue_rules: &HashMap<(LaneKey, i64), RuleChoice>,
) -> PatchSimulation {
    let mut patched = base_blob.to_vec();
    let mut byte_tiers: HashMap<usize, PatchTier> = HashMap::new();

    for ctx in contexts {
        let rule_ref = offset_rules.get(&ctx.offset).or_else(|| {
            ctx.residue
                .and_then(|res| residue_rules.get(&(ctx.lane, res)))
        });
        let (pred_word, _, _, _, _) = predict_word(
            ctx,
            global_mode,
            low_dim,
            high_dim,
            mid_dim,
            target_dim,
            tile_size,
            rule_ref,
        );
        write_word_le(&mut patched, ctx.offset, ctx.lane_bytes, pred_word);

        let tier = ctx.mono_class.tier();
        for idx in 0..ctx.lane_bytes {
            let off = ctx.offset + idx;
            if ((pred_word >> (idx * 8)) & 0xff) as u8 != base_blob[off] {
                byte_tiers.insert(off, tier);
            } else {
                byte_tiers.remove(&off);
            }
        }
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
        byte_tiers,
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

fn write_patchspec(
    path: &str,
    payload_len: usize,
    predict_mode: PredictMode,
    target_dim: i64,
    low_dim: i64,
    high_dim: i64,
    lane_priority: &str,
    changed_from_base: &[usize],
    patched: &[u8],
    byte_tiers: &HashMap<usize, PatchTier>,
    tier_filter: Option<PatchTier>,
) -> Result<usize, Box<dyn Error>> {
    ensure_parent_dir(path)?;

    let mut lines = vec![
        "# emitted by word_field_spec_v2.rs".to_string(),
        format!(
            "# mode={} target_dim={} low_dim={} high_dim={}",
            predict_mode.as_str(),
            target_dim,
            low_dim,
            high_dim
        ),
        format!("# lane_priority={}", lane_priority),
    ];
    if let Some(t) = tier_filter {
        lines.push(format!("# tier_filter={}", t.as_str()));
    }
    lines.push(String::new());

    let mut emitted = 0usize;
    for off in changed_from_base {
        if let Some(filter) = tier_filter {
            let tier = byte_tiers.get(off).copied().unwrap_or(PatchTier::Unknown);
            if tier != filter {
                continue;
            }
        }
        lines.push(format!("{} {} 0x{:02x}", payload_len, off, patched[*off]));
        emitted += 1;
    }

    fs::write(path, lines.join("\n") + "\n")?;
    Ok(emitted)
}

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let analysis: AnalysisReport = serde_json::from_slice(&fs::read(&config.analysis_json)?)?;
    let base_exec_blob = fs::read(&config.base_exec)?;
    let target_exec_blob = fs::read(&config.target_exec)?;
    let base_blob =
        extract_instruction_chunk_from_serialized_executable(&base_exec_blob, config.chunk_index)?;
    let target_blob = extract_instruction_chunk_from_serialized_executable(
        &target_exec_blob,
        config.chunk_index,
    )?;

    let mid_blob: Option<Vec<u8>> = if let Some(mid_exec) = &config.mid_exec {
        let mid_exec_blob = fs::read(mid_exec)?;
        Some(extract_instruction_chunk_from_serialized_executable(
            &mid_exec_blob,
            config.chunk_index,
        )?)
    } else {
        None
    };

    if base_blob.len() != target_blob.len() {
        return Err(format!(
            "base/target chunk size mismatch: base={} target={}",
            base_blob.len(),
            target_blob.len()
        )
        .into());
    }
    if let Some(mid) = &mid_blob {
        if mid.len() != base_blob.len() {
            return Err(format!(
                "base/mid chunk size mismatch: base={} mid={}",
                base_blob.len(),
                mid.len()
            )
            .into());
        }
    }

    if matches!(config.predict_mode, PredictMode::ThreePoint) {
        if config.mid_dim == Some(config.target_dim) {
            eprintln!("warning: --mid-dim equals --target-dim; threepoint prediction is exact by construction (self-validation mode)");
        } else if config.mid_dim.is_none() {
            eprintln!("warning: --predict-mode threepoint without --mid-dim; falling back to best-mode prediction");
        }
    }

    let contexts = build_contexts(
        &analysis,
        &base_blob,
        &target_blob,
        config.low_dim,
        config.high_dim,
        config.mid_dim,
        mid_blob.as_deref(),
        &parse_lane_priority(&config.lane_priority)?,
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
        config.mid_dim,
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
            config.mid_dim,
            config.target_dim,
            config.tile_size,
        );
        if let Some(best_eval) = best {
            chosen_offset_rules.insert(ctx.offset, best_eval.rule.clone());
            notes.push(OffsetNote {
                offset: ctx.offset,
                lane: ctx.lane.as_str().to_string(),
                residue: ctx.residue,
                mono_class: ctx.mono_class.as_str().to_string(),
                patch_tier: ctx.mono_class.tier().as_str().to_string(),
                baseline_word_mismatch_bytes: baseline_word_mismatch,
                selected: true,
                selected_rule: Some(rule_to_output(&best_eval.rule, true, true)),
                selected_mismatch_bytes: Some(best_eval.mismatch_bytes),
            });
        } else {
            notes.push(OffsetNote {
                offset: ctx.offset,
                lane: ctx.lane.as_str().to_string(),
                residue: ctx.residue,
                mono_class: ctx.mono_class.as_str().to_string(),
                patch_tier: ctx.mono_class.tier().as_str().to_string(),
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
        let mut sample_rule: Option<RuleChoice> = None;
        let mut chosen_for_all = true;
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
        config.mid_dim,
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

    let safe_core_byte_count = improved
        .byte_tiers
        .values()
        .filter(|t| **t == PatchTier::SafeCore)
        .count();
    let discrete_flags_byte_count = improved
        .byte_tiers
        .values()
        .filter(|t| **t == PatchTier::DiscreteFlags)
        .count();
    let unknown_byte_count = improved
        .byte_tiers
        .values()
        .filter(|t| **t == PatchTier::Unknown)
        .count();

    if let Some(out_patchspec) = &config.out_patchspec {
        write_patchspec(
            out_patchspec,
            base_blob.len(),
            config.predict_mode,
            config.target_dim,
            config.low_dim,
            config.high_dim,
            &config.lane_priority,
            &improved.changed_from_base,
            &improved.patched,
            &improved.byte_tiers,
            None,
        )?;
    }
    if let Some(out_patchspec_safe) = &config.out_patchspec_safe {
        write_patchspec(
            out_patchspec_safe,
            base_blob.len(),
            config.predict_mode,
            config.target_dim,
            config.low_dim,
            config.high_dim,
            &config.lane_priority,
            &improved.changed_from_base,
            &improved.patched,
            &improved.byte_tiers,
            Some(PatchTier::SafeCore),
        )?;
    }
    if let Some(out_patchspec_discrete) = &config.out_patchspec_discrete {
        write_patchspec(
            out_patchspec_discrete,
            base_blob.len(),
            config.predict_mode,
            config.target_dim,
            config.low_dim,
            config.high_dim,
            &config.lane_priority,
            &improved.changed_from_base,
            &improved.patched,
            &improved.byte_tiers,
            Some(PatchTier::DiscreteFlags),
        )?;
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
        safe_core_byte_count,
        discrete_flags_byte_count,
        unknown_byte_count,
        residue_rule_count: spec.residue_rules.len(),
        offset_rule_count: spec.offset_rules.len(),
        per_offset_notes: notes,
        out_spec: config.out_spec.clone(),
        out_patchspec: config.out_patchspec.clone(),
        out_patchspec_safe: config.out_patchspec_safe.clone(),
        out_patchspec_discrete: config.out_patchspec_discrete.clone(),
    };

    if let Some(out_report) = &config.out_report {
        ensure_parent_dir(out_report)?;
        fs::write(out_report, serde_json::to_string_pretty(&report)? + "\n")?;
    }

    println!("Wrote spec: {}", config.out_spec);
    if let Some(out_patchspec) = &config.out_patchspec {
        println!("Wrote patchspec: {}", out_patchspec);
    }
    if let Some(out_patchspec_safe) = &config.out_patchspec_safe {
        println!("Wrote safe patchspec: {}", out_patchspec_safe);
    }
    if let Some(out_patchspec_discrete) = &config.out_patchspec_discrete {
        println!("Wrote discrete patchspec: {}", out_patchspec_discrete);
    }
    println!(
        "baseline_mismatch={} v2_mismatch={} residue_rules={} offset_rules={} changed_bytes={} safe_core_bytes={} discrete_flags_bytes={} unknown_bytes={}",
        report.baseline.mismatch_vs_target,
        report.with_v2_spec.mismatch_vs_target,
        report.residue_rule_count,
        report.offset_rule_count,
        report.v2_changed_byte_count,
        report.safe_core_byte_count,
        report.discrete_flags_byte_count,
        report.unknown_byte_count,
    );

    Ok(())
}
