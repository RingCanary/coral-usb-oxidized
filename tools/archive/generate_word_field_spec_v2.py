#!/usr/bin/env python3
"""Generate v2 word-field spec from holdout deltas.

This tool inspects per-offset endpoint prediction errors and proposes
field-spec overrides that are:
- offset-aware,
- residue-aware (when safely compressible),
- bit-range-aware (for packed lane32 words).

It emits JSON compatible with `tools/emit_word_field_patch_spec.py`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import emit_word_field_patch_spec as emitter


@dataclass(frozen=True)
class WordCtx:
    offset: int
    lane: str
    lane_bytes: int
    residue: Optional[int]
    low_val: int
    high_val: int
    base_word: int
    target_word: int
    best_formula: Dict[str, Any]
    group_bitfield_fits: List[Dict[str, Any]]


@dataclass(frozen=True)
class RuleChoice:
    lane: str
    residue: Optional[int]
    offset: Optional[int]
    policy: Optional[str]
    model: Optional[str]
    div: Optional[int]
    domain: Optional[str]
    bits: Optional[int]
    bit_lo: Optional[int]
    bit_hi: Optional[int]
    source: str


@dataclass(frozen=True)
class CandidateEval:
    rule: RuleChoice
    pred_word: int
    mismatch_bytes: int


def _iter_group_rows(lane_obj: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for g in lane_obj.get("top_groups", []):
        per_offset = g.get("per_offset_fits", [])
        if per_offset:
            for per in per_offset:
                row = dict(per)
                row["stride_residue"] = g.get("stride_residue")
                yield row, g
            continue

        offsets = g.get("offsets", [])
        grouped_values = g.get("values_by_dim", [])
        if not offsets or not grouped_values:
            continue

        dim_to_values: Dict[int, Sequence[int]] = {}
        for row in grouped_values:
            dim = row.get("dim")
            vals = row.get("values")
            if dim is None or not isinstance(vals, list):
                continue
            dim_to_values[int(dim)] = [int(v) for v in vals]

        for idx, off in enumerate(offsets):
            vals_by_dim: List[Dict[str, int]] = []
            for dim, vals in dim_to_values.items():
                if idx < len(vals):
                    vals_by_dim.append({"dim": dim, "value": int(vals[idx])})
            if not vals_by_dim:
                continue
            yield (
                {
                    "offset": int(off),
                    "values_by_dim": vals_by_dim,
                    "best_formula": g.get("best_formula", {}),
                    "stride_residue": g.get("stride_residue"),
                },
                g,
            )


def _complexity_rank(model: Optional[str]) -> int:
    if model == "const":
        return 0
    if model == "tile-linear":
        return 1
    if model == "tile2div-linear":
        return 2
    if model == "tile-quadratic":
        return 3
    return 9


def _word_mismatch_bytes(pred: int, target: int, lane_bytes: int) -> int:
    pb = pred.to_bytes(lane_bytes, "little", signed=False)
    tb = target.to_bytes(lane_bytes, "little", signed=False)
    return sum(1 for i in range(lane_bytes) if pb[i] != tb[i])


def _collect_endpoint_models(formula: Dict[str, Any]) -> List[Tuple[str, Optional[int]]]:
    cands = [formula] + list(formula.get("top_candidates", []))
    out: List[Tuple[str, Optional[int]]] = []
    seen: set[Tuple[str, Optional[int]]] = set()

    def push(model: str, div: Optional[int]) -> None:
        key = (model, div)
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    for c in cands:
        m = str(c.get("model", ""))
        p = dict(c.get("params", {}))
        if m == "tile2div-linear":
            try:
                d = int(p.get("div", 1))
            except Exception:
                d = 1
            push("tile2div-linear", d)
        elif m == "tile-linear":
            push("tile-linear", None)
        elif m == "const":
            push("const", None)
        elif m == "tile-quadratic":
            # endpoint mode uses linear fallback for quadratic.
            push("tile-linear", None)

    # ensure at least one candidate from endpoint picker.
    pm, pp = emitter._pick_formula(formula, mode="endpoint")
    if pm == "tile2div-linear":
        push("tile2div-linear", int(pp.get("div", 1)))
    elif pm in ("tile-linear", "const", "tile-quadratic"):
        push("tile-linear" if pm == "tile-quadratic" else pm, None)

    return out


def _predict_word(
    ctx: WordCtx,
    global_mode: str,
    low_dim: int,
    high_dim: int,
    target_dim: int,
    tile_size: int,
    rule: Optional[RuleChoice],
) -> Tuple[int, str, Dict[str, Any], str, int]:
    mode_use = global_mode
    if rule is not None and rule.policy is not None:
        mode_use = rule.policy

    if mode_use == "best":
        model, params = emitter._pick_best_candidate(
            best=ctx.best_formula,
            model_override=(rule.model if rule is not None else None),
            div_override=(rule.div if rule is not None else None),
        )
    else:
        if rule is not None and rule.model is not None:
            model = rule.model
            params: Dict[str, Any] = {}
            if rule.div is not None:
                params["div"] = int(rule.div)
        else:
            model, params = emitter._pick_formula(ctx.best_formula, mode=mode_use)

    lane_bits = ctx.lane_bytes * 8
    bit_lo = rule.bit_lo if rule is not None else None
    bit_hi = rule.bit_hi if rule is not None else None
    has_bit_range = (bit_lo is not None) and (bit_hi is not None)

    if has_bit_range:
        width = int(bit_hi - bit_lo + 1)
        field_mask = (1 << width) - 1
        low_eval = (ctx.low_val >> bit_lo) & field_mask
        high_eval = (ctx.high_val >> bit_lo) & field_mask
        bits = width
    else:
        width = lane_bits
        field_mask = (1 << width) - 1
        low_eval = ctx.low_val
        high_eval = ctx.high_val
        bits = lane_bits

    if rule is not None and rule.bits is not None:
        bits = int(rule.bits)

    domain = "u"
    if rule is not None and rule.domain is not None:
        domain = rule.domain

    pred_field = emitter._predict_value(
        model=model,
        params=params,
        low_dim=low_dim,
        low_val=low_eval,
        high_dim=high_dim,
        high_val=high_eval,
        target_dim=target_dim,
        tile_size=tile_size,
        mode=mode_use,
        domain=domain,
        bits=bits,
    )
    pred_field &= field_mask

    if has_bit_range:
        pred = (ctx.base_word & ~(field_mask << bit_lo)) | ((pred_field & field_mask) << bit_lo)
    else:
        pred = pred_field

    pred &= (1 << lane_bits) - 1
    return pred, model, params, domain, bits


def _lane_bit_options(lane_bits: int) -> List[int]:
    if lane_bits <= 16:
        vals = [8, 12, 16]
    else:
        vals = [8, 12, 16, 20, 24, 28, 32]
    out = sorted({x for x in vals if x <= lane_bits})
    if lane_bits not in out:
        out.append(lane_bits)
    return out


def _collect_bit_ranges(ctx: WordCtx, base_pred: int) -> List[Tuple[int, int, str, Dict[str, Any]]]:
    out: List[Tuple[int, int, str, Dict[str, Any]]] = []

    # ranges mined by word-field analyzer
    for bf in ctx.group_bitfield_fits:
        br = bf.get("bit_range")
        if not isinstance(br, list) or len(br) != 2:
            continue
        lo = int(br[0])
        hi = int(br[1])
        out.append((lo, hi, "group_bitfield", dict(bf.get("best_formula", {}))))

    # fallback from byte-level mismatches in baseline prediction
    pb = base_pred.to_bytes(ctx.lane_bytes, "little", signed=False)
    tb = ctx.target_word.to_bytes(ctx.lane_bytes, "little", signed=False)
    mism_bytes = [i for i in range(ctx.lane_bytes) if pb[i] != tb[i]]
    if mism_bytes:
        for b in mism_bytes:
            lo = b * 8
            hi = lo + 7
            out.append((lo, hi, "mismatch_byte", ctx.best_formula))
        lo = min(mism_bytes) * 8
        hi = (max(mism_bytes) * 8) + 7
        out.append((lo, hi, "mismatch_span", ctx.best_formula))

    # de-dup by (lo,hi,source,best_formula model/div)
    uniq: Dict[Tuple[int, int, str, str, Optional[int]], Tuple[int, int, str, Dict[str, Any]]] = {}
    for lo, hi, src, f in out:
        m = str(f.get("model", ""))
        d = None
        try:
            if m == "tile2div-linear":
                d = int(f.get("params", {}).get("div", 1))
        except Exception:
            d = None
        key = (lo, hi, src, m, d)
        if key not in uniq:
            uniq[key] = (lo, hi, src, f)

    return list(uniq.values())


def _rank_candidate(c: CandidateEval, lane_bits: int) -> Tuple[int, int, int, int, int]:
    bit_width = lane_bits
    if c.rule.bit_lo is not None and c.rule.bit_hi is not None:
        bit_width = c.rule.bit_hi - c.rule.bit_lo + 1
    return (
        c.mismatch_bytes,
        0 if c.rule.bit_lo is not None else 1,
        _complexity_rank(c.rule.model),
        bit_width,
        c.rule.bits if c.rule.bits is not None else lane_bits,
    )


def _build_contexts(
    analysis: Dict[str, Any],
    base_blob: bytes,
    target_blob: bytes,
    low_dim: int,
    high_dim: int,
    target_dim: int,
    lane_priority: str,
) -> List[WordCtx]:
    lanes = emitter._lane_plan(lane_priority)
    assigned_offsets: set[int] = set()
    out: List[WordCtx] = []

    for lane in lanes:
        lane_obj = analysis.get(lane.lane_key)
        if not isinstance(lane_obj, dict):
            continue
        for per, group in _iter_group_rows(lane_obj):
            off = int(per["offset"])
            if off in assigned_offsets:
                continue
            if off < 0 or off + lane.lane_bytes > len(base_blob):
                continue

            vals = emitter._values_map(per)
            if low_dim not in vals or high_dim not in vals:
                continue

            target_word = int.from_bytes(target_blob[off : off + lane.lane_bytes], "little", signed=False)
            base_word = int.from_bytes(base_blob[off : off + lane.lane_bytes], "little", signed=False)

            out.append(
                WordCtx(
                    offset=off,
                    lane=lane.lane_key,
                    lane_bytes=lane.lane_bytes,
                    residue=per.get("stride_residue"),
                    low_val=int(vals[low_dim]),
                    high_val=int(vals[high_dim]),
                    base_word=base_word,
                    target_word=target_word,
                    best_formula=dict(per.get("best_formula", {})),
                    group_bitfield_fits=[dict(x) for x in group.get("bitfield_fits", [])],
                )
            )
            assigned_offsets.add(off)

    return out


def _evaluate_rule_on_ctx(
    ctx: WordCtx,
    global_mode: str,
    low_dim: int,
    high_dim: int,
    target_dim: int,
    tile_size: int,
    rule: RuleChoice,
) -> CandidateEval:
    pred, _model, _params, _domain, _bits = _predict_word(
        ctx=ctx,
        global_mode=global_mode,
        low_dim=low_dim,
        high_dim=high_dim,
        target_dim=target_dim,
        tile_size=tile_size,
        rule=rule,
    )
    mism = _word_mismatch_bytes(pred, ctx.target_word, ctx.lane_bytes)
    return CandidateEval(rule=rule, pred_word=pred, mismatch_bytes=mism)


def _propose_offset_rule(
    ctx: WordCtx,
    global_mode: str,
    low_dim: int,
    high_dim: int,
    target_dim: int,
    tile_size: int,
) -> Tuple[int, Optional[CandidateEval]]:
    baseline_pred, _m, _p, _d, _b = _predict_word(
        ctx=ctx,
        global_mode=global_mode,
        low_dim=low_dim,
        high_dim=high_dim,
        target_dim=target_dim,
        tile_size=tile_size,
        rule=None,
    )
    baseline_mism = _word_mismatch_bytes(baseline_pred, ctx.target_word, ctx.lane_bytes)
    if baseline_mism == 0:
        return baseline_mism, None

    lane_bits = ctx.lane_bytes * 8
    evals: Dict[Tuple[Any, ...], CandidateEval] = {}

    # full-word candidates
    for model, div in _collect_endpoint_models(ctx.best_formula):
        for domain in ("u", "s", "mod"):
            for bits in _lane_bit_options(lane_bits):
                rule = RuleChoice(
                    lane=ctx.lane,
                    residue=ctx.residue,
                    offset=ctx.offset,
                    policy="endpoint",
                    model=model,
                    div=div,
                    domain=domain,
                    bits=bits,
                    bit_lo=None,
                    bit_hi=None,
                    source="full_word",
                )
                ev = _evaluate_rule_on_ctx(
                    ctx=ctx,
                    global_mode=global_mode,
                    low_dim=low_dim,
                    high_dim=high_dim,
                    target_dim=target_dim,
                    tile_size=tile_size,
                    rule=rule,
                )
                if ev.mismatch_bytes >= baseline_mism:
                    continue
                key = (
                    rule.policy,
                    rule.model,
                    rule.div,
                    rule.domain,
                    rule.bits,
                    rule.bit_lo,
                    rule.bit_hi,
                )
                old = evals.get(key)
                if old is None or ev.mismatch_bytes < old.mismatch_bytes:
                    evals[key] = ev

    # bit-range candidates
    for lo, hi, src, formula in _collect_bit_ranges(ctx, baseline_pred):
        width = hi - lo + 1
        for model, div in _collect_endpoint_models(formula):
            for domain in ("u", "s", "mod"):
                # width default + smaller byte-friendly fallback widths
                bit_opts = {width}
                for b in _lane_bit_options(width):
                    if b <= width:
                        bit_opts.add(b)
                for bits in sorted(bit_opts):
                    rule = RuleChoice(
                        lane=ctx.lane,
                        residue=ctx.residue,
                        offset=ctx.offset,
                        policy="endpoint",
                        model=model,
                        div=div,
                        domain=domain,
                        bits=bits,
                        bit_lo=lo,
                        bit_hi=hi,
                        source=src,
                    )
                    ev = _evaluate_rule_on_ctx(
                        ctx=ctx,
                        global_mode=global_mode,
                        low_dim=low_dim,
                        high_dim=high_dim,
                        target_dim=target_dim,
                        tile_size=tile_size,
                        rule=rule,
                    )
                    if ev.mismatch_bytes >= baseline_mism:
                        continue
                    key = (
                        rule.policy,
                        rule.model,
                        rule.div,
                        rule.domain,
                        rule.bits,
                        rule.bit_lo,
                        rule.bit_hi,
                    )
                    old = evals.get(key)
                    if old is None or ev.mismatch_bytes < old.mismatch_bytes:
                        evals[key] = ev

    if not evals:
        return baseline_mism, None

    best = sorted(evals.values(), key=lambda c: _rank_candidate(c, lane_bits))[0]
    return baseline_mism, best


def _rule_signature(rule: RuleChoice) -> Tuple[Any, ...]:
    return (
        rule.lane,
        rule.policy,
        rule.model,
        rule.div,
        rule.domain,
        rule.bits,
        rule.bit_lo,
        rule.bit_hi,
    )


def _rule_to_obj(rule: RuleChoice, include_offset: bool, include_residue: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"lane": rule.lane}
    if include_offset:
        out["offset"] = int(rule.offset) if rule.offset is not None else None
    if include_residue:
        out["residue"] = int(rule.residue) if rule.residue is not None else None
    if rule.policy is not None:
        out["policy"] = rule.policy
    if rule.model is not None:
        out["model"] = rule.model
    if rule.div is not None:
        out["div"] = int(rule.div)
    if rule.domain is not None:
        out["domain"] = rule.domain
    if rule.bits is not None:
        out["bits"] = int(rule.bits)
    if rule.bit_lo is not None and rule.bit_hi is not None:
        out["bit_range"] = [int(rule.bit_lo), int(rule.bit_hi)]
    return out


def _simulate_patch(
    contexts: Sequence[WordCtx],
    base_blob: bytes,
    target_blob: bytes,
    global_mode: str,
    low_dim: int,
    high_dim: int,
    target_dim: int,
    tile_size: int,
    offset_rules: Dict[int, RuleChoice],
    residue_rules: Dict[Tuple[str, int], RuleChoice],
) -> Dict[str, Any]:
    patched = bytearray(base_blob)

    for ctx in contexts:
        rule = offset_rules.get(ctx.offset)
        if rule is None and ctx.residue is not None:
            rule = residue_rules.get((ctx.lane, int(ctx.residue)))

        pred, _m, _p, _d, _b = _predict_word(
            ctx=ctx,
            global_mode=global_mode,
            low_dim=low_dim,
            high_dim=high_dim,
            target_dim=target_dim,
            tile_size=tile_size,
            rule=rule,
        )
        bts = pred.to_bytes(ctx.lane_bytes, "little", signed=False)
        patched[ctx.offset : ctx.offset + ctx.lane_bytes] = bts

    mism = [i for i in range(len(target_blob)) if patched[i] != target_blob[i]]
    return {
        "mismatch_vs_target": len(mism),
        "mismatch_ratio_vs_target": len(mism) / float(len(target_blob)) if target_blob else 0.0,
        "mismatch_preview": mism[:200],
        "patched_sha256": emitter.pe._sha256_bytes(bytes(patched)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-json", required=True)
    ap.add_argument("--base-exec", required=True)
    ap.add_argument("--target-exec", required=True)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--low-dim", type=int, required=True)
    ap.add_argument("--high-dim", type=int, required=True)
    ap.add_argument("--target-dim", type=int, required=True)
    ap.add_argument("--tile-size", type=int, default=64)
    ap.add_argument("--predict-mode", choices=["endpoint", "best"], default="endpoint")
    ap.add_argument("--lane-priority", default="lane32,lane16")
    ap.add_argument("--out-spec", required=True)
    ap.add_argument("--out-report", default=None)
    args = ap.parse_args()

    analysis = json.loads(Path(args.analysis_json).read_text())
    base_blob = emitter._read_chunk_bytes(Path(args.base_exec), args.chunk_index)
    target_blob = emitter._read_chunk_bytes(Path(args.target_exec), args.chunk_index)
    if len(base_blob) != len(target_blob):
        raise SystemExit(
            f"base/target chunk length mismatch: base={len(base_blob)} target={len(target_blob)}"
        )

    contexts = _build_contexts(
        analysis=analysis,
        base_blob=base_blob,
        target_blob=target_blob,
        low_dim=args.low_dim,
        high_dim=args.high_dim,
        target_dim=args.target_dim,
        lane_priority=args.lane_priority,
    )

    if not contexts:
        raise SystemExit("no assignable offsets found")

    baseline = _simulate_patch(
        contexts=contexts,
        base_blob=base_blob,
        target_blob=target_blob,
        global_mode=args.predict_mode,
        low_dim=args.low_dim,
        high_dim=args.high_dim,
        target_dim=args.target_dim,
        tile_size=args.tile_size,
        offset_rules={},
        residue_rules={},
    )

    chosen_offset_rules: Dict[int, RuleChoice] = {}
    per_offset_notes: List[Dict[str, Any]] = []

    for ctx in contexts:
        baseline_word_mism, best = _propose_offset_rule(
            ctx=ctx,
            global_mode=args.predict_mode,
            low_dim=args.low_dim,
            high_dim=args.high_dim,
            target_dim=args.target_dim,
            tile_size=args.tile_size,
        )
        note: Dict[str, Any] = {
            "offset": ctx.offset,
            "lane": ctx.lane,
            "residue": ctx.residue,
            "baseline_word_mismatch_bytes": baseline_word_mism,
        }
        if best is not None:
            chosen_offset_rules[ctx.offset] = best.rule
            note.update(
                {
                    "selected": True,
                    "selected_rule": _rule_to_obj(best.rule, include_offset=True, include_residue=True),
                    "selected_mismatch_bytes": best.mismatch_bytes,
                }
            )
        else:
            note["selected"] = False
        per_offset_notes.append(note)

    # Promote to residue rules only when every assigned offset in residue has the same chosen rule signature.
    residue_to_offsets: Dict[Tuple[str, int], List[int]] = {}
    for ctx in contexts:
        if ctx.residue is None:
            continue
        residue_to_offsets.setdefault((ctx.lane, int(ctx.residue)), []).append(ctx.offset)

    residue_rules: Dict[Tuple[str, int], RuleChoice] = {}
    for rk, offs in residue_to_offsets.items():
        chosen = [chosen_offset_rules.get(off) for off in offs]
        if any(r is None for r in chosen):
            continue
        sigs = {_rule_signature(r) for r in chosen if r is not None}
        if len(sigs) != 1:
            continue
        sample = chosen[0]
        if sample is None:
            continue
        residue_rule = RuleChoice(
            lane=sample.lane,
            residue=rk[1],
            offset=None,
            policy=sample.policy,
            model=sample.model,
            div=sample.div,
            domain=sample.domain,
            bits=sample.bits,
            bit_lo=sample.bit_lo,
            bit_hi=sample.bit_hi,
            source=sample.source,
        )
        residue_rules[rk] = residue_rule
        for off in offs:
            chosen_offset_rules.pop(off, None)

    improved = _simulate_patch(
        contexts=contexts,
        base_blob=base_blob,
        target_blob=target_blob,
        global_mode=args.predict_mode,
        low_dim=args.low_dim,
        high_dim=args.high_dim,
        target_dim=args.target_dim,
        tile_size=args.tile_size,
        offset_rules=chosen_offset_rules,
        residue_rules=residue_rules,
    )

    residue_objs = [
        _rule_to_obj(r, include_offset=False, include_residue=True)
        for _, r in sorted(residue_rules.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]
    offset_objs = [
        _rule_to_obj(r, include_offset=True, include_residue=False)
        for _, r in sorted(chosen_offset_rules.items(), key=lambda kv: kv[0])
    ]

    spec = {
        "offset_rules": offset_objs,
        "residue_rules": residue_objs,
    }

    out_spec = Path(args.out_spec)
    out_spec.parent.mkdir(parents=True, exist_ok=True)
    out_spec.write_text(json.dumps(spec, indent=2, sort_keys=False) + "\n")

    report: Dict[str, Any] = {
        "analysis_json": args.analysis_json,
        "base_exec": args.base_exec,
        "target_exec": args.target_exec,
        "chunk_index": args.chunk_index,
        "dims": {
            "low": args.low_dim,
            "high": args.high_dim,
            "target": args.target_dim,
            "tile_size": args.tile_size,
        },
        "predict_mode": args.predict_mode,
        "lane_priority": args.lane_priority,
        "assigned_word_offsets": len(contexts),
        "baseline": baseline,
        "with_v2_spec": improved,
        "residue_rule_count": len(residue_objs),
        "offset_rule_count": len(offset_objs),
        "per_offset_notes": per_offset_notes,
        "out_spec": str(out_spec),
    }

    if args.out_report:
        out_report = Path(args.out_report)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")

    print(f"Wrote spec: {out_spec}")
    print(
        "baseline_mismatch={} v2_mismatch={} residue_rules={} offset_rules={}".format(
            baseline["mismatch_vs_target"],
            improved["mismatch_vs_target"],
            len(residue_objs),
            len(offset_objs),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
