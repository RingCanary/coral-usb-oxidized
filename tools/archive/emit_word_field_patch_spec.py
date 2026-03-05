#!/usr/bin/env python3
"""Emit replay byte patchspec from word-field analysis JSON.

Uses per-offset u16/u32 formulas to predict target-dimension lane values and
writes `<payload_len> <offset> <value>` rules consumable by
`rusb_serialized_exec_replay --instruction-patch-spec`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import parse_edgetpu_executable as pe


@dataclass(frozen=True)
class LanePick:
    lane_key: str
    lane_bytes: int


@dataclass(frozen=True)
class OverrideRule:
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


def _read_chunk_bytes(path: Path, chunk_index: int) -> bytes:
    blob = path.read_bytes()
    root = pe._parse_root_table(blob, 0, file_identifier=None)
    tables = pe._read_vector_table_field(root, 5)
    if chunk_index < 0 or chunk_index >= len(tables):
        raise SystemExit(
            f"chunk index {chunk_index} out of range for {path} (count={len(tables)})"
        )
    return pe._read_vector_bytes_field(tables[chunk_index], 0)


def _tiles(dim: int, tile_size: int) -> int:
    return dim // tile_size


def _fit_linear(x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float]]:
    if x1 == x2:
        return None
    a = (y2 - y1) / (x2 - x1)
    b = y1 - (a * x1)
    return (a, b)


def _predict_linear(a: float, b: float, x: float) -> int:
    return int(round((a * x) + b))


def _load_field_spec(path: Optional[str]) -> Dict[str, List[OverrideRule]]:
    if not path:
        return {"offset_rules": [], "residue_rules": []}

    raw = json.loads(Path(path).read_text())
    offset_rules: List[OverrideRule] = []
    residue_rules: List[OverrideRule] = []

    def parse_rule(obj: Dict[str, Any]) -> OverrideRule:
        lane = str(obj.get("lane", ""))
        if lane not in ("lane16", "lane32"):
            raise SystemExit(f"field-spec rule lane must be lane16 or lane32, got: {lane}")

        residue = obj.get("residue")
        offset = obj.get("offset")
        policy = obj.get("policy")
        model = obj.get("model")
        div = obj.get("div")
        domain = obj.get("domain")
        bits = obj.get("bits")
        bit_lo = obj.get("bit_lo")
        bit_hi = obj.get("bit_hi")
        bit_range = obj.get("bit_range")

        if bit_range is not None:
            if bit_lo is not None or bit_hi is not None:
                raise SystemExit(
                    "field-spec rule must use either bit_range or bit_lo/bit_hi (not both)"
                )
            if (not isinstance(bit_range, list)) or len(bit_range) != 2:
                raise SystemExit(f"field-spec bit_range must be [lo,hi], got: {bit_range}")
            bit_lo = bit_range[0]
            bit_hi = bit_range[1]

        if policy is not None and policy not in ("endpoint", "best"):
            raise SystemExit(f"field-spec policy must be endpoint|best, got: {policy}")
        if domain is not None and domain not in ("u", "s", "mod"):
            raise SystemExit(f"field-spec domain must be u|s|mod, got: {domain}")
        if model is not None and model not in ("const", "tile-linear", "tile-quadratic", "tile2div-linear"):
            raise SystemExit(f"field-spec model unsupported: {model}")

        lane_bits = 16 if lane == "lane16" else 32
        if (bit_lo is None) != (bit_hi is None):
            raise SystemExit("field-spec requires both bit_lo and bit_hi when selecting bit ranges")
        if bit_lo is not None and bit_hi is not None:
            bit_lo = int(bit_lo)
            bit_hi = int(bit_hi)
            if bit_lo < 0 or bit_hi < bit_lo or bit_hi >= lane_bits:
                raise SystemExit(
                    f"field-spec bit range out of bounds for {lane}: [{bit_lo},{bit_hi}]"
                )

        if bits is not None:
            bits = int(bits)
            if bits <= 0 or bits > lane_bits:
                raise SystemExit(f"field-spec bits out of bounds for {lane}: {bits}")

        return OverrideRule(
            lane=lane,
            residue=int(residue) if residue is not None else None,
            offset=int(offset) if offset is not None else None,
            policy=str(policy) if policy is not None else None,
            model=str(model) if model is not None else None,
            div=int(div) if div is not None else None,
            domain=str(domain) if domain is not None else None,
            bits=bits,
            bit_lo=bit_lo,
            bit_hi=bit_hi,
        )

    for obj in raw.get("offset_rules", []):
        offset_rules.append(parse_rule(obj))
    for obj in raw.get("residue_rules", []):
        residue_rules.append(parse_rule(obj))

    return {"offset_rules": offset_rules, "residue_rules": residue_rules}


def _match_override(
    spec: Dict[str, List[OverrideRule]],
    lane: str,
    offset: int,
    residue: Optional[int],
) -> Optional[OverrideRule]:
    for rule in spec["offset_rules"]:
        if rule.lane == lane and rule.offset == offset:
            return rule
    if residue is not None:
        for rule in spec["residue_rules"]:
            if rule.lane == lane and rule.residue == residue:
                return rule
    return None


def _pick_best_candidate(
    best: Dict[str, Any],
    model_override: Optional[str],
    div_override: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    cands = [best] + list(best.get("top_candidates", []))
    filtered = []
    for c in cands:
        model = str(c.get("model", "none"))
        params = dict(c.get("params", {}))
        if model_override is not None and model != model_override:
            continue
        if div_override is not None:
            if model != "tile2div-linear":
                continue
            if int(params.get("div", -1)) != int(div_override):
                continue
        filtered.append(c)

    if not filtered:
        return (str(best.get("model", "none")), dict(best.get("params", {})))

    def rank(c: Dict[str, Any]) -> Tuple[float, float, int, str]:
        exact = float(c.get("exact_ratio", 0.0))
        mae = float(c.get("mae", 1e9))
        model = str(c.get("model", "none"))
        complexity = 0 if model == "const" else (1 if model == "tile-linear" else 2)
        return (-exact, mae, complexity, model)

    chosen = sorted(filtered, key=rank)[0]
    return (str(chosen.get("model", "none")), dict(chosen.get("params", {})))


def _pick_formula(best: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    if mode == "best":
        return (str(best.get("model", "none")), dict(best.get("params", {})))

    # mode == endpoint: prefer more interpretable exact candidates when present
    top = best.get("top_candidates", [])
    exact = [
        c
        for c in top
        if float(c.get("exact_ratio", 0.0)) == 1.0 and float(c.get("mae", 1.0)) == 0.0
    ]
    if exact:
        # prioritize tile2div-linear with integer-ish params; then tile-linear; then others
        def key(c: Dict[str, Any]) -> Tuple[int, float, float, int, str]:
            m = str(c.get("model"))
            p = dict(c.get("params", {}))
            a = float(p.get("a", 0.0))
            b = float(p.get("b", 0.0))
            d = int(p.get("div", 1))
            int_err = abs(a - round(a)) + abs(b - round(b))
            m_rank = 0 if m == "tile2div-linear" else (1 if m == "tile-linear" else 2)
            return (m_rank, int_err, abs(a), d, m)

        exact = sorted(exact, key=key)
        c = exact[0]
        return (str(c.get("model", "none")), dict(c.get("params", {})))

    return (str(best.get("model", "none")), dict(best.get("params", {})))


def _to_signed(v: int, bits: int) -> int:
    mask = (1 << bits) - 1
    v &= mask
    sign = 1 << (bits - 1)
    return v - (1 << bits) if (v & sign) else v


def _from_signed(v: int, bits: int) -> int:
    return v & ((1 << bits) - 1)


def _interp_domain(
    low: int,
    high: int,
    xl: float,
    xh: float,
    xt: float,
    bits: int,
    domain: str,
) -> int:
    if xh == xl:
        return low
    frac = (xt - xl) / (xh - xl)

    if domain == "u":
        return int(round(low + (high - low) * frac))
    if domain == "s":
        l = _to_signed(low, bits)
        h = _to_signed(high, bits)
        p = int(round(l + (h - l) * frac))
        return _from_signed(p, bits)
    if domain == "mod":
        ring = 1 << bits
        half = ring // 2
        delta = ((high - low + half) % ring) - half
        p = int(round(low + (delta * frac)))
        return p % ring

    return int(round(low + (high - low) * frac))


def _predict_value(
    model: str,
    params: Dict[str, Any],
    low_dim: int,
    low_val: int,
    high_dim: int,
    high_val: int,
    target_dim: int,
    tile_size: int,
    mode: str,
    domain: str,
    bits: int,
) -> int:
    # all-points: evaluate paramized model directly
    if mode == "best":
        t = float(_tiles(target_dim, tile_size))
        if model == "const":
            return int(round(float(params.get("c", low_val))))
        if model == "tile-linear":
            return int(round(float(params.get("a", 0.0)) * t + float(params.get("b", low_val))))
        if model == "tile-quadratic":
            a = float(params.get("a", 0.0))
            b = float(params.get("b", 0.0))
            c = float(params.get("c", low_val))
            return int(round(a * t * t + b * t + c))
        if model == "tile2div-linear":
            d = int(params.get("div", 1))
            x = float((_tiles(target_dim, tile_size) ** 2) // d)
            return int(round(float(params.get("a", 0.0)) * x + float(params.get("b", low_val))))

    # endpoint-fit mode: derive only from low/high endpoint values
    tl = _tiles(low_dim, tile_size)
    th = _tiles(high_dim, tile_size)
    tt = _tiles(target_dim, tile_size)

    if model == "const":
        return int(low_val)
    if model == "tile-linear":
        return _interp_domain(
            low=low_val,
            high=high_val,
            xl=float(tl),
            xh=float(th),
            xt=float(tt),
            bits=bits,
            domain=domain,
        )
    if model == "tile2div-linear":
        d = int(params.get("div", 1))
        return _interp_domain(
            low=low_val,
            high=high_val,
            xl=float((tl * tl) // d),
            xh=float((th * th) // d),
            xt=float((tt * tt) // d),
            bits=bits,
            domain=domain,
        )
    if model == "tile-quadratic":
        # underconstrained with 2 points; fallback to tile-linear in tiles domain
        return _interp_domain(
            low=low_val,
            high=high_val,
            xl=float(tl),
            xh=float(th),
            xt=float(tt),
            bits=bits,
            domain=domain,
        )

    return int(low_val)


def _values_map(per: Dict[str, Any]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for row in per.get("values_by_dim", []):
        out[int(row["dim"])] = int(row["value"])
    return out


def _iter_per_offsets(lane_obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for g in lane_obj.get("top_groups", []):
        per_offset = g.get("per_offset_fits", [])
        if per_offset:
            for per in per_offset:
                row = dict(per)
                row["stride_residue"] = g.get("stride_residue")
                yield row
            continue

        # Backward-compatible path for grouped analysis JSON that stores:
        #   offsets=[...], values_by_dim=[{"dim": d, "values": [...]}]
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
            yield {
                "offset": int(off),
                "values_by_dim": vals_by_dim,
                "best_formula": g.get("best_formula", {}),
                "stride_residue": g.get("stride_residue"),
            }


def _lane_plan(priority: str) -> List[LanePick]:
    items = [x.strip() for x in priority.split(",") if x.strip()]
    out: List[LanePick] = []
    for it in items:
        if it == "lane32":
            out.append(LanePick(lane_key="lane32", lane_bytes=4))
        elif it == "lane16":
            out.append(LanePick(lane_key="lane16", lane_bytes=2))
        else:
            raise SystemExit(f"invalid lane priority token: {it} (use lane32,lane16)")
    if not out:
        raise SystemExit("empty --lane-priority")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-json", required=True)
    ap.add_argument("--base-exec", required=True, help="serialized_executable_XXX.bin used as patch base")
    ap.add_argument("--target-exec", default=None, help="optional ground-truth executable for mismatch reporting")
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--low-dim", type=int, required=True)
    ap.add_argument("--high-dim", type=int, required=True)
    ap.add_argument("--target-dim", type=int, required=True)
    ap.add_argument("--tile-size", type=int, default=64)
    ap.add_argument("--predict-mode", choices=["best", "endpoint"], default="endpoint")
    ap.add_argument("--lane-priority", default="lane32,lane16")
    ap.add_argument("--field-spec", default=None, help="optional JSON override for residue/offset rules")
    ap.add_argument("--out-spec", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    analysis = json.loads(Path(args.analysis_json).read_text())
    base = bytearray(_read_chunk_bytes(Path(args.base_exec), args.chunk_index))
    target = (
        _read_chunk_bytes(Path(args.target_exec), args.chunk_index)
        if args.target_exec is not None
        else None
    )

    lanes = _lane_plan(args.lane_priority)
    field_spec = _load_field_spec(args.field_spec)
    assigned: Dict[int, Dict[str, Any]] = {}

    for lane in lanes:
        lane_obj = analysis.get(lane.lane_key)
        if not isinstance(lane_obj, dict):
            continue
        for per in _iter_per_offsets(lane_obj):
            off = int(per["offset"])
            if off in assigned:
                continue
            if off < 0 or off + lane.lane_bytes > len(base):
                continue

            vals = _values_map(per)
            if args.low_dim not in vals or args.high_dim not in vals:
                continue
            # for mode=best we can still evaluate even if target dim absent in values
            low_val = int(vals[args.low_dim])
            high_val = int(vals[args.high_dim])

            best = dict(per.get("best_formula", {}))
            override = _match_override(
                spec=field_spec,
                lane=lane.lane_key,
                offset=off,
                residue=per.get("stride_residue"),
            )

            mode_use = args.predict_mode
            model: str
            p: Dict[str, Any]
            if override is not None and override.policy is not None:
                mode_use = override.policy

            if mode_use == "best":
                model, p = _pick_best_candidate(
                    best=best,
                    model_override=override.model if override else None,
                    div_override=override.div if override else None,
                )
            else:
                if override is not None and override.model is not None:
                    model = override.model
                    p = {}
                    if override.div is not None:
                        p["div"] = int(override.div)
                else:
                    model, p = _pick_formula(best, mode=mode_use)

            lane_bits = (lane.lane_bytes * 8)
            bit_lo = override.bit_lo if override is not None else None
            bit_hi = override.bit_hi if override is not None else None
            has_bit_range = (bit_lo is not None) and (bit_hi is not None)

            if has_bit_range:
                field_width = int(bit_hi - bit_lo + 1)
                field_mask = (1 << field_width) - 1
                low_eval = (low_val >> bit_lo) & field_mask
                high_eval = (high_val >> bit_lo) & field_mask
                bits = field_width
            else:
                field_width = lane_bits
                field_mask = (1 << field_width) - 1
                low_eval = low_val
                high_eval = high_val
                bits = lane_bits

            if override is not None and override.bits is not None:
                bits = int(override.bits)

            domain = "u"
            if override is not None and override.domain is not None:
                domain = override.domain

            pred_field = _predict_value(
                model=model,
                params=p,
                low_dim=args.low_dim,
                low_val=low_eval,
                high_dim=args.high_dim,
                high_val=high_eval,
                target_dim=args.target_dim,
                tile_size=args.tile_size,
                mode=mode_use,
                domain=domain,
                bits=bits,
            )
            pred_field &= field_mask

            if has_bit_range:
                base_word = int.from_bytes(
                    base[off : off + lane.lane_bytes],
                    "little",
                    signed=False,
                )
                pred_val = (base_word & ~(field_mask << bit_lo)) | ((pred_field & field_mask) << bit_lo)
            else:
                pred_val = pred_field

            max_val = (1 << lane_bits) - 1
            pred_val &= max_val
            pred_bytes = pred_val.to_bytes(lane.lane_bytes, "little", signed=False)

            assigned[off] = {
                "offset": off,
                "lane": lane.lane_key,
                "lane_bytes": lane.lane_bytes,
                "model": model,
                "params": p,
                "policy": mode_use,
                "domain": domain,
                "bits": bits,
                "bit_lo": bit_lo,
                "bit_hi": bit_hi,
                "pred_field_value": int(pred_field),
                "pred_value": pred_val,
                "pred_bytes": list(pred_bytes),
                "stride_residue": per.get("stride_residue"),
                "override_applied": override is not None,
                "known_values": vals,
            }

    patched = bytearray(base)
    for off, row in assigned.items():
        bts = bytes(row["pred_bytes"])
        patched[off : off + len(bts)] = bts

    changed = [i for i in range(len(base)) if patched[i] != base[i]]

    spec_lines = [
        "# emitted by emit_word_field_patch_spec.py",
        f"# mode={args.predict_mode} target_dim={args.target_dim} low_dim={args.low_dim} high_dim={args.high_dim}",
        f"# lane_priority={args.lane_priority}",
        "",
    ]
    for off in changed:
        spec_lines.append(f"{len(base)} {off} 0x{patched[off]:02x}")

    out_spec = Path(args.out_spec)
    out_spec.parent.mkdir(parents=True, exist_ok=True)
    out_spec.write_text("\n".join(spec_lines).rstrip() + "\n")

    report: Dict[str, Any] = {
        "analysis_json": args.analysis_json,
        "base_exec": args.base_exec,
        "target_exec": args.target_exec,
        "chunk_index": args.chunk_index,
        "predict_mode": args.predict_mode,
        "field_spec": args.field_spec,
        "dims": {
            "low": args.low_dim,
            "high": args.high_dim,
            "target": args.target_dim,
            "tile_size": args.tile_size,
        },
        "payload_len": len(base),
        "assigned_word_offsets": len(assigned),
        "changed_byte_count": len(changed),
        "lane_histogram": {
            "lane32": sum(1 for x in assigned.values() if x["lane"] == "lane32"),
            "lane16": sum(1 for x in assigned.values() if x["lane"] == "lane16"),
        },
        "base_sha256": pe._sha256_bytes(bytes(base)),
        "patched_sha256": pe._sha256_bytes(bytes(patched)),
        "changed_offsets_preview": changed[:200],
    }

    if target is not None:
        if len(target) != len(base):
            raise SystemExit(
                f"target length mismatch: target={len(target)} base={len(base)}"
            )
        mism = [i for i in range(len(target)) if patched[i] != target[i]]
        report.update(
            {
                "target_sha256": pe._sha256_bytes(target),
                "mismatch_vs_target": len(mism),
                "mismatch_ratio_vs_target": len(mism) / float(len(target)),
                "mismatch_preview": mism[:200],
            }
        )

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Wrote patch spec: {out_spec}")
    print(
        "assigned_word_offsets={} changed_byte_count={} mismatch_vs_target={}".format(
            report["assigned_word_offsets"],
            report["changed_byte_count"],
            report.get("mismatch_vs_target", "n/a"),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
