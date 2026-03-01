#!/usr/bin/env python3
"""Validate 2-point holdout prediction on word-field analysis JSON.

Given analysis JSON (typically 3 dims), this tool predicts target-dim values
from low/high endpoints using model policies derived from group best candidates,
and reports per-lane/per-group mismatch counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _tiles(dim: int, tile_size: int) -> int:
    return dim // tile_size


def _fit_linear(x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float]]:
    if x1 == x2:
        return None
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return (a, b)


def _predict_linear(a: float, b: float, x: float) -> int:
    return int(round((a * x) + b))


def _pick_policy_formula(best: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    model = str(best.get("model", "none"))
    if model in ("const", "tile-linear", "tile2div-linear"):
        div = int(best.get("params", {}).get("div", 1)) if model == "tile2div-linear" else None
        return (model, div)

    # For tile-quadratic: prefer exact tile2div-linear alternate if present.
    for cand in best.get("top_candidates", []):
        if (
            cand.get("model") == "tile2div-linear"
            and float(cand.get("exact_ratio", 0.0)) == 1.0
            and float(cand.get("mae", 1.0)) == 0.0
        ):
            div = int(cand.get("params", {}).get("div", 1))
            return ("tile2div-linear", div)

    # fallback: tile-linear in tiles domain
    return ("tile-linear", None)


def _map_values_by_dim(values_by_dim: Sequence[Dict[str, Any]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for row in values_by_dim:
        d = int(row["dim"])
        vals = [int(v) for v in row["values"]]
        out[d] = vals
    return out


def _map_scalar_values(values_by_dim: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for row in values_by_dim:
        out[int(row["dim"])] = int(row["value"])
    return out


def _predict_value(
    model: str,
    div: Optional[int],
    low_dim: int,
    low_val: int,
    high_dim: int,
    high_val: int,
    target_dim: int,
    tile_size: int,
) -> int:
    if model == "const":
        return low_val

    tl = _tiles(low_dim, tile_size)
    th = _tiles(high_dim, tile_size)
    tt = _tiles(target_dim, tile_size)

    if model == "tile-linear":
        fit = _fit_linear(float(tl), float(low_val), float(th), float(high_val))
        if fit is None:
            return low_val
        return _predict_linear(fit[0], fit[1], float(tt))

    if model == "tile2div-linear":
        d = int(div or 1)
        xl = float((tl * tl) // d)
        xh = float((th * th) // d)
        xt = float((tt * tt) // d)
        fit = _fit_linear(xl, float(low_val), xh, float(high_val))
        if fit is None:
            return low_val
        return _predict_linear(fit[0], fit[1], xt)

    # fallback
    fit = _fit_linear(float(tl), float(low_val), float(th), float(high_val))
    if fit is None:
        return low_val
    return _predict_linear(fit[0], fit[1], float(tt))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-json", required=True)
    ap.add_argument("--lane", choices=["lane16", "lane32"], default="lane16")
    ap.add_argument("--low-dim", type=int, required=True)
    ap.add_argument("--target-dim", type=int, required=True)
    ap.add_argument("--high-dim", type=int, required=True)
    ap.add_argument("--tile-size", type=int, default=64)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    report = json.loads(Path(args.analysis_json).read_text())
    lane = report[args.lane]

    total_offsets = 0
    mismatches = 0
    unresolved = 0
    group_rows: List[Dict[str, Any]] = []

    for g in lane["top_groups"]:
        vals_map = _map_values_by_dim(g["values_by_dim"])
        offs: List[int] = [int(x) for x in g["offsets"]]
        if args.low_dim not in vals_map or args.target_dim not in vals_map or args.high_dim not in vals_map:
            continue

        gm = 0
        for i, off in enumerate(offs):
            # Prefer per-offset formula/values when present; fallback to group vectors.
            per = None
            if "per_offset_fits" in g and i < len(g["per_offset_fits"]):
                per = g["per_offset_fits"][i]

            if per is not None:
                scalars = _map_scalar_values(per.get("values_by_dim", []))
                if args.low_dim in scalars and args.target_dim in scalars and args.high_dim in scalars:
                    lv = scalars[args.low_dim]
                    tv = scalars[args.target_dim]
                    hv = scalars[args.high_dim]
                else:
                    lv = int(vals_map[args.low_dim][i])
                    tv = int(vals_map[args.target_dim][i])
                    hv = int(vals_map[args.high_dim][i])
                best = per.get("best_formula", g.get("best_formula", {}))
            else:
                lv = int(vals_map[args.low_dim][i])
                tv = int(vals_map[args.target_dim][i])
                hv = int(vals_map[args.high_dim][i])
                best = g.get("best_formula", {})

            model, div = _pick_policy_formula(best)

            pred = _predict_value(
                model=model,
                div=div,
                low_dim=args.low_dim,
                low_val=lv,
                high_dim=args.high_dim,
                high_val=hv,
                target_dim=args.target_dim,
                tile_size=args.tile_size,
            )

            total_offsets += 1
            if pred != tv:
                mismatches += 1
                gm += 1

        if model not in ("const", "tile-linear", "tile2div-linear"):
            unresolved += len(offs)

        group_rows.append(
            {
                "stride_residue": g["stride_residue"],
                "offset_count": len(offs),
                "model_used": "mixed-per-offset" if "per_offset_fits" in g else model,
                "div": None,
                "group_mismatches": gm,
                "group_match_ratio": (len(offs) - gm) / float(len(offs)),
            }
        )

    out = {
        "analysis_json": args.analysis_json,
        "lane": args.lane,
        "low_dim": args.low_dim,
        "target_dim": args.target_dim,
        "high_dim": args.high_dim,
        "total_offsets": total_offsets,
        "mismatches": mismatches,
        "match_ratio": (total_offsets - mismatches) / float(total_offsets) if total_offsets else 0.0,
        "unresolved_offsets": unresolved,
        "groups": group_rows,
    }

    print(json.dumps(out, indent=2, sort_keys=True))

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
