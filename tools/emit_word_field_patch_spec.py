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
        fit = _fit_linear(float(tl), float(low_val), float(th), float(high_val))
        if fit is None:
            return int(low_val)
        return _predict_linear(fit[0], fit[1], float(tt))
    if model == "tile2div-linear":
        d = int(params.get("div", 1))
        xl = float((tl * tl) // d)
        xh = float((th * th) // d)
        xt = float((tt * tt) // d)
        fit = _fit_linear(xl, float(low_val), xh, float(high_val))
        if fit is None:
            return int(low_val)
        return _predict_linear(fit[0], fit[1], xt)
    if model == "tile-quadratic":
        # underconstrained with 2 points; fallback to tile-linear in tiles domain
        fit = _fit_linear(float(tl), float(low_val), float(th), float(high_val))
        if fit is None:
            return int(low_val)
        return _predict_linear(fit[0], fit[1], float(tt))

    return int(low_val)


def _values_map(per: Dict[str, Any]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for row in per.get("values_by_dim", []):
        out[int(row["dim"])] = int(row["value"])
    return out


def _iter_per_offsets(lane_obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for g in lane_obj.get("top_groups", []):
        for per in g.get("per_offset_fits", []):
            row = dict(per)
            row["stride_residue"] = g.get("stride_residue")
            yield row


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
            model, p = _pick_formula(best, mode=args.predict_mode)
            pred_val = _predict_value(
                model=model,
                params=p,
                low_dim=args.low_dim,
                low_val=low_val,
                high_dim=args.high_dim,
                high_val=high_val,
                target_dim=args.target_dim,
                tile_size=args.tile_size,
                mode=args.predict_mode,
            )

            max_val = (1 << (8 * lane.lane_bytes)) - 1
            pred_val &= max_val
            pred_bytes = pred_val.to_bytes(lane.lane_bytes, "little", signed=False)

            assigned[off] = {
                "offset": off,
                "lane": lane.lane_key,
                "lane_bytes": lane.lane_bytes,
                "model": model,
                "params": p,
                "pred_value": pred_val,
                "pred_bytes": list(pred_bytes),
                "stride_residue": per.get("stride_residue"),
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
