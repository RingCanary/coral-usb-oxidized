#!/usr/bin/env python3
"""Word/halfword-level dimension analysis for EdgeTPU instruction bitstreams.

This complements byte-wise diff tools by operating on u16/u32 lanes, detecting
repeating record-stride groups, and fitting simple tile-domain formulas.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import parse_edgetpu_executable as pe


@dataclass(frozen=True)
class Entry:
    dim: int
    path: Path


@dataclass
class Bitstream:
    dim: int
    path: str
    size: int
    sha256: str
    bytes_data: bytes


def _iso_utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _parse_entry(raw: str) -> Entry:
    if ":" not in raw:
        raise SystemExit(f"invalid --entry '{raw}', expected DIM:PATH")
    left, right = raw.split(":", 1)
    try:
        dim = int(left)
    except ValueError as exc:
        raise SystemExit(f"invalid DIM in --entry '{raw}': {exc}") from exc
    path = Path(right)
    if not path.is_file():
        raise SystemExit(f"entry path not found: {path}")
    return Entry(dim=dim, path=path)


def _extract_bitstream(path: Path, chunk_index: int) -> Bitstream:
    blob = path.read_bytes()
    root = pe._parse_root_table(blob, 0, file_identifier=None)
    instruction_tables = pe._read_vector_table_field(root, 5)
    if chunk_index < 0 or chunk_index >= len(instruction_tables):
        raise SystemExit(
            f"chunk index {chunk_index} out of range for {path} (count={len(instruction_tables)})"
        )
    table = instruction_tables[chunk_index]
    bitstream = pe._read_vector_bytes_field(table, 0)
    return Bitstream(
        dim=-1,
        path=str(path),
        size=len(bitstream),
        sha256=pe._sha256_bytes(bitstream),
        bytes_data=bitstream,
    )


def _lane_values(data: bytes, lane_bytes: int) -> List[int]:
    usable = len(data) - (len(data) % lane_bytes)
    out: List[int] = []
    for off in range(0, usable, lane_bytes):
        out.append(int.from_bytes(data[off : off + lane_bytes], "little", signed=False))
    return out


def _changed_lane_indexes(values_by_dim: Sequence[List[int]]) -> List[int]:
    count = min(len(v) for v in values_by_dim)
    out: List[int] = []
    for idx in range(count):
        vals = {v[idx] for v in values_by_dim}
        if len(vals) > 1:
            out.append(idx)
    return out


def _candidate_strides(total_size: int) -> List[int]:
    cands = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
    return [c for c in cands if c <= total_size]


def _detect_stride(offsets: Sequence[int], total_size: int) -> int:
    if not offsets:
        return 64
    best_stride = 64
    best_score = -1.0
    offset_set = set(offsets)
    for stride in _candidate_strides(total_size):
        progression_hits = sum(1 for off in offsets if (off + stride) in offset_set)
        buckets: Dict[int, List[int]] = {}
        for off in offsets:
            buckets.setdefault(off % stride, []).append(off)
        repeated = [len(v) for v in buckets.values() if len(v) >= 2]
        repeated_weight = sum(x - 1 for x in repeated)
        bucket_count = len(repeated)
        # Prioritize true periodic progression (off+s exists), then residue grouping.
        score = (progression_hits * 1_000_000.0) + (repeated_weight * 1000.0) + float(bucket_count)
        if score > best_score:
            best_score = score
            best_stride = stride
    return best_stride


def _solve_3x3(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        p = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= p
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def _fit_const(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float]]:
    if not ys:
        return None
    return (sum(ys) / len(ys),)


def _fit_linear(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float]]:
    n = len(xs)
    if n < 2:
        return None
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = (n * sxx) - (sx * sx)
    if abs(den) < 1e-12:
        return None
    a = ((n * sxy) - (sx * sy)) / den
    b = (sy - a * sx) / n
    return (a, b)


def _fit_quadratic(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    if len(xs) < 3:
        return None
    s1 = float(len(xs))
    sx = sum(xs)
    sx2 = sum(x * x for x in xs)
    sx3 = sum(x * x * x for x in xs)
    sx4 = sum(x * x * x * x for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2y = sum((x * x) * y for x, y in zip(xs, ys))
    sol = _solve_3x3(
        a=[[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, s1]],
        b=[sx2y, sxy, sy],
    )
    if sol is None:
        return None
    return (sol[0], sol[1], sol[2])


def _predict_model(name: str, x: float, params: Tuple[float, ...]) -> float:
    if name == "const":
        return params[0]
    if name in ("tile-linear", "tile2div-linear"):
        return (params[0] * x) + params[1]
    if name == "tile-quadratic":
        return (params[0] * x * x) + (params[1] * x) + params[2]
    raise ValueError(name)


def _fit_best_formula(
    dims: Sequence[int],
    vals: Sequence[int],
    tile_size: int,
    t2_divs: Sequence[int],
) -> Dict[str, Any]:
    tiles = [float(d // tile_size) for d in dims]
    ys = [float(v) for v in vals]

    candidates: List[Dict[str, Any]] = []

    c = _fit_const(tiles, ys)
    if c is not None:
        preds = [int(round(_predict_model("const", x, c))) for x in tiles]
        err = [abs(p - v) for p, v in zip(preds, vals)]
        candidates.append(
            {
                "model": "const",
                "params": {"c": float(c[0])},
                "exact_ratio": sum(1 for e in err if e == 0) / len(err),
                "mae": sum(err) / len(err),
                "complexity": 0,
            }
        )

    lin = _fit_linear(tiles, ys)
    if lin is not None:
        preds = [int(round(_predict_model("tile-linear", x, lin))) for x in tiles]
        err = [abs(p - v) for p, v in zip(preds, vals)]
        candidates.append(
            {
                "model": "tile-linear",
                "params": {"a": float(lin[0]), "b": float(lin[1])},
                "exact_ratio": sum(1 for e in err if e == 0) / len(err),
                "mae": sum(err) / len(err),
                "complexity": 1,
            }
        )

    quad = _fit_quadratic(tiles, ys)
    if quad is not None:
        preds = [int(round(_predict_model("tile-quadratic", x, quad))) for x in tiles]
        err = [abs(p - v) for p, v in zip(preds, vals)]
        candidates.append(
            {
                "model": "tile-quadratic",
                "params": {"a": float(quad[0]), "b": float(quad[1]), "c": float(quad[2])},
                "exact_ratio": sum(1 for e in err if e == 0) / len(err),
                "mae": sum(err) / len(err),
                "complexity": 2,
            }
        )

    for div in t2_divs:
        x2 = [float((int(t) * int(t)) // div) for t in tiles]
        tl = _fit_linear(x2, ys)
        if tl is None:
            continue
        preds = [int(round(_predict_model("tile2div-linear", x, tl))) for x in x2]
        err = [abs(p - v) for p, v in zip(preds, vals)]
        candidates.append(
            {
                "model": "tile2div-linear",
                "params": {"a": float(tl[0]), "b": float(tl[1]), "div": int(div)},
                "exact_ratio": sum(1 for e in err if e == 0) / len(err),
                "mae": sum(err) / len(err),
                "complexity": 2,
            }
        )

    if not candidates:
        return {
            "model": "none",
            "params": {},
            "exact_ratio": 0.0,
            "mae": float("inf"),
            "complexity": 99,
            "candidate_count": 0,
            "top_candidates": [],
        }

    candidates.sort(
        key=lambda x: (
            -float(x["exact_ratio"]),
            float(x["mae"]),
            int(x["complexity"]),
            str(x["model"]),
        )
    )
    best = dict(candidates[0])
    best["candidate_count"] = len(candidates)
    best["top_candidates"] = candidates[:8]
    return best


def _bit_ranges(mask: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    b = 0
    while b < 32:
        if ((mask >> b) & 1) == 0:
            b += 1
            continue
        start = b
        while b < 32 and ((mask >> b) & 1) == 1:
            b += 1
        ranges.append((start, b - 1))
    return ranges


def _analyze_lane(
    bitstreams: Dict[int, Bitstream],
    lane_bytes: int,
    tile_size: int,
    stride: Optional[int],
    top: int,
    t2_divs: Sequence[int],
) -> Dict[str, Any]:
    dims = sorted(bitstreams)
    payloads = [bitstreams[d].bytes_data for d in dims]
    lanes = [_lane_values(p, lane_bytes) for p in payloads]
    changed_idx = _changed_lane_indexes(lanes)
    changed_offsets = [idx * lane_bytes for idx in changed_idx]

    payload_size = min(len(p) for p in payloads)
    stride_used = stride if stride is not None else _detect_stride(changed_offsets, payload_size)

    groups: Dict[int, List[int]] = {}
    for off in changed_offsets:
        groups.setdefault(off % stride_used, []).append(off)

    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    group_reports: List[Dict[str, Any]] = []
    for residue, offs in ranked[:top]:
        offs_sorted = sorted(offs)
        values_by_dim: Dict[int, List[int]] = {}
        for d, lane in zip(dims, lanes):
            values_by_dim[d] = [lane[o // lane_bytes] for o in offs_sorted]

        repeated_identical = all(len(set(values_by_dim[d])) == 1 for d in dims)

        rep_vals: List[int]
        if repeated_identical:
            rep_vals = [values_by_dim[d][0] for d in dims]
        else:
            rep_vals = [values_by_dim[d][0] for d in dims]

        best_formula = _fit_best_formula(dims, rep_vals, tile_size=tile_size, t2_divs=t2_divs)

        row: Dict[str, Any] = {
            "lane_bytes": lane_bytes,
            "stride_residue": residue,
            "occurrence_count": len(offs_sorted),
            "offsets": offs_sorted,
            "repeated_identical_per_dim": repeated_identical,
            "values_by_dim": [{"dim": d, "values": values_by_dim[d]} for d in dims],
            "representative_values_by_dim": [{"dim": d, "value": rep_vals[i]} for i, d in enumerate(dims)],
            "best_formula": best_formula,
        }

        per_offset_fits: List[Dict[str, Any]] = []
        for idx, off in enumerate(offs_sorted):
            vals = [int(values_by_dim[d][idx]) for d in dims]
            off_best = _fit_best_formula(dims, vals, tile_size=tile_size, t2_divs=t2_divs)
            per_offset_fits.append(
                {
                    "offset": off,
                    "values_by_dim": [{"dim": d, "value": vals[i]} for i, d in enumerate(dims)],
                    "best_formula": off_best,
                }
            )
        row["per_offset_fits"] = per_offset_fits

        if lane_bytes == 4:
            wvals = rep_vals
            vary_mask = 0
            ref = wvals[0]
            for v in wvals[1:]:
                vary_mask |= (ref ^ v)
            ranges = _bit_ranges(vary_mask)
            bitfield_rows: List[Dict[str, Any]] = []
            for lo, hi in ranges:
                width = hi - lo + 1
                if width > 16:
                    continue
                mask = (1 << width) - 1
                sub_vals = [int((v >> lo) & mask) for v in wvals]
                sub_formula = _fit_best_formula(dims, sub_vals, tile_size=tile_size, t2_divs=t2_divs)
                bitfield_rows.append(
                    {
                        "bit_range": [lo, hi],
                        "width": width,
                        "values_by_dim": [{"dim": d, "value": sub_vals[i]} for i, d in enumerate(dims)],
                        "best_formula": sub_formula,
                    }
                )
            row["representative_vary_mask_hex"] = f"0x{vary_mask:08x}"
            row["bitfield_fits"] = bitfield_rows

        group_reports.append(row)

    return {
        "lane_bytes": lane_bytes,
        "lane_count": min(len(x) for x in lanes),
        "changed_lane_count": len(changed_idx),
        "changed_offset_count": len(changed_offsets),
        "stride_used": stride_used,
        "top_groups": group_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="append", required=True, help="DIM:PATH (repeat)")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--stride", default="auto", help="record stride bytes or 'auto' (default)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--t2-divs",
        default="1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256",
        help="comma-separated divisors for tile2div-linear fitting",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    entries = [_parse_entry(x) for x in args.entry]
    by_dim: Dict[int, Entry] = {}
    for e in entries:
        if e.dim in by_dim:
            raise SystemExit(f"duplicate dimension: {e.dim}")
        by_dim[e.dim] = e

    bitstreams: Dict[int, Bitstream] = {}
    for dim, ent in sorted(by_dim.items()):
        bs = _extract_bitstream(ent.path, args.chunk_index)
        bs.dim = dim
        bitstreams[dim] = bs

    sizes = {bs.size for bs in bitstreams.values()}
    if len(sizes) != 1:
        raise SystemExit(f"all entries must have same bitstream size, got: {sorted(sizes)}")

    if args.tile_size <= 0:
        raise SystemExit("--tile-size must be > 0")

    stride: Optional[int]
    if args.stride == "auto":
        stride = None
    else:
        try:
            stride = int(args.stride)
        except ValueError as exc:
            raise SystemExit(f"invalid --stride: {args.stride}") from exc
        if stride <= 0:
            raise SystemExit("--stride must be > 0")

    try:
        t2_divs = [int(x.strip()) for x in args.t2_divs.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --t2-divs: {args.t2_divs}") from exc
    if not t2_divs:
        raise SystemExit("--t2-divs must contain at least one divisor")

    lane16 = _analyze_lane(
        bitstreams=bitstreams,
        lane_bytes=2,
        tile_size=args.tile_size,
        stride=stride,
        top=args.top,
        t2_divs=t2_divs,
    )
    lane32 = _analyze_lane(
        bitstreams=bitstreams,
        lane_bytes=4,
        tile_size=args.tile_size,
        stride=stride,
        top=args.top,
        t2_divs=t2_divs,
    )

    report: Dict[str, Any] = {
        "generated_at_utc": _iso_utc_now(),
        "chunk_index": args.chunk_index,
        "tile_size": args.tile_size,
        "entries": [
            {
                "dim": d,
                "path": bitstreams[d].path,
                "size": bitstreams[d].size,
                "sha256": bitstreams[d].sha256,
            }
            for d in sorted(bitstreams)
        ],
        "lane16": lane16,
        "lane32": lane32,
    }

    print("Instruction word-field analysis")
    print(f"generated_at_utc={report['generated_at_utc']}")
    print(f"entry_count={len(report['entries'])} size={report['entries'][0]['size']}")
    for e in report["entries"]:
        print(f"  dim={e['dim']} path={e['path']}")

    for lane_key in ["lane16", "lane32"]:
        lane = report[lane_key]
        print(
            f"{lane_key}: changed={lane['changed_lane_count']} stride={lane['stride_used']} top_groups={len(lane['top_groups'])}"
        )
        for g in lane["top_groups"][: min(8, len(lane["top_groups"]))]:
            bf = g["best_formula"]
            print(
                "  residue={} count={} repeat_identical={} model={} exact={:.3f} mae={:.3f}".format(
                    g["stride_residue"],
                    g["occurrence_count"],
                    g["repeated_identical_per_dim"],
                    bf["model"],
                    bf["exact_ratio"],
                    bf["mae"],
                )
            )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"json_out={out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
