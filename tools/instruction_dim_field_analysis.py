#!/usr/bin/env python3
"""Analyze dimension-dependent instruction-byte offsets across compiled executables.

Input entries are `dim:path_to_serialized_executable_XXX.bin`.
The tool extracts instruction bitstream chunk 0, computes changed-offset sets for
pairwise dimension comparisons, then reports intersection/union and per-offset
value trajectories/correlation across dimensions.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

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
    relocation_bytes: List[int]


def _iso_utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _parse_entry(raw: str) -> Entry:
    if ":" not in raw:
        raise SystemExit(f"invalid --entry '{raw}', expected DIM:PATH")
    left, right = raw.split(":", 1)
    try:
        dim = int(left)
    except ValueError as exc:
        raise SystemExit(f"invalid dimension in --entry '{raw}': {exc}")
    path = Path(right)
    if not path.is_file():
        raise SystemExit(f"entry path not found: {path}")
    return Entry(dim=dim, path=path)


def _parse_pair(raw: str) -> Tuple[int, int]:
    if ":" not in raw:
        raise SystemExit(f"invalid --pair '{raw}', expected DIMA:DIMB")
    left, right = raw.split(":", 1)
    try:
        a = int(left)
        b = int(right)
    except ValueError as exc:
        raise SystemExit(f"invalid --pair '{raw}': {exc}")
    if a == b:
        raise SystemExit(f"invalid --pair '{raw}': dimensions must differ")
    return (a, b)


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
    field_tables = pe._read_vector_table_field(table, 1)

    reloc: Set[int] = set()
    for ft in field_tables:
        parsed = pe._parse_field_offset(ft)
        bit_off = parsed.get("offset_bit")
        if isinstance(bit_off, int) and bit_off >= 0:
            b = bit_off // 8
            if b < len(bitstream):
                reloc.add(b)

    return Bitstream(
        dim=-1,
        path=str(path),
        size=len(bitstream),
        sha256=pe._sha256_bytes(bitstream),
        bytes_data=bitstream,
        relocation_bytes=sorted(reloc),
    )


def _changed_offsets(a: bytes, b: bytes) -> Set[int]:
    n = min(len(a), len(b))
    return {i for i in range(n) if a[i] != b[i]}


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs)
    deny = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(denx * deny)
    if den == 0.0:
        return 0.0
    return num / den


def _default_pairs(bitstreams: Dict[int, Bitstream]) -> List[Tuple[int, int]]:
    by_size: Dict[int, List[int]] = {}
    for dim, bs in bitstreams.items():
        by_size.setdefault(bs.size, []).append(dim)

    out: List[Tuple[int, int]] = []
    for _, dims in sorted(by_size.items()):
        if len(dims) < 2:
            continue
        dims = sorted(dims)
        out.append((dims[0], dims[-1]))
    if not out:
        raise SystemExit(
            "no default pairs resolved (need at least one bitstream-size bucket with >=2 dimensions)"
        )
    return out


def _run_length_summary(offsets: Sequence[int]) -> List[Tuple[int, int, int]]:
    if not offsets:
        return []
    runs: List[Tuple[int, int, int]] = []
    s = offsets[0]
    p = offsets[0]
    for off in offsets[1:]:
        if off == p + 1:
            p = off
            continue
        runs.append((s, p, p - s + 1))
        s = off
        p = off
    runs.append((s, p, p - s + 1))
    return runs


def _analyze_offsets(
    bitstreams: Dict[int, Bitstream],
    offsets: Sequence[int],
    top_n: int,
    min_abs_corr: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dims = sorted(bitstreams)
    x_dims = [float(d) for d in dims]
    x_tiles = [float(d // 64) for d in dims]

    all_rows: List[Dict[str, Any]] = []
    for off in offsets:
        vals: List[int] = []
        for d in dims:
            bs = bitstreams[d].bytes_data
            if off >= len(bs):
                vals = []
                break
            vals.append(bs[off])
        if not vals:
            continue

        row = {
            "offset": off,
            "values_by_dim": [{"dim": d, "value": vals[i]} for i, d in enumerate(dims)],
            "unique_values": len(set(vals)),
            "corr_dim": _corr(x_dims, [float(v) for v in vals]),
            "corr_tiles": _corr(x_tiles, [float(v) for v in vals]),
            "monotonic_non_decreasing": all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)),
            "monotonic_non_increasing": all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)),
        }
        all_rows.append(row)

    top_rows = [
        row
        for row in sorted(
            all_rows,
            key=lambda r: (abs(r["corr_dim"]), r["unique_values"], r["offset"]),
            reverse=True,
        )
        if abs(row["corr_dim"]) >= min_abs_corr
    ][:top_n]

    return all_rows, top_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dimension-diff analyzer for EdgeTPU instruction bitstreams"
    )
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="DIM:PATH to serialized_executable_XXX.bin (repeat)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="explicit DIMA:DIMB pair (repeat); default pairs by equal-size buckets",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="instruction bitstream chunk index within executable (default: 0)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="number of top correlated offsets to print/store (default: 40)",
    )
    parser.add_argument(
        "--min-abs-corr",
        type=float,
        default=0.9,
        help="minimum absolute dim-correlation for top section (default: 0.9)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="write full JSON report to this path",
    )
    args = parser.parse_args()

    entries = [_parse_entry(raw) for raw in args.entry]
    by_dim: Dict[int, Entry] = {}
    for e in entries:
        if e.dim in by_dim:
            raise SystemExit(f"duplicate dimension in --entry: {e.dim}")
        by_dim[e.dim] = e

    bitstreams: Dict[int, Bitstream] = {}
    for dim, entry in by_dim.items():
        bs = _extract_bitstream(entry.path, chunk_index=args.chunk_index)
        bs.dim = dim
        bitstreams[dim] = bs

    explicit_pairs = [_parse_pair(raw) for raw in args.pair]
    if explicit_pairs:
        pairs = explicit_pairs
    else:
        pairs = _default_pairs(bitstreams)

    for a, b in pairs:
        if a not in bitstreams or b not in bitstreams:
            raise SystemExit(f"pair references missing dimension: {a}:{b}")
        sa = bitstreams[a].size
        sb = bitstreams[b].size
        if sa != sb:
            raise SystemExit(
                f"pair {a}:{b} has size mismatch ({sa} vs {sb}); supply same-size pairs"
            )

    pair_summaries: List[Dict[str, Any]] = []
    changed_sets: List[Set[int]] = []
    union_reloc: Set[int] = set()
    for dim, bs in bitstreams.items():
        union_reloc.update(bs.relocation_bytes)

    for a, b in pairs:
        ba = bitstreams[a].bytes_data
        bb = bitstreams[b].bytes_data
        changed = _changed_offsets(ba, bb)
        changed_sets.append(changed)
        changed_sorted = sorted(changed)
        runs = _run_length_summary(changed_sorted)
        pair_summaries.append(
            {
                "pair": [a, b],
                "size": bitstreams[a].size,
                "changed_count": len(changed_sorted),
                "min_offset": changed_sorted[0] if changed_sorted else None,
                "max_offset": changed_sorted[-1] if changed_sorted else None,
                "changed_relocation_overlap": len(changed & union_reloc),
                "run_count": len(runs),
                "runs_preview": runs[:30],
                "offsets_preview": changed_sorted[:80],
            }
        )

    intersection = sorted(set.intersection(*changed_sets)) if changed_sets else []
    union = sorted(set.union(*changed_sets)) if changed_sets else []

    all_rows, top_rows = _analyze_offsets(
        bitstreams=bitstreams,
        offsets=intersection,
        top_n=args.top,
        min_abs_corr=args.min_abs_corr,
    )

    report: Dict[str, Any] = {
        "generated_at_utc": _iso_utc_now(),
        "chunk_index": args.chunk_index,
        "entries": [
            {
                "dim": dim,
                "path": bs.path,
                "size": bs.size,
                "sha256": bs.sha256,
                "relocation_byte_count": len(bs.relocation_bytes),
                "relocation_bytes_preview": bs.relocation_bytes[:64],
            }
            for dim, bs in sorted(bitstreams.items())
        ],
        "pairs": pair_summaries,
        "intersection_count": len(intersection),
        "union_count": len(union),
        "intersection_offsets": intersection,
        "union_offsets_preview": union[:300],
        "intersection_relocation_overlap": len(set(intersection) & union_reloc),
        "top_offsets": top_rows,
        "all_intersection_offsets": all_rows,
    }

    print("Instruction dimension field analysis")
    print(f"generated_at_utc={report['generated_at_utc']}")
    print(f"chunk_index={args.chunk_index}")
    print(f"entry_count={len(report['entries'])}")
    for e in report["entries"]:
        print(
            f"  dim={e['dim']} size={e['size']} reloc={e['relocation_byte_count']} path={e['path']}"
        )
    print(f"pair_count={len(pair_summaries)}")
    for p in pair_summaries:
        pair = p["pair"]
        print(
            f"  pair={pair[0]}:{pair[1]} size={p['size']} changed={p['changed_count']} runs={p['run_count']} reloc_overlap={p['changed_relocation_overlap']}"
        )
    print(
        f"intersection={report['intersection_count']} union={report['union_count']} intersection_reloc_overlap={report['intersection_relocation_overlap']}"
    )
    print(f"top_offsets={len(top_rows)} (|corr_dim| >= {args.min_abs_corr})")
    for row in top_rows:
        vals = [f"{kv['dim']}:{kv['value']}" for kv in row["values_by_dim"]]
        mono = row["monotonic_non_decreasing"] or row["monotonic_non_increasing"]
        print(
            "  off={} corr_dim={:+.3f} corr_tiles={:+.3f} uniq={} mono={} vals=[{}]".format(
                row["offset"],
                row["corr_dim"],
                row["corr_tiles"],
                row["unique_values"],
                mono,
                ", ".join(vals),
            )
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"json_out={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
