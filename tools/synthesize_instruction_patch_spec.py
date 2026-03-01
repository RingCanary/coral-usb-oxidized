#!/usr/bin/env python3
"""Synthesize instruction patch spec for replay from endpoint dimensions.

Given low/high endpoint instruction streams and a target dimension, this tool:
- predicts target bytes by linear interpolation per byte,
- optionally compares prediction against real target bytes,
- emits replay patch spec (<payload_len> <offset> <value>) relative to base bytes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import parse_edgetpu_executable as pe


@dataclass(frozen=True)
class StreamConfig:
    name: str
    low_path: Path
    high_path: Path
    base_path: Path
    target_path: Optional[Path]


def _read_chunk_bytes(path: Path, chunk_index: int) -> bytes:
    blob = path.read_bytes()
    root = pe._parse_root_table(blob, 0, file_identifier=None)
    tables = pe._read_vector_table_field(root, 5)
    if chunk_index < 0 or chunk_index >= len(tables):
        raise SystemExit(
            f"chunk index {chunk_index} out of range for {path} (count={len(tables)})"
        )
    return pe._read_vector_bytes_field(tables[chunk_index], 0)


def _predict_byte(low: int, high: int, low_dim: int, high_dim: int, target_dim: int) -> int:
    if high_dim == low_dim:
        return low
    frac = (target_dim - low_dim) / float(high_dim - low_dim)
    val = round(low + (high - low) * frac)
    return max(0, min(255, int(val)))


def _parse_stream_args(args: argparse.Namespace) -> List[StreamConfig]:
    streams = [
        StreamConfig(
            name="eo",
            low_path=Path(args.low_eo),
            high_path=Path(args.high_eo),
            base_path=Path(args.base_eo) if args.base_eo else Path(args.target_eo),
            target_path=Path(args.target_eo) if args.target_eo else None,
        ),
        StreamConfig(
            name="pc",
            low_path=Path(args.low_pc),
            high_path=Path(args.high_pc),
            base_path=Path(args.base_pc) if args.base_pc else Path(args.target_pc),
            target_path=Path(args.target_pc) if args.target_pc else None,
        ),
    ]

    for s in streams:
        for p in [s.low_path, s.high_path, s.base_path]:
            if not p.is_file():
                raise SystemExit(f"missing file for stream '{s.name}': {p}")
        if s.target_path is not None and not s.target_path.is_file():
            raise SystemExit(f"missing target file for stream '{s.name}': {s.target_path}")
    return streams


def _offset_set(mode: str, low: bytes, high: bytes) -> List[int]:
    n = min(len(low), len(high))
    if mode == "all":
        return list(range(n))
    if mode == "changed-extrema":
        return [i for i in range(n) if low[i] != high[i]]
    raise SystemExit(f"unsupported --offset-mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-dim", type=int, required=True)
    parser.add_argument("--high-dim", type=int, required=True)
    parser.add_argument("--target-dim", type=int, required=True)

    parser.add_argument("--low-eo", required=True)
    parser.add_argument("--high-eo", required=True)
    parser.add_argument("--target-eo", required=True)
    parser.add_argument("--base-eo", default=None)

    parser.add_argument("--low-pc", required=True)
    parser.add_argument("--high-pc", required=True)
    parser.add_argument("--target-pc", required=True)
    parser.add_argument("--base-pc", default=None)

    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument(
        "--offset-mode",
        default="changed-extrema",
        choices=["changed-extrema", "all"],
        help="which offsets to synthesize/emit (default: changed-extrema)",
    )
    parser.add_argument("--out-spec", required=True)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    if args.low_dim == args.high_dim:
        raise SystemExit("--low-dim and --high-dim must differ")

    streams = _parse_stream_args(args)

    spec_lines: List[str] = []
    report: Dict[str, object] = {
        "low_dim": args.low_dim,
        "high_dim": args.high_dim,
        "target_dim": args.target_dim,
        "offset_mode": args.offset_mode,
        "streams": [],
    }

    total_rules = 0

    for s in streams:
        low = _read_chunk_bytes(s.low_path, args.chunk_index)
        high = _read_chunk_bytes(s.high_path, args.chunk_index)
        base = _read_chunk_bytes(s.base_path, args.chunk_index)
        target = _read_chunk_bytes(s.target_path, args.chunk_index) if s.target_path else None

        if not (len(low) == len(high) == len(base)):
            raise SystemExit(
                f"length mismatch in stream '{s.name}': low={len(low)} high={len(high)} base={len(base)}"
            )
        if target is not None and len(target) != len(low):
            raise SystemExit(
                f"length mismatch in stream '{s.name}': target={len(target)} vs {len(low)}"
            )

        offsets = _offset_set(args.offset_mode, low, high)

        predicted = bytearray(base)
        for off in offsets:
            predicted[off] = _predict_byte(
                low=low[off],
                high=high[off],
                low_dim=args.low_dim,
                high_dim=args.high_dim,
                target_dim=args.target_dim,
            )

        changed_vs_base = [i for i in range(len(base)) if predicted[i] != base[i]]
        for off in changed_vs_base:
            spec_lines.append(f"{len(base)} {off} 0x{predicted[off]:02x}")

        stream_report: Dict[str, object] = {
            "name": s.name,
            "payload_len": len(base),
            "offset_count_input": len(offsets),
            "rules_emitted": len(changed_vs_base),
            "base_sha256": pe._sha256_bytes(base),
            "predicted_sha256": pe._sha256_bytes(bytes(predicted)),
            "low_sha256": pe._sha256_bytes(low),
            "high_sha256": pe._sha256_bytes(high),
            "changed_vs_base_preview": changed_vs_base[:80],
        }

        if target is not None:
            mismatches = [i for i in range(len(target)) if predicted[i] != target[i]]
            stream_report.update(
                {
                    "target_sha256": pe._sha256_bytes(target),
                    "mismatch_vs_target": len(mismatches),
                    "mismatch_ratio_vs_target": len(mismatches) / float(len(target)),
                    "mismatch_preview": mismatches[:80],
                }
            )

        report["streams"].append(stream_report)
        total_rules += len(changed_vs_base)

    out_spec = Path(args.out_spec)
    out_spec.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# synthesized by synthesize_instruction_patch_spec.py",
        f"# low_dim={args.low_dim} high_dim={args.high_dim} target_dim={args.target_dim}",
        f"# offset_mode={args.offset_mode}",
    ]
    out_spec.write_text("\n".join(header + [""] + spec_lines).rstrip() + "\n")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Wrote patch spec: {out_spec}")
    print(f"Rules emitted: {total_rules}")
    for s in report["streams"]:  # type: ignore[index]
        print(
            "  stream={} len={} input_offsets={} emitted_rules={} mismatch_vs_target={}".format(
                s["name"],
                s["payload_len"],
                s["offset_count_input"],
                s["rules_emitted"],
                s.get("mismatch_vs_target", "n/a"),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
