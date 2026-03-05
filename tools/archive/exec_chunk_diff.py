#!/usr/bin/env python3
"""Phase-B helper: diff EdgeTPU EXECUTION_ONLY bitstream chunks.

This tool compares instruction bitstream chunks from serialized executable blobs
(`serialized_executable_*.bin`) and reports byte/word-level deltas, including
how much of the delta overlaps known relocation-byte positions.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import parse_edgetpu_executable as pe


@dataclass
class ChunkData:
    index: int
    bitstream: bytes
    field_offsets_bits: List[int]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.bitstream).hexdigest()

    @property
    def size(self) -> int:
        return len(self.bitstream)

    @property
    def relocation_bytes(self) -> List[int]:
        out = sorted(
            {
                int(bit // 8)
                for bit in self.field_offsets_bits
                if bit is not None and bit >= 0 and (bit // 8) < len(self.bitstream)
            }
        )
        return out


@dataclass
class ExecutableData:
    path: Path
    chunks: List[ChunkData]


def _iso_utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _path_label(path: Path) -> str:
    parts = [part for part in path.parts if part not in ("/", "\\", "")]
    if not parts:
        return "root"
    return "__".join(parts).replace(":", "_")


def _resolve_input_paths(inputs: Sequence[str], only_exec_index: int | None) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"input path not found: {raw}")
        if path.is_file():
            if path.name.startswith("serialized_executable_") and path.suffix == ".bin":
                out.append(path)
            elif path.name == "serialized_multi_executable.bin":
                raise SystemExit(
                    f"unsupported input (needs executable blobs, not multi-executable): {path}"
                )
            else:
                raise SystemExit(
                    f"unsupported file input (expected serialized_executable_*.bin): {path}"
                )
            continue

        # directory input: collect serialized_executable_*.bin recursively
        found = sorted(path.rglob("serialized_executable_*.bin"))
        if only_exec_index is not None:
            filtered = [p for p in found if p.stem == f"serialized_executable_{only_exec_index:03d}"]
            if not filtered:
                raise SystemExit(
                    f"no serialized_executable_{only_exec_index:03d}.bin found under directory: {path}"
                )
            found = filtered
        if not found:
            raise SystemExit(f"no serialized_executable_*.bin found under directory: {path}")
        out.extend(found)

    # stable ordering improves pair reproducibility
    return sorted(out)


def _extract_executable(path: Path) -> ExecutableData:
    blob = path.read_bytes()
    root = pe._parse_root_table(blob, 0, file_identifier=None)
    instruction_tables = pe._read_vector_table_field(root, 5)
    chunks: List[ChunkData] = []
    for idx, table in enumerate(instruction_tables):
        bitstream = pe._read_vector_bytes_field(table, 0)
        field_tables = pe._read_vector_table_field(table, 1)
        field_offsets_bits: List[int] = []
        for field_table in field_tables:
            parsed = pe._parse_field_offset(field_table)
            bit_off = parsed.get("offset_bit")
            if isinstance(bit_off, int):
                field_offsets_bits.append(bit_off)
        chunks.append(
            ChunkData(index=idx, bitstream=bitstream, field_offsets_bits=sorted(field_offsets_bits))
        )
    return ExecutableData(path=path, chunks=chunks)


def _changed_positions(a: bytes, b: bytes) -> List[int]:
    n = min(len(a), len(b))
    out = [idx for idx in range(n) if a[idx] != b[idx]]
    if len(a) != len(b):
        out.extend(range(n, max(len(a), len(b))))
    return out


def _instruction_delta_count(changed_positions: Iterable[int], width: int, min_len: int) -> int:
    if width <= 0:
        return 0
    limit = (min_len // width) * width
    changed_instr = {pos // width for pos in changed_positions if pos < limit}
    return len(changed_instr)


def _diff_chunk(a: ChunkData, b: ChunkData, instruction_width: int) -> Dict[str, Any]:
    changed = _changed_positions(a.bitstream, b.bitstream)
    changed_set = set(changed)
    a_reloc = set(a.relocation_bytes)
    b_reloc = set(b.relocation_bytes)
    reloc_union = a_reloc | b_reloc

    min_len = min(a.size, b.size)
    max_len = max(a.size, b.size)
    byte_diff_count = len(changed)
    byte_diff_ratio = (byte_diff_count / max_len) if max_len > 0 else 0.0

    reloc_overlap = len(changed_set & reloc_union)
    non_reloc_changes = byte_diff_count - reloc_overlap

    instruction_count = min_len // instruction_width if instruction_width > 0 else 0
    changed_instruction_count = _instruction_delta_count(changed, instruction_width, min_len)

    return {
        "chunk_index": a.index,
        "a_size": a.size,
        "b_size": b.size,
        "a_sha256": a.sha256,
        "b_sha256": b.sha256,
        "equal": a.bitstream == b.bitstream,
        "byte_diff_count": byte_diff_count,
        "byte_diff_ratio": byte_diff_ratio,
        "relocation_bytes_a": len(a_reloc),
        "relocation_bytes_b": len(b_reloc),
        "relocation_bytes_union": len(reloc_union),
        "changed_relocation_bytes": reloc_overlap,
        "changed_non_relocation_bytes": non_reloc_changes,
        "instruction_width": instruction_width,
        "instruction_count_min_len": instruction_count,
        "changed_instruction_count": changed_instruction_count,
    }


def _build_pairs(executables: Sequence[ExecutableData], explicit_pairs: Sequence[str]) -> List[Tuple[int, int]]:
    if explicit_pairs:
        index_by_path = {str(exe.path): idx for idx, exe in enumerate(executables)}
        out: List[Tuple[int, int]] = []
        for pair in explicit_pairs:
            if ":" not in pair:
                raise SystemExit(f"invalid --pair '{pair}', expected A:B")
            left, right = pair.split(":", 1)
            if left not in index_by_path:
                raise SystemExit(f"--pair left path not found in inputs: {left}")
            if right not in index_by_path:
                raise SystemExit(f"--pair right path not found in inputs: {right}")
            out.append((index_by_path[left], index_by_path[right]))
        return out

    # default: consecutive diffs
    return [(idx, idx + 1) for idx in range(len(executables) - 1)]


def _dump_chunks(executables: Sequence[ExecutableData], out_dir: Path) -> Dict[str, Any]:
    dump_root = out_dir / "chunks"
    dump_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"executables": []}
    for exe in executables:
        exe_label = _path_label(exe.path)
        exe_dir = dump_root / exe_label
        exe_dir.mkdir(parents=True, exist_ok=True)

        chunk_entries: List[Dict[str, Any]] = []
        for chunk in exe.chunks:
            bin_name = f"chunk_{chunk.index:03d}.bin"
            json_name = f"chunk_{chunk.index:03d}.json"
            bin_path = exe_dir / bin_name
            meta_path = exe_dir / json_name
            bin_path.write_bytes(chunk.bitstream)
            meta = {
                "chunk_index": chunk.index,
                "size": chunk.size,
                "sha256": chunk.sha256,
                "field_offsets_bits": chunk.field_offsets_bits,
                "relocation_bytes": chunk.relocation_bytes,
            }
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            chunk_entries.append(
                {
                    "chunk_index": chunk.index,
                    "bin_path": str(bin_path),
                    "meta_path": str(meta_path),
                    "size": chunk.size,
                    "sha256": chunk.sha256,
                }
            )

        manifest["executables"].append(
            {
                "path": str(exe.path),
                "label": exe_label,
                "chunk_count": len(exe.chunks),
                "chunks": chunk_entries,
            }
        )
    return manifest


def _summarize_pair_diff(
    left: ExecutableData, right: ExecutableData, instruction_width: int
) -> Dict[str, Any]:
    max_chunks = max(len(left.chunks), len(right.chunks))
    chunk_diffs: List[Dict[str, Any]] = []
    missing_in_left: List[int] = []
    missing_in_right: List[int] = []
    for idx in range(max_chunks):
        a = left.chunks[idx] if idx < len(left.chunks) else None
        b = right.chunks[idx] if idx < len(right.chunks) else None
        if a is None:
            missing_in_left.append(idx)
            continue
        if b is None:
            missing_in_right.append(idx)
            continue
        chunk_diffs.append(_diff_chunk(a, b, instruction_width))

    total_changed_bytes = int(sum(item["byte_diff_count"] for item in chunk_diffs))
    total_changed_instr = int(sum(item["changed_instruction_count"] for item in chunk_diffs))
    total_instr = int(sum(item["instruction_count_min_len"] for item in chunk_diffs))
    equal_chunks = int(sum(1 for item in chunk_diffs if item["equal"]))

    return {
        "left_path": str(left.path),
        "right_path": str(right.path),
        "left_chunk_count": len(left.chunks),
        "right_chunk_count": len(right.chunks),
        "missing_in_left": missing_in_left,
        "missing_in_right": missing_in_right,
        "equal_chunks": equal_chunks,
        "compared_chunks": len(chunk_diffs),
        "total_changed_bytes": total_changed_bytes,
        "total_changed_instruction_words": total_changed_instr,
        "total_instruction_words_compared": total_instr,
        "chunk_diffs": chunk_diffs,
    }


def _print_human_summary(report: Dict[str, Any]) -> None:
    print("EdgeTPU EXECUTION_ONLY chunk diff")
    print(f"generated_at_utc={report['generated_at_utc']}")
    print(f"instruction_width={report['instruction_width']}")
    print(f"input_count={len(report['inputs'])}")
    for item in report["inputs"]:
        print(
            f"  input={item['path']} chunks={item['chunk_count']} instruction_total_bytes={item['instruction_total_bytes']}"
        )

    for idx, pair in enumerate(report["pairs"]):
        print()
        print(f"[pair {idx}] {pair['left_path']} -> {pair['right_path']}")
        print(
            "  chunks compared={} equal={} missing_left={} missing_right={}".format(
                pair["compared_chunks"],
                pair["equal_chunks"],
                len(pair["missing_in_left"]),
                len(pair["missing_in_right"]),
            )
        )
        print(
            "  changed_bytes={} changed_instr_words={} / {}".format(
                pair["total_changed_bytes"],
                pair["total_changed_instruction_words"],
                pair["total_instruction_words_compared"],
            )
        )
        for chunk in pair["chunk_diffs"]:
            print(
                "    chunk={} equal={} diff_bytes={} diff_ratio={:.6f} reloc_changed={} non_reloc_changed={} instr_changed={}/{}".format(
                    chunk["chunk_index"],
                    chunk["equal"],
                    chunk["byte_diff_count"],
                    chunk["byte_diff_ratio"],
                    chunk["changed_relocation_bytes"],
                    chunk["changed_non_relocation_bytes"],
                    chunk["changed_instruction_count"],
                    chunk["instruction_count_min_len"],
                )
            )

    if report.get("chunk_dump_manifest_path"):
        print()
        print(f"chunk_dump_manifest={report['chunk_dump_manifest_path']}")
    if report.get("json_out_path"):
        print(f"json_report={report['json_out_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diff EdgeTPU serialized executable instruction chunks for Phase-B RE."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="serialized_executable_*.bin files and/or directories containing them",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="explicit comparison pair A:B where A/B are full input paths from this invocation",
    )
    parser.add_argument(
        "--instruction-width",
        type=int,
        default=8,
        help="word width (bytes) for changed-instruction aggregation (default: 8)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory for JSON report and optional chunk dumps",
    )
    parser.add_argument(
        "--only-exec-index",
        type=int,
        default=None,
        help="when directory inputs contain multiple executables, keep only serialized_executable_<N>.bin",
    )
    parser.add_argument(
        "--dump-chunks",
        action="store_true",
        help="dump extracted chunk bytes + metadata under <out-dir>/chunks",
    )
    parser.add_argument("--json", action="store_true", help="print JSON report to stdout")
    args = parser.parse_args()

    if args.instruction_width <= 0:
        raise SystemExit("--instruction-width must be >= 1")
    if args.dump_chunks and not args.out_dir:
        raise SystemExit("--dump-chunks requires --out-dir")

    if args.only_exec_index is not None and args.only_exec_index < 0:
        raise SystemExit("--only-exec-index must be >= 0")

    input_paths = _resolve_input_paths(args.paths, only_exec_index=args.only_exec_index)
    if len(input_paths) < 2:
        raise SystemExit("need at least two executable inputs to diff")

    executables = [_extract_executable(path) for path in input_paths]
    pairs = _build_pairs(executables, args.pair)
    if not pairs:
        raise SystemExit("no comparison pairs resolved")

    report: Dict[str, Any] = {
        "generated_at_utc": _iso_utc_now(),
        "instruction_width": args.instruction_width,
        "inputs": [
            {
                "path": str(exe.path),
                "chunk_count": len(exe.chunks),
                "instruction_total_bytes": int(sum(chunk.size for chunk in exe.chunks)),
                "chunks": [
                    {
                        "chunk_index": chunk.index,
                        "size": chunk.size,
                        "sha256": chunk.sha256,
                        "relocation_byte_count": len(chunk.relocation_bytes),
                    }
                    for chunk in exe.chunks
                ],
            }
            for exe in executables
        ],
        "pairs": [],
    }

    for left_idx, right_idx in pairs:
        pair_summary = _summarize_pair_diff(
            executables[left_idx], executables[right_idx], instruction_width=args.instruction_width
        )
        report["pairs"].append(pair_summary)

    out_dir: Path | None = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.dump_chunks:
            manifest = _dump_chunks(executables, out_dir=out_dir)
            manifest_path = out_dir / "chunk_dump_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            report["chunk_dump_manifest_path"] = str(manifest_path)
        json_out_path = out_dir / "exec_chunk_diff_report.json"
        json_out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report["json_out_path"] = str(json_out_path)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_summary(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
