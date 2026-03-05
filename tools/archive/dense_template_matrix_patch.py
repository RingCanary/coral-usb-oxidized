#!/usr/bin/env python3
"""Patch Dense(256x256) template parameter bytes using recovered layout mapping."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inspect_model(model_path: Path, repo_root: Path) -> Dict:
    proc = subprocess.run(
        [
            "python3",
            str(repo_root / "tools" / "tensorizer_patch_edgetpu.py"),
            "inspect",
            "--json",
            str(model_path),
        ],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"inspect failed for {model_path}:\n{proc.stderr}\n{proc.stdout}")
    return json.loads(proc.stdout)


def _pick_parameter_region(inspect_json: Dict) -> Tuple[int, int, Dict]:
    first_nonempty = None
    for pkg in inspect_json.get("packages", []):
        for exe in pkg.get("executables", []):
            preg = exe.get("parameter_region")
            if not preg:
                continue
            size = int(preg.get("size", 0))
            if size <= 0:
                continue
            if first_nonempty is None:
                first_nonempty = (int(preg["start"]), int(preg["end"]), exe)
            if exe.get("type_name") == "PARAMETER_CACHING":
                return int(preg["start"]), int(preg["end"]), exe
    if first_nonempty is None:
        raise RuntimeError("no non-empty parameter_region found")
    return first_nonempty


def dense_256_param_offset(row: int, col: int) -> int:
    """Recovered mapping for Dense(256x256) template payload.

    Offset formula validated across multi-run single-hot probes:
      offset = (col//64)*16384 + (row//64)*4096 + ((row%64)//4)*256 + (col%64)*4 + (row%4)
    """

    if not (0 <= row < 256 and 0 <= col < 256):
        raise ValueError(f"row/col out of range for 256x256: row={row} col={col}")
    return (col // 64) * 16384 + (row // 64) * 4096 + ((row % 64) // 4) * 256 + (col % 64) * 4 + (row % 4)


def _iter_nonzero(mode: str, hot_row: int, hot_col: int) -> Iterable[Tuple[int, int]]:
    if mode == "identity":
        for i in range(256):
            yield i, i
        return
    if mode == "zero":
        return
    if mode == "single_hot":
        if not (0 <= hot_row < 256 and 0 <= hot_col < 256):
            raise ValueError(f"--hot-row/--hot-col must be in [0,255], got ({hot_row},{hot_col})")
        yield hot_row, hot_col
        return
    if mode == "shift_plus1":
        # output[j] = input[(j + 1) % 256]
        for col in range(256):
            row = (col + 1) % 256
            yield row, col
        return
    if mode == "shift_minus1":
        # output[j] = input[(j - 1) % 256]
        for col in range(256):
            row = (col - 1) % 256
            yield row, col
        return
    raise ValueError(f"unsupported mode: {mode}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Patch Dense(256x256) template parameter payload with structured matrix modes.",
    )
    p.add_argument("input", help="Input compiled *_edgetpu.tflite template.")
    p.add_argument("--output", "-o", required=True, help="Output patched model path.")
    p.add_argument(
        "--mode",
        choices=["identity", "zero", "single_hot", "shift_plus1", "shift_minus1"],
        default="identity",
    )
    p.add_argument("--hot-row", type=int, default=0, help="Used when --mode single_hot.")
    p.add_argument("--hot-col", type=int, default=0, help="Used when --mode single_hot.")
    p.add_argument("--zero-byte", type=int, default=128, help="Byte value for zero weights.")
    p.add_argument("--one-byte", type=int, default=255, help="Byte value for non-zero weights.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--metadata-out", help="Optional metadata JSON output path.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"input model not found: {in_path}")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {out_path} (pass --overwrite)")
    if not (0 <= args.zero_byte <= 255 and 0 <= args.one_byte <= 255):
        raise ValueError("--zero-byte and --one-byte must be in [0,255]")

    inspect_json = _inspect_model(in_path, repo_root)
    start, end, exe = _pick_parameter_region(inspect_json)
    size = end - start
    if size != 256 * 256:
        raise RuntimeError(
            f"expected parameter payload size 65536 for Dense(256x256), got {size} "
            f"(exe={exe.get('type_name')} index={exe.get('index')})"
        )

    payload = bytearray([args.zero_byte] * size)
    nonzero_pairs = list(_iter_nonzero(args.mode, args.hot_row, args.hot_col))
    for row, col in nonzero_pairs:
        payload[dense_256_param_offset(row, col)] = args.one_byte

    blob = bytearray(in_path.read_bytes())
    before_sha = _sha256(blob[start:end])
    blob[start:end] = payload
    after_sha = _sha256(blob[start:end])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)

    metadata = {
        "tool": "dense_template_matrix_patch.py",
        "generated_at_utc": dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat(),
        "input": str(in_path),
        "output": str(out_path),
        "mode": args.mode,
        "hot_row": args.hot_row,
        "hot_col": args.hot_col,
        "zero_byte": args.zero_byte,
        "one_byte": args.one_byte,
        "parameter_region": {"start": start, "end": end, "size": size},
        "target_executable": {
            "index": exe.get("index"),
            "type_name": exe.get("type_name"),
            "type_value": exe.get("type_value"),
        },
        "nonzero_count": len(nonzero_pairs),
        "nonzero_preview": [[r, c, dense_256_param_offset(r, c)] for r, c in nonzero_pairs[:16]],
        "payload_sha256_before": before_sha,
        "payload_sha256_after": after_sha,
    }

    if args.metadata_out:
        meta_path = Path(args.metadata_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote patched model: {out_path}")
    print(f"  mode={args.mode} nonzero_count={len(nonzero_pairs)}")
    print(f"  parameter_region=[{start},{end}) size={size}")
    print(f"  payload_sha256_before={before_sha}")
    print(f"  payload_sha256_after={after_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
