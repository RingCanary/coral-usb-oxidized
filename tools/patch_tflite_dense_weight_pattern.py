#!/usr/bin/env python3
"""Patch Dense quantized weight tensor bytes in a TFLite model to a deterministic pattern.

This tool performs an in-place byte patch on the selected constant weight buffer
without rebuilding FlatBuffers, preserving file structure and offsets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Patch Dense quantized weight bytes to index_mod pattern (i %% modulus).",
    )
    p.add_argument("--input", required=True, type=Path, help="Input quantized .tflite path")
    p.add_argument("--output", required=True, type=Path, help="Output patched .tflite path")
    p.add_argument("--input-dim", type=int, required=True)
    p.add_argument("--output-dim", type=int, required=True)
    p.add_argument("--subgraph-index", type=int, default=0)
    p.add_argument("--modulus", type=int, default=251, help="Pattern modulus (default: 251)")
    p.add_argument(
        "--tensor-name-contains",
        default="",
        help="Optional preferred substring for selecting weight tensor name",
    )
    p.add_argument("--metadata-out", type=Path, help="Optional JSON metadata output path")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.modulus <= 0 or args.modulus > 256:
        raise ValueError(f"--modulus must be in [1,256], got {args.modulus}")

    # tensorflow import is intentionally local so help works without TF runtime.
    from tensorflow.lite.python import schema_py_generated as schema

    inp = args.input
    out = args.output

    blob = bytearray(inp.read_bytes())
    input_sha = _sha256_bytes(bytes(blob))

    model = schema.Model.GetRootAsModel(blob, 0)
    if not (0 <= args.subgraph_index < model.SubgraphsLength()):
        raise ValueError(
            f"--subgraph-index out of range: {args.subgraph_index} (subgraphs={model.SubgraphsLength()})"
        )
    subgraph = model.Subgraphs(args.subgraph_index)

    expected_len = args.input_dim * args.output_dim

    candidates = []
    for tidx in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(tidx)
        shape = [t.Shape(i) for i in range(t.ShapeLength())]
        if shape != [args.input_dim, args.output_dim]:
            continue
        bidx = t.Buffer()
        b = model.Buffers(bidx)
        dlen = b.DataLength()
        if dlen != expected_len:
            continue
        name = t.Name().decode("utf-8", errors="replace") if t.Name() else ""
        candidates.append((tidx, name, bidx, dlen, b))

    if not candidates:
        raise RuntimeError(
            f"no candidate weight tensor found with shape [{args.input_dim},{args.output_dim}] and buffer_size={expected_len}"
        )

    pick = None
    preferred = args.tensor_name_contains.strip()
    if preferred:
        for c in candidates:
            if preferred in c[1]:
                pick = c
                break
    if pick is None:
        for c in candidates:
            if c[1].startswith("tfl.pseudo_qconst"):
                pick = c
                break
    if pick is None:
        pick = candidates[0]

    tensor_index, tensor_name, buffer_index, data_len, buffer_obj = pick

    vec_field_off = buffer_obj._tab.Offset(4)
    if vec_field_off == 0:
        raise RuntimeError("selected buffer has no Data vector")
    vec_start = buffer_obj._tab.Vector(vec_field_off)
    vec_len = buffer_obj._tab.VectorLen(vec_field_off)
    if vec_len != data_len:
        raise RuntimeError(f"vector length mismatch: vec_len={vec_len} data_len={data_len}")

    old_preview = bytes(blob[vec_start : vec_start + 16])
    old_sha = _sha256_bytes(bytes(blob[vec_start : vec_start + vec_len]))

    for i in range(vec_len):
        blob[vec_start + i] = i % args.modulus

    new_preview = bytes(blob[vec_start : vec_start + 16])
    new_sha = _sha256_bytes(bytes(blob[vec_start : vec_start + vec_len]))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(blob))
    out_sha = _sha256_bytes(out.read_bytes())

    metadata = {
        "tool": "patch_tflite_dense_weight_pattern.py",
        "generated_utc": _utc_now(),
        "input": {
            "path": str(inp),
            "sha256": input_sha,
            "size": len(blob),
        },
        "output": {
            "path": str(out),
            "sha256": out_sha,
            "size": out.stat().st_size,
        },
        "pattern": {
            "mode": "index_mod",
            "modulus": args.modulus,
        },
        "selection": {
            "subgraph_index": args.subgraph_index,
            "tensor_index": tensor_index,
            "tensor_name": tensor_name,
            "buffer_index": buffer_index,
            "tensor_shape": [args.input_dim, args.output_dim],
            "buffer_data_len": data_len,
            "buffer_vector_start": vec_start,
            "buffer_vector_len": vec_len,
            "candidate_count": len(candidates),
        },
        "patch": {
            "old_param_sha256": old_sha,
            "new_param_sha256": new_sha,
            "old_preview_hex": old_preview.hex(),
            "new_preview_hex": new_preview.hex(),
        },
    }

    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
