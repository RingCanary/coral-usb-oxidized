#!/usr/bin/env python3
"""Patch 1x1 Conv2D quantized weight tensor bytes in a TFLite model to a deterministic pattern.

This performs an in-place patch on the selected constant tensor buffer and can also
write the flattened row-major `[in_channel, out_channel]` bytes used for patching.
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
        description="Patch 1x1 Conv2D quantized weight bytes to a deterministic index_mod pattern.",
    )
    p.add_argument("--input", required=True, type=Path, help="Input quantized .tflite path")
    p.add_argument("--output", required=True, type=Path, help="Output patched .tflite path")
    p.add_argument("--in-channels", type=int, required=True)
    p.add_argument("--out-channels", type=int, required=True)
    p.add_argument("--subgraph-index", type=int, default=0)
    p.add_argument("--modulus", type=int, default=251, help="Pattern modulus (default: 251)")
    p.add_argument(
        "--signed-reinterpret",
        action="store_true",
        help="Use ((i %% modulus) - 128) rem_euclid 256 instead of raw i %% modulus.",
    )
    p.add_argument(
        "--tensor-name-contains",
        default="",
        help="Optional preferred substring for selecting weight tensor name",
    )
    p.add_argument(
        "--row-major-out",
        type=Path,
        help="Optional output path for the flattened row-major patched bytes",
    )
    p.add_argument("--metadata-out", type=Path, help="Optional JSON metadata output path")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.modulus <= 0 or args.modulus > 256:
        raise ValueError(f"--modulus must be in [1,256], got {args.modulus}")

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

    expected_len = args.in_channels * args.out_channels
    accepted_shapes = [
        [1, 1, args.in_channels, args.out_channels],
        [args.out_channels, 1, 1, args.in_channels],
    ]

    candidates = []
    for tidx in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(tidx)
        shape = [t.Shape(i) for i in range(t.ShapeLength())]
        if shape not in accepted_shapes:
            continue
        bidx = t.Buffer()
        b = model.Buffers(bidx)
        dlen = b.DataLength()
        if dlen != expected_len:
            continue
        name = t.Name().decode("utf-8", errors="replace") if t.Name() else ""
        candidates.append((tidx, name, bidx, dlen, b, shape))

    if not candidates:
        raise RuntimeError(
            f"no candidate 1x1 Conv2D weight tensor found with shapes {accepted_shapes} and buffer_size={expected_len}"
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

    tensor_index, tensor_name, buffer_index, data_len, buffer_obj, selected_shape = pick

    vec_field_off = buffer_obj._tab.Offset(4)
    if vec_field_off == 0:
        raise RuntimeError("selected buffer has no Data vector")
    vec_start = buffer_obj._tab.Vector(vec_field_off)
    vec_len = buffer_obj._tab.VectorLen(vec_field_off)
    if vec_len != data_len:
        raise RuntimeError(f"vector length mismatch: vec_len={vec_len} data_len={data_len}")

    old_preview = bytes(blob[vec_start : vec_start + 16])
    old_sha = _sha256_bytes(bytes(blob[vec_start : vec_start + vec_len]))

    row_major = bytearray(vec_len)
    for i in range(vec_len):
        v = i % args.modulus
        if args.signed_reinterpret:
            row_major[i] = ((v - 128) % 256)
        else:
            row_major[i] = v

    if selected_shape == [1, 1, args.in_channels, args.out_channels]:
        patched_buffer = bytes(row_major)
    elif selected_shape == [args.out_channels, 1, 1, args.in_channels]:
        patched = bytearray(vec_len)
        for oc in range(args.out_channels):
            for ic in range(args.in_channels):
                src_idx = ic * args.out_channels + oc
                dst_idx = oc * args.in_channels + ic
                patched[dst_idx] = row_major[src_idx]
        patched_buffer = bytes(patched)
    else:
        raise RuntimeError(f"unsupported selected_shape after candidate filtering: {selected_shape}")

    blob[vec_start : vec_start + vec_len] = patched_buffer

    new_preview = bytes(blob[vec_start : vec_start + 16])
    new_sha = _sha256_bytes(bytes(blob[vec_start : vec_start + vec_len]))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(blob))
    out_sha = _sha256_bytes(out.read_bytes())

    if args.row_major_out:
        args.row_major_out.parent.mkdir(parents=True, exist_ok=True)
        args.row_major_out.write_bytes(bytes(row_major))

    metadata = {
        "tool": "patch_tflite_conv1x1_weight_pattern.py",
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
            "signed_reinterpret": args.signed_reinterpret,
        },
        "selection": {
            "subgraph_index": args.subgraph_index,
            "tensor_index": tensor_index,
            "tensor_name": tensor_name,
            "buffer_index": buffer_index,
            "tensor_shape": selected_shape,
            "buffer_data_len": data_len,
            "buffer_vector_start": vec_start,
            "buffer_vector_len": vec_len,
            "candidate_count": len(candidates),
        },
        "row_major": {
            "path": str(args.row_major_out) if args.row_major_out else None,
            "sha256": _sha256_bytes(bytes(row_major)),
            "len": len(row_major),
            "preview_hex": bytes(row_major[:16]).hex(),
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
