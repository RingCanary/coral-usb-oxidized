#!/usr/bin/env python3
"""Dump 1x1 Conv2D quantized weight tensor bytes from a TFLite model.

Outputs the current proven stored tensor order used by tested TFLite Conv2D 1x1 models.
For the common converted form `[out, 1, 1, in]`, the dumped byte stream is therefore
flattened in `[out_channel, in_channel]` order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump 1x1 Conv2D quantized weight bytes from a TFLite model.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--in-channels", type=int, required=True)
    p.add_argument("--out-channels", type=int, required=True)
    p.add_argument("--subgraph-index", type=int, default=0)
    p.add_argument("--tensor-name-contains", default="")
    p.add_argument("--metadata-out", type=Path)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    from tensorflow.lite.python import schema_py_generated as schema

    blob = args.input.read_bytes()
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
    stored = bytes(blob[vec_start : vec_start + vec_len])

    if selected_shape not in accepted_shapes:
        raise RuntimeError(f"unsupported selected_shape after candidate filtering: {selected_shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(stored)

    metadata = {
        "tool": "dump_tflite_conv1x1_weights.py",
        "input": str(args.input),
        "output": str(args.output),
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "tensor_index": tensor_index,
        "tensor_name": tensor_name,
        "buffer_index": buffer_index,
        "stored_shape": selected_shape,
        "stored_sha256": _sha256_bytes(stored),
        "len": len(stored),
        "preview_hex": stored[:16].hex(),
    }
    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
