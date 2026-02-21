#!/usr/bin/env python3
"""Generate a single-layer Dense model and export fully-int8 TFLite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator, List

import numpy as np
import tensorflow as tf


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_kernel(
    init_mode: str,
    input_dim: int,
    output_dim: int,
    seed: int,
    diag_scale: float,
    hot_row: int,
    hot_col: int,
    hot_value: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kernel = np.zeros((input_dim, output_dim), dtype=np.float32)

    if init_mode == "identity":
        for i in range(min(input_dim, output_dim)):
            kernel[i, i] = diag_scale
        return kernel
    if init_mode == "permutation":
        perm = rng.permutation(input_dim)
        for out_idx in range(output_dim):
            src_idx = int(perm[out_idx % input_dim])
            kernel[src_idx, out_idx] = diag_scale
        return kernel
    if init_mode == "ones":
        kernel.fill(diag_scale)
        return kernel
    if init_mode == "zero":
        return kernel
    if init_mode == "single_hot":
        if not (0 <= hot_row < input_dim):
            raise ValueError(f"--hot-row out of range: {hot_row} (input_dim={input_dim})")
        if not (0 <= hot_col < output_dim):
            raise ValueError(f"--hot-col out of range: {hot_col} (output_dim={output_dim})")
        kernel[hot_row, hot_col] = hot_value
        return kernel
    if init_mode == "random_uniform":
        return rng.uniform(-1.0, 1.0, size=(input_dim, output_dim)).astype(np.float32)
    raise ValueError(f"unsupported init_mode: {init_mode}")


def _representative_dataset(
    input_dim: int,
    samples: int,
    value_range: float,
    seed: int,
) -> Iterator[List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        batch = rng.uniform(
            -value_range,
            value_range,
            size=(1, input_dim),
        ).astype(np.float32)
        yield [batch]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate single-layer Dense INT8 TFLite model for EdgeTPU template experiments.",
    )
    p.add_argument("--output", required=True, help="Output .tflite path.")
    p.add_argument("--metadata-out", help="Optional JSON metadata output path.")
    p.add_argument("--input-dim", type=int, default=256)
    p.add_argument("--output-dim", type=int, default=256)
    p.add_argument(
        "--init-mode",
        choices=["identity", "permutation", "ones", "zero", "single_hot", "random_uniform"],
        default="identity",
        help="Kernel initialization mode.",
    )
    p.add_argument(
        "--diag-scale",
        type=float,
        default=1.0,
        help="Diagonal/fill scale for identity/permutation/ones init.",
    )
    p.add_argument("--use-bias", action="store_true", help="Enable Dense bias.")
    p.add_argument("--hot-row", type=int, default=0, help="Row index for --init-mode single_hot.")
    p.add_argument("--hot-col", type=int, default=0, help="Column index for --init-mode single_hot.")
    p.add_argument("--hot-value", type=float, default=1.0, help="Value for --init-mode single_hot.")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed.")
    p.add_argument("--rep-samples", type=int, default=256)
    p.add_argument("--rep-range", type=float, default=1.0)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(
                input_shape=(args.input_dim,),
                batch_size=1,
                dtype=tf.float32,
                name="input",
            ),
            tf.keras.layers.Dense(
                args.output_dim,
                activation=None,
                use_bias=args.use_bias,
                name="dense",
            ),
        ],
        name=f"dense_{args.input_dim}x{args.output_dim}",
    )

    # Materialize variables before setting deterministic weights.
    _ = model(np.zeros((1, args.input_dim), dtype=np.float32))

    kernel = _make_kernel(
        init_mode=args.init_mode,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seed=args.seed,
        diag_scale=args.diag_scale,
        hot_row=args.hot_row,
        hot_col=args.hot_col,
        hot_value=args.hot_value,
    )
    dense_layer = model.layers[-1]
    if args.use_bias:
        bias = np.zeros((args.output_dim,), dtype=np.float32)
        dense_layer.set_weights([kernel, bias])
    else:
        dense_layer.set_weights([kernel])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(
        input_dim=args.input_dim,
        samples=args.rep_samples,
        value_range=args.rep_range,
        seed=args.seed + 1,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    metadata = {
        "tool": "generate_dense_quant_tflite.py",
        "model_name": model.name,
        "output_path": str(output_path),
        "output_sha256": _sha256_bytes(tflite_model),
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "init_mode": args.init_mode,
        "diag_scale": args.diag_scale,
        "hot_row": args.hot_row,
        "hot_col": args.hot_col,
        "hot_value": args.hot_value,
        "use_bias": args.use_bias,
        "seed": args.seed,
        "rep_samples": args.rep_samples,
        "rep_range": args.rep_range,
        "kernel_shape": list(kernel.shape),
        "kernel_sha256": _sha256_bytes(kernel.tobytes()),
        "kernel_min": float(kernel.min()),
        "kernel_max": float(kernel.max()),
        "input_tensor": {
            "name": input_details.get("name"),
            "shape": input_details.get("shape", []).tolist(),
            "dtype": str(input_details.get("dtype")),
            "quantization": list(input_details.get("quantization", ())),
        },
        "output_tensor": {
            "name": output_details.get("name"),
            "shape": output_details.get("shape", []).tolist(),
            "dtype": str(output_details.get("dtype")),
            "quantization": list(output_details.get("quantization", ())),
        },
    }

    print(
        f"Wrote quantized model: {output_path}\n"
        f"  sha256: {metadata['output_sha256']}\n"
        f"  input shape: {metadata['input_tensor']['shape']} int8\n"
        f"  output shape: {metadata['output_tensor']['shape']} int8"
    )

    if args.metadata_out:
        meta_path = Path(args.metadata_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metadata: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
