#!/usr/bin/env python3
"""Generate a small Conv2D->Dense model and export fully-int8 TFLite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator, List


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Conv2D->Dense INT8 TFLite model for multi-op EdgeTPU experiments.",
    )
    p.add_argument("--output", required=True, help="Output .tflite path.")
    p.add_argument("--metadata-out", help="Optional JSON metadata output path.")

    p.add_argument("--height", type=int, default=16)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--in-channels", type=int, default=16)

    p.add_argument("--conv-filters", type=int, default=64)
    p.add_argument("--conv-kernel-size", type=int, default=1)
    p.add_argument("--conv-stride", type=int, default=1)
    p.add_argument("--conv-padding", choices=["same", "valid"], default="same")
    p.add_argument(
        "--conv-init-mode",
        choices=["delta", "ones", "zero", "random_uniform"],
        default="delta",
    )
    p.add_argument("--conv-diag-scale", type=float, default=1.0)

    p.add_argument("--dense-units", type=int, default=256)
    p.add_argument(
        "--dense-init-mode",
        choices=["identity", "permutation", "ones", "zero", "random_uniform"],
        default="identity",
    )
    p.add_argument("--dense-diag-scale", type=float, default=1.0)

    p.add_argument("--use-bias", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rep-samples", type=int, default=128)
    p.add_argument("--rep-range", type=float, default=1.0)
    return p


def _make_conv_kernel(
    np,
    init_mode: str,
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    seed: int,
    diag_scale: float,
):
    rng = np.random.default_rng(seed)
    kernel = np.zeros((kernel_size, kernel_size, in_channels, out_channels), dtype=np.float32)

    if init_mode == "zero":
        return kernel
    if init_mode == "ones":
        kernel.fill(diag_scale)
        return kernel
    if init_mode == "random_uniform":
        return rng.uniform(-1.0, 1.0, size=kernel.shape).astype(np.float32)
    if init_mode == "delta":
        center = kernel_size // 2
        for c in range(min(in_channels, out_channels)):
            kernel[center, center, c, c] = diag_scale
        return kernel

    raise ValueError(f"unsupported conv init mode: {init_mode}")


def _make_dense_kernel(
    np,
    init_mode: str,
    input_dim: int,
    output_dim: int,
    seed: int,
    diag_scale: float,
):
    rng = np.random.default_rng(seed)
    kernel = np.zeros((input_dim, output_dim), dtype=np.float32)

    if init_mode == "zero":
        return kernel
    if init_mode == "ones":
        kernel.fill(diag_scale)
        return kernel
    if init_mode == "random_uniform":
        return rng.uniform(-1.0, 1.0, size=kernel.shape).astype(np.float32)
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

    raise ValueError(f"unsupported dense init mode: {init_mode}")


def _representative_dataset(np, height: int, width: int, channels: int, samples: int, value_range: float, seed: int) -> Iterator[List[object]]:
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        batch = rng.uniform(
            -value_range,
            value_range,
            size=(1, height, width, channels),
        ).astype(np.float32)
        yield [batch]


def main() -> int:
    args = _build_parser().parse_args()
    global np, tf
    import numpy as np
    import tensorflow as tf

    if args.height <= 0 or args.width <= 0:
        raise ValueError("height/width must be > 0")
    if args.in_channels <= 0 or args.conv_filters <= 0 or args.dense_units <= 0:
        raise ValueError("channels/filters/dense-units must be > 0")

    tf.keras.utils.set_random_seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(
                input_shape=(args.height, args.width, args.in_channels),
                batch_size=1,
                dtype=tf.float32,
                name="input",
            ),
            tf.keras.layers.Conv2D(
                filters=args.conv_filters,
                kernel_size=(args.conv_kernel_size, args.conv_kernel_size),
                strides=(args.conv_stride, args.conv_stride),
                padding=args.conv_padding,
                activation=None,
                use_bias=args.use_bias,
                name="conv2d",
            ),
            tf.keras.layers.GlobalAveragePooling2D(name="gap2d"),
            tf.keras.layers.Dense(
                args.dense_units,
                activation=None,
                use_bias=args.use_bias,
                name="dense",
            ),
        ],
        name=(
            f"conv_dense_{args.height}x{args.width}x{args.in_channels}"
            f"_conv{args.conv_filters}_k{args.conv_kernel_size}"
            f"_dense{args.dense_units}"
        ),
    )

    _ = model(np.zeros((1, args.height, args.width, args.in_channels), dtype=np.float32))

    conv_kernel = _make_conv_kernel(
        np=np,
        init_mode=args.conv_init_mode,
        kernel_size=args.conv_kernel_size,
        in_channels=args.in_channels,
        out_channels=args.conv_filters,
        seed=args.seed,
        diag_scale=args.conv_diag_scale,
    )
    dense_kernel = _make_dense_kernel(
        np=np,
        init_mode=args.dense_init_mode,
        input_dim=args.conv_filters,
        output_dim=args.dense_units,
        seed=args.seed + 1,
        diag_scale=args.dense_diag_scale,
    )

    conv_layer = model.get_layer("conv2d")
    dense_layer = model.get_layer("dense")

    if args.use_bias:
        conv_bias = np.zeros((args.conv_filters,), dtype=np.float32)
        dense_bias = np.zeros((args.dense_units,), dtype=np.float32)
        conv_layer.set_weights([conv_kernel, conv_bias])
        dense_layer.set_weights([dense_kernel, dense_bias])
    else:
        conv_layer.set_weights([conv_kernel])
        dense_layer.set_weights([dense_kernel])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(
        np=np,
        height=args.height,
        width=args.width,
        channels=args.in_channels,
        samples=args.rep_samples,
        value_range=args.rep_range,
        seed=args.seed + 2,
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
        "tool": "generate_dense_conv_quant_tflite.py",
        "model_name": model.name,
        "output_path": str(output_path),
        "output_sha256": _sha256_bytes(tflite_model),
        "height": args.height,
        "width": args.width,
        "in_channels": args.in_channels,
        "conv_filters": args.conv_filters,
        "conv_kernel_size": args.conv_kernel_size,
        "conv_stride": args.conv_stride,
        "conv_padding": args.conv_padding,
        "conv_init_mode": args.conv_init_mode,
        "conv_diag_scale": args.conv_diag_scale,
        "dense_units": args.dense_units,
        "dense_init_mode": args.dense_init_mode,
        "dense_diag_scale": args.dense_diag_scale,
        "use_bias": args.use_bias,
        "seed": args.seed,
        "rep_samples": args.rep_samples,
        "rep_range": args.rep_range,
        "conv_kernel_shape": list(conv_kernel.shape),
        "conv_kernel_sha256": _sha256_bytes(conv_kernel.tobytes()),
        "dense_kernel_shape": list(dense_kernel.shape),
        "dense_kernel_sha256": _sha256_bytes(dense_kernel.tobytes()),
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
