#!/usr/bin/env python3
"""Generate a single-layer Conv2D model and export fully-int8 TFLite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator, List


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_kernel(
    init_mode: str,
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    seed: int,
    diag_scale: float,
    hot_y: int,
    hot_x: int,
    hot_in_channel: int,
    hot_out_channel: int,
    hot_value: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kernel = np.zeros(
        (kernel_size, kernel_size, in_channels, out_channels), dtype=np.float32
    )

    if init_mode == "zero":
        return kernel
    if init_mode == "ones":
        kernel.fill(diag_scale)
        return kernel
    if init_mode == "random_uniform":
        return rng.uniform(-1.0, 1.0, size=kernel.shape).astype(np.float32)
    if init_mode == "delta":
        center = kernel_size // 2
        for channel in range(min(in_channels, out_channels)):
            kernel[center, center, channel, channel] = diag_scale
        return kernel
    if init_mode == "single_hot":
        if not (0 <= hot_y < kernel_size):
            raise ValueError(f"--hot-y out of range: {hot_y} (kernel_size={kernel_size})")
        if not (0 <= hot_x < kernel_size):
            raise ValueError(f"--hot-x out of range: {hot_x} (kernel_size={kernel_size})")
        if not (0 <= hot_in_channel < in_channels):
            raise ValueError(
                f"--hot-in-channel out of range: {hot_in_channel} (in_channels={in_channels})"
            )
        if not (0 <= hot_out_channel < out_channels):
            raise ValueError(
                f"--hot-out-channel out of range: {hot_out_channel} (out_channels={out_channels})"
            )
        kernel[hot_y, hot_x, hot_in_channel, hot_out_channel] = hot_value
        return kernel

    raise ValueError(f"unsupported init_mode: {init_mode}")


def _representative_dataset(
    height: int,
    width: int,
    channels: int,
    samples: int,
    value_range: float,
    seed: int,
) -> Iterator[List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        batch = rng.uniform(
            -value_range,
            value_range,
            size=(1, height, width, channels),
        ).astype(np.float32)
        yield [batch]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate single-layer Conv2D INT8 TFLite model for EdgeTPU template experiments."
        ),
    )
    p.add_argument("--output", required=True, help="Output .tflite path.")
    p.add_argument("--metadata-out", help="Optional JSON metadata output path.")
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--out-channels", type=int, default=16)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--padding", choices=["same", "valid"], default="same")
    p.add_argument(
        "--init-mode",
        choices=["delta", "ones", "zero", "random_uniform", "single_hot"],
        default="delta",
        help="Kernel initialization mode.",
    )
    p.add_argument(
        "--diag-scale",
        type=float,
        default=1.0,
        help="Scale for delta/ones init modes.",
    )
    p.add_argument("--hot-y", type=int, default=0, help="Y index for --init-mode single_hot.")
    p.add_argument("--hot-x", type=int, default=0, help="X index for --init-mode single_hot.")
    p.add_argument(
        "--hot-in-channel",
        type=int,
        default=0,
        help="Input channel index for --init-mode single_hot.",
    )
    p.add_argument(
        "--hot-out-channel",
        type=int,
        default=0,
        help="Output channel index for --init-mode single_hot.",
    )
    p.add_argument(
        "--hot-value", type=float, default=1.0, help="Value for --init-mode single_hot."
    )
    p.add_argument("--use-bias", action="store_true", help="Enable Conv2D bias.")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed.")
    p.add_argument("--rep-samples", type=int, default=128)
    p.add_argument("--rep-range", type=float, default=1.0)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    # Keep dependency imports local so `--help` works without TensorFlow installed.
    global np, tf
    import numpy as np
    import tensorflow as tf

    if args.height <= 0 or args.width <= 0:
        raise ValueError("height/width must be > 0")
    if args.in_channels <= 0 or args.out_channels <= 0:
        raise ValueError("in/out channels must be > 0")
    if args.kernel_size <= 0 or args.stride <= 0:
        raise ValueError("kernel-size/stride must be > 0")

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
                filters=args.out_channels,
                kernel_size=(args.kernel_size, args.kernel_size),
                strides=(args.stride, args.stride),
                padding=args.padding,
                activation=None,
                use_bias=args.use_bias,
                name="conv2d",
            ),
        ],
        name=(
            f"conv2d_{args.height}x{args.width}x{args.in_channels}"
            f"_to_{args.out_channels}_k{args.kernel_size}_s{args.stride}_{args.padding}"
        ),
    )

    _ = model(
        np.zeros((1, args.height, args.width, args.in_channels), dtype=np.float32)
    )

    kernel = _make_kernel(
        init_mode=args.init_mode,
        kernel_size=args.kernel_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        seed=args.seed,
        diag_scale=args.diag_scale,
        hot_y=args.hot_y,
        hot_x=args.hot_x,
        hot_in_channel=args.hot_in_channel,
        hot_out_channel=args.hot_out_channel,
        hot_value=args.hot_value,
    )
    conv_layer = model.layers[-1]
    if args.use_bias:
        bias = np.zeros((args.out_channels,), dtype=np.float32)
        conv_layer.set_weights([kernel, bias])
    else:
        conv_layer.set_weights([kernel])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(
        height=args.height,
        width=args.width,
        channels=args.in_channels,
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
        "tool": "generate_conv2d_quant_tflite.py",
        "model_name": model.name,
        "output_path": str(output_path),
        "output_sha256": _sha256_bytes(tflite_model),
        "height": args.height,
        "width": args.width,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "kernel_size": args.kernel_size,
        "stride": args.stride,
        "padding": args.padding,
        "init_mode": args.init_mode,
        "diag_scale": args.diag_scale,
        "hot_y": args.hot_y,
        "hot_x": args.hot_x,
        "hot_in_channel": args.hot_in_channel,
        "hot_out_channel": args.hot_out_channel,
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
