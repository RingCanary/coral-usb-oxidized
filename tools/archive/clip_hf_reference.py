#!/usr/bin/env python3
"""Generate a CLIP image embedding reference with HuggingFace Transformers.

Expected input is raw f32 CHW tensor (3x224x224) in little-endian format.
"""

import argparse
import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection


def read_image_f32le(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f4")
    expected = 3 * 224 * 224
    if arr.size != expected:
        raise ValueError(f"expected {expected} float32 values, got {arr.size}")
    return arr.reshape(1, 3, 224, 224)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="HF model id or local directory")
    parser.add_argument("image_f32le", help="Input image tensor (f32le, CHW 3x224x224)")
    parser.add_argument("out_f32le", help="Output embedding path (f32le, len 512)")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize output embedding before writing",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download from hub",
    )
    args = parser.parse_args()

    pixel_values = torch.from_numpy(read_image_f32le(args.image_f32le))
    model = CLIPVisionModelWithProjection.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    model.eval()

    with torch.no_grad():
        out = model(pixel_values=pixel_values)
        embed = out.image_embeds[0].detach().cpu().numpy().astype(np.float32)

    if args.normalize:
        norm = np.linalg.norm(embed)
        if norm > 0:
            embed = embed / norm

    embed.astype("<f4").tofile(args.out_f32le)
    print(f"wrote {args.out_f32le} ({embed.shape[0]} floats)")


if __name__ == "__main__":
    main()
