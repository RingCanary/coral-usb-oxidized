#!/usr/bin/env python3
"""Compatibility shim for archived tool 'patch_tflite_dense_weight_pattern.py'."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parent / "archive" / "patch_tflite_dense_weight_pattern.py"
    if not target.exists():
        print(f"error: missing archived tool: {target}", file=sys.stderr)
        return 2
    print(
        "note: 'patch_tflite_dense_weight_pattern.py' is archived; forwarding to tools/archive/patch_tflite_dense_weight_pattern.py",
        file=sys.stderr,
    )
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
