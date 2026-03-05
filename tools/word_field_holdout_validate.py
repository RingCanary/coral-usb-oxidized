#!/usr/bin/env python3
"""Compatibility shim for the archived holdout validator.

The implementation was moved to `tools/archive/word_field_holdout_validate.py`
as part of a safe tool-prune pass.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parent / "archive" / "word_field_holdout_validate.py"
    if not target.exists():
        print(f"error: missing archived tool: {target}", file=sys.stderr)
        return 2
    print(
        "note: `tools/word_field_holdout_validate.py` is archived; forwarding to tools/archive/...",
        file=sys.stderr,
    )
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
