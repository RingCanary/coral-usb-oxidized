#!/usr/bin/env python3
"""Generate replay instruction patch spec from good vs replay payload dumps.

Patch spec format (consumed by rusb_serialized_exec_replay --instruction-patch-spec):
  <payload_len> <offset> <byte_value>
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple


def parse_pair(raw: str) -> Tuple[int, Path, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"invalid --pair '{raw}' (expected LEN:GOOD_PATH:REPLAY_PATH)"
        )
    try:
        length = int(parts[0], 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid LEN in --pair '{raw}': {exc}") from exc
    if length <= 0:
        raise argparse.ArgumentTypeError(f"LEN must be > 0 in --pair '{raw}'")
    return length, Path(parts[1]), Path(parts[2])


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        type=parse_pair,
        help="LEN:GOOD_PATH:REPLAY_PATH (repeat for multiple payload lengths)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output patch spec path",
    )
    args = parser.parse_args()

    lines: List[str] = []
    total_rules = 0

    for payload_len, good_path, replay_path in args.pair:
        if not good_path.is_file():
            raise SystemExit(f"good payload file not found: {good_path}")
        if not replay_path.is_file():
            raise SystemExit(f"replay payload file not found: {replay_path}")

        good = good_path.read_bytes()
        replay = replay_path.read_bytes()

        if len(good) != payload_len:
            raise SystemExit(
                f"good payload length mismatch for {good_path}: expected {payload_len}, got {len(good)}"
            )
        if len(replay) != payload_len:
            raise SystemExit(
                f"replay payload length mismatch for {replay_path}: expected {payload_len}, got {len(replay)}"
            )

        changed = [idx for idx in range(payload_len) if good[idx] != replay[idx]]
        lines.append(
            f"# len={payload_len} good={good_path.name} sha={sha256_hex(good)} replay={replay_path.name} sha={sha256_hex(replay)} diff_bytes={len(changed)}"
        )
        for idx in changed:
            lines.append(f"{payload_len} {idx} 0x{good[idx]:02x}")
        lines.append("")
        total_rules += len(changed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n")

    print(f"Wrote patch spec: {args.out}")
    print(f"Pairs: {len(args.pair)}")
    print(f"Total rules: {total_rules}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
