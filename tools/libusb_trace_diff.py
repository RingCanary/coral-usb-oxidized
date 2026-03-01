#!/usr/bin/env python3
"""Diff OUT payload hashes from libusb interposer TSV logs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Row:
    ts_us: int
    seq: int
    api: str
    ep: str
    direction: str
    length: int
    sha256: str
    first16: str
    last16: str
    is_header: bool
    hdr_len: int
    hdr_tag: int
    stream_tag: int
    stream_rem_before: int
    stream_rem_after: int
    ret: int
    transferred: int


FIELDS = [
    "ts_us",
    "seq",
    "pid",
    "tid",
    "api",
    "ep",
    "dir",
    "len",
    "sha256",
    "first16",
    "last16",
    "is_header",
    "hdr_len",
    "hdr_tag",
    "stream_tag",
    "stream_rem_before",
    "stream_rem_after",
    "ret",
    "transferred",
    "timeout_ms",
]


def parse_log(path: Path) -> List[Row]:
    rows: List[Row] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < len(FIELDS):
            continue
        record = dict(zip(FIELDS, parts))
        rows.append(
            Row(
                ts_us=int(record["ts_us"]),
                seq=int(record["seq"]),
                api=record["api"],
                ep=record["ep"],
                direction=record["dir"],
                length=int(record["len"]),
                sha256=record["sha256"],
                first16=record["first16"],
                last16=record["last16"],
                is_header=record["is_header"] == "1",
                hdr_len=int(record["hdr_len"]),
                hdr_tag=int(record["hdr_tag"]),
                stream_tag=int(record["stream_tag"]),
                stream_rem_before=int(record["stream_rem_before"]),
                stream_rem_after=int(record["stream_rem_after"]),
                ret=int(record["ret"]),
                transferred=int(record["transferred"]),
            )
        )
    return rows


def out_rows(rows: Iterable[Row]) -> List[Row]:
    return [r for r in rows if r.direction == "O"]


def payload_rows(rows: Iterable[Row]) -> List[Row]:
    return [r for r in rows if not r.is_header]


def header_rows(rows: Iterable[Row]) -> List[Row]:
    return [r for r in rows if r.is_header]


def summarize(rows: List[Row], label: str) -> None:
    print(f"\n[{label}] out_rows={len(rows)}")
    lengths = Counter(r.length for r in rows)
    top = ", ".join(f"{k}:{v}" for k, v in lengths.most_common(12))
    print(f"  len_counts: {top}")
    headers = [r for r in rows if r.is_header]
    if headers:
        print("  headers:")
        for r in headers[:12]:
            print(
                f"    seq={r.seq} hdr_len={r.hdr_len} hdr_tag={r.hdr_tag} sha={r.sha256[:16]}..."
            )


def compare_first(rows_a: List[Row], rows_b: List[Row], length: int, stream_tag: int | None = None) -> None:
    def pick(rows: List[Row]) -> List[Row]:
        base = [r for r in payload_rows(rows) if r.length == length]
        if stream_tag is not None:
            base = [r for r in base if r.stream_tag == stream_tag]
        return base

    a = pick(rows_a)
    b = pick(rows_b)
    tag_label = f" stream_tag={stream_tag}" if stream_tag is not None else ""
    print(f"\n[compare len={length}{tag_label}] a={len(a)} b={len(b)}")
    if not a or not b:
        return
    x, y = a[0], b[0]
    same = x.sha256 == y.sha256
    print(f"  first_sha_same={same}")
    print(f"  a: seq={x.seq} api={x.api} sha={x.sha256} first16={x.first16} last16={x.last16}")
    print(f"  b: seq={y.seq} api={y.api} sha={y.sha256} first16={y.first16} last16={y.last16}")


def compare_multiset(
    rows_a: List[Row],
    rows_b: List[Row],
    length: int,
    stream_tag: int | None = None,
    *,
    include_headers: bool = False,
) -> None:
    def hashes(rows: List[Row]) -> Counter:
        source = rows if include_headers else payload_rows(rows)
        base = [r for r in source if r.length == length]
        if stream_tag is not None:
            base = [r for r in base if r.stream_tag == stream_tag]
        return Counter(r.sha256 for r in base)

    ha = hashes(rows_a)
    hb = hashes(rows_b)
    tag_label = f" stream_tag={stream_tag}" if stream_tag is not None else ""
    kind = "all" if include_headers else "payload"
    same = ha == hb
    print(
        f"\n[multiset kind={kind} len={length}{tag_label}] equal={same} a_unique={len(ha)} b_unique={len(hb)}"
    )
    if not same:
        only_a = ha - hb
        only_b = hb - ha
        if only_a:
            print("  only_a:")
            for digest, count in only_a.items():
                print(f"    {digest} x{count}")
        if only_b:
            print("  only_b:")
            for digest, count in only_b.items():
                print(f"    {digest} x{count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("good", type=Path, help="Known-good interposer TSV log")
    parser.add_argument("replay", type=Path, help="Replay interposer TSV log")
    args = parser.parse_args()

    good = out_rows(parse_log(args.good))
    replay = out_rows(parse_log(args.replay))

    summarize(good, "good")
    summarize(replay, "replay")

    # Decisive targets requested: 2608 and 9872 instruction payloads.
    compare_first(good, replay, 2608, stream_tag=0)
    compare_first(good, replay, 9872, stream_tag=0)

    # Also compare descriptor headers and 1 MiB param chunks for context.
    compare_multiset(good, replay, 8, include_headers=True)
    compare_multiset(good, replay, 1048576, stream_tag=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
