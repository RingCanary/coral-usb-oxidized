#!/usr/bin/env python3
"""Small utility helpers for USB debug workflows.

Commands:
- summarize-usbmon: count endpoint events from usbmon text logs.
- summarize-strace: count ioctl/read/write calls from strace logs.
- report: merge simple key/value summaries into markdown.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
from typing import Iterable

USBMON_EP = re.compile(r"\b(?:s|c)\s+\S+\s+\S+\s+\S+\s+(Bo|Bi|Co|Ci):\d+:(\d+):")
STRACE_CALL = re.compile(r"^(ioctl|read|write|openat|close)\(")


def read_lines(path: pathlib.Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\n")


def summarize_usbmon(path: pathlib.Path) -> str:
    endpoint_counts: collections.Counter[str] = collections.Counter()
    direction_counts: collections.Counter[str] = collections.Counter()

    for line in read_lines(path):
        match = USBMON_EP.search(line)
        if not match:
            continue
        direction, endpoint = match.groups()
        endpoint_counts[f"ep{endpoint}"] += 1
        direction_counts[direction] += 1

    rows = ["# usbmon summary", "", f"source: `{path}`", "", "## Direction counts"]
    rows.extend(f"- {name}: {count}" for name, count in sorted(direction_counts.items()))
    rows.append("")
    rows.append("## Endpoint counts")
    rows.extend(f"- {name}: {count}" for name, count in sorted(endpoint_counts.items()))
    return "\n".join(rows) + "\n"


def summarize_strace(path: pathlib.Path) -> str:
    call_counts: collections.Counter[str] = collections.Counter()
    for line in read_lines(path):
        match = STRACE_CALL.match(line)
        if match:
            call_counts[match.group(1)] += 1

    rows = ["# strace summary", "", f"source: `{path}`", "", "## syscall counts"]
    rows.extend(f"- {name}: {count}" for name, count in sorted(call_counts.items()))
    return "\n".join(rows) + "\n"


def report(files: list[pathlib.Path]) -> str:
    sections = ["# USB debug report", ""]
    for file in files:
        content = file.read_text(encoding="utf-8", errors="replace").strip()
        sections.append(f"## {file.name}")
        sections.append("")
        sections.append(content if content else "(empty)")
        sections.append("")
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    usbmon_p = sub.add_parser("summarize-usbmon", help="summarize usbmon text log")
    usbmon_p.add_argument("input", type=pathlib.Path)
    usbmon_p.add_argument("--out", type=pathlib.Path)

    strace_p = sub.add_parser("summarize-strace", help="summarize strace text log")
    strace_p.add_argument("input", type=pathlib.Path)
    strace_p.add_argument("--out", type=pathlib.Path)

    report_p = sub.add_parser("report", help="merge markdown/text snippets")
    report_p.add_argument("inputs", nargs="+", type=pathlib.Path)
    report_p.add_argument("--out", type=pathlib.Path)

    args = parser.parse_args()
    if args.cmd == "summarize-usbmon":
        output = summarize_usbmon(args.input)
    elif args.cmd == "summarize-strace":
        output = summarize_strace(args.input)
    else:
        output = report(args.inputs)

    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
