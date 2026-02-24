#!/usr/bin/env python3
"""Extract bulk payload signatures from usbmon text logs.

Focuses on short payload prefixes (default first 2 dwords) from:
- submit events: `S Bo:... = <hex words...>`
- complete events with payload preview: `C Bi:... = <hex words...>`
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

HEX_WORD_RE = re.compile(r"^[0-9a-fA-F]{8}$")

HEADER_TAG_NAMES = {
    0: "Instructions",
    1: "InputActivations",
    2: "Parameters",
    3: "OutputActivations",
    4: "Interrupt0",
    5: "Interrupt1",
    6: "Interrupt2",
    7: "Interrupt3",
}


@dataclass
class Entry:
    ts: int
    event: str
    transfer: str
    bus: str
    dev: str
    ep: str
    status: str
    size: Optional[int]
    tokens: List[str]
    payload_words: List[str]


def parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


def parse_payload_words(parts: List[str]) -> List[str]:
    if "=" not in parts:
        return []
    idx = parts.index("=")
    out: List[str] = []
    for tok in parts[idx + 1 :]:
        t = tok.strip().lower()
        if HEX_WORD_RE.match(t):
            out.append(t)
    return out


def parse_line(line: str) -> Optional[Entry]:
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    ts = parse_int(parts[1])
    if ts is None:
        return None
    xfer = parts[3].split(":")
    if len(xfer) != 4:
        return None
    size = parse_int(parts[5]) if len(parts) > 5 else None
    return Entry(
        ts=ts,
        event=parts[2],
        transfer=xfer[0],
        bus=xfer[1],
        dev=xfer[2],
        ep=xfer[3],
        status=parts[4],
        size=size,
        tokens=parts,
        payload_words=parse_payload_words(parts),
    )


def load_entries(path: Path) -> List[Entry]:
    out: List[Entry] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            e = parse_line(line)
            if e is not None:
                out.append(e)
    return out


def dominant_device(entries: Iterable[Entry], bus: Optional[str]) -> Optional[str]:
    c = Counter()
    for e in entries:
        if bus is not None and e.bus != bus:
            continue
        c[e.dev] += 1
    if not c:
        return None
    return c.most_common(1)[0][0]


def detect_loop_window(entries: List[Entry], bo_b: int, bi_out: int) -> Tuple[Optional[int], Optional[int]]:
    first_bo_b = None
    last_bi_out = None
    for e in entries:
        if e.event == "C" and e.status == "0" and e.transfer == "Bo" and e.size == bo_b:
            if first_bo_b is None:
                first_bo_b = e.ts
    for e in entries:
        if e.event == "C" and e.status == "0" and e.transfer == "Bi" and e.size == bi_out:
            last_bi_out = e.ts
    return first_bo_b, last_bi_out


def phase_for_ts(ts: int, first_bo_b: Optional[int], last_bi_out: Optional[int]) -> str:
    if first_bo_b is None or last_bi_out is None:
        return "setup_only"
    if ts < first_bo_b:
        return "pre_loop"
    if ts <= last_bi_out:
        return "loop"
    return "post_loop"


def signature(words: List[str], prefix_words: int) -> str:
    if not words:
        return "(no-payload)"
    return " ".join(words[:prefix_words])


def decode_ml_header_from_sig(sig: str) -> Optional[Dict[str, object]]:
    parts = sig.split()
    if len(parts) < 2:
        return None
    try:
        w0 = bytes.fromhex(parts[0])
        w1 = bytes.fromhex(parts[1])
    except ValueError:
        return None
    if len(w0) != 4 or len(w1) != 4:
        return None
    length = int.from_bytes(w0, byteorder="little", signed=False)
    tag = w1[0]
    return {
        "payload_length": length,
        "tag": tag,
        "tag_name": HEADER_TAG_NAMES.get(tag, f"UnknownTag{tag}"),
    }


def analyze_log(
    path: Path,
    bus: Optional[str],
    device: Optional[str],
    bo_b: int,
    bi_out: int,
    prefix_words: int,
) -> Dict[str, object]:
    entries_all = load_entries(path)
    if bus is not None:
        entries_bus = [e for e in entries_all if e.bus == bus]
    else:
        entries_bus = entries_all
    dev = device or dominant_device(entries_bus, bus)
    if dev is None:
        return {"log_path": str(path), "error": "no matching bus/device entries"}
    entries = [e for e in entries_bus if e.dev == dev]
    if not entries:
        return {"log_path": str(path), "bus": bus, "device": dev, "error": "no entries for device"}

    first_bo_b, last_bi_out = detect_loop_window(entries, bo_b=bo_b, bi_out=bi_out)

    submit_by_key = Counter()
    submit_by_key_phase = Counter()
    complete_by_key = Counter()
    complete_by_key_phase = Counter()

    submit_sizes = Counter()
    complete_sizes = Counter()

    for e in entries:
        if e.transfer not in {"Bo", "Bi"}:
            continue
        if e.size is None:
            continue
        phase = phase_for_ts(e.ts, first_bo_b=first_bo_b, last_bi_out=last_bi_out)

        if e.event == "S":
            submit_sizes[(e.transfer, e.size)] += 1
            if e.payload_words:
                sig = signature(e.payload_words, prefix_words=prefix_words)
                submit_by_key[(e.transfer, e.size, sig)] += 1
                submit_by_key_phase[(e.transfer, e.size, sig, phase)] += 1
        elif e.event == "C" and e.status == "0":
            complete_sizes[(e.transfer, e.size)] += 1
            if e.payload_words:
                sig = signature(e.payload_words, prefix_words=prefix_words)
                complete_by_key[(e.transfer, e.size, sig)] += 1
                complete_by_key_phase[(e.transfer, e.size, sig, phase)] += 1

    return {
        "log_path": str(path),
        "bus": bus,
        "device": dev,
        "line_count": len(entries),
        "loop_window": {"first_bo_b_ts": first_bo_b, "last_bi_out_ts": last_bi_out},
        "prefix_words": prefix_words,
        "submit_sizes": {f"{k[0]}::{k[1]}": v for k, v in sorted(submit_sizes.items())},
        "complete_sizes": {f"{k[0]}::{k[1]}": v for k, v in sorted(complete_sizes.items())},
        "submit_signatures": {
            f"{k[0]}::{k[1]}::{k[2]}": v
            for k, v in sorted(submit_by_key.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "submit_signatures_phase": {
            f"{k[0]}::{k[1]}::{k[2]}::{k[3]}": v
            for k, v in sorted(submit_by_key_phase.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "complete_signatures": {
            f"{k[0]}::{k[1]}::{k[2]}": v
            for k, v in sorted(complete_by_key.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "complete_signatures_phase": {
            f"{k[0]}::{k[1]}::{k[2]}::{k[3]}": v
            for k, v in sorted(complete_by_key_phase.items(), key=lambda kv: (-kv[1], kv[0]))
        },
    }


def format_phase_counts(data: Dict[str, int], transfer: str, size: int, sig: str) -> str:
    parts: List[str] = []
    for phase in ("setup_only", "pre_loop", "loop", "post_loop"):
        key = f"{transfer}::{size}::{sig}::{phase}"
        v = data.get(key, 0)
        if v:
            parts.append(f"{phase}:{v}")
    return ",".join(parts) if parts else "-"


def render_report_text(data: Dict[str, object], top: int) -> str:
    if "error" in data:
        return f"ERROR: {data['error']}"

    lines: List[str] = []
    lines.append(f"log={data['log_path']}")
    lines.append(f"bus={data['bus'] or '*'} device={data['device']} lines={data['line_count']}")
    lw = data["loop_window"]
    lines.append(
        "loop_window first_bo_b_ts={} last_bi_out_ts={}".format(
            lw["first_bo_b_ts"], lw["last_bi_out_ts"]
        )
    )
    lines.append("submit_sizes=" + json.dumps(data["submit_sizes"], sort_keys=True))
    lines.append("complete_sizes=" + json.dumps(data["complete_sizes"], sort_keys=True))

    lines.append("top_submit_signatures:")
    submit_items = []
    for key, count in data["submit_signatures"].items():
        transfer, size_s, sig = key.split("::", 2)
        submit_items.append((count, transfer, int(size_s), sig))
    submit_items.sort(reverse=True)
    for count, transfer, size, sig in submit_items[:top]:
        phases = format_phase_counts(data["submit_signatures_phase"], transfer, size, sig)
        extra = ""
        if transfer == "Bo" and size == 8:
            decoded = decode_ml_header_from_sig(sig)
            if decoded is not None:
                extra = " header_len={} tag={}({})".format(
                    decoded["payload_length"], decoded["tag"], decoded["tag_name"]
                )
        lines.append(f"  {transfer} size={size} sig={sig} count={count} phases={phases}{extra}")

    lines.append("top_complete_signatures:")
    complete_items = []
    for key, count in data["complete_signatures"].items():
        transfer, size_s, sig = key.split("::", 2)
        complete_items.append((count, transfer, int(size_s), sig))
    complete_items.sort(reverse=True)
    for count, transfer, size, sig in complete_items[:top]:
        phases = format_phase_counts(data["complete_signatures_phase"], transfer, size, sig)
        lines.append(f"  {transfer} size={size} sig={sig} count={count} phases={phases}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract bulk payload signatures from usbmon logs.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--bus")
    parser.add_argument("--device")
    parser.add_argument("--bo-b", type=int, default=150_528)
    parser.add_argument("--bi-out", type=int, default=1_008)
    parser.add_argument("--prefix-words", type=int, default=2, help="Words to keep in signature prefix.")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    data = analyze_log(
        path=args.log,
        bus=args.bus,
        device=args.device,
        bo_b=args.bo_b,
        bi_out=args.bi_out,
        prefix_words=args.prefix_words,
    )
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(render_report_text(data, top=args.top))
    return 0 if "error" not in data else 1


if __name__ == "__main__":
    raise SystemExit(main())
