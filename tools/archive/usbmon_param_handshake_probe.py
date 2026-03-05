#!/usr/bin/env python3
"""Compare usbmon parameter-stream handshake behavior around a byte threshold.

This tool compares two usbmon text logs ("good" and "bad") and focuses on the
descriptor tag=2 parameter stream phase:
1. auto-select bus/device (unless overridden),
2. find Bo submit header (size=8) with descriptor tag=2,
3. accumulate subsequent Bo submit bytes (size>8) in that phase,
4. find the first payload write whose cumulative bytes cross threshold,
5. summarize tuple counts in the full phase and near-anchor window,
6. print side-by-side text diffs (or JSON via --json).
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


@dataclass
class Entry:
    line_no: int
    raw: str
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


@dataclass
class PayloadWrite:
    entry_idx: int
    line_no: int
    ts: int
    size: int
    cumulative_bytes: int


@dataclass
class ParamPhase:
    phase_index: int
    start_idx: int
    end_idx: int
    start_line_no: int
    start_ts: int
    header_payload_length: int
    payload_writes: List[PayloadWrite]


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


def parse_line(line: str, line_no: int) -> Optional[Entry]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    ts = parse_int(parts[1])
    if ts is None:
        return None
    xfer = parts[3].split(":")
    if len(xfer) != 4:
        return None
    size = parse_int(parts[5]) if len(parts) > 5 else None
    return Entry(
        line_no=line_no,
        raw=line.rstrip("\n"),
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
        for line_no, line in enumerate(handle, start=1):
            e = parse_line(line, line_no=line_no)
            if e is not None:
                out.append(e)
    return out


def decode_descriptor_header(words: List[str]) -> Optional[Dict[str, int]]:
    if len(words) < 2:
        return None
    try:
        w0 = bytes.fromhex(words[0])
        w1 = bytes.fromhex(words[1])
    except ValueError:
        return None
    if len(w0) != 4 or len(w1) != 4:
        return None
    payload_length = int.from_bytes(w0, byteorder="little", signed=False)
    tag = int(w1[0])
    return {"payload_length": payload_length, "tag": tag}


def descriptor_header_submit(entry: Entry) -> Optional[Dict[str, int]]:
    if entry.event != "S":
        return None
    if entry.transfer != "Bo":
        return None
    if entry.size != 8:
        return None
    return decode_descriptor_header(entry.payload_words)


def is_param_header_submit(entry: Entry) -> bool:
    header = descriptor_header_submit(entry)
    return header is not None and header["tag"] == 2


def select_bus_device(
    entries: Iterable[Entry], bus_override: Optional[str], device_override: Optional[str]
) -> Optional[Tuple[str, str]]:
    total = Counter()
    param_headers = Counter()
    for e in entries:
        if bus_override is not None and e.bus != bus_override:
            continue
        if device_override is not None and e.dev != device_override:
            continue
        key = (e.bus, e.dev)
        total[key] += 1
        if is_param_header_submit(e):
            param_headers[key] += 1
    if not total:
        return None
    ranked = sorted(
        total.keys(),
        key=lambda key: (param_headers[key], total[key], key[0], key[1]),
        reverse=True,
    )
    return ranked[0]


def detect_param_phases(entries: List[Entry]) -> List[ParamPhase]:
    phases: List[ParamPhase] = []
    open_phase: Optional[ParamPhase] = None
    phase_counter = 0
    cumulative = 0

    for idx, e in enumerate(entries):
        header = descriptor_header_submit(e)
        if header is not None:
            if open_phase is not None:
                open_phase.end_idx = idx - 1
                phases.append(open_phase)
                open_phase = None
                cumulative = 0
            if header["tag"] == 2:
                phase_counter += 1
                cumulative = 0
                open_phase = ParamPhase(
                    phase_index=phase_counter,
                    start_idx=idx,
                    end_idx=idx,
                    start_line_no=e.line_no,
                    start_ts=e.ts,
                    header_payload_length=header["payload_length"],
                    payload_writes=[],
                )
            continue

        if open_phase is None:
            continue
        open_phase.end_idx = idx
        if e.event == "S" and e.transfer == "Bo" and e.size is not None and e.size > 8:
            cumulative += e.size
            open_phase.payload_writes.append(
                PayloadWrite(
                    entry_idx=idx,
                    line_no=e.line_no,
                    ts=e.ts,
                    size=e.size,
                    cumulative_bytes=cumulative,
                )
            )

    if open_phase is not None:
        open_phase.end_idx = len(entries) - 1
        phases.append(open_phase)
    return phases


def pick_phase(phases: List[ParamPhase], threshold: int) -> Optional[ParamPhase]:
    if not phases:
        return None
    for phase in phases:
        for write in phase.payload_writes:
            if write.cumulative_bytes >= threshold:
                return phase
    return max(phases, key=lambda p: p.payload_writes[-1].cumulative_bytes if p.payload_writes else 0)


def find_anchor_write(
    phase: ParamPhase, threshold: int
) -> Tuple[Optional[PayloadWrite], Optional[PayloadWrite], bool]:
    anchor: Optional[PayloadWrite] = None
    for write in phase.payload_writes:
        if write.cumulative_bytes >= threshold:
            anchor = write
            break
    if phase.payload_writes:
        fallback = phase.payload_writes[-1]
    else:
        fallback = None
    reached = anchor is not None
    return anchor, fallback, reached


def control_tuple(entry: Entry) -> Optional[str]:
    if entry.event != "S":
        return None
    if entry.transfer not in {"Co", "Ci"}:
        return None
    if len(entry.tokens) < 10:
        return None
    if entry.tokens[4] != "s":
        return None
    bm = entry.tokens[5].lower()
    breq = entry.tokens[6].lower()
    wval = entry.tokens[7].lower()
    widx = entry.tokens[8].lower()
    wlen = entry.tokens[9].lower()
    return f"{entry.transfer}:{bm}:{breq}:{wval}:{widx}:{wlen}"


def summarize_tuples(entries: List[Entry]) -> Dict[str, Dict[str, int]]:
    transfer = Counter()
    status = Counter()
    transfer_status = Counter()
    control = Counter()

    for e in entries:
        transfer[e.transfer] += 1
        status[e.status] += 1
        transfer_status[f"{e.event}:{e.transfer}:{e.status}"] += 1
        ctl = control_tuple(e)
        if ctl is not None:
            control[ctl] += 1

    return {
        "transfer": dict(sorted(transfer.items())),
        "status": dict(sorted(status.items())),
        "transfer_status": dict(
            sorted(transfer_status.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "control_request": dict(sorted(control.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def counter_diff(
    a: Dict[str, int], b: Dict[str, int], top: Optional[int] = None
) -> List[Dict[str, int | str]]:
    keys = set(a.keys()) | set(b.keys())
    rows: List[Dict[str, int | str]] = []
    for key in keys:
        av = int(a.get(key, 0))
        bv = int(b.get(key, 0))
        rows.append({"tuple": key, "good": av, "bad": bv, "delta_bad_minus_good": bv - av})
    rows.sort(key=lambda row: (-abs(int(row["delta_bad_minus_good"])), str(row["tuple"])))
    if top is not None:
        return rows[:top]
    return rows


def only_in(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {key: int(a[key]) for key in sorted(a.keys()) if key not in b}


def format_context_entry(e: Entry) -> str:
    return f"L{e.line_no} {e.raw}"


def extract_context(
    entries: List[Entry], center_idx: int, context: int
) -> Tuple[Tuple[int, int], Dict[int, Dict[str, object]]]:
    lo = max(0, center_idx - context)
    hi = min(len(entries) - 1, center_idx + context)
    out: Dict[int, Dict[str, object]] = {}
    for idx in range(lo, hi + 1):
        rel = idx - center_idx
        out[rel] = {
            "line_no": entries[idx].line_no,
            "raw": entries[idx].raw,
            "render": format_context_entry(entries[idx]),
        }
    return (lo, hi), out


def analyze_one(
    label: str,
    path: Path,
    bus_override: Optional[str],
    device_override: Optional[str],
    threshold: int,
    context: int,
) -> Dict[str, object]:
    entries_all = load_entries(path)
    selected = select_bus_device(entries_all, bus_override=bus_override, device_override=device_override)
    if selected is None:
        return {
            "label": label,
            "log_path": str(path),
            "error": "no entries matched selected bus/device filter",
        }
    bus, device = selected
    entries = [e for e in entries_all if e.bus == bus and e.dev == device]
    if not entries:
        return {
            "label": label,
            "log_path": str(path),
            "bus": bus,
            "device": device,
            "error": "selected bus/device has no entries",
        }

    phases = detect_param_phases(entries)
    selected_phase = pick_phase(phases, threshold=threshold)
    if selected_phase is None:
        return {
            "label": label,
            "log_path": str(path),
            "bus": bus,
            "device": device,
            "line_count": len(entries),
            "param_phase_count": 0,
            "error": "no parameter phase start found (Bo submit header tag=2)",
        }

    anchor, fallback, reached = find_anchor_write(selected_phase, threshold=threshold)
    center_idx = (
        anchor.entry_idx
        if anchor is not None
        else (fallback.entry_idx if fallback is not None else selected_phase.start_idx)
    )
    _, context_map = extract_context(entries, center_idx=center_idx, context=context)

    phase_entries = entries[selected_phase.start_idx : selected_phase.end_idx + 1]
    near_entries = entries[
        max(0, center_idx - context) : min(len(entries), center_idx + context + 1)
    ]

    phase_summary = summarize_tuples(phase_entries)
    near_summary = summarize_tuples(near_entries)
    total_payload_bytes = (
        selected_phase.payload_writes[-1].cumulative_bytes if selected_phase.payload_writes else 0
    )

    return {
        "label": label,
        "log_path": str(path),
        "bus": bus,
        "device": device,
        "line_count": len(entries),
        "param_phase_count": len(phases),
        "selected_phase_index": selected_phase.phase_index,
        "phase_start": {
            "entry_index": selected_phase.start_idx,
            "line_no": selected_phase.start_line_no,
            "ts": selected_phase.start_ts,
            "header_payload_length": selected_phase.header_payload_length,
        },
        "phase_end": {
            "entry_index": selected_phase.end_idx,
            "line_no": entries[selected_phase.end_idx].line_no,
            "ts": entries[selected_phase.end_idx].ts,
        },
        "phase_payload_bytes_total": total_payload_bytes,
        "payload_write_count": len(selected_phase.payload_writes),
        "anchor_reached_threshold": reached,
        "anchor_write": None
        if anchor is None
        else {
            "entry_index": anchor.entry_idx,
            "line_no": anchor.line_no,
            "ts": anchor.ts,
            "size": anchor.size,
            "cumulative_bytes": anchor.cumulative_bytes,
        },
        "anchor_fallback": None
        if fallback is None
        else {
            "entry_index": fallback.entry_idx,
            "line_no": fallback.line_no,
            "ts": fallback.ts,
            "size": fallback.size,
            "cumulative_bytes": fallback.cumulative_bytes,
        },
        "anchor_center": {
            "entry_index": center_idx,
            "line_no": entries[center_idx].line_no,
            "ts": entries[center_idx].ts,
            "reason": "threshold_cross"
            if anchor is not None
            else ("phase_last_payload" if fallback is not None else "phase_start"),
        },
        "phase_tuple_counts": phase_summary,
        "near_anchor_tuple_counts": near_summary,
        "context": {
            "radius": context,
            "lines": {str(k): v for k, v in sorted(context_map.items())},
        },
    }


def trim_cell(value: str, width: int) -> str:
    if len(value) <= width:
        return value.ljust(width)
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def render_metric_table(good: Dict[str, object], bad: Dict[str, object], threshold: int, context: int) -> str:
    rows = [
        ("bus/device", f"{good.get('bus')}/{good.get('device')}", f"{bad.get('bus')}/{bad.get('device')}"),
        ("line_count", str(good.get("line_count")), str(bad.get("line_count"))),
        ("param_phase_count", str(good.get("param_phase_count")), str(bad.get("param_phase_count"))),
        (
            "selected_phase_index",
            str(good.get("selected_phase_index")),
            str(bad.get("selected_phase_index")),
        ),
        (
            "phase_start_line",
            str(((good.get("phase_start") or {}).get("line_no"))),
            str(((bad.get("phase_start") or {}).get("line_no"))),
        ),
        (
            "phase_payload_total",
            str(good.get("phase_payload_bytes_total")),
            str(bad.get("phase_payload_bytes_total")),
        ),
        (
            "payload_write_count",
            str(good.get("payload_write_count")),
            str(bad.get("payload_write_count")),
        ),
        (
            f"anchor>=threshold({threshold})",
            "yes" if good.get("anchor_reached_threshold") else "no",
            "yes" if bad.get("anchor_reached_threshold") else "no",
        ),
        (
            "anchor_line",
            str(((good.get("anchor_write") or {}).get("line_no"))),
            str(((bad.get("anchor_write") or {}).get("line_no"))),
        ),
        (
            "anchor_cumulative",
            str(((good.get("anchor_write") or {}).get("cumulative_bytes"))),
            str(((bad.get("anchor_write") or {}).get("cumulative_bytes"))),
        ),
        (
            "anchor_center_line",
            str(((good.get("anchor_center") or {}).get("line_no"))),
            str(((bad.get("anchor_center") or {}).get("line_no"))),
        ),
        (
            f"context_radius({context})",
            str(((good.get("context") or {}).get("radius"))),
            str(((bad.get("context") or {}).get("radius"))),
        ),
    ]
    lines = ["Metric                          Good                 Bad"]
    lines.append("----------------------------------------------------------")
    for metric, gv, bv in rows:
        lines.append(f"{metric:<30} {gv:<20} {bv:<20}")
    return "\n".join(lines)


def render_counter_diff(
    title: str, good_map: Dict[str, int], bad_map: Dict[str, int], top: int
) -> str:
    rows = counter_diff(good_map, bad_map, top=top)
    lines = [title]
    if not rows:
        lines.append("  (none)")
        return "\n".join(lines)
    lines.append("  tuple | good | bad | delta_bad_minus_good")
    for row in rows:
        lines.append(
            f"  {row['tuple']} | {row['good']} | {row['bad']} | {row['delta_bad_minus_good']}"
        )
    return "\n".join(lines)


def render_only_sets(title: str, data: Dict[str, int]) -> str:
    lines = [title]
    if not data:
        lines.append("  (none)")
        return "\n".join(lines)
    for key, value in data.items():
        lines.append(f"  {key} -> {value}")
    return "\n".join(lines)


def render_context_side_by_side(good: Dict[str, object], bad: Dict[str, object], width: int = 98) -> str:
    good_lines = (good.get("context") or {}).get("lines") or {}
    bad_lines = (bad.get("context") or {}).get("lines") or {}
    keys = sorted(
        {int(k) for k in good_lines.keys()} | {int(k) for k in bad_lines.keys()}
    )
    lines = []
    lines.append("Near-anchor context (good | bad)")
    lines.append(
        f"{'offset':<7} {'good':<{width}} | {'bad':<{width}}"
    )
    lines.append("-" * (8 + width + 3 + width))
    for key in keys:
        g_obj = good_lines.get(str(key))
        b_obj = bad_lines.get(str(key))
        g_text = g_obj["render"] if isinstance(g_obj, dict) and "render" in g_obj else ""
        b_text = b_obj["render"] if isinstance(b_obj, dict) and "render" in b_obj else ""
        lines.append(
            f"{key:+06d} {trim_cell(g_text, width)} | {trim_cell(b_text, width)}"
        )
    return "\n".join(lines)


def build_output(
    good: Dict[str, object], bad: Dict[str, object], threshold: int, context: int
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "threshold": threshold,
        "context": context,
        "good": good,
        "bad": bad,
    }
    if "error" in good or "error" in bad:
        return out

    good_phase = (good["phase_tuple_counts"] if isinstance(good.get("phase_tuple_counts"), dict) else {})
    bad_phase = (bad["phase_tuple_counts"] if isinstance(bad.get("phase_tuple_counts"), dict) else {})
    good_near = (
        good["near_anchor_tuple_counts"] if isinstance(good.get("near_anchor_tuple_counts"), dict) else {}
    )
    bad_near = (
        bad["near_anchor_tuple_counts"] if isinstance(bad.get("near_anchor_tuple_counts"), dict) else {}
    )

    good_near_control = good_near.get("control_request", {}) if isinstance(good_near, dict) else {}
    bad_near_control = bad_near.get("control_request", {}) if isinstance(bad_near, dict) else {}

    out["diff"] = {
        "phase_transfer": counter_diff(
            good_phase.get("transfer", {}), bad_phase.get("transfer", {})
        ),
        "phase_status": counter_diff(
            good_phase.get("status", {}), bad_phase.get("status", {})
        ),
        "phase_transfer_status": counter_diff(
            good_phase.get("transfer_status", {}), bad_phase.get("transfer_status", {})
        ),
        "near_transfer": counter_diff(
            good_near.get("transfer", {}), bad_near.get("transfer", {})
        ),
        "near_status": counter_diff(
            good_near.get("status", {}), bad_near.get("status", {})
        ),
        "near_transfer_status": counter_diff(
            good_near.get("transfer_status", {}), bad_near.get("transfer_status", {})
        ),
        "phase_control_request": counter_diff(
            good_phase.get("control_request", {}), bad_phase.get("control_request", {})
        ),
        "near_control_request": counter_diff(good_near_control, bad_near_control),
        "near_control_only_good": only_in(good_near_control, bad_near_control),
        "near_control_only_bad": only_in(bad_near_control, good_near_control),
    }
    return out


def render_text(report: Dict[str, object], top: int) -> str:
    good = report.get("good", {})
    bad = report.get("bad", {})
    threshold = int(report.get("threshold", 0))
    context = int(report.get("context", 0))

    lines: List[str] = []
    lines.append("usbmon parameter handshake probe")
    lines.append(f"good_log={good.get('log_path')}")
    lines.append(f"bad_log={bad.get('log_path')}")
    lines.append(f"threshold={threshold} context={context}")
    lines.append("")

    if "error" in good or "error" in bad:
        if "error" in good:
            lines.append(f"good_error={good['error']}")
        if "error" in bad:
            lines.append(f"bad_error={bad['error']}")
        return "\n".join(lines)

    lines.append(render_metric_table(good, bad, threshold=threshold, context=context))
    lines.append("")

    good_phase = good["phase_tuple_counts"]
    bad_phase = bad["phase_tuple_counts"]
    good_near = good["near_anchor_tuple_counts"]
    bad_near = bad["near_anchor_tuple_counts"]

    lines.append(
        render_counter_diff(
            "Phase transfer tuple diff (top by |delta|)",
            good_phase.get("transfer", {}),
            bad_phase.get("transfer", {}),
            top=top,
        )
    )
    lines.append("")
    lines.append(
        render_counter_diff(
            "Phase status tuple diff (top by |delta|)",
            good_phase.get("status", {}),
            bad_phase.get("status", {}),
            top=top,
        )
    )
    lines.append("")
    lines.append(
        render_counter_diff(
            "Phase transfer/status tuple diff (top by |delta|)",
            good_phase.get("transfer_status", {}),
            bad_phase.get("transfer_status", {}),
            top=top,
        )
    )
    lines.append("")
    lines.append(
        render_counter_diff(
            "Near-anchor transfer tuple diff (top by |delta|)",
            good_near.get("transfer", {}),
            bad_near.get("transfer", {}),
            top=top,
        )
    )
    lines.append("")
    lines.append(
        render_counter_diff(
            "Near-anchor status tuple diff (top by |delta|)",
            good_near.get("status", {}),
            bad_near.get("status", {}),
            top=top,
        )
    )
    lines.append("")
    lines.append(
        render_counter_diff(
            "Near-anchor transfer/status tuple diff (top by |delta|)",
            good_near.get("transfer_status", {}),
            bad_near.get("transfer_status", {}),
            top=top,
        )
    )
    lines.append("")
    lines.append(
        render_counter_diff(
            "Phase control-request tuple diff (top by |delta|)",
            good_phase.get("control_request", {}),
            bad_phase.get("control_request", {}),
            top=top,
        )
    )
    lines.append("")
    lines.append(
        render_counter_diff(
            "Near-anchor control-request tuple diff (top by |delta|)",
            good_near.get("control_request", {}),
            bad_near.get("control_request", {}),
            top=top,
        )
    )
    lines.append("")

    only_good = only_in(
        good_near.get("control_request", {}), bad_near.get("control_request", {})
    )
    only_bad = only_in(
        bad_near.get("control_request", {}), good_near.get("control_request", {})
    )
    lines.append(render_only_sets("Near-anchor control tuples only in good:", only_good))
    lines.append("")
    lines.append(render_only_sets("Near-anchor control tuples only in bad:", only_bad))
    lines.append("")
    lines.append(render_context_side_by_side(good, bad))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two usbmon logs around the tag=2 parameter-stream threshold anchor."
    )
    parser.add_argument("good_log", type=Path, help="Known-good usbmon text log path.")
    parser.add_argument("bad_log", type=Path, help="Known-bad usbmon text log path.")
    parser.add_argument("--bus", help="USB bus filter (auto-selected if omitted).")
    parser.add_argument("--device", help="USB device filter (auto-selected if omitted).")
    parser.add_argument(
        "--threshold",
        type=int,
        default=49_152,
        help="Cumulative Bo submit bytes threshold in parameter phase (default: 49152).",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=8,
        help="Number of lines before/after anchor line for near-window context (default: 8).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Top tuple diffs to print per section (default: 30).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    args = parser.parse_args()

    if args.threshold <= 0:
        raise SystemExit("--threshold must be > 0")
    if args.context < 0:
        raise SystemExit("--context must be >= 0")
    if args.top <= 0:
        raise SystemExit("--top must be > 0")

    good = analyze_one(
        label="good",
        path=args.good_log,
        bus_override=args.bus,
        device_override=args.device,
        threshold=args.threshold,
        context=args.context,
    )
    bad = analyze_one(
        label="bad",
        path=args.bad_log,
        bus_override=args.bus,
        device_override=args.device,
        threshold=args.threshold,
        context=args.context,
    )
    report = build_output(good, bad, threshold=args.threshold, context=args.context)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report, top=args.top))

    return 0 if "error" not in good and "error" not in bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
