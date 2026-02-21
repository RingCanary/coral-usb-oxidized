#!/usr/bin/env python3
"""Phase-oriented analyzer for Linux usbmon captures.

This tool is intentionally opinionated for Coral EdgeTPU reverse-engineering:
- segment timeline by inactivity gaps
- summarize transfer/status patterns per device
- detect per-inference cycles (default Bo 225824 -> Bo 150528 -> Bi 1008)
- compare two runs
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Entry:
    ts: int
    event: str
    transfer: str
    bus: str
    dev: str
    ep: str
    status_raw: str
    size: Optional[int]
    tokens: List[str]


@dataclass
class Segment:
    index: int
    start_ts: int
    end_ts: int
    duration_ms: float
    lines: int
    by_transfer: Dict[str, int]


@dataclass
class Cycle:
    index: int
    bo_a_ts: int
    bo_b_ts: int
    bi_out_ts: int
    bo_a_to_bo_b_ms: float
    bo_b_to_bi_out_ms: float
    bo_a_to_bi_out_ms: float


def parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


def parse_line(line: str) -> Optional[Entry]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    ts = parse_int(parts[1])
    if ts is None:
        return None
    event = parts[2]
    if ":" not in parts[3]:
        return None
    transfer_parts = parts[3].split(":")
    if len(transfer_parts) != 4:
        return None
    transfer, bus, dev, ep = transfer_parts
    status_raw = parts[4]
    size = None
    if len(parts) > 5:
        size = parse_int(parts[5])
    return Entry(
        ts=ts,
        event=event,
        transfer=transfer,
        bus=bus,
        dev=dev,
        ep=ep,
        status_raw=status_raw,
        size=size,
        tokens=parts,
    )


def load_entries(path: Path) -> List[Entry]:
    entries: List[Entry] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            entry = parse_line(line)
            if entry is not None:
                entries.append(entry)
    return entries


def fmt_ms_from_us(us: int) -> float:
    return us / 1000.0


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    idx = int(round((len(sorted_values) - 1) * p))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": float(len(values)),
        "min": sorted_values[0],
        "p50": percentile(sorted_values, 0.5),
        "p95": percentile(sorted_values, 0.95),
        "avg": statistics.fmean(values),
        "max": sorted_values[-1],
    }


def dominant_device(entries: Iterable[Entry], bus_filter: Optional[str]) -> Optional[str]:
    c = Counter()
    for e in entries:
        if bus_filter is not None and e.bus != bus_filter:
            continue
        c[e.dev] += 1
    if not c:
        return None
    return c.most_common(1)[0][0]


def build_segments(entries: List[Entry], gap_us: int) -> List[Segment]:
    if not entries:
        return []
    segments: List[Segment] = []
    seg_index = 1
    seg_start = entries[0].ts
    seg_end = entries[0].ts
    transfer_counts: Counter[str] = Counter()
    transfer_counts[entries[0].transfer] += 1
    seg_lines = 1
    prev_ts = entries[0].ts

    for entry in entries[1:]:
        if entry.ts - prev_ts > gap_us:
            segments.append(
                Segment(
                    index=seg_index,
                    start_ts=seg_start,
                    end_ts=seg_end,
                    duration_ms=fmt_ms_from_us(seg_end - seg_start),
                    lines=seg_lines,
                    by_transfer=dict(transfer_counts),
                )
            )
            seg_index += 1
            seg_start = entry.ts
            seg_end = entry.ts
            seg_lines = 1
            transfer_counts = Counter()
            transfer_counts[entry.transfer] += 1
        else:
            seg_end = entry.ts
            seg_lines += 1
            transfer_counts[entry.transfer] += 1
        prev_ts = entry.ts

    segments.append(
        Segment(
            index=seg_index,
            start_ts=seg_start,
            end_ts=seg_end,
            duration_ms=fmt_ms_from_us(seg_end - seg_start),
            lines=seg_lines,
            by_transfer=dict(transfer_counts),
        )
    )
    return segments


def detect_cycles(
    entries: List[Entry], bo_a: int, bo_b: int, bi_out: int
) -> List[Cycle]:
    cycles: List[Cycle] = []
    state = 0
    bo_a_ts = 0
    bo_b_ts = 0

    for entry in entries:
        if entry.event != "C":
            continue
        if entry.status_raw != "0":
            continue
        if entry.size is None:
            continue

        if state == 0:
            if entry.transfer == "Bo" and entry.size == bo_a:
                bo_a_ts = entry.ts
                state = 1
        elif state == 1:
            if entry.transfer == "Bo" and entry.size == bo_b:
                bo_b_ts = entry.ts
                state = 2
            elif entry.transfer == "Bo" and entry.size == bo_a:
                bo_a_ts = entry.ts
        elif state == 2:
            if entry.transfer == "Bi" and entry.size == bi_out:
                cycle_idx = len(cycles) + 1
                cycles.append(
                    Cycle(
                        index=cycle_idx,
                        bo_a_ts=bo_a_ts,
                        bo_b_ts=bo_b_ts,
                        bi_out_ts=entry.ts,
                        bo_a_to_bo_b_ms=fmt_ms_from_us(bo_b_ts - bo_a_ts),
                        bo_b_to_bi_out_ms=fmt_ms_from_us(entry.ts - bo_b_ts),
                        bo_a_to_bi_out_ms=fmt_ms_from_us(entry.ts - bo_a_ts),
                    )
                )
                state = 0
            elif entry.transfer == "Bo" and entry.size == bo_a:
                bo_a_ts = entry.ts
                state = 1
    return cycles


def analyze_log(
    path: Path,
    bus: Optional[str],
    device: Optional[str],
    gap_us: int,
    bo_a: int,
    bo_b: int,
    bi_out: int,
) -> Dict[str, object]:
    entries_all = load_entries(path)
    if bus is not None:
        entries_bus = [e for e in entries_all if e.bus == bus]
    else:
        entries_bus = entries_all

    selected_device = device or dominant_device(entries_bus, bus)
    if selected_device is None:
        return {
            "log_path": str(path),
            "error": "no entries matched selected bus/device",
        }

    entries = [e for e in entries_bus if e.dev == selected_device]
    if not entries:
        return {
            "log_path": str(path),
            "bus": bus,
            "device": selected_device,
            "error": "no entries for selected device",
        }

    ts_first = entries[0].ts
    ts_last = entries[-1].ts

    by_transfer = Counter(e.transfer for e in entries)
    by_event = Counter(e.event for e in entries)
    by_status = Counter(e.status_raw for e in entries if parse_int(e.status_raw) is not None)
    by_transfer_status: Dict[str, Counter[str]] = defaultdict(Counter)
    for e in entries:
        if parse_int(e.status_raw) is not None:
            by_transfer_status[e.transfer][e.status_raw] += 1

    bulk_c_sizes = Counter(
        e.size
        for e in entries
        if e.event == "C"
        and e.status_raw == "0"
        and e.size is not None
        and e.transfer in {"Bo", "Bi"}
    )
    bulk_c_sizes_by_transfer: Dict[str, Counter[int]] = {"Bo": Counter(), "Bi": Counter()}
    for e in entries:
        if (
            e.event == "C"
            and e.status_raw == "0"
            and e.size is not None
            and e.transfer in bulk_c_sizes_by_transfer
        ):
            bulk_c_sizes_by_transfer[e.transfer][e.size] += 1

    segments = build_segments(entries, gap_us=gap_us)
    cycles = detect_cycles(entries, bo_a=bo_a, bo_b=bo_b, bi_out=bi_out)
    cycle_intervals = []
    if len(cycles) > 1:
        for prev, cur in zip(cycles, cycles[1:]):
            cycle_intervals.append(fmt_ms_from_us(cur.bo_b_ts - prev.bo_b_ts))

    first_bo_b_ts = next(
        (
            e.ts
            for e in entries
            if e.event == "C"
            and e.status_raw == "0"
            and e.transfer == "Bo"
            and e.size == bo_b
        ),
        None,
    )
    pre_first_bo_b_bytes = 0
    if first_bo_b_ts is not None:
        pre_first_bo_b_bytes = sum(
            e.size or 0
            for e in entries
            if e.event == "C"
            and e.status_raw == "0"
            and e.transfer == "Bo"
            and e.size is not None
            and e.ts < first_bo_b_ts
        )

    result = {
        "log_path": str(path),
        "bus": bus,
        "device": selected_device,
        "line_count": len(entries),
        "time": {
            "first_ts": ts_first,
            "last_ts": ts_last,
            "duration_ms": fmt_ms_from_us(ts_last - ts_first),
        },
        "counts": {
            "event": dict(by_event),
            "transfer": dict(by_transfer),
            "status": dict(by_status),
            "transfer_status": {k: dict(v) for k, v in by_transfer_status.items()},
        },
        "bulk_complete_sizes": {
            "all": {str(k): v for k, v in sorted(bulk_c_sizes.items())},
            "Bo": {str(k): v for k, v in sorted(bulk_c_sizes_by_transfer["Bo"].items())},
            "Bi": {str(k): v for k, v in sorted(bulk_c_sizes_by_transfer["Bi"].items())},
        },
        "segments": [asdict(seg) for seg in segments],
        "cycle_pattern": {
            "bo_a": bo_a,
            "bo_b": bo_b,
            "bi_out": bi_out,
            "count": len(cycles),
            "timing_ms": {
                "bo_a_to_bo_b": stats([c.bo_a_to_bo_b_ms for c in cycles]),
                "bo_b_to_bi_out": stats([c.bo_b_to_bi_out_ms for c in cycles]),
                "bo_a_to_bi_out": stats([c.bo_a_to_bi_out_ms for c in cycles]),
                "bo_b_interval": stats(cycle_intervals),
            },
            "first_three": [asdict(c) for c in cycles[:3]],
            "last_three": [asdict(c) for c in cycles[-3:]],
        },
        "pre_first_bo_b_bulk_out_bytes": pre_first_bo_b_bytes,
    }
    return result


def render_report_text(data: Dict[str, object]) -> str:
    if "error" in data:
        return f"ERROR: {data['error']}"

    lines: List[str] = []
    lines.append(f"log={data['log_path']}")
    lines.append(f"bus={data['bus'] or '*'} device={data['device']}")
    time = data["time"]
    lines.append(
        "duration_ms={:.3f} first_ts={} last_ts={} lines={}".format(
            time["duration_ms"], time["first_ts"], time["last_ts"], data["line_count"]
        )
    )

    counts = data["counts"]
    lines.append("transfer_counts=" + json.dumps(counts["transfer"], sort_keys=True))
    lines.append("event_counts=" + json.dumps(counts["event"], sort_keys=True))
    lines.append("status_counts=" + json.dumps(counts["status"], sort_keys=True))

    lines.append("bulk_complete_sizes.Bo=" + json.dumps(data["bulk_complete_sizes"]["Bo"], sort_keys=True))
    lines.append("bulk_complete_sizes.Bi=" + json.dumps(data["bulk_complete_sizes"]["Bi"], sort_keys=True))
    lines.append(f"pre_first_bo_b_bulk_out_bytes={data['pre_first_bo_b_bulk_out_bytes']}")

    pattern = data["cycle_pattern"]
    lines.append(
        "cycle_pattern=Bo({}) -> Bo({}) -> Bi({}), count={}".format(
            pattern["bo_a"], pattern["bo_b"], pattern["bi_out"], pattern["count"]
        )
    )
    timing = pattern["timing_ms"]
    for key in ("bo_a_to_bo_b", "bo_b_to_bi_out", "bo_a_to_bi_out", "bo_b_interval"):
        s = timing[key]
        if s.get("count", 0) == 0:
            lines.append(f"{key}: none")
            continue
        lines.append(
            "{}: n={} min={:.3f} p50={:.3f} p95={:.3f} avg={:.3f} max={:.3f}".format(
                key,
                int(s["count"]),
                s["min"],
                s["p50"],
                s["p95"],
                s["avg"],
                s["max"],
            )
        )

    lines.append("segments:")
    for seg in data["segments"]:
        lines.append(
            "  seg={} start={} end={} dur_ms={:.3f} lines={} transfers={}".format(
                seg["index"],
                seg["start_ts"],
                seg["end_ts"],
                seg["duration_ms"],
                seg["lines"],
                json.dumps(seg["by_transfer"], sort_keys=True),
            )
        )
    return "\n".join(lines)


def diff_reports(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, object]:
    def g(d: Dict[str, object], *path: str, default=None):
        cur = d
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    return {
        "a_log": a.get("log_path"),
        "b_log": b.get("log_path"),
        "a_device": a.get("device"),
        "b_device": b.get("device"),
        "duration_ms": {
            "a": g(a, "time", "duration_ms"),
            "b": g(b, "time", "duration_ms"),
            "delta_b_minus_a": (g(b, "time", "duration_ms", default=0) or 0)
            - (g(a, "time", "duration_ms", default=0) or 0),
        },
        "lines": {
            "a": a.get("line_count"),
            "b": b.get("line_count"),
            "delta_b_minus_a": (b.get("line_count") or 0) - (a.get("line_count") or 0),
        },
        "cycle_count": {
            "a": g(a, "cycle_pattern", "count"),
            "b": g(b, "cycle_pattern", "count"),
            "delta_b_minus_a": (g(b, "cycle_pattern", "count", default=0) or 0)
            - (g(a, "cycle_pattern", "count", default=0) or 0),
        },
        "cycle_interval_avg_ms": {
            "a": g(a, "cycle_pattern", "timing_ms", "bo_b_interval", "avg"),
            "b": g(b, "cycle_pattern", "timing_ms", "bo_b_interval", "avg"),
        },
        "cycle_inference_leg_avg_ms": {
            "a": g(a, "cycle_pattern", "timing_ms", "bo_b_to_bi_out", "avg"),
            "b": g(b, "cycle_pattern", "timing_ms", "bo_b_to_bi_out", "avg"),
        },
        "pre_first_bo_b_bulk_out_bytes": {
            "a": a.get("pre_first_bo_b_bulk_out_bytes"),
            "b": b.get("pre_first_bo_b_bulk_out_bytes"),
            "delta_b_minus_a": (b.get("pre_first_bo_b_bulk_out_bytes") or 0)
            - (a.get("pre_first_bo_b_bulk_out_bytes") or 0),
        },
        "transfer_counts": {
            "a": g(a, "counts", "transfer"),
            "b": g(b, "counts", "transfer"),
        },
        "bo_sizes": {
            "a": g(a, "bulk_complete_sizes", "Bo"),
            "b": g(b, "bulk_complete_sizes", "Bo"),
        },
        "bi_sizes": {
            "a": g(a, "bulk_complete_sizes", "Bi"),
            "b": g(b, "bulk_complete_sizes", "Bi"),
        },
    }


def render_diff_text(data: Dict[str, object]) -> str:
    lines = [
        f"a={data['a_log']}",
        f"b={data['b_log']}",
        f"a_device={data['a_device']} b_device={data['b_device']}",
        "duration_ms: a={a:.3f} b={b:.3f} delta={d:.3f}".format(
            a=data["duration_ms"]["a"] or 0.0,
            b=data["duration_ms"]["b"] or 0.0,
            d=data["duration_ms"]["delta_b_minus_a"] or 0.0,
        ),
        "lines: a={a} b={b} delta={d}".format(
            a=data["lines"]["a"], b=data["lines"]["b"], d=data["lines"]["delta_b_minus_a"]
        ),
        "cycle_count: a={a} b={b} delta={d}".format(
            a=data["cycle_count"]["a"],
            b=data["cycle_count"]["b"],
            d=data["cycle_count"]["delta_b_minus_a"],
        ),
        "cycle_interval_avg_ms: a={a:.3f} b={b:.3f}".format(
            a=data["cycle_interval_avg_ms"]["a"] or 0.0,
            b=data["cycle_interval_avg_ms"]["b"] or 0.0,
        ),
        "cycle_inference_leg_avg_ms: a={a:.3f} b={b:.3f}".format(
            a=data["cycle_inference_leg_avg_ms"]["a"] or 0.0,
            b=data["cycle_inference_leg_avg_ms"]["b"] or 0.0,
        ),
        "pre_first_bo_b_bulk_out_bytes: a={a} b={b} delta={d}".format(
            a=data["pre_first_bo_b_bulk_out_bytes"]["a"],
            b=data["pre_first_bo_b_bulk_out_bytes"]["b"],
            d=data["pre_first_bo_b_bulk_out_bytes"]["delta_b_minus_a"],
        ),
        "transfer_counts.a=" + json.dumps(data["transfer_counts"]["a"], sort_keys=True),
        "transfer_counts.b=" + json.dumps(data["transfer_counts"]["b"], sort_keys=True),
        "bo_sizes.a=" + json.dumps(data["bo_sizes"]["a"], sort_keys=True),
        "bo_sizes.b=" + json.dumps(data["bo_sizes"]["b"], sort_keys=True),
        "bi_sizes.a=" + json.dumps(data["bi_sizes"]["a"], sort_keys=True),
        "bi_sizes.b=" + json.dumps(data["bi_sizes"]["b"], sort_keys=True),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Coral-oriented usbmon phase behavior.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="Generate phase report for one usbmon log.")
    rep.add_argument("log", type=Path, help="Path to usbmon .log file.")
    rep.add_argument("--bus", help="Filter bus number (for example 4).")
    rep.add_argument("--device", help="Filter USB device id on bus (for example 005).")
    rep.add_argument("--gap-us", type=int, default=100_000, help="Gap threshold for segment splitting.")
    rep.add_argument("--bo-a", type=int, default=225_824, help="First Bo completion size in cycle pattern.")
    rep.add_argument("--bo-b", type=int, default=150_528, help="Second Bo completion size in cycle pattern.")
    rep.add_argument("--bi-out", type=int, default=1_008, help="Bi completion size in cycle pattern.")
    rep.add_argument("--json", action="store_true", help="Emit JSON output.")

    dff = sub.add_parser("diff", help="Compare two usbmon logs using phase metrics.")
    dff.add_argument("a_log", type=Path, help="Path to baseline usbmon .log.")
    dff.add_argument("b_log", type=Path, help="Path to comparison usbmon .log.")
    dff.add_argument("--bus", help="Filter bus number (for example 4).")
    dff.add_argument("--device", help="Filter USB device id on bus (for example 005).")
    dff.add_argument("--gap-us", type=int, default=100_000, help="Gap threshold for segment splitting.")
    dff.add_argument("--bo-a", type=int, default=225_824, help="First Bo completion size in cycle pattern.")
    dff.add_argument("--bo-b", type=int, default=150_528, help="Second Bo completion size in cycle pattern.")
    dff.add_argument("--bi-out", type=int, default=1_008, help="Bi completion size in cycle pattern.")
    dff.add_argument("--json", action="store_true", help="Emit JSON output.")

    args = parser.parse_args()

    if args.cmd == "report":
        result = analyze_log(
            path=args.log,
            bus=args.bus,
            device=args.device,
            gap_us=args.gap_us,
            bo_a=args.bo_a,
            bo_b=args.bo_b,
            bi_out=args.bi_out,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(render_report_text(result))
        return 0

    if args.cmd == "diff":
        a = analyze_log(
            path=args.a_log,
            bus=args.bus,
            device=args.device,
            gap_us=args.gap_us,
            bo_a=args.bo_a,
            bo_b=args.bo_b,
            bi_out=args.bi_out,
        )
        b = analyze_log(
            path=args.b_log,
            bus=args.bus,
            device=args.device,
            gap_us=args.gap_us,
            bo_a=args.bo_a,
            bo_b=args.bo_b,
            bi_out=args.bi_out,
        )
        if "error" in a or "error" in b:
            print(json.dumps({"a": a, "b": b}, indent=2, sort_keys=True))
            return 1
        d = diff_reports(a, b)
        if args.json:
            print(json.dumps(d, indent=2, sort_keys=True))
        else:
            print(render_diff_text(d))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
