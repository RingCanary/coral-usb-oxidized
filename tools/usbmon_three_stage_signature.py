#!/usr/bin/env python3
"""Detect repeated 3-stage EdgeTPU bulk cycle signatures in usbmon logs.

Pattern form:
- Bo(size1) -> Bo(size2) -> Bo(size3) -> Bi(size_out)

The tool supports:
- automatic candidate discovery from repeated Bo/Bo/Bo/Bi completion windows
- explicit pattern matching via CLI sizes
- per-cycle timing and interval statistics
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
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
    status: str
    size: Optional[int]
    tokens: List[str]


@dataclass
class Cycle:
    index: int
    bo1_ts: int
    bo2_ts: int
    bo3_ts: int
    bi_out_ts: int
    bo1_to_bo2_ms: float
    bo2_to_bo3_ms: float
    bo3_to_bi_out_ms: float
    bo1_to_bi_out_ms: float
    gap_events_1_2: int
    gap_events_2_3: int
    gap_events_3_4: int


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
    counts = Counter()
    for e in entries:
        if bus is not None and e.bus != bus:
            continue
        counts[e.dev] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


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
        return {"count": 0.0}
    sorted_values = sorted(values)
    return {
        "count": float(len(values)),
        "min": sorted_values[0],
        "p50": percentile(sorted_values, 0.5),
        "p95": percentile(sorted_values, 0.95),
        "avg": statistics.fmean(values),
        "max": sorted_values[-1],
    }


def bulk_success(entries: List[Entry]) -> List[Entry]:
    return [
        e
        for e in entries
        if e.event == "C" and e.status == "0" and e.size is not None and e.transfer in {"Bo", "Bi"}
    ]


def pattern_label(pattern: Tuple[int, int, int, int]) -> str:
    return f"Bo({pattern[0]}) -> Bo({pattern[1]}) -> Bo({pattern[2]}) -> Bi({pattern[3]})"


def discover_candidates(events: List[Entry], min_size: int) -> Counter[Tuple[int, int, int, int]]:
    filtered = [e for e in events if (e.size or 0) >= min_size]
    counts: Counter[Tuple[int, int, int, int]] = Counter()
    for i in range(len(filtered) - 3):
        a, b, c, d = filtered[i], filtered[i + 1], filtered[i + 2], filtered[i + 3]
        if a.transfer == "Bo" and b.transfer == "Bo" and c.transfer == "Bo" and d.transfer == "Bi":
            counts[(int(a.size), int(b.size), int(c.size), int(d.size))] += 1
    return counts


def detect_cycles(
    events: List[Entry],
    pattern: Tuple[int, int, int, int],
    max_stage_gap_us: int,
) -> List[Cycle]:
    expected = [("Bo", pattern[0]), ("Bo", pattern[1]), ("Bo", pattern[2]), ("Bi", pattern[3])]
    cycles: List[Cycle] = []
    state = 0
    matched_events: List[Entry] = []
    matched_idx: List[int] = []

    for idx, e in enumerate(events):
        if state > 0 and max_stage_gap_us > 0 and matched_events:
            if e.ts - matched_events[-1].ts > max_stage_gap_us:
                state = 0
                matched_events = []
                matched_idx = []

        exp_transfer, exp_size = expected[state]
        if e.transfer == exp_transfer and e.size == exp_size:
            matched_events.append(e)
            matched_idx.append(idx)
            state += 1
            if state == len(expected):
                bo1, bo2, bo3, bi = matched_events
                i1, i2, i3, i4 = matched_idx
                cycles.append(
                    Cycle(
                        index=len(cycles) + 1,
                        bo1_ts=bo1.ts,
                        bo2_ts=bo2.ts,
                        bo3_ts=bo3.ts,
                        bi_out_ts=bi.ts,
                        bo1_to_bo2_ms=fmt_ms_from_us(bo2.ts - bo1.ts),
                        bo2_to_bo3_ms=fmt_ms_from_us(bo3.ts - bo2.ts),
                        bo3_to_bi_out_ms=fmt_ms_from_us(bi.ts - bo3.ts),
                        bo1_to_bi_out_ms=fmt_ms_from_us(bi.ts - bo1.ts),
                        gap_events_1_2=max(0, i2 - i1 - 1),
                        gap_events_2_3=max(0, i3 - i2 - 1),
                        gap_events_3_4=max(0, i4 - i3 - 1),
                    )
                )
                state = 0
                matched_events = []
                matched_idx = []
            continue

        if state > 0 and e.transfer == expected[0][0] and e.size == expected[0][1]:
            # restart from a fresh cycle anchor
            state = 1
            matched_events = [e]
            matched_idx = [idx]

    return cycles


def cycle_anchor_ts(c: Cycle, anchor: int) -> int:
    if anchor == 1:
        return c.bo1_ts
    if anchor == 2:
        return c.bo2_ts
    if anchor == 3:
        return c.bo3_ts
    return c.bi_out_ts


def analyze_log(
    path: Path,
    bus: Optional[str],
    device: Optional[str],
    bo1: Optional[int],
    bo2: Optional[int],
    bo3: Optional[int],
    bi_out: Optional[int],
    min_size: int,
    min_count: int,
    top_candidates: int,
    interval_anchor: int,
    max_stage_gap_us: int,
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

    events = bulk_success(entries)
    bo_sizes = Counter(int(e.size) for e in events if e.transfer == "Bo")
    bi_sizes = Counter(int(e.size) for e in events if e.transfer == "Bi")
    candidates = discover_candidates(events, min_size=min_size)

    candidate_items = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    top_items = candidate_items[:top_candidates]
    top_candidate_data = [
        {"pattern": {"bo": [p[0], p[1], p[2]], "bi_out": p[3]}, "count": count, "label": pattern_label(p)}
        for p, count in top_items
    ]

    explicit = all(v is not None for v in (bo1, bo2, bo3, bi_out))
    selected_pattern: Optional[Tuple[int, int, int, int]] = None
    selected_source = "auto"
    warnings: List[str] = []

    if explicit:
        selected_pattern = (int(bo1), int(bo2), int(bo3), int(bi_out))
        selected_source = "explicit"
    elif top_items and top_items[0][1] >= min_count:
        selected_pattern = top_items[0][0]
    else:
        warnings.append(
            "no auto-selected pattern met min-count; pass explicit --bo-1/--bo-2/--bo-3/--bi-out to force."
        )

    cycles: List[Cycle] = []
    interval_stats = {"count": 0.0}
    if selected_pattern is not None:
        cycles = detect_cycles(events, pattern=selected_pattern, max_stage_gap_us=max_stage_gap_us)
        if len(cycles) > 1:
            intervals = [
                fmt_ms_from_us(cycle_anchor_ts(cur, interval_anchor) - cycle_anchor_ts(prev, interval_anchor))
                for prev, cur in zip(cycles, cycles[1:])
            ]
            interval_stats = stats(intervals)
        if selected_source == "auto":
            auto_count = candidates.get(selected_pattern, 0)
            if auto_count and auto_count != len(cycles):
                warnings.append(
                    "auto window count differs from extracted non-overlapping cycles "
                    f"({auto_count} vs {len(cycles)})."
                )

    result: Dict[str, object] = {
        "log_path": str(path),
        "bus": bus,
        "device": dev,
        "line_count": len(entries),
        "bulk_complete_total": len(events),
        "bulk_complete_sizes": {
            "Bo": {str(k): v for k, v in sorted(bo_sizes.items())},
            "Bi": {str(k): v for k, v in sorted(bi_sizes.items())},
        },
        "candidate_scan": {
            "min_size": min_size,
            "min_count": min_count,
            "top_candidates": top_candidate_data,
            "unique_candidates": len(candidates),
        },
        "selected_pattern": None,
        "cycle_count": 0,
        "timing_ms": {},
        "first_three": [],
        "last_three": [],
        "warnings": warnings,
    }

    if selected_pattern is not None:
        result["selected_pattern"] = {
            "bo": [selected_pattern[0], selected_pattern[1], selected_pattern[2]],
            "bi_out": selected_pattern[3],
            "label": pattern_label(selected_pattern),
            "source": selected_source,
            "candidate_count": candidates.get(selected_pattern, 0),
        }
        result["cycle_count"] = len(cycles)
        result["timing_ms"] = {
            "bo1_to_bo2": stats([c.bo1_to_bo2_ms for c in cycles]),
            "bo2_to_bo3": stats([c.bo2_to_bo3_ms for c in cycles]),
            "bo3_to_bi_out": stats([c.bo3_to_bi_out_ms for c in cycles]),
            "bo1_to_bi_out": stats([c.bo1_to_bi_out_ms for c in cycles]),
            "interval": {"anchor_stage": interval_anchor, "stats": interval_stats},
            "gap_events": {
                "stage_1_2": stats([float(c.gap_events_1_2) for c in cycles]),
                "stage_2_3": stats([float(c.gap_events_2_3) for c in cycles]),
                "stage_3_4": stats([float(c.gap_events_3_4) for c in cycles]),
            },
        }
        result["first_three"] = [asdict(c) for c in cycles[:3]]
        result["last_three"] = [asdict(c) for c in cycles[-3:]]

    return result


def format_stats_line(name: str, s: Dict[str, float]) -> str:
    if s.get("count", 0) == 0:
        return f"{name}: none"
    return (
        f"{name}: n={int(s['count'])} min={s['min']:.3f} "
        f"p50={s['p50']:.3f} p95={s['p95']:.3f} avg={s['avg']:.3f} max={s['max']:.3f}"
    )


def render_text(data: Dict[str, object], show_candidates: int) -> str:
    if "error" in data:
        return f"ERROR: {data['error']}"

    lines: List[str] = []
    lines.append(f"log={data['log_path']}")
    lines.append(f"bus={data['bus'] or '*'} device={data['device']} lines={data['line_count']}")
    lines.append(f"bulk_complete_total={data['bulk_complete_total']}")
    lines.append("bulk_complete_sizes.Bo=" + json.dumps(data["bulk_complete_sizes"]["Bo"], sort_keys=True))
    lines.append("bulk_complete_sizes.Bi=" + json.dumps(data["bulk_complete_sizes"]["Bi"], sort_keys=True))

    scan = data["candidate_scan"]
    lines.append(
        "candidate_scan min_size={} min_count={} unique={}".format(
            scan["min_size"], scan["min_count"], scan["unique_candidates"]
        )
    )
    lines.append("top_candidates:")
    top = scan["top_candidates"][:show_candidates]
    if not top:
        lines.append("  (none)")
    else:
        for i, item in enumerate(top, start=1):
            lines.append(f"  {i}. {item['label']} count={item['count']}")

    if data["selected_pattern"] is None:
        lines.append("selected_pattern=(none)")
    else:
        selected = data["selected_pattern"]
        lines.append(
            "selected_pattern={} source={} candidate_count={} cycle_count={}".format(
                selected["label"], selected["source"], selected["candidate_count"], data["cycle_count"]
            )
        )
        timing = data["timing_ms"]
        lines.append(format_stats_line("bo1_to_bo2_ms", timing["bo1_to_bo2"]))
        lines.append(format_stats_line("bo2_to_bo3_ms", timing["bo2_to_bo3"]))
        lines.append(format_stats_line("bo3_to_bi_out_ms", timing["bo3_to_bi_out"]))
        lines.append(format_stats_line("bo1_to_bi_out_ms", timing["bo1_to_bi_out"]))
        iv = timing["interval"]
        lines.append(
            format_stats_line(f"cycle_interval_ms(anchor_stage={iv['anchor_stage']})", iv["stats"])
        )
        lines.append(format_stats_line("gap_events_1_2", timing["gap_events"]["stage_1_2"]))
        lines.append(format_stats_line("gap_events_2_3", timing["gap_events"]["stage_2_3"]))
        lines.append(format_stats_line("gap_events_3_4", timing["gap_events"]["stage_3_4"]))

    warnings = data.get("warnings", [])
    if warnings:
        lines.append("warnings:")
        for w in warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect repeated 3-stage bulk signatures in usbmon logs.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--bus", help="Filter bus number (for example 4).")
    parser.add_argument("--device", help="Filter USB device id on bus (for example 005).")
    parser.add_argument("--bo-1", type=int, help="First Bo completion size.")
    parser.add_argument("--bo-2", type=int, help="Second Bo completion size.")
    parser.add_argument("--bo-3", type=int, help="Third Bo completion size.")
    parser.add_argument("--bi-out", type=int, help="Bi completion size.")
    parser.add_argument("--min-size", type=int, default=64, help="Minimum completion size for auto candidate scan.")
    parser.add_argument("--min-count", type=int, default=3, help="Minimum candidate count for auto-selection.")
    parser.add_argument("--top-candidates", type=int, default=10, help="How many auto candidates to retain.")
    parser.add_argument(
        "--interval-anchor",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help="Stage used for cycle-interval stats (1=Bo1, 2=Bo2, 3=Bo3, 4=Bi).",
    )
    parser.add_argument(
        "--max-stage-gap-us",
        type=int,
        default=20_000,
        help="Reset partial cycle if stage-to-stage gap exceeds this threshold (0 disables).",
    )
    parser.add_argument("--show-candidates", type=int, default=5, help="Top candidates shown in text output.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()

    data = analyze_log(
        path=args.log,
        bus=args.bus,
        device=args.device,
        bo1=args.bo_1,
        bo2=args.bo_2,
        bo3=args.bo_3,
        bi_out=args.bi_out,
        min_size=args.min_size,
        min_count=args.min_count,
        top_candidates=args.top_candidates,
        interval_anchor=args.interval_anchor,
        max_stage_gap_us=args.max_stage_gap_us,
    )
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(render_text(data, show_candidates=args.show_candidates))
    return 0 if "error" not in data else 1


if __name__ == "__main__":
    raise SystemExit(main())
