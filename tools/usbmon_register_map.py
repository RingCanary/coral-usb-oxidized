#!/usr/bin/env python3
"""Extract candidate register-map activity from usbmon logs.

Focuses on control transfers used by EdgeTPU USB runtime:
- vendor 64-bit read/write: bmRequestType c0/40, bRequest 00, wIndex 0004, wLength 0008
- vendor 32-bit read/write: bmRequestType c0/40, bRequest 01, wIndex 0001, wLength 0004
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
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


def classify_control(parts: List[str]) -> Dict[str, str]:
    # Expected usbmon control submission format:
    # ... S Co:... s bmRequestType bRequest wValue wIndex wLength len ...
    # token indices: [4]=s [5]=bm [6]=bRequest [7]=wValue [8]=wIndex [9]=wLength
    if len(parts) < 10:
        return {"kind": "unknown"}
    setup_tag = parts[4]
    bm = parts[5].lower()
    breq = parts[6].lower()
    wval = parts[7].lower()
    widx = parts[8].lower()
    wlen = parts[9].lower()
    if setup_tag != "s":
        return {"kind": "unknown"}

    # Standard USB descriptor/config requests.
    if bm in {"80", "00"} and breq in {"06", "09", "31"}:
        return {
            "kind": "standard",
            "bm": bm,
            "breq": breq,
            "wval": wval,
            "widx": widx,
            "wlen": wlen,
            "op": "standard",
            "addr": wval,
        }

    # Vendor control path used by EdgeTPU runtime.
    if bm == "40" and breq == "00" and widx == "0004" and wlen == "0008":
        return {
            "kind": "vendor",
            "bm": bm,
            "breq": breq,
            "wval": wval,
            "widx": widx,
            "wlen": wlen,
            "op": "write64",
            "addr": wval,
        }
    if bm == "c0" and breq == "00" and widx == "0004" and wlen == "0008":
        return {
            "kind": "vendor",
            "bm": bm,
            "breq": breq,
            "wval": wval,
            "widx": widx,
            "wlen": wlen,
            "op": "read64",
            "addr": wval,
        }
    if bm == "40" and breq == "01" and widx == "0001" and wlen == "0004":
        return {
            "kind": "vendor",
            "bm": bm,
            "breq": breq,
            "wval": wval,
            "widx": widx,
            "wlen": wlen,
            "op": "write32",
            "addr": wval,
        }
    if bm == "c0" and breq == "01" and widx == "0001" and wlen == "0004":
        return {
            "kind": "vendor",
            "bm": bm,
            "breq": breq,
            "wval": wval,
            "widx": widx,
            "wlen": wlen,
            "op": "read32",
            "addr": wval,
        }

    return {
        "kind": "vendor_other" if bm in {"40", "c0"} else "unknown",
        "bm": bm,
        "breq": breq,
        "wval": wval,
        "widx": widx,
        "wlen": wlen,
        "op": "unknown",
        "addr": wval,
    }


def phase_for_ts(ts: int, first_bo_b: Optional[int], last_bi_out: Optional[int]) -> str:
    if first_bo_b is None or last_bi_out is None:
        return "setup_only"
    if ts < first_bo_b:
        return "pre_loop"
    if ts <= last_bi_out:
        return "loop"
    return "post_loop"


def analyze_log(path: Path, bus: Optional[str], device: Optional[str], bo_b: int, bi_out: int) -> Dict[str, object]:
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

    total_control = 0
    standard_control = 0
    vendor_control = 0
    vendor_other = 0
    unknown_control = 0
    by_op = Counter()
    by_op_phase = Counter()
    by_addr_op = Counter()
    by_addr_op_phase = Counter()
    by_standard_sig = Counter()

    for e in entries:
        if e.event != "S":
            continue
        if e.transfer not in {"Ci", "Co"}:
            continue
        if e.ep != "0":
            continue
        if len(e.tokens) < 10:
            continue
        if e.tokens[4] != "s":
            continue

        total_control += 1
        c = classify_control(e.tokens)
        kind = c["kind"]
        if kind == "standard":
            standard_control += 1
            sig = "{} {} {} {} {}".format(c["bm"], c["breq"], c["wval"], c["widx"], c["wlen"])
            by_standard_sig[sig] += 1
            continue
        if kind == "vendor":
            vendor_control += 1
        elif kind == "vendor_other":
            vendor_other += 1
        else:
            unknown_control += 1

        op = c.get("op", "unknown")
        addr = c.get("addr", "????")
        phase = phase_for_ts(e.ts, first_bo_b=first_bo_b, last_bi_out=last_bi_out)
        by_op[op] += 1
        by_op_phase[(op, phase)] += 1
        by_addr_op[(addr, op)] += 1
        by_addr_op_phase[(addr, op, phase)] += 1

    return {
        "log_path": str(path),
        "bus": bus,
        "device": dev,
        "line_count": len(entries),
        "loop_window": {"first_bo_b_ts": first_bo_b, "last_bi_out_ts": last_bi_out},
        "control_totals": {
            "total_control": total_control,
            "standard_control": standard_control,
            "vendor_control": vendor_control,
            "vendor_other": vendor_other,
            "unknown_control": unknown_control,
        },
        "by_op": dict(by_op),
        "by_op_phase": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_op_phase.items())},
        "by_addr_op": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_addr_op.items())},
        "by_addr_op_phase": {
            f"{k[0]}::{k[1]}::{k[2]}": v for k, v in sorted(by_addr_op_phase.items())
        },
        "standard_signatures": dict(by_standard_sig),
    }


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
    lines.append("control_totals=" + json.dumps(data["control_totals"], sort_keys=True))
    lines.append("ops=" + json.dumps(data["by_op"], sort_keys=True))
    lines.append("op_phase=" + json.dumps(data["by_op_phase"], sort_keys=True))

    pairs = []
    for key, count in data["by_addr_op"].items():
        addr, op = key.split("::", 1)
        pairs.append((count, addr, op))
    pairs.sort(reverse=True)
    lines.append("top_addr_ops:")
    for count, addr, op in pairs[:top]:
        phase_counts = []
        for phase in ("setup_only", "pre_loop", "loop", "post_loop"):
            k = f"{addr}::{op}::{phase}"
            v = data["by_addr_op_phase"].get(k, 0)
            if v:
                phase_counts.append(f"{phase}:{v}")
        phase_s = ",".join(phase_counts) if phase_counts else "-"
        lines.append(f"  {addr} {op} count={count} phases={phase_s}")
    return "\n".join(lines)


def render_matrix_text(matrix: Dict[str, object], top: int) -> str:
    runs = matrix["runs"]
    run_names = list(runs.keys())
    lines: List[str] = []
    lines.append("runs=" + ", ".join(run_names))
    lines.append("")

    rows = []
    for addr_op, total in matrix["addr_op_totals"].items():
        addr, op = addr_op.split("::", 1)
        rows.append((total, addr, op))
    rows.sort(reverse=True)

    header = ["addr", "op"] + run_names + ["total"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for total, addr, op in rows[:top]:
        row = [addr, op]
        for rn in run_names:
            row.append(str(runs[rn]["by_addr_op"].get(f"{addr}::{op}", 0)))
        row.append(str(total))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("phase_breakdown_for_most_active_run:")
    most_active = matrix["most_active_run"]
    lines.append(f"- run={most_active}")
    per = runs[most_active]["by_addr_op_phase"]
    for total, addr, op in rows[:top]:
        phase_items = []
        for phase in ("setup_only", "pre_loop", "loop", "post_loop"):
            v = per.get(f"{addr}::{op}::{phase}", 0)
            if v:
                phase_items.append(f"{phase}:{v}")
        if phase_items:
            lines.append(f"- {addr} {op}: " + ", ".join(phase_items))
    return "\n".join(lines)


def build_matrix(reports: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    totals = Counter()
    active_counts = {}
    for rn, rep in reports.items():
        if "error" in rep:
            continue
        c = rep["control_totals"]["vendor_control"]
        active_counts[rn] = c
        for key, v in rep["by_addr_op"].items():
            totals[key] += v
    most_active = max(active_counts.items(), key=lambda kv: kv[1])[0] if active_counts else ""
    return {
        "runs": reports,
        "addr_op_totals": dict(totals),
        "most_active_run": most_active,
    }


def parse_run_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    p = Path(path)
    return name, p


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract EdgeTPU USB control/register activity from usbmon logs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="Analyze one usbmon log.")
    rep.add_argument("log", type=Path)
    rep.add_argument("--bus")
    rep.add_argument("--device")
    rep.add_argument("--bo-b", type=int, default=150_528)
    rep.add_argument("--bi-out", type=int, default=1_008)
    rep.add_argument("--top", type=int, default=40)
    rep.add_argument("--json", action="store_true")

    mat = sub.add_parser("matrix", help="Analyze multiple logs and produce consolidated table.")
    mat.add_argument("--run", action="append", required=True, type=parse_run_arg, help="NAME=PATH")
    mat.add_argument("--bus")
    mat.add_argument("--device")
    mat.add_argument("--bo-b", type=int, default=150_528)
    mat.add_argument("--bi-out", type=int, default=1_008)
    mat.add_argument("--top", type=int, default=40)
    mat.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.cmd == "report":
        data = analyze_log(
            path=args.log,
            bus=args.bus,
            device=args.device,
            bo_b=args.bo_b,
            bi_out=args.bi_out,
        )
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(render_report_text(data, top=args.top))
        return 0 if "error" not in data else 1

    if args.cmd == "matrix":
        reports: Dict[str, Dict[str, object]] = {}
        for name, path in args.run:
            reports[name] = analyze_log(
                path=path,
                bus=args.bus,
                device=args.device,
                bo_b=args.bo_b,
                bi_out=args.bi_out,
            )
        matrix = build_matrix(reports)
        if args.json:
            print(json.dumps(matrix, indent=2, sort_keys=True))
        else:
            print(render_matrix_text(matrix, top=args.top))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
