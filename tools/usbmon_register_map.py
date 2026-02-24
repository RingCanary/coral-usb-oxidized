#!/usr/bin/env python3
"""Extract candidate register-map activity from usbmon logs.

Focuses on control transfers used by EdgeTPU USB runtime:
- vendor 64-bit read/write: bmRequestType c0/40, bRequest 00, wLength 0008
- vendor 32-bit read/write: bmRequestType c0/40, bRequest 01, wLength 0004

For vendor control requests, the full 32-bit CSR offset is:
  full_offset = (wIndex << 16) | wValue
"""

from __future__ import annotations

import argparse
import json
import re
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


HEX_WORD_RE = re.compile(r"^[0-9a-fA-F]{8}$")


# Curated register names from publicly available DarwiNN/Beagle maps:
# - google-coral/libedgetpu (usb_ml_commands, usb_driver)
# - ricardodeazambuja/libredgetpu (generated beagle register map)
KNOWN_REGISTER_NAMES = {
    0x00044018: "scalarCoreRunControl",
    0x00048788: "tileconfig0",
    0x0001A30C: "scu_ctrl_0",
    0x00048528: "output_actv_queue_base",
    0x00048540: "output_actv_queue_tail",
    0x00048550: "output_actv_queue_completed_head",
    0x00048590: "instruction_queue_base",
    0x000485A8: "instruction_queue_tail",
    0x000485B0: "instruction_queue_fetched_head",
    0x000485B8: "instruction_queue_completed_head",
    0x000485F8: "input_actv_queue_base",
    0x00048610: "input_actv_queue_tail",
    0x00048620: "input_actv_queue_completed_head",
    0x00048660: "param_queue_base",
    0x00048678: "param_queue_tail",
    0x00048688: "param_queue_completed_head",
}


def parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


def parse_hex_u16(value: str) -> Optional[int]:
    try:
        iv = int(value, 16)
    except ValueError:
        return None
    if iv < 0 or iv > 0xFFFF:
        return None
    return iv


def full_offset_from_setup(wval: str, widx: str) -> Optional[int]:
    lo = parse_hex_u16(wval)
    hi = parse_hex_u16(widx)
    if lo is None or hi is None:
        return None
    return (hi << 16) | lo


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


def parse_payload_words(tokens: List[str]) -> List[str]:
    if "=" not in tokens:
        return []
    idx = tokens.index("=")
    out: List[str] = []
    for tok in tokens[idx + 1 :]:
        t = tok.strip().lower()
        if HEX_WORD_RE.match(t):
            out.append(t)
    return out


def decode_write_payload(op: str, payload_words: List[str]) -> Optional[int]:
    if op not in {"write32", "write64"}:
        return None
    want_bytes = 4 if op == "write32" else 8
    raw = bytearray()
    for word in payload_words:
        try:
            raw.extend(bytes.fromhex(word))
        except ValueError:
            return None
        if len(raw) >= want_bytes:
            break
    if len(raw) < want_bytes:
        return None
    return int.from_bytes(bytes(raw[:want_bytes]), byteorder="little", signed=False)


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
            "addr_full": "",
            "reg_name": "",
        }

    full = full_offset_from_setup(wval=wval, widx=widx)
    reg_name = KNOWN_REGISTER_NAMES.get(full, "")
    addr_full = f"{full:08x}" if full is not None else ""

    # Vendor control path used by EdgeTPU runtime.
    # bRequest 0 -> 64-bit register access, expected wLength=0008
    # bRequest 1 -> 32-bit register access, expected wLength=0004
    if bm in {"40", "c0"} and breq in {"00", "01"}:
        if breq == "00":
            op_base = "read64" if bm == "c0" else "write64"
            expect_wlen = "0008"
        else:
            op_base = "read32" if bm == "c0" else "write32"
            expect_wlen = "0004"
        op = op_base if wlen == expect_wlen else f"{op_base}_len_mismatch"
        return {
            "kind": "vendor",
            "bm": bm,
            "breq": breq,
            "wval": wval,
            "widx": widx,
            "wlen": wlen,
            "op": op,
            "addr": wval,
            "addr_full": addr_full,
            "reg_name": reg_name,
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
        "addr_full": addr_full,
        "reg_name": reg_name,
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
    by_addr_op_low16 = Counter()
    by_addr_op_low16_phase = Counter()
    by_reg_name_op = Counter()
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
        addr_low16 = c.get("addr", "????")
        addr_full = c.get("addr_full") or addr_low16
        reg_name = c.get("reg_name", "")
        phase = phase_for_ts(e.ts, first_bo_b=first_bo_b, last_bi_out=last_bi_out)
        by_op[op] += 1
        by_op_phase[(op, phase)] += 1
        by_addr_op[(addr_full, op)] += 1
        by_addr_op_phase[(addr_full, op, phase)] += 1
        by_addr_op_low16[(addr_low16, op)] += 1
        by_addr_op_low16_phase[(addr_low16, op, phase)] += 1
        if reg_name:
            by_reg_name_op[(reg_name, op)] += 1

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
        "by_addr_op_low16": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_addr_op_low16.items())},
        "by_addr_op_low16_phase": {
            f"{k[0]}::{k[1]}::{k[2]}": v for k, v in sorted(by_addr_op_low16_phase.items())
        },
        "by_reg_name_op": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_reg_name_op.items())},
        "standard_signatures": dict(by_standard_sig),
    }


def extract_vendor_sequence(
    path: Path,
    bus: Optional[str],
    device: Optional[str],
    bo_b: int,
    bi_out: int,
    phases: Optional[List[str]] = None,
    writes_only: bool = False,
    known_only: bool = False,
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
    phase_allow = set(phases or ["all"])

    seq: List[Dict[str, object]] = []
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

        c = classify_control(e.tokens)
        if c.get("kind") != "vendor":
            continue
        op = c.get("op", "")
        if writes_only and not str(op).startswith("write"):
            continue
        phase = phase_for_ts(e.ts, first_bo_b=first_bo_b, last_bi_out=last_bi_out)
        if "all" not in phase_allow and phase not in phase_allow:
            continue
        reg_name = c.get("reg_name", "")
        if known_only and not reg_name:
            continue

        payload_words = parse_payload_words(e.tokens)
        value = decode_write_payload(str(op), payload_words)
        row: Dict[str, object] = {
            "ts": e.ts,
            "phase": phase,
            "transfer": e.transfer,
            "op": op,
            "bm": c.get("bm"),
            "breq": c.get("breq"),
            "wlen": c.get("wlen"),
            "addr_low16": c.get("addr"),
            "addr_full": c.get("addr_full", c.get("addr")),
            "reg_name": reg_name,
            "payload_words": payload_words,
        }
        if value is not None:
            row["value_u64"] = int(value)
            row["value_hex"] = f"0x{int(value):x}"
        seq.append(row)

    return {
        "log_path": str(path),
        "bus": bus,
        "device": dev,
        "line_count": len(entries),
        "loop_window": {"first_bo_b_ts": first_bo_b, "last_bi_out_ts": last_bi_out},
        "sequence_count": len(seq),
        "phases": sorted(phase_allow),
        "sequence": seq,
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
        reg_name = ""
        try:
            reg_name = KNOWN_REGISTER_NAMES.get(int(addr, 16), "")
        except ValueError:
            reg_name = ""
        phase_counts = []
        for phase in ("setup_only", "pre_loop", "loop", "post_loop"):
            k = f"{addr}::{op}::{phase}"
            v = data["by_addr_op_phase"].get(k, 0)
            if v:
                phase_counts.append(f"{phase}:{v}")
        phase_s = ",".join(phase_counts) if phase_counts else "-"
        suffix = f" name={reg_name}" if reg_name else ""
        lines.append(f"  {addr} {op} count={count} phases={phase_s}{suffix}")
    if data.get("by_reg_name_op"):
        lines.append("named_register_ops=" + json.dumps(data["by_reg_name_op"], sort_keys=True))
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


def render_sequence_text(data: Dict[str, object], top: int) -> str:
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
    lines.append("phases=" + ",".join(data.get("phases", [])))
    if "writes_only" in data:
        lines.append(f"writes_only={data['writes_only']}")
    if "known_only" in data:
        lines.append(f"known_only={data['known_only']}")
    lines.append(f"sequence_count={data['sequence_count']}")
    lines.append("sequence_preview:")
    for row in data["sequence"][:top]:
        label = f"{row['ts']} {row['phase']} {row['transfer']} {row['op']} {row['addr_full']}"
        if row.get("reg_name"):
            label += f" ({row['reg_name']})"
        if row.get("value_hex"):
            label += f" value={row['value_hex']}"
        lines.append("  " + label)
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

    seq = sub.add_parser("sequence", help="Extract ordered vendor control sequence for replay.")
    seq.add_argument("log", type=Path)
    seq.add_argument("--bus")
    seq.add_argument("--device")
    seq.add_argument("--bo-b", type=int, default=150_528)
    seq.add_argument("--bi-out", type=int, default=1_008)
    seq.add_argument(
        "--phase",
        action="append",
        default=[],
        choices=["setup_only", "pre_loop", "loop", "post_loop", "all"],
        help="Filter sequence by phase (default: all).",
    )
    seq.add_argument(
        "--writes-only",
        action="store_true",
        help="Keep only write32/write64 operations.",
    )
    seq.add_argument(
        "--known-only",
        action="store_true",
        help="Keep only entries with known register-name mapping.",
    )
    seq.add_argument("--top", type=int, default=80)
    seq.add_argument("--json", action="store_true")

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

    if args.cmd == "sequence":
        phases = args.phase if args.phase else ["all"]
        data = extract_vendor_sequence(
            path=args.log,
            bus=args.bus,
            device=args.device,
            bo_b=args.bo_b,
            bi_out=args.bi_out,
            phases=phases,
            writes_only=args.writes_only,
            known_only=args.known_only,
        )
        data["writes_only"] = bool(args.writes_only)
        data["known_only"] = bool(args.known_only)
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(render_sequence_text(data, top=args.top))
        return 0 if "error" not in data else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
