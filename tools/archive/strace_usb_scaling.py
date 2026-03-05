#!/usr/bin/env python3
"""Summarize USBDEVFS ioctl scaling from strace summary files.

Scans run directories produced by tools/usb_syscall_trace.sh and reports:
- model, total invokes, submit/reap counts
- simple linear fit for submit/reap vs invokes
- per-point residuals to highlight outliers
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Row:
    run: str
    model: str
    runs: int
    warmup: int
    total: int
    submiturb: int
    reapurb: int


COMMAND_RE = re.compile(r"\.tflite(?:\\\s+|\s+)(\d+)(?:\\\s+|\s+)(\d+)$")
SUBMIT_RE = re.compile(r"\bUSBDEVFS_SUBMITURB\s+(\d+)")
REAP_RE = re.compile(r"\bUSBDEVFS_REAPURBNDELAY\s+(\d+)")


def parse_command_line(command: str) -> Optional[Tuple[int, int]]:
    m = COMMAND_RE.search(command)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_summary(path: Path) -> Optional[Tuple[int, int, int, str]]:
    command: Optional[str] = None
    submit: Optional[int] = None
    reap: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("command="):
                command = line[len("command=") :].strip()
                continue
            m = SUBMIT_RE.search(line)
            if m:
                submit = int(m.group(1))
                continue
            m = REAP_RE.search(line)
            if m:
                reap = int(m.group(1))
                continue

    if command is None or submit is None or reap is None:
        return None
    parsed = parse_command_line(command)
    if parsed is None:
        return None
    runs, warm = parsed
    return runs, warm, submit, command


def parse_result_model(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "RESULT model=" not in line:
                    continue
                m = re.search(r"RESULT model=([^ ]+)", line)
                if m:
                    return m.group(1)
    except OSError:
        return None
    return None


def load_rows(root: Path, include_prefixes: Optional[List[str]]) -> List[Row]:
    rows: List[Row] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        if include_prefixes is not None and not any(run_name.startswith(p) for p in include_prefixes):
            continue
        summaries = sorted(run_dir.glob("*.summary.txt"))
        if not summaries:
            continue
        parsed = parse_summary(summaries[0])
        if parsed is None:
            continue
        runs, warm, submit, _ = parsed
        reaps = None
        with summaries[0].open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                m = REAP_RE.search(line)
                if m:
                    reaps = int(m.group(1))
        if reaps is None:
            continue

        model = None
        logs = sorted(run_dir.glob("*.log"))
        for log in logs:
            model = parse_result_model(log)
            if model:
                break
        if model is None:
            model = "(unknown)"
        rows.append(
            Row(
                run=run_name,
                model=model,
                runs=runs,
                warmup=warm,
                total=runs + warm,
                submiturb=submit,
                reapurb=reaps,
            )
        )
    return rows


def linear_fit(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, ys[0]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0, y_mean
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def grouped(rows: Iterable[Row]) -> Dict[str, List[Row]]:
    out: Dict[str, List[Row]] = {}
    for r in rows:
        out.setdefault(r.model, []).append(r)
    for model in out:
        out[model].sort(key=lambda r: (r.total, r.run))
    return out


def render(rows: List[Row]) -> str:
    if not rows:
        return "No matching run summaries found."
    lines: List[str] = []
    by_model = grouped(rows)
    for model, rs in by_model.items():
        lines.append(f"model={model}")
        lines.append("| run | total | submiturb | reapurb |")
        lines.append("|---|---:|---:|---:|")
        for r in rs:
            lines.append(f"| {r.run} | {r.total} | {r.submiturb} | {r.reapurb} |")
        xs = [float(r.total) for r in rs]
        ys_submit = [float(r.submiturb) for r in rs]
        ys_reap = [float(r.reapurb) for r in rs]
        s_slope, s_int = linear_fit(xs, ys_submit)
        r_slope, r_int = linear_fit(xs, ys_reap)
        lines.append(f"fit_submiturb = {s_slope:.6f} * invokes + {s_int:.6f}")
        lines.append(f"fit_reapurb   = {r_slope:.6f} * invokes + {r_int:.6f}")
        lines.append("residuals:")
        for r in rs:
            pred_submit = s_slope * r.total + s_int
            pred_reap = r_slope * r.total + r_int
            lines.append(
                "  {}: submit_delta={:+.3f} reap_delta={:+.3f}".format(
                    r.run, r.submiturb - pred_submit, r.reapurb - pred_reap
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize USBDEVFS ioctl scaling by model from strace summaries.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("traces/re-matrix-20260221T092342Z"),
        help="Directory containing R*/ run folders.",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        help="Include only runs whose folder starts with this prefix (can repeat).",
    )
    args = parser.parse_args()

    rows = load_rows(root=args.root, include_prefixes=args.include_prefix)
    print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
