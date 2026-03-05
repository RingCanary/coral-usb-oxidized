#!/usr/bin/env python3
"""Recover Dense(256x256) parameter layout via single-hot compiled probes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _utc_stamp() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(cmd: Sequence[str], *, cwd: Path | None = None, log_path: Path | None = None) -> None:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "$ " + " ".join(cmd) + "\n\n" + proc.stdout + ("\n" + proc.stderr if proc.stderr else ""),
            encoding="utf-8",
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _parse_coord(text: str) -> Tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"invalid coord '{text}' (expected row,col)")
    try:
        row = int(parts[0])
        col = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid coord '{text}': {exc}") from exc
    return row, col


def _coord_tag(row: int, col: int) -> str:
    return f"r{row:03d}_c{col:03d}"


def _resolve_compiler(path_opt: str | None, repo_root: Path) -> Path:
    if path_opt:
        compiler = Path(path_opt)
        if not compiler.exists():
            raise FileNotFoundError(f"--compiler path not found: {compiler}")
        return compiler

    from_path = shutil.which("edgetpu_compiler")
    if from_path:
        return Path(from_path)

    bootstrap = repo_root / "tools" / "bootstrap_edgetpu_compiler.sh"
    _run([str(bootstrap), "install"], cwd=repo_root)
    fallback = Path.home() / ".local" / "bin" / "edgetpu_compiler"
    if not fallback.exists():
        raise FileNotFoundError("could not resolve edgetpu_compiler after bootstrap")
    return fallback


def _pick_parameter_region(inspect_json: Dict) -> Tuple[int, int, Dict]:
    first_nonempty = None
    for pkg in inspect_json.get("packages", []):
        for exe in pkg.get("executables", []):
            preg = exe.get("parameter_region")
            if not preg:
                continue
            size = int(preg.get("size", 0))
            if size <= 0:
                continue
            if first_nonempty is None:
                first_nonempty = (int(preg["start"]), int(preg["end"]), exe)
            if exe.get("type_name") == "PARAMETER_CACHING":
                return int(preg["start"]), int(preg["end"]), exe
    if first_nonempty is None:
        raise RuntimeError("no non-empty parameter_region found in inspect JSON")
    return first_nonempty


def _inspect_model(model_path: Path, repo_root: Path) -> Dict:
    proc = subprocess.run(
        [
            "python3",
            str(repo_root / "tools" / "tensorizer_patch_edgetpu.py"),
            "inspect",
            "--json",
            str(model_path),
        ],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"inspect failed for {model_path}:\n{proc.stderr}\n{proc.stdout}")
    return json.loads(proc.stdout)


def _extract_blob(model_path: Path, start: int, end: int) -> bytes:
    data = model_path.read_bytes()
    if not (0 <= start <= end <= len(data)):
        raise RuntimeError(f"invalid parameter slice [{start},{end}) for {model_path} size={len(data)}")
    return data[start:end]


def _mode_byte(blob: bytes) -> int:
    if not blob:
        return 0
    value, _ = Counter(blob).most_common(1)[0]
    return int(value)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run single-hot Dense probe matrix and recover parameter byte offset behavior.",
    )
    p.add_argument("--out-dir", help="Output directory. Default: traces/dense-layout-probe-<utc>.")
    p.add_argument("--python-version", default="3.9")
    p.add_argument("--tf-version", default="2.10.1")
    p.add_argument("--numpy-version", default="1.23.5")
    p.add_argument("--compiler", help="Path to edgetpu_compiler.")
    p.add_argument("--input-dim", type=int, default=256)
    p.add_argument("--output-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rep-samples", type=int, default=256)
    p.add_argument("--rep-range", type=float, default=1.0)
    p.add_argument("--hot-value", type=float, default=1.0)
    p.add_argument("--reference", type=_parse_coord, default=(0, 0), help="Reference probe row,col.")
    p.add_argument(
        "--probe",
        action="append",
        type=_parse_coord,
        default=[],
        help="Probe row,col (repeatable). Defaults to a built-in spread.",
    )
    p.add_argument("--max-preview", type=int, default=32, help="Max offset preview entries in text report.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "traces" / f"dense-layout-probe-{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.probe:
        args.probe = [(0, 1), (1, 0), (1, 1), (16, 16), (127, 127), (255, 255)]
    if args.reference not in args.probe:
        probes: List[Tuple[int, int]] = [args.reference] + args.probe
    else:
        probes = list(args.probe)

    # De-dup while preserving order.
    seen = set()
    ordered_probes = []
    for probe in probes:
        if probe not in seen:
            seen.add(probe)
            ordered_probes.append(probe)
    probes = ordered_probes

    compiler = _resolve_compiler(args.compiler, repo_root)
    subprocess.run([str(compiler), "--version"], check=True, cwd=str(repo_root))

    # Ensure uv-managed Python exists before loop.
    _run(["uv", "python", "install", args.python_version], cwd=repo_root)

    records = []
    probe_root = out_dir / "probe_models"
    blob_root = out_dir / "blobs"
    probe_root.mkdir(parents=True, exist_ok=True)
    blob_root.mkdir(parents=True, exist_ok=True)

    for row, col in probes:
        tag = _coord_tag(row, col)
        quant_model = probe_root / f"dense_{tag}_quant.tflite"
        quant_meta = probe_root / f"dense_{tag}_quant.metadata.json"
        compile_log = probe_root / f"dense_{tag}_compile.log"

        gen_cmd = [
            "uv",
            "run",
            "--python",
            args.python_version,
            "--with",
            f"tensorflow-cpu=={args.tf_version}",
            "--with",
            f"numpy=={args.numpy_version}",
            str(repo_root / "tools" / "generate_dense_quant_tflite.py"),
            "--output",
            str(quant_model),
            "--metadata-out",
            str(quant_meta),
            "--input-dim",
            str(args.input_dim),
            "--output-dim",
            str(args.output_dim),
            "--init-mode",
            "single_hot",
            "--hot-row",
            str(row),
            "--hot-col",
            str(col),
            "--hot-value",
            str(args.hot_value),
            "--seed",
            str(args.seed),
            "--rep-samples",
            str(args.rep_samples),
            "--rep-range",
            str(args.rep_range),
        ]
        _run(gen_cmd, cwd=repo_root)

        compile_cmd = [str(compiler), "-s", "-o", str(probe_root), str(quant_model)]
        _run(compile_cmd, cwd=repo_root, log_path=compile_log)

        compiled_model = probe_root / f"dense_{tag}_quant_edgetpu.tflite"
        if not compiled_model.exists():
            raise RuntimeError(f"missing compiled model: {compiled_model}")

        inspect_json = _inspect_model(compiled_model, repo_root)
        start, end, exe_info = _pick_parameter_region(inspect_json)
        blob = _extract_blob(compiled_model, start, end)
        blob_path = blob_root / f"dense_{tag}.params.bin"
        blob_path.write_bytes(blob)

        background = _mode_byte(blob)
        active_offsets = [idx for idx, b in enumerate(blob) if b != background]
        active_values = [int(blob[idx]) for idx in active_offsets]
        records.append(
            {
                "row": row,
                "col": col,
                "tag": tag,
                "quant_model": str(quant_model),
                "compiled_model": str(compiled_model),
                "compile_log": str(compile_log),
                "parameter_region": {"start": start, "end": end, "size": end - start},
                "executable": {
                    "index": exe_info.get("index"),
                    "type_name": exe_info.get("type_name"),
                    "type_value": exe_info.get("type_value"),
                },
                "blob_path": str(blob_path),
                "blob_sha256": _sha256(blob),
                "background_byte": background,
                "active_offsets": active_offsets,
                "active_values": active_values,
                "active_count": len(active_offsets),
            }
        )

    # Build diffs against reference probe.
    ref = next((r for r in records if (r["row"], r["col"]) == args.reference), None)
    if ref is None:
        raise RuntimeError(f"reference probe not found in records: {args.reference}")
    ref_blob = Path(ref["blob_path"]).read_bytes()
    ref_bg = int(ref["background_byte"])

    for rec in records:
        blob = Path(rec["blob_path"]).read_bytes()
        if len(blob) != len(ref_blob):
            raise RuntimeError(
                f"blob size mismatch: ref={len(ref_blob)} probe={len(blob)} for {rec['tag']}"
            )
        changed = [i for i, (a, b) in enumerate(zip(ref_blob, blob)) if a != b]
        added = [i for i in changed if ref_blob[i] == ref_bg and blob[i] != ref_bg]
        removed = [i for i in changed if ref_blob[i] != ref_bg and blob[i] == ref_bg]
        rec["diff_vs_reference"] = {
            "reference": _coord_tag(args.reference[0], args.reference[1]),
            "changed_offsets": changed,
            "changed_count": len(changed),
            "added_offsets": added,
            "added_count": len(added),
            "removed_offsets": removed,
            "removed_count": len(removed),
        }
        if len(added) == 1:
            rec["mapping_candidate_offset"] = added[0]
        elif len(rec["active_offsets"]) == 1:
            rec["mapping_candidate_offset"] = rec["active_offsets"][0]
        else:
            rec["mapping_candidate_offset"] = None

    report = {
        "tool": "dense_layout_probe.py",
        "generated_at_utc": dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat(),
        "config": {
            "out_dir": str(out_dir),
            "python_version": args.python_version,
            "tf_version": args.tf_version,
            "numpy_version": args.numpy_version,
            "compiler": str(compiler),
            "input_dim": args.input_dim,
            "output_dim": args.output_dim,
            "seed": args.seed,
            "rep_samples": args.rep_samples,
            "rep_range": args.rep_range,
            "hot_value": args.hot_value,
            "reference": {"row": args.reference[0], "col": args.reference[1]},
        },
        "records": records,
    }

    json_path = out_dir / "layout_probe.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    txt_path = out_dir / "layout_probe.txt"
    lines = []
    lines.append(f"tool={report['tool']}")
    lines.append(f"generated_at_utc={report['generated_at_utc']}")
    lines.append(
        f"reference={_coord_tag(args.reference[0], args.reference[1])} "
        f"python={args.python_version} tf={args.tf_version} numpy={args.numpy_version}"
    )
    lines.append("")
    for rec in records:
        lines.append(f"[probe {rec['tag']}] row={rec['row']} col={rec['col']}")
        lines.append(
            f"  blob_size={rec['parameter_region']['size']} "
            f"bg={rec['background_byte']} active_count={rec['active_count']}"
        )
        if rec["active_values"]:
            lines.append(f"  active_values={rec['active_values'][: args.max_preview]}")
        lines.append(
            f"  diff.changed={rec['diff_vs_reference']['changed_count']} "
            f"added={rec['diff_vs_reference']['added_count']} "
            f"removed={rec['diff_vs_reference']['removed_count']}"
        )
        changed_preview = rec["diff_vs_reference"]["changed_offsets"][: args.max_preview]
        lines.append(f"  changed_preview={changed_preview}")
        if rec["mapping_candidate_offset"] is not None:
            lines.append(f"  mapping_candidate_offset={rec['mapping_candidate_offset']}")
        else:
            lines.append("  mapping_candidate_offset=NONE")
        lines.append("")
    txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
