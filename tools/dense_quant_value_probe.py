#!/usr/bin/env python3
"""Probe Dense single-hot quantization value mapping into compiled payload bytes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _utc_stamp() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _mode_byte(blob: bytes) -> int:
    if not blob:
        return 0
    value, _ = Counter(blob).most_common(1)[0]
    return int(value)


def dense_256_param_offset(row: int, col: int) -> int:
    if not (0 <= row < 256 and 0 <= col < 256):
        raise ValueError(f"row/col out of range for 256x256: row={row} col={col}")
    return (col // 64) * 16384 + (row // 64) * 4096 + ((row % 64) // 4) * 256 + (col % 64) * 4 + (row % 4)


def _resolve_compiler(path_opt: str | None, repo_root: Path) -> Path:
    if path_opt:
        compiler = Path(path_opt)
        if not compiler.exists():
            raise FileNotFoundError(f"--compiler path not found: {compiler}")
        return compiler
    from_path = shutil.which("edgetpu_compiler")
    if from_path:
        return Path(from_path)
    _run([str(repo_root / "tools" / "bootstrap_edgetpu_compiler.sh"), "install"], cwd=repo_root)
    fallback = Path.home() / ".local" / "bin" / "edgetpu_compiler"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("could not resolve edgetpu_compiler")


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


def _pick_parameter_region(inspect_json: Dict) -> Tuple[int, int]:
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
                first_nonempty = (int(preg["start"]), int(preg["end"]))
            if exe.get("type_name") == "PARAMETER_CACHING":
                return int(preg["start"]), int(preg["end"])
    if first_nonempty is None:
        raise RuntimeError("no non-empty parameter region found")
    return first_nonempty


def _parse_weight_tensor(quant_model: Path, repo_root: Path) -> Dict:
    code = r"""
import json
from pathlib import Path
from tensorflow.lite.python import schema_py_generated as schema

model_path = Path(__import__('sys').argv[1])
buf = model_path.read_bytes()
m = schema.Model.GetRootAsModel(buf, 0)
s = m.Subgraphs(0)

def buffer_bytes(buffer_idx: int) -> bytes:
    b = m.Buffers(buffer_idx)
    n = b.DataLength()
    return bytes(b.Data(i) for i in range(n))

candidates = []
for i in range(s.TensorsLength()):
    t = s.Tensors(i)
    name = t.Name().decode('utf-8', errors='replace') if t.Name() else ''
    shape = [t.Shape(j) for j in range(t.ShapeLength())]
    data = buffer_bytes(t.Buffer())
    q = t.Quantization()
    scales = [q.Scale(j) for j in range(q.ScaleLength())] if q else []
    zps = [q.ZeroPoint(j) for j in range(q.ZeroPointLength())] if q else []
    candidates.append({
        'index': i,
        'name': name,
        'type': int(t.Type()),
        'shape': shape,
        'buffer_size': len(data),
        'buffer': list(data),
        'scales': scales,
        'zero_points': zps,
    })

# Prefer the pseudo_qconst tensor with 2D shape 256x256 and full payload.
pick = None
for c in candidates:
    if c['name'].startswith('tfl.pseudo_qconst') and c['shape'] == [256, 256] and c['buffer_size'] == 65536:
        pick = c
        break
if pick is None:
    for c in candidates:
        if c['shape'] == [256, 256] and c['buffer_size'] == 65536:
            pick = c
            break
if pick is None:
    raise SystemExit('could not identify 256x256 weight tensor')

print(json.dumps(pick))
"""
    proc = subprocess.run(
        [
            "uv",
            "run",
            "--python",
            "3.9",
            "--with",
            "tensorflow-cpu==2.10.1",
            "--with",
            "numpy==1.23.5",
            "python",
            "-c",
            code,
            str(quant_model),
        ],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"weight tensor parse failed for {quant_model}:\n{proc.stderr}\n{proc.stdout}")
    return json.loads(proc.stdout)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe mapping between float single-hot weight value and compiled parameter byte value.",
    )
    p.add_argument("--out-dir", help="Output directory. Default: traces/dense-quant-value-probe-<utc>.")
    p.add_argument("--python-version", default="3.9")
    p.add_argument("--tf-version", default="2.10.1")
    p.add_argument("--numpy-version", default="1.23.5")
    p.add_argument("--compiler", help="Path to edgetpu_compiler.")
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--col", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rep-samples", type=int, default=256)
    p.add_argument("--rep-range", type=float, default=1.0)
    p.add_argument("--value", action="append", type=float, default=[], help="Probe float value (repeatable).")
    return p


def _value_tag(v: float) -> str:
    s = f"{v:+.6f}"
    s = s.replace("+", "p").replace("-", "m").replace(".", "d")
    return s


def _clamp_u8(x: int) -> int:
    return max(0, min(255, x))


def _clamp_i8(x: int) -> int:
    return max(-128, min(127, x))


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "traces" / f"dense-quant-value-probe-{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.value:
        args.value = [-1.0, -0.75, -0.5, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]

    compiler = _resolve_compiler(args.compiler, repo_root)
    _run([str(compiler), "--version"], cwd=repo_root)
    _run(["uv", "python", "install", args.python_version], cwd=repo_root)

    byte_offset = dense_256_param_offset(args.row, args.col)
    records: List[Dict] = []

    for value in args.value:
        tag = _value_tag(value)
        quant_model = out_dir / f"dense_hot_r{args.row:03d}_c{args.col:03d}_v{tag}_quant.tflite"
        quant_meta = out_dir / f"dense_hot_r{args.row:03d}_c{args.col:03d}_v{tag}_quant.metadata.json"
        compile_log = out_dir / f"dense_hot_r{args.row:03d}_c{args.col:03d}_v{tag}_compile.log"

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
            "256",
            "--output-dim",
            "256",
            "--init-mode",
            "single_hot",
            "--hot-row",
            str(args.row),
            "--hot-col",
            str(args.col),
            "--hot-value",
            str(value),
            "--seed",
            str(args.seed),
            "--rep-samples",
            str(args.rep_samples),
            "--rep-range",
            str(args.rep_range),
        ]
        _run(gen_cmd, cwd=repo_root)

        compile_cmd = [str(compiler), "-s", "-o", str(out_dir), str(quant_model)]
        _run(compile_cmd, cwd=repo_root, log_path=compile_log)

        compiled_model = out_dir / f"dense_hot_r{args.row:03d}_c{args.col:03d}_v{tag}_quant_edgetpu.tflite"
        if not compiled_model.exists():
            raise RuntimeError(f"missing compiled model: {compiled_model}")

        # Quantized model weight tensor info
        wt = _parse_weight_tensor(quant_model, repo_root)
        wshape = wt["shape"]
        wbytes = bytes(wt["buffer"])
        if len(wbytes) != 65536 or wshape != [256, 256]:
            raise RuntimeError(f"unexpected weight tensor shape/size for {quant_model}: shape={wshape} size={len(wbytes)}")
        q_bg = _mode_byte(wbytes)
        q_active_idxs = [i for i, b in enumerate(wbytes) if b != q_bg]
        q_active_vals = [int(wbytes[i]) for i in q_active_idxs]
        linear_idx = args.row * 256 + args.col
        q_byte_linear = int(wbytes[linear_idx])

        scales = wt.get("scales", [])
        zero_points = wt.get("zero_points", [])
        expected_signed = None
        expected_quant_u8 = None
        expected_compiled_u8 = None
        if len(scales) == 1 and len(zero_points) == 1 and scales[0] != 0:
            expected_signed = _clamp_i8(int(round(value / float(scales[0])) + int(zero_points[0])))
            expected_quant_u8 = expected_signed & 0xFF
            expected_compiled_u8 = (expected_signed + 128) & 0xFF

        # Compiled payload byte at recovered offset
        inspect_json = _inspect_model(compiled_model, repo_root)
        start, end = _pick_parameter_region(inspect_json)
        payload = compiled_model.read_bytes()[start:end]
        if len(payload) != 65536:
            raise RuntimeError(f"unexpected compiled parameter payload size: {len(payload)}")
        c_bg = _mode_byte(payload)
        c_byte_offset = int(payload[byte_offset])

        rec = {
            "value": value,
            "tag": tag,
            "row": args.row,
            "col": args.col,
            "byte_offset": byte_offset,
            "quant_model": str(quant_model),
            "compiled_model": str(compiled_model),
            "compile_log": str(compile_log),
            "weight_quant": {
                "tensor_name": wt["name"],
                "tensor_type": wt.get("type"),
                "shape": wshape,
                "scales": scales,
                "zero_points": zero_points,
                "background_byte": q_bg,
                "active_indices": q_active_idxs,
                "active_values": q_active_vals,
                "byte_at_linear_index": q_byte_linear,
            },
            "compiled_payload": {
                "region_start": start,
                "region_end": end,
                "background_byte": c_bg,
                "byte_at_offset": c_byte_offset,
            },
            "expected_signed_i8_from_scale": expected_signed,
            "expected_quant_u8_from_scale": expected_quant_u8,
            "expected_compiled_u8_from_scale": expected_compiled_u8,
            "delta_quant_minus_expected_quant": (q_byte_linear - expected_quant_u8)
            if expected_quant_u8 is not None
            else None,
            "delta_compiled_minus_expected_compiled": (c_byte_offset - expected_compiled_u8)
            if expected_compiled_u8 is not None
            else None,
            "delta_compiled_minus_quant": c_byte_offset - q_byte_linear,
        }
        records.append(rec)

    report = {
        "tool": "dense_quant_value_probe.py",
        "generated_at_utc": dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat(),
        "config": {
            "out_dir": str(out_dir),
            "python_version": args.python_version,
            "tf_version": args.tf_version,
            "numpy_version": args.numpy_version,
            "compiler": str(compiler),
            "row": args.row,
            "col": args.col,
            "byte_offset": byte_offset,
            "seed": args.seed,
            "rep_samples": args.rep_samples,
            "rep_range": args.rep_range,
            "values": args.value,
        },
        "records": records,
    }

    json_path = out_dir / "value_probe.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    txt_lines = []
    txt_lines.append(f"tool={report['tool']}")
    txt_lines.append(f"generated_at_utc={report['generated_at_utc']}")
    txt_lines.append(
        f"row={args.row} col={args.col} byte_offset={byte_offset} "
        f"python={args.python_version} tf={args.tf_version} numpy={args.numpy_version}"
    )
    txt_lines.append("")
    txt_lines.append(
        "value | exp_i8 | exp_q_u8 | exp_c_u8 | quant_u8 | compiled_u8 | "
        "d(q-expq) | d(c-expc) | d(c-q) | scale | zp"
    )
    for rec in records:
        scales = rec["weight_quant"]["scales"]
        zps = rec["weight_quant"]["zero_points"]
        scale = scales[0] if len(scales) == 1 else None
        zp = zps[0] if len(zps) == 1 else None
        txt_lines.append(
            f"{rec['value']:>7.4f} | "
            f"{str(rec['expected_signed_i8_from_scale']):>6} | "
            f"{str(rec['expected_quant_u8_from_scale']):>8} | "
            f"{str(rec['expected_compiled_u8_from_scale']):>8} | "
            f"{rec['weight_quant']['byte_at_linear_index']:>8} | "
            f"{rec['compiled_payload']['byte_at_offset']:>11} | "
            f"{str(rec['delta_quant_minus_expected_quant']):>9} | "
            f"{str(rec['delta_compiled_minus_expected_compiled']):>9} | "
            f"{rec['delta_compiled_minus_quant']:>6} | "
            f"{str(scale):>8} | "
            f"{str(zp):>3}"
        )
    txt_path = out_dir / "value_probe.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
