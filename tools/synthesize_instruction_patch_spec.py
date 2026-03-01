#!/usr/bin/env python3
"""Synthesize replay instruction patch specs using formula-based byte prediction.

This tool predicts target instruction bytes from dimension-labeled fit points.
Compared to endpoint-only interpolation, it supports non-linear tile formulas
(e.g. floor((tiles^2)/64) patterns observed in parameter-caching streams).

Patch spec format (for `rusb_serialized_exec_replay --instruction-patch-spec`):
  <payload_len> <offset> <value>
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import parse_edgetpu_executable as pe


@dataclass(frozen=True)
class FitPoint:
    dim: int
    eo_path: Path
    pc_path: Path


@dataclass(frozen=True)
class StreamConfig:
    name: str
    base_path: Path
    target_path: Optional[Path]


@dataclass(frozen=True)
class ModelPrediction:
    family: str
    params: Dict[str, float]
    predicted: int
    loo_exact_ratio: float
    loo_mae: float
    in_sample_mae: float
    complexity: int


def _read_chunk_bytes(path: Path, chunk_index: int) -> bytes:
    blob = path.read_bytes()
    root = pe._parse_root_table(blob, 0, file_identifier=None)
    tables = pe._read_vector_table_field(root, 5)
    if chunk_index < 0 or chunk_index >= len(tables):
        raise SystemExit(
            f"chunk index {chunk_index} out of range for {path} (count={len(tables)})"
        )
    return pe._read_vector_bytes_field(tables[chunk_index], 0)


def _parse_fit_point(raw: str) -> FitPoint:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(f"invalid --fit-point '{raw}' (expected DIM:EO_PATH:PC_PATH)")
    try:
        dim = int(parts[0])
    except ValueError as exc:
        raise SystemExit(f"invalid DIM in --fit-point '{raw}': {exc}") from exc
    eo = Path(parts[1])
    pc = Path(parts[2])
    if not eo.is_file():
        raise SystemExit(f"fit-point EO file not found: {eo}")
    if not pc.is_file():
        raise SystemExit(f"fit-point PC file not found: {pc}")
    return FitPoint(dim=dim, eo_path=eo, pc_path=pc)


def _stream_path(fp: FitPoint, stream_name: str) -> Path:
    return fp.eo_path if stream_name == "eo" else fp.pc_path


def _ensure_base_target(base_arg: Optional[str], target_arg: Optional[str], stream_name: str) -> StreamConfig:
    target_path = Path(target_arg) if target_arg else None
    base_path = Path(base_arg) if base_arg else (target_path if target_path else None)
    if base_path is None:
        raise SystemExit(
            f"stream '{stream_name}' needs --base-{stream_name} when --target-{stream_name} is omitted"
        )
    if not base_path.is_file():
        raise SystemExit(f"missing base file for stream '{stream_name}': {base_path}")
    if target_path is not None and not target_path.is_file():
        raise SystemExit(f"missing target file for stream '{stream_name}': {target_path}")
    return StreamConfig(name=stream_name, base_path=base_path, target_path=target_path)


def _build_fit_points(args: argparse.Namespace) -> List[FitPoint]:
    fit: Dict[int, FitPoint] = {}

    for raw in args.fit_point:
        fp = _parse_fit_point(raw)
        fit[fp.dim] = fp

    if not fit:
        required = [
            args.low_dim,
            args.high_dim,
            args.low_eo,
            args.high_eo,
            args.low_pc,
            args.high_pc,
        ]
        if any(v is None for v in required):
            raise SystemExit(
                "either provide --fit-point (repeatable) or provide legacy --low-* / --high-* arguments"
            )

        low_eo = Path(args.low_eo)
        high_eo = Path(args.high_eo)
        low_pc = Path(args.low_pc)
        high_pc = Path(args.high_pc)
        for p in [low_eo, high_eo, low_pc, high_pc]:
            if not p.is_file():
                raise SystemExit(f"legacy fit file not found: {p}")

        fit[int(args.low_dim)] = FitPoint(int(args.low_dim), low_eo, low_pc)
        fit[int(args.high_dim)] = FitPoint(int(args.high_dim), high_eo, high_pc)

    if len(fit) < 2:
        raise SystemExit("need at least 2 fit points")

    return [fit[d] for d in sorted(fit)]


def _offsets_changed_extrema(
    base_len: int,
    extrema_low: Optional[bytes],
    extrema_high: Optional[bytes],
) -> List[int]:
    if extrema_low is None or extrema_high is None:
        return []
    n = min(base_len, len(extrema_low), len(extrema_high))
    return [i for i in range(n) if extrema_low[i] != extrema_high[i]]


def _offsets_changed_fit_union(base_len: int, fit_payloads: Sequence[Tuple[int, bytes]]) -> List[int]:
    out: List[int] = []
    for off in range(base_len):
        vals = {payload[off] for _, payload in fit_payloads if len(payload) > off}
        if len(vals) > 1:
            out.append(off)
    return out


def _values_for_offset(fit_payloads: Sequence[Tuple[int, bytes]], offset: int) -> List[Tuple[int, int]]:
    vals: List[Tuple[int, int]] = []
    for dim, payload in fit_payloads:
        if offset < len(payload):
            vals.append((dim, payload[offset]))
    return vals


def _tiles(dim: int, tile_size: int) -> int:
    return dim // tile_size


def _fit_const(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float]]:
    if not ys:
        return None
    return (sum(ys) / len(ys),)


def _fit_linear(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float]]:
    n = len(xs)
    if n < 2:
        return None
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = (n * sxx) - (sx * sx)
    if abs(den) < 1e-9:
        return None
    a = ((n * sxy) - (sx * sy)) / den
    b = (sy - a * sx) / n
    return (a, b)


def _solve_3x3(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = 3

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-9:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        p = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= p

        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]

    return [m[i][n] for i in range(n)]


def _fit_quadratic(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    if len(xs) < 3:
        return None

    s1 = float(len(xs))
    sx = sum(xs)
    sx2 = sum(x * x for x in xs)
    sx3 = sum(x * x * x for x in xs)
    sx4 = sum(x * x * x * x for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2y = sum((x * x) * y for x, y in zip(xs, ys))

    sol = _solve_3x3(
        a=[[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, s1]],
        b=[sx2y, sxy, sy],
    )
    if sol is None:
        return None
    return (sol[0], sol[1], sol[2])


def _predict_from_params(model: str, x: float, params: Tuple[float, ...]) -> float:
    if model == "const":
        return params[0]
    if model in ("linear", "tile2div-linear"):
        return params[0] * x + params[1]
    if model == "quadratic":
        return params[0] * (x * x) + params[1] * x + params[2]
    raise ValueError(f"unknown model: {model}")


def _fit_and_predict(
    family: str,
    samples: Sequence[Tuple[int, int]],
    target_dim: int,
    tile_size: int,
) -> Optional[Tuple[int, Dict[str, float], int]]:
    dims = [d for d, _ in samples]
    vals = [float(v) for _, v in samples]

    if family == "const":
        params = _fit_const(dims, vals)
        if params is None:
            return None
        pred = _predict_from_params("const", 0.0, params)
        return (int(round(pred)), {"c": float(params[0])}, 0)

    if family == "tile-linear":
        xs = [float(_tiles(d, tile_size)) for d in dims]
        params = _fit_linear(xs, vals)
        if params is None:
            return None
        xt = float(_tiles(target_dim, tile_size))
        pred = _predict_from_params("linear", xt, params)
        return (int(round(pred)), {"a": float(params[0]), "b": float(params[1])}, 1)

    if family == "tile-quadratic":
        xs = [float(_tiles(d, tile_size)) for d in dims]
        params = _fit_quadratic(xs, vals)
        if params is None:
            return None
        xt = float(_tiles(target_dim, tile_size))
        pred = _predict_from_params("quadratic", xt, params)
        return (
            int(round(pred)),
            {"a": float(params[0]), "b": float(params[1]), "c": float(params[2])},
            2,
        )

    if family == "tile2div-linear":
        candidates = [16, 32, 64, 128, 256]
        best: Optional[Tuple[int, Dict[str, float], int, float]] = None
        for div in candidates:
            xs = [float((_tiles(d, tile_size) ** 2) // div) for d in dims]
            params = _fit_linear(xs, vals)
            if params is None:
                continue
            xt = float((_tiles(target_dim, tile_size) ** 2) // div)
            pred = _predict_from_params("tile2div-linear", xt, params)

            mae = sum(abs(v - _predict_from_params("tile2div-linear", x, params)) for v, x in zip(vals, xs)) / len(vals)
            record = (
                int(round(pred)),
                {"a": float(params[0]), "b": float(params[1]), "div": float(div)},
                2,
                mae,
            )
            if best is None or record[3] < best[3]:
                best = record
        if best is None:
            return None
        return (best[0], best[1], best[2])

    raise SystemExit(f"unsupported model family: {family}")


def _eval_model_loo(
    family: str,
    samples: Sequence[Tuple[int, int]],
    tile_size: int,
    min_loo_points: int,
) -> Tuple[float, float]:
    if len(samples) < min_loo_points:
        return (0.0, float("inf"))

    errors: List[float] = []
    exact = 0
    for i in range(len(samples)):
        hold_dim, hold_val = samples[i]
        train = [samples[j] for j in range(len(samples)) if j != i]
        fitted = _fit_and_predict(
            family=family,
            samples=train,
            target_dim=hold_dim,
            tile_size=tile_size,
        )
        if fitted is None:
            return (0.0, float("inf"))
        pred = max(0, min(255, fitted[0]))
        err = abs(pred - hold_val)
        if err == 0:
            exact += 1
        errors.append(float(err))

    exact_ratio = exact / float(len(samples))
    mae = sum(errors) / len(errors)
    return (exact_ratio, mae)


def _eval_model_insample(family: str, samples: Sequence[Tuple[int, int]], tile_size: int) -> float:
    fitted = _fit_and_predict(
        family=family,
        samples=samples,
        target_dim=samples[0][0],
        tile_size=tile_size,
    )
    if fitted is None:
        return float("inf")

    err_sum = 0.0
    for dim, v in samples:
        f = _fit_and_predict(family=family, samples=samples, target_dim=dim, tile_size=tile_size)
        if f is None:
            return float("inf")
        pred = max(0, min(255, f[0]))
        err_sum += abs(pred - v)
    return err_sum / len(samples)


def _select_model(
    samples: Sequence[Tuple[int, int]],
    target_dim: int,
    tile_size: int,
    families: Sequence[str],
    min_loo_points: int,
) -> Optional[ModelPrediction]:
    candidates: List[ModelPrediction] = []
    for family in families:
        fitted = _fit_and_predict(
            family=family,
            samples=samples,
            target_dim=target_dim,
            tile_size=tile_size,
        )
        if fitted is None:
            continue

        loo_exact, loo_mae = _eval_model_loo(
            family=family,
            samples=samples,
            tile_size=tile_size,
            min_loo_points=min_loo_points,
        )
        in_sample_mae = _eval_model_insample(
            family=family,
            samples=samples,
            tile_size=tile_size,
        )
        candidates.append(
            ModelPrediction(
                family=family,
                params=fitted[1],
                predicted=max(0, min(255, int(fitted[0]))),
                loo_exact_ratio=loo_exact,
                loo_mae=loo_mae,
                in_sample_mae=in_sample_mae,
                complexity=fitted[2],
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (
            -c.loo_exact_ratio,
            c.loo_mae,
            c.in_sample_mae,
            c.complexity,
            c.family,
        )
    )
    return candidates[0]


def _predict_linear_extrema(
    low_dim: int,
    high_dim: int,
    target_dim: int,
    low_val: int,
    high_val: int,
) -> int:
    if high_dim == low_dim:
        return int(low_val)
    frac = (target_dim - low_dim) / float(high_dim - low_dim)
    val = round(low_val + (high_val - low_val) * frac)
    return max(0, min(255, int(val)))


def _predict_model_at_dim(
    family: str,
    params: Dict[str, float],
    dim: int,
    tile_size: int,
) -> int:
    t = float(_tiles(dim, tile_size))
    if family == "const":
        y = params["c"]
    elif family == "tile-linear":
        y = params["a"] * t + params["b"]
    elif family == "tile-quadratic":
        y = params["a"] * (t * t) + params["b"] * t + params["c"]
    elif family == "tile2div-linear":
        div = int(round(params["div"]))
        x = float((int(t) * int(t)) // div)
        y = params["a"] * x + params["b"]
    else:
        raise ValueError(f"unsupported family for evaluation: {family}")
    return max(0, min(255, int(round(y))))


def _maybe_extrema_payloads(
    args: argparse.Namespace,
    stream: StreamConfig,
    fit_payloads: Sequence[Tuple[int, bytes]],
    chunk_index: int,
) -> Tuple[Optional[int], Optional[bytes], Optional[int], Optional[bytes]]:
    if args.low_dim is not None and args.high_dim is not None:
        low_dim = int(args.low_dim)
        high_dim = int(args.high_dim)

        if stream.name == "eo":
            low_path = Path(args.low_eo) if args.low_eo else None
            high_path = Path(args.high_eo) if args.high_eo else None
        else:
            low_path = Path(args.low_pc) if args.low_pc else None
            high_path = Path(args.high_pc) if args.high_pc else None

        if low_path is not None and high_path is not None and low_path.is_file() and high_path.is_file():
            return (
                low_dim,
                _read_chunk_bytes(low_path, chunk_index),
                high_dim,
                _read_chunk_bytes(high_path, chunk_index),
            )

    # Do not implicitly derive extrema from fit points. For mixed-family fit sets
    # this can produce large, unsafe baseline drifts.
    return (None, None, None, None)


def _print_stream_summary(stream_report: Dict[str, object]) -> None:
    print(
        "  stream={} len={} fit_points={} input_offsets={} emitted_rules={} mismatch_vs_target={} formula_applied={} fallback_used={}".format(
            stream_report["name"],
            stream_report["payload_len"],
            stream_report["fit_point_count"],
            stream_report["offset_count_input"],
            stream_report["rules_emitted"],
            stream_report.get("mismatch_vs_target", "n/a"),
            stream_report["formula_applied_count"],
            stream_report["fallback_count"],
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--target-dim", type=int, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--fit-point", action="append", default=[], help="DIM:EO_PATH:PC_PATH (repeat)")

    # Legacy endpoint arguments (optional fallback if --fit-point is omitted).
    parser.add_argument("--low-dim", type=int, default=None)
    parser.add_argument("--high-dim", type=int, default=None)
    parser.add_argument("--low-eo", default=None)
    parser.add_argument("--high-eo", default=None)
    parser.add_argument("--low-pc", default=None)
    parser.add_argument("--high-pc", default=None)

    # Stream files.
    parser.add_argument("--target-eo", default=None)
    parser.add_argument("--base-eo", default=None)
    parser.add_argument("--target-pc", default=None)
    parser.add_argument("--base-pc", default=None)

    parser.add_argument(
        "--offset-mode",
        default="changed-fit-union",
        choices=["changed-fit-union", "changed-extrema", "all"],
        help="which offsets to synthesize/emit (default: changed-fit-union)",
    )
    parser.add_argument(
        "--model-family",
        action="append",
        default=[],
        choices=["const", "tile-linear", "tile-quadratic", "tile2div-linear"],
        help="model families to consider (repeatable)",
    )
    parser.add_argument(
        "--min-loo-points",
        type=int,
        default=3,
        help="minimum sample count to run leave-one-out scoring (default: 3)",
    )
    parser.add_argument(
        "--min-apply-loo-exact",
        type=float,
        default=1.0,
        help="minimum LOO exact-match ratio required to apply formula prediction (default: 1.0)",
    )
    parser.add_argument(
        "--max-apply-loo-mae",
        type=float,
        default=0.0,
        help="maximum LOO MAE required to apply formula prediction (default: 0.0)",
    )
    parser.add_argument("--out-spec", required=True)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    if args.tile_size <= 0:
        raise SystemExit("--tile-size must be > 0")

    streams = [
        _ensure_base_target(args.base_eo, args.target_eo, "eo"),
        _ensure_base_target(args.base_pc, args.target_pc, "pc"),
    ]

    fit_points = _build_fit_points(args)
    families = args.model_family if args.model_family else ["const", "tile-linear", "tile-quadratic", "tile2div-linear"]

    spec_lines: List[str] = []
    report: Dict[str, object] = {
        "target_dim": args.target_dim,
        "chunk_index": args.chunk_index,
        "tile_size": args.tile_size,
        "offset_mode": args.offset_mode,
        "families": families,
        "apply_thresholds": {
            "min_loo_exact": args.min_apply_loo_exact,
            "max_loo_mae": args.max_apply_loo_mae,
        },
        "fit_points": [
            {
                "dim": fp.dim,
                "eo_path": str(fp.eo_path),
                "pc_path": str(fp.pc_path),
            }
            for fp in fit_points
        ],
        "streams": [],
    }

    total_rules = 0

    for s in streams:
        base = _read_chunk_bytes(s.base_path, args.chunk_index)
        target = _read_chunk_bytes(s.target_path, args.chunk_index) if s.target_path else None

        fit_payloads: List[Tuple[int, bytes]] = []
        for fp in fit_points:
            payload = _read_chunk_bytes(_stream_path(fp, s.name), args.chunk_index)
            fit_payloads.append((fp.dim, payload))

        low_dim, low_payload, high_dim, high_payload = _maybe_extrema_payloads(
            args=args,
            stream=s,
            fit_payloads=fit_payloads,
            chunk_index=args.chunk_index,
        )

        if args.offset_mode == "all":
            offsets = list(range(len(base)))
        elif args.offset_mode == "changed-extrema":
            offsets = _offsets_changed_extrema(len(base), low_payload, high_payload)
        else:
            offsets = _offsets_changed_fit_union(len(base), fit_payloads)

        predicted = bytearray(base)
        model_hist: Dict[str, int] = {}
        fallback_count = 0
        formula_applied_count = 0
        formula_rejected_count = 0
        prediction_details: List[Dict[str, object]] = []

        for off in offsets:
            samples = _values_for_offset(fit_payloads, off)
            chosen = _select_model(
                samples=samples,
                target_dim=args.target_dim,
                tile_size=args.tile_size,
                families=families,
                min_loo_points=args.min_loo_points,
            )

            if low_payload is not None and high_payload is not None and low_dim is not None and high_dim is not None:
                if off < len(low_payload) and off < len(high_payload):
                    baseline_val = _predict_linear_extrema(
                        low_dim=low_dim,
                        high_dim=high_dim,
                        target_dim=args.target_dim,
                        low_val=low_payload[off],
                        high_val=high_payload[off],
                    )
                    baseline_family = "baseline-linear-extrema"
                else:
                    baseline_val = base[off]
                    baseline_family = "baseline-base"
            else:
                baseline_val = base[off]
                baseline_family = "baseline-base"

            predicted_val = baseline_val
            family_name = baseline_family
            params: Dict[str, float] = {}
            loo_exact_ratio = 0.0
            loo_mae = float("inf")
            in_sample_mae = float("inf")
            complexity = 99
            used_fallback = True

            if chosen is not None:
                endpoint_match = True
                if (
                    low_payload is not None
                    and high_payload is not None
                    and low_dim is not None
                    and high_dim is not None
                    and off < len(low_payload)
                    and off < len(high_payload)
                ):
                    lp = _predict_model_at_dim(
                        family=chosen.family,
                        params=chosen.params,
                        dim=low_dim,
                        tile_size=args.tile_size,
                    )
                    hp = _predict_model_at_dim(
                        family=chosen.family,
                        params=chosen.params,
                        dim=high_dim,
                        tile_size=args.tile_size,
                    )
                    endpoint_match = (lp == low_payload[off]) and (hp == high_payload[off])

                meets_threshold = (
                    math.isfinite(chosen.loo_mae)
                    and chosen.loo_exact_ratio >= args.min_apply_loo_exact
                    and chosen.loo_mae <= args.max_apply_loo_mae
                    and endpoint_match
                )
                if meets_threshold:
                    predicted_val = chosen.predicted
                    family_name = chosen.family
                    params = chosen.params
                    loo_exact_ratio = chosen.loo_exact_ratio
                    loo_mae = chosen.loo_mae
                    in_sample_mae = chosen.in_sample_mae
                    complexity = chosen.complexity
                    used_fallback = False
                    formula_applied_count += 1
                else:
                    formula_rejected_count += 1

            predicted[off] = max(0, min(255, int(predicted_val)))
            model_hist[family_name] = model_hist.get(family_name, 0) + 1
            if used_fallback:
                fallback_count += 1

            if len(prediction_details) < 400:
                prediction_details.append(
                    {
                        "offset": off,
                        "family": family_name,
                        "predicted": int(predicted[off]),
                        "sample_count": len(samples),
                        "params": params,
                        "loo_exact_ratio": loo_exact_ratio,
                        "loo_mae": loo_mae,
                        "in_sample_mae": in_sample_mae,
                        "complexity": complexity,
                    }
                )

        changed_vs_base = [i for i in range(len(base)) if predicted[i] != base[i]]
        for off in changed_vs_base:
            spec_lines.append(f"{len(base)} {off} 0x{predicted[off]:02x}")

        stream_report: Dict[str, object] = {
            "name": s.name,
            "payload_len": len(base),
            "fit_point_count": len(fit_payloads),
            "offset_count_input": len(offsets),
            "rules_emitted": len(changed_vs_base),
            "base_sha256": pe._sha256_bytes(base),
            "predicted_sha256": pe._sha256_bytes(bytes(predicted)),
            "model_histogram": dict(sorted(model_hist.items())),
            "fallback_count": fallback_count,
            "formula_applied_count": formula_applied_count,
            "formula_rejected_count": formula_rejected_count,
            "changed_vs_base_preview": changed_vs_base[:120],
            "prediction_details_preview": prediction_details,
        }

        if target is not None:
            if len(target) != len(base):
                raise SystemExit(
                    f"length mismatch in stream '{s.name}': target={len(target)} vs base={len(base)}"
                )
            mismatches = [i for i in range(len(target)) if predicted[i] != target[i]]
            stream_report.update(
                {
                    "target_sha256": pe._sha256_bytes(target),
                    "mismatch_vs_target": len(mismatches),
                    "mismatch_ratio_vs_target": len(mismatches) / float(len(target)),
                    "mismatch_preview": mismatches[:120],
                }
            )

        report["streams"].append(stream_report)
        total_rules += len(changed_vs_base)

    out_spec = Path(args.out_spec)
    out_spec.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# synthesized by synthesize_instruction_patch_spec.py",
        f"# target_dim={args.target_dim} chunk_index={args.chunk_index} tile_size={args.tile_size}",
        f"# offset_mode={args.offset_mode} families={','.join(families)}",
    ]
    out_spec.write_text("\n".join(header + [""] + spec_lines).rstrip() + "\n")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Wrote patch spec: {out_spec}")
    print(f"Rules emitted: {total_rules}")
    for s in report["streams"]:  # type: ignore[index]
        _print_stream_summary(s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
