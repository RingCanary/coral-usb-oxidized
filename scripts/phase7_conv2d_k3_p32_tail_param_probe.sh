#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ARTIFACT_REL="${SOURCE_ARTIFACT_REL:-traces/analysis/phase4-conv2d-k3-family-scout-20260316T114108Z}"
COMPLETION_ARTIFACT_REL="${COMPLETION_ARTIFACT_REL:-traces/analysis/phase4-conv2d-k3-completion-demo-20260316T114632Z}"
RUN_ID="phase7-conv2d-k3-p32-tail-param-probe-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR"

for w in 176 184 192; do
  cargo run --quiet --bin model_param_stream_dump -- \
    --model "$REPO_ROOT/$SOURCE_ARTIFACT_REL/h12_w${w}_ic32_oc32_k3/conv2d_12x${w}x32_to_32_k3_s1_same_quant_edgetpu.tflite" \
    --out "$OUT_DIR/h12_w${w}.compiler_param.bin" >/dev/null
done

python3 - <<'PY' "$REPO_ROOT" "$SOURCE_ARTIFACT_REL" "$COMPLETION_ARTIFACT_REL" "$OUT_DIR"
import json
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
source_artifact = repo_root / sys.argv[2]
completion_artifact = repo_root / sys.argv[3]
out_dir = pathlib.Path(sys.argv[4])

sys.path.insert(0, str(repo_root / "tools"))
import parse_edgetpu_executable as pe

PREFIX = 256
GROUP = 128
GROUPS = 72


def read_exec_chunk(path: pathlib.Path) -> bytearray:
    blob = path.read_bytes()
    root = pe._parse_root_table(blob, 0, file_identifier=None)
    tables = pe._read_vector_table_field(root, 5)
    return bytearray(pe._read_vector_bytes_field(tables[0], 0))


def apply_patch(base: bytearray, patchspec: pathlib.Path) -> bytearray:
    out = bytearray(base)
    for raw in patchspec.read_text().splitlines():
        clean = raw.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) != 3:
            continue
        _plen, off, val = [int(x, 0) for x in parts]
        out[off] = val
    return out


def mismatch_runs(a: bytes, b: bytes):
    diffs = [idx for idx, (lhs, rhs) in enumerate(zip(a, b)) if lhs != rhs]
    if not diffs:
        return []
    runs = []
    start = prev = diffs[0]
    for idx in diffs[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            runs.append([start, prev])
            start = prev = idx
    runs.append([start, prev])
    return runs


summary = []
summary.append(f"source_artifact={source_artifact}")
summary.append(f"completion_artifact={completion_artifact}")
summary.append("expected_mapping=compiler_group=(kernel_pos*8)+ic_group")

anchor_exec = read_exec_chunk(
    repo_root
    / "templates/phase7_conv2d_k3_h12_corridor_6512/p32/anchor/serialized_executable_000.bin"
)

results = {}
for width in (176, 184, 192):
    target_exec = read_exec_chunk(
        source_artifact
        / f"h12_w{width}_ic32_oc32_k3/extract/package_000/serialized_executable_000.bin"
    )
    patched_exec = apply_patch(
        anchor_exec,
        completion_artifact / f"cases/p32/h12_w{width}/eo.patchspec",
    )
    eo_mismatch = [idx for idx, (lhs, rhs) in enumerate(zip(patched_exec, target_exec)) if lhs != rhs]

    compiler_param = (out_dir / f"h12_w{width}.compiler_param.bin").read_bytes()
    native_param = (
        completion_artifact / f"cases/p32/h12_w{width}/target_param_stream.bin"
    ).read_bytes()
    param_diffs = [idx for idx, (lhs, rhs) in enumerate(zip(compiler_param, native_param)) if lhs != rhs]

    compiler_groups = [
        compiler_param[PREFIX + g * GROUP : PREFIX + (g + 1) * GROUP] for g in range(GROUPS)
    ]
    native_groups = [
        native_param[PREFIX + g * GROUP : PREFIX + (g + 1) * GROUP] for g in range(GROUPS)
    ]
    mapping = []
    for ng in native_groups:
        mapping.append(next(j for j, cg in enumerate(compiler_groups) if ng == cg))

    expected = [((g % 9) * 8) + (g // 9) for g in range(GROUPS)]
    results[f"h12_w{width}"] = {
        "eo_mismatch_count": len(eo_mismatch),
        "param_mismatch_count": len(param_diffs),
        "param_mismatch_runs": mismatch_runs(compiler_param, native_param),
        "same_group_positions": [idx for idx, comp_idx in enumerate(mapping) if idx == comp_idx],
        "mapping_matches_expected": mapping == expected,
        "mapping_first16": mapping[:16],
    }
    summary.append(
        f"h12_w{width}: eo_mismatch_count={len(eo_mismatch)} "
        f"param_mismatch_count={len(param_diffs)} "
        f"mapping_matches_expected={mapping == expected} "
        f"same_group_positions={[idx for idx, comp_idx in enumerate(mapping) if idx == comp_idx]}"
    )

(out_dir / "tail_param_probe.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
(out_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
PY

echo "$OUT_DIR"
