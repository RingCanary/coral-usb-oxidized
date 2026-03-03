#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/m3_param_stream_probe.sh [--out-dir PATH]

Generates controlled dense models (zero/ones at 896 and 1792), compiles them
with edgetpu_compiler, and runs Rust param-stream differential analysis.

Outputs:
  <out-dir>/param_stream_diff.report.json
  <out-dir>/SUMMARY.txt

Default out-dir:
  traces/analysis/m3-param-stream-probe-<utc>
USAGE
}

OUT_DIR=""
while (($# > 0)); do
  case "$1" in
    --out-dir)
      [[ $# -ge 2 ]] || { echo "error: --out-dir requires value" >&2; exit 1; }
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="traces/analysis/m3-param-stream-probe-$(date -u +%Y%m%dT%H%M%SZ)"
fi

command -v uv >/dev/null 2>&1 || { echo "error: uv not found" >&2; exit 1; }
command -v edgetpu_compiler >/dev/null 2>&1 || { echo "error: edgetpu_compiler not found" >&2; exit 1; }

mkdir -p "$OUT_DIR"

PY_VER="3.9"
TF_PKG="tensorflow-cpu==2.10.1"
NP_PKG="numpy==1.23.5"

compile_case() {
  local name="$1"
  local dim="$2"
  local init_mode="$3"

  local case_dir="$OUT_DIR/$name"
  mkdir -p "$case_dir"

  local quant_model="$case_dir/dense_${dim}x${dim}_${init_mode}_quant.tflite"
  local quant_meta="$case_dir/dense_${dim}x${dim}_${init_mode}_quant.metadata.json"
  local compile_log="$case_dir/compile.log"
  local generate_log="$case_dir/generate.log"

  echo "[m3-probe] generate $name"
  uv run --python "$PY_VER" --with "$TF_PKG" --with "$NP_PKG" \
    tools/generate_dense_quant_tflite.py \
    --output "$quant_model" \
    --metadata-out "$quant_meta" \
    --input-dim "$dim" \
    --output-dim "$dim" \
    --init-mode "$init_mode" \
    >"$generate_log" 2>&1

  echo "[m3-probe] compile $name"
  edgetpu_compiler -s -o "$case_dir" "$quant_model" >"$compile_log" 2>&1
}

compile_case d1792_zero 1792 zero
compile_case d1792_ones 1792 ones
compile_case d896_zero 896 zero
compile_case d896_ones 896 ones

REPORT_JSON="$OUT_DIR/param_stream_diff.report.json"
SUMMARY_TXT="$OUT_DIR/SUMMARY.txt"

echo "[m3-probe] running Rust param_stream_diff"
cargo run --bin param_stream_diff -- \
  --model d1792_zero="$OUT_DIR/d1792_zero/dense_1792x1792_zero_quant_edgetpu.tflite" \
  --model d1792_ones="$OUT_DIR/d1792_ones/dense_1792x1792_ones_quant_edgetpu.tflite" \
  --model d896_zero="$OUT_DIR/d896_zero/dense_896x896_zero_quant_edgetpu.tflite" \
  --model d896_ones="$OUT_DIR/d896_ones/dense_896x896_ones_quant_edgetpu.tflite" \
  --compare d1792_zero:d1792_ones \
  --compare d896_zero:d896_ones \
  --compare d896_ones:d1792_ones \
  --compare d896_zero:d1792_zero \
  --out-json "$REPORT_JSON"

python - <<'PY' "$REPORT_JSON" "$SUMMARY_TXT"
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
r = json.loads(report_path.read_text())

models = {m["name"]: m for m in r.get("models", [])}
comparisons = {(c["lhs"], c["rhs"]): c for c in r.get("comparisons", [])}

lines = []
lines.append("Milestone 3 Param Stream Probe Summary")
lines.append(f"report={report_path}")
lines.append("")
for name in ["d896_zero", "d896_ones", "d1792_zero", "d1792_ones"]:
    m = models.get(name)
    if not m:
        continue
    lines.append(
        f"model {name}: len={m['param_len']} fnv1a64={m['param_fnv1a64_hex']} "
        f"instr_chunks={m['instruction_chunk_lens']}"
    )

lines.append("")
for pair in [
    ("d1792_zero", "d1792_ones"),
    ("d896_zero", "d896_ones"),
    ("d896_ones", "d1792_ones"),
    ("d896_zero", "d1792_zero"),
]:
    c = comparisons.get(pair)
    if not c:
        continue
    lines.append(
        f"compare {pair[0]} vs {pair[1]}: changed={c['changed_in_overlap']}/{c['overlap_len']} "
        f"({c['changed_fraction_in_overlap']:.6f}) extra_rhs={c['extra_bytes_rhs']} "
        f"prefix_equal={c['equal_prefix_len']}"
    )

mi = r.get("multi_model_invariants")
if mi:
    lines.append("")
    lines.append(
        f"multi-model invariants: count={mi['invariant_offset_count']}/{mi['common_prefix_len']} "
        f"({mi['invariant_fraction']:.6f})"
    )

summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY

echo "[m3-probe] done"
echo "  out_dir:   $OUT_DIR"
echo "  report:    $REPORT_JSON"
echo "  summary:   $SUMMARY_TXT"
