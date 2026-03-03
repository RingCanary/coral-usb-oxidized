#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_ID="m6-instruction-axis-probe-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
VAR_DIR="$OUT_DIR/variants"
mkdir -p "$VAR_DIR"

echo "run_id=$RUN_ID"
echo "out_dir=$OUT_DIR"

compile_variant() {
  local name="$1"
  shift
  local dir="$VAR_DIR/$name"
  mkdir -p "$dir"
  echo "[compile] $name"
  "$REPO_ROOT/tools/dense_template_pipeline.sh" \
    --out-dir "$dir" \
    --input-dim 1792 \
    --output-dim 1792 \
    --init-mode ones \
    --rep-samples 64 \
    --rep-range 1.0 \
    "$@" \
    > "$dir.pipeline.log" 2>&1
}

compile_variant baseline
compile_variant quant_range_half --rep-range 0.5
compile_variant quant_range_double --rep-range 2.0
compile_variant quant_offset_pos --rep-offset 0.5
compile_variant act_relu --activation relu
compile_variant act_relu6 --activation relu6
compile_variant bias_on --use-bias

BASE_EO="$VAR_DIR/baseline/extract/package_000/serialized_executable_000.bin"
BASE_PC="$VAR_DIR/baseline/extract/package_000/serialized_executable_001.bin"

for p in "$BASE_EO" "$BASE_PC"; do
  [[ -f "$p" ]] || { echo "missing baseline executable: $p" >&2; exit 1; }
done

build_variant_args() {
  local kind="$1" # eo|pc
  local exe_idx
  if [[ "$kind" == "eo" ]]; then
    exe_idx=000
  else
    exe_idx=001
  fi
  local args=()
  for name in quant_range_half quant_range_double quant_offset_pos act_relu act_relu6 bias_on; do
    local p="$VAR_DIR/$name/extract/package_000/serialized_executable_${exe_idx}.bin"
    [[ -f "$p" ]] || { echo "missing variant executable: $p" >&2; exit 1; }
    args+=(--variant "$name:$p")
  done
  printf '%s\n' "${args[@]}"
}

mapfile -t EO_ARGS < <(build_variant_args eo)
mapfile -t PC_ARGS < <(build_variant_args pc)

cargo run --bin instruction_chunk_diff -- \
  --baseline "$BASE_EO" \
  "${EO_ARGS[@]}" \
  --chunk-index 0 \
  --out-json "$OUT_DIR/eo_axis_diff.json" \
  > "$OUT_DIR/eo_axis_diff.stdout.log" 2>&1

cargo run --bin instruction_chunk_diff -- \
  --baseline "$BASE_PC" \
  "${PC_ARGS[@]}" \
  --chunk-index 0 \
  --out-json "$OUT_DIR/pc_axis_diff.json" \
  > "$OUT_DIR/pc_axis_diff.stdout.log" 2>&1

python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import collections
import sys

out = pathlib.Path(sys.argv[1])
eo = json.loads((out / "eo_axis_diff.json").read_text())
pc = json.loads((out / "pc_axis_diff.json").read_text())

axis_of = {
    "quant_range_half": "quantization",
    "quant_range_double": "quantization",
    "quant_offset_pos": "quantization",
    "act_relu": "activation",
    "act_relu6": "activation",
    "bias_on": "bias",
}

def classify(report):
    by_offset = collections.defaultdict(set)
    changed_counts = {}
    for v in report["variants"]:
        changed_counts[v["name"]] = v["changed_count"]
        axis = axis_of.get(v["name"], "unknown")
        for off in v["changed_offsets"]:
            by_offset[int(off)].add(axis)

    sig_counts = collections.Counter()
    for off, axes in by_offset.items():
        sig = "+".join(sorted(axes)) if axes else "none"
        sig_counts[sig] += 1

    return {
        "payload_len": report["payload_len"],
        "variant_changed_counts": changed_counts,
        "changed_offset_count": len(by_offset),
        "axis_signature_histogram": dict(sorted(sig_counts.items())),
        "offset_axes": [
            {"offset": off, "axes": sorted(list(axes))}
            for off, axes in sorted(by_offset.items())
        ],
    }

summary = {
    "run_id": out.name,
    "eo": classify(eo),
    "pc": classify(pc),
    "variant_model_paths": {
        name: str(out / "variants" / name / f"dense_1792x1792_quant_edgetpu.tflite")
        for name in ["baseline", "quant_range_half", "quant_range_double", "quant_offset_pos", "act_relu", "act_relu6", "bias_on"]
    },
}

(out / "semantic_labels.json").write_text(json.dumps(summary, indent=2) + "\n")
with (out / "SUMMARY.txt").open("w", encoding="utf-8") as f:
    f.write(f"run_id={out.name}\n")
    for plane in ["eo", "pc"]:
        section = summary[plane]
        f.write(f"[{plane}] payload_len={section['payload_len']} changed_offsets={section['changed_offset_count']}\n")
        for name, cnt in section["variant_changed_counts"].items():
            f.write(f"  variant {name}: changed={cnt}\n")
        f.write(f"  axis_signature_histogram={section['axis_signature_histogram']}\n")
print(out / "SUMMARY.txt")
PY

echo "done: $OUT_DIR"
