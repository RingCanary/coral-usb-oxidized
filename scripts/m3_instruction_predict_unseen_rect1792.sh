#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PIECEWISE_TARGET_INPUT="${PIECEWISE_TARGET_INPUT:-1344}"
FIXED_OUTPUT="${FIXED_OUTPUT:-1792}"
LOW_INPUT="${LOW_INPUT:-896}"
MID_INPUT="${MID_INPUT:-1792}"
HIGH_INPUT="${HIGH_INPUT:-2688}"

RUN_ID="m3-unseen-predict-rect${FIXED_OUTPUT}-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "run_id=$RUN_ID"
echo "out_dir=$OUT_DIR"

compile_model() {
  local in_dim="$1"
  local out_dim="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  cargo run --example rust_dense_template_compile -- \
    --out-dir "$out_dir" \
    --input-dim "$in_dim" \
    --output-dim "$out_dim" \
    --init-mode zero \
    > "$out_dir/compile_driver.log" 2>&1 || true

  local compiled="$out_dir/dense_${in_dim}x${out_dim}_quant_edgetpu.tflite"
  if [[ ! -f "$compiled" ]]; then
    local fb="$out_dir/$out_dir/dense_${in_dim}x${out_dim}_quant_edgetpu.tflite"
    if [[ -f "$fb" ]]; then
      compiled="$fb"
    fi
  fi
  if [[ ! -f "$compiled" ]]; then
    echo "missing compiled model for ${in_dim}x${out_dim}: $out_dir" >&2
    exit 1
  fi

  python3 "$REPO_ROOT/tools/extract_edgetpu_package.py" extract \
    "$compiled" \
    --out "$out_dir/extract" \
    --overwrite \
    > "$out_dir/extract.log" 2>&1
}

# Existing training artifacts.
RECT_896_MODEL="$REPO_ROOT/traces/analysis/m3-param-permutation-rect-20260303T151814Z/dense_896x1792_mod251_quant_edgetpu.tflite"
if [[ ! -f "$RECT_896_MODEL" ]]; then
  echo "missing required model: $RECT_896_MODEL" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/traces/analysis/m3-param-permutation-rect-20260303T151814Z/extract/package_000/serialized_executable_000.bin" ]]; then
  python3 "$REPO_ROOT/tools/extract_edgetpu_package.py" extract \
    "$RECT_896_MODEL" \
    --out "$REPO_ROOT/traces/analysis/m3-param-permutation-rect-20260303T151814Z/extract" \
    --overwrite \
    > "$OUT_DIR/extract_rect896.log" 2>&1
fi

# Compile missing training high-point and target unseen point.
TRAIN_HIGH_DIR="$OUT_DIR/train_${HIGH_INPUT}x${FIXED_OUTPUT}"
TARGET_DIR="$OUT_DIR/target_${PIECEWISE_TARGET_INPUT}x${FIXED_OUTPUT}"
compile_model "$HIGH_INPUT" "$FIXED_OUTPUT" "$TRAIN_HIGH_DIR"
compile_model "$PIECEWISE_TARGET_INPUT" "$FIXED_OUTPUT" "$TARGET_DIR"

TRAIN_896_EO="$REPO_ROOT/traces/analysis/m3-param-permutation-rect-20260303T151814Z/extract/package_000/serialized_executable_000.bin"
TRAIN_896_PC="$REPO_ROOT/traces/analysis/m3-param-permutation-rect-20260303T151814Z/extract/package_000/serialized_executable_001.bin"
TRAIN_1792_EO="$REPO_ROOT/traces/dense-template-1792x1792-20260301T141847Z/extract/package_000/serialized_executable_000.bin"
TRAIN_1792_PC="$REPO_ROOT/traces/dense-template-1792x1792-20260301T141847Z/extract/package_000/serialized_executable_001.bin"
TRAIN_2688_EO="$TRAIN_HIGH_DIR/extract/package_000/serialized_executable_000.bin"
TRAIN_2688_PC="$TRAIN_HIGH_DIR/extract/package_000/serialized_executable_001.bin"
TGT_EO="$TARGET_DIR/extract/package_000/serialized_executable_000.bin"
TGT_PC="$TARGET_DIR/extract/package_000/serialized_executable_001.bin"

for p in "$TRAIN_896_EO" "$TRAIN_896_PC" "$TRAIN_1792_EO" "$TRAIN_1792_PC" "$TRAIN_2688_EO" "$TRAIN_2688_PC" "$TGT_EO" "$TGT_PC"; do
  if [[ ! -f "$p" ]]; then
    echo "missing executable: $p" >&2
    exit 1
  fi
done

python3 "$REPO_ROOT/tools/instruction_word_field_analysis.py" \
  --entry ${LOW_INPUT}:"$TRAIN_896_PC" \
  --entry ${MID_INPUT}:"$TRAIN_1792_PC" \
  --entry ${HIGH_INPUT}:"$TRAIN_2688_PC" \
  --chunk-index 0 \
  --json-out "$OUT_DIR/pc_wordfield_rect${FIXED_OUTPUT}.json" \
  > "$OUT_DIR/pc_wordfield.stdout.txt"

python3 "$REPO_ROOT/tools/instruction_word_field_analysis.py" \
  --entry ${LOW_INPUT}:"$TRAIN_896_EO" \
  --entry ${MID_INPUT}:"$TRAIN_1792_EO" \
  --entry ${HIGH_INPUT}:"$TRAIN_2688_EO" \
  --chunk-index 0 \
  --json-out "$OUT_DIR/eo_wordfield_rect${FIXED_OUTPUT}.json" \
  > "$OUT_DIR/eo_wordfield.stdout.txt"

for mode in endpoint best strict threepoint; do
  cargo run --bin word_field_spec_v2 -- \
    --analysis-json "$OUT_DIR/pc_wordfield_rect${FIXED_OUTPUT}.json" \
    --base-exec "$TRAIN_896_PC" \
    --target-exec "$TGT_PC" \
    --low-dim "$LOW_INPUT" \
    --high-dim "$HIGH_INPUT" \
    --target-dim "$PIECEWISE_TARGET_INPUT" \
    --mid-dim "$MID_INPUT" \
    --mid-exec "$TRAIN_1792_PC" \
    --predict-mode "$mode" \
    --lane-priority lane32,lane16 \
    --out-spec "$OUT_DIR/pc_${mode}.fieldspec.json" \
    --out-report "$OUT_DIR/pc_${mode}.report.json" \
    --out-patchspec "$OUT_DIR/pc_${mode}.full.patchspec" \
    --out-patchspec-safe "$OUT_DIR/pc_${mode}.safe.patchspec" \
    --out-patchspec-discrete "$OUT_DIR/pc_${mode}.discrete.patchspec" \
    > "$OUT_DIR/pc_${mode}.stdout.log" 2>&1
  echo "pc mode=$mode done"
done

set +e
cargo run --bin word_field_spec_v2 -- \
  --analysis-json "$OUT_DIR/eo_wordfield_rect${FIXED_OUTPUT}.json" \
  --base-exec "$TRAIN_896_EO" \
  --target-exec "$TGT_EO" \
  --low-dim "$LOW_INPUT" \
  --high-dim "$HIGH_INPUT" \
  --target-dim "$PIECEWISE_TARGET_INPUT" \
  --mid-dim "$MID_INPUT" \
  --mid-exec "$TRAIN_1792_EO" \
  --predict-mode threepoint \
  --lane-priority lane32,lane16 \
  --out-spec "$OUT_DIR/eo_threepoint.fieldspec.json" \
  --out-report "$OUT_DIR/eo_threepoint.report.json" \
  --out-patchspec "$OUT_DIR/eo_threepoint.full.patchspec" \
  > "$OUT_DIR/eo_threepoint.stdout.log" 2>&1
EO_RC=$?
set -e

echo "$EO_RC" > "$OUT_DIR/eo_threepoint.rc"

python3 - "$OUT_DIR" "$PIECEWISE_TARGET_INPUT" "$FIXED_OUTPUT" <<'PY'
import json, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
target_input = int(sys.argv[2])
out_dim = int(sys.argv[3])
rows = []
for mode in ["endpoint", "best", "strict", "threepoint"]:
    rep = json.loads((out / f"pc_{mode}.report.json").read_text())
    rows.append({
        "mode": mode,
        "baseline_mismatch": rep["baseline"]["mismatch_vs_target"],
        "v2_mismatch": rep["with_v2_spec"]["mismatch_vs_target"],
        "changed_bytes": rep["v2_changed_byte_count"],
        "safe_core_bytes": rep["safe_core_byte_count"],
        "discrete_flags_bytes": rep["discrete_flags_byte_count"],
        "unknown_bytes": rep["unknown_byte_count"],
        "residue_rule_count": rep["residue_rule_count"],
        "offset_rule_count": rep["offset_rule_count"],
    })
best = sorted(rows, key=lambda r: (r["v2_mismatch"], r["changed_bytes"], r["mode"]))[0]

eo_log = (out / "eo_threepoint.stdout.log").read_text(errors="ignore")
eo_errs = re.findall(r"error: (.+)", eo_log)

data = {
    "target_input": target_input,
    "target_output": out_dim,
    "pc_rows": rows,
    "pc_best": best,
    "eo_boundary_error": eo_errs[-1] if eo_errs else None,
}
(out / "SUMMARY.json").write_text(json.dumps(data, indent=2) + "\n")
with (out / "SUMMARY.txt").open("w", encoding="utf-8") as f:
    f.write(f"target_input={target_input} target_output={out_dim}\n")
    for r in rows:
        f.write(
            f"pc mode={r['mode']} baseline={r['baseline_mismatch']} v2={r['v2_mismatch']} changed={r['changed_bytes']} "
            f"safe={r['safe_core_bytes']} discrete={r['discrete_flags_bytes']} unknown={r['unknown_bytes']} "
            f"residue_rules={r['residue_rule_count']} offset_rules={r['offset_rule_count']}\n"
        )
    f.write(f"pc best_mode={best['mode']} v2_mismatch={best['v2_mismatch']} changed={best['changed_bytes']}\n")
    if data["eo_boundary_error"]:
        f.write(f"eo_boundary_error={data['eo_boundary_error']}\n")
print(out / "SUMMARY.json")
PY

echo "done: $OUT_DIR"
