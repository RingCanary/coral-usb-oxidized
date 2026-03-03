#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_ID="m5-family-patchspecs-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "run_id=$RUN_ID"
echo "out_dir=$OUT_DIR"

merge_patchspecs() {
  local out_path="$1"
  shift
  python3 - "$out_path" "$@" <<'PY'
import sys
from pathlib import Path
out = Path(sys.argv[1])
paths = [Path(p) for p in sys.argv[2:]]
rules = {}
for p in paths:
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        plen = int(parts[0])
        off = int(parts[1])
        val = int(parts[2], 16)
        key = (plen, off)
        if key in rules and rules[key] != val:
            raise SystemExit(f"conflict at {key}: {rules[key]:02x} vs {val:02x}")
        rules[key] = val
lines = ["# merged by scripts/m5_build_family_patchspecs.sh"]
for (plen, off), val in sorted(rules.items()):
    lines.append(f"{plen} {off} 0x{val:02x}")
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out} rules={len(rules)}")
PY
}

run_wordfield() {
  local fam="$1"
  local kind="$2" # eo|pc
  local low_dim="$3"
  local mid_dim="$4"
  local high_dim="$5"
  local target_dim="$6"
  local low_exec="$7"
  local mid_exec="$8"
  local high_exec="$9"
  local target_exec="${10}"

  local fam_dir="$OUT_DIR/$fam"
  mkdir -p "$fam_dir"

  local analysis_json="$fam_dir/${kind}_wordfield.json"
  local spec_json="$fam_dir/${kind}_strict.fieldspec.json"
  local report_json="$fam_dir/${kind}_strict.report.json"
  local patch_full="$fam_dir/${kind}_strict.full.patchspec"
  local patch_safe="$fam_dir/${kind}_strict.safe.patchspec"
  local patch_discrete="$fam_dir/${kind}_strict.discrete.patchspec"

  python3 "$REPO_ROOT/tools/instruction_word_field_analysis.py" \
    --entry ${low_dim}:"$low_exec" \
    --entry ${mid_dim}:"$mid_exec" \
    --entry ${high_dim}:"$high_exec" \
    --chunk-index 0 \
    --json-out "$analysis_json" \
    > "$fam_dir/${kind}_wordfield.stdout.log"

  cargo run --bin word_field_spec_v2 -- \
    --analysis-json "$analysis_json" \
    --base-exec "$low_exec" \
    --target-exec "$target_exec" \
    --low-dim "$low_dim" \
    --mid-dim "$mid_dim" \
    --mid-exec "$mid_exec" \
    --high-dim "$high_dim" \
    --target-dim "$target_dim" \
    --predict-mode strict \
    --lane-priority lane32,lane16 \
    --out-spec "$spec_json" \
    --out-report "$report_json" \
    --out-patchspec "$patch_full" \
    --out-patchspec-safe "$patch_safe" \
    --out-patchspec-discrete "$patch_discrete" \
    > "$fam_dir/${kind}_strict.stdout.log" 2>&1
}

# -------- Family 7056/1840 (synthetic 3-point axis over known same-family members)
F7056_LOW_EO="$REPO_ROOT/traces/analysis/m5-family-rect-scan-o640-20260303T183401Z/i640_o640/extract/package_000/serialized_executable_000.bin"
F7056_LOW_PC="$REPO_ROOT/traces/analysis/m5-family-rect-scan-o640-20260303T183401Z/i640_o640/extract/package_000/serialized_executable_001.bin"
F7056_MID_EO="$REPO_ROOT/traces/analysis/m5-family-rect-scan-20260303T183223Z/i640_o1280/extract/package_000/serialized_executable_000.bin"
F7056_MID_PC="$REPO_ROOT/traces/analysis/m5-family-rect-scan-20260303T183223Z/i640_o1280/extract/package_000/serialized_executable_001.bin"
F7056_HIGH_EO="$REPO_ROOT/traces/analysis/m5-family-rect-scan-20260303T183223Z/i1280_o1280/extract/package_000/serialized_executable_000.bin"
F7056_HIGH_PC="$REPO_ROOT/traces/analysis/m5-family-rect-scan-20260303T183223Z/i1280_o1280/extract/package_000/serialized_executable_001.bin"

run_wordfield f7056 eo 640 960 1280 960 "$F7056_LOW_EO" "$F7056_MID_EO" "$F7056_HIGH_EO" "$F7056_MID_EO"
run_wordfield f7056 pc 640 960 1280 960 "$F7056_LOW_PC" "$F7056_MID_PC" "$F7056_HIGH_PC" "$F7056_MID_PC"
merge_patchspecs "$OUT_DIR/f7056/f7056_strict.full.patchspec" "$OUT_DIR/f7056/eo_strict.full.patchspec" "$OUT_DIR/f7056/pc_strict.full.patchspec"
merge_patchspecs "$OUT_DIR/f7056/f7056_strict.safe.patchspec" "$OUT_DIR/f7056/eo_strict.safe.patchspec" "$OUT_DIR/f7056/pc_strict.safe.patchspec"
merge_patchspecs "$OUT_DIR/f7056/f7056_strict.discrete.patchspec" "$OUT_DIR/f7056/eo_strict.discrete.patchspec" "$OUT_DIR/f7056/pc_strict.discrete.patchspec"

# -------- Family 7952/2096 (fixed output 1536, input 768/1536/2304)
F7952_LOW_EO="$REPO_ROOT/traces/analysis/m5-family-7952-check-o1536-20260303T185105Z/i768_o1536/extract/package_000/serialized_executable_000.bin"
F7952_LOW_PC="$REPO_ROOT/traces/analysis/m5-family-7952-check-o1536-20260303T185105Z/i768_o1536/extract/package_000/serialized_executable_001.bin"
F7952_MID_EO="$REPO_ROOT/traces/analysis/m5-family-7952-check-o1536-20260303T185105Z/i1536_o1536/extract/package_000/serialized_executable_000.bin"
F7952_MID_PC="$REPO_ROOT/traces/analysis/m5-family-7952-check-o1536-20260303T185105Z/i1536_o1536/extract/package_000/serialized_executable_001.bin"
F7952_HIGH_EO="$REPO_ROOT/traces/analysis/m5-family-7952-check-o1536-20260303T185105Z/i2304_o1536/extract/package_000/serialized_executable_000.bin"
F7952_HIGH_PC="$REPO_ROOT/traces/analysis/m5-family-7952-check-o1536-20260303T185105Z/i2304_o1536/extract/package_000/serialized_executable_001.bin"

run_wordfield f7952 eo 768 1536 2304 1536 "$F7952_LOW_EO" "$F7952_MID_EO" "$F7952_HIGH_EO" "$F7952_MID_EO"
run_wordfield f7952 pc 768 1536 2304 1536 "$F7952_LOW_PC" "$F7952_MID_PC" "$F7952_HIGH_PC" "$F7952_MID_PC"
merge_patchspecs "$OUT_DIR/f7952/f7952_strict.full.patchspec" "$OUT_DIR/f7952/eo_strict.full.patchspec" "$OUT_DIR/f7952/pc_strict.full.patchspec"
merge_patchspecs "$OUT_DIR/f7952/f7952_strict.safe.patchspec" "$OUT_DIR/f7952/eo_strict.safe.patchspec" "$OUT_DIR/f7952/pc_strict.safe.patchspec"
merge_patchspecs "$OUT_DIR/f7952/f7952_strict.discrete.patchspec" "$OUT_DIR/f7952/eo_strict.discrete.patchspec" "$OUT_DIR/f7952/pc_strict.discrete.patchspec"

# -------- Family 9872/2608 (fixed output 2048, input 1024/2048/3072)
F9872_LOW_EO="$REPO_ROOT/traces/analysis/m5-family-rect-scan-20260303T183223Z/i1024_o2048/extract/package_000/serialized_executable_000.bin"
F9872_LOW_PC="$REPO_ROOT/traces/analysis/m5-family-rect-scan-20260303T183223Z/i1024_o2048/extract/package_000/serialized_executable_001.bin"
F9872_MID_EO="$REPO_ROOT/traces/dense-template-2048x2048-20260222T062027Z/extract/package_000/serialized_executable_000.bin"
F9872_MID_PC="$REPO_ROOT/traces/dense-template-2048x2048-20260222T062027Z/extract/package_000/serialized_executable_001.bin"
F9872_HIGH_EO="$REPO_ROOT/traces/analysis/m5-family-rect-scan-hi-20260303T183324Z/i3072_o2048/extract/package_000/serialized_executable_000.bin"
F9872_HIGH_PC="$REPO_ROOT/traces/analysis/m5-family-rect-scan-hi-20260303T183324Z/i3072_o2048/extract/package_000/serialized_executable_001.bin"

run_wordfield f9872 eo 1024 2048 3072 2048 "$F9872_LOW_EO" "$F9872_MID_EO" "$F9872_HIGH_EO" "$F9872_MID_EO"
run_wordfield f9872 pc 1024 2048 3072 2048 "$F9872_LOW_PC" "$F9872_MID_PC" "$F9872_HIGH_PC" "$F9872_MID_PC"
merge_patchspecs "$OUT_DIR/f9872/f9872_strict.full.patchspec" "$OUT_DIR/f9872/eo_strict.full.patchspec" "$OUT_DIR/f9872/pc_strict.full.patchspec"
merge_patchspecs "$OUT_DIR/f9872/f9872_strict.safe.patchspec" "$OUT_DIR/f9872/eo_strict.safe.patchspec" "$OUT_DIR/f9872/pc_strict.safe.patchspec"
merge_patchspecs "$OUT_DIR/f9872/f9872_strict.discrete.patchspec" "$OUT_DIR/f9872/eo_strict.discrete.patchspec" "$OUT_DIR/f9872/pc_strict.discrete.patchspec"

python3 - "$OUT_DIR" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
families = ["f7056", "f7952", "f9872"]
rows = []
for fam in families:
    row = {"family": fam}
    for kind in ["eo", "pc"]:
        rep = json.loads((out / fam / f"{kind}_strict.report.json").read_text())
        row[f"{kind}_baseline_mismatch"] = rep["baseline"]["mismatch_vs_target"]
        row[f"{kind}_v2_mismatch"] = rep["with_v2_spec"]["mismatch_vs_target"]
        row[f"{kind}_safe_core"] = rep["safe_core_byte_count"]
        row[f"{kind}_discrete"] = rep["discrete_flags_byte_count"]
        row[f"{kind}_unknown"] = rep["unknown_byte_count"]
    rows.append(row)
(out / "SUMMARY.json").write_text(json.dumps({"run_dir": str(out), "families": rows}, indent=2) + "\n")
with (out / "SUMMARY.txt").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(
            f"{r['family']}: "
            f"eo baseline={r['eo_baseline_mismatch']} v2={r['eo_v2_mismatch']} safe={r['eo_safe_core']} discrete={r['eo_discrete']} unknown={r['eo_unknown']} | "
            f"pc baseline={r['pc_baseline_mismatch']} v2={r['pc_v2_mismatch']} safe={r['pc_safe_core']} discrete={r['pc_discrete']} unknown={r['pc_unknown']}\n"
        )
print(out / "SUMMARY.txt")
PY

echo "done: $OUT_DIR"
