#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="phase4-conv2d-k3-family-scout-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "run_id=$RUN_ID"

MODE="${MODE:-fixed}"
SAME_PRODUCT="${SAME_PRODUCT:-1024}"
HEIGHTS_CSV="${HEIGHTS_CSV:-1,2,4,8,16,32,64,128}"
CHANNELS_CSV="${CHANNELS_CSV:-64}"
OUT_CHANNELS_CSV="${OUT_CHANNELS_CSV:-}"
CASES_CSV="${CASES_CSV:-}"

cases=()

if [[ "$MODE" == "sameprod" ]]; then
  IFS=',' read -r -a heights <<<"$HEIGHTS_CSV"
  IFS=',' read -r -a channels <<<"$CHANNELS_CSV"
  if [[ -n "$OUT_CHANNELS_CSV" ]]; then
    IFS=',' read -r -a out_channels <<<"$OUT_CHANNELS_CSV"
  else
    out_channels=("${channels[@]}")
  fi
  if (( ${#channels[@]} != ${#out_channels[@]} )); then
    echo "CHANNELS_CSV and OUT_CHANNELS_CSV length mismatch" >&2
    exit 1
  fi
  for idx in "${!channels[@]}"; do
    ic="${channels[$idx]}"
    oc="${out_channels[$idx]}"
    for H in "${heights[@]}"; do
      if (( H <= 0 )) || (( SAME_PRODUCT % H != 0 )); then
        echo "invalid height for same-product scan: $H (same_product=$SAME_PRODUCT)" >&2
        exit 1
      fi
      W=$((SAME_PRODUCT / H))
      cases+=("$H $W $ic $oc")
    done
  done
elif [[ "$MODE" == "explicit" ]]; then
  [[ -n "$CASES_CSV" ]] || {
    echo "CASES_CSV is required when MODE=explicit" >&2
    exit 1
  }
  IFS=',' read -r -a explicit_cases <<<"$CASES_CSV"
  for entry in "${explicit_cases[@]}"; do
    [[ "$entry" =~ ^([0-9]+)x([0-9]+):([0-9]+):([0-9]+)$ ]] || {
      echo "invalid explicit case: $entry (expected HxW:IC:OC)" >&2
      exit 1
    }
    H="${BASH_REMATCH[1]}"
    W="${BASH_REMATCH[2]}"
    ic="${BASH_REMATCH[3]}"
    oc="${BASH_REMATCH[4]}"
    cases+=("$H $W $ic $oc")
  done
else
  cases=(
    "32 32 64 64"
    "32 32 64 128"
    "32 32 128 64"
  )
fi

printf "case_id\theight\twidth\tin_channels\tout_channels\tkernel_size\teo_instr\tpc_instr\tparam_bytes\tinput_bytes\toutput_bytes\n" > "$OUT_DIR/size_table.tsv"

for spec in "${cases[@]}"; do
  read -r H W IC OC <<<"$spec"
  CASE_ID="h${H}_w${W}_ic${IC}_oc${OC}_k3"
  CASE_DIR="$OUT_DIR/$CASE_ID"
  echo "[$CASE_ID] compile/extract"
  "$REPO_ROOT/tools/conv_template_pipeline.sh" \
    --out-dir "$CASE_DIR" \
    --height "$H" \
    --width "$W" \
    --in-channels "$IC" \
    --out-channels "$OC" \
    --kernel-size 3 \
    --stride 1 \
    --padding same \
    --init-mode random_uniform >/dev/null
  python3 - <<'PY' "$CASE_ID" "$CASE_DIR/exec_parse.json" "$OUT_DIR/size_table.tsv"
import json, sys, pathlib
case_id, parse_path, tsv_path = sys.argv[1:4]
obj = json.loads(pathlib.Path(parse_path).read_text())
dir_entry = next(iter(obj['directories'].values()))
execs = dir_entry['executables']
eo_exec = next(e for e in execs if e['type_name'] == 'EXECUTION_ONLY')
pc_exec = next(e for e in execs if e['type_name'] == 'PARAMETER_CACHING')
reports = obj['reports']
eo = next(r for r in reports if r['type_name'] == 'EXECUTION_ONLY')
row = [
    case_id,
    str(eo['input_layers']['dims_yxz'][0][0]),
    str(eo['input_layers']['dims_yxz'][0][1]),
    str(eo['input_layers']['dims_yxz'][0][2]),
    str(eo['output_layers']['dims_yxz'][0][2]),
    '3',
    str(eo_exec['instruction_total_bytes']),
    str(pc_exec['instruction_total_bytes']),
    str(pc_exec['parameters_size_bytes']),
    str(eo['input_layers']['layers'][0]['size_bytes']),
    str(eo['output_layers']['layers'][0]['size_bytes']),
]
with open(tsv_path, 'a', encoding='utf-8') as f:
    f.write('\t'.join(row) + '\n')
PY
done

SUMMARY_SAME_PRODUCT="$SAME_PRODUCT"
if [[ "$MODE" == "explicit" ]]; then
  SUMMARY_SAME_PRODUCT="mixed"
fi

python3 - <<'PY' "$OUT_DIR/size_table.tsv" "$OUT_DIR/SUMMARY.txt" "$OUT_DIR/families.json" "$MODE" "$SUMMARY_SAME_PRODUCT"
import csv, json, pathlib, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
mode = sys.argv[4]
same_product = sys.argv[5]
families = {}
for row in rows:
    key = f"k{row['kernel_size']}_eo{row['eo_instr']}_pc{row['pc_instr']}_param{row['param_bytes']}"
    families.setdefault(key, []).append({
        'case_id': row['case_id'],
        'height': int(row['height']),
        'width': int(row['width']),
        'in_channels': int(row['in_channels']),
        'out_channels': int(row['out_channels']),
        'input_bytes': int(row['input_bytes']),
        'output_bytes': int(row['output_bytes']),
    })
pathlib.Path(sys.argv[3]).write_text(json.dumps({'families': families}, indent=2) + '\n')
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    f.write(f"run_id={pathlib.Path(sys.argv[1]).parent.name}\n")
    f.write(f"mode={mode}\n")
    f.write(f"same_product={same_product}\n")
    f.write(f"case_count={len(rows)}\n")
    f.write(f"family_count={len(families)}\n")
    for key, vals in sorted(families.items()):
        dims = [f"{v['height']}x{v['width']}x{v['in_channels']}->{v['out_channels']}" for v in vals]
        f.write(f"{key}: {dims}\n")
print(sys.argv[2])
PY

echo "done: $OUT_DIR"
