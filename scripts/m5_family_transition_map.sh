#!/usr/bin/env bash
set -euo pipefail

RUN_ID="m5-family-transition-map-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="traces/analysis/${RUN_ID}"
INPUT_TSV="docs/artifacts/instruction-dim-field-20260301/dense_instruction_size_table.tsv"

mkdir -p "${OUT_DIR}"

cargo run --bin family_transition_map -- \
  --input "${INPUT_TSV}" \
  --out-json "${OUT_DIR}/transition_map.json" \
  --out-md "${OUT_DIR}/transition_map.md" \
  > "${OUT_DIR}/family_transition_map.stdout.log" 2>&1

python3 - <<'PY' "${OUT_DIR}/transition_map.json" "${OUT_DIR}/SUMMARY.txt"
import json, sys
src = sys.argv[1]
out = sys.argv[2]
obj = json.load(open(src, 'r', encoding='utf-8'))
recurrent = obj.get('recurrent_families', [f for f in obj['families'] if len(f['dims']) >= 2])
with open(out, 'w', encoding='utf-8') as f:
    f.write(f"run_id={src.split('/')[-2]}\n")
    f.write(f"family_count={obj['family_count']}\n")
    f.write(f"recurrent_family_count={obj.get('recurrent_family_count', len(recurrent))}\n")
    f.write(f"record_count={len(obj['records'])}\n")
    f.write("families:\n")
    for fam in obj['families']:
        f.write(f"  - {fam['family_id']}: dims={fam['dims']}\n")
    f.write("recurrent_families:\n")
    for fam in recurrent:
        f.write(f"  - {fam['family_id']}: dims={fam['dims']}\n")
    f.write("transitions:\n")
    for tr in obj['transitions']:
        f.write(f"  - {tr['from_family_id']}@{tr['from_dim']} -> {tr['to_family_id']}@{tr['to_dim']}\n")
print(out)
PY

echo "run_dir=${OUT_DIR}"
