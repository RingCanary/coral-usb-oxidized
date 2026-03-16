#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="phase3-conv2d-layout-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "run_id=$RUN_ID"

cases=(
  "32 32"
  "64 64"
  "64 128"
  "128 64"
  "128 128"
)

for spec in "${cases[@]}"; do
  read -r IC OC <<<"$spec"
  CASE_ID="ic${IC}_oc${OC}"
  CASE_DIR="$OUT_DIR/$CASE_ID"
  "$REPO_ROOT/tools/archive/conv_layout_probe.py" \
    --out-dir "$CASE_DIR" \
    --height 32 --width 32 \
    --in-channels "$IC" --out-channels "$OC" \
    --kernel-size 1 --stride 1 --padding same \
    --reference 0,0,0,0 \
    --probe 0,0,0,1 \
    --probe 0,0,1,0 \
    --probe 0,0,1,1 \
    --probe 0,0,$((IC/2-1)),$((OC/2-1)) \
    --probe 0,0,$((IC-1)),$((OC-1)) >/dev/null
  python3 - <<'PY' "$CASE_DIR/layout_probe.json" "$CASE_DIR/SUMMARY.txt" "$IC" "$OC"
import json, sys, pathlib
probe_path, summary_path, ic_s, oc_s = sys.argv[1:5]
ic = int(ic_s); oc = int(oc_s)
obj = json.loads(pathlib.Path(probe_path).read_text())
def block_widths(out_channels):
    rem = out_channels
    out = []
    while rem > 64:
        out.append(64)
        rem -= 64
    out.append(rem)
    return out

def expected_offset(ic0, oc0, out_channels):
    block_start = 0
    rem_out = oc0
    for bw in block_widths(out_channels):
        if rem_out < bw:
            return block_start + (bw * 8) + ((ic0 // 4) * (bw * 4)) + (rem_out * 4) + (ic0 % 4)
        block_start += bw * (8 + ic)
        rem_out -= bw
    raise RuntimeError('bad oc')
checks = [
    (0, 1),
    (1, 0),
    (1, 1),
    (ic//2 - 1, oc//2 - 1),
    (ic - 1, oc - 1),
]
by_pair = {(r['in_channel'], r['out_channel']): r for r in obj['records']}
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(f"in_channels={ic} out_channels={oc} total_prefix={oc * 8}\n")
    for ic0, oc0 in checks:
        rec = by_pair[(ic0, oc0)]
        expected = expected_offset(ic0, oc0, oc)
        actual = rec.get('mapping_candidate_offset')
        f.write(f"ic={ic0} oc={oc0} expected={expected} actual={actual} match={expected == actual}\n")
PY
done

python3 - <<'PY' "$OUT_DIR" "$OUT_DIR/SUMMARY.txt"
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
lines = [f"run_id={root.name}"]
for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    txt = (case_dir / 'SUMMARY.txt').read_text()
    lines.append(f"[{case_dir.name}]")
    lines.extend('  ' + line for line in txt.strip().splitlines())
    matches = re.findall(r'match=(True|False)', txt)
    lines.append(f"  all_match={all(m == 'True' for m in matches)}")
pathlib.Path(sys.argv[2]).write_text('\n'.join(lines) + '\n')
print(sys.argv[2])
PY

echo "done: $OUT_DIR"
