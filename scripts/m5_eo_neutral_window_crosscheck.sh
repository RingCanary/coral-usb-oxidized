#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARTIFACT_REL="traces/analysis/m5-crossdim-oracle-matrix-20260306T103420Z"
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/tmp/coral-m5-eo-probe-src}"

RUN_ID="m5-eo-neutral-window-crosscheck-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_REL="traces/analysis/$RUN_ID"
OUT_DIR="$REPO_ROOT/$OUT_REL"
DUT_REL="traces/analysis/specv3-$RUN_ID-dut"
DUT_DIR="$REPO_ROOT/$DUT_REL"
ARTIFACT_DIR="$REPO_ROOT/$ARTIFACT_REL"
REMOTE_ARTIFACT_DIR="$REMOTE_REPO/$ARTIFACT_REL"
REMOTE_OUT_DIR="$REMOTE_REPO/$OUT_REL"
REMOTE_DUT_DIR="$REMOTE_REPO/$DUT_REL"
mkdir -p "$OUT_DIR" "$DUT_DIR"

echo "run_id=$RUN_ID"
echo "artifact_rel=$ARTIFACT_REL"
echo "pi_host=$PI_HOST"

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o IdentitiesOnly=yes
  -i "$SSH_KEY"
)

ssh_run() {
  ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"
}

rsync_pi() {
  rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"
}

python3 - "$ARTIFACT_DIR" "$OUT_DIR" <<'PY'
import json, pathlib, sys
artifact_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])

families = {
    'f7056': [(1208, 1448), (4258, 4476), (5137, 5204)],
    'f9872': [(2862, 3206), (3210, 3548), (5090, 5318), (5322, 5660)],
}

for family, windows in families.items():
    prep = json.loads((artifact_dir / family / 'PREP_SUMMARY.json').read_text())
    fam_out = out_dir / family
    fam_out.mkdir(parents=True, exist_ok=True)

    anchor_in, anchor_out = prep['anchor_dims']
    target_in, target_out = prep['target_dims']

    reverse = {
        'family_id': family,
        'forward_anchor_dims': prep['anchor_dims'],
        'forward_target_dims': prep['target_dims'],
        'anchor_dims': prep['target_dims'],
        'target_dims': prep['anchor_dims'],
        'windows': [
            {'name': f'w{i:02d}', 'start': a, 'end': b}
            for i, (a, b) in enumerate(windows)
        ],
        'anchor_model_rel': f"{artifact_dir.relative_to(pathlib.Path.cwd())}/build/i{target_in}_o{target_out}/dense_{target_in}x{target_out}_quant_edgetpu.tflite",
        'target_model_rel': f"{artifact_dir.relative_to(pathlib.Path.cwd())}/build/i{anchor_in}_o{anchor_out}/dense_{anchor_in}x{anchor_out}_quant_edgetpu.tflite",
        'anchor_eo_exec': artifact_dir / 'build' / f'i{target_in}_o{target_out}' / 'extract/package_000/serialized_executable_000.bin',
        'target_eo_exec': artifact_dir / 'build' / f'i{anchor_in}_o{anchor_out}' / 'extract/package_000/serialized_executable_000.bin',
        'target_model_abs': artifact_dir / 'build' / f'i{anchor_in}_o{anchor_out}' / f'dense_{anchor_in}x{anchor_out}_quant_edgetpu.tflite',
        'target_param_len': prep['target_param_len'],
    }
    (fam_out / 'REVERSE_META.json').write_text(json.dumps({
        **reverse,
        'anchor_eo_exec': str(reverse['anchor_eo_exec']),
        'target_eo_exec': str(reverse['target_eo_exec']),
        'target_model_abs': str(reverse['target_model_abs']),
    }, indent=2) + '\n')
PY

prepare_family() {
  local family="$1"
  local fam_out="$OUT_DIR/$family"
  local meta="$fam_out/REVERSE_META.json"
  local anchor_eo_exec target_eo_exec target_model_abs
  anchor_eo_exec=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['anchor_eo_exec'])
PY
)
  target_eo_exec=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['target_eo_exec'])
PY
)
  target_model_abs=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['target_model_abs'])
PY
)

  cargo run --quiet --bin instruction_chunk_patchspec -- \
    --base-exec "$anchor_eo_exec" \
    --target-exec "$target_eo_exec" \
    --chunk-index 0 \
    --out-patchspec "$fam_out/eo_full_reverse.patchspec" \
    > "$fam_out/eo_full_reverse.stdout.log"

  cargo run --quiet --bin model_param_stream_dump -- \
    --model "$target_model_abs" \
    --out "$fam_out/target_param_stream.bin" \
    --metadata-out "$fam_out/target_param_stream.json" \
    > "$fam_out/target_param_stream.stdout.log"

  python3 - "$fam_out" <<'PY'
import json, pathlib, sys
fam_out = pathlib.Path(sys.argv[1])
meta = json.loads((fam_out / 'REVERSE_META.json').read_text())
full = fam_out / 'eo_full_reverse.patchspec'
lines = []
for line in full.read_text().splitlines():
    s = line.strip()
    if s and not s.startswith('#'):
        plen_s, off_s, val_s = s.split()[:3]
        lines.append((int(plen_s), int(off_s), int(val_s, 16)))
for w in meta['windows']:
    kept = [x for x in lines if not (w['start'] <= x[1] <= w['end'])]
    removed = [x for x in lines if (w['start'] <= x[1] <= w['end'])]
    path = fam_out / f"eo_minus_{w['name']}.patchspec"
    out = [f"# reverse neutral-window crosscheck minus {w['name']} {w['start']}..{w['end']}"]
    for plen, off, val in kept:
        out.append(f"{plen} {off} 0x{val:02x}")
    path.write_text('\n'.join(out) + '\n')
    w['removed_rule_count'] = len(removed)
param_meta = json.loads((fam_out / 'target_param_stream.json').read_text())
meta['eo_full_rule_count'] = len(lines)
meta['target_param_len_dumped'] = param_meta['param_len']
(fam_out / 'REVERSE_META.json').write_text(json.dumps(meta, indent=2) + '\n')
with (fam_out / 'REVERSE_META.txt').open('w', encoding='utf-8') as f:
    f.write(f"family={meta['family_id']} reverse_anchor={meta['anchor_dims']} reverse_target={meta['target_dims']} eo_full_rule_count={meta['eo_full_rule_count']}\n")
    for w in meta['windows']:
        f.write(f"{w['name']}: {w['start']}..{w['end']} removed_rules={w['removed_rule_count']}\n")
PY
}

prepare_family f7056
prepare_family f9872

ssh_run "mkdir -p '$REMOTE_OUT_DIR' '$REMOTE_DUT_DIR' '$REMOTE_SRC_DIR'"
rsync_pi "$ARTIFACT_DIR/" "$PI_HOST:$REMOTE_ARTIFACT_DIR/" >/dev/null
rsync_pi "$OUT_DIR/" "$PI_HOST:$REMOTE_OUT_DIR/" >/dev/null
rsync -av --delete --exclude '.git' --exclude 'target' --exclude 'traces' \
  -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" \
  "$REPO_ROOT/" "$PI_HOST:$REMOTE_SRC_DIR/" >/dev/null
ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --example rusb_serialized_exec_replay >/dev/null"

run_case() {
  local family="$1"
  local case_name="$2"
  local model_remote="$3"
  local input_bytes="$4"
  local output_bytes="$5"
  local param_len="$6"
  shift 6
  local extra_args=("$@")
  local local_log="$DUT_DIR/$family/${case_name}.log"
  local remote_log="$REMOTE_DUT_DIR/$family/${case_name}.log"
  mkdir -p "$(dirname "$local_log")"
  local cmd="cd '$REMOTE_SRC_DIR' && sudo cargo run --example rusb_serialized_exec_replay -- --model '$model_remote' --firmware '$FIRMWARE_REMOTE' --chunk-size 1048576 --param-stream-max-bytes '$param_len' --bootstrap-known-good-order --input-bytes '$input_bytes' --output-bytes '$output_bytes' --reset-before-claim --post-reset-sleep-ms 1200"
  for arg in "${extra_args[@]}"; do
    cmd+=" '$arg'"
  done
  echo "[$family] $case_name"
  ssh_run "$cmd" </dev/null > "$local_log" 2>&1 || true
  cat "$local_log" | ssh "${SSH_OPTS[@]}" "$PI_HOST" "mkdir -p '$REMOTE_DUT_DIR/$family' && cat > '$remote_log'"
}

for family in f7056 f9872; do
  meta="$OUT_DIR/$family/REVERSE_META.json"
  anchor_model_rel=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['anchor_model_rel'])
PY
)
  target_model_rel=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['target_model_rel'])
PY
)
  target_in=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['target_dims'][0])
PY
)
  target_out=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['target_dims'][1])
PY
)
  param_len=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['target_param_len'])
PY
)
  remote_fam_dir="$REMOTE_OUT_DIR/$family"
  anchor_model_remote="$REMOTE_REPO/$anchor_model_rel"
  target_model_remote="$REMOTE_REPO/$target_model_rel"
  param_remote="$remote_fam_dir/target_param_stream.bin"

  run_case "$family" target_baseline "$target_model_remote" "$target_in" "$target_out" "$param_len"
  run_case "$family" anchor_param_only "$anchor_model_remote" "$target_in" "$target_out" "$param_len" \
    --param-stream-override-file "$param_remote"
  run_case "$family" eo_full_reverse "$anchor_model_remote" "$target_in" "$target_out" "$param_len" \
    --instruction-patch-spec "$remote_fam_dir/eo_full_reverse.patchspec" \
    --param-stream-override-file "$param_remote"

  while IFS= read -r item; do
    name="$(echo "$item" | cut -d' ' -f1)"
    run_case "$family" "eo_minus_${name}" "$anchor_model_remote" "$target_in" "$target_out" "$param_len" \
      --instruction-patch-spec "$remote_fam_dir/eo_minus_${name}.patchspec" \
      --param-stream-override-file "$param_remote"
  done < <(python3 - <<'PY' "$meta"
import json,sys
for w in json.load(open(sys.argv[1]))['windows']:
    print(w['name'])
PY
)
done

rsync_pi "$PI_HOST:$REMOTE_DUT_DIR/" "$DUT_DIR/" >/dev/null

python3 - "$OUT_DIR" "$DUT_DIR" <<'PY'
import json, pathlib, re, sys
out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])
rows = []
for fam_dir in sorted(out_dir.iterdir()):
    if not fam_dir.is_dir():
        continue
    meta = json.loads((fam_dir / 'REVERSE_META.json').read_text())
    logs_dir = dut_dir / fam_dir.name
    cases = {}
    for log_path in sorted(logs_dir.glob('*.log')):
        txt = log_path.read_text(errors='ignore')
        out = re.findall(r'Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)', txt)
        err = re.findall(r'Error: (.+)', txt)
        cases[log_path.stem] = {
            'pass': bool(out),
            'bytes': int(out[-1][0]) if out else None,
            'hash': out[-1][1] if out else None,
            'error': err[-1] if err else None,
        }
    target_hash = cases['target_baseline']['hash']
    evals = []
    for w in meta['windows']:
        name = f"eo_minus_{w['name']}"
        c = cases.get(name, {})
        evals.append({
            'name': w['name'],
            'start': w['start'],
            'end': w['end'],
            'removed_rule_count': w['removed_rule_count'],
            'pass': c.get('pass', False),
            'hash': c.get('hash'),
            'error': c.get('error'),
            'hash_eq_target': c.get('hash') == target_hash,
        })
    summary = {
        'family_id': fam_dir.name,
        'meta': meta,
        'target_baseline': cases.get('target_baseline', {}),
        'anchor_param_only': cases.get('anchor_param_only', {}),
        'eo_full_reverse': cases.get('eo_full_reverse', {}),
        'window_cases': evals,
    }
    (fam_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (fam_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
        f.write(f"family={fam_dir.name} target_hash={target_hash}\n")
        for k in ['target_baseline','anchor_param_only','eo_full_reverse']:
            r = summary[k]
            f.write(f"{k}: pass={r.get('pass')} hash={r.get('hash')} error={r.get('error')}\n")
        for w in evals:
            f.write(f"eo_minus_{w['name']}: range={w['start']}..{w['end']} removed_rules={w['removed_rule_count']} pass={w['pass']} hash={w['hash']} hash_eq_target={w['hash_eq_target']} error={w['error']}\n")
    rows.append(summary)
root = {'run_id': out_dir.name, 'families': rows}
(out_dir / 'SUMMARY.json').write_text(json.dumps(root, indent=2) + '\n')
with (out_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
    f.write(f"run_id={out_dir.name}\n")
    for fam in rows:
        f.write(f"[{fam['family_id']}] target_hash={fam['target_baseline'].get('hash')} eo_full_hash_eq_target={fam['eo_full_reverse'].get('hash') == fam['target_baseline'].get('hash')}\n")
        retained = [w['name'] for w in fam['window_cases'] if w['hash_eq_target']]
        f.write(f"  reverse_hash_neutral_windows={retained}\n")
print(out_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
