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

RUN_ID="m5-eo-window-refine-probe-$(date -u +%Y%m%dT%H%M%SZ)"
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

# Windows chosen from prior 16/32-way DUT refinement runs.
windows = {
    'f7056': [(338, 540), (902, 1176), (5300, 5658)],
    'f8976': [(338, 680), (902, 1784), (6830, 8501)],
    'f9872': [(338, 646), (872, 1304), (1336, 1784)],
}
subsplit = 4

for family, fam_windows in windows.items():
    prep = json.loads((artifact_dir / family / 'PREP_SUMMARY.json').read_text())
    fam_out = out_dir / family
    fam_out.mkdir(parents=True, exist_ok=True)

    rules = []
    for line in (artifact_dir / family / 'eo_oracle.patchspec').read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        plen_s, off_s, val_s = s.split()[:3]
        rules.append((int(plen_s), int(off_s), int(val_s, 16)))

    meta = {
        'family_id': family,
        'anchor_dims': prep['anchor_dims'],
        'target_dims': prep['target_dims'],
        'anchor_model_rel': str(pathlib.Path(prep['anchor_model']).resolve().relative_to(pathlib.Path.cwd())),
        'target_model_rel': str(pathlib.Path(prep['target_model']).resolve().relative_to(pathlib.Path.cwd())),
        'target_param_len': prep['target_param_len'],
        'windows': [],
    }

    out_lines = ['# copied full eo oracle patchspec']
    for plen, off, val in rules:
        out_lines.append(f'{plen} {off} 0x{val:02x}')
    (fam_out / 'eo_full.patchspec').write_text('\n'.join(out_lines) + '\n')

    for widx, (start, end) in enumerate(fam_windows):
        window_rules = [r for r in rules if start <= r[1] <= end]
        bins = []
        n = len(window_rules)
        for idx in range(subsplit):
            lo = (idx * n) // subsplit
            hi = ((idx + 1) * n) // subsplit
            subset = window_rules[lo:hi]
            if subset:
                sub_start = subset[0][1]
                sub_end = subset[-1][1]
            else:
                sub_start = sub_end = None
            bins.append({
                'name': f'w{widx:02d}_g{idx:02d}',
                'parent_window_index': widx,
                'parent_range': [start, end],
                'start': sub_start,
                'end': sub_end,
                'rule_count': len(subset),
            })
            kept = [r for r in rules if r not in subset]
            lines = [f'# minus subset {family} {widx}:{idx} {sub_start}..{sub_end}']
            for plen, off, val in kept:
                lines.append(f'{plen} {off} 0x{val:02x}')
            (fam_out / f"eo_minus_w{widx:02d}_g{idx:02d}.patchspec").write_text('\n'.join(lines) + '\n')
        meta['windows'].append({
            'window_index': widx,
            'start': start,
            'end': end,
            'rule_count': len(window_rules),
            'bins': bins,
        })

    (fam_out / 'REFINE_META.json').write_text(json.dumps(meta, indent=2) + '\n')
    with (fam_out / 'REFINE_META.txt').open('w', encoding='utf-8') as f:
        f.write(f"family={family} anchor={prep['anchor_dims']} target={prep['target_dims']}\n")
        for w in meta['windows']:
            f.write(f"w{w['window_index']:02d}: {w['start']}..{w['end']} rule_count={w['rule_count']}\n")
            for b in w['bins']:
                f.write(f"  {b['name']}: {b['start']}..{b['end']} rule_count={b['rule_count']}\n")
PY

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

for family in f7056 f8976 f9872; do
  meta="$OUT_DIR/$family/REFINE_META.json"
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
  param_remote="$REMOTE_ARTIFACT_DIR/$family/target_param_stream.bin"

  run_case "$family" target_baseline "$target_model_remote" "$target_in" "$target_out" "$param_len"
  run_case "$family" anchor_param_only "$anchor_model_remote" "$target_in" "$target_out" "$param_len" \
    --param-stream-override-file "$param_remote"
  run_case "$family" eo_full "$anchor_model_remote" "$target_in" "$target_out" "$param_len" \
    --instruction-patch-spec "$remote_fam_dir/eo_full.patchspec" \
    --param-stream-override-file "$param_remote"

  while IFS= read -r name; do
    run_case "$family" "$name" "$anchor_model_remote" "$target_in" "$target_out" "$param_len" \
      --instruction-patch-spec "$remote_fam_dir/${name}.patchspec" \
      --param-stream-override-file "$param_remote"
  done < <(python3 - <<'PY' "$meta"
import json,sys
meta=json.load(open(sys.argv[1]))
for w in meta['windows']:
    for b in w['bins']:
        print(f"eo_minus_{b['name']}")
PY
)
done

rsync_pi "$PI_HOST:$REMOTE_DUT_DIR/" "$DUT_DIR/" >/dev/null

python3 - "$OUT_DIR" "$DUT_DIR" <<'PY'
import json, pathlib, re, sys
out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])
root = {'run_id': out_dir.name, 'families': []}
for fam_dir in sorted(out_dir.iterdir()):
    if not fam_dir.is_dir():
        continue
    meta = json.loads((fam_dir / 'REFINE_META.json').read_text())
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
    target_hash = cases.get('target_baseline', {}).get('hash')
    windows = []
    for w in meta['windows']:
        bins = []
        for b in w['bins']:
            c = cases.get(f"eo_minus_{b['name']}", {})
            bins.append({
                **b,
                'pass': c.get('pass', False),
                'hash': c.get('hash'),
                'error': c.get('error'),
                'hash_eq_target': c.get('hash') == target_hash,
            })
        windows.append({**w, 'bins': bins})
    fam = {
        'family_id': fam_dir.name,
        'target_baseline': cases.get('target_baseline', {}),
        'anchor_param_only': cases.get('anchor_param_only', {}),
        'eo_full': cases.get('eo_full', {}),
        'windows': windows,
    }
    (fam_dir / 'SUMMARY.json').write_text(json.dumps(fam, indent=2) + '\n')
    with (fam_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
        f.write(f"family={fam_dir.name} target_hash={target_hash}\n")
        for k in ['target_baseline','anchor_param_only','eo_full']:
            r = fam[k]
            f.write(f"{k}: pass={r.get('pass')} hash={r.get('hash')} error={r.get('error')}\n")
        for w in windows:
            f.write(f"window w{w['window_index']:02d}: {w['start']}..{w['end']} rule_count={w['rule_count']}\n")
            for b in w['bins']:
                f.write(f"  {b['name']}: {b['start']}..{b['end']} rule_count={b['rule_count']} pass={b['pass']} hash={b['hash']} hash_eq_target={b['hash_eq_target']} error={b['error']}\n")
    root['families'].append(fam)
(out_dir / 'SUMMARY.json').write_text(json.dumps(root, indent=2) + '\n')
with (out_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
    f.write(f"run_id={out_dir.name}\n")
    for fam in root['families']:
        f.write(f"[{fam['family_id']}] target_hash={fam['target_baseline'].get('hash')}\n")
        for w in fam['windows']:
            fatal = [b['name'] for b in w['bins'] if not b['pass']]
            neutral = [b['name'] for b in w['bins'] if b['hash_eq_target']]
            f.write(f"  w{w['window_index']:02d} fatal_bins={fatal} neutral_bins={neutral}\n")
print(out_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
