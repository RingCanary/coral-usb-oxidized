#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_REL="traces/analysis/phase3-conv2d-crossdim-oracle-matrix-20260306T132611Z"
PAIRS=(p32 p64 p128)
SUBSPLIT=4
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"

while (($# > 0)); do
  case "$1" in
    --artifact-rel)
      ARTIFACT_REL="$2"
      shift 2
      ;;
    --pairs)
      IFS=',' read -r -a PAIRS <<< "$2"
      shift 2
      ;;
    --pair)
      PAIRS+=("$2")
      shift 2
      ;;
    --subsplit)
      SUBSPLIT="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

ARTIFACT_DIR="$REPO_ROOT/$ARTIFACT_REL"
[[ -d "$ARTIFACT_DIR" ]] || { echo "artifact dir not found: $ARTIFACT_DIR" >&2; exit 1; }

RUN_ID="phase3-conv2d-eo-window-refine-probe-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_REL="traces/analysis/$RUN_ID"
OUT_DIR="$REPO_ROOT/$OUT_REL"
DUT_REL="traces/analysis/specv3-$RUN_ID-dut"
DUT_DIR="$REPO_ROOT/$DUT_REL"
REMOTE_ARTIFACT_DIR="$REMOTE_REPO/$ARTIFACT_REL"
REMOTE_OUT_DIR="$REMOTE_REPO/$OUT_REL"
REMOTE_DUT_DIR="$REMOTE_REPO/$DUT_REL"
mkdir -p "$OUT_DIR" "$DUT_DIR"

echo "run_id=$RUN_ID"
echo "artifact_rel=$ARTIFACT_REL"
echo "pairs=${PAIRS[*]}"
echo "subsplit=$SUBSPLIT"
echo "pi_host=$PI_HOST"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }
rsync_pi() { rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"; }

python3 - "$ARTIFACT_DIR" "$OUT_DIR" "$SUBSPLIT" "${PAIRS[@]}" <<'PY'
import json, pathlib, sys
artifact_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
subsplit = int(sys.argv[3])
pairs = sys.argv[4:]

windows = {
    'p32': [(242, 2292), (2292, 3236)],
    'p64': [(242, 2297), (2297, 3236)],
    'p128': [(242, 2289), (2289, 3236)],
}

for pair in pairs:
    prep = json.loads((artifact_dir / pair / 'PREP_SUMMARY.json').read_text())
    pair_out = out_dir / pair
    pair_out.mkdir(parents=True, exist_ok=True)

    rules = []
    for line in (artifact_dir / pair / 'eo_oracle.patchspec').read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        plen_s, off_s, val_s = s.split()[:3]
        rules.append((int(plen_s), int(off_s), int(val_s, 16)))
    rules.sort(key=lambda x: x[1])

    def write_patch(path, subset):
        lines = [f'# auto-generated subset for {pair}']
        for plen, off, val in subset:
            lines.append(f'{plen} {off} 0x{val:02x}')
        path.write_text('\n'.join(lines) + '\n')

    write_patch(pair_out / 'eo_full.patchspec', rules)
    meta = {
        'pair_id': pair,
        'artifact_rel': str(artifact_dir.relative_to(pathlib.Path.cwd())),
        'anchor_model_rel': str(pathlib.Path(prep['anchor_model']).resolve().relative_to(pathlib.Path.cwd())),
        'target_model_rel': str(pathlib.Path(prep['target_model']).resolve().relative_to(pathlib.Path.cwd())),
        'eo_rule_count': prep['eo_rule_count'],
        'pc_rule_count': prep['pc_rule_count'],
        'param_equal': prep['param_equal'],
        'windows': [],
    }
    if pair == 'p32':
        meta['input_bytes'] = 32 * 32 * 32
        meta['output_bytes'] = 32 * 32 * 32
    elif pair == 'p64':
        meta['input_bytes'] = 32 * 32 * 64
        meta['output_bytes'] = 32 * 32 * 64
    elif pair == 'p128':
        meta['input_bytes'] = 32 * 32 * 128
        meta['output_bytes'] = 32 * 32 * 128
    else:
        raise RuntimeError(f'unknown pair: {pair}')

    for widx, (start, end) in enumerate(windows[pair]):
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
            kept = [r for r in rules if r not in subset]
            name = f'eo_minus_w{widx:02d}_g{idx:02d}'
            write_patch(pair_out / f'{name}.patchspec', kept)
            bins.append({
                'name': name,
                'start': sub_start,
                'end': sub_end,
                'rule_count': len(subset),
            })
        meta['windows'].append({
            'window_index': widx,
            'start': start,
            'end': end,
            'rule_count': len(window_rules),
            'bins': bins,
        })

    (pair_out / 'REFINE_META.json').write_text(json.dumps(meta, indent=2) + '\n')
    with (pair_out / 'REFINE_META.txt').open('w', encoding='utf-8') as f:
        f.write(f'pair={pair} eo_rule_count={prep["eo_rule_count"]}\n')
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
  local pair="$1"
  local case_name="$2"
  local model_remote="$3"
  local input_bytes="$4"
  local output_bytes="$5"
  shift 5
  local extra_args=("$@")
  local local_log="$DUT_DIR/$pair/${case_name}.log"
  mkdir -p "$(dirname "$local_log")"
  local cmd="cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$model_remote' --firmware '$FIRMWARE_REMOTE' --bootstrap-known-good-order --input-bytes '$input_bytes' --output-bytes '$output_bytes' --reset-before-claim --post-reset-sleep-ms 1200"
  for arg in "${extra_args[@]}"; do
    cmd+=" '$arg'"
  done
  echo "[$pair] $case_name"
  ssh_run "$cmd" </dev/null > "$local_log" 2>&1 || true
}

for pair in "${PAIRS[@]}"; do
  meta="$OUT_DIR/$pair/REFINE_META.json"
  anchor_rel=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['anchor_model_rel'])
PY
)
  target_rel=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['target_model_rel'])
PY
)
  input_bytes=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['input_bytes'])
PY
)
  output_bytes=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['output_bytes'])
PY
)
  remote_pair_dir="$REMOTE_OUT_DIR/$pair"
  anchor_model_remote="$REMOTE_REPO/$anchor_rel"
  target_model_remote="$REMOTE_REPO/$target_rel"

  run_case "$pair" target_baseline "$target_model_remote" "$input_bytes" "$output_bytes"
  run_case "$pair" anchor_baseline "$anchor_model_remote" "$input_bytes" "$output_bytes"
  run_case "$pair" eo_full "$anchor_model_remote" "$input_bytes" "$output_bytes" \
    --instruction-patch-spec "$remote_pair_dir/eo_full.patchspec"

  while IFS= read -r case_name; do
    run_case "$pair" "$case_name" "$anchor_model_remote" "$input_bytes" "$output_bytes" \
      --instruction-patch-spec "$remote_pair_dir/${case_name}.patchspec"
  done < <(python3 - <<'PY' "$meta"
import json,sys
meta=json.load(open(sys.argv[1]))
for w in meta['windows']:
    for b in w['bins']:
        print(b['name'])
PY
)
done

python3 - "$OUT_DIR" "$DUT_DIR" <<'PY'
import json, pathlib, re, sys
out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])
root_lines = [f'run_id={out_dir.name}']
for pair_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
    meta = json.loads((pair_dir / 'REFINE_META.json').read_text())
    logs_dir = dut_dir / pair_dir.name
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
    with (pair_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
        f.write(f'pair={pair_dir.name} target_hash={target_hash}\n')
        for fixed in ['target_baseline','anchor_baseline','eo_full']:
            r = cases.get(fixed, {})
            f.write(f"{fixed}: pass={r.get('pass')} hash={r.get('hash')} error={r.get('error')}\n")
        for w in meta['windows']:
            f.write(f"window w{w['window_index']:02d}: {w['start']}..{w['end']} rule_count={w['rule_count']}\n")
            for b in w['bins']:
                c = cases.get(b['name'], {})
                f.write(
                    f"  {b['name']}: {b['start']}..{b['end']} rule_count={b['rule_count']} pass={c.get('pass', False)} hash={c.get('hash')} hash_eq_target={c.get('hash') == target_hash} error={c.get('error')}\n"
                )
    root_lines.append(f'[{pair_dir.name}] target_hash={target_hash}')
    for w in meta['windows']:
        fatal=[]; semantic=[]; exact=[]
        for b in w['bins']:
            c=cases.get(b['name'], {})
            if not c.get('pass', False):
                fatal.append(b['name'])
            elif c.get('hash') == target_hash:
                exact.append(b['name'])
            else:
                semantic.append(b['name'])
        root_lines.append(f"  w{w['window_index']:02d} fatal_bins={fatal} semantic_bins={semantic} exact_bins={exact}")
(out_dir / 'SUMMARY.txt').write_text('\n'.join(root_lines) + '\n')
print(out_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
