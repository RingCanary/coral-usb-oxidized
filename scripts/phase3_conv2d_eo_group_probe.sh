#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_REL="traces/analysis/phase3-conv2d-crossdim-oracle-matrix-20260306T132611Z"
PAIRS=(p32 p64 p128)
GROUP_COUNT=8
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/tmp/coral-phase3-conv2d-eo-probe-src}"

while (($# > 0)); do
  case "$1" in
    --artifact-rel)
      ARTIFACT_REL="$2"
      shift 2
      ;;
    --pair)
      PAIRS+=("$2")
      shift 2
      ;;
    --pairs)
      IFS=',' read -r -a PAIRS <<< "$2"
      shift 2
      ;;
    --groups)
      GROUP_COUNT="$2"
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

RUN_ID="phase3-conv2d-eo-group-probe-$(date -u +%Y%m%dT%H%M%SZ)"
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
echo "groups=$GROUP_COUNT"
echo "pi_host=$PI_HOST"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }
rsync_pi() { rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"; }

python3 - "$ARTIFACT_DIR" "$OUT_DIR" "$GROUP_COUNT" "${PAIRS[@]}" <<'PY'
import json, pathlib, sys
artifact_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
group_count = int(sys.argv[3])
pairs = sys.argv[4:]

for pair in pairs:
    prep = json.loads((artifact_dir / pair / 'PREP_SUMMARY.json').read_text())
    src_patch = artifact_dir / pair / 'eo_oracle.patchspec'
    pair_out = out_dir / pair
    pair_out.mkdir(parents=True, exist_ok=True)

    rules = []
    for line in src_patch.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        plen_s, off_s, val_s = s.split()[:3]
        rules.append((int(plen_s), int(off_s), int(val_s, 16)))
    rules.sort(key=lambda x: x[1])
    n = len(rules)
    groups = []
    for idx in range(group_count):
        start = (idx * n) // group_count
        end = ((idx + 1) * n) // group_count
        groups.append({
            'group_index': idx,
            'rule_start': start,
            'rule_end': end,
            'rule_count': end - start,
            'offset_start': rules[start][1] if start < end else None,
            'offset_end': rules[end - 1][1] if start < end else None,
        })

    def write_patch(path: pathlib.Path, subset):
        lines = [f'# auto-generated subset for {pair}']
        for plen, off, val in subset:
            lines.append(f'{plen} {off} 0x{val:02x}')
        path.write_text('\n'.join(lines) + '\n')

    write_patch(pair_out / 'eo_full.patchspec', rules)
    cases = []
    for g in groups:
        start, end = g['rule_start'], g['rule_end']
        only_rules = rules[start:end]
        minus_rules = rules[:start] + rules[end:]
        only_name = f"eo_g{g['group_index']:02d}_only"
        minus_name = f"eo_minus_g{g['group_index']:02d}"
        write_patch(pair_out / f'{only_name}.patchspec', only_rules)
        write_patch(pair_out / f'{minus_name}.patchspec', minus_rules)
        cases.append({'name': only_name, 'mode': 'only', **g})
        cases.append({'name': minus_name, 'mode': 'minus', **g})

    anchor_model = pathlib.Path(prep['anchor_model'])
    target_model = pathlib.Path(prep['target_model'])
    meta = {
        'pair_id': pair,
        'artifact_rel': str(artifact_dir.relative_to(pathlib.Path.cwd())),
        'anchor_model_rel': str(anchor_model.resolve().relative_to(pathlib.Path.cwd())),
        'target_model_rel': str(target_model.resolve().relative_to(pathlib.Path.cwd())),
        'param_equal': prep['param_equal'],
        'eo_rule_count': prep['eo_rule_count'],
        'pc_rule_count': prep['pc_rule_count'],
        'group_count': group_count,
        'groups': groups,
        'cases': cases,
    }
    # recover input/output bytes from model file name context encoded in pair id metadata
    # p32/p64/p128 all use the target spatial dims and channels from PREP_SUMMARY consumer script.
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
        raise RuntimeError(f'unknown pair id without explicit byte metadata: {pair}')
    (pair_out / 'GROUPS.json').write_text(json.dumps(meta, indent=2) + '\n')
    with (pair_out / 'GROUPS.txt').open('w', encoding='utf-8') as f:
        f.write(f"pair={pair} eo_rule_count={prep['eo_rule_count']} group_count={group_count}\n")
        for g in groups:
            f.write(f"g{g['group_index']:02d}: rule_count={g['rule_count']} offset_range={g['offset_start']}..{g['offset_end']}\n")
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
  meta="$OUT_DIR/$pair/GROUPS.json"
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
for case in json.load(open(sys.argv[1]))['cases']:
    print(case['name'])
PY
)
done

python3 - "$OUT_DIR" "$DUT_DIR" <<'PY'
import json, pathlib, re, sys
out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])
root_lines = [f"run_id={out_dir.name}"]
for pair_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
    meta = json.loads((pair_dir / 'GROUPS.json').read_text())
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
    rows = []
    for case in meta['cases']:
        result = cases.get(case['name'], {})
        rows.append({
            **case,
            **result,
            'hash_eq_target': result.get('hash') == target_hash,
        })
    summary = {
        'pair_id': pair_dir.name,
        'meta': meta,
        'target_baseline': cases.get('target_baseline', {}),
        'anchor_baseline': cases.get('anchor_baseline', {}),
        'eo_full': cases.get('eo_full', {}),
        'group_cases': rows,
    }
    (pair_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (pair_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
        f.write(f"pair={pair_dir.name} target_hash={target_hash}\n")
        for fixed in ['target_baseline', 'anchor_baseline', 'eo_full']:
            r = cases.get(fixed, {})
            f.write(f"{fixed}: pass={r.get('pass')} hash={r.get('hash')} error={r.get('error')}\n")
        for row in rows:
            f.write(
                f"{row['name']}: mode={row['mode']} rule_count={row['rule_count']} offset_range={row['offset_start']}..{row['offset_end']} pass={row.get('pass')} hash={row.get('hash')} hash_eq_target={row['hash_eq_target']} error={row.get('error')}\n"
            )
    root_lines.append(
        f"[{pair_dir.name}] target_hash={target_hash} eo_full_hash_eq_target={cases.get('eo_full', {}).get('hash') == target_hash} "
        f"target_only_groups={[r['name'] for r in rows if r['mode']=='only' and r['hash_eq_target']]} "
        f"target_minus_groups={[r['name'] for r in rows if r['mode']=='minus' and r['hash_eq_target']]} "
        f"failing_minus_groups={[r['name'] for r in rows if r['mode']=='minus' and not r.get('pass', False)]}"
    )
(out_dir / 'SUMMARY.txt').write_text('\n'.join(root_lines) + '\n')
print(out_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
