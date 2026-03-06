#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARTIFACT_REL="traces/analysis/m5-crossdim-oracle-matrix-20260306T103420Z"
FAMILIES=(f7056 f9872)
GROUP_COUNT=4
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="/tmp/coral-m5-eo-probe-src"

while (($# > 0)); do
  case "$1" in
    --artifact-rel)
      ARTIFACT_REL="$2"
      shift 2
      ;;
    --family)
      FAMILIES+=("$2")
      shift 2
      ;;
    --families)
      IFS=',' read -r -a FAMILIES <<< "$2"
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

RUN_ID="m5-eo-oracle-group-probe-$(date -u +%Y%m%dT%H%M%SZ)"
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
echo "families=${FAMILIES[*]}"
echo "groups=$GROUP_COUNT"
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

python3 - "$ARTIFACT_DIR" "$OUT_DIR" "$GROUP_COUNT" "${FAMILIES[@]}" <<'PY'
import json, pathlib, sys
artifact_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
group_count = int(sys.argv[3])
families = sys.argv[4:]

for family in families:
    prep = json.loads((artifact_dir / family / 'PREP_SUMMARY.json').read_text())
    src_patch = artifact_dir / family / 'eo_oracle.patchspec'
    fam_out = out_dir / family
    fam_out.mkdir(parents=True, exist_ok=True)

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
        lines = [f'# auto-generated subset for {family}']
        for plen, off, val in subset:
            lines.append(f'{plen} {off} 0x{val:02x}')
        path.write_text('\n'.join(lines) + '\n')

    write_patch(fam_out / 'eo_full.patchspec', rules)
    cases = []
    for g in groups:
        start, end = g['rule_start'], g['rule_end']
        only_rules = rules[start:end]
        minus_rules = rules[:start] + rules[end:]
        only_name = f"eo_g{g['group_index']:02d}_only"
        minus_name = f"eo_minus_g{g['group_index']:02d}"
        write_patch(fam_out / f'{only_name}.patchspec', only_rules)
        write_patch(fam_out / f'{minus_name}.patchspec', minus_rules)
        cases.append({'name': only_name, 'mode': 'only', **g})
        cases.append({'name': minus_name, 'mode': 'minus', **g})

    meta = {
        'family_id': family,
        'artifact_rel': str(artifact_dir.relative_to(pathlib.Path.cwd())),
        'anchor_model_rel': str(pathlib.Path(prep['anchor_model']).resolve().relative_to(pathlib.Path.cwd())),
        'target_model_rel': str(pathlib.Path(prep['target_model']).resolve().relative_to(pathlib.Path.cwd())),
        'anchor_dims': prep['anchor_dims'],
        'target_dims': prep['target_dims'],
        'target_param_len': prep['target_param_len'],
        'eo_rule_count': prep['eo_rule_count'],
        'group_count': group_count,
        'groups': groups,
        'cases': cases,
    }
    (fam_out / 'GROUPS.json').write_text(json.dumps(meta, indent=2) + '\n')
    with (fam_out / 'GROUPS.txt').open('w', encoding='utf-8') as f:
        f.write(f"family={family} eo_rule_count={prep['eo_rule_count']} group_count={group_count}\n")
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

for family in "${FAMILIES[@]}"; do
  meta="$OUT_DIR/$family/GROUPS.json"
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
  anchor_model_remote="$REMOTE_REPO/$anchor_rel"
  target_model_remote="$REMOTE_REPO/$target_rel"
  param_remote="$REMOTE_ARTIFACT_DIR/$family/target_param_stream.bin"

  run_case "$family" target_baseline "$target_model_remote" "$target_in" "$target_out" "$param_len"
  run_case "$family" anchor_param_only "$anchor_model_remote" "$target_in" "$target_out" "$param_len" \
    --param-stream-override-file "$param_remote"
  run_case "$family" eo_full "$anchor_model_remote" "$target_in" "$target_out" "$param_len" \
    --instruction-patch-spec "$remote_fam_dir/eo_full.patchspec" \
    --param-stream-override-file "$param_remote"

  while IFS= read -r case_name; do
    run_case "$family" "$case_name" "$anchor_model_remote" "$target_in" "$target_out" "$param_len" \
      --instruction-patch-spec "$remote_fam_dir/${case_name}.patchspec" \
      --param-stream-override-file "$param_remote"
  done < <(python3 - <<'PY' "$meta"
import json,sys
meta=json.load(open(sys.argv[1]))
for case in meta['cases']:
    print(case['name'])
PY
)
done

rsync_pi "$PI_HOST:$REMOTE_DUT_DIR/" "$DUT_DIR/" >/dev/null

python3 - "$OUT_DIR" "$DUT_DIR" <<'PY'
import json, pathlib, re, sys
out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])

for fam_dir in sorted(out_dir.iterdir()):
    if not fam_dir.is_dir():
        continue
    meta = json.loads((fam_dir / 'GROUPS.json').read_text())
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
    rows = []
    for case in meta['cases']:
        result = cases.get(case['name'], {})
        rows.append({
            **case,
            **result,
            'hash_eq_target': result.get('hash') == target_hash,
            'pass': result.get('pass', False),
            'bytes': result.get('bytes'),
            'hash': result.get('hash'),
            'error': result.get('error'),
        })
    summary = {
        'family_id': fam_dir.name,
        'meta': meta,
        'target_baseline': cases.get('target_baseline', {}),
        'anchor_param_only': cases.get('anchor_param_only', {}),
        'eo_full': cases.get('eo_full', {}),
        'group_cases': rows,
    }
    (fam_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (fam_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
        f.write(f"family={fam_dir.name} target_hash={target_hash}\n")
        for fixed in ['target_baseline', 'anchor_param_only', 'eo_full']:
            r = cases.get(fixed, {})
            f.write(f"{fixed}: pass={r.get('pass')} hash={r.get('hash')} error={r.get('error')}\n")
        for row in rows:
            f.write(
                f"{row['name']}: mode={row['mode']} rule_count={row['rule_count']} offset_range={row['offset_start']}..{row['offset_end']} pass={row['pass']} hash={row['hash']} hash_eq_target={row['hash_eq_target']} error={row['error']}\n"
            )

root = {
    'run_id': out_dir.name,
    'families': [json.loads((p / 'SUMMARY.json').read_text()) for p in sorted(out_dir.iterdir()) if p.is_dir() and (p / 'SUMMARY.json').exists()],
}
(out_dir / 'SUMMARY.json').write_text(json.dumps(root, indent=2) + '\n')
with (out_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
    f.write(f"run_id={out_dir.name}\n")
    for fam in root['families']:
        f.write(f"[{fam['family_id']}] target_hash={fam['target_baseline'].get('hash')} anchor_param_only_pass={fam['anchor_param_only'].get('pass')} eo_full_hash_eq_target={fam['eo_full'].get('hash')==fam['target_baseline'].get('hash')}\n")
        good_only = [r['name'] for r in fam['group_cases'] if r['mode']=='only' and r['hash_eq_target']]
        good_minus = [r['name'] for r in fam['group_cases'] if r['mode']=='minus' and r['hash_eq_target']]
        bad_minus = [r['name'] for r in fam['group_cases'] if r['mode']=='minus' and not r['pass']]
        f.write(f"  target_only_groups={good_only}\n")
        f.write(f"  target_minus_groups={good_minus}\n")
        f.write(f"  failing_minus_groups={bad_minus}\n")
print(out_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
