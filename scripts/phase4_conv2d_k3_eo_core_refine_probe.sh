#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
K1_ARTIFACT_REL="${K1_ARTIFACT_REL:-traces/analysis/phase3-conv2d-crossdim-oracle-matrix-20260306T132611Z}"
K3_ARTIFACT_REL="${K3_ARTIFACT_REL:-traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z}"
PAIR="${PAIR:-p32}"
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"

while (($# > 0)); do
  case "$1" in
    --pair)
      PAIR="$2"
      shift 2
      ;;
    --k1-artifact-rel)
      K1_ARTIFACT_REL="$2"
      shift 2
      ;;
    --k3-artifact-rel)
      K3_ARTIFACT_REL="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

K1_ARTIFACT_DIR="$REPO_ROOT/$K1_ARTIFACT_REL"
K3_ARTIFACT_DIR="$REPO_ROOT/$K3_ARTIFACT_REL"
[[ -d "$K1_ARTIFACT_DIR" ]] || { echo "k1 artifact dir not found: $K1_ARTIFACT_DIR" >&2; exit 1; }
[[ -d "$K3_ARTIFACT_DIR" ]] || { echo "k3 artifact dir not found: $K3_ARTIFACT_DIR" >&2; exit 1; }

RUN_ID="phase4-conv2d-k3-eo-core-refine-probe-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_REL="traces/analysis/$RUN_ID"
OUT_DIR="$REPO_ROOT/$OUT_REL"
DUT_REL="traces/analysis/specv3-$RUN_ID-dut"
DUT_DIR="$REPO_ROOT/$DUT_REL"
REMOTE_OUT_DIR="$REMOTE_REPO/$OUT_REL"
REMOTE_DUT_DIR="$REMOTE_REPO/$DUT_REL"
mkdir -p "$OUT_DIR" "$DUT_DIR"

echo "run_id=$RUN_ID"
echo "k1_artifact_rel=$K1_ARTIFACT_REL"
echo "k3_artifact_rel=$K3_ARTIFACT_REL"
echo "pair=$PAIR"
echo "pi_host=$PI_HOST"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }
rsync_pi() { rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"; }

python3 - "$K1_ARTIFACT_DIR" "$K3_ARTIFACT_DIR" "$OUT_DIR" "$PAIR" <<'PY'
import json, pathlib, sys

def read_patch(path: pathlib.Path):
    rules = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        plen_s, off_s, val_s = s.split()[:3]
        rules.append((int(plen_s), int(off_s), int(val_s, 16)))
    rules.sort(key=lambda x: x[1])
    return rules

k1_dir = pathlib.Path(sys.argv[1])
k3_dir = pathlib.Path(sys.argv[2])
out_dir = pathlib.Path(sys.argv[3])
pair = sys.argv[4]
pairs = ['p32', 'p64', 'p128']

k1 = {p: {off: val for _, off, val in read_patch(k1_dir / p / 'eo_oracle.patchspec')} for p in pairs}
k3_rules = {p: read_patch(k3_dir / p / 'eo_oracle.patchspec') for p in pairs}
k3 = {p: {off: val for _, off, val in k3_rules[p]} for p in pairs}

common_k1 = set(k1[pairs[0]])
common_k3 = set(k3[pairs[0]])
for p in pairs[1:]:
    common_k1 &= set(k1[p])
    common_k3 &= set(k3[p])

B = []
C = []
for off in sorted(common_k3):
    vals3 = [k3[p][off] for p in pairs]
    if off in common_k1:
        vals1 = [k1[p][off] for p in pairs]
        if vals1 != vals3:
            B.append(off)
    else:
        C.append(off)

C_mid = [o for o in C if 2315 <= o < 3422]
remaining = set(C_mid)
chains = []
for off in sorted(C_mid):
    if off not in remaining:
        continue
    chain = [off]
    remaining.remove(off)
    nxt = off + 96
    while nxt in remaining:
        chain.append(nxt)
        remaining.remove(nxt)
        nxt += 96
    chains.append(chain)

pair_out = out_dir / pair
pair_out.mkdir(parents=True, exist_ok=True)
pair_rules = k3_rules[pair]
pair_rule_map = {off: (plen, off, val) for plen, off, val in pair_rules}

def write_minus_patch(path: pathlib.Path, remove_offsets):
    remove = set(remove_offsets)
    lines = [f'# auto-generated minus subset for {pair}: {path.stem}']
    for plen, off, val in pair_rules:
        if off in remove:
            continue
        lines.append(f'{plen} {off} 0x{val:02x}')
    path.write_text('\n'.join(lines) + '\n')

write_minus_patch(pair_out / 'eo_full.patchspec', [])
cases = []
for off in B:
    name = f'eo_minus_B_off_{off}'
    write_minus_patch(pair_out / f'{name}.patchspec', [off])
    cases.append({
        'name': name,
        'mode': 'minus',
        'class_name': 'B_single',
        'offsets': [off],
        'rule_count': 1,
        'offset_start': off,
        'offset_end': off,
    })
for idx, chain in enumerate(chains):
    name = f'eo_minus_Cmid_chain_{idx:02d}'
    write_minus_patch(pair_out / f'{name}.patchspec', chain)
    cases.append({
        'name': name,
        'mode': 'minus',
        'class_name': 'C_mid_chain',
        'offsets': chain,
        'rule_count': len(chain),
        'offset_start': chain[0],
        'offset_end': chain[-1],
    })

native_param = k3_dir / pair / 'target_param_stream.native.bin'
compiler_param = k3_dir / pair / 'target_param_stream.bin'
param_source = native_param if native_param.exists() else compiler_param
if pair == 'p32':
    input_bytes = output_bytes = 32 * 32 * 32
elif pair == 'p64':
    input_bytes = output_bytes = 32 * 32 * 64
elif pair == 'p128':
    input_bytes = output_bytes = 32 * 32 * 128
else:
    raise RuntimeError(f'unsupported pair: {pair}')

prep = json.loads((k3_dir / pair / 'PREP_SUMMARY.json').read_text())
meta = {
    'pair_id': pair,
    'anchor_model_rel': str(pathlib.Path(prep['anchor_model']).resolve().relative_to(pathlib.Path.cwd())),
    'target_model_rel': str(pathlib.Path(prep['target_model']).resolve().relative_to(pathlib.Path.cwd())),
    'param_source_rel': str(param_source.resolve().relative_to(pathlib.Path.cwd())),
    'param_source_kind': 'native' if param_source.name.endswith('.native.bin') else 'compiler',
    'param_len': len(param_source.read_bytes()),
    'B_offsets': B,
    'C_mid_chains': chains,
    'cases': cases,
    'input_bytes': input_bytes,
    'output_bytes': output_bytes,
}
(pair_out / 'CORE_META.json').write_text(json.dumps(meta, indent=2) + '\n')
with (pair_out / 'CORE_META.txt').open('w', encoding='utf-8') as f:
    f.write(f"pair={pair} B_count={len(B)} C_mid_chain_count={len(chains)} param_source={meta['param_source_kind']}\n")
    for case in cases:
        f.write(f"{case['name']}: class={case['class_name']} rule_count={case['rule_count']} offsets={case['offsets']}\n")
PY

ssh_run "mkdir -p '$REMOTE_OUT_DIR' '$REMOTE_DUT_DIR' '$REMOTE_SRC_DIR'"
rsync_pi "$K3_ARTIFACT_DIR/" "$PI_HOST:$REMOTE_REPO/$K3_ARTIFACT_REL/" >/dev/null
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
  local param_len="$6"
  shift 6
  local extra_args=("$@")
  local local_log="$DUT_DIR/$pair/${case_name}.log"
  mkdir -p "$(dirname "$local_log")"
  local cmd="cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$model_remote' --firmware '$FIRMWARE_REMOTE' --chunk-size 1048576 --param-stream-max-bytes '$param_len' --bootstrap-known-good-order --input-bytes '$input_bytes' --output-bytes '$output_bytes' --reset-before-claim --post-reset-sleep-ms 1200"
  for arg in "${extra_args[@]}"; do
    cmd+=" '$arg'"
  done
  echo "[$pair] $case_name"
  ssh_run "$cmd" </dev/null > "$local_log" 2>&1 || true
}

meta="$OUT_DIR/$PAIR/CORE_META.json"
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
param_rel=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['param_source_rel'])
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
param_len=$(python3 - <<'PY' "$meta"
import json,sys
print(json.load(open(sys.argv[1]))['param_len'])
PY
)

remote_pair_dir="$REMOTE_OUT_DIR/$PAIR"
anchor_model_remote="$REMOTE_REPO/$anchor_rel"
target_model_remote="$REMOTE_REPO/$target_rel"
param_remote="$REMOTE_REPO/$param_rel"

run_case "$PAIR" target_baseline "$target_model_remote" "$input_bytes" "$output_bytes" "$param_len"
run_case "$PAIR" anchor_param_only "$anchor_model_remote" "$input_bytes" "$output_bytes" "$param_len" \
  --param-stream-override-file "$param_remote"
run_case "$PAIR" eo_full "$anchor_model_remote" "$input_bytes" "$output_bytes" "$param_len" \
  --instruction-patch-spec "$remote_pair_dir/eo_full.patchspec" \
  --param-stream-override-file "$param_remote"

while IFS= read -r case_name; do
  run_case "$PAIR" "$case_name" "$anchor_model_remote" "$input_bytes" "$output_bytes" "$param_len" \
    --instruction-patch-spec "$remote_pair_dir/${case_name}.patchspec" \
    --param-stream-override-file "$param_remote"
done < <(python3 - <<'PY' "$meta"
import json,sys
for case in json.load(open(sys.argv[1]))['cases']:
    print(case['name'])
PY
)

python3 - "$OUT_DIR" "$DUT_DIR" "$PAIR" <<'PY'
import json, pathlib, re, sys
out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])
pair = sys.argv[3]
pair_dir = out_dir / pair
meta = json.loads((pair_dir / 'CORE_META.json').read_text())
cases = {}
for log_path in sorted((dut_dir / pair).glob('*.log')):
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
summary = {
    'pair_id': pair,
    'meta': meta,
    'target_baseline': cases.get('target_baseline', {}),
    'anchor_param_only': cases.get('anchor_param_only', {}),
    'eo_full': cases.get('eo_full', {}),
    'cases': [],
}
with (pair_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
    f.write(f"pair={pair} target_hash={target_hash} param_source={meta['param_source_kind']}\n")
    for fixed in ['target_baseline', 'anchor_param_only', 'eo_full']:
        r = cases.get(fixed, {})
        f.write(f"{fixed}: pass={r.get('pass')} hash={r.get('hash')} error={r.get('error')}\n")
    for case in meta['cases']:
        r = cases.get(case['name'], {})
        row = {
            **case,
            'pass': r.get('pass', False),
            'bytes': r.get('bytes'),
            'hash': r.get('hash'),
            'error': r.get('error'),
            'hash_eq_target': r.get('hash') == target_hash,
        }
        summary['cases'].append(row)
        f.write(
            f"{row['name']}: class={row['class_name']} rule_count={row['rule_count']} "
            f"range={row['offset_start']}..{row['offset_end']} pass={row['pass']} "
            f"hash={row['hash']} hash_eq_target={row['hash_eq_target']} error={row['error']}\n"
        )
(pair_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
print(pair_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
