#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
CROSSDIM_INIT_MODE="${CROSSDIM_INIT_MODE:-random_uniform}"
CROSSDIM_SEED="${CROSSDIM_SEED:-1337}"

RUN_ID="m5-crossdim-oracle-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_REL="traces/analysis/$RUN_ID"
OUT_DIR="$REPO_ROOT/$OUT_REL"
BUILD_DIR="$OUT_DIR/build"
DUT_RUN_ID="specv3-$RUN_ID-dut"
DUT_REL="traces/analysis/$DUT_RUN_ID"
DUT_DIR="$REPO_ROOT/$DUT_REL"
REMOTE_OUT_DIR="$REMOTE_REPO/$OUT_REL"
REMOTE_DUT_DIR="$REMOTE_REPO/$DUT_REL"
REMOTE_SRC_DIR="/tmp/coral-m5-crossdim-src"
mkdir -p "$OUT_DIR" "$BUILD_DIR" "$DUT_DIR"

echo "run_id=$RUN_ID"
echo "out_dir=$OUT_DIR"
echo "pi_host=$PI_HOST"
echo "init_mode=$CROSSDIM_INIT_MODE seed=$CROSSDIM_SEED"

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o IdentitiesOnly=yes
  -i "$SSH_KEY"
)

ssh_run() {
  ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"
}

rsync_to_pi() {
  rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"
}

compile_model() {
  local input_dim="$1"
  local output_dim="$2"
  local dir="$BUILD_DIR/i${input_dim}_o${output_dim}"
  if [[ -f "$dir/exec_parse.json" ]]; then
    echo "reuse compile i${input_dim}_o${output_dim}"
    return
  fi
  echo "compile i${input_dim}_o${output_dim}"
  mkdir -p "$dir"
  "$REPO_ROOT/tools/dense_template_pipeline.sh" \
    --out-dir "$dir" \
    --input-dim "$input_dim" \
    --output-dim "$output_dim" \
    --init-mode "$CROSSDIM_INIT_MODE" \
    --seed "$CROSSDIM_SEED" \
    --rep-samples 64 \
    --rep-range 1.0 \
    > "$dir.pipeline.log" 2>&1
}

patchspec_rule_count() {
  local path="$1"
  python3 - "$path" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
count = 0
for line in path.read_text().splitlines():
    s = line.strip()
    if s and not s.startswith('#'):
        count += 1
print(count)
PY
}

merge_patchspecs() {
  local out="$1"
  shift
  python3 - "$out" "$@" <<'PY'
import pathlib, sys
out = pathlib.Path(sys.argv[1])
lines = ["# merged by scripts/m5_crossdim_oracle_matrix.sh"]
for raw in sys.argv[2:]:
    for line in pathlib.Path(raw).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            lines.append(s)
out.write_text("\n".join(lines) + "\n")
print(out)
PY
}

prepare_family() {
  local family_id="$1"
  local anchor_in="$2"
  local anchor_out="$3"
  local target_in="$4"
  local target_out="$5"

  compile_model "$anchor_in" "$anchor_out"
  compile_model "$target_in" "$target_out"

  local fam_dir="$OUT_DIR/$family_id"
  mkdir -p "$fam_dir"

  local anchor_model="$BUILD_DIR/i${anchor_in}_o${anchor_out}/dense_${anchor_in}x${anchor_out}_quant_edgetpu.tflite"
  local target_model="$BUILD_DIR/i${target_in}_o${target_out}/dense_${target_in}x${target_out}_quant_edgetpu.tflite"
  local anchor_eo="$BUILD_DIR/i${anchor_in}_o${anchor_out}/extract/package_000/serialized_executable_000.bin"
  local anchor_pc="$BUILD_DIR/i${anchor_in}_o${anchor_out}/extract/package_000/serialized_executable_001.bin"
  local target_eo="$BUILD_DIR/i${target_in}_o${target_out}/extract/package_000/serialized_executable_000.bin"
  local target_pc="$BUILD_DIR/i${target_in}_o${target_out}/extract/package_000/serialized_executable_001.bin"

  cargo run --quiet --bin instruction_chunk_patchspec -- \
    --base-exec "$anchor_eo" \
    --target-exec "$target_eo" \
    --chunk-index 0 \
    --out-patchspec "$fam_dir/eo_oracle.patchspec" \
    > "$fam_dir/eo_oracle.stdout.log"

  cargo run --quiet --bin instruction_chunk_patchspec -- \
    --base-exec "$anchor_pc" \
    --target-exec "$target_pc" \
    --chunk-index 0 \
    --out-patchspec "$fam_dir/pc_oracle.patchspec" \
    > "$fam_dir/pc_oracle.stdout.log"

  merge_patchspecs "$fam_dir/eo_pc_oracle.patchspec" \
    "$fam_dir/eo_oracle.patchspec" \
    "$fam_dir/pc_oracle.patchspec" \
    > "$fam_dir/merge.stdout.log"

  cargo run --quiet --bin model_param_stream_dump -- \
    --model "$target_model" \
    --out "$fam_dir/target_param_stream.bin" \
    --metadata-out "$fam_dir/target_param_stream.json" \
    > "$fam_dir/target_param_stream.stdout.log"

  python3 - "$fam_dir" "$family_id" "$anchor_model" "$target_model" "$anchor_in" "$anchor_out" "$target_in" "$target_out" <<'PY'
import json, pathlib, sys
fam_dir = pathlib.Path(sys.argv[1])
family_id = sys.argv[2]
anchor_model = sys.argv[3]
target_model = sys.argv[4]
anchor_in = int(sys.argv[5])
anchor_out = int(sys.argv[6])
target_in = int(sys.argv[7])
target_out = int(sys.argv[8])

def rule_count(path):
    c = 0
    for line in pathlib.Path(path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            c += 1
    return c

param_meta = json.loads((fam_dir / 'target_param_stream.json').read_text())
summary = {
    'family_id': family_id,
    'anchor_model': anchor_model,
    'target_model': target_model,
    'anchor_dims': [anchor_in, anchor_out],
    'target_dims': [target_in, target_out],
    'eo_rule_count': rule_count(fam_dir / 'eo_oracle.patchspec'),
    'pc_rule_count': rule_count(fam_dir / 'pc_oracle.patchspec'),
    'merged_rule_count': rule_count(fam_dir / 'eo_pc_oracle.patchspec'),
    'target_param_len': param_meta['param_len'],
    'target_param_fnv1a64_hex': param_meta['param_fnv1a64_hex'],
}
(fam_dir / 'PREP_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
with (fam_dir / 'PREP_SUMMARY.txt').open('w', encoding='utf-8') as f:
    for k, v in summary.items():
        f.write(f'{k}={v}\n')
print(fam_dir / 'PREP_SUMMARY.txt')
PY
}

run_family_case() {
  local family_id="$1"
  local case_name="$2"
  local model_remote="$3"
  local input_bytes="$4"
  local output_bytes="$5"
  local param_max_bytes="$6"
  shift 6
  local extra_args=("$@")

  local common=(
    --model "$model_remote"
    --firmware "$FIRMWARE_REMOTE"
    --chunk-size 1048576
    --param-stream-max-bytes "$param_max_bytes"
    --bootstrap-known-good-order
    --input-bytes "$input_bytes"
    --output-bytes "$output_bytes"
    --reset-before-claim
    --post-reset-sleep-ms 1200
  )
  local remote_log="$REMOTE_DUT_DIR/$family_id/${case_name}.log"
  local local_log="$DUT_DIR/$family_id/${case_name}.log"
  mkdir -p "$(dirname "$local_log")"
  echo "[$family_id] $case_name"
  ssh_run "cd '$REMOTE_REPO' && sudo cargo run --example rusb_serialized_exec_replay -- ${common[*]} ${extra_args[*]}" > "$local_log" 2>&1 || true
  cat "$local_log" | ssh_run "mkdir -p '$REMOTE_DUT_DIR/$family_id' && cat > '$remote_log'"
}

prepare_family f7056 640 1280 1280 640
prepare_family f7952 768 1536 1536 768
prepare_family f8976 896 1792 1792 896
prepare_family f9872 1024 2048 2048 1024

ssh_run "mkdir -p '$REMOTE_OUT_DIR' '$REMOTE_DUT_DIR' '$REMOTE_SRC_DIR'"
rsync_to_pi "$OUT_DIR/" "$PI_HOST:$REMOTE_OUT_DIR/" >/dev/null
rsync -av --delete \
  --exclude '.git' \
  --exclude 'target' \
  --exclude 'traces' \
  -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" \
  "$REPO_ROOT/" "$PI_HOST:$REMOTE_SRC_DIR/" >/dev/null

ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --example rusb_serialized_exec_replay >/dev/null"

run_remote_replay() {
  local family_id="$1"
  local case_name="$2"
  local model_remote="$3"
  local input_bytes="$4"
  local output_bytes="$5"
  local param_max_bytes="$6"
  shift 6
  local extra_args=("$@")
  local local_log="$DUT_DIR/$family_id/${case_name}.log"
  local remote_log="$REMOTE_DUT_DIR/$family_id/${case_name}.log"
  mkdir -p "$(dirname "$local_log")"
  echo "[$family_id] $case_name"
  local remote_cmd="cd '$REMOTE_SRC_DIR' && sudo cargo run --example rusb_serialized_exec_replay -- --model '$model_remote' --firmware '$FIRMWARE_REMOTE' --chunk-size 1048576 --param-stream-max-bytes '$param_max_bytes' --bootstrap-known-good-order --input-bytes '$input_bytes' --output-bytes '$output_bytes' --reset-before-claim --post-reset-sleep-ms 1200"
  for arg in "${extra_args[@]}"; do
    remote_cmd+=" '$arg'"
  done
  ssh_run "$remote_cmd" > "$local_log" 2>&1 || true
  cat "$local_log" | ssh_run "mkdir -p '$REMOTE_DUT_DIR/$family_id' && cat > '$remote_log'"
}

run_family_matrix() {
  local family_id="$1"
  local anchor_in="$2"
  local anchor_out="$3"
  local target_in="$4"
  local target_out="$5"
  local param_max="$6"

  local remote_fam_dir="$REMOTE_OUT_DIR/$family_id"
  local anchor_model_remote="$REMOTE_OUT_DIR/build/i${anchor_in}_o${anchor_out}/dense_${anchor_in}x${anchor_out}_quant_edgetpu.tflite"
  local target_model_remote="$REMOTE_OUT_DIR/build/i${target_in}_o${target_out}/dense_${target_in}x${target_out}_quant_edgetpu.tflite"

  run_remote_replay "$family_id" target_baseline "$target_model_remote" "$target_in" "$target_out" "$param_max"
  run_remote_replay "$family_id" target_override "$target_model_remote" "$target_in" "$target_out" "$param_max" \
    --param-stream-override-file "$remote_fam_dir/target_param_stream.bin"
  run_remote_replay "$family_id" anchor_param_only "$anchor_model_remote" "$target_in" "$target_out" "$param_max" \
    --param-stream-override-file "$remote_fam_dir/target_param_stream.bin"
  run_remote_replay "$family_id" anchor_pc_oracle "$anchor_model_remote" "$target_in" "$target_out" "$param_max" \
    --instruction-patch-spec "$remote_fam_dir/pc_oracle.patchspec" \
    --param-stream-override-file "$remote_fam_dir/target_param_stream.bin"
  run_remote_replay "$family_id" anchor_eo_oracle "$anchor_model_remote" "$target_in" "$target_out" "$param_max" \
    --instruction-patch-spec "$remote_fam_dir/eo_oracle.patchspec" \
    --param-stream-override-file "$remote_fam_dir/target_param_stream.bin"
  run_remote_replay "$family_id" anchor_eopc_oracle "$anchor_model_remote" "$target_in" "$target_out" "$param_max" \
    --instruction-patch-spec "$remote_fam_dir/eo_pc_oracle.patchspec" \
    --param-stream-override-file "$remote_fam_dir/target_param_stream.bin"
}

run_family_matrix f7056 640 1280 1280 640 819200
run_family_matrix f7952 768 1536 1536 768 1179648
run_family_matrix f8976 896 1792 1792 896 1605632
run_family_matrix f9872 1024 2048 2048 1024 2097152

rsync_to_pi "$PI_HOST:$REMOTE_DUT_DIR/" "$DUT_DIR/" >/dev/null

python3 - "$OUT_DIR" "$DUT_DIR" <<'PY'
import json, pathlib, re, sys
out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])

families = []
for fam_dir in sorted(dut_dir.iterdir()):
    if not fam_dir.is_dir():
        continue
    family_id = fam_dir.name
    prep = json.loads((out_dir / family_id / 'PREP_SUMMARY.json').read_text())
    cases = {}
    for log_path in sorted(fam_dir.glob('*.log')):
        text = log_path.read_text(errors='ignore')
        outs = re.findall(r'Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)', text)
        errs = re.findall(r'Error: (.+)', text)
        cases[log_path.stem] = {
            'pass': bool(outs),
            'bytes': int(outs[-1][0]) if outs else None,
            'hash': outs[-1][1] if outs else None,
            'error': errs[-1] if errs else None,
        }
    target_hash = cases.get('target_baseline', {}).get('hash')
    conclusions = {
        'target_override_hash_eq_target': cases.get('target_override', {}).get('hash') == target_hash,
        'anchor_param_only_pass': cases.get('anchor_param_only', {}).get('pass', False),
        'anchor_pc_oracle_pass': cases.get('anchor_pc_oracle', {}).get('pass', False),
        'anchor_eo_oracle_pass': cases.get('anchor_eo_oracle', {}).get('pass', False),
        'anchor_eopc_oracle_hash_eq_target': (
            cases.get('anchor_eopc_oracle', {}).get('pass', False)
            and cases.get('anchor_eopc_oracle', {}).get('hash') == target_hash
        ),
    }
    families.append({
        'family_id': family_id,
        'prep': prep,
        'cases': cases,
        'conclusions': conclusions,
    })

summary = {
    'run_id': out_dir.name,
    'dut_run_id': dut_dir.name,
    'family_count': len(families),
    'families': families,
}
(out_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
with (out_dir / 'SUMMARY.txt').open('w', encoding='utf-8') as f:
    f.write(f"run_id={out_dir.name}\n")
    f.write(f"dut_run_id={dut_dir.name}\n")
    for fam in families:
        f.write(
            f"{fam['family_id']}: eo_rules={fam['prep']['eo_rule_count']} pc_rules={fam['prep']['pc_rule_count']} merged={fam['prep']['merged_rule_count']} param_len={fam['prep']['target_param_len']}\n"
        )
        for case_name in [
            'target_baseline',
            'target_override',
            'anchor_param_only',
            'anchor_pc_oracle',
            'anchor_eo_oracle',
            'anchor_eopc_oracle',
        ]:
            case = fam['cases'].get(case_name, {})
            f.write(
                f"  {case_name}: pass={case.get('pass')} bytes={case.get('bytes')} hash={case.get('hash')} error={case.get('error')}\n"
            )
        f.write(f"  conclusions={fam['conclusions']}\n")
print(out_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
echo "dut_logs: $DUT_DIR"
