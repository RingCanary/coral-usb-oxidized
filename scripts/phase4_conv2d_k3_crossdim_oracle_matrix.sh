#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"
RUN_ID="phase4-conv2d-k3-crossdim-oracle-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
DUT_DIR="$REPO_ROOT/traces/analysis/specv3-$RUN_ID-dut"
mkdir -p "$OUT_DIR" "$DUT_DIR"

echo "run_id=$RUN_ID"
echo "pi_host=$PI_HOST"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }
rsync_pi() { rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"; }

pairs=(
  "p32 16 64 32 32 32 32"
  "p64 16 64 64 32 32 64"
  "p128 16 64 128 32 32 128"
)
# format: name anchor_h anchor_w channels target_h target_w out_channels

for spec in "${pairs[@]}"; do
  read -r NAME AH AW CH TH TW OC <<<"$spec"
  PAIR_DIR="$OUT_DIR/$NAME"
  mkdir -p "$PAIR_DIR/anchor" "$PAIR_DIR/target"
  "$REPO_ROOT/tools/conv_template_pipeline.sh" --out-dir "$PAIR_DIR/anchor" --height "$AH" --width "$AW" --in-channels "$CH" --out-channels "$OC" --kernel-size 3 --stride 1 --padding same --init-mode random_uniform >/dev/null
  "$REPO_ROOT/tools/conv_template_pipeline.sh" --out-dir "$PAIR_DIR/target" --height "$TH" --width "$TW" --in-channels "$CH" --out-channels "$OC" --kernel-size 3 --stride 1 --padding same --init-mode random_uniform >/dev/null

  cargo run --quiet --bin instruction_chunk_patchspec -- --base-exec "$PAIR_DIR/anchor/extract/package_000/serialized_executable_000.bin" --target-exec "$PAIR_DIR/target/extract/package_000/serialized_executable_000.bin" --out-patchspec "$PAIR_DIR/eo_oracle.patchspec" >/dev/null
  cargo run --quiet --bin instruction_chunk_patchspec -- --base-exec "$PAIR_DIR/anchor/extract/package_000/serialized_executable_001.bin" --target-exec "$PAIR_DIR/target/extract/package_000/serialized_executable_001.bin" --out-patchspec "$PAIR_DIR/pc_oracle.patchspec" >/dev/null

  ANCHOR_MODEL=$(find "$PAIR_DIR/anchor" -maxdepth 1 -name '*_edgetpu.tflite' | head -n 1)
  TARGET_MODEL=$(find "$PAIR_DIR/target" -maxdepth 1 -name '*_edgetpu.tflite' | head -n 1)
  cargo run --quiet --bin model_param_stream_dump -- --model "$ANCHOR_MODEL" --out "$PAIR_DIR/anchor_param_stream.bin" --metadata-out "$PAIR_DIR/anchor_param_meta.json" >/dev/null
  cargo run --quiet --bin model_param_stream_dump -- --model "$TARGET_MODEL" --out "$PAIR_DIR/target_param_stream.bin" --metadata-out "$PAIR_DIR/target_param_meta.json" >/dev/null

  python3 - <<'PY' "$PAIR_DIR"
import hashlib, json, pathlib, sys
pair_dir = pathlib.Path(sys.argv[1])
anchor = (pair_dir / 'anchor_param_stream.bin').read_bytes()
target = (pair_dir / 'target_param_stream.bin').read_bytes()
eo = [line for line in (pair_dir / 'eo_oracle.patchspec').read_text().splitlines() if line.strip() and not line.startswith('#')]
pc = [line for line in (pair_dir / 'pc_oracle.patchspec').read_text().splitlines() if line.strip() and not line.startswith('#')]
(pair_dir / 'eopc_oracle.patchspec').write_text((pair_dir / 'eo_oracle.patchspec').read_text() + (pair_dir / 'pc_oracle.patchspec').read_text())
summary = {
  'anchor_model': str(next(pair_dir.joinpath('anchor').glob('*_edgetpu.tflite'))),
  'target_model': str(next(pair_dir.joinpath('target').glob('*_edgetpu.tflite'))),
  'param_len': len(anchor),
  'anchor_param_sha256': hashlib.sha256(anchor).hexdigest(),
  'target_param_sha256': hashlib.sha256(target).hexdigest(),
  'param_equal': anchor == target,
  'eo_rule_count': len(eo),
  'pc_rule_count': len(pc),
}
(pair_dir / 'PREP_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
with (pair_dir / 'PREP_SUMMARY.txt').open('w', encoding='utf-8') as f:
  for k,v in summary.items():
    f.write(f'{k}={v}\n')
PY
done

ssh_run "mkdir -p '$REMOTE_SRC_DIR' '$REMOTE_REPO/traces/analysis/$RUN_ID' '$REMOTE_REPO/traces/analysis/specv3-$RUN_ID-dut'"
rsync -av --delete --exclude '.git' --exclude 'target' --exclude 'traces' -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$REPO_ROOT/" "$PI_HOST:$REMOTE_SRC_DIR/" >/dev/null
ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --example rusb_serialized_exec_replay >/dev/null"

run_case() {
  local pair="$1"; local name="$2"; local model_remote="$3"; local inb="$4"; local outb="$5"; shift 5
  local extra=("$@")
  local log="$DUT_DIR/$pair/${name}.log"
  mkdir -p "$(dirname "$log")"
  local cmd="cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$model_remote' --firmware '$FIRMWARE_REMOTE' --input-bytes '$inb' --output-bytes '$outb' --bootstrap-known-good-order --reset-before-claim --post-reset-sleep-ms 1200"
  for arg in "${extra[@]}"; do cmd+=" '$arg'"; done
  ssh_run "$cmd" > "$log" 2>&1 || true
}

for spec in "${pairs[@]}"; do
  read -r NAME AH AW CH TH TW OC <<<"$spec"
  PAIR_DIR="$OUT_DIR/$NAME"
  rsync_pi "$PAIR_DIR/" "$PI_HOST:$REMOTE_REPO/traces/analysis/$RUN_ID/$NAME/" >/dev/null
  TARGET_MODEL_REMOTE=$(ssh "${SSH_OPTS[@]}" "$PI_HOST" "find '$REMOTE_REPO/traces/analysis/$RUN_ID/$NAME/target' -maxdepth 1 -name '*_edgetpu.tflite' | head -n 1")
  ANCHOR_MODEL_REMOTE=$(ssh "${SSH_OPTS[@]}" "$PI_HOST" "find '$REMOTE_REPO/traces/analysis/$RUN_ID/$NAME/anchor' -maxdepth 1 -name '*_edgetpu.tflite' | head -n 1")
  IN_BYTES=$((TH * TW * CH))
  OUT_BYTES=$((TH * TW * OC))
  REMOTE_PAIR_DIR="$REMOTE_REPO/traces/analysis/$RUN_ID/$NAME"
  run_case "$NAME" target_baseline "$TARGET_MODEL_REMOTE" "$IN_BYTES" "$OUT_BYTES"
  run_case "$NAME" anchor_baseline "$ANCHOR_MODEL_REMOTE" "$IN_BYTES" "$OUT_BYTES"
  run_case "$NAME" anchor_param_only "$ANCHOR_MODEL_REMOTE" "$IN_BYTES" "$OUT_BYTES" --param-stream-override-file "$REMOTE_PAIR_DIR/target_param_stream.bin"
  run_case "$NAME" anchor_pc_oracle "$ANCHOR_MODEL_REMOTE" "$IN_BYTES" "$OUT_BYTES" --instruction-patch-spec "$REMOTE_PAIR_DIR/pc_oracle.patchspec"
  run_case "$NAME" anchor_eo_oracle "$ANCHOR_MODEL_REMOTE" "$IN_BYTES" "$OUT_BYTES" --instruction-patch-spec "$REMOTE_PAIR_DIR/eo_oracle.patchspec"
  run_case "$NAME" anchor_eopc_oracle "$ANCHOR_MODEL_REMOTE" "$IN_BYTES" "$OUT_BYTES" --instruction-patch-spec "$REMOTE_PAIR_DIR/eopc_oracle.patchspec"
  run_case "$NAME" anchor_param_eo_oracle "$ANCHOR_MODEL_REMOTE" "$IN_BYTES" "$OUT_BYTES" --instruction-patch-spec "$REMOTE_PAIR_DIR/eo_oracle.patchspec" --param-stream-override-file "$REMOTE_PAIR_DIR/target_param_stream.bin"
done

python3 - <<'PY' "$OUT_DIR" "$DUT_DIR"
import json, pathlib, re, sys
out_dir = pathlib.Path(sys.argv[1]); dut_dir = pathlib.Path(sys.argv[2])
lines = [f"run_id={out_dir.name}"]
for pair_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
    prep = json.loads((pair_dir / 'PREP_SUMMARY.json').read_text())
    lines.append(f"[{pair_dir.name}] param_equal={prep['param_equal']} eo_rule_count={prep['eo_rule_count']} pc_rule_count={prep['pc_rule_count']}")
    target_hash = None
    for name in ['target_baseline','anchor_baseline','anchor_param_only','anchor_pc_oracle','anchor_eo_oracle','anchor_eopc_oracle','anchor_param_eo_oracle']:
        log_path = dut_dir / pair_dir.name / f'{name}.log'
        txt = log_path.read_text(errors='ignore') if log_path.exists() else ''
        out = re.findall(r'Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)', txt)
        err = re.findall(r'Error: (.+)', txt)
        if out:
            h = out[-1][1]
            if name == 'target_baseline':
                target_hash = h
            lines.append(f"  {name}: pass=True bytes={out[-1][0]} hash={h} hash_eq_target={h == target_hash if target_hash else None}")
        else:
            lines.append(f"  {name}: pass=False error={err[-1] if err else None}")
(out_dir / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n')
print(out_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
