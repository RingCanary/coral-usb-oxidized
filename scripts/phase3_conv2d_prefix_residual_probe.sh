#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/tmp/coral-m5-eo-probe-src}"
RUN_ID="phase3-conv2d-prefix-residual-probe-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR/a" "$OUT_DIR/b"

echo "run_id=$RUN_ID"
echo "pi_host=$PI_HOST"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }
rsync_pi() { rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"; }

for seed_dir_seed in "a 1337" "b 2024"; do
  read -r D S <<<"$seed_dir_seed"
  "$REPO_ROOT/tools/conv_template_pipeline.sh" \
    --out-dir "$OUT_DIR/$D" \
    --height 32 --width 32 --in-channels 64 --out-channels 64 \
    --kernel-size 1 --stride 1 --padding same --init-mode random_uniform --seed "$S" >/dev/null
  MODEL=$(find "$OUT_DIR/$D" -maxdepth 1 -name '*_edgetpu.tflite' | head -n 1)
  cargo run --quiet --bin model_param_stream_dump -- --model "$MODEL" --out "$OUT_DIR/${D}_param.bin" >/dev/null
  if [[ "$D" == "b" ]]; then
    B_MODEL="$MODEL"
  fi
done

python3 - <<'PY' "$OUT_DIR/a_param.bin" "$OUT_DIR/b_param.bin" "$OUT_DIR/hybrid_prefixA_weightsB.bin" "$OUT_DIR/SUMMARY.txt"
from pathlib import Path
import hashlib, sys
A=Path(sys.argv[1]).read_bytes(); B=Path(sys.argv[2]).read_bytes(); hybrid=A[:512]+B[512:]
Path(sys.argv[3]).write_bytes(hybrid)
with open(sys.argv[4],'w', encoding='utf-8') as f:
    f.write(f'len={len(A)}\n')
    f.write(f'prefix_mismatches={sum(1 for x,y in zip(A[:512],B[:512]) if x!=y)}\n')
    f.write(f'weights_mismatches={sum(1 for x,y in zip(A[512:],B[512:]) if x!=y)}\n')
    f.write(f'hybrid_eq_B={hybrid==B}\n')
    f.write(f'hybrid_sha256={hashlib.sha256(hybrid).hexdigest()}\n')
print(sys.argv[4])
PY

REMOTE_OUT="$REMOTE_REPO/traces/analysis/$RUN_ID"
REMOTE_MODEL="$REMOTE_REPO/traces/analysis/$RUN_ID/b/$(basename "$B_MODEL")"
ssh_run "mkdir -p '$REMOTE_OUT/b' '$REMOTE_SRC_DIR'"
rsync_pi "$OUT_DIR/" "$PI_HOST:$REMOTE_OUT/" >/dev/null
ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --example rusb_serialized_exec_replay >/dev/null"

ssh_run "cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$REMOTE_MODEL' --firmware '$FIRMWARE_REMOTE' --input-bytes 65536 --output-bytes 65536 --bootstrap-known-good-order --reset-before-claim --post-reset-sleep-ms 1200" > "$OUT_DIR/target_baseline.log" 2>&1 || true
ssh_run "cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$REMOTE_MODEL' --firmware '$FIRMWARE_REMOTE' --input-bytes 65536 --output-bytes 65536 --bootstrap-known-good-order --reset-before-claim --post-reset-sleep-ms 1200 --param-stream-override-file '$REMOTE_OUT/hybrid_prefixA_weightsB.bin'" > "$OUT_DIR/hybrid_override.log" 2>&1 || true

python3 - <<'PY' "$OUT_DIR/target_baseline.log" "$OUT_DIR/hybrid_override.log" "$OUT_DIR/SUMMARY.txt"
import pathlib, re, sys
with open(sys.argv[3], 'a', encoding='utf-8') as f:
    for name,path in [('target_baseline',sys.argv[1]),('hybrid_override',sys.argv[2])]:
        txt=pathlib.Path(path).read_text(errors='ignore')
        out=re.findall(r'Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)', txt)
        err=re.findall(r'Error: (.+)', txt)
        if out:
            f.write(f'{name}: pass=True bytes={out[-1][0]} hash={out[-1][1]}\n')
        else:
            f.write(f'{name}: pass=False error={err[-1] if err else None}\n')
print(pathlib.Path(sys.argv[3]).read_text())
PY

echo "done: $OUT_DIR"
