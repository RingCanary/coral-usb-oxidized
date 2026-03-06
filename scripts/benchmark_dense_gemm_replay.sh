#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
RUNS="${RUNS:-10}"
POST_RESET_SLEEP_MS="${POST_RESET_SLEEP_MS:-1200}"

cases=(
  "d2048 templates/dense_2048x2048_quant_edgetpu.tflite 2048 2048"
  "d2304 templates/dense_2304x2304_quant_edgetpu.tflite 2304 2304"
  "d2688 templates/dense_2688x2688_quant_edgetpu.tflite 2688 2688"
)

RUN_ID="benchmark-dense-gemm-replay-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "run_id=$RUN_ID"
echo "pi_host=$PI_HOST"
echo "runs=$RUNS"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }

ssh_run "mkdir -p '$REMOTE_SRC_DIR'"
rsync -av --delete --exclude '.git' --exclude 'target' --exclude 'traces' \
  -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" \
  "$REPO_ROOT/" "$PI_HOST:$REMOTE_SRC_DIR/" >/dev/null
ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --example rusb_serialized_exec_replay >/dev/null"

for spec in "${cases[@]}"; do
  read -r NAME MODEL IN_BYTES OUT_BYTES <<<"$spec"
  LOG="$OUT_DIR/${NAME}.log"
  echo "[$NAME] model=$MODEL"
  ssh_run "cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$MODEL' --firmware '$FIRMWARE_REMOTE' --input-bytes '$IN_BYTES' --output-bytes '$OUT_BYTES' --bootstrap-known-good-order --reset-before-claim --post-reset-sleep-ms '$POST_RESET_SLEEP_MS' --runs '$RUNS'" > "$LOG" 2>&1 || true
done

python3 - <<'PY' "$OUT_DIR" "$RUNS"
import json, math, pathlib, re, statistics, sys
out_dir = pathlib.Path(sys.argv[1])
runs = int(sys.argv[2])
lines = [f"run_id={out_dir.name}", f"runs={runs}"]
rows = []
for log_path in sorted(out_dir.glob('d*.log')):
    name = log_path.stem
    dim = int(name[1:])
    txt = log_path.read_text(errors='ignore')
    run_ms = [float(v) for v in re.findall(r'Run timing: run_ms=([0-9.]+)', txt)]
    summary = re.search(r'Run timing summary: runs=([0-9]+) avg_ms=([0-9.]+) min_ms=([0-9.]+) max_ms=([0-9.]+)', txt)
    outputs = re.findall(r'Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)', txt)
    errors = re.findall(r'Error: (.+)', txt)
    row = {
        'case': name,
        'dim': dim,
        'pass': bool(outputs),
        'output_hashes': [h for _, h in outputs],
        'output_hash_stable': len({h for _, h in outputs}) == 1 if outputs else False,
        'error': errors[-1] if errors else None,
        'run_count_observed': len(run_ms),
        'run_ms_values': run_ms,
    }
    if summary:
        avg_ms = float(summary.group(2))
        min_ms = float(summary.group(3))
        max_ms = float(summary.group(4))
    elif run_ms:
        avg_ms = sum(run_ms) / len(run_ms)
        min_ms = min(run_ms)
        max_ms = max(run_ms)
    else:
        avg_ms = min_ms = max_ms = None
    row['avg_ms'] = avg_ms
    row['min_ms'] = min_ms
    row['max_ms'] = max_ms
    if run_ms:
        row['p50_ms'] = statistics.median(run_ms)
        row['p95_ms'] = sorted(run_ms)[max(0, math.ceil(len(run_ms) * 0.95) - 1)]
    else:
        row['p50_ms'] = None
        row['p95_ms'] = None
    if avg_ms and avg_ms > 0:
        macs = dim * dim
        row['gmac_per_s'] = macs / (avg_ms * 1_000_000.0)
    else:
        row['gmac_per_s'] = None
    rows.append(row)
    lines.append(
        f"[{name}] pass={row['pass']} hash_stable={row['output_hash_stable']} runs={row['run_count_observed']} avg_ms={row['avg_ms']} min_ms={row['min_ms']} p50_ms={row['p50_ms']} p95_ms={row['p95_ms']} max_ms={row['max_ms']} gmac_per_s={row['gmac_per_s']} error={row['error']}"
    )
(out_dir / 'SUMMARY.json').write_text(json.dumps({'run_id': out_dir.name, 'runs': runs, 'cases': rows}, indent=2) + '\n')
(out_dir / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n')
print(out_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
