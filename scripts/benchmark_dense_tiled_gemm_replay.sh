#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_HOST="${PI_HOST:-rpc@10.76.127.205}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
MODEL_PATH="${MODEL_PATH:-templates/dense_2688x2688_quant_edgetpu.tflite}"
TILE_DIM="${TILE_DIM:-2688}"
LOGICAL_RUNS="${LOGICAL_RUNS:-2}"
POST_RESET_SLEEP_MS="${POST_RESET_SLEEP_MS:-1200}"

cases=(
  "g2688x2688x2688 2688 2688 2688"
  "g5376x2688x2688 5376 2688 2688"
  "g8064x2688x2688 8064 2688 2688"
)

RUN_ID="benchmark-dense-tiled-gemm-replay-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "run_id=$RUN_ID"
echo "pi_host=$PI_HOST"
echo "tile_dim=$TILE_DIM"
echo "logical_runs=$LOGICAL_RUNS"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }

ssh_run "mkdir -p '$REMOTE_SRC_DIR'"
rsync -av --delete --exclude '.git' --exclude 'target' --exclude 'traces' \
  -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" \
  "$REPO_ROOT/" "$PI_HOST:$REMOTE_SRC_DIR/" >/dev/null
ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --example rusb_serialized_exec_replay >/dev/null"

for spec in "${cases[@]}"; do
  read -r NAME M K N <<<"$spec"
  TILE_ROWS=$(( (M + TILE_DIM - 1) / TILE_DIM ))
  DEVICE_RUNS_PER_LOGICAL=$(( TILE_ROWS * N ))
  TOTAL_RUNS=$(( DEVICE_RUNS_PER_LOGICAL * LOGICAL_RUNS ))
  LOG="$OUT_DIR/${NAME}.log"
  echo "[$NAME] m=$M k=$K n=$N tile_rows=$TILE_ROWS device_runs_per_logical=$DEVICE_RUNS_PER_LOGICAL total_runs=$TOTAL_RUNS"
  ssh_run "cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$MODEL_PATH' --firmware '$FIRMWARE_REMOTE' --input-bytes '$TILE_DIM' --output-bytes '$TILE_DIM' --bootstrap-known-good-order --reset-before-claim --post-reset-sleep-ms '$POST_RESET_SLEEP_MS' --runs '$TOTAL_RUNS'" > "$LOG" 2>&1 || true
done

python3 - <<'PY' "$OUT_DIR" "$TILE_DIM" "$LOGICAL_RUNS"
import json
import math
import pathlib
import re
import statistics
import sys

out_dir = pathlib.Path(sys.argv[1])
tile_dim = int(sys.argv[2])
logical_runs = int(sys.argv[3])
rows = []
lines = [f"run_id={out_dir.name}", f"tile_dim={tile_dim}", f"logical_runs={logical_runs}"]

for log_path in sorted(out_dir.glob("g*.log")):
    stem = log_path.stem[1:]
    m_str, k_str, n_str = stem.split("x")
    m = int(m_str)
    k = int(k_str)
    n = int(n_str)
    tile_rows = math.ceil(m / tile_dim)
    device_runs_per_logical = tile_rows * n
    txt = log_path.read_text(errors="ignore")
    run_ms = [float(v) for v in re.findall(r"Run timing: run_ms=([0-9.]+)", txt)]
    outputs = re.findall(r"Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)", txt)
    logical_ms = []
    for idx in range(0, len(run_ms), device_runs_per_logical):
        group = run_ms[idx:idx + device_runs_per_logical]
        if len(group) == device_runs_per_logical:
            logical_ms.append(sum(group))
    avg_device_ms = sum(run_ms) / len(run_ms) if run_ms else None
    avg_logical_ms = sum(logical_ms) / len(logical_ms) if logical_ms else None
    median_logical_ms = statistics.median(logical_ms) if logical_ms else None
    row = {
        "case": log_path.stem,
        "m": m,
        "k": k,
        "n": n,
        "tile_rows": tile_rows,
        "device_runs_per_logical": device_runs_per_logical,
        "device_runs_observed": len(run_ms),
        "logical_runs_observed": len(logical_ms),
        "avg_device_run_ms": avg_device_ms,
        "avg_logical_ms": avg_logical_ms,
        "median_logical_ms": median_logical_ms,
        "min_logical_ms": min(logical_ms) if logical_ms else None,
        "max_logical_ms": max(logical_ms) if logical_ms else None,
        "logical_gmac_per_s_mean": (m * k * n) / (avg_logical_ms * 1_000_000.0) if avg_logical_ms else None,
        "logical_gmac_per_s_median": (m * k * n) / (median_logical_ms * 1_000_000.0) if median_logical_ms else None,
        "output_hash_stable": len({h for _, h in outputs}) == 1 if outputs else False,
    }
    rows.append(row)
    lines.append(
        f"{log_path.stem}: m={m} k={k} n={n} tile_rows={tile_rows} "
        f"device_runs_per_logical={device_runs_per_logical} logical_runs={len(logical_ms)} "
        f"avg_device_run_ms={avg_device_ms} avg_logical_ms={avg_logical_ms} "
        f"logical_gmac_per_s_mean={row['logical_gmac_per_s_mean']} "
        f"logical_gmac_per_s_median={row['logical_gmac_per_s_median']} "
        f"hash_stable={row['output_hash_stable']}"
    )

(out_dir / "SUMMARY.json").write_text(
    json.dumps(
        {
            "run_id": out_dir.name,
            "tile_dim": tile_dim,
            "logical_runs_requested": logical_runs,
            "cases": rows,
        },
        indent=2,
    )
    + "\n"
)
(out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
print(out_dir / "SUMMARY.txt")
PY

echo "done: $OUT_DIR"
