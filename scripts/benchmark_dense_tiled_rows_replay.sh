#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
ROW_CASES="${ROW_CASES:-1 8 32 128}"
REPLAY_REPEATS="${REPLAY_REPEATS:-5}"
CPU_REPEATS="${CPU_REPEATS:-10}"
CPU_WARMUP="${CPU_WARMUP:-2}"
POST_RESET_SLEEP_MS="${POST_RESET_SLEEP_MS:-1200}"
BUILD_PROFILE="${BUILD_PROFILE:-release}"
INPUT_DIM=2688
OUTPUT_DIM=2688

RUN_ID="benchmark-dense-tiled-rows-replay-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR"

echo "run_id=$RUN_ID"
echo "pi_host=$PI_HOST"
echo "row_cases=$ROW_CASES"
echo "replay_repeats=$REPLAY_REPEATS cpu_repeats=$CPU_REPEATS"
echo "build_profile=$BUILD_PROFILE"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY")
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }

rsync -av --delete --exclude '.git' --exclude 'target' --exclude 'traces' \
  -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" \
  "$REPO_ROOT/" "$PI_HOST:$REMOTE_SRC_DIR/" >/dev/null

ssh_run "mkdir -p '$REMOTE_SRC_DIR/bench'"
ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --profile '$BUILD_PROFILE' --bin i8_matrix_pattern --example cpu_gemm_baseline --example rusb_serialized_exec_replay >/dev/null"
ssh_run "cd '$REMOTE_SRC_DIR' && target/$BUILD_PROFILE/i8_matrix_pattern --rows $INPUT_DIM --cols $OUTPUT_DIM --mode shift_plus1_cycle --out bench/weights_shift_plus1_2688.i8 > bench/weights_shift_plus1_2688.log"

for ROWS in $ROW_CASES; do
  echo "[rows=$ROWS] generating inputs"
  ssh_run "cd '$REMOTE_SRC_DIR' && target/$BUILD_PROFILE/i8_matrix_pattern --rows '$ROWS' --cols $INPUT_DIM --mode row_index_mod --row-step 17 --modulus 251 --out bench/inputs_rows${ROWS}_2688.i8 > bench/inputs_rows${ROWS}_2688.log"

  CPU_LOG="$OUT_DIR/cpu_rows${ROWS}.log"
  ssh_run "cd '$REMOTE_SRC_DIR' && target/$BUILD_PROFILE/examples/cpu_gemm_baseline --input-dim $INPUT_DIM --output-dim $OUTPUT_DIM --rows '$ROWS' --inputs-i8-file bench/inputs_rows${ROWS}_2688.i8 --weights-row-major-i8-file bench/weights_shift_plus1_2688.i8 --warmup '$CPU_WARMUP' --repeats '$CPU_REPEATS'" > "$CPU_LOG" 2>&1

  for REP in $(seq 1 "$REPLAY_REPEATS"); do
    REPLAY_LOG="$OUT_DIR/replay_rows${ROWS}_rep${REP}.log"
    echo "[rows=$ROWS] replay repeat $REP"
    ssh_run "cd '$REMOTE_SRC_DIR' && sudo target/$BUILD_PROFILE/examples/rusb_serialized_exec_replay --family-profile templates/dense_2688x2688_family_profile.json --weights-row-major-i8-file bench/weights_shift_plus1_2688.i8 --input-batch-file bench/inputs_rows${ROWS}_2688.i8 --firmware '$FIRMWARE_REMOTE' --runs '$ROWS' --bootstrap-known-good-order --reset-before-claim --post-reset-sleep-ms '$POST_RESET_SLEEP_MS'" > "$REPLAY_LOG" 2>&1 || true
  done
done

python3 - <<'PY' "$OUT_DIR" "$INPUT_DIM" "$OUTPUT_DIM" "$ROW_CASES"
import json
import math
import pathlib
import re
import statistics
import sys

out_dir = pathlib.Path(sys.argv[1])
input_dim = int(sys.argv[2])
output_dim = int(sys.argv[3])
row_cases = [int(v) for v in sys.argv[4].split()]
activation_bytes_per_row = input_dim + output_dim

def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * p) - 1))
    return ordered[idx]

rows_out = []
lines = [f"run_id={out_dir.name}"]
for rows in row_cases:
    cpu_log = out_dir / f"cpu_rows{rows}.log"
    cpu_text = cpu_log.read_text(errors="ignore") if cpu_log.exists() else ""
    cpu_summary = re.search(
        r"Summary: repeats=(\d+) mean_ms=([0-9.]+) p50_ms=([0-9.]+) p95_ms=([0-9.]+) min_ms=([0-9.]+) max_ms=([0-9.]+) effective_gmac_per_s=([0-9.]+)",
        cpu_text,
    )
    cpu = None
    if cpu_summary:
        cpu = {
            "repeats": int(cpu_summary.group(1)),
            "mean_ms": float(cpu_summary.group(2)),
            "p50_ms": float(cpu_summary.group(3)),
            "p95_ms": float(cpu_summary.group(4)),
            "min_ms": float(cpu_summary.group(5)),
            "max_ms": float(cpu_summary.group(6)),
            "effective_gmac_per_s": float(cpu_summary.group(7)),
        }

    replay_batch_ms = []
    replay_rows_per_s = []
    replay_activation_mbps = []
    replay_gmac = []
    replay_run_ms = []
    replay_logs = sorted(out_dir.glob(f"replay_rows{rows}_rep*.log"))
    replay_failures = []
    for log_path in replay_logs:
        text = log_path.read_text(errors="ignore")
        run_ms = [float(v) for v in re.findall(r"Run timing: run_ms=([0-9.]+)", text)]
        errors = re.findall(r"Error: (.+)", text)
        if errors:
            replay_failures.extend(errors)
        if len(run_ms) != rows:
            continue
        replay_run_ms.extend(run_ms)
        batch_ms = sum(run_ms)
        replay_batch_ms.append(batch_ms)
        replay_rows_per_s.append(rows / (batch_ms / 1000.0))
        replay_activation_mbps.append((rows * activation_bytes_per_row) / (batch_ms / 1000.0) / 1_000_000.0)
        replay_gmac.append((rows * input_dim * output_dim) / (batch_ms * 1_000_000.0))

    replay = None
    if replay_batch_ms:
        replay = {
            "repeats": len(replay_batch_ms),
            "batch_ms_mean": statistics.mean(replay_batch_ms),
            "batch_ms_p50": statistics.median(replay_batch_ms),
            "batch_ms_p95": percentile(replay_batch_ms, 0.95),
            "batch_ms_min": min(replay_batch_ms),
            "batch_ms_max": max(replay_batch_ms),
            "rows_per_s_mean": statistics.mean(replay_rows_per_s),
            "activation_mb_s_mean": statistics.mean(replay_activation_mbps),
            "effective_gmac_per_s_mean": statistics.mean(replay_gmac),
            "effective_gmac_per_s_p50": statistics.median(replay_gmac),
            "per_invoke_ms_mean": statistics.mean(replay_run_ms),
            "per_invoke_ms_p50": statistics.median(replay_run_ms),
            "per_invoke_ms_p95": percentile(replay_run_ms, 0.95),
            "per_invoke_ms_min": min(replay_run_ms),
            "per_invoke_ms_max": max(replay_run_ms),
            "per_invoke_gmac_per_s_p50": (input_dim * output_dim) / (statistics.median(replay_run_ms) * 1_000_000.0),
            "per_invoke_activation_mb_s_p50": activation_bytes_per_row / (statistics.median(replay_run_ms) / 1000.0) / 1_000_000.0,
        }

    row_record = {
        "rows": rows,
        "macs": rows * input_dim * output_dim,
        "activation_bytes": rows * activation_bytes_per_row,
        "cpu": cpu,
        "replay": replay,
        "replay_failures": replay_failures,
    }
    if cpu and replay:
        row_record["speedup_vs_cpu_gmac"] = replay["effective_gmac_per_s_mean"] / cpu["effective_gmac_per_s"]
        row_record["speedup_vs_cpu_latency"] = cpu["mean_ms"] / replay["batch_ms_mean"]
    rows_out.append(row_record)

    lines.append(
        "[rows={rows}] cpu_gmac={cpu_gmac} replay_gmac={replay_gmac} replay_batch_ms={replay_ms} replay_invoke_p50_ms={invoke_p50_ms} replay_invoke_gmac_p50={invoke_gmac_p50} rows_per_s={rows_per_s} activation_mb_s={activation_mb_s} speedup_vs_cpu_gmac={speedup}".format(
            rows=rows,
            cpu_gmac=None if not cpu else cpu["effective_gmac_per_s"],
            replay_gmac=None if not replay else replay["effective_gmac_per_s_mean"],
            replay_ms=None if not replay else replay["batch_ms_mean"],
            invoke_p50_ms=None if not replay else replay["per_invoke_ms_p50"],
            invoke_gmac_p50=None if not replay else replay["per_invoke_gmac_per_s_p50"],
            rows_per_s=None if not replay else replay["rows_per_s_mean"],
            activation_mb_s=None if not replay else replay["activation_mb_s_mean"],
            speedup=None if "speedup_vs_cpu_gmac" not in row_record else row_record["speedup_vs_cpu_gmac"],
        )
    )

(out_dir / "SUMMARY.json").write_text(json.dumps({"run_id": out_dir.name, "cases": rows_out}, indent=2) + "\n")
(out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
print(out_dir / "SUMMARY.txt")
PY

echo "done: $OUT_DIR"
