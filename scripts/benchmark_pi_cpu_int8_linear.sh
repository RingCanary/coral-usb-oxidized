#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_HOST="${PI_HOST:-rpc@10.76.127.205}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"
WARMUP="${WARMUP:-2}"
RUNS_GEMV="${RUNS_GEMV:-9}"
RUNS_GEMM="${RUNS_GEMM:-5}"
THREADS="${THREADS:-4}"

gemv_cases=(
  "2048 2048"
  "2304 2304"
  "2688 2688"
)

gemm_cases=(
  "2688 2688 2688"
  "5376 2688 2688"
  "8064 2688 2688"
)

RUN_ID="benchmark-pi-cpu-int8-linear-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
mkdir -p "$OUT_DIR"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }

echo "run_id=$RUN_ID"
echo "pi_host=$PI_HOST"
echo "threads=$THREADS"

ssh_run "mkdir -p '$REMOTE_SRC_DIR/tools' '$REMOTE_SRC_DIR/target'"
rsync -av \
  -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" \
  "$REPO_ROOT/tools/pi_cpu_int8_bench.c" "$PI_HOST:$REMOTE_SRC_DIR/tools/" >/dev/null
ssh_run "gcc -O3 -mcpu=cortex-a76+dotprod -fopenmp '$REMOTE_SRC_DIR/tools/pi_cpu_int8_bench.c' -o '$REMOTE_SRC_DIR/target/pi_cpu_int8_bench'"

for spec in "${gemv_cases[@]}"; do
  read -r M K <<<"$spec"
  LOG="$OUT_DIR/gemv_${M}x${K}.log"
  ssh_run "OMP_NUM_THREADS='$THREADS' '$REMOTE_SRC_DIR/target/pi_cpu_int8_bench' gemv '$M' '$K' '$WARMUP' '$RUNS_GEMV'" > "$LOG"
done

for spec in "${gemm_cases[@]}"; do
  read -r M K N <<<"$spec"
  LOG="$OUT_DIR/gemm_${M}x${K}x${N}.log"
  ssh_run "OMP_NUM_THREADS='$THREADS' '$REMOTE_SRC_DIR/target/pi_cpu_int8_bench' gemm '$M' '$K' '$N' '$WARMUP' '$RUNS_GEMM'" > "$LOG"
done

python3 - <<'PY' "$OUT_DIR" "$THREADS"
import json
import pathlib
import re
import sys

out_dir = pathlib.Path(sys.argv[1])
threads = int(sys.argv[2])
rows = []
for log_path in sorted(out_dir.glob("*.log")):
    txt = log_path.read_text()
    match = re.search(
        r"mode=(\w+) m=(\d+) k=(\d+) n=(\d+) warmup=(\d+) runs=(\d+) threads=(\d+) "
        r"min_ms=([0-9.]+) median_ms=([0-9.]+) p95_ms=([0-9.]+) max_ms=([0-9.]+) mean_ms=([0-9.]+) "
        r"gmac_per_s_median=([0-9.]+) gmac_per_s_mean=([0-9.]+) checksum=(-?\d+)",
        txt,
    )
    if not match:
        rows.append({"path": log_path.name, "parse_error": True, "raw": txt})
        continue
    row = {
        "path": log_path.name,
        "mode": match.group(1),
        "m": int(match.group(2)),
        "k": int(match.group(3)),
        "n": int(match.group(4)),
        "warmup": int(match.group(5)),
        "runs": int(match.group(6)),
        "threads": int(match.group(7)),
        "min_ms": float(match.group(8)),
        "median_ms": float(match.group(9)),
        "p95_ms": float(match.group(10)),
        "max_ms": float(match.group(11)),
        "mean_ms": float(match.group(12)),
        "gmac_per_s_median": float(match.group(13)),
        "gmac_per_s_mean": float(match.group(14)),
        "checksum": int(match.group(15)),
    }
    rows.append(row)

summary = {
    "run_id": out_dir.name,
    "threads": threads,
    "cases": rows,
}
summary_json = out_dir / "SUMMARY.json"
summary_json.write_text(json.dumps(summary, indent=2) + "\n")

lines = [f"run_id={out_dir.name}", f"threads={threads}"]
for row in rows:
    if row.get("parse_error"):
        lines.append(f"{row['path']}: parse_error=true")
        continue
    lines.append(
        f"{row['path']}: mode={row['mode']} shape={row['m']}x{row['k']}x{row['n']} "
        f"median_ms={row['median_ms']:.3f} mean_ms={row['mean_ms']:.3f} "
        f"gmac_per_s_median={row['gmac_per_s_median']:.3f} gmac_per_s_mean={row['gmac_per_s_mean']:.3f}"
    )
(out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
print(out_dir / "SUMMARY.txt")
PY

echo "done: $OUT_DIR"
