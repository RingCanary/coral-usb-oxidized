#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_REL="${ARTIFACT_REL:-traces/analysis/phase3-conv2d-crossdim-oracle-matrix-20260306T132611Z}"
PAIR="${PAIR:-p32}"
WINDOW_START="${WINDOW_START:-1430}"
WINDOW_END="${WINDOW_END:-1526}"
SUBSPLIT="${SUBSPLIT:-4}"
PI_HOST="${PI_HOST:-rpc@10.76.127.205}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"

ARTIFACT_DIR="$REPO_ROOT/$ARTIFACT_REL"
[[ -d "$ARTIFACT_DIR" ]] || { echo "artifact dir not found: $ARTIFACT_DIR" >&2; exit 1; }

RUN_ID="phase3-conv2d-eo-p32-core-refine-probe-$(date -u +%Y%m%dT%H%M%SZ)"
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
echo "pair=$PAIR"
echo "window=$WINDOW_START..$WINDOW_END"
echo "subsplit=$SUBSPLIT"
echo "pi_host=$PI_HOST"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }
rsync_pi() { rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"; }

python3 - "$ARTIFACT_DIR" "$OUT_DIR" "$PAIR" "$WINDOW_START" "$WINDOW_END" "$SUBSPLIT" <<'PY'
import json
import pathlib
import sys

artifact_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
pair = sys.argv[3]
window_start = int(sys.argv[4])
window_end = int(sys.argv[5])
subsplit = int(sys.argv[6])

prep = json.loads((artifact_dir / pair / "PREP_SUMMARY.json").read_text())
pair_out = out_dir / pair
pair_out.mkdir(parents=True, exist_ok=True)

rules = []
for line in (artifact_dir / pair / "eo_oracle.patchspec").read_text().splitlines():
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    plen_s, off_s, val_s = s.split()[:3]
    rules.append((int(plen_s), int(off_s), int(val_s, 16)))
rules.sort(key=lambda item: item[1])

window_rules = [r for r in rules if window_start <= r[1] <= window_end]
if not window_rules:
    raise SystemExit(f"no rules inside window {window_start}..{window_end}")

def write_patch(path: pathlib.Path, selected):
    lines = [f"# auto-generated p32 core subset for {pair}"]
    for plen, off, val in selected:
        lines.append(f"{plen} {off} 0x{val:02x}")
    path.write_text("\n".join(lines) + "\n")

write_patch(pair_out / "eo_full.patchspec", rules)

bins = []
count = len(window_rules)
for idx in range(subsplit):
    lo = (idx * count) // subsplit
    hi = ((idx + 1) * count) // subsplit
    removed = window_rules[lo:hi]
    removed_start = removed[0][1] if removed else None
    removed_end = removed[-1][1] if removed else None
    kept = [r for r in rules if r not in removed]
    name = f"eo_minus_core_g{idx:02d}"
    write_patch(pair_out / f"{name}.patchspec", kept)
    bins.append(
        {
            "name": name,
            "removed_start": removed_start,
            "removed_end": removed_end,
            "removed_rule_count": len(removed),
        }
    )

meta = {
    "pair_id": pair,
    "artifact_rel": str(artifact_dir.relative_to(pathlib.Path.cwd())),
    "anchor_model_rel": str(pathlib.Path(prep["anchor_model"]).resolve().relative_to(pathlib.Path.cwd())),
    "target_model_rel": str(pathlib.Path(prep["target_model"]).resolve().relative_to(pathlib.Path.cwd())),
    "eo_rule_count": prep["eo_rule_count"],
    "pc_rule_count": prep["pc_rule_count"],
    "param_equal": prep["param_equal"],
    "input_bytes": 32 * 32 * 32,
    "output_bytes": 32 * 32 * 32,
    "window_start": window_start,
    "window_end": window_end,
    "window_rule_count": len(window_rules),
    "bins": bins,
}
(pair_out / "CORE_REFINE_META.json").write_text(json.dumps(meta, indent=2) + "\n")
PY

ssh_run "mkdir -p '$REMOTE_OUT_DIR' '$REMOTE_DUT_DIR' '$REMOTE_SRC_DIR'"
rsync_pi "$ARTIFACT_DIR/" "$PI_HOST:$REMOTE_ARTIFACT_DIR/" >/dev/null
rsync_pi "$OUT_DIR/" "$PI_HOST:$REMOTE_OUT_DIR/" >/dev/null
rsync -av --delete --exclude '.git' --exclude 'target' --exclude 'traces' \
  -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" \
  "$REPO_ROOT/" "$PI_HOST:$REMOTE_SRC_DIR/" >/dev/null
ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --example rusb_serialized_exec_replay >/dev/null"

run_case() {
  local case_name="$1"
  local model_remote="$2"
  local input_bytes="$3"
  local output_bytes="$4"
  shift 4
  local extra_args=("$@")
  local local_log="$DUT_DIR/$PAIR/${case_name}.log"
  mkdir -p "$(dirname "$local_log")"
  local cmd="cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$model_remote' --firmware '$FIRMWARE_REMOTE' --bootstrap-known-good-order --input-bytes '$input_bytes' --output-bytes '$output_bytes' --reset-before-claim --post-reset-sleep-ms 1200"
  for arg in "${extra_args[@]}"; do
    cmd+=" '$arg'"
  done
  echo "[$PAIR] $case_name"
  ssh_run "$cmd" </dev/null > "$local_log" 2>&1 || true
}

meta="$OUT_DIR/$PAIR/CORE_REFINE_META.json"
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
remote_pair_dir="$REMOTE_OUT_DIR/$PAIR"
anchor_model_remote="$REMOTE_REPO/$anchor_rel"
target_model_remote="$REMOTE_REPO/$target_rel"

run_case target_baseline "$target_model_remote" "$input_bytes" "$output_bytes"
run_case anchor_baseline "$anchor_model_remote" "$input_bytes" "$output_bytes"
run_case eo_full "$anchor_model_remote" "$input_bytes" "$output_bytes" \
  --instruction-patch-spec "$remote_pair_dir/eo_full.patchspec"

while IFS= read -r case_name; do
  run_case "$case_name" "$anchor_model_remote" "$input_bytes" "$output_bytes" \
    --instruction-patch-spec "$remote_pair_dir/${case_name}.patchspec"
done < <(python3 - <<'PY' "$meta"
import json,sys
for bin_meta in json.load(open(sys.argv[1]))['bins']:
    print(bin_meta['name'])
PY
)

python3 - "$OUT_DIR" "$DUT_DIR" "$PAIR" <<'PY'
import json
import pathlib
import re
import sys

out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])
pair = sys.argv[3]
pair_dir = out_dir / pair
meta = json.loads((pair_dir / "CORE_REFINE_META.json").read_text())
logs_dir = dut_dir / pair
cases = {}
for log_path in sorted(logs_dir.glob("*.log")):
    txt = log_path.read_text(errors="ignore")
    out = re.findall(r"Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)", txt)
    err = re.findall(r"Error: (.+)", txt)
    cases[log_path.stem] = {
        "pass": bool(out),
        "bytes": int(out[-1][0]) if out else None,
        "hash": out[-1][1] if out else None,
        "error": err[-1] if err else None,
    }
target_hash = cases.get("target_baseline", {}).get("hash")
summary = {
    "pair_id": pair,
    "target_hash": target_hash,
    "target_baseline": cases.get("target_baseline", {}),
    "anchor_baseline": cases.get("anchor_baseline", {}),
    "eo_full": cases.get("eo_full", {}),
    "bins": [],
}
with (pair_dir / "SUMMARY.txt").open("w", encoding="utf-8") as f:
    f.write(f"pair={pair} target_hash={target_hash}\n")
    for fixed in ["target_baseline", "anchor_baseline", "eo_full"]:
        result = cases.get(fixed, {})
        f.write(f"{fixed}: pass={result.get('pass')} hash={result.get('hash')} error={result.get('error')}\n")
    f.write(
        f"core_window: {meta['window_start']}..{meta['window_end']} rule_count={meta['window_rule_count']}\n"
    )
    for bin_meta in meta["bins"]:
        result = cases.get(bin_meta["name"], {})
        row = {**bin_meta, **result, "hash_eq_target": result.get("hash") == target_hash}
        summary["bins"].append(row)
        f.write(
            f"  {bin_meta['name']}: {bin_meta['removed_start']}..{bin_meta['removed_end']} "
            f"removed_rule_count={bin_meta['removed_rule_count']} pass={result.get('pass')} "
            f"hash={result.get('hash')} hash_eq_target={result.get('hash') == target_hash} "
            f"error={result.get('error')}\n"
        )
(pair_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
(out_dir / "SUMMARY.txt").write_text(f"run_id={out_dir.name}\n[{pair}] target_hash={target_hash}\n")
PY

echo "done: $OUT_DIR"
