#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_REL="traces/analysis/phase3-conv2d-crossdim-oracle-matrix-20260306T132611Z"
PAIRS=(p32 p64 p128)
SUBSPLIT=4
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"

while (($# > 0)); do
  case "$1" in
    --artifact-rel)
      ARTIFACT_REL="$2"
      shift 2
      ;;
    --pairs)
      IFS=',' read -r -a PAIRS <<< "$2"
      shift 2
      ;;
    --pair)
      PAIRS+=("$2")
      shift 2
      ;;
    --subsplit)
      SUBSPLIT="$2"
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

RUN_ID="phase3-conv2d-eo-targeted-refine-probe-$(date -u +%Y%m%dT%H%M%SZ)"
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
echo "pairs=${PAIRS[*]}"
echo "subsplit=$SUBSPLIT"
echo "pi_host=$PI_HOST"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }
rsync_pi() { rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"; }

python3 - "$ARTIFACT_DIR" "$OUT_DIR" "$SUBSPLIT" "${PAIRS[@]}" <<'PY'
import json
import pathlib
import sys

artifact_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
subsplit = int(sys.argv[3])
pairs = sys.argv[4:]

window_specs = {
    "p32": [
        ("body_semantic_core", 1298, 1526),
        ("tail_semantic_a", 2315, 2344),
        ("tail_semantic_b", 2386, 3236),
    ],
    "p64": [
        ("body_full", 242, 2297),
        ("tail_semantic_a", 2321, 2349),
        ("tail_semantic_b", 2354, 2377),
        ("tail_semantic_c", 2386, 3236),
    ],
    "p128": [
        ("body_full", 242, 2289),
        ("tail_semantic_a", 2289, 2306),
        ("tail_semantic_b", 2311, 2345),
        ("tail_semantic_c", 2386, 3236),
    ],
}

io_bytes = {
    "p32": 32 * 32 * 32,
    "p64": 32 * 32 * 64,
    "p128": 32 * 32 * 128,
}

def write_patch(path: pathlib.Path, rules, pair: str):
    lines = [f"# auto-generated targeted subset for {pair}"]
    for plen, off, val in rules:
        lines.append(f"{plen} {off} 0x{val:02x}")
    path.write_text("\n".join(lines) + "\n")

for pair in pairs:
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
    write_patch(pair_out / "eo_full.patchspec", rules, pair)

    windows = []
    for label, start, end in window_specs[pair]:
        window_rules = [r for r in rules if start <= r[1] <= end]
        bins = []
        count = len(window_rules)
        for idx in range(subsplit):
            lo = (idx * count) // subsplit
            hi = ((idx + 1) * count) // subsplit
            removed = window_rules[lo:hi]
            if removed:
                removed_start = removed[0][1]
                removed_end = removed[-1][1]
            else:
                removed_start = None
                removed_end = None
            kept = [r for r in rules if r not in removed]
            name = f"eo_minus_{label}_g{idx:02d}"
            write_patch(pair_out / f"{name}.patchspec", kept, pair)
            bins.append(
                {
                    "name": name,
                    "removed_start": removed_start,
                    "removed_end": removed_end,
                    "removed_rule_count": len(removed),
                }
            )
        windows.append(
            {
                "label": label,
                "start": start,
                "end": end,
                "rule_count": len(window_rules),
                "bins": bins,
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
        "input_bytes": io_bytes[pair],
        "output_bytes": io_bytes[pair],
        "windows": windows,
    }
    (pair_out / "TARGETED_REFINE_META.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (pair_out / "TARGETED_REFINE_META.txt").open("w", encoding="utf-8") as f:
        f.write(f"pair={pair} eo_rule_count={prep['eo_rule_count']}\n")
        for window in windows:
            f.write(
                f"{window['label']}: {window['start']}..{window['end']} rule_count={window['rule_count']}\n"
            )
            for bin_meta in window["bins"]:
                f.write(
                    f"  {bin_meta['name']}: {bin_meta['removed_start']}..{bin_meta['removed_end']} "
                    f"removed_rule_count={bin_meta['removed_rule_count']}\n"
                )
PY

ssh_run "mkdir -p '$REMOTE_OUT_DIR' '$REMOTE_DUT_DIR' '$REMOTE_SRC_DIR'"
rsync_pi "$ARTIFACT_DIR/" "$PI_HOST:$REMOTE_ARTIFACT_DIR/" >/dev/null
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
  shift 5
  local extra_args=("$@")
  local local_log="$DUT_DIR/$pair/${case_name}.log"
  mkdir -p "$(dirname "$local_log")"
  local cmd="cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$model_remote' --firmware '$FIRMWARE_REMOTE' --bootstrap-known-good-order --input-bytes '$input_bytes' --output-bytes '$output_bytes' --reset-before-claim --post-reset-sleep-ms 1200"
  for arg in "${extra_args[@]}"; do
    cmd+=" '$arg'"
  done
  echo "[$pair] $case_name"
  ssh_run "$cmd" </dev/null > "$local_log" 2>&1 || true
}

for pair in "${PAIRS[@]}"; do
  meta="$OUT_DIR/$pair/TARGETED_REFINE_META.json"
  anchor_rel=$(python3 - <<'PY' "$meta"
import json, sys
print(json.load(open(sys.argv[1]))['anchor_model_rel'])
PY
)
  target_rel=$(python3 - <<'PY' "$meta"
import json, sys
print(json.load(open(sys.argv[1]))['target_model_rel'])
PY
)
  input_bytes=$(python3 - <<'PY' "$meta"
import json, sys
print(json.load(open(sys.argv[1]))['input_bytes'])
PY
)
  output_bytes=$(python3 - <<'PY' "$meta"
import json, sys
print(json.load(open(sys.argv[1]))['output_bytes'])
PY
)
  remote_pair_dir="$REMOTE_OUT_DIR/$pair"
  anchor_model_remote="$REMOTE_REPO/$anchor_rel"
  target_model_remote="$REMOTE_REPO/$target_rel"

  run_case "$pair" target_baseline "$target_model_remote" "$input_bytes" "$output_bytes"
  run_case "$pair" anchor_baseline "$anchor_model_remote" "$input_bytes" "$output_bytes"
  run_case "$pair" eo_full "$anchor_model_remote" "$input_bytes" "$output_bytes" \
    --instruction-patch-spec "$remote_pair_dir/eo_full.patchspec"

  while IFS= read -r case_name; do
    run_case "$pair" "$case_name" "$anchor_model_remote" "$input_bytes" "$output_bytes" \
      --instruction-patch-spec "$remote_pair_dir/${case_name}.patchspec"
  done < <(python3 - <<'PY' "$meta"
import json, sys
meta = json.load(open(sys.argv[1]))
for window in meta['windows']:
    for bin_meta in window['bins']:
        print(bin_meta['name'])
PY
)
done

python3 - "$OUT_DIR" "$DUT_DIR" <<'PY'
import json
import pathlib
import re
import sys

out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])
root_lines = [f"run_id={out_dir.name}"]

for pair_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
    meta = json.loads((pair_dir / "TARGETED_REFINE_META.json").read_text())
    logs_dir = dut_dir / pair_dir.name
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
        "pair_id": pair_dir.name,
        "target_hash": target_hash,
        "target_baseline": cases.get("target_baseline", {}),
        "anchor_baseline": cases.get("anchor_baseline", {}),
        "eo_full": cases.get("eo_full", {}),
        "windows": [],
    }
    with (pair_dir / "SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write(f"pair={pair_dir.name} target_hash={target_hash}\n")
        for fixed in ["target_baseline", "anchor_baseline", "eo_full"]:
            result = cases.get(fixed, {})
            f.write(
                f"{fixed}: pass={result.get('pass')} hash={result.get('hash')} error={result.get('error')}\n"
            )
        for window in meta["windows"]:
            window_summary = {
                "label": window["label"],
                "start": window["start"],
                "end": window["end"],
                "rule_count": window["rule_count"],
                "bins": [],
            }
            f.write(
                f"{window['label']}: {window['start']}..{window['end']} rule_count={window['rule_count']}\n"
            )
            for bin_meta in window["bins"]:
                result = cases.get(bin_meta["name"], {})
                row = {
                    **bin_meta,
                    **result,
                    "hash_eq_target": result.get("hash") == target_hash,
                }
                window_summary["bins"].append(row)
                f.write(
                    f"  {bin_meta['name']}: {bin_meta['removed_start']}..{bin_meta['removed_end']} "
                    f"removed_rule_count={bin_meta['removed_rule_count']} pass={result.get('pass')} "
                    f"hash={result.get('hash')} hash_eq_target={result.get('hash') == target_hash} "
                    f"error={result.get('error')}\n"
                )
            summary["windows"].append(window_summary)
    (pair_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    root_lines.append(f"[{pair_dir.name}] target_hash={target_hash}")

(out_dir / "SUMMARY.txt").write_text("\n".join(root_lines) + "\n")
PY

echo "done: $OUT_DIR"
