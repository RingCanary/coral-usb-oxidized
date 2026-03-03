#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"

MODEL_REMOTE="${MODEL_REMOTE:-/home/rpc/coral-usb-oxidized/dense_1792x1792_quant_edgetpu.tflite}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"

PATCH_EO_TOXIC4="${PATCH_EO_TOXIC4:-traces/analysis/specv2-holdout-family8976-20260301T2030Z/eo_v2_toxic4.rust.patchspec}"
PATCH_PC_SAFE14="${PATCH_PC_SAFE14:-traces/analysis/specv3-tiered-holdout-family8976-20260301T170035Z/pc_strict.full.patchspec}"
PATCH_PC_RES39="${PATCH_PC_RES39:-traces/analysis/specv2-holdout-family8976-20260301T2030Z/pc_v2_res39.rust.patchspec}"
PATCH_EO_NONTOXIC6="${PATCH_EO_NONTOXIC6:-traces/analysis/specv2-holdout-family8976-20260301T2030Z/eo_v2_nontoxic6.rust.patchspec}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/m3_instruction_coupling_probe.sh

Environment overrides:
  PI_HOST, SSH_KEY, REMOTE_REPO
  MODEL_REMOTE, FIRMWARE_REMOTE
  PATCH_EO_TOXIC4, PATCH_PC_SAFE14, PATCH_PC_RES39, PATCH_EO_NONTOXIC6
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SSH_OPTS=(-o IdentitiesOnly=yes -i "$SSH_KEY")
RSYNC_SSH="ssh -o IdentitiesOnly=yes -i $SSH_KEY"

ssh_run() {
  ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"
}

abs_path() {
  local p="$1"
  if [[ "$p" == /* ]]; then
    echo "$p"
  else
    echo "$REPO_ROOT/$p"
  fi
}

parse_patchspec_to_json() {
  local in_path="$1"
  python3 - "$in_path" <<'PY'
import json, sys
path=sys.argv[1]
rows=[]
for raw in open(path, 'r', encoding='utf-8'):
    clean=raw.split('#',1)[0].strip()
    if not clean:
        continue
    parts=clean.replace(',', ' ').split()
    if len(parts) != 3:
        raise SystemExit(f"invalid patch line in {path}: {raw.rstrip()}")
    plen=int(parts[0],0)
    off=int(parts[1],0)
    val=int(parts[2],0)
    rows.append((plen,off,val))
print(json.dumps(rows))
PY
}

combine_patchspec() {
  local out_path="$1"
  shift
  python3 - "$out_path" "$@" <<'PY'
import json, sys
out=sys.argv[1]
inputs=sys.argv[2:]
merged={}
for path in inputs:
    rows=[]
    for raw in open(path, 'r', encoding='utf-8'):
        clean=raw.split('#',1)[0].strip()
        if not clean:
            continue
        parts=clean.replace(',', ' ').split()
        if len(parts) != 3:
            raise SystemExit(f"invalid patch line in {path}: {raw.rstrip()}")
        plen=int(parts[0],0)
        off=int(parts[1],0)
        val=int(parts[2],0)
        key=(plen,off)
        prev=merged.get(key)
        if prev is not None and prev != val:
            raise SystemExit(f"conflict for payload_len={plen} offset={off}: {prev:#x} vs {val:#x}")
        merged[key]=val

rows=sorted((plen,off,val) for (plen,off),val in merged.items())
with open(out, 'w', encoding='utf-8') as f:
    f.write('# combined patchspec (auto-generated)\n')
    for plen,off,val in rows:
        f.write(f"{plen} {off} 0x{val:02x}\n")
print(len(rows))
PY
}

for rel in "$PATCH_EO_TOXIC4" "$PATCH_PC_SAFE14" "$PATCH_PC_RES39" "$PATCH_EO_NONTOXIC6"; do
  ap="$(abs_path "$rel")"
  if [[ ! -f "$ap" ]]; then
    echo "missing patchspec: $ap" >&2
    exit 1
  fi
done

RUN_ID="specv3-m3-instr-coupling-probe-$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_RUN_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
REMOTE_RUN_DIR="$REMOTE_REPO/traces/analysis/$RUN_ID"
REMOTE_PATCH_DIR="$REMOTE_RUN_DIR/patches"
REMOTE_WT="/tmp/coral-m3-coupling-probe-main"

mkdir -p "$LOCAL_RUN_DIR/patches"

EO_TOXIC4_ABS="$(abs_path "$PATCH_EO_TOXIC4")"
PC_SAFE14_ABS="$(abs_path "$PATCH_PC_SAFE14")"
PC_RES39_ABS="$(abs_path "$PATCH_PC_RES39")"
EO_NONTOXIC6_ABS="$(abs_path "$PATCH_EO_NONTOXIC6")"

cp "$EO_TOXIC4_ABS" "$LOCAL_RUN_DIR/patches/eo_toxic4.patchspec"
cp "$PC_SAFE14_ABS" "$LOCAL_RUN_DIR/patches/pc_safe14.patchspec"
cp "$PC_RES39_ABS" "$LOCAL_RUN_DIR/patches/pc_res39.patchspec"
cp "$EO_NONTOXIC6_ABS" "$LOCAL_RUN_DIR/patches/eo_nontoxic6.patchspec"

combine_patchspec "$LOCAL_RUN_DIR/patches/eo_toxic4_plus_pc_safe14.patchspec" \
  "$LOCAL_RUN_DIR/patches/eo_toxic4.patchspec" \
  "$LOCAL_RUN_DIR/patches/pc_safe14.patchspec" >/dev/null

combine_patchspec "$LOCAL_RUN_DIR/patches/pc_res39_plus_eo_nontoxic6.patchspec" \
  "$LOCAL_RUN_DIR/patches/pc_res39.patchspec" \
  "$LOCAL_RUN_DIR/patches/eo_nontoxic6.patchspec" >/dev/null

ssh_run "mkdir -p '$REMOTE_PATCH_DIR'"
rsync -av -e "$RSYNC_SSH" "$LOCAL_RUN_DIR/patches/" "$PI_HOST:$REMOTE_PATCH_DIR/" >/dev/null

ssh_run "set -e; git -C '$REMOTE_REPO' fetch origin main >/dev/null; git -C '$REMOTE_REPO' worktree remove --force '$REMOTE_WT' >/dev/null 2>&1 || true; rm -rf '$REMOTE_WT'; git -C '$REMOTE_REPO' worktree add '$REMOTE_WT' origin/main >/dev/null; mkdir -p '$REMOTE_RUN_DIR'"

echo "run_id=$RUN_ID"
echo "remote_run_dir=$REMOTE_RUN_DIR"

build_cmd() {
  local patch_remote="${1:-}"
  local cmd=(
    sudo cargo run --example rusb_serialized_exec_replay --
    --model "$MODEL_REMOTE"
    --firmware "$FIRMWARE_REMOTE"
    --bootstrap-known-good-order
    --chunk-size 1048576
    --param-stream-max-bytes 3211264
    --input-bytes 1792
    --output-bytes 1792
    --reset-before-claim
    --post-reset-sleep-ms 1200
  )
  if [[ -n "$patch_remote" ]]; then
    cmd+=(--instruction-patch-spec "$patch_remote")
  fi
  printf '%q ' "${cmd[@]}"
}

run_case() {
  local name="$1"
  local patch_file="${2:-}"
  local remote_patch=""
  if [[ -n "$patch_file" ]]; then
    remote_patch="$REMOTE_PATCH_DIR/$patch_file"
  fi
  local cmd
  cmd="$(build_cmd "$remote_patch")"

  echo "=== $name ==="
  ssh_run "cd '$REMOTE_WT' && $cmd" >"$LOCAL_RUN_DIR/${name}.log" 2>&1 || true
  rsync -av -e "$RSYNC_SSH" "$LOCAL_RUN_DIR/${name}.log" "$PI_HOST:$REMOTE_RUN_DIR/${name}.log" >/dev/null

  if rg -n "Output: bytes=1792" "$LOCAL_RUN_DIR/${name}.log" >/dev/null; then
    local hash
    hash="$(rg -n "Output: bytes=1792" "$LOCAL_RUN_DIR/${name}.log" | tail -n1 | sed -E 's/.*fnv1a64=([^ ]+).*/\1/')"
    echo "$name => PASS hash=$hash"
  elif rg -n "Error:" "$LOCAL_RUN_DIR/${name}.log" >/dev/null; then
    local err
    err="$(rg -n "Error:" "$LOCAL_RUN_DIR/${name}.log" | tail -n1 | sed 's/.*Error: //')"
    echo "$name => FAIL error=$err"
  else
    echo "$name => UNKNOWN"
  fi
}

run_case baseline ""
run_case pc_safe14 "pc_safe14.patchspec"
run_case eo_toxic4 "eo_toxic4.patchspec"
run_case eo_toxic4_plus_pc_safe14 "eo_toxic4_plus_pc_safe14.patchspec"
run_case eo_nontoxic6 "eo_nontoxic6.patchspec"
run_case pc_res39 "pc_res39.patchspec"
run_case pc_res39_plus_eo_nontoxic6 "pc_res39_plus_eo_nontoxic6.patchspec"

python3 - "$LOCAL_RUN_DIR" <<'PY'
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
cases = [
    "baseline",
    "pc_safe14",
    "eo_toxic4",
    "eo_toxic4_plus_pc_safe14",
    "eo_nontoxic6",
    "pc_res39",
    "pc_res39_plus_eo_nontoxic6",
]
out = {"cases": []}
for name in cases:
    text = (root / f"{name}.log").read_text(errors="ignore")
    output = re.findall(r"Output: bytes=1792 fnv1a64=(0x[0-9a-fA-F]+)", text)
    errors = re.findall(r"Error: (.+)", text)
    out["cases"].append(
        {
            "name": name,
            "pass": bool(output),
            "output_hash": output[-1] if output else None,
            "error": errors[-1] if errors else None,
            "has_event_tag4": "Event: tag=4" in text,
        }
    )

# Simple coupling verdicts per requested probes.
lookup = {row["name"]: row for row in out["cases"]}
probe = []
probe.append(
    {
        "name": "eo_toxic4_vs_with_pc_safe14",
        "a": lookup["eo_toxic4"],
        "b": lookup["eo_toxic4_plus_pc_safe14"],
        "coupling_evidence": (lookup["eo_toxic4"]["pass"] != lookup["eo_toxic4_plus_pc_safe14"]["pass"]) or (lookup["eo_toxic4"]["output_hash"] != lookup["eo_toxic4_plus_pc_safe14"]["output_hash"]),
    }
)
probe.append(
    {
        "name": "pc_res39_vs_with_eo_nontoxic6",
        "a": lookup["pc_res39"],
        "b": lookup["pc_res39_plus_eo_nontoxic6"],
        "coupling_evidence": (lookup["pc_res39"]["pass"] != lookup["pc_res39_plus_eo_nontoxic6"]["pass"]) or (lookup["pc_res39"]["output_hash"] != lookup["pc_res39_plus_eo_nontoxic6"]["output_hash"]),
    }
)
out["coupling_probes"] = probe

(root / "SUMMARY.json").write_text(json.dumps(out, indent=2) + "\n")
with (root / "SUMMARY.txt").open("w", encoding="utf-8") as f:
    for row in out["cases"]:
        f.write(f"{row['name']}: pass={row['pass']} hash={row['output_hash']} error={row['error']} event_tag4={row['has_event_tag4']}\n")
    for p in out["coupling_probes"]:
        f.write(f"probe[{p['name']}]: coupling_evidence={p['coupling_evidence']}\n")
print(root / "SUMMARY.json")
PY

rsync -av -e "$RSYNC_SSH" "$PI_HOST:$REMOTE_RUN_DIR/" "$LOCAL_RUN_DIR/" >/dev/null
ssh_run "set -e; sudo rm -rf '$REMOTE_WT'; git -C '$REMOTE_REPO' worktree prune"

echo "local_run_dir=$LOCAL_RUN_DIR"
echo "done"
