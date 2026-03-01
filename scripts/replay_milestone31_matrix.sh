#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_ROOT/docs/milestone_manifest_8976_2352_2026-03-01.json"
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/replay_milestone31_matrix.sh [--manifest <path>]

Environment overrides:
  PI_HOST      (default: rpc@rpilm3.local)
  SSH_KEY      (default: ~/.ssh/id_rsa_glmpitwo)
  REMOTE_REPO  (default: /home/rpc/coral-usb-oxidized)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

# Load manifest values into shell variables.
eval "$({
python - "$MANIFEST" <<'PY'
import json, shlex, sys
m = json.load(open(sys.argv[1]))

def emit(k, v):
    print(f"{k}={shlex.quote(str(v))}")

emit("MODEL_REMOTE", m["artifacts"]["model"]["remote_path"])
emit("FIRMWARE_REMOTE", m["artifacts"]["firmware"]["remote_path"])

rp = m["replay_protocol"]
emit("CHUNK_SIZE", rp["chunk_size"])
emit("PARAM_STREAM_MAX_BYTES", rp["param_stream_max_bytes"])
emit("INPUT_BYTES", rp["input_bytes"])
emit("OUTPUT_BYTES", rp["output_bytes"])

patches = m["artifacts"]["patchspecs"]
emit("PATCH_KEYS", ";".join(patches.keys()))
for key, obj in patches.items():
    up = key.upper()
    emit(f"PATCH_{up}_PATH", obj["path"])
    emit(f"PATCH_{up}_SHA", obj["sha256"])

cases = []
for c in m["m2"]["matrix_cases"]:
    key = c.get("patchspec_key") or ""
    cases.append(f"{c['name']}|{key}")
emit("MATRIX_CASES", ";".join(cases))
PY
} )"

SSH_OPTS=(-o IdentitiesOnly=yes -i "$SSH_KEY")
RSYNC_SSH="ssh -o IdentitiesOnly=yes -i $SSH_KEY"

ssh_run() {
  ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"
}

wait_for_pi() {
  for _ in $(seq 1 220); do
    if ssh_run "echo up" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

reboot_and_wait() {
  ssh_run "sudo reboot" >/dev/null 2>&1 || true
  sleep 8
  wait_for_pi
}

build_replay_cmd() {
  local patch="${1:-}"
  local cmd=(
    sudo cargo run --example rusb_serialized_exec_replay --
    --model "$MODEL_REMOTE"
    --firmware "$FIRMWARE_REMOTE"
    --bootstrap-known-good-order
    --chunk-size "$CHUNK_SIZE"
    --param-stream-max-bytes "$PARAM_STREAM_MAX_BYTES"
    --input-bytes "$INPUT_BYTES"
    --output-bytes "$OUTPUT_BYTES"
  )
  if [[ -n "$patch" ]]; then
    cmd+=(--instruction-patch-spec "$patch")
  fi
  printf '%q ' "${cmd[@]}"
}

# Verify local patchspec checksums and sync to remote.
IFS=';' read -r -a PATCH_KEY_ARR <<< "$PATCH_KEYS"
for key in "${PATCH_KEY_ARR[@]}"; do
  up="$(echo "$key" | tr '[:lower:]' '[:upper:]')"
  path_var="PATCH_${up}_PATH"
  sha_var="PATCH_${up}_SHA"
  rel_path="${!path_var}"
  exp_sha="${!sha_var}"
  abs_path="$REPO_ROOT/$rel_path"
  if [[ ! -f "$abs_path" ]]; then
    echo "missing patchspec: $abs_path" >&2
    exit 1
  fi
  got_sha="$(sha256sum "$abs_path" | awk '{print $1}')"
  if [[ "$got_sha" != "$exp_sha" ]]; then
    echo "sha mismatch for $rel_path: got=$got_sha expected=$exp_sha" >&2
    exit 1
  fi
  remote_dir="$REMOTE_REPO/$(dirname "$rel_path")"
  ssh_run "mkdir -p '$remote_dir'"
  rsync -av -e "$RSYNC_SSH" "$abs_path" "$PI_HOST:$REMOTE_REPO/$rel_path" >/dev/null
  echo "synced $rel_path"
done

RUN_ID="specv3-m2-milestone31-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_RUN_DIR="$REMOTE_REPO/traces/analysis/$RUN_ID"
LOCAL_RUN_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
ssh_run "mkdir -p '$REMOTE_RUN_DIR'"

echo "remote run dir: $REMOTE_RUN_DIR"

run_case() {
  local name="$1"
  local patch="${2:-}"
  local remote_log="$REMOTE_RUN_DIR/${name}.log"

  echo "=== CASE: $name ==="
  reboot_and_wait

  local replay_cmd
  replay_cmd="$(build_replay_cmd "$patch")"
  ssh_run "cd '$REMOTE_REPO' && $replay_cmd 2>&1 | tee '$remote_log'"
}

IFS=';' read -r -a CASE_ENTRIES <<< "$MATRIX_CASES"
for entry in "${CASE_ENTRIES[@]}"; do
  name="${entry%%|*}"
  key="${entry#*|}"
  patch_remote=""
  if [[ -n "$key" ]]; then
    up="$(echo "$key" | tr '[:lower:]' '[:upper:]')"
    path_var="PATCH_${up}_PATH"
    patch_remote="$REMOTE_REPO/${!path_var}"
  fi
  run_case "$name" "$patch_remote"
done

mkdir -p "$LOCAL_RUN_DIR"
rsync -av -e "$RSYNC_SSH" "$PI_HOST:$REMOTE_RUN_DIR/" "$LOCAL_RUN_DIR/" >/dev/null

echo "local run dir: $LOCAL_RUN_DIR"
"$REPO_ROOT/scripts/verify_milestone31_signature.sh" "$LOCAL_RUN_DIR" --manifest "$MANIFEST" --require-count "${#CASE_ENTRIES[@]}"

echo "M2 matrix PASS: $RUN_ID"
