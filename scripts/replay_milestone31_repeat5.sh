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
  scripts/replay_milestone31_repeat5.sh [--manifest <path>] [--runs <n>]

Environment overrides:
  PI_HOST      (default: rpc@rpilm3.local)
  SSH_KEY      (default: ~/.ssh/id_rsa_glmpitwo)
  REMOTE_REPO  (default: /home/rpc/coral-usb-oxidized)
USAGE
}

OVERRIDE_RUNS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --runs)
      OVERRIDE_RUNS="$2"
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

patch_key = m["m2"]["repeat5_patchspec_key"]
patch_obj = m["artifacts"]["patchspecs"][patch_key]
emit("REPEAT_PATCH_KEY", patch_key)
emit("REPEAT_PATCH_PATH", patch_obj["path"])
emit("REPEAT_PATCH_SHA", patch_obj["sha256"])
emit("REPEAT_RUNS", m["m2"]["repeat5_runs"])
PY
} )"

if [[ -n "$OVERRIDE_RUNS" ]]; then
  REPEAT_RUNS="$OVERRIDE_RUNS"
fi

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
  local patch="$1"
  local cmd=(
    sudo cargo run --example rusb_serialized_exec_replay --
    --model "$MODEL_REMOTE"
    --firmware "$FIRMWARE_REMOTE"
    --bootstrap-known-good-order
    --chunk-size "$CHUNK_SIZE"
    --param-stream-max-bytes "$PARAM_STREAM_MAX_BYTES"
    --input-bytes "$INPUT_BYTES"
    --output-bytes "$OUTPUT_BYTES"
    --instruction-patch-spec "$patch"
  )
  printf '%q ' "${cmd[@]}"
}

# Verify patch checksum and sync.
abs_patch="$REPO_ROOT/$REPEAT_PATCH_PATH"
if [[ ! -f "$abs_patch" ]]; then
  echo "missing patchspec: $abs_patch" >&2
  exit 1
fi
got_sha="$(sha256sum "$abs_patch" | awk '{print $1}')"
if [[ "$got_sha" != "$REPEAT_PATCH_SHA" ]]; then
  echo "sha mismatch for $REPEAT_PATCH_PATH: got=$got_sha expected=$REPEAT_PATCH_SHA" >&2
  exit 1
fi
remote_dir="$REMOTE_REPO/$(dirname "$REPEAT_PATCH_PATH")"
ssh_run "mkdir -p '$remote_dir'"
rsync -av -e "$RSYNC_SSH" "$abs_patch" "$PI_HOST:$REMOTE_REPO/$REPEAT_PATCH_PATH" >/dev/null

echo "synced $REPEAT_PATCH_PATH"

RUN_ID="specv3-m2-milestone31-repeat${REPEAT_RUNS}-$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_RUN_DIR="$REMOTE_REPO/traces/analysis/$RUN_ID"
LOCAL_RUN_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
ssh_run "mkdir -p '$REMOTE_RUN_DIR'"

echo "remote run dir: $REMOTE_RUN_DIR"

patch_remote="$REMOTE_REPO/$REPEAT_PATCH_PATH"
replay_cmd="$(build_replay_cmd "$patch_remote")"

for i in $(seq 1 "$REPEAT_RUNS"); do
  case_name="$(printf '%s_run%02d' "$REPEAT_PATCH_KEY" "$i")"
  remote_log="$REMOTE_RUN_DIR/${case_name}.log"
  echo "=== CASE: $case_name ==="
  reboot_and_wait
  ssh_run "cd '$REMOTE_REPO' && $replay_cmd 2>&1 | tee '$remote_log'"
done

mkdir -p "$LOCAL_RUN_DIR"
rsync -av -e "$RSYNC_SSH" "$PI_HOST:$REMOTE_RUN_DIR/" "$LOCAL_RUN_DIR/" >/dev/null

echo "local run dir: $LOCAL_RUN_DIR"
"$REPO_ROOT/scripts/verify_milestone31_signature.sh" "$LOCAL_RUN_DIR" --manifest "$MANIFEST" --require-count "$REPEAT_RUNS"

echo "M2 repeat PASS: $RUN_ID"
