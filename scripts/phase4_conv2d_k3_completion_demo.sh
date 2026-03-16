#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FAMILY_SPEC_REL="${FAMILY_SPEC_REL:-templates/phase4_conv2d_k3_sameprod_6512/family.json}"
FAMILY_SPEC="$REPO_ROOT/$FAMILY_SPEC_REL"
PI_HOST="${PI_HOST:-rpc@192.168.29.216}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/home/rpc/coral-rusb-replay-src}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
BUILD_PROFILE="${BUILD_PROFILE:-debug}"
POST_RESET_SLEEP_MS="${POST_RESET_SLEEP_MS:-1200}"
RUN_ID="phase4-conv2d-k3-completion-demo-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
LOCAL_CASES_DIR="$OUT_DIR/cases"
LOCAL_DUT_DIR="$OUT_DIR/dut"
mkdir -p "$LOCAL_CASES_DIR" "$LOCAL_DUT_DIR"

pairs=(p32 p64 p128)
requested_target_heights=()
requested_target_widths=()

while (($# > 0)); do
  case "$1" in
    --family-spec)
      FAMILY_SPEC="$2"
      shift 2
      ;;
    --pair)
      pairs=("$2")
      shift 2
      ;;
    --pairs)
      IFS=',' read -r -a pairs <<<"$2"
      shift 2
      ;;
    --target-height)
      requested_target_heights=("$2")
      shift 2
      ;;
    --target-heights)
      IFS=',' read -r -a requested_target_heights <<<"$2"
      shift 2
      ;;
    --target-width)
      requested_target_widths=("$2")
      shift 2
      ;;
    --target-widths)
      IFS=',' read -r -a requested_target_widths <<<"$2"
      shift 2
      ;;
    --pi-host)
      PI_HOST="$2"
      shift 2
      ;;
    --build-profile)
      BUILD_PROFILE="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

[[ -f "$FAMILY_SPEC" ]] || {
  echo "family spec not found: $FAMILY_SPEC" >&2
  exit 1
}

channels_for_pair() {
  case "$1" in
    p32) echo 32 ;;
    p64) echo 64 ;;
    p128) echo 128 ;;
    *)
      echo "unknown pair: $1" >&2
      exit 1
      ;;
  esac
}

resolve_target_shapes() {
  python3 - <<'PY' "$FAMILY_SPEC" "${requested_target_heights[*]:-}" "${requested_target_widths[*]:-}"
import json, pathlib, sys
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
height_arg = sys.argv[2].strip()
width_arg = sys.argv[3].strip()
targets = spec["regimes"][0]["targets"]
heights = [int(x) for x in height_arg.split()] if height_arg else []
widths = [int(x) for x in width_arg.split()] if width_arg else []

def die(msg):
    raise SystemExit(msg)

if not heights and not widths:
    for target in targets:
        print(f"{target['height']} {target['width']}")
    raise SystemExit(0)

if widths and not heights:
    die("target widths require target heights")

pairs = []
if widths:
    if len(heights) == 1:
        pairs = [(heights[0], w) for w in widths]
    elif len(heights) == len(widths):
        pairs = list(zip(heights, widths))
    else:
        die("target heights/widths length mismatch")
else:
    for h in heights:
        matches = [(t["height"], t["width"]) for t in targets if int(t["height"]) == h]
        if len(matches) != 1:
            die(f"height {h} is ambiguous or unsupported; pass target width")
        pairs.extend(matches)

for h, w in pairs:
    if not any(int(t["height"]) == h and int(t["width"]) == w for t in targets):
        die(f"unsupported target shape {h}x{w}")
    print(f"{h} {w}")
PY
}

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY")
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }
rsync_pi() { rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"; }

emit_case_assets() {
  local pair="$1"
  local target_height="$2"
  local target_width="$3"
  local channels="$4"
  local case_dir="$LOCAL_CASES_DIR/$pair/h${target_height}_w${target_width}"
  mkdir -p "$case_dir"
  cargo run --quiet --bin conv_k3_eo_emit -- \
    --family-spec "$FAMILY_SPEC" \
    --channels "$channels" \
    --target-height "$target_height" \
    --target-width "$target_width" \
    --out-patchspec "$case_dir/eo.patchspec" \
    --out-report "$case_dir/eo_report.json" >/dev/null
  python3 - <<'PY' "$case_dir/eo_report.json" "$case_dir/eo_report.txt"
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
lines = [f"{k}={v}" for k, v in report.items()]
pathlib.Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  local target_model target_metadata
  target_model=$(python3 - <<'PY' "$case_dir/eo_report.json"
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(report["target_model"])
PY
)
  target_metadata=$(python3 - <<'PY' "$case_dir/eo_report.json"
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(report["target_metadata"])
PY
)
  cargo run --quiet --bin conv_k_param_materialize -- \
    --model "$target_model" \
    --metadata "$target_metadata" \
    --out "$case_dir/target_param_stream.bin" >/dev/null
}

run_remote_case() {
  local name="$1"
  local model_remote="$2"
  local case_remote_dir="$3"
  local input_bytes="$4"
  local output_bytes="$5"
  local rule_count="$6"
  local log_path="$7"
  local cmd="cd '$REMOTE_SRC_DIR' && sudo target/$BUILD_PROFILE/examples/rusb_serialized_exec_replay --model '$model_remote' --firmware '$FIRMWARE_REMOTE' --input-bytes '$input_bytes' --output-bytes '$output_bytes' --bootstrap-known-good-order --reset-before-claim --post-reset-sleep-ms '$POST_RESET_SLEEP_MS'"
  if [[ "$name" == "native_completion" ]]; then
    cmd+=" --param-stream-override-file '$case_remote_dir/target_param_stream.bin'"
    if (( rule_count > 0 )); then
      cmd+=" --instruction-patch-spec '$case_remote_dir/eo.patchspec'"
    fi
  fi
  ssh_run "$cmd" > "$log_path" 2>&1 || true
}

mapfile -t target_shapes < <(resolve_target_shapes)

for pair in "${pairs[@]}"; do
  channels=$(channels_for_pair "$pair")
  for shape in "${target_shapes[@]}"; do
    read -r target_height target_width <<<"$shape"
    emit_case_assets "$pair" "$target_height" "$target_width" "$channels"
  done
done

ssh_run "mkdir -p '$REMOTE_SRC_DIR/traces/analysis/$RUN_ID'"
rsync -av --delete --exclude '.git' --exclude 'target' --exclude 'traces' \
  -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" \
  "$REPO_ROOT/" "$PI_HOST:$REMOTE_SRC_DIR/" >/dev/null
ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --example rusb_serialized_exec_replay $([[ "$BUILD_PROFILE" == release ]] && echo --release) >/dev/null"

for pair in "${pairs[@]}"; do
  for shape in "${target_shapes[@]}"; do
    read -r target_height target_width <<<"$shape"
    case_dir="$LOCAL_CASES_DIR/$pair/h${target_height}_w${target_width}"
    report_json="$case_dir/eo_report.json"
    read -r report_height report_width rule_count anchor_model target_compiled_model < <(
      python3 - <<'PY' "$report_json"
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(report["target_height"], report["target_width"], report["rule_count"], report["anchor_compiled_model"], report["target_compiled_model"])
PY
    )
    input_bytes=$((report_height * report_width * $(channels_for_pair "$pair")))
    output_bytes=$input_bytes
    case_remote_dir="$REMOTE_SRC_DIR/traces/analysis/$RUN_ID/$pair/h${report_height}_w${report_width}"
    mkdir -p "$LOCAL_DUT_DIR/$pair"
    ssh_run "mkdir -p '$case_remote_dir'"
    rsync_pi "$case_dir/" "$PI_HOST:$case_remote_dir/" >/dev/null
    anchor_rel="${anchor_model#$REPO_ROOT/}"
    target_rel="${target_compiled_model#$REPO_ROOT/}"
    anchor_remote="$REMOTE_SRC_DIR/$anchor_rel"
    target_remote="$REMOTE_SRC_DIR/$target_rel"
    run_remote_case target_baseline "$target_remote" "$case_remote_dir" "$input_bytes" "$output_bytes" "$rule_count" "$LOCAL_DUT_DIR/$pair/h${report_height}_w${report_width}_target_baseline.log"
    run_remote_case native_completion "$anchor_remote" "$case_remote_dir" "$input_bytes" "$output_bytes" "$rule_count" "$LOCAL_DUT_DIR/$pair/h${report_height}_w${report_width}_native_completion.log"
  done
done

python3 - <<'PY' "$OUT_DIR" "$LOCAL_DUT_DIR"
import pathlib
import re
import sys

out_dir = pathlib.Path(sys.argv[1])
dut_dir = pathlib.Path(sys.argv[2])
cases_dir = out_dir / "cases"
summary_lines = [f"run_id={out_dir.name}"]
ok = True

def parse_hash(path: pathlib.Path):
    if not path.exists():
        return None, "missing log"
    txt = path.read_text(errors="ignore")
    m = re.findall(r"Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)", txt)
    if m:
        return m[-1][1], None
    err = re.findall(r"Error: (.+)", txt)
    return None, (err[-1] if err else "no output hash found")

for pair_dir in sorted(cases_dir.iterdir()):
    if not pair_dir.is_dir():
        continue
    summary_lines.append(f"[{pair_dir.name}]")
    for case_dir in sorted(pair_dir.iterdir()):
        report = (case_dir / "eo_report.json")
        if not report.exists():
            continue
        base_log = dut_dir / pair_dir.name / f"{case_dir.name}_target_baseline.log"
        native_log = dut_dir / pair_dir.name / f"{case_dir.name}_native_completion.log"
        base_hash, base_err = parse_hash(base_log)
        native_hash, native_err = parse_hash(native_log)
        hash_eq = base_hash is not None and native_hash == base_hash
        ok = ok and hash_eq
        summary_lines.append(
            f"  {case_dir.name}: target_hash={base_hash} native_hash={native_hash} hash_eq_target={hash_eq} target_err={base_err} native_err={native_err}"
        )

(out_dir / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print(out_dir / "SUMMARY.txt")
print(f"all_hash_eq_target={ok}")
PY

echo "completion demo artifact: $OUT_DIR"
