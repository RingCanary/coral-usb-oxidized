#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  ./tools/re_capture_decode_lm_compare.sh \
    --bus <bus> \
    --model <model.safetensors> \
    --templates-dir <dir> \
    --prompt <csv_token_ids> \
    --lm-template <dense_640x2624_quant_edgetpu.tflite> \
    [--out-dir <dir>] [--max-layers <n>] [--steps <n>]

Captures usbmon for Function-Gemma decode in two modes:
  1) CPU LM-head
  2) Coral LM-head (preload)

Then generates phase reports and run-to-run diff.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

BUS=""
MODEL=""
TEMPLATES_DIR=""
PROMPT=""
LM_TEMPLATE=""
MAX_LAYERS=1
STEPS=1
OUT_DIR=""

while (($# > 0)); do
  case "$1" in
    --bus)
      [[ $# -ge 2 ]] || die "missing value for --bus"
      BUS="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || die "missing value for --model"
      MODEL="$2"
      shift 2
      ;;
    --templates-dir)
      [[ $# -ge 2 ]] || die "missing value for --templates-dir"
      TEMPLATES_DIR="$2"
      shift 2
      ;;
    --prompt)
      [[ $# -ge 2 ]] || die "missing value for --prompt"
      PROMPT="$2"
      shift 2
      ;;
    --lm-template)
      [[ $# -ge 2 ]] || die "missing value for --lm-template"
      LM_TEMPLATE="$2"
      shift 2
      ;;
    --max-layers)
      [[ $# -ge 2 ]] || die "missing value for --max-layers"
      MAX_LAYERS="$2"
      shift 2
      ;;
    --steps)
      [[ $# -ge 2 ]] || die "missing value for --steps"
      STEPS="$2"
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || die "missing value for --out-dir"
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$BUS" ]] || die "--bus is required"
[[ -n "$MODEL" ]] || die "--model is required"
[[ -n "$TEMPLATES_DIR" ]] || die "--templates-dir is required"
[[ -n "$PROMPT" ]] || die "--prompt is required"
[[ -n "$LM_TEMPLATE" ]] || die "--lm-template is required"

need_cmd sudo
need_cmd python3
need_cmd cargo

TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="traces/re-decode-lm-compare-${TS}"
fi
mkdir -p "$OUT_DIR"

run_decode() {
  local mode="$1"
  local round_dir="$2"
  mkdir -p "$round_dir"
  local cmd="eval \"\$(./tools/bootstrap_arch_stack.sh print-env)\"; cargo run --example function_gemma_decode_loop -- ${MODEL} ${TEMPLATES_DIR} ${PROMPT} --max-layers ${MAX_LAYERS} --steps ${STEPS} --weight-quant per-channel --lm-head ${mode}"
  if [[ "$mode" != "cpu" ]]; then
    cmd+=" --lm-template ${LM_TEMPLATE}"
  fi
  sudo ./tools/usbmon_capture.sh -b "$BUS" -o "$round_dir" -- bash -lc "$cmd"
}

CPU_DIR="$OUT_DIR/cpu_lmhead"
CORAL_DIR="$OUT_DIR/coral_lmhead"

run_decode "cpu" "$CPU_DIR"
run_decode "coral-preload" "$CORAL_DIR"

CPU_LOG="$(ls -1 "$CPU_DIR"/usbmon-bus${BUS}-*.log | tail -n 1)"
CORAL_LOG="$(ls -1 "$CORAL_DIR"/usbmon-bus${BUS}-*.log | tail -n 1)"

python3 tools/usbmon_phase_report.py report "$CPU_LOG" --bus "$BUS" --json > "$OUT_DIR/cpu_phase.json"
python3 tools/usbmon_phase_report.py report "$CORAL_LOG" --bus "$BUS" --json > "$OUT_DIR/coral_phase.json"
python3 tools/usbmon_phase_report.py diff "$CPU_LOG" "$CORAL_LOG" --bus "$BUS" --json > "$OUT_DIR/cpu_vs_coral_diff.json"
python3 tools/usbmon_bulk_signature.py "$CPU_LOG" --bus "$BUS" > "$OUT_DIR/cpu_bulk.txt"
python3 tools/usbmon_bulk_signature.py "$CORAL_LOG" --bus "$BUS" > "$OUT_DIR/coral_bulk.txt"

cat > "$OUT_DIR/README.txt" <<TXT
RE decode LM compare capture

bus=$BUS
model=$MODEL
templates_dir=$TEMPLATES_DIR
prompt=$PROMPT
lm_template=$LM_TEMPLATE
max_layers=$MAX_LAYERS
steps=$STEPS

cpu_log=$CPU_LOG
coral_log=$CORAL_LOG

outputs:
- cpu_phase.json
- coral_phase.json
- cpu_vs_coral_diff.json
- cpu_bulk.txt
- coral_bulk.txt
TXT

echo "Capture + diff complete: $OUT_DIR"
