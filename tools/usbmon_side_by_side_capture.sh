#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./tools/usbmon_side_by_side_capture.sh --bus <bus> [options]

Run side-by-side usbmon captures in one timestamped root:
  1) pure-rusb deterministic descriptor-tag sweep (`rusb_serialized_exec_replay`)
  2) known-good libedgetpu delegate+invoke (`inference_benchmark`)

Options:
  --bus <bus>                           USB bus number (required)
  --out-dir <dir>                       Output root (default: traces/usbmon-side-by-side-<ts>-bus<bus>)

  --rusb-model <path>                   Model for pure-rusb sweep
                                        (default: templates/dense_2048x2048_quant_edgetpu.tflite)
  --rusb-tags <csv>                     Descriptor tag sweep list for --parameters-tag
                                        (default: 2,0,1,3,4)
  --rusb-runs <n>                       --runs for replay example (default: 1)
  --rusb-input-bytes <n>                --input-bytes for replay (default: 2048)
  --rusb-output-bytes <n>               --output-bytes for replay (default: 2048)
  --rusb-timeout-ms <n>                 --timeout-ms for replay (default: 6000)
  --rusb-param-stream-chunk-size <n>    --param-stream-chunk-size (default: 1024)
  --rusb-param-stream-max-bytes <n>     --param-stream-max-bytes (default: 65536)
  --firmware <path>                     Forwarded as --firmware to replay runs
  --rusb-extra-arg <arg>                Extra replay arg; repeat as needed

  --libedgetpu-model <path>             Model for known-good invoke
                                        (default: templates/dense_2048x2048_quant_edgetpu.tflite)
  --libedgetpu-runs <n>                 Positional [runs] for inference_benchmark (default: 20)
  --libedgetpu-warmup <n>               Positional [warmup] for inference_benchmark (default: 5)
  --libedgetpu-extra-arg <arg>          Extra arg for inference_benchmark; repeat as needed

  --bootstrap-env                        Prefix each captured command with:
                                          eval "$(./tools/bootstrap_arch_stack.sh print-env)"
                                         (default: on)
  --no-bootstrap-env                     Disable bootstrap env prefix
  --strict-exit                          Exit non-zero if any captured command fails
  -h, --help                             Show this help text

Examples:
  ./tools/usbmon_side_by_side_capture.sh --bus 4

  ./tools/usbmon_side_by_side_capture.sh \
    --bus 4 \
    --rusb-model templates/dense_2048x2048_quant_edgetpu.tflite \
    --libedgetpu-model templates/dense_2048x2048_quant_edgetpu.tflite \
    --libedgetpu-runs 30 \
    --libedgetpu-warmup 0 \
    --rusb-extra-arg --setup-include-reads
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

validate_uint() {
  local value="$1"
  local label="$2"
  case "$value" in
    ''|*[!0-9]*)
      die "$label must be a non-negative integer"
      ;;
  esac
}

parse_csv_uints() {
  local csv="$1"
  local -n out_ref="$2"
  out_ref=()

  local token trimmed
  local -a raw_tags=()
  IFS=',' read -r -a raw_tags <<< "$csv"
  for token in "${raw_tags[@]}"; do
    trimmed="${token//[[:space:]]/}"
    [[ -n "$trimmed" ]] || continue
    validate_uint "$trimmed" "--rusb-tags entry"
    out_ref+=("$((10#$trimmed))")
  done

  ((${#out_ref[@]} > 0)) || die "--rusb-tags must include at least one tag"
}

join_by_comma() {
  local -n values_ref="$1"
  local joined=""
  local value
  for value in "${values_ref[@]}"; do
    if [[ -z "$joined" ]]; then
      joined="$value"
    else
      joined="${joined},${value}"
    fi
  done
  printf '%s' "$joined"
}

BUS_RAW=""
OUT_DIR=""

RUSB_MODEL="templates/dense_2048x2048_quant_edgetpu.tflite"
RUSB_TAGS_CSV="2,0,1,3,4"
RUSB_RUNS=1
RUSB_INPUT_BYTES=2048
RUSB_OUTPUT_BYTES=2048
RUSB_TIMEOUT_MS=6000
RUSB_PARAM_STREAM_CHUNK_SIZE=1024
RUSB_PARAM_STREAM_MAX_BYTES=65536
FIRMWARE_PATH=""
declare -a RUSB_EXTRA_ARGS=()

LIBEDGETPU_MODEL="templates/dense_2048x2048_quant_edgetpu.tflite"
LIBEDGETPU_RUNS=20
LIBEDGETPU_WARMUP=5
declare -a LIBEDGETPU_EXTRA_ARGS=()

USE_BOOTSTRAP_ENV=1
STRICT_EXIT=0

while (($# > 0)); do
  case "$1" in
    --bus)
      [[ $# -ge 2 ]] || die "missing value for --bus"
      BUS_RAW="$2"
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || die "missing value for --out-dir"
      OUT_DIR="$2"
      shift 2
      ;;
    --rusb-model)
      [[ $# -ge 2 ]] || die "missing value for --rusb-model"
      RUSB_MODEL="$2"
      shift 2
      ;;
    --rusb-tags)
      [[ $# -ge 2 ]] || die "missing value for --rusb-tags"
      RUSB_TAGS_CSV="$2"
      shift 2
      ;;
    --rusb-runs)
      [[ $# -ge 2 ]] || die "missing value for --rusb-runs"
      RUSB_RUNS="$2"
      shift 2
      ;;
    --rusb-input-bytes)
      [[ $# -ge 2 ]] || die "missing value for --rusb-input-bytes"
      RUSB_INPUT_BYTES="$2"
      shift 2
      ;;
    --rusb-output-bytes)
      [[ $# -ge 2 ]] || die "missing value for --rusb-output-bytes"
      RUSB_OUTPUT_BYTES="$2"
      shift 2
      ;;
    --rusb-timeout-ms)
      [[ $# -ge 2 ]] || die "missing value for --rusb-timeout-ms"
      RUSB_TIMEOUT_MS="$2"
      shift 2
      ;;
    --rusb-param-stream-chunk-size)
      [[ $# -ge 2 ]] || die "missing value for --rusb-param-stream-chunk-size"
      RUSB_PARAM_STREAM_CHUNK_SIZE="$2"
      shift 2
      ;;
    --rusb-param-stream-max-bytes)
      [[ $# -ge 2 ]] || die "missing value for --rusb-param-stream-max-bytes"
      RUSB_PARAM_STREAM_MAX_BYTES="$2"
      shift 2
      ;;
    --firmware)
      [[ $# -ge 2 ]] || die "missing value for --firmware"
      FIRMWARE_PATH="$2"
      shift 2
      ;;
    --rusb-extra-arg)
      [[ $# -ge 2 ]] || die "missing value for --rusb-extra-arg"
      RUSB_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --libedgetpu-model)
      [[ $# -ge 2 ]] || die "missing value for --libedgetpu-model"
      LIBEDGETPU_MODEL="$2"
      shift 2
      ;;
    --libedgetpu-runs)
      [[ $# -ge 2 ]] || die "missing value for --libedgetpu-runs"
      LIBEDGETPU_RUNS="$2"
      shift 2
      ;;
    --libedgetpu-warmup)
      [[ $# -ge 2 ]] || die "missing value for --libedgetpu-warmup"
      LIBEDGETPU_WARMUP="$2"
      shift 2
      ;;
    --libedgetpu-extra-arg)
      [[ $# -ge 2 ]] || die "missing value for --libedgetpu-extra-arg"
      LIBEDGETPU_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --bootstrap-env)
      USE_BOOTSTRAP_ENV=1
      shift
      ;;
    --no-bootstrap-env)
      USE_BOOTSTRAP_ENV=0
      shift
      ;;
    --strict-exit)
      STRICT_EXIT=1
      shift
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

[[ -n "$BUS_RAW" ]] || die "--bus is required"
validate_uint "$BUS_RAW" "--bus"
BUS="$((10#$BUS_RAW))"

validate_uint "$RUSB_RUNS" "--rusb-runs"
validate_uint "$RUSB_INPUT_BYTES" "--rusb-input-bytes"
validate_uint "$RUSB_OUTPUT_BYTES" "--rusb-output-bytes"
validate_uint "$RUSB_TIMEOUT_MS" "--rusb-timeout-ms"
validate_uint "$RUSB_PARAM_STREAM_CHUNK_SIZE" "--rusb-param-stream-chunk-size"
validate_uint "$RUSB_PARAM_STREAM_MAX_BYTES" "--rusb-param-stream-max-bytes"
validate_uint "$LIBEDGETPU_RUNS" "--libedgetpu-runs"
validate_uint "$LIBEDGETPU_WARMUP" "--libedgetpu-warmup"

declare -a RUSB_TAGS=()
parse_csv_uints "$RUSB_TAGS_CSV" RUSB_TAGS

[[ -f "$RUSB_MODEL" ]] || die "pure-rusb model not found: $RUSB_MODEL"
[[ -f "$LIBEDGETPU_MODEL" ]] || die "libedgetpu model not found: $LIBEDGETPU_MODEL"
if [[ -n "$FIRMWARE_PATH" ]]; then
  [[ -f "$FIRMWARE_PATH" ]] || die "firmware file not found: $FIRMWARE_PATH"
fi

need_cmd cargo
need_cmd sudo
[[ -x "./tools/usbmon_capture.sh" ]] || die "missing executable: ./tools/usbmon_capture.sh"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="traces/usbmon-side-by-side-${TS}-bus${BUS}"
fi
mkdir -p "$OUT_DIR/pure_rusb_deterministic_sweep"
mkdir -p "$OUT_DIR/libedgetpu_known_good_invoke"

MANIFEST_FILE="$OUT_DIR/capture_manifest.tsv"
printf 'lane\trun_id\texit_code\tcapture_dir\tlog_file\tsummary_file\tcommand\n' > "$MANIFEST_FILE"

declare -a WRAPPED_COMMAND=()
declare -a FAILED_RUNS=()

build_wrapped_command() {
  local -a raw_command=("$@")
  if [[ "$USE_BOOTSTRAP_ENV" -eq 0 ]]; then
    WRAPPED_COMMAND=("${raw_command[@]}")
    return
  fi

  local quoted
  quoted="$(printf '%q ' "${raw_command[@]}")"
  quoted="${quoted% }"
  WRAPPED_COMMAND=(bash -lc "eval \"\$(./tools/bootstrap_arch_stack.sh print-env)\"; ${quoted}")
}

capture_run() {
  local lane="$1"
  local run_id="$2"
  shift 2

  local run_dir="${OUT_DIR}/${lane}/${run_id}"
  mkdir -p "$run_dir"

  local -a raw_command=("$@")
  build_wrapped_command "${raw_command[@]}"

  local raw_display wrapped_display
  raw_display="$(printf '%q ' "${raw_command[@]}")"
  raw_display="${raw_display% }"
  wrapped_display="$(printf '%q ' "${WRAPPED_COMMAND[@]}")"
  wrapped_display="${wrapped_display% }"

  printf '%s\n' "$raw_display" > "${run_dir}/command.raw.txt"
  printf '%s\n' "$wrapped_display" > "${run_dir}/command.capture.txt"

  echo "[capture] lane=${lane} run=${run_id}"
  set +e
  sudo ./tools/usbmon_capture.sh -b "$BUS" -o "$run_dir" -- "${WRAPPED_COMMAND[@]}"
  local rc=$?
  set -e

  local log_file summary_file
  log_file="$(find "$run_dir" -maxdepth 1 -type f -name "usbmon-bus${BUS}-*.log" | sort | tail -n 1)"
  summary_file="$(find "$run_dir" -maxdepth 1 -type f -name "usbmon-bus${BUS}-*.summary.txt" | sort | tail -n 1)"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$lane" \
    "$run_id" \
    "$rc" \
    "$run_dir" \
    "$log_file" \
    "$summary_file" \
    "$wrapped_display" \
    >> "$MANIFEST_FILE"

  if [[ "$rc" -ne 0 ]]; then
    FAILED_RUNS+=("${lane}/${run_id}:${rc}")
    echo "[capture] warning: ${lane}/${run_id} exited with ${rc}"
  fi
}

declare -a RUSB_BASE_ARGS=(
  cargo run --example rusb_serialized_exec_replay -- --model "$RUSB_MODEL"
  --runs "$RUSB_RUNS"
  --input-bytes "$RUSB_INPUT_BYTES"
  --output-bytes "$RUSB_OUTPUT_BYTES"
  --timeout-ms "$RUSB_TIMEOUT_MS"
  --param-stream-chunk-size "$RUSB_PARAM_STREAM_CHUNK_SIZE"
  --param-stream-max-bytes "$RUSB_PARAM_STREAM_MAX_BYTES"
)
if [[ -n "$FIRMWARE_PATH" ]]; then
  RUSB_BASE_ARGS+=(--firmware "$FIRMWARE_PATH")
fi
if ((${#RUSB_EXTRA_ARGS[@]} > 0)); then
  RUSB_BASE_ARGS+=("${RUSB_EXTRA_ARGS[@]}")
fi

for tag in "${RUSB_TAGS[@]}"; do
  capture_run \
    "pure_rusb_deterministic_sweep" \
    "tag${tag}" \
    "${RUSB_BASE_ARGS[@]}" \
    --parameters-tag "$tag"
done

declare -a LIBEDGETPU_ARGS=(
  cargo run --example inference_benchmark -- "$LIBEDGETPU_MODEL" "$LIBEDGETPU_RUNS" "$LIBEDGETPU_WARMUP"
)
if ((${#LIBEDGETPU_EXTRA_ARGS[@]} > 0)); then
  LIBEDGETPU_ARGS+=("${LIBEDGETPU_EXTRA_ARGS[@]}")
fi
capture_run "libedgetpu_known_good_invoke" "inference" "${LIBEDGETPU_ARGS[@]}"

RUSB_TAGS_JOINED="$(join_by_comma RUSB_TAGS)"
FAILED_COUNT="${#FAILED_RUNS[@]}"

README_FILE="$OUT_DIR/README.txt"
{
  echo "USBMON Side-by-Side Capture"
  echo
  echo "timestamp_utc=${TS}"
  echo "bus=${BUS}"
  echo "output_root=${OUT_DIR}"
  echo "manifest=${MANIFEST_FILE}"
  echo
  echo "pure_rusb_example=rusb_serialized_exec_replay"
  echo "pure_rusb_model=${RUSB_MODEL}"
  echo "pure_rusb_tags=${RUSB_TAGS_JOINED}"
  echo "pure_rusb_runs=${RUSB_RUNS}"
  echo "pure_rusb_input_bytes=${RUSB_INPUT_BYTES}"
  echo "pure_rusb_output_bytes=${RUSB_OUTPUT_BYTES}"
  echo "pure_rusb_timeout_ms=${RUSB_TIMEOUT_MS}"
  echo "pure_rusb_param_stream_chunk_size=${RUSB_PARAM_STREAM_CHUNK_SIZE}"
  echo "pure_rusb_param_stream_max_bytes=${RUSB_PARAM_STREAM_MAX_BYTES}"
  if [[ -n "$FIRMWARE_PATH" ]]; then
    echo "pure_rusb_firmware=${FIRMWARE_PATH}"
  fi
  echo
  echo "libedgetpu_example=inference_benchmark"
  echo "libedgetpu_model=${LIBEDGETPU_MODEL}"
  echo "libedgetpu_runs=${LIBEDGETPU_RUNS}"
  echo "libedgetpu_warmup=${LIBEDGETPU_WARMUP}"
  echo
  echo "bootstrap_env=$([[ "$USE_BOOTSTRAP_ENV" -eq 1 ]] && echo on || echo off)"
  echo "failed_runs_count=${FAILED_COUNT}"
  if ((FAILED_COUNT > 0)); then
    echo "failed_runs=$(IFS=,; echo "${FAILED_RUNS[*]}")"
  fi
  echo
  echo "Subdirectories:"
  echo "  pure_rusb_deterministic_sweep/tag<descriptor_tag>/"
  echo "  libedgetpu_known_good_invoke/inference/"
} > "$README_FILE"

echo "Side-by-side capture complete: ${OUT_DIR}"
echo "Manifest: ${MANIFEST_FILE}"
if ((FAILED_COUNT > 0)); then
  echo "Runs with non-zero exit: $(IFS=,; echo "${FAILED_RUNS[*]}")"
  if [[ "$STRICT_EXIT" -eq 1 ]]; then
    exit 1
  fi
fi
