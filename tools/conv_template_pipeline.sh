#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./tools/conv_template_pipeline.sh [options]

Build a single-layer Conv2D INT8 template model, compile it for EdgeTPU, and
run extraction/parser/tensorizer-inspect steps.

Options:
  --out-dir <dir>        Output directory
                         (default: traces/conv-template-<UTC timestamp>)
  --python-version <v>   uv Python version (default: 3.9)
  --tf-version <v>       tensorflow-cpu version for uv run (default: 2.10.1)
  --numpy-version <v>    numpy version for uv run (default: 1.23.5)
  --height <n>           Input height (default: 224)
  --width <n>            Input width (default: 224)
  --in-channels <n>      Input channels (default: 3)
  --out-channels <n>     Output channels / filters (default: 16)
  --kernel-size <n>      Kernel size (default: 3)
  --stride <n>           Stride (default: 1)
  --padding <mode>       same|valid (default: same)
  --init-mode <mode>     delta|ones|zero|random_uniform (default: delta)
  --diag-scale <f>       Scale for delta/ones init (default: 1.0)
  --use-bias             Enable Conv2D bias.
  --seed <n>             RNG seed (default: 1337)
  --rep-samples <n>      Representative dataset sample count (default: 128)
  --rep-range <f>        Representative sample value range (default: 1.0)
  --compiler <path>      Use explicit edgetpu_compiler path
  --run-benchmark        Run Rust benchmark on compiled model.
  -h, --help             Show this help text

Examples:
  ./tools/conv_template_pipeline.sh
  ./tools/conv_template_pipeline.sh --out-channels 32 --kernel-size 1 --init-mode ones
  ./tools/conv_template_pipeline.sh --height 128 --width 128 --run-benchmark
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

timestamp_utc() {
  date -u +"%Y%m%dT%H%M%SZ"
}

OUT_DIR="traces/conv-template-$(timestamp_utc)"
PYTHON_VERSION="3.9"
TF_VERSION="2.10.1"
NUMPY_VERSION="1.23.5"
HEIGHT=224
WIDTH=224
IN_CHANNELS=3
OUT_CHANNELS=16
KERNEL_SIZE=3
STRIDE=1
PADDING="same"
INIT_MODE="delta"
DIAG_SCALE="1.0"
USE_BIAS=0
SEED=1337
REP_SAMPLES=128
REP_RANGE="1.0"
COMPILER_PATH=""
RUN_BENCHMARK=0

while (($# > 0)); do
  case "$1" in
    --out-dir)
      [[ $# -ge 2 ]] || die "missing value for --out-dir"
      OUT_DIR="$2"
      shift 2
      ;;
    --python-version)
      [[ $# -ge 2 ]] || die "missing value for --python-version"
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --tf-version)
      [[ $# -ge 2 ]] || die "missing value for --tf-version"
      TF_VERSION="$2"
      shift 2
      ;;
    --numpy-version)
      [[ $# -ge 2 ]] || die "missing value for --numpy-version"
      NUMPY_VERSION="$2"
      shift 2
      ;;
    --height)
      [[ $# -ge 2 ]] || die "missing value for --height"
      HEIGHT="$2"
      shift 2
      ;;
    --width)
      [[ $# -ge 2 ]] || die "missing value for --width"
      WIDTH="$2"
      shift 2
      ;;
    --in-channels)
      [[ $# -ge 2 ]] || die "missing value for --in-channels"
      IN_CHANNELS="$2"
      shift 2
      ;;
    --out-channels)
      [[ $# -ge 2 ]] || die "missing value for --out-channels"
      OUT_CHANNELS="$2"
      shift 2
      ;;
    --kernel-size)
      [[ $# -ge 2 ]] || die "missing value for --kernel-size"
      KERNEL_SIZE="$2"
      shift 2
      ;;
    --stride)
      [[ $# -ge 2 ]] || die "missing value for --stride"
      STRIDE="$2"
      shift 2
      ;;
    --padding)
      [[ $# -ge 2 ]] || die "missing value for --padding"
      PADDING="$2"
      shift 2
      ;;
    --init-mode)
      [[ $# -ge 2 ]] || die "missing value for --init-mode"
      INIT_MODE="$2"
      shift 2
      ;;
    --diag-scale)
      [[ $# -ge 2 ]] || die "missing value for --diag-scale"
      DIAG_SCALE="$2"
      shift 2
      ;;
    --use-bias)
      USE_BIAS=1
      shift
      ;;
    --seed)
      [[ $# -ge 2 ]] || die "missing value for --seed"
      SEED="$2"
      shift 2
      ;;
    --rep-samples)
      [[ $# -ge 2 ]] || die "missing value for --rep-samples"
      REP_SAMPLES="$2"
      shift 2
      ;;
    --rep-range)
      [[ $# -ge 2 ]] || die "missing value for --rep-range"
      REP_RANGE="$2"
      shift 2
      ;;
    --compiler)
      [[ $# -ge 2 ]] || die "missing value for --compiler"
      COMPILER_PATH="$2"
      shift 2
      ;;
    --run-benchmark)
      RUN_BENCHMARK=1
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

case "${PADDING}" in
  same|valid) ;;
  *) die "invalid --padding '${PADDING}'" ;;
esac

case "${INIT_MODE}" in
  delta|ones|zero|random_uniform) ;;
  *) die "invalid --init-mode '${INIT_MODE}'" ;;
esac

need_cmd uv
need_cmd python3

mkdir -p "${OUT_DIR}"

MODEL_BASENAME="conv2d_${HEIGHT}x${WIDTH}x${IN_CHANNELS}_to_${OUT_CHANNELS}_k${KERNEL_SIZE}_s${STRIDE}_${PADDING}_quant"
QUANT_MODEL="${OUT_DIR}/${MODEL_BASENAME}.tflite"
CONV_META="${OUT_DIR}/${MODEL_BASENAME}.metadata.json"
COMPILE_LOG="${OUT_DIR}/edgetpu_compile.log"
EXTRACT_DIR="${OUT_DIR}/extract"
PARSE_TXT="${OUT_DIR}/exec_parse.txt"
PARSE_JSON="${OUT_DIR}/exec_parse.json"
INSPECT_TXT="${OUT_DIR}/tensorizer_inspect.txt"
INSPECT_JSON="${OUT_DIR}/tensorizer_inspect.json"
PIPELINE_SUMMARY="${OUT_DIR}/PIPELINE_SUMMARY.txt"
BENCH_COMPILED_LOG=""

echo "[1/6] Generating Conv2D quantized model via uv..."
uv python install "${PYTHON_VERSION}" >/dev/null
uv run --python "${PYTHON_VERSION}" \
  --with "tensorflow-cpu==${TF_VERSION}" \
  --with "numpy==${NUMPY_VERSION}" \
  tools/generate_conv2d_quant_tflite.py \
  --output "${QUANT_MODEL}" \
  --metadata-out "${CONV_META}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --in-channels "${IN_CHANNELS}" \
  --out-channels "${OUT_CHANNELS}" \
  --kernel-size "${KERNEL_SIZE}" \
  --stride "${STRIDE}" \
  --padding "${PADDING}" \
  --init-mode "${INIT_MODE}" \
  --diag-scale "${DIAG_SCALE}" \
  --seed "${SEED}" \
  --rep-samples "${REP_SAMPLES}" \
  --rep-range "${REP_RANGE}" \
  $( ((USE_BIAS == 1)) && printf '%s' '--use-bias' )

if [[ -z "${COMPILER_PATH}" ]]; then
  if command -v edgetpu_compiler >/dev/null 2>&1; then
    COMPILER_PATH="$(command -v edgetpu_compiler)"
  else
    echo "[2/6] edgetpu_compiler not found in PATH; bootstrapping local binary..."
    ./tools/bootstrap_edgetpu_compiler.sh install
    COMPILER_PATH="${HOME}/.local/bin/edgetpu_compiler"
  fi
fi

[[ -x "${COMPILER_PATH}" ]] || die "edgetpu_compiler not executable: ${COMPILER_PATH}"
echo "[2/6] Using compiler: ${COMPILER_PATH}"
"${COMPILER_PATH}" --version

echo "[3/6] Compiling model for EdgeTPU..."
"${COMPILER_PATH}" -s -o "${OUT_DIR}" "${QUANT_MODEL}" 2>&1 | tee "${COMPILE_LOG}"

EDGETPU_MODEL="${OUT_DIR}/${MODEL_BASENAME}_edgetpu.tflite"
[[ -f "${EDGETPU_MODEL}" ]] || die "compiled model not found: ${EDGETPU_MODEL}"

echo "[4/6] Extracting DWN1 package..."
python3 tools/extract_edgetpu_package.py extract "${EDGETPU_MODEL}" --out "${EXTRACT_DIR}" --overwrite

echo "[5/6] Parsing executable internals..."
python3 tools/parse_edgetpu_executable.py "${EXTRACT_DIR}/package_000" > "${PARSE_TXT}"
python3 tools/parse_edgetpu_executable.py --json "${EXTRACT_DIR}/package_000" > "${PARSE_JSON}"

echo "[6/6] Tensorizer inspect..."
python3 tools/tensorizer_patch_edgetpu.py inspect "${EDGETPU_MODEL}" > "${INSPECT_TXT}"
python3 tools/tensorizer_patch_edgetpu.py inspect --json "${EDGETPU_MODEL}" > "${INSPECT_JSON}"

if ((RUN_BENCHMARK == 1)); then
  BENCH_COMPILED_LOG="${OUT_DIR}/bench_compiled.log"
  echo "[bench] Running inference_benchmark on compiled model..."
  cargo run --example inference_benchmark -- "${EDGETPU_MODEL}" 5 1 | tee "${BENCH_COMPILED_LOG}"
fi

{
  echo "Conv2D Template Pipeline Summary"
  echo "out_dir=${OUT_DIR}"
  echo "quant_model=${QUANT_MODEL}"
  echo "compiled_model=${EDGETPU_MODEL}"
  echo "compiler=${COMPILER_PATH}"
  echo "python_version=${PYTHON_VERSION}"
  echo "tensorflow_cpu_version=${TF_VERSION}"
  echo "numpy_version=${NUMPY_VERSION}"
  echo "extract_dir=${EXTRACT_DIR}"
  echo "parse_text=${PARSE_TXT}"
  echo "parse_json=${PARSE_JSON}"
  echo "inspect_text=${INSPECT_TXT}"
  echo "inspect_json=${INSPECT_JSON}"
  if [[ -n "${BENCH_COMPILED_LOG}" ]]; then
    echo "bench_compiled_log=${BENCH_COMPILED_LOG}"
  fi
} | tee "${PIPELINE_SUMMARY}"

echo
echo "Done. Summary: ${PIPELINE_SUMMARY}"
