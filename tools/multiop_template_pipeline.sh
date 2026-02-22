#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./tools/multiop_template_pipeline.sh [options]

Build a Conv2D->Dense INT8 template model, compile it for EdgeTPU, and run
extraction/parser/tensorizer-inspect steps.

Options:
  --out-dir <dir>          Output directory
                           (default: traces/multiop-template-<UTC timestamp>)
  --python-version <v>     uv Python version (default: 3.9)
  --tf-version <v>         tensorflow-cpu version for uv run (default: 2.10.1)
  --numpy-version <v>      numpy version for uv run (default: 1.23.5)

  --height <n>             Input height (default: 16)
  --width <n>              Input width (default: 16)
  --in-channels <n>        Input channels (default: 16)

  --conv-filters <n>       Conv2D filters (default: 64)
  --conv-kernel-size <n>   Conv2D kernel size (default: 1)
  --conv-stride <n>        Conv2D stride (default: 1)
  --conv-padding <mode>    same|valid (default: same)
  --conv-init-mode <mode>  delta|ones|zero|random_uniform (default: delta)
  --conv-diag-scale <f>    Conv2D init scale (default: 1.0)

  --dense-units <n>        Dense output units (default: 256)
  --dense-init-mode <mode> identity|permutation|ones|zero|random_uniform
                           (default: identity)
  --dense-diag-scale <f>   Dense init scale (default: 1.0)

  --use-bias               Enable layer bias.
  --seed <n>               RNG seed (default: 1337)
  --rep-samples <n>        Representative dataset samples (default: 128)
  --rep-range <f>          Representative value range (default: 1.0)

  --compiler <path>        Use explicit edgetpu_compiler path
  --run-benchmark          Run Rust benchmark on compiled model.
  -h, --help               Show this help text

Examples:
  ./tools/multiop_template_pipeline.sh
  ./tools/multiop_template_pipeline.sh --conv-init-mode ones --dense-init-mode permutation
  ./tools/multiop_template_pipeline.sh --height 32 --width 32 --in-channels 8 --dense-units 128
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

OUT_DIR="traces/multiop-template-$(timestamp_utc)"
PYTHON_VERSION="3.9"
TF_VERSION="2.10.1"
NUMPY_VERSION="1.23.5"

HEIGHT=16
WIDTH=16
IN_CHANNELS=16

CONV_FILTERS=64
CONV_KERNEL_SIZE=1
CONV_STRIDE=1
CONV_PADDING="same"
CONV_INIT_MODE="delta"
CONV_DIAG_SCALE="1.0"

DENSE_UNITS=256
DENSE_INIT_MODE="identity"
DENSE_DIAG_SCALE="1.0"

USE_BIAS=0
SEED=1337
REP_SAMPLES=128
REP_RANGE="1.0"

COMPILER_PATH=""
RUN_BENCHMARK=0

while (($# > 0)); do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --python-version) PYTHON_VERSION="$2"; shift 2 ;;
    --tf-version) TF_VERSION="$2"; shift 2 ;;
    --numpy-version) NUMPY_VERSION="$2"; shift 2 ;;

    --height) HEIGHT="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --in-channels) IN_CHANNELS="$2"; shift 2 ;;

    --conv-filters) CONV_FILTERS="$2"; shift 2 ;;
    --conv-kernel-size) CONV_KERNEL_SIZE="$2"; shift 2 ;;
    --conv-stride) CONV_STRIDE="$2"; shift 2 ;;
    --conv-padding) CONV_PADDING="$2"; shift 2 ;;
    --conv-init-mode) CONV_INIT_MODE="$2"; shift 2 ;;
    --conv-diag-scale) CONV_DIAG_SCALE="$2"; shift 2 ;;

    --dense-units) DENSE_UNITS="$2"; shift 2 ;;
    --dense-init-mode) DENSE_INIT_MODE="$2"; shift 2 ;;
    --dense-diag-scale) DENSE_DIAG_SCALE="$2"; shift 2 ;;

    --use-bias) USE_BIAS=1; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --rep-samples) REP_SAMPLES="$2"; shift 2 ;;
    --rep-range) REP_RANGE="$2"; shift 2 ;;

    --compiler) COMPILER_PATH="$2"; shift 2 ;;
    --run-benchmark) RUN_BENCHMARK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "${CONV_PADDING}" in
  same|valid) ;;
  *) die "invalid --conv-padding '${CONV_PADDING}'" ;;
esac

case "${CONV_INIT_MODE}" in
  delta|ones|zero|random_uniform) ;;
  *) die "invalid --conv-init-mode '${CONV_INIT_MODE}'" ;;
esac

case "${DENSE_INIT_MODE}" in
  identity|permutation|ones|zero|random_uniform) ;;
  *) die "invalid --dense-init-mode '${DENSE_INIT_MODE}'" ;;
esac

need_cmd uv
need_cmd python3

mkdir -p "${OUT_DIR}"

MODEL_BASENAME="denseconv_${HEIGHT}x${WIDTH}x${IN_CHANNELS}_conv${CONV_FILTERS}_k${CONV_KERNEL_SIZE}_dense${DENSE_UNITS}_quant"
QUANT_MODEL="${OUT_DIR}/${MODEL_BASENAME}.tflite"
META_JSON="${OUT_DIR}/${MODEL_BASENAME}.metadata.json"
COMPILE_LOG="${OUT_DIR}/edgetpu_compile.log"
EXTRACT_DIR="${OUT_DIR}/extract"
PARSE_TXT="${OUT_DIR}/exec_parse.txt"
PARSE_JSON="${OUT_DIR}/exec_parse.json"
INSPECT_TXT="${OUT_DIR}/tensorizer_inspect.txt"
INSPECT_JSON="${OUT_DIR}/tensorizer_inspect.json"
PIPELINE_SUMMARY="${OUT_DIR}/PIPELINE_SUMMARY.txt"
BENCH_COMPILED_LOG=""

echo "[1/6] Generating Conv2D->Dense quantized model via uv..."
uv python install "${PYTHON_VERSION}" >/dev/null
uv run --python "${PYTHON_VERSION}" \
  --with "tensorflow-cpu==${TF_VERSION}" \
  --with "numpy==${NUMPY_VERSION}" \
  tools/generate_dense_conv_quant_tflite.py \
  --output "${QUANT_MODEL}" \
  --metadata-out "${META_JSON}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --in-channels "${IN_CHANNELS}" \
  --conv-filters "${CONV_FILTERS}" \
  --conv-kernel-size "${CONV_KERNEL_SIZE}" \
  --conv-stride "${CONV_STRIDE}" \
  --conv-padding "${CONV_PADDING}" \
  --conv-init-mode "${CONV_INIT_MODE}" \
  --conv-diag-scale "${CONV_DIAG_SCALE}" \
  --dense-units "${DENSE_UNITS}" \
  --dense-init-mode "${DENSE_INIT_MODE}" \
  --dense-diag-scale "${DENSE_DIAG_SCALE}" \
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
  echo "Multi-op Template Pipeline Summary"
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
