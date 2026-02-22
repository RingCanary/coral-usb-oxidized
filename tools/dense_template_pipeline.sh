#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./tools/dense_template_pipeline.sh [options]

Build a single-layer Dense INT8 template model, compile it for EdgeTPU, and
run extraction/parser/tensorizer-inspect steps.

Options:
  --out-dir <dir>        Output directory
                         (default: traces/dense-template-<UTC timestamp>)
  --python-version <v>   uv Python version (default: 3.9)
  --tf-package <name>    TensorFlow package for uv run (default: tensorflow-cpu
                         on x86_64, tensorflow on aarch64/arm64)
  --tf-version <v>       TensorFlow version for uv run (default: package-specific)
  --numpy-version <v>    numpy version for uv run (default: package-specific)
  --input-dim <n>        Dense input dimension (default: 256)
  --output-dim <n>       Dense output dimension (default: 256)
  --init-mode <mode>     identity|permutation|ones|random_uniform
                         (default: identity)
  --diag-scale <f>       Scale for identity/permutation/ones init (default: 1.0)
  --seed <n>             RNG seed (default: 1337)
  --rep-samples <n>      Representative dataset sample count (default: 256)
  --rep-range <f>        Representative sample value range (default: 1.0)
  --compiler <path>      Use explicit edgetpu_compiler path
  --patch-mode <mode>    none|zero|byte|ramp|xor|random (default: none)
  --patch-byte <0-255>   Byte value for patch mode byte/xor (default: 255)
  --patch-seed <n>       Seed for patch mode random (default: 1337)
  --run-benchmark        Run Rust benchmark for compiled and patched model.
  -h, --help             Show this help text

Examples:
  ./tools/dense_template_pipeline.sh
  ./tools/dense_template_pipeline.sh --init-mode permutation --patch-mode zero
  ./tools/dense_template_pipeline.sh --run-benchmark --patch-mode ramp
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

OUT_DIR="traces/dense-template-$(timestamp_utc)"
PYTHON_VERSION="3.9"
TF_PACKAGE="tensorflow-cpu"
TF_VERSION=""
NUMPY_VERSION="1.23.5"
INPUT_DIM=256
OUTPUT_DIM=256
INIT_MODE="identity"
DIAG_SCALE="1.0"
SEED=1337
REP_SAMPLES=256
REP_RANGE="1.0"
COMPILER_PATH=""
PATCH_MODE="none"
PATCH_BYTE=255
PATCH_SEED=1337
RUN_BENCHMARK=0
TF_PACKAGE_SET=0
TF_VERSION_SET=0
NUMPY_VERSION_SET=0

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
    --tf-package)
      [[ $# -ge 2 ]] || die "missing value for --tf-package"
      TF_PACKAGE="$2"
      TF_PACKAGE_SET=1
      shift 2
      ;;
    --tf-version)
      [[ $# -ge 2 ]] || die "missing value for --tf-version"
      TF_VERSION="$2"
      TF_VERSION_SET=1
      shift 2
      ;;
    --numpy-version)
      [[ $# -ge 2 ]] || die "missing value for --numpy-version"
      NUMPY_VERSION="$2"
      NUMPY_VERSION_SET=1
      shift 2
      ;;
    --input-dim)
      [[ $# -ge 2 ]] || die "missing value for --input-dim"
      INPUT_DIM="$2"
      shift 2
      ;;
    --output-dim)
      [[ $# -ge 2 ]] || die "missing value for --output-dim"
      OUTPUT_DIM="$2"
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
    --patch-mode)
      [[ $# -ge 2 ]] || die "missing value for --patch-mode"
      PATCH_MODE="$2"
      shift 2
      ;;
    --patch-byte)
      [[ $# -ge 2 ]] || die "missing value for --patch-byte"
      PATCH_BYTE="$2"
      shift 2
      ;;
    --patch-seed)
      [[ $# -ge 2 ]] || die "missing value for --patch-seed"
      PATCH_SEED="$2"
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

case "${INIT_MODE}" in
  identity|permutation|ones|random_uniform) ;;
  *) die "invalid --init-mode '${INIT_MODE}'" ;;
esac

case "${PATCH_MODE}" in
  none|zero|byte|ramp|xor|random) ;;
  *) die "invalid --patch-mode '${PATCH_MODE}'" ;;
esac

need_cmd uv
need_cmd python3

# TensorFlow wheel availability differs by architecture. Fall back to the
# monolithic tensorflow package on arm64 when package wasn't explicitly set.
arch="$(uname -m)"
if [[ "${TF_PACKAGE_SET}" -eq 0 ]]; then
  case "${arch}" in
    aarch64|arm64)
      TF_PACKAGE="tensorflow"
      ;;
    *)
      TF_PACKAGE="tensorflow-cpu"
      ;;
  esac
fi

# If version wasn't explicitly provided, choose a package-compatible default.
if [[ "${TF_VERSION_SET}" -eq 0 ]]; then
  case "${TF_PACKAGE}" in
    tensorflow-cpu)
      TF_VERSION="2.10.1"
      ;;
    tensorflow)
      TF_VERSION="2.19.0"
      ;;
    *)
      die "no default --tf-version for --tf-package ${TF_PACKAGE}; set --tf-version explicitly"
      ;;
  esac
fi

# Keep numpy defaults aligned with the TensorFlow package/version family.
if [[ "${NUMPY_VERSION_SET}" -eq 0 ]]; then
  case "${TF_PACKAGE}" in
    tensorflow-cpu)
      NUMPY_VERSION="1.23.5"
      ;;
    tensorflow)
      NUMPY_VERSION="1.26.4"
      ;;
    *)
      die "no default --numpy-version for --tf-package ${TF_PACKAGE}; set --numpy-version explicitly"
      ;;
  esac
fi

mkdir -p "${OUT_DIR}"

MODEL_BASENAME="dense_${INPUT_DIM}x${OUTPUT_DIM}_quant"
QUANT_MODEL="${OUT_DIR}/${MODEL_BASENAME}.tflite"
DENSE_META="${OUT_DIR}/${MODEL_BASENAME}.metadata.json"
COMPILE_LOG="${OUT_DIR}/edgetpu_compile.log"
EXTRACT_DIR="${OUT_DIR}/extract"
PARSE_TXT="${OUT_DIR}/exec_parse.txt"
PARSE_JSON="${OUT_DIR}/exec_parse.json"
INSPECT_TXT="${OUT_DIR}/tensorizer_inspect.txt"
INSPECT_JSON="${OUT_DIR}/tensorizer_inspect.json"
PIPELINE_SUMMARY="${OUT_DIR}/PIPELINE_SUMMARY.txt"

echo "[1/6] Generating dense quantized model via uv..."
uv python install "${PYTHON_VERSION}" >/dev/null
uv run --python "${PYTHON_VERSION}" \
  --with "${TF_PACKAGE}==${TF_VERSION}" \
  --with "numpy==${NUMPY_VERSION}" \
  tools/generate_dense_quant_tflite.py \
  --output "${QUANT_MODEL}" \
  --metadata-out "${DENSE_META}" \
  --input-dim "${INPUT_DIM}" \
  --output-dim "${OUTPUT_DIM}" \
  --init-mode "${INIT_MODE}" \
  --diag-scale "${DIAG_SCALE}" \
  --seed "${SEED}" \
  --rep-samples "${REP_SAMPLES}" \
  --rep-range "${REP_RANGE}"

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

PATCHED_MODEL=""
PATCH_META=""
BENCH_COMPILED_LOG=""
BENCH_PATCHED_LOG=""

if [[ "${PATCH_MODE}" != "none" ]]; then
  PATCHED_MODEL="${OUT_DIR}/${MODEL_BASENAME}_edgetpu_patched_${PATCH_MODE}.tflite"
  PATCH_META="${OUT_DIR}/patch_${PATCH_MODE}.json"
  patch_args=(
    --mode "${PATCH_MODE}"
    --exec-type parameter_caching
  )
  if [[ "${PATCH_MODE}" == "byte" || "${PATCH_MODE}" == "xor" ]]; then
    patch_args+=(--byte-value "${PATCH_BYTE}")
  fi
  if [[ "${PATCH_MODE}" == "random" ]]; then
    patch_args+=(--seed "${PATCH_SEED}")
  fi

  python3 tools/tensorizer_patch_edgetpu.py patch "${EDGETPU_MODEL}" \
    --output "${PATCHED_MODEL}" \
    --overwrite \
    --metadata-out "${PATCH_META}" \
    "${patch_args[@]}" >/dev/null
fi

if ((RUN_BENCHMARK == 1)); then
  BENCH_COMPILED_LOG="${OUT_DIR}/bench_compiled.log"
  echo "[bench] Running inference_benchmark on compiled model..."
  cargo run --example inference_benchmark -- "${EDGETPU_MODEL}" 5 1 | tee "${BENCH_COMPILED_LOG}"
  if [[ -n "${PATCHED_MODEL}" ]]; then
    BENCH_PATCHED_LOG="${OUT_DIR}/bench_patched.log"
    echo "[bench] Running inference_benchmark on patched model..."
    cargo run --example inference_benchmark -- "${PATCHED_MODEL}" 5 1 | tee "${BENCH_PATCHED_LOG}"
  fi
fi

{
  echo "Dense Template Pipeline Summary"
  echo "out_dir=${OUT_DIR}"
  echo "quant_model=${QUANT_MODEL}"
  echo "compiled_model=${EDGETPU_MODEL}"
  echo "compiler=${COMPILER_PATH}"
  echo "python_version=${PYTHON_VERSION}"
  echo "tensorflow_package=${TF_PACKAGE}"
  echo "tensorflow_version=${TF_VERSION}"
  echo "numpy_version=${NUMPY_VERSION}"
  echo "extract_dir=${EXTRACT_DIR}"
  echo "parse_text=${PARSE_TXT}"
  echo "parse_json=${PARSE_JSON}"
  echo "inspect_text=${INSPECT_TXT}"
  echo "inspect_json=${INSPECT_JSON}"
  if [[ -n "${PATCHED_MODEL}" ]]; then
    echo "patched_model=${PATCHED_MODEL}"
    echo "patched_metadata=${PATCH_META}"
  fi
  if [[ -n "${BENCH_COMPILED_LOG}" ]]; then
    echo "bench_compiled_log=${BENCH_COMPILED_LOG}"
  fi
  if [[ -n "${BENCH_PATCHED_LOG}" ]]; then
    echo "bench_patched_log=${BENCH_PATCHED_LOG}"
  fi
} | tee "${PIPELINE_SUMMARY}"

echo
echo "Done. Summary: ${PIPELINE_SUMMARY}"
