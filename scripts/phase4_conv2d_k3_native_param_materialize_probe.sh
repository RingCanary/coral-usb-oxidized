#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_REL="${ARTIFACT_REL:-traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z}"
ARTIFACT_DIR="${ROOT_DIR}/${ARTIFACT_REL}"

if [[ ! -d "${ARTIFACT_DIR}" ]]; then
  echo "artifact dir not found: ${ARTIFACT_DIR}" >&2
  exit 1
fi

echo "artifact_rel=${ARTIFACT_REL}"

for pair_dir in "${ARTIFACT_DIR}"/p*; do
  [[ -d "${pair_dir}" ]] || continue
  pair_id="$(basename "${pair_dir}")"
  model_path="$(find "${pair_dir}/target" -maxdepth 1 -name '*.tflite' ! -name '*_edgetpu.tflite' | sort | head -n1)"
  metadata_path="$(find "${pair_dir}/target" -maxdepth 1 -name '*.metadata.json' | sort | head -n1)"
  out_path="${pair_dir}/target_param_stream.native.bin"
  summary_path="${pair_dir}/NATIVE_PARAM_MATERIALIZE.txt"
  cargo run --quiet --bin conv_k_param_materialize -- \
    --model "${model_path}" \
    --metadata "${metadata_path}" \
    --out "${out_path}" \
    --verify-against "${pair_dir}/target_param_stream.bin" \
    > "${summary_path}"
  cat "${summary_path}"
done
