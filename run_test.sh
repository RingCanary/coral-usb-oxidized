#!/usr/bin/env bash
set -euo pipefail

# Optional override if tensorflowlite_c is installed in a non-standard path.
# Example (Raspberry Pi 5):
#   TFLITE_LIB_DIR=/usr/lib/aarch64-linux-gnu ./run_test.sh
if [[ -n "${TFLITE_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${TFLITE_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

cargo run --example tflite_test
