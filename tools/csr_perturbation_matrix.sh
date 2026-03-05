#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/archive/csr_perturbation_matrix.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "error: missing archived tool: $TARGET" >&2
  exit 2
fi

echo "note: 'csr_perturbation_matrix.sh' is archived; forwarding to tools/archive/csr_perturbation_matrix.sh" >&2
exec "$TARGET" "$@"
