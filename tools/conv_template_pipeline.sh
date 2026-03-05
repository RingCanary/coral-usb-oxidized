#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/archive/conv_template_pipeline.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "error: missing archived tool: $TARGET" >&2
  exit 2
fi

echo "note: 'conv_template_pipeline.sh' is archived; forwarding to tools/archive/conv_template_pipeline.sh" >&2
exec "$TARGET" "$@"
