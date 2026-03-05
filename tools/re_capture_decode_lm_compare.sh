#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/archive/re_capture_decode_lm_compare.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "error: missing archived tool: $TARGET" >&2
  exit 2
fi

echo "note: 're_capture_decode_lm_compare.sh' is archived; forwarding to tools/archive/re_capture_decode_lm_compare.sh" >&2
exec "$TARGET" "$@"
