#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/archive/bootstrap_arch_stack.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "error: missing archived tool: $TARGET" >&2
  exit 2
fi

echo "note: 'bootstrap_arch_stack.sh' is archived; forwarding to tools/archive/bootstrap_arch_stack.sh" >&2
exec "$TARGET" "$@"
