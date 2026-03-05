#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mapfile -t TOOL_PATHS < <(
  rg -n --no-heading -o 'tools/[A-Za-z0-9_./-]+' src scripts tools/*.sh tools/archive/*.sh 2>/dev/null \
    | awk -F: '{print $NF}' \
    | sort -u
)

missing=0
for path in "${TOOL_PATHS[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "missing tool path: $path" >&2
    rg -nF "$path" src scripts tools/*.sh tools/archive/*.sh 2>/dev/null || true
    missing=1
  fi
done

if (( missing != 0 )); then
  echo "tool prune safety check failed" >&2
  exit 1
fi

echo "tool prune safety check passed (${#TOOL_PATHS[@]} referenced paths)"
