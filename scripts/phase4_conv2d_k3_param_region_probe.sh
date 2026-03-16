#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_REL="${ARTIFACT_REL:-traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z}"
ARTIFACT_DIR="$REPO_ROOT/$ARTIFACT_REL"

[[ -d "$ARTIFACT_DIR" ]] || { echo "artifact dir not found: $ARTIFACT_DIR" >&2; exit 1; }

echo "artifact_rel=$ARTIFACT_REL"
cargo run --quiet --bin conv_k3_param_anatomy -- --run-dir "$ARTIFACT_DIR"
