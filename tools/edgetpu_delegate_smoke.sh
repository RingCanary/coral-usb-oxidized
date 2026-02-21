#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'USAGE'
Usage:
  ./tools/edgetpu_delegate_smoke.sh [options]

Compile and run a minimal libedgetpu delegate smoke binary. This is useful
when TensorFlow Lite C library is not installed yet, but libedgetpu is present.

Options:
  -l, --lib-dir <dir>   Directory containing libedgetpu.so (default: EDGETPU_LIB_DIR or /usr/lib)
  -o, --out <path>      Output binary path (default: /tmp/edgetpu_delegate_smoke)
  -h, --help            Show this help text

Examples:
  ./tools/edgetpu_delegate_smoke.sh
  EDGETPU_LIB_DIR=$HOME/.local/lib ./tools/edgetpu_delegate_smoke.sh
  ./tools/edgetpu_delegate_smoke.sh -l $HOME/.local/lib -o /tmp/smoke
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

LIB_DIR="${EDGETPU_LIB_DIR:-/usr/lib}"
OUT_BIN="/tmp/edgetpu_delegate_smoke"

while (($# > 0)); do
  case "$1" in
    -l|--lib-dir)
      [[ $# -ge 2 ]] || die "missing value for $1"
      LIB_DIR="$2"
      shift 2
      ;;
    -o|--out)
      [[ $# -ge 2 ]] || die "missing value for $1"
      OUT_BIN="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ -d "$LIB_DIR" ]] || die "library directory not found: $LIB_DIR"
[[ -f "$LIB_DIR/libedgetpu.so" || -f "$LIB_DIR/libedgetpu.so.1" || -f "$LIB_DIR/libedgetpu.so.1.0" ]] \
  || die "libedgetpu not found in $LIB_DIR"

gcc -O2 -Wall -Wextra -std=c11 \
  tools/edgetpu_delegate_smoke.c \
  -L"$LIB_DIR" \
  -Wl,-rpath,"$LIB_DIR" \
  -ledgetpu \
  -o "$OUT_BIN"

echo "Built $OUT_BIN"
echo "Running with LD_LIBRARY_PATH=$LIB_DIR:${LD_LIBRARY_PATH:-}"
LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" "$OUT_BIN"
