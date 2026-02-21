#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'USAGE'
Usage:
  ./tools/usb_syscall_trace.sh [options] -- command [args...]
  ./tools/usb_syscall_trace.sh [options] -c "command string"

Trace user-space USB activity with strace (no root required for your own process).

Options:
  -o, --out-dir <dir>      Output directory (default: ./traces/usb-syscall-<timestamp>)
  -c, --command <string>   Command string to run under strace
  -h, --help               Show this help text

Examples:
  ./tools/usb_syscall_trace.sh -- cargo run --example verify_device
  ./tools/usb_syscall_trace.sh -c "cargo run --example delegate_usage"
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

ensure_dep() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "required command not found: ${cmd}"
}

OUT_DIR=""
COMMAND_STRING=""
declare -a COMMAND_ARGS=()

while (($# > 0)); do
  case "$1" in
    -o|--out-dir)
      [[ $# -ge 2 ]] || die "missing value for $1"
      OUT_DIR="$2"
      shift 2
      ;;
    -c|--command)
      [[ $# -ge 2 ]] || die "missing value for $1"
      COMMAND_STRING="$2"
      shift 2
      ;;
    --)
      shift
      if (($# > 0)); then
        COMMAND_ARGS=("$@")
      fi
      break
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

if [[ -n "$COMMAND_STRING" && ${#COMMAND_ARGS[@]} -gt 0 ]]; then
  die "use either -c/--command or -- command [args...], not both"
fi

if [[ -z "$COMMAND_STRING" && ${#COMMAND_ARGS[@]} -eq 0 ]]; then
  die "no command provided"
fi

ensure_dep strace
ensure_dep awk
ensure_dep sort

TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="traces/usb-syscall-${TS}"
fi
mkdir -p "$OUT_DIR"

TRACE_FILE="${OUT_DIR}/strace-usb-${TS}.log"
SUMMARY_FILE="${OUT_DIR}/strace-usb-${TS}.summary.txt"

COMMAND_DISPLAY=""
if [[ -n "$COMMAND_STRING" ]]; then
  COMMAND_DISPLAY="$COMMAND_STRING"
else
  COMMAND_DISPLAY="$(printf '%q ' "${COMMAND_ARGS[@]}")"
  COMMAND_DISPLAY="${COMMAND_DISPLAY% }"
fi

STRACE_FILTER='open,openat,openat2,close,ioctl,read,write,pread64,pwrite64,poll,ppoll,select,pselect6'
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

echo "Tracing command with strace to ${TRACE_FILE}"
echo "Command: ${COMMAND_DISPLAY}"

set +e
if [[ -n "$COMMAND_STRING" ]]; then
  strace -f -yy -tt -T -s 256 -o "$TRACE_FILE" -e "trace=${STRACE_FILTER}" -- bash -lc "$COMMAND_STRING"
  COMMAND_EXIT=$?
else
  strace -f -yy -tt -T -s 256 -o "$TRACE_FILE" -e "trace=${STRACE_FILTER}" -- "${COMMAND_ARGS[@]}"
  COMMAND_EXIT=$?
fi
set -e

END_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"
DURATION_SEC=$((END_EPOCH - START_EPOCH))

TOTAL_LINES="$(wc -l < "$TRACE_FILE" | tr -d ' ')"
USB_RELATED_LINES="$(awk '/\/dev\/bus\/usb\/|USBDEVFS_/ {count++} END {print count+0}' "$TRACE_FILE")"
USB_ERROR_LINES="$(awk '/\/dev\/bus\/usb\/|USBDEVFS_/ && / = -1 / {count++} END {print count+0}' "$TRACE_FILE")"

DEVICE_FILE="$(mktemp)"
SYSCALL_FILE="$(mktemp)"
IOCTL_FILE="$(mktemp)"

awk '{
  line = $0
  while (match(line, /\/dev\/bus\/usb\/[0-9][0-9][0-9]\/[0-9][0-9][0-9]/)) {
    print substr(line, RSTART, RLENGTH)
    line = substr(line, RSTART + RLENGTH)
  }
}' "$TRACE_FILE" | sort -u >"$DEVICE_FILE"

awk '
  /\/dev\/bus\/usb\/|USBDEVFS_/ {
    if (match($0, /[A-Za-z_][A-Za-z0-9_]*\(/)) {
      call = substr($0, RSTART, RLENGTH - 1)
      counts[call]++
    }
  }
  END {
    for (k in counts) {
      printf "%s %d\n", k, counts[k]
    }
  }
' "$TRACE_FILE" | sort >"$SYSCALL_FILE"

awk '
  /ioctl\(.*USBDEVFS_/ {
    if (match($0, /USBDEVFS_[A-Z0-9_]+/)) {
      request = substr($0, RSTART, RLENGTH)
      counts[request]++
    }
  }
  END {
    for (k in counts) {
      printf "%s %d\n", k, counts[k]
    }
  }
' "$TRACE_FILE" | sort >"$IOCTL_FILE"

{
  echo "USB Syscall Trace Summary"
  echo "start_utc=${START_TS}"
  echo "end_utc=${END_TS}"
  echo "duration_seconds=${DURATION_SEC}"
  echo "trace_file=${TRACE_FILE}"
  echo "command=${COMMAND_DISPLAY}"
  echo "command_exit=${COMMAND_EXIT}"
  echo "total_strace_lines=${TOTAL_LINES}"
  echo "usb_related_lines=${USB_RELATED_LINES}"
  echo "usb_related_error_lines=${USB_ERROR_LINES}"
  echo
  echo "USB device nodes touched:"
  if [[ -s "$DEVICE_FILE" ]]; then
    sed 's/^/  /' "$DEVICE_FILE"
  else
    echo "  (none detected)"
  fi
  echo
  echo "USB-related syscall counts:"
  if [[ -s "$SYSCALL_FILE" ]]; then
    sed 's/^/  /' "$SYSCALL_FILE"
  else
    echo "  (none detected)"
  fi
  echo
  echo "USBDEVFS ioctl request counts:"
  if [[ -s "$IOCTL_FILE" ]]; then
    sed 's/^/  /' "$IOCTL_FILE"
  else
    echo "  (none detected)"
  fi
} >"$SUMMARY_FILE"

rm -f "$DEVICE_FILE" "$SYSCALL_FILE" "$IOCTL_FILE"

echo "Summary written to ${SUMMARY_FILE}"
echo "Trace log written to ${TRACE_FILE}"

exit "$COMMAND_EXIT"

