#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'USAGE'
Usage:
  sudo ./tools/usbmon_capture.sh -b <bus> [options] [-- command [args...]]

Capture raw kernel usbmon traffic for one USB bus and write a summary.

Options:
  -b, --bus <bus>          USB bus number (for example 1 from lsusb "Bus 001")
  -o, --out-dir <dir>      Output directory (default: ./traces/usbmon-<timestamp>-bus<bus>)
  -d, --duration <seconds> Capture duration in seconds when no command is provided
  -c, --command <string>   Command string to run while capture is active
  -h, --help               Show this help text

Behavior:
  - When launched with sudo, command execution defaults to the invoking user
    (`$SUDO_USER`) so user-local toolchains and libraries are preserved.
  - Set `USBMON_RUN_COMMAND_AS_ROOT=1` to force running the traced command as root.

Command forms:
  - Preferred: -- command [args...]
  - Alternative: -c "command string"

Examples:
  sudo ./tools/usbmon_capture.sh -b 1 -d 15
  sudo ./tools/usbmon_capture.sh -b 1 -- cargo run --example delegate_usage
  sudo ./tools/usbmon_capture.sh -b 2 -c "cargo run --example tflite_test"
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    die "usbmon capture requires root. Re-run with sudo."
  fi
}

ensure_debugfs_mounted() {
  if grep -qsE '[[:space:]]/sys/kernel/debug[[:space:]]+debugfs[[:space:]]' /proc/mounts; then
    return
  fi

  echo "debugfs not mounted at /sys/kernel/debug; attempting to mount it"
  mount -t debugfs debugfs /sys/kernel/debug \
    || die "failed to mount debugfs at /sys/kernel/debug"
}

ensure_usbmon_available() {
  local usbmon_dir="/sys/kernel/debug/usb/usbmon"
  if [[ -d "$usbmon_dir" ]]; then
    return
  fi

  if command -v modprobe >/dev/null 2>&1; then
    echo "usbmon not available; attempting 'modprobe usbmon'"
    modprobe usbmon || die "failed to load usbmon via modprobe"
  fi

  [[ -d "$usbmon_dir" ]] || die "usbmon directory not found at ${usbmon_dir}. Load usbmon first: modprobe usbmon"
}

validate_uint() {
  local value="$1"
  local label="$2"
  case "$value" in
    ''|*[!0-9]*)
      die "$label must be a non-negative integer"
      ;;
  esac
}

BUS_RAW=""
OUT_DIR=""
DURATION=""
COMMAND_STRING=""
declare -a COMMAND_ARGS=()

while (($# > 0)); do
  case "$1" in
    -b|--bus)
      [[ $# -ge 2 ]] || die "missing value for $1"
      BUS_RAW="$2"
      shift 2
      ;;
    -o|--out-dir)
      [[ $# -ge 2 ]] || die "missing value for $1"
      OUT_DIR="$2"
      shift 2
      ;;
    -d|--duration)
      [[ $# -ge 2 ]] || die "missing value for $1"
      DURATION="$2"
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

[[ -n "$BUS_RAW" ]] || die "bus is required (use -b <bus>)"
validate_uint "$BUS_RAW" "bus"
BUS=$((10#$BUS_RAW))

if [[ -n "$DURATION" ]]; then
  validate_uint "$DURATION" "duration"
fi

if [[ -n "$COMMAND_STRING" && ${#COMMAND_ARGS[@]} -gt 0 ]]; then
  die "use either -c/--command or -- command [args...], not both"
fi

require_root

ensure_debugfs_mounted
ensure_usbmon_available

INVOKING_USER="${SUDO_USER:-}"
INVOKING_HOME=""
RUN_COMMAND_AS_INVOKING_USER=0
if [[ -n "$INVOKING_USER" && "$INVOKING_USER" != "root" && "${USBMON_RUN_COMMAND_AS_ROOT:-0}" != "1" ]]; then
  INVOKING_HOME="$(getent passwd "$INVOKING_USER" 2>/dev/null | awk -F: '{print $6}' || true)"
  if [[ -z "$INVOKING_HOME" ]]; then
    INVOKING_HOME="/home/${INVOKING_USER}"
  fi
  RUN_COMMAND_AS_INVOKING_USER=1
fi

USBMON_DIR="/sys/kernel/debug/usb/usbmon"
MON_FILE="${USBMON_DIR}/${BUS}u"

if [[ ! -d "$USBMON_DIR" ]]; then
  die "usbmon directory not found at ${USBMON_DIR}. Load usbmon first: modprobe usbmon"
fi

if [[ ! -r "$MON_FILE" ]]; then
  AVAILABLE_BUSES="$(ls "$USBMON_DIR" 2>/dev/null | awk '/u$/ {print}' | tr '\n' ' ' || true)"
  die "usbmon bus ${BUS} is not available. Found: ${AVAILABLE_BUSES:-none}"
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="traces/usbmon-${TS}-bus${BUS}"
fi
mkdir -p "$OUT_DIR"

CAPTURE_FILE="${OUT_DIR}/usbmon-bus${BUS}-${TS}.log"
SUMMARY_FILE="${OUT_DIR}/usbmon-bus${BUS}-${TS}.summary.txt"

COMMAND_MODE="none"
COMMAND_DISPLAY="(none)"
if [[ -n "$COMMAND_STRING" ]]; then
  COMMAND_MODE="string"
  COMMAND_DISPLAY="$COMMAND_STRING"
elif ((${#COMMAND_ARGS[@]} > 0)); then
  COMMAND_MODE="argv"
  COMMAND_DISPLAY="$(printf '%q ' "${COMMAND_ARGS[@]}")"
  COMMAND_DISPLAY="${COMMAND_DISPLAY% }"
fi

CAPTURE_PID=""
CAPTURE_STOPPED=0

stop_capture() {
  if [[ "$CAPTURE_STOPPED" -eq 1 ]]; then
    return
  fi
  CAPTURE_STOPPED=1

  if [[ -n "$CAPTURE_PID" ]] && kill -0 "$CAPTURE_PID" 2>/dev/null; then
    kill "$CAPTURE_PID" 2>/dev/null || true
    wait "$CAPTURE_PID" 2>/dev/null || true
  fi
}

cleanup() {
  stop_capture
}

INTERRUPTED=0
on_interrupt() {
  INTERRUPTED=1
}

trap cleanup EXIT
trap on_interrupt INT TERM

START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

cat "$MON_FILE" >"$CAPTURE_FILE" &
CAPTURE_PID=$!

echo "Capturing usbmon bus ${BUS} to ${CAPTURE_FILE}"

COMMAND_EXIT=0
if [[ "$COMMAND_MODE" == "string" ]]; then
  echo "Running command: ${COMMAND_DISPLAY}"
  if [[ "$RUN_COMMAND_AS_INVOKING_USER" -eq 1 ]]; then
    echo "Executing command as invoking user: ${INVOKING_USER}"
  fi
  set +e
  if [[ "$RUN_COMMAND_AS_INVOKING_USER" -eq 1 ]]; then
    sudo -u "$INVOKING_USER" \
      env HOME="$INVOKING_HOME" USER="$INVOKING_USER" LOGNAME="$INVOKING_USER" \
      bash -lc "$COMMAND_STRING"
    COMMAND_EXIT=$?
  else
    bash -lc "$COMMAND_STRING"
    COMMAND_EXIT=$?
  fi
  set -e
elif [[ "$COMMAND_MODE" == "argv" ]]; then
  echo "Running command: ${COMMAND_DISPLAY}"
  if [[ "$RUN_COMMAND_AS_INVOKING_USER" -eq 1 ]]; then
    echo "Executing command as invoking user: ${INVOKING_USER}"
  fi
  set +e
  if [[ "$RUN_COMMAND_AS_INVOKING_USER" -eq 1 ]]; then
    USER_COMMAND="$(printf '%q ' "${COMMAND_ARGS[@]}")"
    USER_COMMAND="${USER_COMMAND% }"
    sudo -u "$INVOKING_USER" \
      env HOME="$INVOKING_HOME" USER="$INVOKING_USER" LOGNAME="$INVOKING_USER" \
      bash -lc "$USER_COMMAND"
    COMMAND_EXIT=$?
  else
    "${COMMAND_ARGS[@]}"
    COMMAND_EXIT=$?
  fi
  set -e
elif [[ -n "$DURATION" ]]; then
  echo "No command provided; capturing for ${DURATION}s"
  sleep "$DURATION"
else
  echo "No command or duration provided; press Ctrl+C to stop capture."
  while [[ "$INTERRUPTED" -eq 0 ]]; do
    sleep 1 || true
  done
fi

stop_capture
END_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"
DURATION_SEC=$((END_EPOCH - START_EPOCH))

TOTAL_LINES="$(wc -l < "$CAPTURE_FILE" | tr -d ' ')"
NEGATIVE_STATUS_COUNT="$(awk '$5 ~ /^-?[0-9]+$/ && $5 < 0 {count++} END {print count+0}' "$CAPTURE_FILE")"

EVENT_COUNTS_FILE="$(mktemp)"
TRANSFER_COUNTS_FILE="$(mktemp)"
DEVICE_COUNTS_FILE="$(mktemp)"

awk '{if (NF >= 3) c[$3]++} END {for (k in c) printf "%s %d\n", k, c[k]}' "$CAPTURE_FILE" | sort >"$EVENT_COUNTS_FILE"
awk '{if (NF >= 4) c[substr($4, 1, 2)]++} END {for (k in c) printf "%s %d\n", k, c[k]}' "$CAPTURE_FILE" | sort >"$TRANSFER_COUNTS_FILE"
awk '{
  if (NF >= 4) {
    split($4, parts, ":")
    if (length(parts) >= 3) {
      dev = parts[3]
      gsub(/[^0-9]/, "", dev)
      if (dev != "") c[dev]++
    }
  }
} END {for (k in c) printf "%s %d\n", k, c[k]}' "$CAPTURE_FILE" | sort -n >"$DEVICE_COUNTS_FILE"

{
  echo "USBMON Capture Summary"
  echo "start_utc=${START_TS}"
  echo "end_utc=${END_TS}"
  echo "duration_seconds=${DURATION_SEC}"
  echo "bus=${BUS}"
  echo "capture_file=${CAPTURE_FILE}"
  echo "command=${COMMAND_DISPLAY}"
  echo "command_exit=${COMMAND_EXIT}"
  echo "total_lines=${TOTAL_LINES}"
  echo "negative_status_lines=${NEGATIVE_STATUS_COUNT}"
  echo
  echo "Event counts (field 3):"
  if [[ -s "$EVENT_COUNTS_FILE" ]]; then
    sed 's/^/  /' "$EVENT_COUNTS_FILE"
  else
    echo "  (none)"
  fi
  echo
  echo "Transfer-direction prefix counts (first 2 chars of field 4):"
  if [[ -s "$TRANSFER_COUNTS_FILE" ]]; then
    sed 's/^/  /' "$TRANSFER_COUNTS_FILE"
  else
    echo "  (none)"
  fi
  echo
  echo "Device id counts (parsed from field 4):"
  if [[ -s "$DEVICE_COUNTS_FILE" ]]; then
    sed 's/^/  /' "$DEVICE_COUNTS_FILE"
  else
    echo "  (none)"
  fi
} >"$SUMMARY_FILE"

rm -f "$EVENT_COUNTS_FILE" "$TRANSFER_COUNTS_FILE" "$DEVICE_COUNTS_FILE"

echo "Summary written to ${SUMMARY_FILE}"
echo "Capture log written to ${CAPTURE_FILE}"

if [[ "$COMMAND_MODE" != "none" ]]; then
  exit "$COMMAND_EXIT"
fi

if [[ "$INTERRUPTED" -eq 1 ]]; then
  exit 130
fi
