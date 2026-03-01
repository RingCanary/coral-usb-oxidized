#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./tools/usb_wedge_diag.sh [--bus N] [--out-dir DIR]

Collects USB/xHCI wedge diagnostics into a timestamped directory.

Options:
  --bus N        USB bus number (default: 4)
  --out-dir DIR  Output directory (default: traces/usb-wedge-diag-<timestamp>-busN)
USAGE
}

BUS=4
OUT_DIR=""

while (($# > 0)); do
  case "$1" in
    --bus)
      shift
      BUS="${1:-}"
      ;;
    --out-dir)
      shift
      OUT_DIR="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

case "$BUS" in
  ''|*[!0-9]*)
    echo "bus must be numeric, got: $BUS" >&2
    exit 1
    ;;
esac

TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="traces/usb-wedge-diag-${TS}-bus${BUS}"
fi
mkdir -p "$OUT_DIR"

exec_cmd() {
  local name="$1"
  shift
  local out="${OUT_DIR}/${name}.txt"
  {
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# $*"
    "$@"
  } >"$out" 2>&1 || true
}

exec_shell() {
  local name="$1"
  local cmd="$2"
  local out="${OUT_DIR}/${name}.txt"
  {
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# $cmd"
    bash -lc "$cmd"
  } >"$out" 2>&1 || true
}

exec_shell "host" "hostname; whoami; uname -a; date -u +%Y-%m-%dT%H:%M:%SZ"
exec_shell "lsusb_all" "lsusb"
exec_shell "lsusb_coral" "lsusb | grep -E '1a6e:089a|18d1:9302' || echo 'no-coral-found'"
exec_shell "sysfs_usb_devices" "ls -la /sys/bus/usb/devices"
exec_shell "sysfs_bus_paths" "for d in /sys/bus/usb/devices/usb* /sys/bus/usb/devices/${BUS}-*; do [ -e \"\$d\" ] || continue; echo \"== \$d\"; readlink -f \"\$d\"; done"
exec_shell "xhci_driver_links" "ls -la /sys/bus/platform/drivers/xhci-hcd"
exec_shell "uhubctl_l${BUS}" "timeout 12s sudo uhubctl -l ${BUS} || timeout 12s uhubctl -l ${BUS} || true"
exec_shell "journal_usb_tail" "sudo journalctl -k -n 400 --no-pager | grep -Ei 'usb|xhci|error|timeout|disconnect|reset|firmware changed' || journalctl -k -n 400 --no-pager | grep -Ei 'usb|xhci|error|timeout|disconnect|reset|firmware changed' || true"
exec_shell "dmesg_usb_tail" "sudo dmesg | tail -n 400 | grep -Ei 'usb|xhci|error|timeout|disconnect|reset|firmware changed' || dmesg | tail -n 400 | grep -Ei 'usb|xhci|error|timeout|disconnect|reset|firmware changed' || true"

printf 'USB wedge diagnostics written to %s\n' "$OUT_DIR"
