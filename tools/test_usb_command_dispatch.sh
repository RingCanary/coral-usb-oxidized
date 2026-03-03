#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

assert_contains() {
  local file="$1"
  local needle="$2"
  if ! grep -Fq -- "$needle" "$file"; then
    echo "assertion failed: expected '$needle' in $file" >&2
    exit 1
  fi
}

assert_not_contains() {
  local file="$1"
  local needle="$2"
  if grep -Fq -- "$needle" "$file"; then
    echo "assertion failed: did not expect '$needle' in $file" >&2
    exit 1
  fi
}

assert_fails() {
  set +e
  "$@" >/dev/null 2>"$TMP_DIR/fail.stderr"
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    echo "assertion failed: command unexpectedly succeeded: $*" >&2
    exit 1
  fi
}

# --- usb_syscall_trace.sh dispatch tests ---
FAKE_BIN="$TMP_DIR/fakebin"
mkdir -p "$FAKE_BIN"

cat > "$FAKE_BIN/strace" <<'STRACE'
#!/usr/bin/env bash
set -euo pipefail
: "${STRACE_ARGS_LOG:?STRACE_ARGS_LOG must be set}"
printf '%s\n' "$@" > "$STRACE_ARGS_LOG"
out_file=""
args=("$@")
idx=0
while (( idx < ${#args[@]} )); do
  if [[ "${args[$idx]}" == "-o" && $((idx + 1)) -lt ${#args[@]} ]]; then
    out_file="${args[$((idx + 1))]}"
    break
  fi
  ((idx += 1))
done
if [[ -n "$out_file" ]]; then
  mkdir -p "$(dirname "$out_file")"
  : > "$out_file"
fi
while (($# > 0)); do
  if [[ "$1" == "--" ]]; then
    shift
    break
  fi
  shift
done
"$@"
STRACE
chmod +x "$FAKE_BIN/strace"

PATH="$FAKE_BIN:$PATH" STRACE_ARGS_LOG="$TMP_DIR/strace-argv.log" \
  "$REPO_ROOT/tools/usb_syscall_trace.sh" -o "$TMP_DIR/trace-argv" -- /bin/echo argv-mode >/dev/null
assert_contains "$TMP_DIR/strace-argv.log" "/bin/echo"
assert_not_contains "$TMP_DIR/strace-argv.log" "-lc"

assert_fails env PATH="$FAKE_BIN:$PATH" STRACE_ARGS_LOG="$TMP_DIR/strace-denied.log" \
  "$REPO_ROOT/tools/usb_syscall_trace.sh" -o "$TMP_DIR/trace-denied" -c 'echo denied'
assert_contains "$TMP_DIR/fail.stderr" "--allow-shell-command-string"

PATH="$FAKE_BIN:$PATH" STRACE_ARGS_LOG="$TMP_DIR/strace-string.log" \
  "$REPO_ROOT/tools/usb_syscall_trace.sh" -o "$TMP_DIR/trace-string" \
  --allow-shell-command-string -c 'echo allowed' >/dev/null
assert_contains "$TMP_DIR/strace-string.log" "bash"
assert_contains "$TMP_DIR/strace-string.log" "-lc"

# --- usbmon_capture.sh dispatch tests ---
mkdir -p "$TMP_DIR/usbmon"
cat > "$TMP_DIR/usbmon/1u" <<'MON'
ffff S Bi:001:001:1 -115 8 = 80 06 00 01 00 00 12 00
ffff C Bi:001:001:1 0 8 = 12 01 00 02 00 00 00 40
MON

BASH_CALL_LOG="$TMP_DIR/bash-calls.log"
: > "$BASH_CALL_LOG"

cat > "$FAKE_BIN/bash" <<'BASHWRAP'
#!/bin/bash
set -euo pipefail
: "${USB_TEST_BASH_LOG:?USB_TEST_BASH_LOG must be set}"
printf '%s\n' "$*" >> "$USB_TEST_BASH_LOG"
exec /bin/bash "$@"
BASHWRAP
chmod +x "$FAKE_BIN/bash"

PATH="$FAKE_BIN:$PATH" USB_TEST_BASH_LOG="$BASH_CALL_LOG" \
  USBMON_SKIP_DEBUGFS_CHECK=1 USBMON_BASE_DIR="$TMP_DIR/usbmon" USBMON_RUN_COMMAND_AS_ROOT=1 \
  /bin/bash "$REPO_ROOT/tools/usbmon_capture.sh" -b 1 -o "$TMP_DIR/capture-argv" -- /bin/echo usbmon-argv >/dev/null
assert_not_contains "$BASH_CALL_LOG" "-lc"

assert_fails env PATH="$FAKE_BIN:$PATH" USB_TEST_BASH_LOG="$BASH_CALL_LOG" \
  USBMON_SKIP_DEBUGFS_CHECK=1 USBMON_BASE_DIR="$TMP_DIR/usbmon" USBMON_RUN_COMMAND_AS_ROOT=1 \
  /bin/bash "$REPO_ROOT/tools/usbmon_capture.sh" -b 1 -o "$TMP_DIR/capture-denied" -c 'echo denied'
assert_contains "$TMP_DIR/fail.stderr" "--allow-shell-command-string"

PATH="$FAKE_BIN:$PATH" USB_TEST_BASH_LOG="$BASH_CALL_LOG" \
  USBMON_SKIP_DEBUGFS_CHECK=1 USBMON_BASE_DIR="$TMP_DIR/usbmon" USBMON_RUN_COMMAND_AS_ROOT=1 \
  /bin/bash "$REPO_ROOT/tools/usbmon_capture.sh" -b 1 -o "$TMP_DIR/capture-string" \
  --allow-shell-command-string -c 'echo usbmon-string' >/dev/null
assert_contains "$BASH_CALL_LOG" "-lc echo usbmon-string"

echo "All command dispatch tests passed"
