# USB Tracing Toolkit

Practical tracing helpers for Coral USB traffic are available under `tools/`:

- `tools/usbmon_capture.sh`: privileged kernel-level USB packet capture (`usbmon`).
- `tools/usb_syscall_trace.sh`: unprivileged user-space syscall tracing (`strace`) fallback.
- `tools/usbmon_phase_report.py`: phase-oriented analyzer and diff tool for usbmon logs.
- `tools/usbmon_register_map.py`: control-transfer/register access extractor for usbmon logs.
- `tools/usbmon_bulk_signature.py`: bulk payload header/signature extractor by phase.
- `tools/usbmon_three_stage_signature.py`: dedicated parser for repeated `Bo->Bo->Bo->Bi` cycle signatures.
- `tools/strace_usb_scaling.py`: linear-fit summary for `USBDEVFS_SUBMITURB`/`REAPURBNDELAY`.

## 1) Privileged usbmon capture

Use this when you need packet-level USB events for a specific bus.

### Requirements

- Root privileges (`sudo`).
- Linux `debugfs` mounted at `/sys/kernel/debug`.
- `usbmon` support available (usually via `usbmon` kernel module).

Note: `tools/usbmon_capture.sh` will attempt to mount `debugfs` and run
`modprobe usbmon` automatically when needed.
When invoked with `sudo`, it runs the traced command as the invoking user
(`$SUDO_USER`) by default so user-local libraries (for example `~/.local/lib`)
continue to resolve correctly. Set `USBMON_RUN_COMMAND_AS_ROOT=1` to force root.

### Typical flow

1. Find Coral bus with `lsusb`.
2. Run capture on that bus while reproducing behavior.
3. Review generated `.summary.txt` and raw `.log`.

### Examples

Capture bus `1` for 15 seconds:

```bash
sudo ./tools/usbmon_capture.sh -b 1 -d 15
```

Capture while running a command:

```bash
sudo ./tools/usbmon_capture.sh -b 1 -- cargo run --example delegate_usage
```

Alternative command-string form:

```bash
sudo ./tools/usbmon_capture.sh -b 1 -c "cargo run --example tflite_test"
```

If TensorFlow Lite C libs are not installed yet, you can still generate
delegate/USB activity with the minimal smoke tool:

```bash
./tools/edgetpu_delegate_smoke.sh
sudo ./tools/usbmon_capture.sh -b 1 -- ./tools/edgetpu_delegate_smoke.sh
```

### Output

The script writes:

- Raw capture log: `usbmon-bus<bus>-<timestamp>.log`
- Summary: `usbmon-bus<bus>-<timestamp>.summary.txt`

The summary includes event counts, transfer prefix counts, parsed device-id counts,
capture duration, and command exit status.

## 2) Unprivileged syscall fallback (strace)

Use this when root access is unavailable. It traces user-space USB interactions
for a command and summarizes syscall/ioctl activity.

### Requirements

- `strace` installed.
- Ability to trace the target process (typically your own process).

### Examples

Trace a command (argv form):

```bash
./tools/usb_syscall_trace.sh -- cargo run --example verify_device
```

Trace a command string:

```bash
./tools/usb_syscall_trace.sh -c "cargo run --example delegate_usage"
```

Custom output directory:

```bash
./tools/usb_syscall_trace.sh -o traces/my-run -- cargo run --example tflite_test
```

### Output

The script writes:

- Raw trace log: `strace-usb-<timestamp>.log`
- Summary: `strace-usb-<timestamp>.summary.txt`

The summary includes USB-related line counts, related error lines, USB device nodes touched,
USB-related syscall counts, USBDEVFS ioctl counts, runtime duration, and command exit status.

## 3) Phase report and run-to-run diff

Use this to turn raw usbmon logs into reverse-engineering-friendly timelines.

### One-log report

```bash
python3 tools/usbmon_phase_report.py report \
  traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.log \
  --bus 4 --device 005
```

This reports:

- dominant transfer/status patterns
- inactivity-gap segments
- bulk completion size histograms
- inferred per-inference cycle timing using the default pattern:
  - `Bo(225824) -> Bo(150528) -> Bi(1008)`

### Compare two runs

```bash
python3 tools/usbmon_phase_report.py diff \
  traces/run_a.log \
  traces/run_b.log \
  --bus 4 --device 005
```

This highlights deltas in:

- total duration and line count
- cycle count and interval timings
- pre-first-inference bulk upload bytes
- transfer-type and bulk-size distributions

### JSON output

Add `--json` to either subcommand for machine-readable output.

## 4) Control/register map extraction

Use this for vendor control-transfer analysis (`Ci/Co` endpoint 0 setup packets).

### One-log register report

```bash
python3 tools/usbmon_register_map.py report \
  traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.log \
  --bus 4 --device 005
```

Highlights:

- control totals (`standard` vs `vendor`)
- inferred operation classes (`read32`, `write32`, `read64`, `write64`)
- per-address counts with phase tags (`setup_only`, `pre_loop`, `loop`, `post_loop`)

### Multi-run matrix

```bash
python3 tools/usbmon_register_map.py matrix \
  --run U1=traces/run1.log \
  --run U2=traces/run2.log \
  --run U3=traces/run3.log \
  --bus 4 --device 005
```

This emits a consolidated address/op table across runs and a phase breakdown for
the most control-active run.

## 5) Bulk payload signature extraction

Use this to fingerprint repeated bulk submit/completion payload prefixes.

```bash
python3 tools/usbmon_bulk_signature.py \
  traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.log \
  --bus 4 --device 005 --prefix-words 2
```

Highlights:

- submit/complete counts by transfer and size
- top payload signatures (for example recurring 8-byte loop commands)
- phase attribution (`setup_only` / `pre_loop` / `loop` / `post_loop`)

## 6) Dedicated 3-stage signature parser

Use this when the model loop includes at least three major `Bo` completion
stages before `Bi` output completion (for example U5/U6 classes).

```bash
python3 tools/usbmon_three_stage_signature.py \
  traces/usbmon-20260221T103521Z-bus4/usbmon-bus4-20260221T103521Z.log \
  --bus 4 --device 005
```

Optional explicit pattern matching:

```bash
python3 tools/usbmon_three_stage_signature.py \
  traces/usbmon-20260221T103552Z-bus4/usbmon-bus4-20260221T103552Z.log \
  --bus 4 --device 005 \
  --bo-1 254656 --bo-2 150528 --bo-3 393664 --bi-out 1008
```

Highlights:

- auto-discovered top `Bo/Bo/Bo/Bi` candidates
- non-overlapping cycle extraction with timing stats
- per-stage gap counts (useful to detect hidden intermediate completions)
- cycle interval statistics anchored to a selected stage

## 7) Strace ioctl scaling fit

Use this to summarize and fit USBDEVFS ioctl counts vs invoke count across
`usb_syscall_trace.sh` run folders.

```bash
python3 tools/strace_usb_scaling.py
```

Optional filtering:

```bash
python3 tools/strace_usb_scaling.py --include-prefix R1 --include-prefix R2
```

## Known limits

- `usbmon_capture.sh` needs root and available `usbmon`; it does not decode payload semantics.
- Bus-level capture can include traffic from other devices on the same USB bus.
- `usb_syscall_trace.sh` only shows user-space syscall behavior, not full bus packets.
- `usbmon_phase_report.py` infers phases from traffic patterns; it does not decode proprietary EdgeTPU protocol fields.
- `usbmon_register_map.py` infers register semantics from USB setup fields; address meanings remain hypotheses.
- `usbmon_bulk_signature.py` uses prefix signatures only; it does not decode full payload semantics.
- `usbmon_three_stage_signature.py` models repeated completion choreography; it does not decode payload semantics.
- `strace` can miss activity if USB interactions happen in processes you are not tracing.
- `strace` adds overhead and can alter timing-sensitive behavior.
