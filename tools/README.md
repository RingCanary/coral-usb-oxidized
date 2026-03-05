# Tools (condensed)

This folder is intentionally reduced to **10 tools** that cover device testing, trace capture, debugging, and reporting.

## Toolset

1. `edgetpu_delegate_smoke.sh` — quick EdgeTPU delegate smoke test.
2. `test_usb_command_dispatch.sh` — command dispatch sanity checks.
3. `pyusb_parity_harness.py` — compare USB behavior across implementations.
4. `usbmon_capture.sh` — collect usbmon traces.
5. `usb_syscall_trace.sh` — collect syscall-level USB traces with `strace`.
6. `usbmon_bulk_signature.py` — signature extraction from bulk transfers.
7. `usbmon_three_stage_signature.py` — three-stage USB handshake/signature extraction.
8. `libusb_trace_diff.py` — diff two libusb/trace logs.
9. `usb_wedge_diag.sh` — diagnose suspected wedge/stall conditions.
10. `usb_toolkit.py` — compact reporting helpers:
    - `summarize-usbmon`
    - `summarize-strace`
    - `report`

## Typical workflow

```bash
# 1) Run basic health tests
./edgetpu_delegate_smoke.sh
./test_usb_command_dispatch.sh

# 2) Capture traces during a repro
./usbmon_capture.sh
./usb_syscall_trace.sh

# 3) Extract signatures / compare traces
python3 usbmon_bulk_signature.py --help
python3 usbmon_three_stage_signature.py --help
python3 libusb_trace_diff.py --help

# 4) Build concise summaries and a report
python3 usb_toolkit.py summarize-usbmon trace.usbmon --out usbmon_summary.md
python3 usb_toolkit.py summarize-strace trace.strace --out strace_summary.md
python3 usb_toolkit.py report usbmon_summary.md strace_summary.md --out report.md
```

## Notes

- Removed redundant generators/pipelines and oversized analysis scripts to reduce entropy.
- Python scripts in this directory are kept under **500 LOC**.
- Use `--help` on each script for argument details.
