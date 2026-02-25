# Next Privileged USBMON Capture Matrix

Date: 2026-02-25

Purpose: collect a single side-by-side capture root containing:
1. pure-`rusb` deterministic descriptor-tag sweep (`rusb_serialized_exec_replay`),
2. known-good `libedgetpu` delegate+inference invoke (`inference_benchmark`).

## Prerequisites

```bash
lsusb | rg -i '1a6e:089a|18d1:9302'
sudo -v
```

Confirm Coral bus number and replace `<BUS>` below.

## One-command capture

```bash
./tools/usbmon_side_by_side_capture.sh --bus <BUS>
```

Defaults:
1. pure-`rusb` model: `templates/dense_2048x2048_quant_edgetpu.tflite`
2. sweep tags: `2,0,1,3,4`
3. known-good model: `templates/dense_2048x2048_quant_edgetpu.tflite`
4. known-good invoke args: `runs=20 warmup=5`

## Common overrides

```bash
./tools/usbmon_side_by_side_capture.sh \
  --bus <BUS> \
  --out-dir traces/usbmon-side-by-side-custom \
  --rusb-model templates/dense_2048x2048_quant_edgetpu.tflite \
  --rusb-tags 2,3,4 \
  --rusb-extra-arg --setup-include-reads \
  --libedgetpu-model templates/dense_2048x2048_quant_edgetpu.tflite \
  --libedgetpu-runs 30 \
  --libedgetpu-warmup 0
```

If the Coral is in boot mode (`1a6e:089a`), include firmware for the pure-`rusb`
lane:

```bash
./tools/usbmon_side_by_side_capture.sh \
  --bus <BUS> \
  --firmware /path/to/apex_latest_single_ep.bin
```

## Expected output tree

```
traces/usbmon-side-by-side-<timestamp>-bus<BUS>/
  README.txt
  capture_manifest.tsv
  pure_rusb_deterministic_sweep/
    tag2/
    tag0/
    tag1/
    tag3/
    tag4/
  libedgetpu_known_good_invoke/
    inference/
```

Each run directory contains:
1. `usbmon-bus<BUS>-<timestamp>.log`
2. `usbmon-bus<BUS>-<timestamp>.summary.txt`
3. `command.raw.txt`
4. `command.capture.txt`

`capture_manifest.tsv` records lane, run id, exit code, command, and resolved
log/summary paths.

## Expected status pattern

1. `pure_rusb_deterministic_sweep/tag2` is expected to show the current
   parameter-stream stall behavior (non-zero exit may occur).
2. `pure_rusb_deterministic_sweep/tag3` and `tag4` typically complete without
   the immediate bulk timeout.
3. `libedgetpu_known_good_invoke/inference` is expected to exit `0` when
   delegate creation and inference are healthy.

## Immediate analysis

```bash
while IFS=$'\t' read -r lane run_id exit_code _ log_path _; do
  [[ "$lane" == "lane" ]] && continue
  [[ -n "$log_path" ]] || continue
  python3 tools/usbmon_phase_report.py report "$log_path" --bus <BUS> > "${log_path}.phase.txt"
  python3 tools/usbmon_bulk_signature.py "$log_path" --bus <BUS> > "${log_path}.bulk.txt"
done < traces/usbmon-side-by-side-<timestamp>-bus<BUS>/capture_manifest.tsv
```
