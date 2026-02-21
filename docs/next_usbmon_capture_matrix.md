# Next Privileged USBMON Capture Matrix

Date: 2026-02-21

Purpose: packet-level validation of model-dependent USB slope classes discovered
from strace (`+6/+10`, `+8/+12`, `~+10/+15`).

## Prerequisites

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
lsusb | rg -i '1a6e:089a|18d1:9302'
```

Confirm Coral bus number and replace `<BUS>` below.

## Capture runs

### 1) MobileNet v2 bird (EdgeTPU)

```bash
sudo ./tools/usbmon_capture.sh -b <BUS> -- bash -lc 'eval "$(./tools/bootstrap_arch_stack.sh print-env)"; cargo run --example inference_benchmark -- models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite 20 5'
```

### 2) Inception v1 (EdgeTPU)

```bash
sudo ./tools/usbmon_capture.sh -b <BUS> -- bash -lc 'eval "$(./tools/bootstrap_arch_stack.sh print-env)"; cargo run --example inference_benchmark -- models/inception_v1_224_quant_edgetpu.tflite 20 0'
```

### 3) Inception v1 plain (control baseline)

```bash
sudo ./tools/usbmon_capture.sh -b <BUS> -- bash -lc 'eval "$(./tools/bootstrap_arch_stack.sh print-env)"; cargo run --example inference_benchmark -- models/inception_v1_224_quant.tflite 20 0'
```

## Immediate analysis

For each captured `usbmon-bus<BUS>-*.log`:

```bash
python3 tools/usbmon_phase_report.py report <log> --bus <BUS>
python3 tools/usbmon_bulk_signature.py <log> --bus <BUS>
python3 tools/usbmon_three_stage_signature.py <log> --bus <BUS>
python3 tools/usbmon_register_map.py report <log> --bus <BUS>
```

## Comparison checks

1. Does bird model keep the same loop shape as MobileNet v1 (`Bo225824 -> Bo150528 -> Bi1008`) or introduce new sizes/signatures?
2. Does Inception show larger per-invoke cycle structure consistent with the steeper strace slope class?
3. Do plain-model runs remain bulk-submit-free at packet level, even when strace shows small jitter in some runs?

## Optional diff commands

```bash
python3 tools/usbmon_phase_report.py diff <mobilenet_v1_log> <bird_log> --bus <BUS>
python3 tools/usbmon_phase_report.py diff <mobilenet_v1_log> <inception_log> --bus <BUS>
```
