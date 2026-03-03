# Milestone 3 DUT Matrix: Controlled Param-Stream Models (Pi5 + Coral USB)

Date: 2026-03-03

## Goal
Validate the new controlled M3 models (`896`/`1792`, `zero`/`ones`) on real DUT transport/execution path.

## Model set
Source artifacts:
- `traces/analysis/m3-param-stream-probe-r2-20260303T143830Z/`
  - `dense_896x896_zero_quant_edgetpu.tflite`
  - `dense_896x896_ones_quant_edgetpu.tflite`
  - `dense_1792x1792_zero_quant_edgetpu.tflite`
  - `dense_1792x1792_ones_quant_edgetpu.tflite`

## DUT run
Run directory:
- `traces/analysis/specv3-m3-dut-model-matrix-20260303T144416Z/`

Replay controls:
- `--firmware /home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin`
- `--bootstrap-known-good-order`
- `--chunk-size 1048576`
- `--reset-before-claim --post-reset-sleep-ms 1200`
- per-model `--input-bytes/--output-bytes` matched dimension

## Results
All 4/4 cases passed end-to-end (`Event: tag=4` + output bytes present):

1. `d896_zero`
   - output bytes: `896`
   - hash: `0xdd9d1c974751a925`
   - output head: all `0x80`

2. `d896_ones`
   - output bytes: `896`
   - hash: `0x3676a5a2d8c1a925`

3. `d1792_zero`
   - output bytes: `1792`
   - hash: `0xaffad9e3b4d52f25`

4. `d1792_ones`
   - output bytes: `1792`
   - hash: `0x4f160d7274622f25`

## Operational note
- First case started from boot-mode device (`1a6e:089a`), firmware upload moved device to runtime (`18d1:9302`), and all subsequent cases executed successfully.
- Pi host reboot was not required during matrix run (`boot_id` unchanged).

## Interpretation
- Controlled M3 models are valid on real hardware transport/execution path.
- `zero` vs `ones` produce distinct deterministic output hashes at both dimensions, consistent with prior param-stream differential findings that these streams differ globally.
