# USB Invoke Scaling by Model (strace)

Date: 2026-02-21

Scope: `tools/usb_syscall_trace.sh` summaries for `examples/inference_benchmark`.

## Models

1. `mobilenet_v1_1.0_224_quant_edgetpu.tflite` (output `1001`)
2. `mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite` (output `965`)
3. plain quantized variants (non-EdgeTPU-compiled)

## Observed counts

### EdgeTPU model A (`mobilenet_v1_1.0_224_quant_edgetpu.tflite`)

| total invokes (warmup + measured) | SUBMITURB | REAPURBNDELAY |
|---:|---:|---:|
| 1 | 120 | 225 |
| 6 | 150 | 275 |
| 10 | 174 | 315 |
| 25 | 264 | 465 |
| 35 | 324 | 565 |

Exact fit in this run set:

- `SUBMITURB = 114 + 6 * invokes`
- `REAPURBNDELAY = 215 + 10 * invokes`

### EdgeTPU model B (`mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite`)

| total invokes (warmup + measured) | SUBMITURB | REAPURBNDELAY |
|---:|---:|---:|
| 1 | 121 | 225 |
| 6 | 161 | 285 |
| 12 | 211 | 361 |
| 25 | 315 | 517 |

Approximate fit (low-jitter, model-dependent slope):

- `SUBMITURB ≈ 113 + 8 * invokes`
- `REAPURBNDELAY ≈ 213 + 12 * invokes`

Notes:

- The 6-invoke point matches the approximate fit exactly.
- 12 and 25 invoke points show a small positive offset (`+2` submit, `+4` reap)
  relative to that fit, likely from polling/reap timing effects.

### Plain quantized model path

Across plain-model runs (`mobilenet_v1_*_quant.tflite` and
`mobilenet_v2_*_inat_bird_quant.tflite`):

- `SUBMITURB=105`
- `REAPURBNDELAY=201`

These counts remained effectively constant despite large invoke-count changes.

## Interpretation

1. USB syscall growth is strongly model-dependent for EdgeTPU-compiled graphs.
2. The plain model path stays near fixed setup-level USB activity in this
   environment.
3. The difference between model A and model B slopes suggests different
   per-invoke transport choreography (for example different numbers of URB
   submissions/reaps per inference cycle).

## Next packet-level step

To turn this from syscall-level to protocol-level certainty, capture usbmon for
model B and compare loop signatures against model A:

```bash
sudo ./tools/usbmon_capture.sh -b 4 -- bash -lc 'eval "$(./tools/bootstrap_arch_stack.sh print-env)"; cargo run --example inference_benchmark -- models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite 20 5'
```

Then analyze with:

```bash
python3 tools/usbmon_phase_report.py report <usbmon.log> --bus 4
python3 tools/usbmon_bulk_signature.py <usbmon.log> --bus 4
python3 tools/usbmon_register_map.py report <usbmon.log> --bus 4
```
