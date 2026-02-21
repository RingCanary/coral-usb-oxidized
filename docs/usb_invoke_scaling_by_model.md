# USB Invoke Scaling by Model (strace)

Date: 2026-02-21

Scope: `tools/usb_syscall_trace.sh` summaries for `examples/inference_benchmark`.

## Models

1. `mobilenet_v1_1.0_224_quant_edgetpu.tflite` (output `1001`)
2. `mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite` (output `965`)
3. plain quantized variants (non-EdgeTPU-compiled)

## Observed counts

### EdgeTPU model A (`mobilenet_v1_1.0_224_quant_edgetpu.tflite`)

| run | total invokes (warmup + measured) | SUBMITURB | REAPURBNDELAY |
|---|---:|---:|---:|
| `R8` | 1 | 120 | 225 |
| `R12` | 6 | 150 | 275 |
| `R13` | 10 | 174 | 315 |
| `R14` | 25 | 264 | 465 |
| `R9` | 35 | 324 | 565 |

Exact fit in this run set:

- `SUBMITURB = 114 + 6 * invokes`
- `REAPURBNDELAY = 215 + 10 * invokes`

### EdgeTPU model B (`mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite`)

| run | total invokes (warmup + measured) | SUBMITURB | REAPURBNDELAY |
|---|---:|---:|---:|
| `R18` | 1 | 121 | 225 |
| `R24` | 3 | 137 | 249 |
| `R20` | 6 | 161 | 285 |
| `R23` | 10 | 193 | 333 |
| `R25` | 12 | 209 | 357 |
| `R19` | 25 | 315 | 517 |
| `R21` | 35 | 395 | 637 |

Primary fit:

- `SUBMITURB = 113 + 8 * invokes`
- `REAPURBNDELAY = 213 + 12 * invokes`

Notes:

- `R16_infer_edgetpu_bird_10_2` is a repeatable outlier at total invokes `12`:
  - observed `SUBMITURB=211`, `REAPURBNDELAY=361`
  - expected `209` / `357` from the primary fit
- Repeat run `R25_infer_edgetpu_bird_10_2_repeat` returned exactly to fit,
  indicating occasional run-level jitter rather than a different steady-state slope.

Automated fit/report command used:

```bash
python3 tools/strace_usb_scaling.py
```

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
3. The stable slope difference between model A and model B suggests different
   per-invoke transport choreography (`+6/+10` vs `+8/+12`).

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
