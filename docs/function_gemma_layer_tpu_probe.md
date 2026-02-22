# Function Gemma Layer TPU Probe

This example patches one real Function-Gemma linear layer into a compiled
EdgeTPU Dense template and runs it on-device.

It validates:

- SafeTensors loading with BF16/F16/F32 support
- Function-Gemma layer mapping (`q/k/v/o/gate/up/down`)
- quantize -> patch -> invoke path on Coral USB
- CPU accumulator vs TPU output affine fit

## Command

```bash
cargo run --example function_gemma_layer_tpu_probe -- \
  <model.safetensors> \
  <template_edgetpu.tflite> \
  <layer_idx> \
  <stage> \
  [runs] \
  [qmax] \
  [clip_percentile]
```

Stages:

- `q`
- `k`
- `v`
- `o`
- `gate`
- `up`
- `down`

Defaults:

- `runs=20`
- `qmax=32`
- `clip_percentile=100`

## Function-Gemma 270M template mapping

For Gemma-3 270M style dimensions (`hidden=640`, `q_out=1024`, `kv_out=256`,
`mlp_hidden=2048`):

- `q` -> template `640x1024`
- `k` / `v` -> template `640x256`
- `o` -> template `1024x640`
- `gate` / `up` -> template `640x2048`
- `down` -> template `2048x640`

Generate each template with `tools/dense_template_pipeline.sh`.

## Notes

- The official `google/functiongemma-270m-it` checkpoint is gated on HF.
- An ungated derivative that matches these tensor dimensions:
  `distil-labs/distil-home-assistant-functiongemma`.
