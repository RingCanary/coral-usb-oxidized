# CLIP ViT Layer TPU Probe

This example patches one real CLIP ViT-B/32 linear layer from a SafeTensors
checkpoint into a compiled EdgeTPU Dense template, executes it on Coral, and
reports a CPU-accumulator vs TPU affine fit.

## Command

```bash
cargo run --example clip_vit_layer_tpu_probe -- \
  <model.safetensors> \
  <template_edgetpu.tflite> \
  <layer_idx> \
  <stage> \
  [runs] \
  [qmax]
```

Where:

- `stage`: `q|k|v|o|fc1|fc2`
- `runs`: default `20`
- `qmax`: default `127`

## Template dimensions

- `q|k|v|o`: template must be `768x768`
- `fc1`: template must be `768x3072`
- `fc2`: template must be `3072x768`

## Output

The probe prints:

- per-run latency (`avg_ms`, `total_ms`)
- input/output preview
- affine fit statistics between CPU int32 accumulator reference and TPU output:
  - `alpha`, `beta`
  - `corr`
  - `mae`, `rmse`

This gives a quick signal-integrity check before integrating full CLIP forward
execution.
