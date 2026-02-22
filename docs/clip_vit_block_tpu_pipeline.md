# CLIP ViT Block TPU Pipeline

This example wires a full CLIP ViT-B/32 encoder layer linear sequence on Coral
using six stage weights from SafeTensors:

- `q`
- `k`
- `v`
- `o`
- `fc1`
- `fc2`

For each stage, it quantizes `f32` weights to signed `int8`, patches the
matching compiled Dense template, prepares a TPU interpreter, executes batched
row-major inputs, and reports:

- stage setup metadata (`tensor`, `dims`, quantization stats)
- per-stage execution timing
- affine fit (`alpha`, `beta`, `corr`, `mae`, `rmse`) between CPU int32
  accumulators and TPU `int8` outputs

## Command

```bash
cargo run --example clip_vit_block_tpu_pipeline -- \
  <model.safetensors> \
  <template_768x768.tflite> \
  <template_768x3072.tflite> \
  <template_3072x768.tflite> \
  [layer_idx] [rows] [runs] [warmup] [qmax] \
  [--clip-percentile P] [--auto-qmax A,B,C] \
  [--input-q PATH] [--seed N]
```

Defaults:

- `layer_idx=0`
- `rows=8`
- `runs=3`
- `warmup=1`
- `qmax=24`
- `clip_percentile=100`
- synthetic input rows when `--input-q` is omitted

Recommended for CLIP checkpoints:

- use `qmax` in the `20..32` range (defaults to `24`) to reduce accumulator
  saturation
- use `--auto-qmax 16,20,24,32,48,64` to pick per-stage `qmax` based on
  calibration correlation/RMSE on the current stage input

## Template mapping

- `q/k/v/o`: `768x768` template
- `fc1`: `768x3072` template
- `fc2`: `3072x768` template
