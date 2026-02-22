# CLIP ViT Full Forward (Coral Hybrid)

This example runs a full CLIP ViT vision path with a hybrid execution model:

- CPU: patch embedding conv, layer norms, softmax attention, QuickGELU, residuals
- Coral: all linear layers (`q/k/v/o/fc1/fc2`) via patched Dense templates

The final output is a `512`-dim projected embedding.

## Command

```bash
cargo run --example clip_vit_full_forward -- \
  <model.safetensors> \
  <template_768x768.tflite> \
  <template_768x3072.tflite> \
  <template_3072x768.tflite> \
  [--image-f32le PATH] \
  [--out-f32le PATH] \
  [--out-norm-f32le PATH] \
  [--reference-f32le PATH] \
  [--weight-qmax N] [--act-qmax N] \
  [--clip-percentile P] \
  [--calibration-rows N] \
  [--max-layers N] \
  [--seed N]
```

Defaults:

- `weight-qmax=32`
- `act-qmax=32`
- `clip-percentile=100`
- `calibration-rows=8`
- `max-layers=12`
- synthetic image if `--image-f32le` is omitted

## Input format

`--image-f32le` expects raw `f32` CHW tensor:

- shape: `[3, 224, 224]`
- element count: `150528`
- byte size: `602112`

## HF reference compare

Use `--reference-f32le` with a raw `f32` vector of length `512` (projected image embedding).
The example prints cosine/MAE/RMSE for both raw and normalized embeddings.

For consistent comparisons, ensure both paths use the same input tensor and
same CLIP checkpoint.

Helper script (requires Python deps via `uv`):

```bash
uv run --with numpy --with torch --with transformers \
  tools/clip_hf_reference.py \
  Bingsu/clip-vit-base-patch32-ko \
  /tmp/clip_input.f32le \
  /tmp/clip_ref_embed.f32le
```
