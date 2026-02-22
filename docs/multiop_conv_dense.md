# Multi-op Conv2D + Dense Template Path

Date: 2026-02-22

This workflow adds a combined operator graph path for EdgeTPU reverse engineering:

- `CONV_2D` stage (spatial/channel mixing)
- `FULLY_CONNECTED` stage (projection head)

The model shape is intentionally small and deterministic for iterative RE.

## Generator

- `tools/generate_dense_conv_quant_tflite.py`

Default model:

- Input: `1x16x16x16`
- Conv2D: `64` filters, `1x1`, stride `1`, padding `same`
- GlobalAveragePooling2D
- Dense: `256` units

This default `256` projection aligns with common embedding/projection widths used
in lightweight token pipelines (useful when experimenting with microgpt-adjacent
workloads).

## Pipeline

- `tools/multiop_template_pipeline.sh`

One-command flow:

1. generate quantized model (`uv`, `tensorflow-cpu==2.10.1`)
2. compile with `edgetpu_compiler`
3. extract DWN1 package
4. parse executables (`parse_edgetpu_executable.py`)
5. inspect parameter regions (`tensorizer_patch_edgetpu.py`)
6. optional benchmark (`inference_benchmark`)

Example:

```bash
./tools/multiop_template_pipeline.sh --run-benchmark
```

Alternative larger spatial sweep:

```bash
./tools/multiop_template_pipeline.sh \
  --height 32 --width 32 --in-channels 8 \
  --conv-filters 64 --dense-units 256 --run-benchmark
```

## Why this matters

- Bridges Dense-only and Conv-only RE tracks into one graph.
- Allows observing how parameter-caching and executable partitioning behave for
  mixed-operator models.
- Provides a stepping stone toward microgpt-style accelerator integration where
  projection/mixing layers need to be chained, not run in isolation.

## Next RE tasks

1. Extract per-op parameter boundaries from mixed-model executable payloads.
2. Add structured patch modes for both Conv and Dense segments in one model.
3. Compare USB submit/reap scaling for mixed-operator invokes vs single-op templates.
