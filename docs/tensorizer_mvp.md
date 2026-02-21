# Tensorizer MVP (Model A)

Date: 2026-02-21

Goal: validate that compiled EdgeTPU executables can be reused while swapping
parameter payload bytes in-place inside a compiled `*_edgetpu.tflite`.

## Tooling

- Inspector/patcher: `tools/tensorizer_patch_edgetpu.py`
- Executable parser: `tools/parse_edgetpu_executable.py`
- Runtime harness: `cargo run --example inference_benchmark -- <model> <runs> <warmup>`

## Target model

- `models/mobilenet_v1_1.0_224_quant_edgetpu.tflite`

Package/executable layout (from inspect/parse):

- `exe[0]`: `EXECUTION_ONLY` (`233472` bytes), instruction chunk `225824`,
  parameters `0`
- `exe[1]`: `PARAMETER_CACHING` (`4476928` bytes), instruction chunk `7248`,
  parameters `4464000`

Patch target for MVP:

- `exe[1]` `Executable.parameters` region
- absolute byte range: `[13276, 4477276)` in model file

## Commands

Inspect:

```bash
python3 tools/tensorizer_patch_edgetpu.py inspect \
  models/mobilenet_v1_1.0_224_quant_edgetpu.tflite
```

Patch (zero pattern):

```bash
python3 tools/tensorizer_patch_edgetpu.py patch \
  models/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
  --output /tmp/mobilenet_v1_1.0_224_quant_edgetpu_patched_zero.tflite \
  --overwrite \
  --mode zero \
  --metadata-out /tmp/tensorizer_patch_v1_zero.json
```

Patch (ramp pattern):

```bash
python3 tools/tensorizer_patch_edgetpu.py patch \
  models/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
  --output /tmp/mobilenet_v1_1.0_224_quant_edgetpu_patched_ramp.tflite \
  --overwrite \
  --mode ramp \
  --metadata-out /tmp/tensorizer_patch_v1_ramp.json
```

Patch (xor-ff pattern):

```bash
python3 tools/tensorizer_patch_edgetpu.py patch \
  models/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
  --output /tmp/mobilenet_v1_1.0_224_quant_edgetpu_patched_xorff.tflite \
  --overwrite \
  --mode xor \
  --byte-value 255 \
  --metadata-out /tmp/tensorizer_patch_v1_xorff.json
```

Inference checks:

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"

cargo run --example inference_benchmark -- \
  models/mobilenet_v1_1.0_224_quant_edgetpu.tflite 5 1

cargo run --example inference_benchmark -- \
  /tmp/mobilenet_v1_1.0_224_quant_edgetpu_patched_zero.tflite 5 1

cargo run --example inference_benchmark -- \
  /tmp/mobilenet_v1_1.0_224_quant_edgetpu_patched_ramp.tflite 5 1

cargo run --example inference_benchmark -- \
  /tmp/mobilenet_v1_1.0_224_quant_edgetpu_patched_xorff.tflite 5 1
```

## Observed results

Baseline (`original`):

- top output: `index=905 score=38`
- latency average: `~2.72 ms` (`runs=5`, `warmup=1`)

Patched (`zero`, `ramp`, `xorff`):

- top output: `index=1000 score=0`
- inference still executes successfully with similar latency envelope

Interpretation:

1. Parameter bytes are actively consumed by runtime execution path; modifying
   them changes model output behavior while preserving executable structure.
2. The compiled instruction stream remains runnable after payload mutation,
   supporting the tensorizer premise.
3. For this template/model, naive parameter corruption collapses logits, which
   is expected for arbitrary byte patterns.

## Immediate next step

Dense single-op template is now implemented and validated:

- workflow: `docs/tensorizer_dense_template.md`
- pipeline script: `tools/dense_template_pipeline.sh`

Next progression (for GEMM):

1. Keep the Dense template instruction stream fixed.
2. Replace selected parameter subranges with structured matrix writes (not full
   corruption).
3. Add row/column-targeted expected-value checks to recover exact layout /
   stride transform used by the compiled payload.
