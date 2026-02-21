# EdgeTPU Executable vs USB Transport Correlation

Date: 2026-02-21

This note correlates extracted serialized executables with packet-level USB loop
signatures and moves from marker-only scans to schema-aware parsing.

## Inputs

Models:

- `models/mobilenet_v1_1.0_224_quant_edgetpu.tflite` (model A)
- `models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite` (model B)
- `models/inception_v1_224_quant_edgetpu.tflite` (model C)

Extraction outputs:

- `/tmp/edgetpu_extract_v1/package_000/*`
- `/tmp/edgetpu_extract_bird/package_000/*`
- `/tmp/edgetpu_extract_inception/package_000/*`

USB packet-level references:

- `traces/re-matrix-20260221T092342Z/U4_*` (model A)
- `traces/re-matrix-20260221T092342Z/U5_*` (model B)
- `traces/re-matrix-20260221T092342Z/U6_*` (model C)

Schema-aware parser outputs:

- `traces/re-matrix-20260221T092342Z/EXEC_PARSE_MOBILENET_V1.{txt,json}`
- `traces/re-matrix-20260221T092342Z/EXEC_PARSE_BIRD_V2.{txt,json}`
- `traces/re-matrix-20260221T092342Z/EXEC_PARSE_INCEPTION_V1.{txt,json}`

Schema source (vendored):

- `docs/schema/libedgetpu_executable.fbs`

## New parser capability

`tools/parse_edgetpu_executable.py` decodes:

1. Executable type split (`EXECUTION_ONLY` vs `PARAMETER_CACHING`)
2. Instruction chunk sizes (`instruction_bitstreams[].bitstream`)
3. Relocation table (`field_offsets[].meta.{desc,name,position}`)
4. Parameter payload size (`parameters`)
5. Input/output layer metadata (`Layer`, `TensorShape`, numerics)
6. DMA hint structure (`dma_hints`)

## Cross-model executable matrix

### Model A (`mobilenet_v1_1.0_224_quant_edgetpu`)

- `serialized_executable_000.bin`:
  - type: `EXECUTION_ONLY`
  - instruction chunks: `[225824]`
  - parameters: `0`
  - input/output dims: `224x224x3` -> `1001`
- `serialized_executable_001.bin`:
  - type: `PARAMETER_CACHING`
  - instruction chunks: `[7248]`
  - parameters: `4464000`

USB correlation:

- loop stage: `Bo 225824` (matches exec0 chunk size)
- preload stage: `Bo 7248` (matches exec1 chunk size)

### Model B (`mobilenet_v2_1.0_224_inat_bird_quant_edgetpu`)

- `serialized_executable_000.bin`:
  - type: `EXECUTION_ONLY`
  - instruction chunks: `[261920, 10224]`
  - parameters: `0`
  - input/output dims: `224x224x3` -> `965`
- `serialized_executable_001.bin`:
  - type: `PARAMETER_CACHING`
  - instruction chunks: `[10064]`
  - parameters: `3947392`

USB correlation:

- loop stages: `Bo 261920` and `Bo 10224` (match exec0 chunk sizes)
- preload stage: `Bo 10064` (matches exec1 chunk size)

### Model C (`inception_v1_224_quant_edgetpu`)

- `serialized_executable_000.bin`:
  - type: `EXECUTION_ONLY`
  - instruction chunks: `[254656, 103200]`
  - parameters: `393664`
  - input/output dims: `224x224x3` -> `1001`
- `serialized_executable_001.bin`:
  - type: `PARAMETER_CACHING`
  - instruction chunks: `[9680]`
  - parameters: `6581440`

USB correlation:

- loop stages observed in packet capture:
  - explicit anchor path: `Bo 254656 -> Bo 150528 -> Bo 393664 -> Bi 1008`
  - dedicated 3-stage tail parser: `Bo 150528 -> Bo 393664 -> Bo 103200 -> Bi 1008`
- correlation:
  - `254656` matches exec0 instruction chunk 0
  - `393664` matches exec0 parameters payload size
  - `103200` matches exec0 instruction chunk 1
  - `150528` remains host input activation transfer (common across models)

This resolves the previously ambiguous U6 capture and explains why the model C
loop has extra per-invoke transport work.

## Relocation metadata findings

For all three models, `EXECUTION_ONLY` executable relocations include:

- `BASE_ADDRESS_PARAMETER`
- `BASE_ADDRESS_SCRATCH`
- `BASE_ADDRESS_INPUT_ACTIVATION`
- `BASE_ADDRESS_OUTPUT_ACTIVATION`

Pattern:

- 2 relocations per base (lower/upper 32-bit) in these captures
- input/output relocations carry layer names (for example `input`, `prediction`,
  `InceptionV1/Logits/Predictions/Softmax`)

For all `PARAMETER_CACHING` executables in this set:

- relocations are `BASE_ADDRESS_PARAMETER` only

## MultiExecutable size consistency

For each extracted package:

- `serialized_multi_executable.bin` declared executable sizes match extracted
  `serialized_executable_*.bin` sizes exactly.
- This confirms extractor integrity and supports byte-accurate correlation to
  usbmon payload lengths.

## Interpretation

1. Transport headers/payload sizes are compiled artifact structure, not
   runtime-generated magic.
2. The `exec0`/`exec1` split consistently maps to:
   - `exec0`: inference-time instruction path (plus model-dependent extras)
   - `exec1`: parameter-caching/preload path
3. Model C introduces mixed `EXECUTION_ONLY` content (instructions + parameter
   payload), which explains its steeper per-invoke USB slope class.
4. The parser now provides a schema-backed basis for a tensorizer pipeline.

## Reproduction commands

```bash
# Extract model packages
python3 tools/extract_edgetpu_package.py extract models/mobilenet_v1_1.0_224_quant_edgetpu.tflite --out /tmp/edgetpu_extract_v1 --overwrite
python3 tools/extract_edgetpu_package.py extract models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --out /tmp/edgetpu_extract_bird --overwrite
python3 tools/extract_edgetpu_package.py extract models/inception_v1_224_quant_edgetpu.tflite --out /tmp/edgetpu_extract_inception --overwrite

# Parse executables
python3 tools/parse_edgetpu_executable.py /tmp/edgetpu_extract_v1/package_000
python3 tools/parse_edgetpu_executable.py /tmp/edgetpu_extract_bird/package_000
python3 tools/parse_edgetpu_executable.py /tmp/edgetpu_extract_inception/package_000
```
