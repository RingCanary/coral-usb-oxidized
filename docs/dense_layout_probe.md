# Dense Layout Probe (Single-Hot Mapping)

Date: 2026-02-21

Goal: recover the compiled parameter byte layout for the single-op
`Dense(256,256)` EdgeTPU template.

## Tooling

- `tools/dense_layout_probe.py`
- `tools/dense_template_matrix_patch.py`
- `tools/tensorizer_patch_edgetpu.py` (inspect region source of truth)
- `examples/inference_dump.rs` (runtime output verification)

## Probe method

Single-hot models were generated and compiled:

- one non-zero weight at `(row,col)`
- all other weights zero

For each compiled model:

1. locate `PARAMETER_CACHING` payload region (`size=65536`)
2. extract payload bytes
3. diff against reference `(0,0)` payload

## Core observations

1. Every single-hot probe changed exactly one payload byte relative to
   background.
2. Background byte was consistently `128`.
3. Active non-zero byte was consistently `255`.
4. Diff against reference `(0,0)` always produced exactly 2 changed offsets:
   - old active offset removed
   - new active offset added

This gives direct `(row,col) -> payload_offset` mapping.

## Recovered mapping (validated)

For `row,col in [0,255]`:

```text
offset =
  (col // 64) * 16384 +
  (row // 64) * 4096 +
  ((row % 64) // 4) * 256 +
  (col % 64) * 4 +
  (row % 4)
```

Equivalent structure:

- 4 column tiles (`64` columns each), tile stride `16384`
- 4 row tiles (`64` rows each), tile stride `4096`
- inside tile: row blocks of 4 have stride `256`
- inside 4x4 block: column stride `4`, row stride `1`

## Validation runs

Artifacts:

- `traces/dense-layout-probe-20260221T121033Z`
- `traces/dense-layout-probe-20260221T121109Z`
- `traces/dense-layout-probe-20260221T121249Z`
- `traces/dense-layout-probe-20260221T121345Z`
- `traces/dense-layout-probe-20260221T121612Z`

The formula matched all probed points from these runs.

## Structured matrix patch verification

Using `tools/dense_template_matrix_patch.py` on
`traces/dense-template-20260221T120206Z/dense_256x256_quant_edgetpu.tflite`:

- `mode=shift_plus1`: output became input rotated by +1 index
- `mode=shift_minus1`: output became input rotated by -1 index

Runtime verification logs:

- `traces/dense-template-20260221T120206Z/inference_dump_shift_plus1.log`
- `traces/dense-template-20260221T120206Z/inference_dump_shift_minus1.log`

This is strong evidence the recovered mapping is correct for structured
`W @ x`-style behavior on this template.

## Quantization byte mapping verification

Added probe tool:

- `tools/dense_quant_value_probe.py`

Artifact:

- `traces/dense-quant-value-probe-20260221T122533Z/value_probe.txt`

Observed for single-hot `(row=0,col=0)` sweeps:

1. TFLite weight tensor is signed `INT8` (`zero_point=0`).
2. Compiled payload byte is exactly quant byte shifted by `+128`.
   - examples:
     - quant `127` -> compiled `255`
     - quant `0` -> compiled `128`
     - quant `-127` (raw byte `129`) -> compiled `1`
3. This matches:
   - `q_i8 = clamp(round(real / scale) + zp, -128, 127)`
   - `payload_u8 = (q_i8 + 128) & 0xff`

Note on single-hot value sweeps:

- each individually-compiled model re-calibrates weight scale, so many tested
  float magnitudes saturated to `q_i8=+/-127`.
- for tensorizer usage, the relevant path is fixed-template encoding:
  use the template's fixed scale/zero-point and write `payload_u8` directly.

Fixed-template byte sweep validation (`value_byte_sweep`):

- using `tools/dense_template_matrix_patch.py` (`mode=single_hot`, `(0,0)`)
- bytes `255,224,192,160,128,96,...` produced output lane-0 values
  `127,96,64,32,0,-32,...` under `inference_dump alt`
- confirms linear interpretation of payload byte around center `128`.
