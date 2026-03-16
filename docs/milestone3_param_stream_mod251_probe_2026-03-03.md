# Milestone 3 Probe: Non-uniform `i % 251` weight pattern (2026-03-03)

## Objective
Run the recommended maximally-diagnostic non-uniform test:
- set quantized weight byte `i` to `i % 251` (prime modulus),
- compile at `896x896`,
- inspect PARAMETER_CACHING stream,
- test permutation hypothesis and extract mapping evidence.

## Artifacts
Run root:
- `traces/analysis/m3-param-permutation-20260303T145503Z/`

Key files:
- generated base quant model:
  - `dense_896x896_base_quant.tflite`
- patched quant model (`i % 251` pattern):
  - `dense_896x896_mod251_quant.tflite`
  - metadata: `weight_patch.metadata.json`
- compiled model:
  - `dense_896x896_mod251_quant_edgetpu.tflite`
- permutation reports:
  - `param_permutation.report.json` (raw expectation)
  - `param_permutation.signed_reinterpret.report.json` (signed reinterpret expectation)
- deterministic mapping witness:
  - `param_map_witness.signed_reinterpret.u32le.bin`
- DUT run log:
  - `dut_mod251_896.log`

## Tooling added for this probe
- `tools/archive/patch_tflite_dense_weight_pattern.py`
  - patches Dense quantized weight bytes in-place using TFLite schema table offsets.
- `src/bin/param_stream_permutation_probe.rs`
  - extracts parameter stream,
  - checks multiset permutation vs expected byte pattern,
  - emits mapping witness,
  - evaluates candidate tile formulas.

## Results

### 1) Raw `i % 251` bytes are **not** a direct permutation
From `param_permutation.report.json`:
- `is_permutation=false`
- dominant histogram deltas:
  - missing bytes `123..127`
  - extra bytes `251..255`

Interpretation:
- compiler/runtime path effectively reinterprets quantized values as signed int8 bytes.

### 2) Signed reinterpret expectation **is** a permutation
Using expected mode `signed_reinterpret`:
- expected byte formula: `((i % 251) - 128) mod 256`
- `is_permutation=true`
- `histogram_l1_distance=0`

This confirms the stream is a pure reordering (no byte creation/loss) once int8 reinterpretation is modeled correctly.

### 3) Layout formula match is exact (0 mismatches)
From `formula_checks` in `param_permutation.signed_reinterpret.report.json`:
- `tile64_rowmajor_tiles_local_cr4`: `mismatch_count=0`
- `tile64_colmajor_tiles_local_rc4`: high mismatch (`799556`)

For source row-major index `src = r*dim + c` (`dim=896`), matching stream offset is:

`off = (r/64)*(dim/64*4096) + (c/64)*4096 + ((c%64)/4)*256 + (r%64)*4 + (c%4)`

This is an exact byte-for-byte match for the full parameter stream in this probe.

### 4) DUT confirmation
Pi5 + Coral run with compiled mod251 model:
- log: `dut_mod251_896.log`
- result: PASS
- output hash: `0x8d7854bd1eb9c1e2`

### 5) Direct 1792x1792 confirmation
Additional run root:
- `traces/analysis/m3-param-permutation-1792-20260303T151750Z/`

Result (`param_permutation.signed_reinterpret.report.json`):
- `is_permutation=true`
- `tile64_rowmajor_tiles_local_cr4`: `mismatch_count=0`
- control `tile64_colmajor_tiles_local_rc4`: high mismatch (`3198440`)

This directly confirms the same formula at `1792x1792` (not only via prefix inference).

DUT run on compiled `1792x1792` mod251 model:
- log: `traces/analysis/m3-param-permutation-1792-20260303T151750Z/dut_mod251_1792.log`
- result: PASS
- output hash: `0x394aa8758535e7e9`

### 6) Rectangular check (`896x1792` model)
Additional run root:
- `traces/analysis/m3-param-permutation-rect-20260303T151814Z/`

Notes:
- TensorFlow export stores Dense weight tensor with shape `[1792, 896]` for this model.
- Patch/probe operates on flattened tensor-buffer order and then validates against compiled stream.

Rectangular formula check (`rect_formula_check.json`):
- selected tensor shape formula mismatch: `0`
- swapped-shape control mismatch: `1580544`

Using rows/cols from the stored weight tensor shape (`rows=1792`, `cols=896`), the same packing form holds:

`off = (r/64)*(cols/64*4096) + (c/64)*4096 + ((c%64)/4)*256 + (r%64)*4 + (c%4)`

DUT run on compiled rectangular model:
- log: `dut_mod251_896x1792.log`
- result: PASS
- output hash: `0xe0f607a60893b844`

## Conclusion
This probe now supports, across square and rectangular dense cases tested:
1. parameter payload enters as signed int8 byte semantics,
2. stream formation is deterministic tiled reordering,
3. the concrete `64x64` tile + `c-group-of-4` local layout exactly explains compiled parameter streams when applied over the stored weight tensor shape.

This is a major step toward M4 compilerless synthesis for dense GEMM parameter packing.
