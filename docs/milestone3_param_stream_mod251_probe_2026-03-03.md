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
- `tools/patch_tflite_dense_weight_pattern.py`
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

## Conclusion
This probe strongly supports that, for dense `896x896`:
1. parameter payload enters as signed int8 byte semantics,
2. stream formation is deterministic tiled reordering,
3. the concrete `64x64` tile + `c-group-of-4` local layout above exactly explains the compiled parameter stream.

This is a major step toward M4 compilerless synthesis for dense GEMM parameter packing.
