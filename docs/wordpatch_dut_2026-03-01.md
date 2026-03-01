# Word-Patch DUT Validation (Pi5 + Coral) — 2026-03-01

## Goal
Validate a new word/bitfield-based patch path against the dense `1536x1536` DUT model.

## New Tooling
- `tools/emit_word_field_patch_spec.py`
  - Input: word-field analysis JSON (`per_offset_fits`)
  - Output: replay patch spec (`<payload_len> <offset> <byte>`)
  - Modes:
    - `best`: evaluate fitted params directly
    - `endpoint`: fit from low/high endpoints only using selected model family

## Build Inputs (family: EO=7952, PC=2096)
- Dims used for field analysis: `768`, `1536`, `2304`
- Target replay model: `/tmp/dense-1536/dense_1536x1536_quant_edgetpu.tflite`

## Emitted Endpoint Patchset
- Combined patch spec: `traces/analysis/wordpatch-dut-20260301/combined_endpoint.patchspec`
- Rule count: `210`
  - PC (`2096`): `114` changed bytes
  - EO (`7952`): `96` changed bytes
- Ground-truth mismatch vs real 1536 executables (byte-level):
  - PC mismatch: `114`
  - EO mismatch: `96`

## DUT Results (fresh-boot isolation)

### Baseline (no patch)
- PASS
- Event received
- Output hash: `0xdc8c52f84cb2e9c0`

### Combined endpoint patch (PC+EO)
- FAIL during parameter preload
- Error:
  - `descriptor tag=2 payload write failed at offset 1048576 ... Operation timed out`

### PC-only endpoint patch (fresh boot)
- FAIL during parameter preload
- Same 1MiB failure:
  - `descriptor tag=2 payload write failed at offset 1048576 ... Operation timed out`

### EO-only endpoint patch (fresh boot)
- Partial progress
- Completion event observed (`Event: tag=4 ...`)
- Final failure:
  - `UsbError(Timeout)` on output read

## Interpretation
The split behavior is stable and high-signal:
1. **PC field errors are admission-critical** and can wedge class-2 streaming at a deterministic deeper point (`1MiB`), not the old `49KiB` wall.
2. **EO field errors are execution/output-critical**; pipeline progresses through event signaling but output transfer semantics break.

This mirrors the prior byte-level split finding but with a stronger localization at the word/field synthesis layer.

## Current Blocker
Two-point (`768`,`2304`) endpoint-only prediction is still underconstrained for several PC/EO words due modular/bit-packed semantics. Additional constraints are needed before endpoint-only synthesis can produce DUT-safe binaries.

## Next Step
Add explicit field-spec constraints per residue/bit-range (signedness, modulo domain, preferred `tile2/div` form), then re-emit endpoint patches and re-run the same fresh-boot split matrix.
