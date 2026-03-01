# Instruction Word-Field Analysis (2026-03-01)

## Objective
Move from byte interpolation to structured instruction decoding by analyzing `u16/u32` fields with record-stride grouping.

## Tool
- `tools/instruction_word_field_analysis.py`

Inputs:
- `--entry DIM:serialized_executable_XXX.bin` (repeat)
- fixed family size (same instruction payload length)
- `--stride` (or `auto`)

Outputs:
- changed `u16/u32` lane offsets
- grouped by stride residue (record-relative field positions)
- per-group representative values across dimensions
- best formula fit in tile domain (`tiles = dim/64`)
- `u32` bit-variation mask + small bitfield fit probes

## Key Result: PC Family `2096` (`768, 1536, 2304`)
Artifact:
- `docs/artifacts/instruction-word-field-20260301/pc_2096_768_1536_2304_wordfield.json`
- `docs/artifacts/instruction-word-field-20260301/pc_2096_768_1536_2304_wordfield.stdout.txt`

Confirmed structure:
- Repeating record stride: `64` bytes
- Dominant repeated residues: `28, 30, 32, 34, 38, 40, 42, 56`

Important recovered field:
- `u16` residue `34` with offsets:
  - `290, 354, 418, 482, 546, 610, 674, 738, 802, 866, 930, 994`
- Values:
  - `768 -> 65345 (0xff41)`
  - `1536 -> 64769 (0xfd01)`
  - `2304 -> 63809 (0xf941)`
- Exact best fit:
  - `tile-quadratic`: `y = -4/3 * tiles^2 + 65537`
- Equivalent exact candidate emitted by tool:
  - `tile2div-linear`: `y = 65537 - 4 * floor(tiles^2 / 3)`

This directly explains why per-byte interpolation fails: the encoded quantity is multi-byte and quadratic in tiles; byte-wise low/high interpolation breaks on carry/wrap boundaries.

## EO Family `7952` (`768, 1536, 2304`)
Artifact:
- `docs/artifacts/instruction-word-field-20260301/eo_7952_768_1536_2304_wordfield.json`
- `docs/artifacts/instruction-word-field-20260301/eo_7952_768_1536_2304_wordfield.stdout.txt`

Signals:
- Same stride-64 structuring is visible
- Top repeated residues are mostly `tile-linear` in this 3-point family
- Several residues still resolve as exact `tile-quadratic`

## Practical Implication
Patch synthesis should be word/field based, not byte based:
1. Treat each family independently (same instruction length family)
2. Patch record-relative `u16/u32` fields by formula
3. For packed `u32` words, patch only decoded bit ranges that vary with tiles
4. Serialize patched words back to little-endian bytes

## Next Step
Implement a field-aware patch emitter from this analysis:
- input: family entries (`768,1536,2304`) + target dim
- output: deterministic word-level patch spec (`<payload_len> <offset> <byte>`)
- gate: only apply formulas with exact family fit (`exact_ratio=1.0`)
