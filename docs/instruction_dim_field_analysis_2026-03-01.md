# Dense Instruction Dimension Field Analysis (2026-03-01)

## Objective
Build a compiler-independence path by identifying instruction-byte fields that track dimension changes, using only serialized executable diffs.

## Inputs
- Historical templates from `traces/dense-template-*`.
- New local templates compiled via Rust toolchain wrapper:
  - `640, 768, 896, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 3072, 3328, 3584, 3840, 4096`.

## Instruction Size Table
Generated file: `docs/artifacts/instruction-dim-field-20260301/dense_instruction_size_table.tsv`

Observed EO/PC instruction families:
- `1024, 2048 -> EO=9872 PC=2608`
- `896, 1792, 2688 -> EO=8976 PC=2352`
- `768, 1536, 2304 -> EO=7952 PC=2096`
- `640, 1280 -> EO=7056 PC=1840`
- `2752, 2816 -> EO=11200 (STAND_ALONE)`
- `3072, 4096 -> EO=11152 (STAND_ALONE)`
- `3328 -> EO=9616 (STAND_ALONE)`
- `3584 -> EO=10128 (STAND_ALONE)`
- `3840 -> EO=10640 (STAND_ALONE)`

This confirms there is no single linear size regime; synthesis should be family-aware.

## Pairwise Diff Campaign
Same-size pairs used for field-stability analysis:
- Pair A: `1024 <-> 2048` (EO 9872 / PC 2608)
- Pair B: `1536 <-> 2304` (EO 7952 / PC 2096)
- Pair C: `1792 <-> 2688` (EO 8976 / PC 2352)

Per-pair changed-byte counts:
- EO: `446`, `389`, `435`
- PC: `154`, `187`, `202`

Stable changed-offset intersection across all three pairs:
- EO intersection: `114` offsets
- PC intersection: `78` offsets

Relocation overlap for these diffs remained `0`, so these are non-relocation instruction fields.

## High-Signal Offsets
From `tools/instruction_dim_field_analysis.py` (monotonic with dimension/tile count):

EO examples:
- `338`: `[31, 47, 55, 63, 71, 83]` (corr ~ `+1.0`)
- `340`: `[192, 160, 144, 128, 112, 88]` (corr ~ `-1.0`)
- `530`: `[12, 16, 18, 20, 22, 26]`
- `677`: `[6, 8, 9, 10, 11, 13]`
- `971`: `[4, 6, 7, 8, 9, 11]`
- `988`: `[252, 250, 249, 248, 247, 245]`

PC examples:
- `187`: `[4, 9, 12, 16, 20, 27]`
- `295`: `[254, 250, 249, 248, 242, 240]`
- `1308`: `[0, 2, 3, 3, 6, 7]`

Full artifacts:
- `docs/artifacts/instruction-dim-field-20260301/eo_report.json`
- `docs/artifacts/instruction-dim-field-20260301/pc_report.json`
- `docs/artifacts/instruction-dim-field-20260301/eo_stdout.txt`
- `docs/artifacts/instruction-dim-field-20260301/pc_stdout.txt`
- `docs/artifacts/instruction-dim-field-20260301/eo_crossfamily_report.json`
- `docs/artifacts/instruction-dim-field-20260301/pc_crossfamily_report.json`
- `docs/artifacts/instruction-dim-field-20260301/eo_crossfamily_stdout.txt`
- `docs/artifacts/instruction-dim-field-20260301/pc_crossfamily_stdout.txt`

Cross-family (10-dim) high-signal result:
- EO `off=338` scales almost perfectly with dimension:
  - `640:19, 768:23, 896:27, 1024:31, 1280:39, 1536:47, 1792:55, 2048:63, 2304:71, 2688:83`
- EO `off=340` is the mirrored complement:
  - `640:216, 768:208, 896:200, 1024:192, 1280:176, 1536:160, 1792:144, 2048:128, 2304:112, 2688:88`

## Interpretation
- Dimension-sensitive encoding is distributed across many bytes, not a tiny single patch list.
- A stable core exists (EO 114 + PC 78) across independent families.
- Family-specific residual fields remain and must be layered on top of the stable core.

## Tooling Added
`tools/instruction_dim_field_analysis.py`

It computes:
- pairwise changed sets,
- intersection/union offsets,
- relocation overlap,
- per-offset trajectories across dimensions,
- correlation with dimension and tile count.

## Reproduction
EO run:

```bash
tools/instruction_dim_field_analysis.py \
  --entry 1024:traces/dense-template-1024x1024-20260222T062017Z/extract/package_000/serialized_executable_000.bin \
  --entry 1536:/tmp/dense-1536/extract/package_000/serialized_executable_000.bin \
  --entry 1792:/tmp/dense-1792/extract/package_000/serialized_executable_000.bin \
  --entry 2048:traces/dense-template-2048x2048-20260222T062027Z/extract/package_000/serialized_executable_000.bin \
  --entry 2304:traces/dense-template-2304x2304-20260222T062229Z/extract/package_000/serialized_executable_000.bin \
  --entry 2688:traces/dense-template-2688x2688-20260222T062240Z/extract/package_000/serialized_executable_000.bin \
  --pair 1024:2048 --pair 1536:2304 --pair 1792:2688 \
  --json-out docs/artifacts/instruction-dim-field-20260301/eo_report.json
```

PC run is identical but uses `serialized_executable_001.bin` paths.

## Next Step (synthesis path)
1. Treat each instruction-size family as a separate template family.
2. For a target family, apply stable-core offset transforms first.
3. Fit residual offsets with additional family-specific rules.
4. Validate by reconstructing a held-out known dimension before attempting a novel dimension.

Current best synthesis candidates:
- EO/PC families with at least two known endpoints: `9872/2608`, `7952/2096`, `8976/2352`.
- STAND_ALONE family exploration next: validate whether `3328/3584/3840` can be generated from one another by offset patching before targeting unseen dims.

## DUT Validation (Pi5 + Coral)
Prototype synth tool:
- `tools/synthesize_instruction_patch_spec.py`

Tested family:
- Train endpoints: `768` and `2304` (both `EO=7952`, `PC=2096`)
- Target: `1536`
- Synthesized patch rules: `191` total (`EO=94`, `PC=97`)

Replay results (`--bootstrap-known-good-order`):
- Baseline target model (`1536`) passes and returns output hash `0xdc8c52f84cb2e9c0`.
- Full synth patch fails at class-2 payload start (`tag=2` timeout at offset `0`).
- `PC-only` synth patch: same class-2 timeout at offset `0`.
- `EO-only` synth patch: class-2 preload succeeds and completion event arrives, but output read times out.

Implication:
- PARAMETER_CACHING instruction bytes are admission-critical for class-2 ingestion.
- EXECUTION_ONLY instruction bytes are execution/output-critical after preload.

DUT artifacts:
- `docs/artifacts/instruction-synthesis-20260301/pi5_dut_summary.md`
- `docs/artifacts/instruction-synthesis-20260301/pi5-logs/`
