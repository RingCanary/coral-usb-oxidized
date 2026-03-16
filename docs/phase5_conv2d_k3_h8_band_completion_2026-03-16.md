# Phase 5 Conv2D `k=3` `H=8` band completion (2026-03-16)

## Goal
Close Phase 5 after the same-product `6496` family model failed.

The final completion target is:

- single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- symmetric regimes `p32/p64/p128`
- mixed-product `fixed_height=8` family
- pure-`rusb` replay on Pi
- native params + native EO in the active loop

Completion bar:

> exact DUT output-hash equality against the compiled target baseline for the bounded frozen family, using a native EO emitter driven by field rules plus bounded lookup residue.

## What changed

### Discovery result that changed the family model
The decisive discovery runs are:

- `traces/analysis/phase4-conv2d-k3-family-scout-20260316T091008Z/`
- `traces/analysis/phase4-conv2d-k3-family-scout-20260316T091108Z/`
- `traces/analysis/phase4-conv2d-k3-family-scout-20260316T091251Z/`

They establish:

- same-product remains unsalvageable even under bounded channel asymmetry
- the symmetric `H=8` band is stable across `p32/p64/p128`
- tested widths `88..168` all stay in the same `EO=6496 / PC=688` family per regime

That means the real blocker was the old same-product schema, not the absence of a reusable `6496` region.

### New active-family assets
- `templates/phase5_conv2d_k3_h8_band_6496/family.json`
- `templates/phase5_conv2d_k3_h8_band_6496/SUMMARY.txt`
- frozen anchor/target seed models and serialized executables under that family root

The first bounded Phase 5 family is:

- fixed height `8`
- widths `104, 116, 128, 140, 152`
- anchor `8x128`
- regimes `p32/p64/p128`

### New emitter/runtime behavior
- `src/bin/conv_k3_eo_emit.rs`
- `scripts/phase4_conv2d_k3_completion_demo.sh`
- `scripts/phase5_conv2d_k3_freeze_band_assets.sh`

Key runtime changes:

- `conv_k3_eo_emit` now supports `schema_version=2` families with `family_mode=fixed_height_band`
- `--target-width` is supported and required when height alone is ambiguous
- schema-v2 emission uses:
  - checked-in field-analysis JSON
  - checked-in field-spec JSON
  - checked-in anchor serialized executable
  - bounded per-target lookup residue
- the completion demo is now shape-aware (`hX_wY`) instead of height-only

## Scientific result

The Phase 5 family is not “pure formula only.” It is:

- a rule-based EO core from word-field analysis over the width axis
- plus bounded target-specific lookup residue where the rule core does not fully close the gap

For the frozen family:

- endpoint widths `104` and `152` require `0` lookup residue in all three regimes
- interior holdouts `116` and `140` require bounded lookup residue
- the active emitter reconstructs the exact target instruction chunk locally for every frozen case

This is materially stronger than Phase 4’s pure byte-table EO path:

- the active loop is still compiler-free
- final EO bytes are not stored wholesale per target
- the emitted patch is partly predicted from reusable field rules

## Verification

### Local exact instruction reconstruction
For every frozen target in every regime, the active emitter reconstructs the exact target instruction chunk from:

- family spec
- field-analysis JSON
- field-spec JSON
- anchor executable
- lookup residue

### Pi DUT proof
Full artifact:

- `traces/analysis/phase4-conv2d-k3-completion-demo-20260316T092300Z/`

Smoke artifact:

- `traces/analysis/phase4-conv2d-k3-completion-demo-20260316T092204Z/`

Result:

- `all_hash_eq_target=True`

Passed matrix:

- `p32`: `8x104`, `8x116`, `8x128`, `8x140`, `8x152`
- `p64`: `8x104`, `8x116`, `8x128`, `8x140`, `8x152`
- `p128`: `8x104`, `8x116`, `8x128`, `8x140`, `8x152`

## What is now complete

Phase 5 is complete for the deliberately bounded `6496` family:

- fixed-height `H=8` Conv2D family
- widths `104,116,128,140,152`
- symmetric `p32/p64/p128`
- native target params from uncompiled TFLite
- native EO emission from field rules + bounded lookup residue
- pure-`rusb` replay on Pi
- exact target-hash reproduction

## What remains outside Phase 5

This does not yet prove:

- the full `H=8` band beyond the frozen widths
- a family-wide pure closed-form EO law with no lookup residue
- portability to other heights inside `EO=6496`
- portability beyond Conv2D `k=3`, stride `1`, padding `same`, bias `off`

The next phase should therefore focus on:

1. widening the frozen `H=8` width set,
2. compressing lookup residue further toward a more reusable law, and
3. testing whether the mixed-product family model extends beyond `fixed_height=8`.
