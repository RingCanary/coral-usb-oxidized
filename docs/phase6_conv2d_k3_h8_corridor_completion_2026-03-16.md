# Phase 6 Conv2D `k=3` `H=8` Corridor Completion

Date: 2026-03-16

## Scope

This phase widens the active `schema_version=2` / `family_mode=fixed_height_band` path from the bounded Phase 5 `H=8` subset to the full observed all-regime `EO=6496`, `PC=688` corridor.

Frozen family:

- operator: single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- `fixed_height=8`
- widths: `72,76,80,...,192` (step `4`)
- regimes: `p32/p64/p128`
- active family spec: `templates/phase6_conv2d_k3_h8_corridor_6496/family.json`

## Discovery Result

The explicit all-regime scout on `traces/analysis/phase4-conv2d-k3-family-scout-20260316T094618Z/` showed that the `H=8` `EO=6496 / PC=688` family is contiguous across the full scanned corridor `72..192` at step `4`, with no interior all-regime holes.

That turns the earlier bounded Phase 5 band into a true corridor-wide family:

- `p32`: `k3_eo6496_pc688_param9472`
- `p64`: `k3_eo6496_pc688_param37376`
- `p128`: `k3_eo6496_pc688_param148480`

Nearby sampled `H=12` and `H=16` cases remain outside this family, so the widened completion claim remains explicitly scoped to `fixed_height=8`.

## Implementation Changes

Two active-path corrections were required to make the widened family exact instead of merely close:

1. `scripts/phase5_conv2d_k3_freeze_band_assets.sh`
   - generalized corridor freezing with explicit `FAMILY_ID`, `EO_PAYLOAD_LEN`, `LOOKUP_CAP`, and `PREDICT_MODE`
   - enforced `LOOKUP_CAP=96` as a hard freeze gate
   - merged lookup top-up by offset, avoiding duplicate-offset inflation

2. `src/bin/conv_k3_eo_emit.rs`
   - made field-rule writes deterministic, matching `word_field_spec_v2` sequential application semantics
   - added runtime `predict_mode` support and honored the frozen `threepoint` law instead of silently defaulting unresolved contexts to endpoint-mode behavior

The decisive support triplet for the widened law is:

- low width `76`
- mid width `128`
- high width `188`
- `tile_size=4`
- `predict_mode=threepoint`

## Lookup Cap Result

The widened family passes the tighter non-anchor lookup bar comfortably:

- `p32 max_lookup_rules=56`
- `p64 max_lookup_rules=45`
- `p128 max_lookup_rules=22`

These counts are recorded in `templates/phase6_conv2d_k3_h8_corridor_6496/SUMMARY.txt`.

## Validation

Local validation:

- `cargo check --bin conv_k3_eo_emit --bin conv_k_param_materialize --bin word_field_spec_v2`
- `bash -n scripts/phase5_conv2d_k3_freeze_band_assets.sh`
- field-only parity check on `p32 h8_w104`: `conv_k3_eo_emit` now reproduces `field_spec.patchspec` exactly (`missing=0`, `extra=0`, `diff=0`)
- widened family assets frozen under `templates/phase6_conv2d_k3_h8_corridor_6496/`

Pi DUT validation:

- smoke artifact: `traces/analysis/phase4-conv2d-k3-completion-demo-20260316T101340Z/`
  - `p64 × {8x72..8x192 step 4}`
  - `all_hash_eq_target=True`
- full matrix artifact: `traces/analysis/phase4-conv2d-k3-completion-demo-20260316T102326Z/`
  - `p32/p64/p128 × {8x72..8x192 step 4}`
  - `all_hash_eq_target=True`

The full matrix demonstrates exact DUT output `fnv1a64` equality between:

- compiled target baseline replay
- native completion replay (`anchor executable + native params + native EO`)

for every frozen width in every supported regime.

## Conclusion

Phase 6 is complete for the widened `fixed_height=8` corridor.

The active native path now supports:

- bounded same-product family completion at `EO=6512`
- widened mixed-product `H=8` corridor completion at `EO=6496`
- pure-`rusb` Pi replay with no compiler in the active artifact-creation or execution loop

The remaining frontier is no longer the first `6496` family closure. It is:

- extending beyond `fixed_height=8`
- compressing lookup residue further toward a more reusable law
- testing whether corridor-style schema-v2 families generalize to adjacent geometry bands
