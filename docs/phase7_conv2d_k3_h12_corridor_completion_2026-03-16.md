# Phase 7 Conv2D `k=3` `H=12` Corridor Completion

Date: 2026-03-16

## Scope

Phase 7 tests whether the `schema_version=2` / `family_mode=fixed_height_band`
path extends beyond the completed `H=8` corridor.

Final frozen family:

- operator: single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- `fixed_height=12`
- widths: `64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192`
- regimes: `p32/p64/p128`
- active family spec: `templates/phase7_conv2d_k3_h12_corridor_6512/family.json`

## Discovery Result

The explicit all-regime scout on
`traces/analysis/phase4-conv2d-k3-family-scout-20260316T114108Z/` showed a
contiguous `H=12` family across the full scanned width set `64..192` at step
`8`:

- `p32`: `k3_eo6512_pc688_param9472`
- `p64`: `k3_eo6512_pc688_param37376`
- `p128`: `k3_eo6512_pc688_param148480`

This proves the corridor-style schema extends beyond `fixed_height=8`.

## Tail Recovery

The first over-wide freeze attempted to keep the full scanned tail
`12x64..12x192`. Pi replay then isolated a narrower real blocker:

- EO emission was exact for the failing tail widths
- only `p32` at `12x176/184/192` failed
- the failure was hash drift, not transport failure

Follow-up comparison showed the issue is not the EO family law. The emitted EO
patchspecs for `p32 12x176/184/192` reconstruct the target executable chunk
exactly, but native parameter materialization diverged from the compiler stream
for those three widths.

That tail is now closed. The native parameter materializer applies the recovered
`kernel_pos-major` packing switch for `k=3`, `p32`, `H=12`, `W>=176`, and a
fresh local Pi5+Coral proof reran the excluded widths directly on-device.

## Frozen Family

The final checked-in family under
`templates/phase7_conv2d_k3_h12_corridor_6512/` now keeps the full successful
all-regime corridor and stays inside the same lookup cap used for Phase 6:

- `lookup_cap=96`
- `predict_mode=threepoint`
- anchor `12x128`

Observed maxima in `SUMMARY.txt`:

- `p32 max_lookup_rules=91`
- `p64 max_lookup_rules=29`
- `p128 max_lookup_rules=28`

## Validation

Local validation:

- `cargo check --bin conv_k3_eo_emit --bin conv_k_param_materialize --bin word_field_spec_v2`
- `bash -n scripts/phase5_conv2d_k3_freeze_band_assets.sh`
- `bash -n scripts/phase4_conv2d_k3_completion_demo.sh`

Discovery and intermediate artifacts:

- scout artifact:
  `traces/analysis/phase4-conv2d-k3-family-scout-20260316T114108Z/`
- p64 full-slice smoke:
  `traces/analysis/phase4-conv2d-k3-completion-demo-20260316T114345Z/`
- over-wide all-regime attempt exposing the `p32` tail problem:
  `traces/analysis/phase4-conv2d-k3-completion-demo-20260316T114632Z/`

Final Pi DUT validation:

- original bounded matrix artifact:
  `traces/analysis/phase4-conv2d-k3-completion-demo-20260316T115420Z/`
- matrix:
  `p32/p64/p128 × {12x64,12x72,12x80,12x88,12x96,12x104,12x112,12x120,12x128,12x136,12x144,12x152,12x160,12x168}`
- result:
  `all_hash_eq_target=True`

Tail-close Pi DUT validation:

- proof artifact:
  `traces/analysis/phase7-p32-tail-dut-proof-20260316T145105Z/`
- matrix:
  `p32 × {12x176,12x184,12x192}`
- result:
  `all_hash_eq_target=True`

This demonstrates exact DUT output `fnv1a64` equality between:

- compiled target baseline replay
- native completion replay (`anchor executable + native params + native EO`)

for every frozen width in every supported regime, including the recovered
`p32` high tail.

## Conclusion

Phase 7 is complete for a second bounded schema-v2 corridor family:

- the active native path now supports a bounded same-product `EO=6512` family
- a widened `fixed_height=8` `EO=6496` corridor
- and a second bounded `fixed_height=12` `EO=6512` corridor that now spans
  `12x64..12x192`

The next concrete frontier is narrower than before:

- test whether more corridor families exist beyond `H=8` and `H=12`
- continue reducing the gap between bounded family emitters and general native
  executable/codegen control
