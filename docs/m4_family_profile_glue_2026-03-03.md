# M4 Glue Layer: Dense Family Profile + Replay Integration (2026-03-03)

## Summary
Implemented a family-profile glue path so replay can compose:

1. anchor model executable payloads,
2. optional instruction patch spec defaults,
3. compilerless dense parameter stream generation from raw row-major weights.

This narrows deployment dependency to "anchor-per-family" plus profile-driven synthesis inputs.

## New library module
- `src/family_profile.rs`

Exports:
- `DenseFamilyProfile`
- `DenseFamilyReplayDefaults`
- `DenseFamilyProfileError`

Key behavior:
- JSON profile parsing + validation (`schema_version=1`)
- stored weight shape validation through recovered dense pack constraints
- optional expected stream length check
- profile-relative path resolution for:
  - `anchor_model`
  - `instruction_patch_spec`

### Profile schema (v1)
```json
{
  "schema_version": 1,
  "profile_id": "...",
  "anchor_model": "...",
  "instruction_patch_spec": "... (optional)",
  "stored_weight_shape": [rows, cols],
  "expected_param_stream_len": 3211264,
  "replay_defaults": {
    "input_bytes": 1792,
    "output_bytes": 1792,
    "bootstrap_known_good_order": true
  },
  "notes": "optional"
}
```

Example profiles:
- Legacy single-spec style:
  - `docs/artifacts/family_profiles/holdout_family8976_2352_anchor1792_v1.example.json`
- Tiered style (`generic.safe_core` + per-dim overlays):
  - `docs/artifacts/family_profiles/holdout_family8976_2352_tiered_v1.example.json`

## Replay integration
File:
- `examples/rusb_serialized_exec_replay.rs`

### New CLI options
- `--family-profile PATH`
- `--check-profile` (lint/resolve profile plan and exit before USB)
- `--weights-row-major-u8-file PATH`
- `--weights-row-major-i8-file PATH`
- `--weights-pattern-index-mod`
- `--weights-pattern-modulus N`
- `--weights-pattern-signed-reinterpret`

### Behavior
- If `--family-profile` is provided:
  - profile is loaded and validated,
  - `anchor_model` can populate `--model` when omitted,
  - legacy `instruction_patch_spec` can populate replay patch spec when omitted,
  - tiered `instruction_patches` can auto-select generic + per-dim overlay patchspec paths,
  - replay merges selected patchspec sources with conflict checks,
  - merged patch payload lengths are validated against extracted executable chunk lengths,
  - replay defaults can populate `input_bytes/output_bytes/bootstrap_known_good_order` when CLI values are unchanged defaults.
- If `--check-profile` is set:
  - replay resolves model + patch plan,
  - validates referenced files and merged patch compatibility,
  - exits before any USB open/claim/reset activity.
- If a weight source option is used with `--family-profile`:
  - row-major weights are packed with `pack_dense_row_major_*_to_stream`,
  - resulting stream overrides extracted parameter stream before USB replay,
  - hashes are logged for source and final stream.

### Validation guards
- Exactly one weight source allowed
- Weight-source generation requires `--family-profile`
- Weight-source generation cannot be combined with `--param-stream-override-file`
- Pattern controls require `--weights-pattern-index-mod`
- Pattern modulus constrained to `[1, 256]`

## DUT validation (Pi5 + Coral)
Profile artifacts:
- `traces/analysis/m4-family-profile-glue-20260303T161010Z/profile_896.json`
- `traces/analysis/m4-family-profile-glue-20260303T161010Z/profile_1792.json`
- `traces/analysis/m4-family-profile-glue-20260303T161010Z/profile_rect.json`

Replay matrix run:
- `traces/analysis/specv3-m4-family-profile-dut-matrix-20260303T161108Z/`
- summary: `traces/analysis/specv3-m4-family-profile-dut-matrix-20260303T161108Z/SUMMARY.txt`

Results (base vs profile-generated pattern stream):
- `896`: base `0x8d7854bd1eb9c1e2` == profile `0x8d7854bd1eb9c1e2`
- `1792`: base `0x394aa8758535e7e9` == profile `0x394aa8758535e7e9`
- `rect (stored 1792x896)`: base `0xe0f607a60893b844` == profile `0xe0f607a60893b844`

The profile path auto-applied:
- anchor model path,
- replay defaults (`input_bytes`, `output_bytes`, `bootstrap_known_good_order`),
- compilerless param stream generation from `--weights-pattern-index-mod ...`.

## Tiered instruction patch DUT validation
Additional run:
- `traces/analysis/specv3-m4-tiered-profile-device-20260303T165658Z/`
- summary: `.../SUMMARY.txt`

Key outcomes:
- profile with generic safe-core only (PC14) reproduces baseline hash,
- profile with generic safe-core + per-dim overlay (EO nontoxic6) remains transport-stable and changes output hash as expected,
- replay logs show patch source merge (`sources=2`, `rule_count=20`) and per-payload application.

See detailed report:
- `docs/m4_tiered_instruction_profile_device_validation_2026-03-03.md`

Profile lint/check example artifact:
- `traces/analysis/m4-profile-check-local-20260303T171235Z/check_profile.log`
  - confirms `Profile check mode: PASS` and no USB operations executed.

## Build/test status
- `cargo check --example rusb_serialized_exec_replay --bin dense_param_pack --lib` PASS
- `cargo test --lib` PASS (includes new `family_profile` tests)
