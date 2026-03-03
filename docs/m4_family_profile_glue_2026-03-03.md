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

Example profile:
- `docs/artifacts/family_profiles/holdout_family8976_2352_anchor1792_v1.example.json`

## Replay integration
File:
- `examples/rusb_serialized_exec_replay.rs`

### New CLI options
- `--family-profile PATH`
- `--weights-row-major-u8-file PATH`
- `--weights-row-major-i8-file PATH`
- `--weights-pattern-index-mod`
- `--weights-pattern-modulus N`
- `--weights-pattern-signed-reinterpret`

### Behavior
- If `--family-profile` is provided:
  - profile is loaded and validated,
  - `anchor_model` can populate `--model` when omitted,
  - `instruction_patch_spec` can populate replay patch spec when omitted,
  - replay defaults can populate `input_bytes/output_bytes/bootstrap_known_good_order` when CLI values are unchanged defaults.
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

## Build/test status
- `cargo check --example rusb_serialized_exec_replay --bin dense_param_pack --lib` PASS
- `cargo test --lib` PASS (includes new `family_profile` tests)
