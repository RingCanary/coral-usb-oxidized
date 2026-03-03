# M4 Tiered Instruction Profile Validation on DUT (2026-03-03)

## Objective
Validate family-profile tiered instruction patch composition on Pi5 + Coral:
- generic `safe_core` patch applied for in-family dims,
- optional per-dim `discrete_flags` overlay merged on top.

## Code support
- `src/family_profile.rs`
  - Added `instruction_patches` schema support:
    - `generic.safe_core` / `generic.full`
    - `overlays[]` keyed by exact `{input_dim, output_dim}`
- `examples/rusb_serialized_exec_replay.rs`
  - Auto-resolves tiered patch paths from `--family-profile`
  - Merges multiple patchspec sources with conflict detection
  - Logs source + merged rule counts

## Device run
Run dir:
- `traces/analysis/specv3-m4-tiered-profile-device-20260303T165658Z/`
- summary: `traces/analysis/specv3-m4-tiered-profile-device-20260303T165658Z/SUMMARY.txt`

Profiles used:
- `profiles/profile_safe_only.json`
  - generic `safe_core` = `pc_strict.full.patchspec` (14 rules)
- `profiles/profile_safe_plus_overlay.json`
  - generic `safe_core` = `pc_strict.full.patchspec` (14 rules)
  - overlay (`1792x1792`) `discrete_flags` = `eo_v2_nontoxic6.rust.patchspec` (6 rules)

Cases:
1. `baseline`
   - PASS hash `0x67709fedfd103a2d`
2. `profile_safe_only`
   - merged patch sources=1, rule_count=14
   - PASS hash `0x67709fedfd103a2d`
3. `profile_safe_plus_overlay`
   - merged patch sources=2, rule_count=20
   - PASS hash `0xf790ee9e92c4c4f1`

Checks:
- `safe_only_equals_baseline=true`
- `overlay_changes_hash=true`

## Interpretation
Tiered instruction composition works as intended on DUT:
- generic safe_core can be applied dimension-generically in-family,
- per-dim discrete overlay is optional and materially changes semantics when present,
- no reboot/hub power cycle needed (used `--reset-before-claim`).
