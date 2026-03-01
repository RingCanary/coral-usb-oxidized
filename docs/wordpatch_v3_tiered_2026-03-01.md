# Word-Patch v3 Tiered Synthesis + Pi5 DUT Checks (2026-03-01)

## Scope
Implemented tiered/monotonicity-aware synthesis flow and validated generated patch subsets on Pi5 + Coral USB.

## Code Changes

### 1) Rust v2 generator extended (`src/bin/word_field_spec_v2.rs`)
- Added predict modes:
  - `strict` (endpoint interpolation + non-monotonic skip gate)
  - `threepoint` (3-point const/linear/quadratic fit; fallback to `best` if no mid point)
- Added optional mid-point inputs:
  - `--mid-dim`
  - `--mid-exec`
- Added tiered patchspec outputs:
  - `--out-patchspec` (full)
  - `--out-patchspec-safe` (safe_core only)
  - `--out-patchspec-discrete` (discrete_flags only)
- Added monotonicity classes on contexts:
  - `const`, `monotone_up`, `monotone_down`, `midpoint_pulse`, `non_monotone`, `insufficient`
- Added patch tiers:
  - `safe_core`, `discrete_flags`, `unknown`
- Added strict-mode skip gate in `propose_offset_rule()` for non-monotonic contexts.
- Added three-point predictor path in `predict_word()` using 3-point fitting.
- Added report fields:
  - `mono_class`, `patch_tier` per offset note
  - `safe_core_byte_count`, `discrete_flags_byte_count`, `unknown_byte_count`
- Added self-validation warning when `--mid-dim == --target-dim` under `threepoint`.

### 2) Python analysis tool extended (`tools/instruction_word_field_analysis.py`)
- Added `_classify_monotonicity(vals, dims)`.
- Added per-offset fields in `per_offset_fits`:
  - `monotonicity`
  - `is_monotone`
  - `monotonicity_dims`
- Added group-level summary fields:
  - `non_monotone_offset_count`
  - `non_monotone_fraction`

## Safe-Core Composition (no code path)
Composed manual transport-safe union:
- `traces/analysis/specv2-holdout-family8976-20260301T2030Z/safe_core.rust.patchspec`
- Rule count: `43` (`PC=37`, `EO=6`)

## Local Verification

### Three-point self-validation (PC family 2352)
Command used:
- `--predict-mode threepoint --mid-dim 1792 --mid-exec <1792>`
- base=`896`, target=`1792`, high=`2688`

Observed:
- `baseline_mismatch=0`
- `v2_mismatch=0`
- `changed_bytes=229`
- tier split: `safe_core=63`, `discrete_flags=166`, `unknown=0`
- patchspec counts satisfy: `full == safe + discrete`

### Strict mode sanity (same family)
Observed:
- `changed_bytes=49`
- tier split: `safe_core=49`, `discrete_flags=0`, `unknown=0`

## Holdout Artifact Regeneration (strict)
Output dir:
- `traces/analysis/specv3-tiered-holdout-family8976-20260301T170035Z/`

Strict outputs (base=target=1792 executable for each stream):
- PC strict full: `14` bytes (`pc_strict.full.patchspec`)
- EO strict full: `8` bytes (`eo_strict.full.patchspec`)
- Combined strict full: `22` bytes (`both_strict.full.patchspec`)

Notable EO strict bytes include toxic-known offsets:
- `746`, `975`, `1103`, `1231` (present in `eo_strict.full.patchspec`)

## Pi5 Reboot-First Matrix (tiered/strict/safe)
Run dir:
- `traces/analysis/specv3-tiered-strict-safe-matrix-20260301T170113Z/`

Cases:
1. baseline
2. safe_core43 (manual union)
3. pc_strict14
4. eo_strict8
5. both_strict22

Results:
- baseline: **PASS**, hash `0x67709fedfd103a2d`
- safe_core43: **PASS transfer**, hash `0x89c84d0b6795819c` (semantic change)
- pc_strict14: **PASS**, hash `0x67709fedfd103a2d` (baseline-equivalent)
- eo_strict8: **FAIL** event then output timeout
- both_strict22: **FAIL** event then output timeout (EO-dominated)

## Interpretation
- Tiering + strict gating successfully isolated a transport-safe **PC strict subset** (`14` bytes) that preserves baseline hash.
- EO strict still contains known toxic non-linear bytes; it remains transfer-fatal.
- Manual safe_core union is transfer-safe but semantically divergent (hash changed).

## Follow-up Matrix: EO Strict Minus Toxic4
Run dir:
- `traces/analysis/specv3-tiered-eo-minus-toxic-matrix-20260301T170557Z/`

Constructed patchspecs:
- `eo_strict.minus_toxic4.patchspec` (`4` bytes)
  - kept: `615, 642, 866, 6726`
  - removed toxic: `{746, 975, 1103, 1231}`
- `both_pcstrict14_eostrictminus4.patchspec` (`18` bytes)
  - `pc_strict14` + `eo_strict.minus_toxic4`

Results:
- baseline: **PASS**, hash `0x67709fedfd103a2d`
- `eo_minus_toxic4`: **PASS transfer**, hash `0x27c68f0d32ba3e60`
- `both_pc14_eo_minus4`: **PASS transfer**, hash `0x505440f4aab46c09`

## Updated Interpretation
- Removing EO toxic4 converts strict EO from transfer-fatal to transfer-safe.
- `pc_strict14` remains transport-safe and baseline-equivalent when applied alone.
- Combined `pc_strict14 + eo_minus_toxic4` is now transport-safe end-to-end.
- Remaining work is now semantic convergence (hash matching), not transport stability.

## Next Step
Perform semantic delta-debugging on transfer-safe set (`pc_strict14 + eo_minus_toxic4`) to recover baseline hash while preserving stability:
- ablate EO safe bytes first (`615, 642, 866, 6726`)
- then test selective re-introduction of additional non-toxic EO/PC bytes
- keep reboot-first acceptance criteria: admission/event/output stability + hash objective.
