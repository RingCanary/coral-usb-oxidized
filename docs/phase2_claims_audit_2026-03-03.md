# Phase-2 Claims Audit (2026-03-03)

This note tightens wording around what is **proven** vs **not yet proven** after M5-M7.

## 1) M5 profile validation scope

What was run:
- `traces/analysis/specv3-m5-family-profile-dut-matrix-v2-20260303T190420Z/`

What it proves:
- For each recurrent family profile, applying current profile defaults at its own anchor dimension is transport-safe and baseline-hash-preserving.

What it does **not** prove:
- Anchor->non-anchor cross-dim deployment for those families.

## 2) EO toxicity narrative correction

Previous shorthand (“EO toxic because quant/activation fields”) was incomplete.

Evidence:
- Prior EO toxic subset (family 8976): offsets `{746, 975, 1103, 1231}`.
- M6 controlled-axis EO-sensitive set (family 8976 @ 1792x1792): offsets `6810..6836` from
  - `traces/analysis/m6-instruction-axis-probe-20260303T190926Z/semantic_labels.json`
- Intersection is empty.

Conclusion:
- M6 identifies a semantic EO region (activation/quant sensitivity), but does not by itself explain previously isolated EO-toxic bytes.
- EO toxic bytes likely include additional non-linear/safety-critical fields (still unresolved).

## 3) M6 single-family generalization check

Added explicit second-family cross-check (family `7952/2096`, fixed `1536x1536`):
- Local instruction-axis diff:
  - `traces/analysis/m6-crosscheck-f7952-20260303T193142Z/SUMMARY.txt`
- DUT variant matrix:
  - `traces/analysis/m6-crosscheck-f7952-dut-20260303T193230Z/SUMMARY.txt`

Observed pattern matches family 8976:
- PC changed offsets: `0` for quant/activation/bias axis changes.
- EO changed offsets: non-zero, concentrated block (`5978..6004`).
- DUT hashes: only quant-offset variant changes output hash; relu/relu6/bias variants match baseline.

## 4) Rectangular-member caveat

Valid concern: rectangular third-members can confound one-axis scalar fits.

Current status:
- Rectangular points were used to unlock recurrent family coverage and tooling bootstrap.
- They should be treated as provisional training support, not final proof of generalizable transition rules.

## 5) Cross-dim anchor deployment reality check (critical)

Performed direct DUT experiment to test whether EO can stay anchored while changing dimension geometry.

Prep artifact:
- `traces/analysis/m5-crossdim-anchor-vs-target-20260303T193705Z/`
  - anchor: `640x1280`
  - target: `1280x640`
  - same family `7056/1840`, same parameter length (`819200`)
  - direct diffs: EO changed `311` bytes, PC changed `50` bytes

DUT matrix:
- `traces/analysis/specv3-m5-crossdim-anchor-vs-target-20260303T193705Z-dut-20260303T193737Z/SUMMARY.txt`

Results:
- `target_baseline`: PASS
- `anchor_baseline`: FAIL `UsbError(Overflow)`
- `anchor_pc_oracle` (PC exact target diff only): FAIL `UsbError(Overflow)`
- `anchor_eopc_oracle` (EO+PC exact target diffs): PASS and hash == target baseline

Conclusion:
- EO cannot be treated as a fixed anchor when crossing dimensions (even within same EO/PC length family).
- PC-only geometry patching is insufficient for cross-dim execution.

## 6) Safe-core rule-count variance

Rule-count differences (`14` vs `36/60/69`) are real and should not be interpreted as equal confidence.
- `8976` safe subset is heavily DUT-vetted (historical toxic-byte isolation).
- New-family safe subsets are less mature and require deeper byte-ablation before “portable safe-core” claims.

## Updated practical claim boundary

Currently validated:
1. Compilerless parameter stream generation (dense) is solid.
2. PC instruction patches can be transport-safe at anchor-dim profile runs.
3. EO contains unresolved dimension-coupled semantics; EO byte synthesis remains the blocker.

Not yet validated:
- General anchor->non-anchor deployment without compiler-derived EO for each target dim.
