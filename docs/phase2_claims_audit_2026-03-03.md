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

### Early single-family check
Initial DUT experiment:
- prep artifact:
  - `traces/analysis/m5-crossdim-anchor-vs-target-20260303T193705Z/`
- DUT matrix:
  - `traces/analysis/specv3-m5-crossdim-anchor-vs-target-20260303T193705Z-dut-20260303T193737Z/SUMMARY.txt`

That run established the first failure mode:
- stale anchor instructions are not generally transport-safe across dims.

### Stronger family-wide check (superseding the earlier shorthand)
We then ran a non-degenerate family-wide matrix using random-uniform weights and same-family transpose pairs with equal parameter-stream length:
- doc:
  - `docs/phase2_dense_m55_crossdim_oracle_matrix_2026-03-06.md`
- main artifact:
  - `traces/analysis/m5-crossdim-oracle-matrix-20260306T103420Z/`
- DUT logs:
  - `traces/analysis/specv3-m5-crossdim-oracle-matrix-20260306T103420Z-dut/`

Per-family result:
- `target_override`: PASS + hash==target baseline in all 4 families
- `anchor_param_only` (stale EO+PC):
  - `7056/7952/8976`: FAIL `UsbError(Overflow)`
  - `9872`: PASS but hash drift
- `anchor_pc_oracle` (exact PC target bytes only): never reaches target-equivalent replay
- `anchor_eo_oracle` (exact EO target bytes only): PASS + hash==target baseline in all 4 families
- `anchor_eopc_oracle`: same hash as `anchor_eo_oracle`

Conclusion:
- EO cannot be treated as a fixed anchor when crossing dimensions (even within same EO/PC-length family).
- Cross-dim target-equivalent replay was **never** achieved without EO target bytes.
- In these same-product transpose probes, EO exact target bytes were already sufficient; PC exact target bytes were not required for target-equivalent replay.

## 6) Coarse EO block structure is now evidenced

Follow-up DUT ablations on non-degenerate cross-dim oracle EO patchsets:
- `docs/phase2_dense_eo_group_ablation_2026-03-06.md`
- runs:
  - `traces/analysis/m5-eo-oracle-group-probe-20260306T105020Z/`
  - `traces/analysis/m5-eo-oracle-group-probe-20260306T105435Z/`
  - `traces/analysis/m5-eo-oracle-group-probe-20260306T105801Z/`
  - `traces/analysis/m5-eo-oracle-group-probe-20260306T110121Z/`

What this adds:
- EO oracle bytes are not uniformly opaque; removable blocks exist.
- `7056` and `8976` continue to show the same topology class under finer refinement: early prefix + tail-critical structure, with middle transport-safe semantic regions.
- Representative 1/32 refinement for `7056` (`traces/analysis/m5-eo-oracle-group-probe-20260306T112056Z/`) exposed multiple *candidate* hash-neutral removable windows (`1208..1448`, `4258..4476`, `5137..5204`).
- `9872` remains a distinct topology class: stale EO is transport-safe but wrong-hash, and 1/32 refinement exposes both a transport-critical prefix and several *candidate* hash-neutral removable windows (`2862..3548`, `5090..5660`).

Reverse-direction cross-check then tightened the interpretation:
- doc:
  - `docs/phase2_dense_eo_neutral_window_crosscheck_2026-03-06.md`
- run:
  - `traces/analysis/m5-eo-neutral-window-crosscheck-20260306T114730Z/`
- result:
  - `f7056`: none of the forward candidate windows remained reverse-direction neutral
  - `f9872`: only `2862..3206` survived reverse-direction neutral-window validation

So most discovered “neutral windows” are currently best treated as **context-local**, not yet family-level neutral invariants.

This is still not a generator, but it materially narrows where deeper EO minimization should focus.

Targeted refinement then sharpened those critical regions further:
- doc:
  - `docs/phase2_dense_eo_transport_window_refine_2026-03-06.md`
- run:
  - `traces/analysis/m5-eo-window-refine-probe-20260306T115811Z/`
- notable outputs:
  - `f7056`: compact fatal subwindows at `338..340` and `5518..5658`, plus dense fully-critical `902..1176`
  - `f8976`: compact fatal subwindows at `338..504` and `7042..7253`, plus dense fully-critical `902..1784`
  - `f9872`: compact fatal subwindows at `338..442` and `1336..1432`, plus dense fully-critical `872..1304`

So the next useful recursion target is no longer “EO in general”, but these much smaller transport-critical subwindows.

Rule-level refinement inside only the compact fatal subwindows then sharpened the boundary one step further:
- run:
  - `traces/analysis/m5-eo-rule-refine-probe-20260306T122419Z/`
- strongest shared result:
  - removing EO offset `338` is individually transport-fatal in all three studied families/classes (`f7056`, `f8976`, `f9872`)
- strongest family-specific compact fatal offsets:
  - `f7056`: `338`, `5658`
  - `f8976`: `338`, `7194`
  - `f9872`: `338`, `1357`, `1359`, `1415`
- nearby bytes often separate cleanly into neutral or semantic-only roles, e.g.:
  - `340` is removable without hash drift in all three studied families
  - `f7056:5657`, `f8976:{503,504,7193,7253}`, `f9872:1394` are transport-safe but semantic

This does **not** yet produce a generator, but it tightens the residual dependency from broad EO windows to a small set of individually critical bytes plus still-dense fully-critical blocks.

## 7) Safe-core rule-count variance

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
