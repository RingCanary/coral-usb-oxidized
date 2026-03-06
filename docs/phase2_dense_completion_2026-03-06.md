# Phase 2 Dense — Completion / Exit Note (2026-03-06)

## Scope
This closes the Dense-focused Phase 2 work tracked through M5, M5.5, M6, and M7.

Conv2D (previously labeled M8 in the checklist) remains intentionally deferred.

## What Phase 2 now proves

### 1) Dense parameter streaming is compilerless
Proven by local byte-equivalence and DUT hash-equivalence:
- Rust packer reproduces the compiled Dense parameter stream exactly
- square and rectangular stored shapes both validated
- runtime hot path can inject params without compiler involvement

Key references:
- `docs/m4_compilerless_param_override_2026-03-03.md`
- `docs/m4_family_profile_glue_2026-03-03.md`

### 2) Family/profile routing is real, but anchor-dim validation is the correct claim
The recurrent Dense instruction families were mapped and profiled:
- `7056/1840`
- `7952/2096`
- `8976/2352`
- `9872/2608`

Anchor-dim profile validation is solid.
Cross-dim deployment without target EO bytes is not.

Key references:
- `docs/phase2_m5_transition_bootstrap_2026-03-03.md`
- `traces/analysis/specv3-m5-family-profile-dut-matrix-v2-20260303T190420Z/`

### 3) Cross-dim target-equivalent replay needs EO target bytes
This is the critical negative proof from M5.5.
For same-family transpose pairs with equal parameter-stream length:
- target params alone are insufficient
- target PC bytes alone are insufficient
- exact target EO bytes are sufficient

Observed pattern:
- `7056/7952/8976`: stale EO causes `UsbError(Overflow)`
- `9872`: stale EO is transport-safe but hash-wrong
- `anchor_eo_oracle`: PASS + target hash in all tested families

So the remaining blocker is specifically the **EO run-phase instruction state**.

Key reference:
- `docs/phase2_dense_m55_crossdim_oracle_matrix_2026-03-06.md`

### 4) EO is structured, not monolithic
Follow-up DUT ablations showed that EO contains at least three kinds of regions:
- transport-critical
- transport-safe but semantic
- move-local removable/neutral regions

But most apparent neutral regions are not family-level invariants until they survive a second move.

Key references:
- `docs/phase2_dense_eo_group_ablation_2026-03-06.md`
- `docs/phase2_dense_eo_neutral_window_crosscheck_2026-03-06.md`

### 5) Transport-critical EO windows can be narrowed sharply
Targeted refinement reduced the search from broad EO windows to compact fatal subwindows plus dense fully-critical blocks.

Then rule-level probing inside the compact fatal subwindows isolated smaller critical sets.

Key references:
- `docs/phase2_dense_eo_transport_window_refine_2026-03-06.md`
- `traces/analysis/m5-eo-rule-refine-probe-20260306T122419Z/`

## Strongest final EO findings

### Shared high-confidence pattern
Across all three studied families/classes (`f7056`, `f8976`, `f9872`), removing EO patch rule at offset **`338`** is individually transport-fatal.

This is the strongest current candidate for a shared transport-critical EO byte.

By contrast, nearby offset `340` is removable without hash drift in all three tested families.

### Family/class-specific compact fatal bytes
Rule-level refinement produced these strongest compact fatal offsets:

- `f7056`
  - fatal: `338`, `5658`
  - semantic-only neighbor: `5657`
  - neutral neighbors: `340`, `5518`

- `f8976`
  - fatal: `338`, `7194`
  - semantic-only neighbors: `503`, `504`, `7193`, `7253`
  - neutral neighbors: `340`, `441`, `442`, `7042`, `7046`, `7050`, `7054`

- `f9872`
  - fatal: `338`, `1357`, `1359`, `1415`
  - semantic-only neighbor: `1394`
  - neutral neighbors include: `340`, `442`, `1347`, `1348`, `1350`, `1351`, `1352`, `1355`, `1372`, `1374`, `1418`, `1432`
  - whole subwindow `1336..1346` became fully removable at single-rule granularity in this move

### Dense fully-critical blocks still unresolved
These remained fatal at the previous 4-way split and were intentionally not exploded further in this closeout step:
- `f7056`: `902..1176`
- `f8976`: `902..1784`
- `f9872`: `872..1304`

These blocks are likely the next place to look if later work wants deeper EO decoding, but they are not required to state the current boundary precisely.

## Practical compiler-dependency boundary after Phase 2

### Compilerless today
- Dense parameter stream generation
- known-profile replay hot path
- profile validation / patch composition / safe-core PC handling

### Not compilerless today
- generating **new target-dimension EO run-phase bytes** for unseen cross-dim deployment

So the residual non-compilerless surface has been narrowed from “the compiled model” to:
> **EO run-phase target state for a new Dense target dimension**

That is narrower than a full compiler dependency because:
- params are already compilerless,
- PC is not the blocking plane in the tested cross-dim moves,
- the unsolved part is specifically the EO target-state acquisition/synthesis problem.

## What Phase 2 does *not* prove
- a general EO byte generator
- family-general neutrality for most candidate removable windows
- cross-family portability across EO/PC payload-size transitions
- unseen-dimension target-equivalent replay without target EO bytes

## Exit decision
Phase 2 Dense is complete enough to close with a precise boundary:
- Dense deployment is compilerless on the parameter path,
- profile-driven replay is solid at validated anchor dims,
- the remaining unsolved dependency is the EO run-phase state for unseen target dims,
- and EO minimization has already narrowed that dependency to specific transport-critical regions/bytes rather than an undifferentiated blob.

## Recommended next phase
Do **not** continue widening Dense Phase 2.
Treat this as closed and start Conv2D in a separate phase only when needed.

If Dense work is revisited later, the highest-value follow-up would be:
1. structured analysis of the dense fully-critical `902..*` blocks,
2. multi-move validation of any newly discovered removable EO windows,
3. factoring whether the surviving EO-critical bytes map to a smaller reusable template/residual mechanism.
