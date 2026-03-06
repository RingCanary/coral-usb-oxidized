# Phase 3 Conv2D — Completion / Exit Note (2026-03-06)

## Scope actually completed
Phase 3 was intentionally narrowed to the high-value subset:
- single-op Conv2D
- `1x1`
- stride `1`
- padding `same`
- bias `off`

This was a deliberate correction to the original deferred plan.
Multi-op Conv2D->Dense, `k>1`, depthwise, and wider activation/bias variants remain deferred.

## Critical review of the plan after execution
The original deferred Conv2D plan was directionally good, but too broad for a first pass.

What worked:
- start with single-op `1x1` Conv2D
- map families first
- revalidate parameter layout before writing runtime code
- test cross-dim EO dependence explicitly instead of assuming Dense conclusions transfer

What we deliberately did **not** need for closeout:
- multi-op graphs
- depthwise / `k>1`
- large EO minimization campaign
- Conv2D family-profile glue as a prerequisite for stating the residual dependency boundary

The phase is complete because the boundary is now precise, not because every possible Conv2D variant was generalized.

## What Phase 3 now proves

### 1) 1x1 Conv2D family structure is much simpler than Dense in the first tested regime
Kickoff + bootstrap artifacts:
- `docs/phase3_conv2d_kickoff_2026-03-06.md`
- `traces/analysis/phase3-conv2d-family-bootstrap-20260306T125706Z/`

Across the initial `1x1` sweep over spatial/channel changes, EO/PC instruction lengths stayed fixed:
- `EO = 5360`
- `PC = 688`

Only parameter bytes changed with channel regime:
- `1280`
- `4608`
- `8704`
- `9216`
- `17408`

This is already a materially different shape from Dense Phase 2, where recurrent family routing was a larger part of the problem.

### 2) The recovered 1x1 Conv2D packing law is now wider and sharper
Artifact:
- `traces/analysis/phase3-conv2d-layout-matrix-20260306T131544Z/`

The old February clue was revalidated and generalized.
Current recovered layout for tested cases:
- output channels partition into blocks of up to `64`
- each block contributes `block_width * 8` prefix bytes
- within a block, weight bytes use the local mapping:
  - `((ic // 4) * (block_width * 4)) + ((oc % block_width) * 4) + (ic % 4)`

Validated cases:
- `32 -> 32`
- `64 -> 64`
- `64 -> 128`
- `128 -> 64`
- `128 -> 128`

All tested probe points matched the recovered formula.

### 3) The full tested 1x1 Conv2D parameter stream is now compilerless and exact
Artifacts:
- initial narrowing result: `traces/analysis/phase3-conv2d-param-override-matrix-20260306T132851Z/`
- exact-recovery proof: `traces/analysis/phase3-conv2d-param-override-matrix-20260306T135325Z/`
- stale-prefix motivation probe: `traces/analysis/phase3-conv2d-prefix-residual-probe-20260306T132745Z/`

New Rust support:
- `src/param_pack.rs`
- `src/bin/conv1x1_param_pack.rs`

New helper:
- `tools/dump_tflite_conv1x1_weights.py`

Important corrections discovered during implementation:
- the tested TFLite Conv2D `1x1` constant tensor is stored in `[out, 1, 1, in]` order,
- compiled parameter payload weight bytes use a `+128` encoding bias,
- the prefix is blockwise, not global,
- the effective-scale float table matches the compiler only when computed as:
  - `effective_scale = (input_scale * weight_scale) * f32(1 / output_scale)`
  - i.e. using a pre-rounded `f32` reciprocal of `output_scale`

Recovered tested prefix law per output block (up to 64 channels):
- `f32 effective_scale[out]`
- `u32 stored_zero_point[out]`
- with `stored_zero_point = weight_zero_point + 128`

After wiring that in, the compilerless packer reached exact local equivalence in all tested cases:
- `64 -> 64`: `local_byte_equal=True`, `prefix_mismatch_count=0`, `weight_mismatch_count=0`
- `64 -> 128`: `local_byte_equal=True`, `prefix_mismatch_count=0`, `weight_mismatch_count=0`
- `128 -> 64`: `local_byte_equal=True`, `prefix_mismatch_count=0`, `weight_mismatch_count=0`

DUT result for full-stream overrides in all 3 cases:
- replay PASS
- output hash exactly equals the compiled baseline

So for the tested single-op `1x1` regime:
> the full parameter stream is now compilerless and exact.

### 4) Same-product spatial cross-dim replay depends only on EO target bytes in the tested regime
Artifact:
- `traces/analysis/phase3-conv2d-crossdim-oracle-matrix-20260306T132611Z/`

Tested same-product pairs:
- `p32`: `16x64x32 -> 32x32x32`
- `p64`: `16x64x64 -> 32x32x64`
- `p128`: `16x64x128 -> 32x32x128`

Key prep result in all 3 pairs:
- `param_equal=True`
- `pc_rule_count=0`
- EO changed bytes: `164`, `169`, `173`

DUT result in all 3 pairs:
- target baseline: PASS + target hash
- anchor baseline: PASS but wrong hash
- EO oracle only: PASS + target hash
- EO+PC oracle: same as EO oracle
- PC oracle: no-op / zero-rule case

This is a very strong Conv2D result.
For the tested same-product `1x1` spatial moves:
> the cross-dim blocker is **EO only**.

Parameters are already equal.
PC does not change.
EO exact target bytes are sufficient.

## Final practical boundary for Phase 3

### Compilerless today for tested Conv2D 1x1 regime
- family/size bootstrap and routing evidence
- exact blockwise packing formula for the weight region
- exact blockwise packing formula for the prefix region
- full compilerless parameter-stream generation with local byte-equivalence
- DUT hash-equivalence with full-stream parameter override at anchor dims
- same-product cross-dim replay analysis showing PC is not the blocker

### Not compilerless today for tested Conv2D 1x1 regime
- the **EO target-state bytes** for same-product spatial moves

So the residual non-compilerless surface is now narrower still:
> **EO run-phase target bytes for unseen/target Conv2D spatial moves**

That is the correct closeout boundary for this phase.

## What Phase 3 does *not* prove
- general Conv2D (`k>1`, depthwise, stride changes, etc.)
- multi-op Conv2D->Dense behavior
- EO synthesis for unseen Conv2D target dims
- profile-glue generalization across Conv2D operator families

## Exit decision
Phase 3 Conv2D is complete enough to close because:
1. the plan was executed critically rather than mechanically,
2. the high-value `1x1` subset was bounded precisely,
3. the remaining compiler dependency is no longer vague.

## Recommended next Conv2D work, if revisited later
1. focus directly on EO target-state synthesis for same-product spatial moves,
2. only after that widen beyond single-op `1x1`,
3. keep widening conservative (`k>1`, stride changes, depthwise) and evidence-first,
4. defer multi-op Conv2D->Dense until single-op Conv2D target-state dependence is better understood.
