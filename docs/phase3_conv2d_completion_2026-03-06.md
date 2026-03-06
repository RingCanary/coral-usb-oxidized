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

### 3) The Conv2D parameter stream splits into two parts: exact compilerless weight payload + unresolved prefix bytes
Artifact:
- `traces/analysis/phase3-conv2d-param-override-matrix-20260306T132851Z/`

New Rust support:
- `src/param_pack.rs`
- `src/bin/conv1x1_param_pack.rs`

New helper:
- `tools/dump_tflite_conv1x1_weights.py`

Important correction discovered during implementation:
- the tested TFLite Conv2D `1x1` constant tensor is stored in `[out, 1, 1, in]` order,
- and compiled parameter payload bytes use a `+128` encoding bias for the weight region.

With that corrected, the results became clean:
- full stream byte-equivalence: **False**
- weight-region byte-equivalence: **True**
- remaining mismatches are confined to the non-weight prefix bytes

Representative results:
- `64 -> 64`: `prefix_mismatch_count=320`, `weight_mismatch_count=0`
- `64 -> 128`: `prefix_mismatch_count=637`, `weight_mismatch_count=0`
- `128 -> 64`: `prefix_mismatch_count=320`, `weight_mismatch_count=0`

DUT result for zero/stale-prefix overrides:
- replay still passes transport,
- but output hash drifts from the compiled baseline in every tested case.

So the compilerless story for Conv2D `1x1` is:
> the weight payload is now compilerless and exact,
> but the prefix bytes remain a residual dependency.

### 4) The residual Conv2D parameter dependency is semantic, not just cosmetic
Artifact:
- `traces/analysis/phase3-conv2d-prefix-residual-probe-20260306T132745Z/`

This probe held the weight region fixed to the correct target weights, but replaced the prefix with a stale same-shape prefix from another seed.

Result:
- baseline: PASS, hash `0xb57cd4d5e6a691dd`
- stale-prefix hybrid: PASS, hash `0x6586bc8da1c03d1e`

So the unresolved prefix bytes are not transport filler.
They are semantically relevant to target-equivalent replay.

### 5) Same-product spatial cross-dim replay depends only on EO target bytes in the tested regime
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
- weight-region packing formula
- exact compilerless generation of the **weight payload** portion of the parameter stream
- same-product cross-dim replay analysis showing PC is not the blocker

### Not compilerless today for tested Conv2D 1x1 regime
- the **parameter-stream prefix bytes**
- the **EO target-state bytes** for same-product spatial moves

So the residual non-compilerless surface is now much smaller than “Conv2D in general”:
> **(a) Conv2D 1x1 parameter-prefix bytes and (b) EO run-phase target bytes**

That is the correct closeout boundary for this phase.

## What Phase 3 does *not* prove
- full compilerless parameter-stream synthesis for Conv2D `1x1`
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
1. decode or factor the Conv2D parameter-prefix bytes,
2. only then consider whether full compilerless parameter-stream generation is realistic,
3. keep EO work focused on same-product spatial moves before widening to `k>1`,
4. defer multi-op Conv2D->Dense until single-op Conv2D target-state dependence is better understood.
