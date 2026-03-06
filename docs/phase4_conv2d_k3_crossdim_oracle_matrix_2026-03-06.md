# Phase 4 Conv2D `k=3` same-product cross-dim oracle matrix (2026-03-06)

## Goal
Take the first real step beyond `1x1` Conv2D by testing whether the same-product spatial cross-dim pattern survives for single-op `k=3`, `stride=1`, `padding=same`, `bias=off`.

The key questions were:
1. are parameters already equal across same-product spatial moves?
2. does PC change?
3. is EO still the only blocker?

## Helper and artifact
- Helper:
  - `scripts/phase4_conv2d_k3_crossdim_oracle_matrix.sh`
- Artifact:
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/`

DUT host:
- `rpc@10.76.127.205`

## Tested pairs
- `p32`: `16x64x32 -> 32x32x32`
- `p64`: `16x64x64 -> 32x32x64`
- `p128`: `16x64x128 -> 32x32x128`

All cases use:
- `kernel_size = 3`
- `stride = 1`
- `padding = same`
- `bias = off`

## Prep result
Across all 3 tested `k=3` pairs:
- `param_equal = False`
- `pc_rule_count = 0`
- EO changed bytes: nonzero (`255`, `259`, `270`)

This is the first major divergence from the completed `1x1` Conv2D story.

## DUT result matrix
For all 3 pairs:
- target baseline: PASS + target hash
- anchor baseline: PASS but wrong hash
- target params only (`anchor_param_only`): PASS but wrong hash
- EO only (`anchor_eo_oracle`): PASS but wrong hash
- PC only: zero-rule / no-op case
- target params + EO (`anchor_param_eo_oracle`): PASS + target hash

This was consistent for:
- `p32`
- `p64`
- `p128`

## What this proves
For tested same-product `k=3` Conv2D moves:
> **both target params and target EO are required** for target-equivalent replay.

More precisely:
- **PC is not the blocker** in the tested regime (`pc_rule_count = 0`)
- **params alone are not enough**
- **EO alone is not enough**
- **params + EO together are sufficient**

So the residual dependency structure for `k=3` is already different from `1x1`.

## Contrast with `1x1`
### `1x1` tested same-product moves
- `param_equal = True`
- `pc_rule_count = 0`
- EO target bytes alone were sufficient

### `k=3` tested same-product moves
- `param_equal = False`
- `pc_rule_count = 0`
- EO target bytes alone are **not** sufficient
- target params + target EO are required

This is the clearest evidence yet that the completed `1x1` story should not be generalized to larger kernels.

## Interpretation
The first `k=3` widening step reveals a new boundary:
- the `1x1` param invariance across same-product spatial moves does **not** survive to `k=3`
- but the zero-PC result **does** survive in this first tested regime

That means the current `k=3` blocker surface is:
> parameter-stream target state + EO target state

not PC.

## Best next `k=3` step
Before trying any `k=3` EO ablation, first determine **why** the same-product spatial move changes parameters for `k=3`:
1. diff the `k=3` parameter stream structure by region,
2. test whether the changed region is a small prefix/meta section or a larger semantic layout change,
3. only after that decide whether `k=3` parameter recovery should be attacked directly or whether more family scanning is needed.
