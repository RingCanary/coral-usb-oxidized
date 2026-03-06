# Phase 4 Conv2D `k=3` scout (2026-03-06)

## Goal
Start widening beyond the completed single-op `1x1` Conv2D phase in the smallest useful way:
- still single-op
- `stride=1`
- `padding=same`
- `bias=off`
- but now `kernel_size=3`

This is a **scout**, not a full new phase.
The goal is to learn whether the `1x1` conclusions immediately collapse outside the narrow regime.

## Helper and artifact
- New helper:
  - `scripts/phase4_conv2d_k3_family_scout.sh`
- Artifact:
  - `traces/analysis/phase4-conv2d-k3-family-scout-20260306T141320Z/`

## Local scanned cases
- `32x32x64 -> 64, k=3`
- `32x32x64 -> 128, k=3`
- `32x32x128 -> 64, k=3`

## Local size results
All 3 scanned cases landed in the same instruction-size family:
- `EO = 6512`
- `PC = 688`

Parameter bytes varied by channel regime:
- `37376` for `64 -> 64`
- `74752` for `64 -> 128`
- `74240` for `128 -> 64`

So the first non-1x1 scout suggests:
1. moving from `k=1` to `k=3` does change the EO family (`5360 -> 6512`),
2. but the PC size stayed fixed at `688` in this first small scan,
3. and the `k=3` space may itself have simple recurrent families worth mapping before any deeper reverse-engineering.

## First DUT anchor baseline
A first DUT baseline was run for:
- `32x32x64 -> 64, k=3`

Log:
- `traces/analysis/phase4-conv2d-k3-family-scout-20260306T141320Z/h32_w32_ic64_oc64_k3/dut_anchor_baseline.log`

Result:
- PASS
- output bytes: `65536`
- hash: `0x31b571782f19bc2d`
- run timing: `2.005 ms`

## Interpretation
This is only a first scout, but it already gives two useful boundaries:

1. The completed `1x1` recovery should **not** be overgeneralized to `k=3`.
   The EO family changed immediately.
2. The step from `1x1` to `k=3` does **not** look chaotic yet.
   In this first scan, the `k=3` cases share one EO/PC size family while parameter bytes still track the channel regime.

## Best next widening step
Before trying to recover `k=3` parameter layout or EO behavior, first do for `k=3` what worked for `1x1`:
1. expand the local family scan a bit,
2. check whether spatial-only moves at fixed channels keep EO/PC sizes stable,
3. then test whether the parameter stream has a simple compilerless structure or immediately needs a different law.
