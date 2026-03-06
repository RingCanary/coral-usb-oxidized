# Phase 2 Dense — EO Neutral-Window Reverse Cross-Check (2026-03-06)

## Why this check
The earlier EO group-ablation runs identified several **candidate** hash-neutral/removable EO windows in one cross-dim direction.

That was not enough to conclude those windows were generally neutral.
We needed a second in-family move to test whether the same windows remain removable.

Because each tested family only had one equal-parameter-length transpose pair, the natural second move is the **reverse direction**:
- forward: `A -> B`
- reverse: `B -> A`

## Tooling
Added:
- `scripts/m5_eo_neutral_window_crosscheck.sh`

It reuses the M5.5 build artifact, constructs the reverse-direction EO oracle, removes candidate windows discovered in the forward-direction study, and runs the same DUT replay matrix on Pi5 + Coral.

## Source artifact
- `traces/analysis/m5-crossdim-oracle-matrix-20260306T103420Z/`

## Run
- `traces/analysis/m5-eo-neutral-window-crosscheck-20260306T114730Z/`

Pi host:
- `rpc@10.76.127.205`

## Families tested
### `f7056`
Forward-direction candidate removable/hash-neutral windows from 1/32 study:
- `1208..1448`
- `4258..4476`
- `5137..5204`

Reverse-direction results:
- `eo_full_reverse`: PASS, hash == target baseline
- remove `1208..1448`: **FAIL** `UsbError(Timeout)`
- remove `4258..4476`: **FAIL** `UsbError(Timeout)`
- remove `5137..5204`: PASS but **hash != target**

Result:
- **none** of the forward-direction candidate windows remained reverse-direction neutral.

### `f9872`
Forward-direction candidate removable/hash-neutral windows from 1/32 study:
- `2862..3206`
- `3210..3548`
- `5090..5318`
- `5322..5660`

Reverse-direction results:
- `eo_full_reverse`: PASS, hash == target baseline
- remove `2862..3206`: PASS, hash == target baseline
- remove `3210..3548`: **FAIL** `UsbError(Timeout)`
- remove `5090..5318`: **FAIL** `UsbError(Timeout)`
- remove `5322..5660`: **FAIL** `UsbError(Timeout)`

Result:
- only **one** forward-direction candidate window survived reverse-direction cross-check:
  - `2862..3206`

## Interpretation
This materially changes how the earlier EO group-ablation results should be read.

### What is still true
- EO has structure.
- There are transport-critical and semantic-only regions.
- The block-ablation method is useful.

### What is **not** yet justified
- treating most forward-discovered “neutral windows” as family-level neutral EO regions.

### Updated conclusion
Most candidate neutral windows are **context-local**, not yet reusable family-level invariants.

Current status:
- `f7056`: forward neutral windows did **not** survive reverse cross-check
- `f9872`: only `2862..3206` survived reverse cross-check

So the safer interpretation is:
> EO neutrality claims require multi-move validation; one-direction neutrality is not sufficient evidence.

## Implication for next step
The correct follow-up remains:
1. refine transport-critical windows recursively,
2. cross-check any candidate removable windows on a second move before preserving them as neutral/template-excludable regions.
