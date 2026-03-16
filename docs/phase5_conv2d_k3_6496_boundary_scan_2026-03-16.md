# Phase 5 Conv2D `k=3` `6496` boundary scan (2026-03-16)

## Goal
Start Phase 5 by answering the first hard question before touching the active loop:

> does `8x128` have an intra-family `6496` partner, or is it a singleton family boundary point?

The intended Phase 5 path required at least two shapes inside the `6496` family so we could freeze a nontrivial `family.json` and prove one native replay *inside* that family.

## New helper capability
- `scripts/phase4_conv2d_k3_family_scout.sh`

The scout now supports an env-driven same-product mode:

- `MODE=sameprod`
- `SAME_PRODUCT=...`
- `HEIGHTS_CSV=...`
- `CHANNELS_CSV=...`
- optional `OUT_CHANNELS_CSV=...`

This keeps the old fixed-case behavior intact while making the Phase 5 family scan reproducible.

## Artifacts

### Reproducible p64 scans via the generalized scout
- lower/lower-mid height axis:
  - `traces/analysis/phase4-conv2d-k3-family-scout-20260316T071530Z/`
- upper/transposed height axis:
  - `traces/analysis/phase4-conv2d-k3-family-scout-20260316T071607Z/`

### Additional bounded corroboration scans
- `p32` lower-half axis:
  - `traces/analysis/phase5-conv2d-k3-p32-height-scan-20260316T071333Z/`
- `p128` lower-half axis:
  - `traces/analysis/phase5-conv2d-k3-p128-height-scan-20260316T071333Z/`

## Result

### `p64` full same-product power-of-two height axis
For `same_product = 1024`, `in_channels = out_channels = 64`, `kernel_size = 3`, `stride = 1`, `padding = same`:

- `1x1024` -> `EO=4112`, `PC=688`, `params=12800`
- `2x512` -> `EO=5200`, `PC=688`, `params=37376`
- `4x256` -> `EO=6224`, `PC=688`, `params=37376`
- `8x128` -> `EO=6496`, `PC=688`, `params=37376`
- `16x64` -> `EO=6512`, `PC=688`, `params=37376`
- `32x32` -> `EO=6512`, `PC=688`, `params=37376`
- `64x16` -> `EO=6512`, `PC=688`, `params=37376`
- `128x8` -> `EO=6512`, `PC=688`, `params=37376`
- `256x4` -> `EO=6512`, `PC=688`, `params=37376`
- `512x2` -> `EO=5520`, `PC=688`, `params=37376`
- `1024x1` -> `EO=4448`, `PC=688`, `params=12800`

This is the decisive Phase 5 finding:

> on the scanned `p64` same-product power-of-two axis, `8x128` is the only `6496` member.

So the hoped-for first `6496` family is not `{8x128, 4x256, ...}`. It is currently a singleton island.

### `p32` corroboration
The lower-half `p32` scan matches the same qualitative split:

- `1x1024` -> `EO=4112`
- `2x512` -> `EO=5200`
- `4x256` -> `EO=6224`
- `8x128` -> `EO=6496`
- `16x64`, `32x32`, `64x16`, `128x8` -> `EO=6512`

### `p128` corroboration
The lower-half `p128` scan also matches from `2x512` upward:

- `2x512` -> `EO=5200`
- `4x256` -> `EO=6224`
- `8x128` -> `EO=6496`
- `16x64`, `32x32`, `64x16`, `128x8` -> `EO=6512`

`1x1024` did not reach the usual extracted compiled artifact for `p128`; `edgetpu_compile.log` reports:

> `Compilation failed due to large activation tensors in model.`

So for `p128`, the most extreme tall/flat endpoint is currently a compileability limit, not just another family data point.

## Consequence
Phase 5 does **not** yet proceed to:

- freezing a nontrivial `6496` checked-in family spec,
- generalizing the active completion runner for `6496`,
- or attempting a native intra-`6496` Pi replay proof.

That would be premature, because the first chosen discovery axis does not provide a second `6496` partner shape.

The correct scientific outcome of this round is:

> the `6512 -> 6496` boundary is real, but the scanned p64 family does not yet contain a reusable intra-family `6496` pair.

## What is now proven

- `8x128` is not simply the first member of a large nearby `6496` family on the power-of-two same-product axis.
- EO family identity is strongly aspect-ratio-conditioned; it does not evolve monotonically as height halves.
- `PC` stays fixed at `688` across all successful scans here, so the new boundary pressure is still in EO family structure rather than PC size.
- For `p32/p64`, the lower-half axis is regime-stable at the EO-family level.

## Next step
The next valid Phase 5 move is more discovery, not active-loop generalization:

1. widen the local search beyond the current power-of-two same-product axis,
2. find a second nontrivial `6496` partner if one exists,
3. only then freeze `templates/...6496/family.json` and wire the active emitter/runner to it.

Until that exists, the bounded Phase 4 `6512` family remains the last completed active path.
