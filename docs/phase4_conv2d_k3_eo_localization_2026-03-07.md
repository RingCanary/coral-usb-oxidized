# Phase 4 Conv2D `k=3` EO localization milestone (2026-03-07)

## Goal
Advance `P4-M3` from “EO unsolved” to a bounded, experimentally justified decomposition for the current Phase 4 family:

- single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- same-product spatial move `16x64 -> 32x32`
- channel regimes `p32/p64/p128`

The active condition for all EO probes was:

> anchor executable + native target parameter stream + EO subset

This removes the old parameter confound. `PC` remains zero-rule in this bounded family.

## Helpers and artifacts

### New helpers
- `scripts/phase4_conv2d_k3_eo_group_probe.sh`
- `scripts/phase4_conv2d_k3_eo_provenance_probe.sh`
- `scripts/phase4_conv2d_k3_eo_core_refine_probe.sh`

### Supporting local family analysis
- `traces/analysis/phase4-conv2d-k3-eo-local-family-20260306T193647Z/`

This local widening showed an important boundary before any DUT work:

- `16x64`, `32x32`, `64x16`, and `128x8` stay in EO family `6512`
- `8x128` leaves that family (`6496`)

So the first bounded EO synthesis claim should stay inside the `6512` family.

### DUT artifacts
- coarse `p32` group probe:
  - `traces/analysis/phase4-conv2d-k3-eo-group-probe-20260306T193945Z/`
- provenance-class probes:
  - `traces/analysis/phase4-conv2d-k3-eo-provenance-probe-20260306T194306Z/` (`p32`)
  - `traces/analysis/phase4-conv2d-k3-eo-provenance-probe-20260306T194739Z/` (`p64`)
  - `traces/analysis/phase4-conv2d-k3-eo-provenance-probe-20260306T194836Z/` (`p128`)
- `p32` transport-core refinement:
  - `traces/analysis/phase4-conv2d-k3-eo-core-refine-probe-20260306T194520Z/`

## Step 1: Local mathematical decomposition
Comparing the completed `1x1` EO oracle family (`phase3`) against the current `k=3` EO oracle family (`phase4`) gives a clean shared-offset decomposition:

- `220` EO offsets are common across `p32/p64/p128` in `k=3`
- those split into:
  - `A = 132` offsets shared with `1x1` and byte-identical
  - `B = 13` offsets shared with `1x1` but kernel-conditioned
  - `C = 75` offsets that are common `k=3`-only bytes
- per-pair overlays are small:
  - `p32 = 35`
  - `p64 = 39`
  - `p128 = 50`

The `C` set is highly structured:

- `C_early = 9` offsets before `2315`
- `C_mid = 42` offsets in `2586..3416`
- `C_tail = 24` offsets in `3422..4388`

`C_mid` is dominated by a 96-byte lattice, which justified chain-based refinement instead of more blind contiguous splitting.

## Step 2: Coarse DUT topology (`p32`)
The first `8`-way coarse `minus` probe on `p32` gave:

- removing any of `g00..g06` is transport-fatal
- removing only `g07 = 3422..4388` is transport-safe but hash-wrong

So the late tail is immediately separated from the transport-critical body.

## Step 3: Provenance-class DUT probes (`p32/p64/p128`)
The class result is stable across all three channel regimes:

- `eo_minus_A_common_same`: transport-safe, hash-wrong
- `eo_minus_B_common_kernel_conditioned`: transport-fatal
- `eo_minus_C_common_k3_only`: transport-fatal
- `eo_minus_C_early`: transport-safe, hash-wrong
- `eo_minus_C_mid`: transport-fatal
- `eo_minus_C_tail`: transport-safe, hash-wrong
- `eo_minus_overlay_pair_unique`: transport-safe, hash-wrong

This is the decisive family-level localization result.

It means the bounded-family EO problem is no longer “all EO bytes.” The transport-critical surface is now:

> `B + C_mid`

Everything else is transport-safe and semantic-only.

## Step 4: `p32` transport-core refinement
Refining the `p32` transport core gave:

### `B` single-byte split

Exact-removable:
- `242`
- `338`
- `434`
- `530`
- `809`
- `813`
- `818`
- `827`
- `1465`
- `1577`
- `1689`

Semantic-only:
- `2281`

Transport-fatal:
- `3232`

### `C_mid` chain split

Transport-fatal chains:
- `2586,2682,2778,2874,2970,3066,3162,3258,3354`
- `2656,2752,2848,2944`
- `3161,3257,3353`
- `3328`

Exact-removable chains:
- `2588,2684,2780,2876,2972,3068`
- `2592,2688,2784,2880,2976,3072`

Semantic-only chains:
- `2648,2744,2840,2936`
- `3154,3250,3346`
- `3160,3256,3352`
- `3230,3326`
- `3416`

## `p32` byte-class milestone
For `p32`, the full EO oracle now splits into:

- exact-removable: `23` bytes
- semantic-only: `214` bytes
- transport-fatal: `18` bytes

Transport-fatal `p32` EO bytes are exactly:

- `3232`
- `2586,2682,2778,2874,2970,3066,3162,3258,3354`
- `2656,2752,2848,2944`
- `3161,3257,3353`
- `3328`

That is the first Phase 4 byte-scale EO transport core.

## What is now proven

### Proven family-level result
Across `p32/p64/p128`, the `k=3` EO oracle cleanly decomposes into:

- transport-safe semantic sets:
  - `A`
  - `C_early`
  - `C_tail`
  - per-pair overlay
- transport-critical sets:
  - `B`
  - `C_mid`

### Proven `p32` result
Inside the `p32` transport-critical surface, most of the feared `55` bytes are already gone:

- `11` kernel-conditioned bytes are exact-removable
- `1` is semantic-only
- only `1` remains individually transport-fatal
- among `C_mid`, only `17` bytes remain transport-fatal at chain granularity

So the `p32` EO transport core is now `18` bytes total.

## What is not solved yet
`P4-M3` is not fully complete yet.

What remains unsolved is native emission of the **semantic-only** EO bytes with exact target hash, especially:

- the `A` shared retarget set,
- the `C_tail` late semantic tail,
- the semantic-only `C_mid` chains,
- the small per-pair overlays.

The failed local holdout fit (`311 -> 276` mismatch only) shows that broad geometry-only exact EO synthesis is still too weak. After this localization milestone, the remaining synthesis task is much narrower:

> emit the transport-fatal core exactly, then fit or table-drive only the remaining semantic-only bytes.

## Phase 4 consequence
Phase 4 is materially closer to completion:

- params are solved natively (`P4-M2`)
- EO is now localized to a family-level class split
- `p32` transport-critical EO is reduced to `18` bytes

The remaining blocker is no longer “find the EO core.” It is:

> native emission of the semantic-only EO target state for the bounded family.
