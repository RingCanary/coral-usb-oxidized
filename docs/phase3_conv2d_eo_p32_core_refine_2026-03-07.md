# Phase 3 Conv2D EO p32 core refine (2026-03-07)

## Goal
Continue the narrowest justified Phase 3 follow-up from the targeted refine note:

- recurse only inside `p32`'s newly isolated transport-fatal core,
- avoid reopening any already-settled `p64/p128` windows,
- check whether `1430..1526` contains a smaller toxic subset.

## Helper and artifact

- Helper:
  - `scripts/phase3_conv2d_eo_p32_core_refine_probe.sh`
- Artifact:
  - `traces/analysis/phase3-conv2d-eo-p32-core-refine-probe-20260306T190118Z/`

## Tested window

- `p32` core window:
  - `1430..1526`

The window was split into 4 `minus` bins.

## Result

### Transport-fatal bins

- `1430..1436`
- `1440..1456`
- `1458..1512`

All three removals fail with `UsbError(Timeout)`.

### Transport-safe but semantic-only bin

- `1516..1526`

Removing this bin still passes transport, but the resulting hash is wrong:

- target hash: `0x37d44edf947eeac7`
- minus-bin hash: `0x7aa0efdfc18e6332`

## Interpretation

This contracts the currently known `p32` interior toxic region again:

- previous targeted note: `1430..1526` transport-fatal
- current result: only `1430..1512` remains transport-fatal at this resolution

So the upper tail of that earlier toxic band is no longer transport-critical:

- `1516..1526` is semantic-only

## Current best p32-only boundary

For the previously exposed interior body region:

- `1298..1419`: transport-safe, hash-wrong
- `1430..1512`: transport-fatal
- `1516..1526`: transport-safe, hash-wrong

## Best next move

If this line is pushed one step further, the best next recursion is now:

1. split only `1430..1512`,
2. stop immediately if all sub-bins remain fatal,
3. otherwise record the first smaller transport-critical core and stop there.

That keeps the work aligned with the Phase 3 closeout discipline: narrow, evidence-first, and no broad repartitioning.
