# Phase 3 Conv2D EO p32 core refine r2 (2026-03-07)

## Goal
Run one last justified recursion inside the remaining `p32` interior toxic band from the previous note:

- previous core candidate: `1430..1512`
- stop after the first split that exposes a non-fatal sub-bin

## Helper and artifact

- Helper:
  - `scripts/phase3_conv2d_eo_p32_core_refine_probe.sh`
- Artifact:
  - `traces/analysis/phase3-conv2d-eo-p32-core-refine-probe-20260306T190231Z/`

Run parameters:

- `WINDOW_START=1430`
- `WINDOW_END=1512`

## Result

### Transport-fatal bins

- `1430..1433`
- `1449..1456`
- `1458..1512`

All three removals fail with `UsbError(Timeout)`.

### Exact-removable bin

- `1436..1444`

Removing this bin still passes and preserves the target hash exactly:

- target hash: `0x37d44edf947eeac7`
- minus-bin hash: `0x37d44edf947eeac7`

## Interpretation

This is the first split inside the `p32` interior toxic band that exposes an exact-removable gap.

So the currently known `p32` interior structure is no longer:

- one contiguous toxic region

but instead:

- `1430..1433`: transport-fatal
- `1436..1444`: exact-removable
- `1449..1456`: transport-fatal
- `1458..1512`: transport-fatal

Combined with the earlier note, the current best p32-only boundary is:

- `1298..1419`: transport-safe, hash-wrong
- `1430..1433`: transport-fatal
- `1436..1444`: exact-removable
- `1449..1456`: transport-fatal
- `1458..1512`: transport-fatal
- `1516..1526`: transport-safe, hash-wrong

## Stop condition

This is a reasonable stopping point for the Phase 3 `p32` recursion line.

Why:

1. the first exact-removable gap inside the interior toxic band is now isolated,
2. the remaining fatal region is already fragmented enough to support the boundary claim,
3. further recursion would be byte-minimization work, not a boundary-setting necessity.
