# Phase 3 Conv2D EO p32 interior byte map (2026-03-07)

## Milestone
Complete the `p32` interior EO recursion to a byte-level classification and stop at the first conclusive boundary-setting point.

This closes the remaining narrow Phase 3 `p32` follow-up line from:

- `docs/phase3_conv2d_eo_targeted_refine_2026-03-07.md`
- `docs/phase3_conv2d_eo_p32_core_refine_2026-03-07.md`
- `docs/phase3_conv2d_eo_p32_core_refine_r2_2026-03-07.md`

## Artifacts

Single-byte probes:

- `traces/analysis/phase3-conv2d-eo-p32-core-refine-probe-20260306T190449Z/`
- `traces/analysis/phase3-conv2d-eo-p32-core-refine-probe-20260306T190526Z/`
- `traces/analysis/phase3-conv2d-eo-p32-core-refine-probe-20260306T190555Z/`

Helper:

- `scripts/phase3_conv2d_eo_p32_core_refine_probe.sh`

## Final byte-level classification

Inside the previously unresolved `p32` interior body band:

- transport-fatal:
  - `1430`
  - `1433`
  - `1456`
  - `1463`
- exact-removable:
  - `1449`
  - `1458`
  - `1465`
- semantic-only:
  - `1432`
  - `1454`
  - `1512`

Transport-fatal means removing the byte causes replay failure (`UsbError(Timeout)`).

Exact-removable means removing the byte still yields the exact target hash:

- target hash: `0x37d44edf947eeac7`

Semantic-only means replay still passes transport but the output hash changes.

## Combined p32 interior map

Putting the full p32 interior region together:

- `1298..1332`: semantic-only
- `1337..1419`: semantic-only
- `1430`: transport-fatal
- `1432`: semantic-only
- `1433`: transport-fatal
- `1436..1444`: exact-removable
- `1449`: exact-removable
- `1454`: semantic-only
- `1456`: transport-fatal
- `1458`: exact-removable
- `1463`: transport-fatal
- `1465`: exact-removable
- `1512`: semantic-only
- `1516..1526`: semantic-only

This is now a fragmented mixed-meaning region, not a contiguous toxic block.

## Interpretation

This is the first conclusive milestone for the p32 interior EO line.

Why it is conclusive:

1. the remaining uncertain region was reduced to individual rule bytes,
2. each byte now has a behavioral class,
3. the result already demonstrates the structural point that mattered:
   the p32 interior body region is a fine-grained mix of transport-fatal, exact-removable, and semantic-only bytes.

Further recursion would no longer sharpen the Phase 3 boundary in a meaningful way.

## Final Phase 3-consistent claim

For tested same-product `1x1` Conv2D spatial moves:

- `p64/p128` still retain a densely transport-critical body at the current useful granularity,
- `p32` does not: its interior body region is now proven to be mixed at byte scale,
- the tail remains transport-safe and also decomposes into exact-removable and semantic-only bytes/subwindows.

That is strong enough to stop this line of work without reopening whole-stream EO minimization.
