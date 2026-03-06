# Phase 3 Conv2D EO group probe (2026-03-06)

## Goal
Start the next Conv2D follow-up on the remaining unsolved surface:
- EO target-state dependence for same-product spatial moves.

This probe deliberately reuses the existing exact-oracle artifact and asks a conservative first question:
- if we partition the EO oracle into coarse contiguous rule groups, which groups are transport-critical, and which are only semantic?

## Artifact and helper
- Source oracle artifact:
  - `traces/analysis/phase3-conv2d-crossdim-oracle-matrix-20260306T132611Z/`
- New helper:
  - `scripts/phase3_conv2d_eo_group_probe.sh`
- New run:
  - `traces/analysis/phase3-conv2d-eo-group-probe-20260306T140308Z/`

All DUT runs used Pi `rpc@10.76.127.205` with:
- `--reset-before-claim`
- no hub power-cycle
- same pure-rusb replay path as the Phase-3 oracle matrix

## Tested pairs
- `p32`: `16x64x32 -> 32x32x32`
- `p64`: `16x64x64 -> 32x32x64`
- `p128`: `16x64x128 -> 32x32x128`

Each EO oracle patchset was partitioned into 8 contiguous rule groups.
For each group we tested:
- `only`: keep only that group
- `minus`: remove only that group from the full EO oracle

## High-level result
Across all 3 tested same-product pairs:
- the full EO oracle still gives target-equivalent replay
- **no single group is sufficient** by itself for target-equivalent replay
- removing groups `g00..g06` is transport-fatal in all 3 pairs
- removing only the final tail group `g07` is transport-safe but hash-wrong in all 3 pairs

This means the tested Conv2D EO structure is **not** “one small magic island”.
The target-equivalent state is distributed across a broad transport-critical prefix/middle region, while the tail group is semantic-only at this granularity.

## Detailed observations

### Shared across `p32`, `p64`, `p128`
For all 3 pairs:
- `eo_full`: PASS + target hash
- `eo_minus_g07`: PASS but wrong hash
- `eo_minus_g00..g06`: `UsbError(Timeout)`

This is the strongest shared signal in the run.
At 1/8 granularity, the last EO group is removable without breaking transport, but it is still needed for target-equivalent semantics.

### `p32`
Representative ranges:
- `g00`: `242..808`
- `g01`: `809..1320`
- `g02`: `1321..1449`
- `g03`: `1454..1568`
- `g04`: `1570..1682`
- `g05`: `1687..1992`
- `g06`: `1996..2289`
- `g07`: `2292..3236`

Results:
- `eo_g06_only`: PASS but wrong hash
- `eo_minus_g07`: PASS but wrong hash
- everything else in `only` or `minus` form: transport-fatal

### `p64`
Representative ranges:
- `g00`: `242..805`
- `g01`: `809..1307`
- `g02`: `1318..1445`
- `g03`: `1449..1566`
- `g04`: `1568..1682`
- `g05`: `1687..1992`
- `g06`: `1996..2292`
- `g07`: `2297..3236`

Results mirror `p32`:
- `eo_g06_only`: PASS but wrong hash
- `eo_minus_g07`: PASS but wrong hash
- everything else: transport-fatal

### `p128`
Representative ranges:
- `g00`: `242..813`
- `g01`: `818..1324`
- `g02`: `1329..1456`
- `g03`: `1458..1571`
- `g04`: `1575..1687`
- `g05`: `1689..1902`
- `g06`: `1992..2281`
- `g07`: `2289..3236`

Results:
- `eo_g05_only`: PASS but wrong hash
- `eo_g06_only`: PASS but wrong hash
- `eo_minus_g07`: PASS but wrong hash
- everything else: transport-fatal

So `p128` keeps the same broad class, but the semantic-only / transport-safe interior appears slightly wider than `p32/p64` at this coarse granularity.

## Interpretation
This does **not** solve EO synthesis, but it narrows the structure:

1. The Conv2D EO residual is again **family-structured**, not arbitrary byte noise.
2. The broad region from roughly `242..2290` behaves as transport-critical at 1/8 granularity in all tested same-product pairs.
3. The tail region around roughly `2290..3236` is semantic-only at this granularity:
   - removable without breaking transport,
   - but still required for target-equivalent replay.
4. `p128` suggests some non-tail semantic-only structure also exists earlier in the stream.

## Practical consequence
For tested same-product `1x1` Conv2D moves:
- exact target EO bytes are still the only known way to reach target-equivalent replay,
- but the EO stream now has its first coarse topology:
  - broad transport-critical body
  - semantic-only tail block
  - possible additional semantic interior in the larger-channel case

## Best next step
Do not split the whole EO stream uniformly again.
The next useful refinement is to recurse only inside:
- the transport-critical body, and
- the semantic-only tail window,

to find smaller reusable EO substructures or field-like regions.
