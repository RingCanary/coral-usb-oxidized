# Phase 3 Conv2D EO window refinement (2026-03-06)

## Goal
Refine the coarse Conv2D `1x1` EO topology discovered in the earlier 8-way group probe.

Previous coarse result:
- broad transport-critical body
- transport-safe but semantic-only tail

This follow-up asks a narrower question:
- which subwindows inside those two coarse regions are still transport-critical,
- and which are exact-removable or semantic-only?

## Helper and artifact
- Helper:
  - `scripts/phase3_conv2d_eo_window_refine_probe.sh`
- Artifact:
  - `traces/analysis/phase3-conv2d-eo-window-refine-probe-20260306T142438Z/`

DUT host:
- `rpc@10.76.127.205`

## Tested pairs
- `p32`: `16x64x32 -> 32x32x32`
- `p64`: `16x64x64 -> 32x32x64`
- `p128`: `16x64x128 -> 32x32x128`

For each pair we refined two coarse windows:
- `w00`: broad body
- `w01`: semantic tail

Each window was subdivided into 4 bins, and we tested `minus` cases only.

## High-level result
### 1) The broad body is still mostly transport-critical
- `p64`: all 4 body bins fatal when removed
- `p128`: all 4 body bins fatal when removed
- `p32`: 3 of 4 body bins fatal; one middle body bin became transport-safe but hash-wrong

So the Conv2D EO body is **not** uniformly fatal, but for `p64/p128` it still behaves as a densely transport-critical region at this granularity.

### 2) The tail is fully transport-safe at this granularity
All tail-bin removals were transport-safe in all 3 pairs.

This confirms and sharpens the earlier coarse result:
- the tail is a semantic region, not a transport gate.

### 3) Some tail bins are already exact-removable in the current tested moves
Exact-removable bins at this granularity:
- `p32`: `2292..2310`, `2349..2376`
- `p64`: `2297..2315`
- `p128`: `2349..2376`

Other tail bins remain semantic-only:
- transport-safe, but hash-wrong if removed.

## Detailed results

### `p32`
#### Body `w00 = 242..2292`
- fatal when removed:
  - `242..1296`
  - `1531..1744`
  - `1746..2292`
- semantic-only when removed:
  - `1298..1526`

#### Tail `w01 = 2292..3236`
- exact-removable:
  - `2292..2310`
  - `2349..2376`
- semantic-only:
  - `2315..2344`
  - `2386..3236`

### `p64`
#### Body `w00 = 242..2297`
All 4 bins fatal when removed:
- `242..1292`
- `1294..1522`
- `1526..1744`
- `1746..2297`

#### Tail `w01 = 2297..3236`
- exact-removable:
  - `2297..2315`
- semantic-only:
  - `2321..2349`
  - `2354..2377`
  - `2386..3236`

### `p128`
#### Body `w00 = 242..2289`
All 4 bins fatal when removed:
- `242..1302`
- `1307..1542`
- `1544..1744`
- `1746..2289`

#### Tail `w01 = 2289..3236`
- exact-removable:
  - `2349..2376`
- semantic-only:
  - `2289..2306`
  - `2311..2345`
  - `2386..3236`

## Interpretation
This is the strongest Conv2D EO structural result so far.

### Stable claim now supported
For tested same-product `1x1` Conv2D spatial moves:
- the tail region around roughly `2290..3236` is **transport-safe**,
- but not uniformly removable,
- and contains at least some small exact-removable subwindows.

### More conservative claim
The earlier broad body around `242..2290` is still mostly transport-critical, but not monolithic:
- `p32` already exposes one semantic-only interior subwindow (`1298..1526`),
- so deeper recursive refinement should focus there rather than treating the whole body as uniformly fatal.

## Best next EO step
Recurse again only inside:
1. the large transport-critical body for `p64/p128`, and
2. the semantic-only body window `1298..1526` for `p32`, and
3. the semantic tail bins that are not yet exact-removable.

That is likely a better next move than any further whole-stream uniform partitioning.
