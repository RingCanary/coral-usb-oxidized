# Phase 3 Conv2D EO targeted refine follow-up (2026-03-07)

## Goal
Follow the documented Phase 3 next step without widening scope:

- recurse only inside the already-isolated EO windows,
- avoid broad whole-stream repartitioning,
- sharpen the residual target-state boundary for same-product `1x1` Conv2D spatial moves.

This follow-up uses the exact same same-product oracle artifact from Phase 3 and only tests `minus` cases inside previously identified body/tail windows.

## Helper and artifact

- Helper:
  - `scripts/phase3_conv2d_eo_targeted_refine_probe.sh`
- Artifact:
  - `traces/analysis/phase3-conv2d-eo-targeted-refine-probe-20260306T185512Z/`

## Tested windows

### `p32`

- semantic-only interior body window from the previous pass:
  - `1298..1526`
- remaining semantic tail bins:
  - `2315..2344`
  - `2386..3236`

### `p64`

- full transport-critical body window:
  - `242..2297`
- remaining semantic tail bins:
  - `2321..2349`
  - `2354..2377`
  - `2386..3236`

### `p128`

- full transport-critical body window:
  - `242..2289`
- remaining semantic tail bins:
  - `2289..2306`
  - `2311..2345`
  - `2386..3236`

Each targeted window was split into 4 bins and tested as `minus` cases only.

## High-level result

### 1. `p32` semantic-only interior body window is not uniformly semantic

Previous Phase 3 result had identified `1298..1526` as transport-safe but hash-wrong when removed as a whole.

After splitting:

- semantic-only:
  - `1298..1332`
  - `1337..1419`
- transport-fatal:
  - `1430..1456`
  - `1458..1526`

So the earlier semantic-only interior window actually contains a smaller transport-critical core concentrated in its upper half.

### 2. `p64/p128` body windows remain densely transport-critical at this resolution

All 4 body-quarter removals still fail transport:

- `p64`: all `242..2297` quarters fatal
- `p128`: all `242..2289` quarters fatal

This strengthens the existing conservative claim:

> for `p64/p128`, the EO body still behaves as a densely transport-critical region at this granularity.

### 3. Tail windows continue to decompose into exact-removable and semantic-only microbins

All tested tail removals remain transport-safe.

But many of them now split cleanly into:

- exact-removable microbins, and
- semantic-only microbins that still change the DUT hash.

## Detailed results

### `p32`

#### Body semantic core `1298..1526`

- semantic-only:
  - `1298..1332`
  - `1337..1419`
- transport-fatal:
  - `1430..1456`
  - `1458..1526`

#### Tail semantic window `2315..2344`

- exact-removable:
  - `2315`
  - `2321`
- semantic-only:
  - `2328`
  - `2337..2344`

#### Tail semantic window `2386..3236`

- exact-removable:
  - `2457`
  - `3232..3236`
- semantic-only:
  - `2386`
  - `2424..2455`

### `p64`

#### Body `242..2297`

All 4 quarters remain transport-fatal:

- `242..1292`
- `1294..1522`
- `1526..1744`
- `1746..2297`

#### Tail semantic window `2321..2349`

- exact-removable:
  - `2321`
  - `2345..2349`
- semantic-only:
  - `2322..2328`
  - `2337`

#### Tail semantic window `2354..2377`

- exact-removable:
  - `2354`
  - `2358`
  - `2363`
- semantic-only:
  - `2376..2377`

#### Tail semantic window `2386..3236`

- exact-removable:
  - `2457`
  - `3232..3236`
- semantic-only:
  - `2386`
  - `2424..2455`

### `p128`

#### Body `242..2289`

All 4 quarters remain transport-fatal:

- `242..1302`
- `1307..1542`
- `1544..1744`
- `1746..2289`

#### Tail semantic window `2289..2306`

- exact-removable:
  - `2292`
  - `2297`
  - `2301..2306`
- semantic-only:
  - `2289`

#### Tail semantic window `2311..2345`

- exact-removable:
  - `2311`
  - `2315..2322`
- semantic-only:
  - `2329`
  - `2337..2345`

#### Tail semantic window `2386..3236`

- exact-removable:
  - `2457`
  - `3232..3236`
- semantic-only:
  - `2386`
  - `2424..2455`

## Interpretation

This follow-up sharpens the Phase 3 EO boundary in two important ways.

### 1. The residual unknown body dependence is now narrower for `p32`

The only newly exposed interior transport-critical region is:

- `p32`: roughly `1430..1526`

That is much tighter than the earlier `1298..1526` semantic-only-body claim and is the best current candidate for any deeper p32-only recursion.

### 2. Much of the semantic tail now contains stable exact-removable microbins

The repeated exact-removable pattern across pairs is notable:

- `2457`
- `3232..3236`

and several pair-specific exact-removable microbins now appear in the `2290..2377` region.

This suggests the tail is not just “transport-safe but semantic” in a coarse sense; it contains a mix of:

- exact-removable bookkeeping,
- semantic-only output-state bytes,
- and little or no transport gating.

### 3. Best next Phase 3-consistent move

If Conv2D `1x1` EO work is revisited again, the next highest-value move is now:

1. recurse only inside `p32: 1430..1526`,
2. optionally recurse inside one `p64/p128` body quarter only if a tighter toxic core is needed,
3. otherwise stop here and treat the remaining tail map as sufficiently characterized for the Phase 3 boundary.

The current evidence does not justify broad further partitioning of the whole EO stream.
