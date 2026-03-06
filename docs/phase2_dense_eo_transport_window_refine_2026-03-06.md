# Phase 2 Dense — EO Transport-Window Refinement (2026-03-06)

## Goal
After coarse 8/16/32-way EO ablations and reverse neutral-window cross-checks, recurse only inside the currently transport-critical EO windows to isolate smaller transport-critical vs removable subranges.

## Tooling
Added:
- `scripts/m5_eo_window_refine_probe.sh`

This probe reuses the non-degenerate M5.5 cross-dim artifact and, for each family, partitions only the currently transport-critical EO windows into 4 bins. It then removes one sub-bin at a time from the full EO oracle patchspec and runs the result on Pi5 + Coral.

## Source artifact
- `traces/analysis/m5-crossdim-oracle-matrix-20260306T103420Z/`

## Run
- `traces/analysis/m5-eo-window-refine-probe-20260306T115811Z/`

Pi host:
- `rpc@10.76.127.205`

## Family results

### `f7056`
Targeted windows:
- `338..540`
- `902..1176`
- `5300..5658`

Findings:
- `338..540`
  - removing `338..340` is transport-fatal
  - removing `441..540` is hash-neutral in this move
- `902..1176`
  - **all four sub-bins are transport-fatal**
- `5300..5658`
  - removing `5518..5658` is transport-fatal
  - removing `5500..5514` is hash-neutral in this move
  - removing `5300..5496` is transport-safe but semantic

Interpretation:
- two very compact transport-critical EO zones now stand out strongly:
  - `338..340`
  - `5518..5658`
- plus a dense fully-critical mid block:
  - `902..1176`

### `f8976`
Targeted windows:
- `338..680`
- `902..1784`
- `6830..8501`

Findings:
- `338..680`
  - removing `338..504` is transport-fatal
  - removing `530..548` is hash-neutral in this move
  - removing later parts is transport-safe but semantic
- `902..1784`
  - **all four sub-bins are transport-fatal**
- `6830..8501`
  - removing `7042..7253` is transport-fatal
  - removing the other 3 bins is transport-safe but semantic

Interpretation:
- `f8976` remains in the same topology class as `f7056`:
  - compact early transport-critical block
  - dense fully-critical mid block
  - compact tail-critical block

### `f9872`
Targeted windows:
- `338..646`
- `872..1304`
- `1336..1784`

Findings:
- `338..646`
  - removing `338..442` is transport-fatal
  - removing `504..548` is hash-neutral in this move
  - removing `552..646` is transport-safe but semantic
- `872..1304`
  - **all four sub-bins are transport-fatal**
- `1336..1784`
  - removing `1336..1432` is transport-fatal
  - removing `1574..1784` is transport-safe but semantic

Interpretation:
- `f9872` still differs from the overflow families overall,
- but it also contains compact transport-critical subwindows plus one dense fully-critical block.

## Current best transport-critical candidates
These are still move-specific, but they are the strongest current refinement targets.

| Family | Strong compact fatal subwindows | Dense fully-critical block |
|---|---|---|
| `f7056` | `338..340`, `5518..5658` | `902..1176` |
| `f8976` | `338..504`, `7042..7253` | `902..1784` |
| `f9872` | `338..442`, `1336..1432` | `872..1304` |

## What changed vs previous step
This is more actionable than the earlier 16/32-way summary because we now know which critical windows are:
- compact and likely worth byte-level or word-level minimization first,
- vs broad dense regions that are still too large for direct interpretation.

## Next step
1. Recurse again only inside the compact fatal subwindows.
2. Leave the dense fully-critical blocks for later structured analysis.
3. Keep treating move-local hash-neutral bins as provisional unless they survive reverse validation.
