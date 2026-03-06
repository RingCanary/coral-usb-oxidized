# Phase 2 Dense — EO Oracle Group Ablation (2026-03-06)

## Goal
Start the post-M5.5 EO minimization pass by coarse-clustering EO oracle bytes into:
- transport-critical
- transport-safe but semantically active
- neutral/removable (if any)

## Tooling
Added:
- `scripts/m5_eo_oracle_group_probe.sh`

This script reuses an existing M5.5 cross-dim oracle artifact, partitions `eo_oracle.patchspec` into contiguous offset groups, generates `only` and `minus` subset patchspecs, syncs the current source tree to the Pi, and runs DUT replay for each subset.

## Source artifact
- `traces/analysis/m5-crossdim-oracle-matrix-20260306T103420Z/`
- non-degenerate random-uniform weights (`seed=1337`)

## Runs
### 1) `f7056` + `f9872`, 4-way split
- `traces/analysis/m5-eo-oracle-group-probe-20260306T105020Z/`

### 2) `f9872`, 8-way split
- `traces/analysis/m5-eo-oracle-group-probe-20260306T105435Z/`

### 3) `f7056`, 8-way split
- `traces/analysis/m5-eo-oracle-group-probe-20260306T105801Z/`

### 4) `f8976`, 8-way split
- `traces/analysis/m5-eo-oracle-group-probe-20260306T110121Z/`

All DUT runs used the Pi at `rpc@10.76.127.205` with `--reset-before-claim` and no hub power-cycle.

### 5) `f7056` + `f8976` + `f9872`, 16-way split
- `traces/analysis/m5-eo-oracle-group-probe-20260306T111016Z/`

### 6) `f7056` + `f9872`, 32-way split
- `traces/analysis/m5-eo-oracle-group-probe-20260306T112056Z/`

## High-level findings

### Family `f7056` (`640x1280 -> 1280x640`)
#### 8-way
- removing `g00,g01,g02,g07` is transport-fatal
- removing `g03,g04,g05,g06` is transport-safe
- `eo_minus_g03` is hash-neutral

#### 16-way
- fatal when removed: `g00,g05,g06,g07,g08,g30`
- hash-neutral when removed: `g09,g23,g28`

#### 32-way
- fatal when removed: `g00,g05,g06,g07,g08,g30`
- hash-neutral when removed:
  - `g09` (`1208..1448`)
  - `g23` (`4258..4476`)
  - `g28` (`5137..5204`)

Interpretation:
- transport-critical EO state is not one contiguous blob; it clusters into an early prefix and one later tail block,
- but there are already multiple small removable/hash-neutral windows.

### Family `f8976` (`896x1792 -> 1792x896`)
#### 8-way
- removing `g00,g01,g02,g07` is transport-fatal
- removing `g03,g04,g05,g06` is transport-safe

#### 16-way
- fatal when removed: `g00,g02,g03,g04,g15`
- no 1/16 hash-neutral removable block observed

Interpretation:
- `f8976` follows the same coarse pattern as `f7056`:
  - early prefix critical,
  - tail critical,
  - middle transport-safe but semantically active.

### Family `f9872` (`1024x2048 -> 2048x1024`)
This family differs from the overflow-fatal families.

Baseline behavior:
- stale EO (`anchor_param_only`) is transport-safe but semantically wrong.

#### 8-way
- fatal when removed: `g00,g01`
- hash-neutral when removed: `g03`

#### 16-way
- fatal when removed: `g00,g01,g02,g03,g04,g05,g06,g07,g08,g09`
- hash-neutral when removed: `g07,g14,g15,g21,g22`

#### 32-way
- fatal when removed: `g00,g03,g04,g05,g06,g07,g08,g09`
- hash-neutral when removed:
  - `g14` (`2862..3206`)
  - `g15` (`3210..3548`)
  - `g21` (`5090..5318`)
  - `g22` (`5322..5660`)
- notable transport-safe but semantic-only removals include `g01`, `g02`, and several later blocks.

Interpretation:
- `f9872` has a transport-critical prefix, but unlike `f7056/f8976`, stale EO is not transport-fatal globally;
- the family exposes both transport-critical and semantic-only EO windows much more clearly.

## Refined clustering summary
| Family | Coarse transport-critical shape | Finer neutral/removable evidence |
|---|---|---|
| `f7056` | early critical prefix + critical tail | neutral windows at `1208..1448`, `4258..4476`, `5137..5204` |
| `f8976` | same topology as `f7056` | no neutral 1/16 block yet |
| `f9872` | smaller transport-critical prefix; stale EO transport-safe but wrong | neutral windows at `2862..3548` and `5090..5660` |

## What this means
1. EO oracle bytes are **not uniformly opaque**.
2. Transport-critical EO bytes are **structured and partially localizable**.
3. `f7056` and `f8976` continue to look like one topology class.
4. `f9872` remains a distinct topology class with more semantic-only removals.

## Next step
Refine only the transport-critical windows, not the whole EO stream:
- `f7056/f8976`: recurse on the early prefix + tail-critical windows
- `f9872`: recurse on the critical prefix only
- preserve discovered hash-neutral windows as candidate removable EO subsets for later synthesis attempts
