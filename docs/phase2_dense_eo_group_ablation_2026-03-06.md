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

## High-level findings

### Family `f7056` (`640x1280 -> 1280x640`)
8-way split pattern:
- removing groups `g00`, `g01`, `g02`, or `g07` is transport-fatal
- removing groups `g03`, `g04`, `g05`, `g06` remains transport-safe
- removing `g03` is **hash-neutral** (`eo_minus_g03` still matches target)
- `g07` alone is transport-safe but semantically wrong

Interpretation:
- coarse transport-critical EO state is concentrated in the early ranges plus the tail block,
- while at least one middle block (`g03`) is removable at this granularity.

### Family `f8976` (`896x1792 -> 1792x896`)
8-way split pattern is very similar to `f7056`:
- removing groups `g00`, `g01`, `g02`, or `g07` is transport-fatal
- removing groups `g03`, `g04`, `g05`, `g06` remains transport-safe
- none of the removable 1/8 groups are hash-neutral alone
- `g07` alone is transport-safe but semantically wrong

Interpretation:
- `f7056` and `f8976` share a common coarse EO topology:
  - early blocks + tail block are transport-critical,
  - middle blocks are transport-safe but semantically active.

### Family `f9872` (`1024x2048 -> 2048x1024`)
This family differs from the overflow-fatal families.

Baseline behavior:
- stale EO (`anchor_param_only`) is already transport-safe but semantically wrong.

8-way split:
- removing `g00` or `g01` is transport-fatal
- removing `g02`, `g04`, `g05`, `g06`, `g07` remains transport-safe but changes hash
- removing `g03` remains transport-safe **and hash-neutral** (`eo_minus_g03` matches target)
- `g03` alone and `g07` alone are transport-safe but semantically wrong

Interpretation:
- `f9872` has a smaller transport-critical prefix,
- plus at least one removable neutral block (`g03`) at this granularity.

## Coarse clustering summary
| Family | Transport-fatal when removed | Transport-safe when removed | Hash-neutral removable block |
|---|---|---|---|
| `f7056` | `g00,g01,g02,g07` | `g03,g04,g05,g06` | `g03` |
| `f8976` | `g00,g01,g02,g07` | `g03,g04,g05,g06` | none at 1/8 split |
| `f9872` | `g00,g01` | `g02,g03,g04,g05,g06,g07` | `g03` |

## What this means
1. EO oracle bytes are **not uniformly opaque**.
   - coarse removable regions exist.
2. Transport-critical EO bytes are **not randomly scattered**.
   - there is observable block structure.
3. The transport-critical topology differs by family.
   - `7056/8976` share one pattern,
   - `9872` is less transport-fragile.

## Next step
Refine the transport-critical coarse blocks recursively:
- start with `f7056/f8976` blocks `g00,g01,g02,g07`
- start with `f9872` blocks `g00,g01`
- within removable blocks, search for smaller hash-neutral subsets before attempting any EO template rule synthesis.
