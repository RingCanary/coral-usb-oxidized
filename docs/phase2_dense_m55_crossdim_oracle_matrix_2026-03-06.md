# Phase 2 Dense M5.5 — Cross-Dim Oracle Matrix (2026-03-06)

## Objective
Stress-test the precise claim boundary for compilerless Dense deployment:

> Given a compiled anchor in the same EO/PC-length family, can we replay a different dimension without compiler-derived EO bytes?

We tested this on Pi5 + Coral using same-family **transpose pairs with equal parameter-stream length**, so the target parameter stream can be reused exactly.

## Method

### Families and transpose pairs
- `f7056`: `640x1280` -> `1280x640`
- `f7952`: `768x1536` -> `1536x768`
- `f8976`: `896x1792` -> `1792x896`
- `f9872`: `1024x2048` -> `2048x1024`

All models were rebuilt with non-degenerate weights:
- `init_mode=random_uniform`
- `seed=1337`

This supersedes an earlier all-zero-output transport-only matrix.

### Tooling added
- `src/bin/instruction_chunk_patchspec.rs`
  - exact serialized-executable instruction diff -> patchspec
- `src/bin/model_param_stream_dump.rs`
  - dump exact parameter stream from a compiled model
- `scripts/m5_crossdim_oracle_matrix.sh`
  - compile families, generate EO/PC oracle patchspecs, sync current source tree to Pi, run DUT matrix, summarize results

### Cases per family
1. `target_baseline`
2. `target_override` (target model + exact dumped target param stream)
3. `anchor_param_only` (anchor model + target param stream, no instruction patches)
4. `anchor_pc_oracle` (anchor model + exact target PC bytes only)
5. `anchor_eo_oracle` (anchor model + exact target EO bytes only)
6. `anchor_eopc_oracle` (anchor model + exact target EO+PC bytes)

## Artifacts
- Main run:
  - `traces/analysis/m5-crossdim-oracle-matrix-20260306T103420Z/`
- DUT logs:
  - `traces/analysis/specv3-m5-crossdim-oracle-matrix-20260306T103420Z-dut/`

## Results

### Summary table
| Family | target_override | anchor_param_only | anchor_pc_oracle | anchor_eo_oracle | anchor_eopc_oracle |
|---|---:|---:|---:|---:|---:|
| `f7056` | PASS, hash==target | FAIL `UsbError(Overflow)` | FAIL `UsbError(Overflow)` | PASS, hash==target | PASS, hash==target |
| `f7952` | PASS, hash==target | FAIL `UsbError(Overflow)` | FAIL `UsbError(Overflow)` | PASS, hash==target | PASS, hash==target |
| `f8976` | PASS, hash==target | FAIL `UsbError(Overflow)` | FAIL `UsbError(Overflow)` | PASS, hash==target | PASS, hash==target |
| `f9872` | PASS, hash==target | PASS, **hash!=target** | PASS, **hash!=target** | PASS, hash==target | PASS, hash==target |

### Key observations
1. `target_override` passes in all families.
   - Confirms exact target parameter stream extraction/injection is sound.
2. No family reaches **target-equivalent** cross-dim replay without EO target bytes.
   - `7056/7952/8976`: stale-EO runs are transport-fatal (`Overflow`)
   - `9872`: stale-EO runs are transport-safe but semantically wrong (hash drift)
3. `anchor_pc_oracle` never rescues stale EO.
   - PC exact target bytes alone are insufficient in every family.
4. `anchor_eo_oracle` is already sufficient for target-equivalent replay in every tested family.
   - Adding PC exact target bytes changes nothing further for these same-product transpose pairs.

## Interpretation
This sharpens the instruction-plane dependency boundary:

- **Parameters:** compilerless and exact
- **PC:** non-semantic under M6 axis probes, but also **not sufficient** for cross-dim target equivalence here
- **EO:** decisive cross-dim plane
  - stale EO causes either transport failure or semantic drift
  - exact target EO bytes are sufficient for target-equivalent replay in all tested transpose pairs

So the practical blocker is more precise than “instruction synthesis in general”:

> For same-family cross-dim Dense replay, the unresolved dependency is specifically the EO run-phase instruction state.

## Implication for Phase 2
M5.5 is complete as a **negative proof**:
- cross-dim deployment **without compiler-derived EO target bytes is not yet available**.

This keeps the Phase-2 decision on the minimal-template path, but with tighter wording:
- the residual compiler dependency is the **EO anchor/template per target dimension family member**,
- not the PC plane for these tested same-product transpose moves.
