# Phase 2 Dense (M5→M7) Status Update (2026-03-03)

## Scope
This update completes M5/M6/M7 tasks from `phase2-todo.md` for Dense-family profiling and controlled instruction semantics, with DUT validation on Pi5 + Coral.

## M5 — Multi-family profiling and coverage

### 1) Family transition function artifact
- Tooling:
  - `src/bin/family_transition_map.rs`
  - `scripts/m5_family_transition_map.sh`
- Run:
  - `traces/analysis/m5-family-transition-map-20260303T182605Z/`
- Key output:
  - recurrent family count = 4
  - recurrent families:
    - `eo7056_pc1840`
    - `eo7952_pc2096`
    - `eo8976_pc2352`
    - `eo9872_pc2608`

### 2) Third-member coverage for previously 2-member families
- `7056/1840`: added rectangular 3rd member `640x1280` (same EO/PC family)
  - search artifacts include:
    - `traces/analysis/m5-family-rect-scan-20260303T183223Z/`
    - `traces/analysis/m5-family-7056-exhaustive-o640-20260303T183712Z/`
    - `traces/analysis/m5-family-7056-exhaustive-o1280-20260303T184059Z/`
- `9872/2608`: added 3rd member `3072x2048`
  - artifacts:
    - `traces/analysis/m5-family-rect-scan-hi-20260303T183324Z/`
    - `traces/analysis/m5-family-9872-sweep-o2048-20260303T185235Z/`

### 3) Safe-core extraction and family profile artifacts
- Patchspec generation:
  - `scripts/m5_build_family_patchspecs.sh`
  - run: `traces/analysis/m5-family-patchspecs-20260303T185627Z/`
- Initial strict-safe synthesis showed EO/PC candidates for `f7056`, `f7952`, `f9872`.

### 4) DUT toxicity split (PC-safe vs EO-toxic)
- Run:
  - `traces/analysis/specv3-m5-patch-toxicity-matrix-20260303T190208Z/`
- Result pattern (all three families):
  - PC safe patch: PASS (baseline hash preserved)
  - EO safe patch: FAIL `UsbError(Timeout)`

### 5) Final validated profile matrix (4 recurrent families)
- Run:
  - `traces/analysis/specv3-m5-family-profile-dut-matrix-v2-20260303T190420Z/`
- Result:
  - all four family profiles PASS,
  - baseline hash == profile hash in every family:
    - `f7056`: `0x963548a7b7b20725`
    - `f7952`: `0x6ab05ef9aa8b9b25`
    - `f8976`: `0x67709fedfd103a2d`
    - `f9872`: `0x3ce2a859ce7ed025`

## M6 — Controlled instruction probes (single-axis)

### Probe setup
- Added:
  - `src/bin/instruction_chunk_diff.rs`
  - `scripts/m6_instruction_axis_probe.sh`
- Extended template generator/pipeline controls:
  - `tools/generate_dense_quant_tflite.py`: `--activation`, `--rep-offset`
  - `tools/dense_template_pipeline.sh`: passthrough for `--activation`, `--use-bias`, `--rep-offset`

### Axis run
- Run:
  - `traces/analysis/m6-instruction-axis-probe-20260303T190926Z/`
- Family under test: `8976/2352` at fixed `1792x1792`.
- EO results:
  - changed offsets = 14
  - signature histogram:
    - `activation+quantization`: 11
    - `quantization`: 3
  - concentrated near offsets `6810..6836`
- PC results:
  - changed offsets = 0 across quant/activation/bias probes

### DUT run for all axis variants
- Run:
  - `traces/analysis/specv3-m6-axis-variant-dut-matrix-20260303T191041Z/`
- All variants PASS transport/signature gates.
- Only `quant_offset_pos` changes output hash (`0x3c5a2635cd3a2e25`), others match baseline hash (`0x26b62556f7b52f25`).

## Cross-family semantic classification (M5+M6 synthesis)

| Class | Evidence | Current interpretation |
|---|---|---|
| Hardware-constant | large unchanged majority in EO/PC payloads across dim and axis sweeps | transport/scaffold bytes, not currently synthesis targets |
| Dim-scaling | PC strict-safe patches are transport-safe across recurrent families; EO/PC family lengths gate applicability | primary candidate set for safe in-family scaling |
| Family-specific/config | EO strict-safe candidates repeatedly timeout on DUT (new families + prior 8976 toxicity history); EO offsets around `6810..6836` move with activation/quantization | unresolved EO config region; keep anchored templates per family |

## M7 — Residual compiler dependency and decision

### Quantification snapshot
- PC safe-core rule counts (strict-safe, anchor-target synthesis):
  - `f7056`: 69 rules over payload 1840
  - `f7952`: 36 rules over payload 2096
  - `f8976`: 14-rule validated safe subset (baseline-equivalent PC14)
  - `f9872`: 60 rules over payload 2608
- EO strict-safe candidates for new families are currently transport-unsafe (timeout).

### Decision
- **Selected path:** minimal instruction-template dependency (not full parametric instruction generator yet).
- Practical dependency state:
  - parameter stream: compilerless (Rust packer)
  - instruction plane: keep per-family anchor templates; apply validated PC-safe subsets; EO remains anchored until further decode.

## M8
- Not started (intentionally gated on Dense M5–M7 stabilization).
