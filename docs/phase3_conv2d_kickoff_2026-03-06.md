# Phase 3 Conv2D Kickoff (2026-03-06)

## Critical evaluation of the starting plan
The deferred Conv2D plan was reviewed against the actual repository state before starting.

### What held up
- Existing tooling for single-op Conv2D is real:
  - `tools/generate_conv2d_quant_tflite.py`
  - `tools/conv_template_pipeline.sh`
  - `tools/conv_layout_probe.py`
- Prior evidence already suggested that **1x1 Conv2D channel mixing may reuse Dense-like inner packing** with a fixed parameter prefix.
- The repo already had a clean Dense lesson to apply: map families first, then test compilerless params, then test cross-dim EO dependence.

### What needed tightening
- The older Conv2D evidence was too narrow to justify a broad “general Conv2D” push.
- Mixed-op Conv2D->Dense tooling exists, but it is too confounded for the first serious Conv2D phase.
- Some Conv2D tool entrypoints were archived/shimmed and had not been revalidated recently.

### Phase-3 decision
The phase starts with a deliberately narrow scope:
- **single-op** Conv2D
- **1x1** kernel
- **stride 1**
- **padding same**
- **no bias**

Multi-op, `k>1`, depthwise, and activation/bias widening stay deferred until this base path is bounded.

## Kickoff artifacts produced

### 1) Local anchor compile/extract
Artifact:
- `traces/analysis/phase3-conv2d-kickoff-anchor-20260306T125400Z/`

Anchor model:
- input: `1x32x32x64`
- Conv2D: `64` filters, `1x1`, stride `1`, padding `same`, bias `off`

Key extracted facts:
- EO instruction bytes: `5360`
- PC instruction bytes: `688`
- parameter bytes: `4608`
- input bytes: `65536`
- output bytes: `65536`

Executable split:
- `EXECUTION_ONLY`: `5360` instruction bytes
- `PARAMETER_CACHING`: `688` instruction bytes + `4608` parameter bytes

### 2) First DUT anchor baseline
Artifact:
- `traces/analysis/specv3-phase3-conv2d-anchor-baseline-20260306T125446Z/`

Result:
- PASS
- output bytes: `65536`
- output hash: `0x1f518203ddfe8154`

This establishes the first stable Conv2D replay signature on Pi5 + Coral for the new phase.

### 3) Initial 1x1 Conv2D family bootstrap
Added helper:
- `scripts/phase3_conv2d_family_bootstrap.sh`

Artifact:
- `traces/analysis/phase3-conv2d-family-bootstrap-20260306T125706Z/`

Cases scanned:
- `16x16x32 -> 32`
- `32x32x32 -> 32`
- `32x32x64 -> 64`
- `32x32x64 -> 128`
- `32x32x128 -> 64`
- `32x32x128 -> 128`
- `64x64x64 -> 64`

Key result:
- in this initial 1x1 sweep, **EO and PC instruction sizes stayed fixed**:
  - `EO = 5360`
  - `PC = 688`
- only parameter bytes changed across the scanned cases:
  - `1280`, `4608`, `8704`, `9216`, `17408`

Initial family summary:
- `eo5360_pc688_param1280`: `16x16x32->32`, `32x32x32->32`
- `eo5360_pc688_param4608`: `32x32x64->64`, `64x64x64->64`
- `eo5360_pc688_param8704`: `32x32x128->64`
- `eo5360_pc688_param9216`: `32x32x64->128`
- `eo5360_pc688_param17408`: `32x32x128->128`

Interpretation:
- for the scanned 1x1 cases, spatial size changes and channel-size changes did **not** change EO/PC instruction payload lengths,
- which makes 1x1 Conv2D look structurally friendlier than Dense at the first family-mapping pass.

This is only a bootstrap result, not a full family law.

### 4) Revalidated 1x1 layout clue with the current toolchain
Artifact:
- `traces/analysis/phase3-conv2d-layout-probe-20260306T125758Z/`

Before running it, a stale archived-path bug had to be fixed:
- `tools/archive/conv_layout_probe.py`
  - corrected repo-root resolution when invoked via the top-level shim

Current 1x1 `64x64` probe still supports the older layout clue:
- parameter region size: `4608`
- candidate mapping examples reproduced:
  - `(ic=0, oc=1) -> 516`
  - `(ic=1, oc=0) -> 513`
  - `(ic=31, oc=31) -> 2431`
  - `(ic=63, oc=63) -> 4607`

This remains consistent with the prior interpretation:
> 1x1 Conv2D channel mixing appears Dense-like in the inner lane order, with a fixed `512`-byte prefix before the channel-mixing payload.

Still, this is not yet a general packing proof; it only justifies moving to wider `32/64/128` validation next.

## Current critical assessment

### What is encouraging
1. Conv2D 1x1 already has a stable DUT anchor baseline.
2. The initial 1x1 scan suggests EO/PC size stability across the first tested spatial/channel changes.
3. The old layout clue survives current-toolchain revalidation.

### What is still unknown
1. Whether the `Dense-like + 512-byte prefix` packing rule holds across more channel regimes.
2. Whether compilerless param generation can be ported cleanly into Rust and match DUT hashes.
3. Whether cross-dim target-equivalent replay for Conv2D also collapses specifically to an EO target-state dependency.

### Main risk
The current evidence is good enough to start, but still too narrow to claim that 1x1 Conv2D will follow the same dependency story as Dense.
That must be tested explicitly; it should not be assumed.

## Next step
Proceed in this order:
1. widen layout validation to more `Cin/Cout` regimes,
2. port compilerless 1x1 Conv2D parameter packing into Rust,
3. run the Conv2D cross-dim oracle matrix before doing any EO ablation.
