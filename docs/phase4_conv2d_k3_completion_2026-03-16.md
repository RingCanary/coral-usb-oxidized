# Phase 4 Conv2D `k=3` bounded completion (2026-03-16)

## Goal
Close Phase 4 for the deliberately bounded family:

- single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- same-product spatial family with `EO=6512`
- anchor shape `16x64`
- target shapes `32x32`, `64x16`, `128x8`
- channel regimes `p32/p64/p128`

Completion means:

> no `edgetpu_compiler` in the active loop, no legacy runtime in the active loop, pure-`rusb` replay on Pi, and target-hash reproduction from native params + native EO.

## New active-path pieces

### Frozen family assets
- `templates/phase4_conv2d_k3_sameprod_6512/family.json`
- `templates/phase4_conv2d_k3_sameprod_6512/SUMMARY.txt`
- curated anchor/target seed models under `templates/phase4_conv2d_k3_sameprod_6512/`

These are the bounded checked-in seed artifacts for the active completion path.

### Offline freeze helper
- `scripts/phase4_conv2d_k3_freeze_family_assets.sh`

This is the one offline compiler-assisted step used to curate the bounded family tables and seed models. It is not part of the active replay/materialization loop.

### Native EO emitter
- `src/bin/conv_k3_eo_emit.rs`

This emits the EO patchspec for a requested bounded-family target from the frozen repo-native spec. In the current completion path, all supported targets are emitted from bounded native tables.

### One-command completion runner
- `scripts/phase4_conv2d_k3_completion_demo.sh`

This script:

1. resolves the bounded family target,
2. materializes target params natively from the uncompiled target `.tflite`,
3. emits target EO from the frozen native family spec,
4. syncs the current source tree and case files to Pi,
5. runs target baseline and native completion replay,
6. verifies target-hash equivalence.

## Verification artifact
- `traces/analysis/phase4-conv2d-k3-completion-demo-20260316T065217Z/`

The bounded-family matrix passed in all `9` cases:

- `p32`: `32x32`, `64x16`, `128x8`
- `p64`: `32x32`, `64x16`, `128x8`
- `p128`: `32x32`, `64x16`, `128x8`

Each native run matched the compiled target baseline hash exactly.

Representative hashes:

- `p32 h32`: `0xa6e87a2685fa95a5`
- `p64 h64`: `0x606ffcd9fe25c9b5`
- `p128 h128`: `0xa9049eadd42ddd61`

See `SUMMARY.txt` inside the artifact for the full matrix.

## What is now complete

For this bounded family, the active path now has:

- pure-`rusb` runtime execution
- native target parameter emission from uncompiled TFLite
- native EO emission from repo-native bounded family tables
- Pi replay that reproduces the target hash
- no `edgetpu_compiler` in the active loop
- no TensorFlow/Bazel/legacy runtime in the active loop

This is a real bounded completion milestone, not just “compiler-assisted artifact control.”

## What this completion is not

This does **not** prove:

- a formulaic EO generator for the family
- portability outside the `EO=6512` same-product family
- support for `8x128` (already outside the family)
- support for other kernels, stride changes, padding changes, bias, or multi-op graphs

The current EO emitter is intentionally table-driven for the bounded family. That is acceptable for Phase 4 completion because the active path is compiler-free even though the frozen family spec was curated offline.

## Next frontier

Phase 4 should now be treated as closed for the bounded family.

The next phase should start with generalization pressure, not more Phase 4 minimization:

1. cross the `6512 -> 6496` family boundary (`8x128`),
2. test whether the bounded EO tables factor into a smaller reusable law,
3. only then widen beyond this single-op `k=3` same-product regime.
