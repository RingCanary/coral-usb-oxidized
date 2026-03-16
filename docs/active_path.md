# Active Path

This is the Phase 4 front door.

The active path in this repo is:

- pure-`rusb` runtime execution
- native Rust reverse-engineering and materialization helpers
- bounded checked-in seed artifacts only when unavoidable
- Pi replay helpers under `scripts/`

The active path is **not**:

- `libedgetpu`
- TensorFlow Lite delegate execution
- `edgetpu_compiler`
- legacy Python/Bazel stacks

Those remain compatibility or archival surfaces until Phase 4 native artifact generation is complete.

## Start Here

For current status and frontier:

- `docs/phase4_completion_control_plan_2026-03-07.md`
- `docs/phase4_conv2d_k3_scout_2026-03-06.md`
- `docs/phase4_conv2d_k3_crossdim_oracle_matrix_2026-03-06.md`
- `docs/phase4_conv2d_k3_param_region_2026-03-07.md`
- `docs/phase4_conv2d_k3_native_param_materialize_2026-03-07.md`
- `docs/phase4_conv2d_k3_eo_localization_2026-03-07.md`
- `WORKLOG.md`

For active runtime smoke checks:

```bash
cargo check --lib
cargo run --example rusb_control_plane_probe -- --verbose-configs
cargo run --example rusb_serialized_exec_replay -- --help
```

For active Phase 4 helpers:

```bash
bash scripts/phase4_conv2d_k3_family_scout.sh
bash scripts/phase4_conv2d_k3_crossdim_oracle_matrix.sh
bash scripts/phase4_conv2d_k3_param_region_probe.sh
```

## Current Bounded Completion Target

Phase 4 completion currently means:

- single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- same-product spatial moves
- pure-`rusb` replay on Pi
- no `edgetpu_compiler` in the active artifact-creation loop

## Compatibility Boundary

If you need delegate examples, `edgetpu_compiler`, or TensorFlow Lite interoperability, use `docs/legacy_compatibility.md`.
