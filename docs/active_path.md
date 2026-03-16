# Active Path

This is the active front door.

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

Those remain compatibility or archival surfaces outside the active bounded completion path.

## Start Here

For current status and frontier:

- `docs/phase4_conv2d_k3_completion_2026-03-16.md`
- `docs/phase5_conv2d_k3_6496_boundary_scan_2026-03-16.md`
- `docs/phase4_completion_control_plan_2026-03-07.md`
- `templates/phase4_conv2d_k3_sameprod_6512/family.json`
- `WORKLOG.md`

For active runtime smoke checks:

```bash
cargo check --lib
cargo run --example rusb_control_plane_probe -- --verbose-configs
cargo run --example rusb_serialized_exec_replay -- --help
```

For active bounded completion helpers:

```bash
cargo run --bin conv_k3_eo_emit -- --family-spec templates/phase4_conv2d_k3_sameprod_6512/family.json --channels 64 --target-height 64 --out-report /tmp/conv_k3_eo_emit.json
bash scripts/phase4_conv2d_k3_completion_demo.sh
```

## Current Bounded Status

Bounded Phase 4 completion is now achieved for:

- single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- same-product spatial moves
- pure-`rusb` replay on Pi
- no `edgetpu_compiler` in the active artifact-creation loop

The current frontier is no longer bounded-family completion. It is family-boundary discovery beyond this frozen family.

The current Phase 5 result is narrower than hoped:

- `8x128` is confirmed as `EO=6496`
- on the scanned p64 power-of-two same-product axis, it has no second `6496` partner yet
- so the active loop remains intentionally frozen at the checked-in `6512` family until a nontrivial `6496` family is discovered

## Compatibility Boundary

If you need delegate examples, `edgetpu_compiler`, or TensorFlow Lite interoperability, use `docs/legacy_compatibility.md`.
