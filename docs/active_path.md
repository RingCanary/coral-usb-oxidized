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
- `docs/phase5_conv2d_k3_h8_band_completion_2026-03-16.md`
- `docs/phase5_conv2d_k3_6496_h8_band_2026-03-16.md`
- `docs/phase5_conv2d_k3_6496_boundary_scan_2026-03-16.md`
- `docs/phase4_completion_control_plan_2026-03-07.md`
- `templates/phase4_conv2d_k3_sameprod_6512/family.json`
- `templates/phase5_conv2d_k3_h8_band_6496/family.json`
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

cargo run --bin conv_k3_eo_emit -- --family-spec templates/phase5_conv2d_k3_h8_band_6496/family.json --channels 64 --target-height 8 --target-width 140 --out-report /tmp/conv_k3_eo_emit_phase5.json
bash scripts/phase4_conv2d_k3_completion_demo.sh --family-spec templates/phase5_conv2d_k3_h8_band_6496/family.json --pairs p32,p64,p128 --target-height 8 --target-widths 104,116,128,140,152
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

Bounded Phase 5 completion is now also achieved for:

- single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- mixed-product `fixed_height=8` family
- frozen widths `104,116,128,140,152`
- symmetric `p32/p64/p128`
- pure-`rusb` replay on Pi
- no `edgetpu_compiler` in the active artifact-creation loop

The active frontier is now beyond the first `6496` family closure:

- widen the frozen `H=8` width set
- compress lookup residue further toward a more reusable law
- test whether the mixed-product family model extends beyond `fixed_height=8`

## Compatibility Boundary

If you need delegate examples, `edgetpu_compiler`, or TensorFlow Lite interoperability, use `docs/legacy_compatibility.md`.
