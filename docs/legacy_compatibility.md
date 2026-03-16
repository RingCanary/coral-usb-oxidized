# Legacy Compatibility

This document covers repo surfaces that remain useful, but are not the active repo path.

## Compatibility-Only Surfaces

- `legacy-runtime` Cargo feature
- delegate/TensorFlow Lite examples
- `edgetpu_compiler` bootstrap and compile pipelines
- Python helpers that inspect or patch compiled `*_edgetpu.tflite` artifacts
- archive-only scripts and tooling under `tools/archive/`

These exist for interoperability, historical reproduction, and artifact extraction. They should not be described as the native/custom control path.

## When To Use Them

Use the legacy stack only when you specifically need one of:

- a compiled reference artifact for reverse-engineering
- delegate interoperability checks
- old experiment reproduction
- compatibility validation against the historical Coral toolchain

## Non-Goals For The Active Path

Using any of the following in the active artifact-creation loop means Phase 4 completion has not been reached yet:

- `edgetpu_compiler`
- `libedgetpu`
- TensorFlow Lite delegate execution
- old TensorFlow/Python/Bazel build flows

## Current Legacy Entry Points

- `cargo run --features legacy-runtime --example ...`
- `cargo run --features legacy-runtime --example cpu_vs_edgetpu_mvp -- --help`
- `tools/archive/bootstrap_edgetpu_compiler.sh`
- `tools/archive/dense_template_pipeline.sh`
- `tools/archive/conv_template_pipeline.sh`
- `tools/archive/multiop_template_pipeline.sh`

Treat them as compatibility surfaces, not as the repo front door.
