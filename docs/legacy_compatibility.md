# Legacy Compatibility

This document covers repo surfaces that remain useful, but are not the active Phase 4 path.

## Compatibility-Only Surfaces

- `legacy-runtime` Cargo feature
- delegate/TensorFlow Lite examples
- `edgetpu_compiler` bootstrap and compile pipelines
- Python helpers that inspect or patch compiled `*_edgetpu.tflite` artifacts
- archive-forwarding tool shims under `tools/`

These exist for interoperability, historical reproduction, and artifact extraction. They should not be described as the native/custom control path.

## When To Use Them

Use the legacy stack only when you specifically need one of:

- a compiled reference artifact for reverse-engineering
- delegate interoperability checks
- old experiment reproduction
- compatibility validation against the historical Coral toolchain

## Non-Goals For Phase 4

Using any of the following in the active artifact-creation loop means Phase 4 completion has not been reached yet:

- `edgetpu_compiler`
- `libedgetpu`
- TensorFlow Lite delegate execution
- old TensorFlow/Python/Bazel build flows

## Current Legacy Entry Points

- `cargo run --features legacy-runtime --example ...`
- `tools/bootstrap_edgetpu_compiler.sh`
- `tools/dense_template_pipeline.sh`
- `tools/conv_template_pipeline.sh`
- `tools/multiop_template_pipeline.sh`

Treat them as compatibility surfaces, not as the repo front door.
