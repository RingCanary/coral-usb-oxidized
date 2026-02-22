# Bundled Dense Templates

These EdgeTPU-compiled TFLite templates are bundled for zero-setup GEMM experiments:

- `dense_2048x2048_quant_edgetpu.tflite`
- `dense_2304x2304_quant_edgetpu.tflite`
- `dense_2688x2688_quant_edgetpu.tflite`

Generation provenance:

- Produced locally with `tools/dense_template_pipeline.sh`
- Fully INT8 single-layer Dense models
- Compiled by `edgetpu_compiler` and validated on Coral USB hardware

Rust API access is exposed from `src/gemm.rs` via:

- `TEMPLATE_2048`
- `TEMPLATE_2304`
- `TEMPLATE_2688`

Use `DenseGemmTemplate::from_bundled_2048()` / `from_bundled_2304()` / `from_bundled_2688()` to instantiate directly.
