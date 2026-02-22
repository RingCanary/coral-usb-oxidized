# RE Frontier Platform (Dense + Conv + Tiling)

Date: 2026-02-22

This note captures the next-stage exploration platform built on top of the
Dense template path.

## Scope in this increment

1. Faster Dense weight restriding in Rust (`set_weights_from_slice`).
2. Host-side batched execution helper (`PreparedDenseGemm::execute_batch_rows`).
3. Row-tiled large matrix-vector example using bundled `2688x2688` template.
4. New Conv2D template toolchain with `uv` + `edgetpu_compiler`.

## 1) Fast restride path

`DenseGemmTemplate::set_weights_from_slice` now writes payload bytes via a
tile-native loop (64x64 tiles with 4-lane inner ordering), instead of calling
`dense_param_offset` per element.

Validation is covered by unit test:

- `fast_restride_matches_formula_mapping`

The test generates a synthetic `128x128` matrix and verifies each payload byte
against the formula-based offset mapping.

## 2) Batch helper (host loop)

`PreparedDenseGemm` now provides:

- `execute_batch_rows(&self, inputs_row_major_q: &[i8]) -> Result<Vec<i8>, DenseGemmError>`

Behavior:

- Input is concatenated row-major vectors (`N * input_dim` bytes).
- Output is concatenated row-major vectors (`N * output_dim` bytes).
- Reuses one prepared interpreter/delegate path across all rows.

Note:

- This is a host-loop batch helper, not a compiled multi-batch graph.

## 3) Large-matrix tiling example

New example:

- `examples/gemm_tiled_rows.rs`

Purpose:

- Demonstrate row-tiling execution for matrices larger than a single on-chip
  parameter block by reusing `DenseGemmTemplate::from_bundled_2688()`.

Run:

```bash
cargo run --example gemm_tiled_rows -- 8192 identity_cycle 1
```

What it does:

1. Builds row tiles of size `2688x2688`.
2. Runs each tile sequentially on EdgeTPU.
3. Stitches first `row_block` outputs from each tile into the full result.

This provides a concrete path for workloads whose total parameter footprint
exceeds the ~7 MiB on-chip cache regime, while keeping each invoke in a known
compiled template envelope.

## 4) Conv2D template workflow

New generator:

- `tools/generate_conv2d_quant_tflite.py`

New pipeline:

- `tools/conv_template_pipeline.sh`

Example run:

```bash
./tools/conv_template_pipeline.sh \
  --height 224 --width 224 \
  --in-channels 3 --out-channels 16 \
  --kernel-size 3 --stride 1 --padding same
```

Pipeline output mirrors Dense flow:

- quantized `.tflite`
- compiled `*_edgetpu.tflite`
- DWN1 extraction
- executable parser text/json
- tensorizer inspect text/json
- optional benchmark log (`--run-benchmark`)

## Immediate next experiments

1. Benchmark `execute_batch_rows` throughput vs single-row `execute` for fixed
   weights and varying batch sizes.
2. Add Conv2D model classes (`1x1`, depthwise-like channel patterns) and compare
   transport signatures against Dense paths.
3. Extend tiling from row-only to full `M x K` block decomposition with
   calibrated partial-sum accumulation strategy.
4. Start a pure-Rust USB control/data-plane prototype behind a feature flag,
   reusing known register setup sequences from captured traces.

## Pure-Rust USB driver milestone plan

1. Baseline + contract capture:
   - map all current `libedgetpu` call sites and expected delegate behavior.
   - lock down expected state transitions (`1a6e:089a` -> `18d1:9302`).
2. Minimal `rusb` prototype:
   - open device, claim interface, and replay captured init transfers.
   - first validation: deterministic state transition + status response.
3. Command-path emulation:
   - implement core control/bulk command sequence in Rust with structured logs.
4. Integration phase:
   - add a feature-gated Rust delegate path compatible with `CoralInterpreter`.
5. Validation + docs:
   - hardware smoke tests against existing examples
   - record setup/runbook and failure modes in `WORKLOG.md`.
