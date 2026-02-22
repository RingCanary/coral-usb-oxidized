# Coral USB Oxidized

Rust SDK/driver layer for Google Coral USB Accelerator discovery, delegate creation, and TensorFlow Lite C API interop.

## What this crate provides

- Coral USB detection (`1a6e:089a` and `18d1:9302`)
- EdgeTPU delegate creation through `edgetpu_c.h`
- TensorFlow Lite C API wrappers for model/interpreter/tensor operations
- Example programs for device verification and delegate/TFLite flows

## Raspberry Pi 5 setup

### 1) System packages

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl build-essential pkg-config libusb-1.0-0-dev clang llvm-dev libclang-dev gnupg
```

### 2) Install EdgeTPU runtime (modern apt keyring flow)

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/coral-edgetpu.gpg
echo "deb [signed-by=/etc/apt/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list >/dev/null
sudo apt-get update
sudo apt-get install -y libedgetpu1-std libedgetpu-dev
```

### 3) Install TensorFlow Lite runtime/dev library

```bash
sudo apt-get install -y libtensorflow-lite-dev
```

On Debian/Raspberry Pi OS this typically installs `libtensorflow-lite.so` under `/usr/lib/aarch64-linux-gnu`.

### 4) USB permissions (required for non-root delegate creation)

```bash
cat <<'EOF' | sudo tee /etc/udev/rules.d/71-edgetpu.rules >/dev/null
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1a6e", ATTRS{idProduct}=="089a", MODE="0664", GROUP="plugdev", TAG+="uaccess"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", MODE="0664", GROUP="plugdev", TAG+="uaccess"
EOF
sudo usermod -aG plugdev "$USER"
sudo udevadm control --reload-rules
sudo udevadm trigger
```

You need a new login session after `usermod -aG`.

### 5) Build and smoke test

```bash
cargo check --lib
cargo run --example basic_usage
cargo run --example simple_delegate
cargo run --example tflite_test
```

## Linking behavior

The build script checks these library locations:

- `/usr/lib`
- `/usr/local/lib`
- `/usr/lib/x86_64-linux-gnu`
- `/usr/lib/aarch64-linux-gnu`
- `/usr/lib/arm-linux-gnueabihf`

Overrides:

- `CORAL_LIB_DIR`
- `EDGETPU_LIB_DIR`
- `TFLITE_LIB_DIR`
- `TFLITE_LINK_LIB` (explicitly choose link name, for example `tensorflowlite_c` or `tensorflow-lite`)

By default, TensorFlow Lite linking prefers `libtensorflowlite_c.so` when present, otherwise falls back to distro naming (`libtensorflow-lite.so`).

## Device behavior

Coral USB commonly appears as:

- Initial: `1a6e:089a`
- After delegate/init/inference: `18d1:9302`

Both IDs are expected and should be included in udev rules.

## Examples

```bash
cargo run --example basic_usage
cargo run --example verify_device
cargo run --example delegate_usage
cargo run --example simple_delegate
cargo run --example tflite_test
cargo run --example tflite_standard_example
cargo run --example cpu_vs_edgetpu_mvp -- --help
cargo run --example gemm_int8 -- <dense_template_edgetpu.tflite> shift_plus1 ramp
cargo run --example gemm_int8_dynamic -- <dense_template_edgetpu.tflite> <input_dim> <output_dim> identity ramp
cargo run --example gemm_int8_bundled -- 2688 identity 30
cargo run --example gemm_tiled_rows -- 8192 identity_cycle 1
cargo run --example transformer_linear_block -- 8 5 1
cargo run --example transformer_linear_block -- 16 3 1 --no-attention --weight-source f32
cargo run --example gemm_weight_load_verify -- 8 3 1 2
cargo run --example clip_vit_safetensors_report -- /path/to/model.safetensors 0 127
cargo run --example clip_vit_layer_tpu_probe -- /path/to/model.safetensors /path/to/template_edgetpu.tflite 0 q 20 127
cargo run --example clip_vit_block_tpu_pipeline -- /path/to/model.safetensors /path/to/template_768x768_edgetpu.tflite /path/to/template_768x3072_edgetpu.tflite /path/to/template_3072x768_edgetpu.tflite 0 8 3 1 32
cargo run --example clip_vit_full_forward -- /path/to/model.safetensors /path/to/template_768x768_edgetpu.tflite /path/to/template_768x3072_edgetpu.tflite /path/to/template_3072x768_edgetpu.tflite --max-layers 12 --out-norm-f32le /tmp/clip_embed_norm.f32le
cargo run --example function_gemma_layer_tpu_probe -- /path/to/model.safetensors /path/to/template_edgetpu.tflite 0 q 20 32 100
cargo run --example function_gemma_lm_head_sanity -- /path/to/model.safetensors 42 10
cargo run --example function_gemma_decode_loop -- /path/to/model.safetensors /path/to/functiongemma-templates-b1 2,2516,29901 --steps 8 --lm-head coral --lm-template /path/to/dense_640x2624_quant_edgetpu.tflite
```

## Offline EdgeTPU package extractor

Use the standalone extractor to inspect `DWN1` package(s) embedded inside
`*_edgetpu.tflite` files and dump serialized executable blobs.

```bash
python3 tools/extract_edgetpu_package.py extract \
  /tmp/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
  --out /tmp/edgetpu_extract
```

Outputs include:

- `/tmp/edgetpu_extract/metadata.json`
- `/tmp/edgetpu_extract/package_000/serialized_multi_executable.bin`
- `/tmp/edgetpu_extract/package_000/serialized_executable_000.bin` (and more)

Quick self-test (skips if default model is absent):

```bash
python3 tools/extract_edgetpu_package.py self-test
```

Parse extracted executables into schema-aware summaries:

```bash
python3 tools/parse_edgetpu_executable.py /tmp/edgetpu_extract/package_000
```

This decodes instruction chunk sizes, relocation metadata (`field_offsets`),
layer metadata, and parameter payload sizes from `serialized_executable_*.bin`.

## USB tracing toolkit

For protocol-level and syscall-level capture helpers, use:

- `tools/usbmon_capture.sh` (root, kernel usbmon capture)
- `tools/usb_syscall_trace.sh` (unprivileged `strace` fallback)
- `tools/usbmon_phase_report.py` (phase-oriented usbmon report and diff)
- `tools/usbmon_register_map.py` (usbmon control/register extraction and run matrix)
- `tools/usbmon_bulk_signature.py` (bulk payload header/signature extraction by phase)
- `tools/usbmon_three_stage_signature.py` (dedicated 3-stage bulk loop signature parser)
- `tools/parse_edgetpu_executable.py` (schema-aware parser for serialized executables)
- `tools/tensorizer_patch_edgetpu.py` (in-place parameter patcher for compiled `*_edgetpu.tflite`)
- `tools/generate_dense_quant_tflite.py` (single-layer Dense INT8 model generator)
- `tools/generate_conv2d_quant_tflite.py` (single-layer Conv2D INT8 model generator)
- `tools/generate_dense_conv_quant_tflite.py` (Conv2D->Dense INT8 multi-op model generator)
- `tools/bootstrap_edgetpu_compiler.sh` (local `edgetpu_compiler` bootstrap from Coral apt repo)
- `tools/dense_template_pipeline.sh` (generate -> compile -> extract -> parse -> inspect pipeline)
- `tools/conv_template_pipeline.sh` (Conv2D generate -> compile -> extract -> parse -> inspect pipeline)
- `tools/multiop_template_pipeline.sh` (Conv2D->Dense generate -> compile -> extract -> parse -> inspect pipeline)
- `tools/dense_layout_probe.py` (single-hot parameter-layout probe and offset mapping extractor)
- `tools/conv_layout_probe.py` (single-hot Conv2D parameter-layout probe and offset candidate extraction)
- `tools/dense_template_matrix_patch.py` (structured Dense matrix patcher using recovered layout map)
- `tools/dense_quant_value_probe.py` (single-hot float->quant->compiled byte mapping verifier)
- `tools/strace_usb_scaling.py` (USBDEVFS submit/reap scaling fit from strace summaries)
- `tools/edgetpu_delegate_smoke.sh` (minimal delegate exercise without TensorFlow Lite C libs)
- `examples/inference_dump.rs` (single-invoke deterministic output dump for tensorizer validation)
- `examples/gemm_int8.rs` (Rust-native template patch + execute + verification loop)
- `examples/gemm_int8_dynamic.rs` (dimension-aware Rust template patch + prepared execution loop)
- `examples/gemm_int8_bundled.rs` (runs bundled 2048/2304/2688 templates with no external model path)
- `examples/gemm_tiled_rows.rs` (row-tiled matrix-vector execution beyond a single 2688x2688 parameter block)
- `examples/transformer_linear_block.rs` (six-stage `2304x2304` transformer-like block benchmark with stage timing and model-switch baseline)
- `examples/gemm_weight_load_verify.rs` (f32 weight/input quantize->patch->execute bridge with CPU reference verification)
- `examples/clip_vit_safetensors_report.rs` (CLIP ViT SafeTensors parser + layer mapping/quantization preflight)
- `examples/clip_vit_layer_tpu_probe.rs` (patch a real CLIP ViT layer into a rectangular template and execute on TPU)
- `examples/clip_vit_block_tpu_pipeline.rs` (full CLIP ViT layer linear-stage pipeline: q/k/v/o/fc1/fc2 with per-stage timing + affine verification)
- `examples/clip_vit_full_forward.rs` (patch embedding + transformer blocks + projection, with Coral-backed linear stages and optional reference compare)
- `examples/function_gemma_layer_tpu_probe.rs` (Function-Gemma BF16 stage loader + quantize/patch/execute probe for q/k/v/o/gate/up/down)
- `examples/function_gemma_lm_head_sanity.rs` (CPU embedding lookup + tied LM-head top-k sanity pass for Function-Gemma checkpoints)
- `examples/function_gemma_decode_loop.rs` (autoregressive decode loop with Coral-backed q/k/v/o/gate/up/down and optional Coral-tiled LM-head)

Detailed workflow and caveats are documented in `docs/usb_tracing.md`.

Current reverse-engineering notes:

- `WORKLOG.md`
- `docs/usb_invoke_scaling_by_model.md`
- `docs/next_usbmon_capture_matrix.md`
- `docs/usb_register_map_candidates.md`
- `docs/usb_executable_transport_correlation.md`
- `docs/tensorizer_mvp.md`
- `docs/tensorizer_dense_template.md`
- `docs/research_frontier_platform.md`
- `docs/multiop_conv_dense.md`
- `docs/conv_layout_probe.md`
- `docs/dense_layout_probe.md`
- `docs/transformer_linear_block.md`
- `docs/gemm_weight_load_verify.md`
- `docs/clip_vit_safetensors_report.md`
- `docs/clip_vit_layer_tpu_probe.md`
- `docs/clip_vit_block_tpu_pipeline.md`
- `docs/clip_vit_full_forward.md`
- `docs/function_gemma_layer_tpu_probe.md`
- `docs/function_gemma_lm_head_sanity.md`
- `docs/function_gemma_decode_loop.md`
- `docs/schema/libedgetpu_executable.fbs`
- `docs/external_research_2026-02-21.md`
- `traces/re-matrix-20260221T092342Z/USBMON_PACKET_VALIDATION_20260221T1035Z.md`

## Arch bootstrap (local prefix)

If distro/AUR packages are out of sync, build the runtime stack into
`$HOME/.local`:

```bash
./tools/bootstrap_arch_stack.sh build-libedgetpu
./tools/bootstrap_arch_stack.sh build-tflite-c
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
```

Bootstrap `edgetpu_compiler` locally (no distro package required):

```bash
./tools/bootstrap_edgetpu_compiler.sh install
eval "$(./tools/bootstrap_edgetpu_compiler.sh print-env)"
```

### Single-layer Dense template workflow (uv-managed)

Generate a compiler-compatible single-op Dense template and run extraction +
schema parsing in one command:

```bash
./tools/dense_template_pipeline.sh --patch-mode zero
```

Notes:

- Pipeline defaults to `python 3.9` + `tensorflow-cpu 2.10.1` via `uv`.
- This converter/runtime pairing is currently required for reliable
  `edgetpu_compiler` acceptance of minimal Dense templates on this repo setup.

Quick output check on hardware:

```bash
latest="$(ls -1dt traces/dense-template-* | head -n1)"
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
cargo run --example inference_dump -- "$latest/dense_256x256_quant_edgetpu.tflite" ramp
cargo run --example inference_dump -- "$latest/dense_256x256_quant_edgetpu_patched_zero.tflite" ramp
```

### Conv2D template workflow (uv-managed)

Generate a single-op Conv2D template and run extraction + parser in one command:

```bash
./tools/conv_template_pipeline.sh --height 224 --width 224 --in-channels 3 --out-channels 16 --kernel-size 3 --stride 1 --padding same
```

This provides a reproducible Conv2D RE path parallel to Dense-template work, including executable extraction and schema-aware parser output for transport/parameter analysis.

### Conv2D layout probe workflow

Recover candidate payload offsets for Conv2D single-hot kernel coordinates:

```bash
./tools/conv_layout_probe.py --height 32 --width 32 --in-channels 64 --out-channels 64 --kernel-size 1
```

### Multi-op Conv2D + Dense workflow

Generate and compile a chained Conv2D->Dense model (useful stepping stone toward
microgpt-adjacent mixed-layer acceleration experiments):

```bash
./tools/multiop_template_pipeline.sh --run-benchmark
```

### Rust-native Dense GEMM template path

Use the recovered 256x256 layout and in-memory patch path directly from Rust
(no Python patcher required for weight replacement):

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
latest="$(ls -1dt traces/dense-template-* | head -n1)"
cargo run --example gemm_int8 -- \
  "$latest/dense_256x256_quant_edgetpu.tflite" shift_plus1 ramp
```

Library API entry points:

- `DenseGemmTemplate` (DWN1 parameter-region discovery + dimension-aware patch/execute path)
- `PreparedDenseGemm` (dimension-aware prepared executor; includes host-loop `execute_batch_rows`)
- `dense_256_param_offset` (recovered restride formula)
- `dense_param_offset` (dimension-aware restride mapping)
- `TEMPLATE_2048`, `TEMPLATE_2304`, `TEMPLATE_2688` (bundled precompiled templates via `include_bytes!`)
- `DenseGemmTemplate::from_bundled_2048/2304/2688()` (zero-path constructors)

### Transformer-like linear block milestone (2304)

Run the six-stage `2304x2304` benchmark (`Q/K/V/O/MLP_up/MLP_down`) with
optional CPU single-head attention in the middle:

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
cargo run --example transformer_linear_block -- 8 5 1
```

Use `--no-attention` to isolate linear-stage timing only.

Wire f32 weight loading for all six stages (generated or file-backed):

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
cargo run --example transformer_linear_block -- 16 3 1 --no-attention --weight-source f32
```

Optional file-backed stage weights:

- `--weights-dir <dir>` expects:
  - `q_proj.f32le`, `k_proj.f32le`, `v_proj.f32le`,
  - `o_proj.f32le`, `mlp_up.f32le`, `mlp_down.f32le`
  each containing `2304*2304` little-endian `f32` row-major values.

### f32 weight-loading verification bridge

Run a full `f32 -> int8 -> template patch -> EdgeTPU execute` path and compare
against a CPU quantized reference:

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
cargo run --example gemm_weight_load_verify -- 8 3 1 2
```

This is the integration bridge between synthetic matrix modes and model-style
weight loading.

Migration note:

- The old fixed-256 wrapper types (`DenseGemm256Template`, `GemmTemplate256`, `PreparedGemm256`)
  were removed in favor of the single dimension-aware `DenseGemmTemplate` +
  `PreparedDenseGemm` path.
- Internal/FFI-only types are no longer exported at crate root (for example
  `TfLiteModel*` and `EdgeTPUDelegate*` raw types). Public API now focuses on
  `CoralDevice`, `EdgeTPUDelegate`, `CoralInterpreter`, and GEMM types.

### Real inference benchmark example

Download a quantized TensorFlow Lite model:

```bash
mkdir -p models
curl -L -o models/mobilenet_v1_1.0_224_quant.tflite \
  https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant.tflite
```

Run repeated inference:

```bash
cargo run --example inference_benchmark -- \
  models/mobilenet_v1_1.0_224_quant.tflite 100 10
```

### MVP CPU vs EdgeTPU comparison harness

Run a minimal two-workload comparison (`sanity_model` + `matrix_model`) across
`cpu_int8` and `edgetpu_int8`, with warmup `10`, runs `100`, repeats `3`:

```bash
cargo run --example cpu_vs_edgetpu_mvp -- \
  --sanity-model models/mobilenet_v1_1.0_224_quant.tflite \
  --matrix-model models/mobilenet_v1_1.0_224_quant.tflite \
  --warmup 10 \
  --runs 100 \
  --repeats 3 \
  --csv mvp_results.csv
```

Output includes per-scenario `RESULT ...` lines and one CSV file with repeat-level summaries.

## Validation snapshot (2026-02-20, Raspberry Pi 5)

Hardware: Raspberry Pi 5 (aarch64, Debian-based Raspberry Pi OS) with Coral USB Accelerator attached.

| Case | Command | Result |
|---|---|---|
| Build check | `cargo check --lib` | Pass |
| Device smoke | `cargo run --example basic_usage` | Pass |
| Delegate smoke | `cargo run --example simple_delegate` | Pass |
| TFLite + delegate creation | `cargo run --example tflite_test` | Pass |
| Benchmark (100 runs) | `inference_benchmark mobilenet_v1_1.0_224_quant.tflite 100 10` | Pass, avg `15.133 ms`, p95 `15.203 ms` |
| Benchmark (500 runs) | `inference_benchmark mobilenet_v1_1.0_224_quant.tflite 500 20` | Pass, avg `15.086 ms`, p95 `15.157 ms` |
| EdgeTPU model load | `inference_benchmark mobilenet_v1_1.0_224_quant_edgetpu.tflite 1 0` | Fail (`SIGSEGV`) at interpreter creation |
| EdgeTPU model load | `inference_benchmark mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite 1 0` | Fail (`SIGSEGV`) at interpreter creation |

## Notes

- Real hardware validation is required for meaningful results.
- API behavior for `CoralDevice`, `EdgeTPUDelegate`, and `CoralInterpreter` remains stable.
- Some distro/runtime combinations can crash when loading certain `*_edgetpu.tflite` models. If you hit that, align `libedgetpu` and TensorFlow Lite versions from the same Coral compatibility set.
