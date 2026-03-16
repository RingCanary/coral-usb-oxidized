# Coral USB Oxidized

Rust SDK/driver layer for Google Coral USB Accelerator:

- active path: pure-rusb control/data-plane tooling and native RE/materialization helpers
- compatibility path: legacy delegate/TFLite/compiler interoperability when explicitly needed

## What this crate provides

- Pure-`rusb` Coral USB detection and runtime control (`1a6e:089a`, `18d1:9302`)
- Native replay, extraction, patching, and RE helpers for executable/parameter-state work
- Compatibility-only `legacy-runtime` examples for delegate/TFLite interop

## Start here

If you are working on the current native path, start with:

- `docs/active_path.md`
- `docs/phase4_conv2d_k3_completion_2026-03-16.md`
- `docs/phase5_conv2d_k3_6496_boundary_scan_2026-03-16.md`
- `docs/phase4_completion_control_plan_2026-03-07.md`
- `docs/phase4_conv2d_k3_param_region_2026-03-07.md`
- `docs/phase4_conv2d_k3_native_param_materialize_2026-03-07.md`
- `docs/phase4_conv2d_k3_eo_localization_2026-03-07.md`
- `WORKLOG.md`

Current bounded completion status:

- Phase 4 is now complete for the bounded single-op Conv2D `k=3`, `same`, same-product family rooted at `16x64`
- active completion assets live under `templates/phase4_conv2d_k3_sameprod_6512/`
- the one-command Pi proof is `scripts/phase4_conv2d_k3_completion_demo.sh`
- Phase 5 boundary discovery currently shows `8x128` as a singleton `EO=6496` island on the scanned p64 power-of-two same-product axis; no nontrivial `6496` family has been frozen yet

## Raspberry Pi 5 setup

### 1) System packages

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl build-essential pkg-config libusb-1.0-0-dev clang llvm-dev libclang-dev gnupg
```

### 2) USB permissions

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

### 3) Build and smoke test the active pure-rusb path

```bash
cargo check --lib
cargo run --example rusb_control_plane_probe -- --verbose-configs
cargo run --example rusb_serialized_exec_replay -- --help
```

### 4) Legacy compatibility stack (only if explicitly needed)

Install this only for compatibility examples that require `legacy-runtime`, TensorFlow Lite, or `libedgetpu`.

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/coral-edgetpu.gpg
echo "deb [signed-by=/etc/apt/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list >/dev/null
sudo apt-get update
sudo apt-get install -y libedgetpu1-std libedgetpu-dev libtensorflow-lite-dev
```

Legacy usage details are in `docs/legacy_compatibility.md`.

## Linking behavior

With `--features legacy-runtime`, the build script links `libedgetpu` and a
TensorFlow Lite C runtime from these locations. This is a compatibility path,
not the active Phase 4 runtime path.

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

By default, TensorFlow Lite linking prefers `libtensorflowlite_c.so` when
present, otherwise falls back to distro naming (`libtensorflow-lite.so`).

Without `legacy-runtime`, no `libedgetpu`/`libtensorflowlite*` link flags are
added. This is the active pure-rusb mode for protocol RE, replay, and native
materialization work.

## Device behavior

Coral USB commonly appears as:

- Initial: `1a6e:089a`
- After delegate/init/inference: `18d1:9302`

Both IDs are expected and should be included in udev rules.

## Examples

Active pure-rusb/native path:

```bash
cargo run --example rusb_control_plane_probe -- --verbose-configs
cargo run --example rusb_serialized_exec_replay -- --help
cargo run --example rusb_param_glitch_fuzz -- --help
cargo run --example rust_dense_template_compile -- --help
```

Compatibility-only legacy delegate/TFLite/GEMM examples (requires `legacy-runtime`):

```bash
cargo run --features legacy-runtime --example basic_usage
cargo run --features legacy-runtime --example verify_device
cargo run --features legacy-runtime --example delegate_usage
cargo run --features legacy-runtime --example simple_delegate
cargo run --features legacy-runtime --example tflite_test
cargo run --features legacy-runtime --example tflite_standard_example
cargo run --features legacy-runtime --example cpu_vs_edgetpu_mvp -- --help
cargo run --features legacy-runtime --example gemm_int8_dynamic -- <dense_template_edgetpu.tflite> <input_dim> <output_dim> identity ramp
```

## Offline EdgeTPU package extractor

This remains useful for inspecting compiler-produced artifacts during RE, but it
is not part of the active native artifact-creation path.

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

This list still mixes active runtime/control helpers with transitional
compiler-assisted research helpers. For the stricter taxonomy, see
`tools/README.md`.

For protocol-level and syscall-level capture helpers, use:

- `tools/usbmon_capture.sh` (root, kernel usbmon capture)
- `tools/usb_syscall_trace.sh` (unprivileged `strace` fallback)
- `tools/usbmon_phase_report.py` (phase-oriented usbmon report and diff)
- `tools/usbmon_register_map.py` (usbmon control/register extraction and run matrix)
- `tools/usbmon_bulk_signature.py` (bulk payload header/signature extraction by phase)
- `tools/usbmon_three_stage_signature.py` (dedicated 3-stage bulk loop signature parser)
- `tools/parse_edgetpu_executable.py` (schema-aware parser for serialized executables)
- `tools/exec_chunk_diff.py` (EXECUTION_ONLY chunk diff + relocation-overlap analysis for Phase-B ISA work)
- `tools/tensorizer_patch_edgetpu.py` (in-place parameter patcher for compiled `*_edgetpu.tflite`)
- `tools/generate_dense_quant_tflite.py` (single-layer Dense INT8 model generator)
- `tools/generate_conv2d_quant_tflite.py` (single-layer Conv2D INT8 model generator)
- `tools/dump_tflite_conv1x1_weights.py` (dump stored 1x1 Conv2D quantized weight bytes from a TFLite model)
- `tools/patch_tflite_conv1x1_weight_pattern.py` (patch stored 1x1 Conv2D quantized weight bytes deterministically)
- `tools/generate_dense_conv_quant_tflite.py` (Conv2D->Dense INT8 multi-op model generator)
- `tools/bootstrap_edgetpu_compiler.sh` (local `edgetpu_compiler` bootstrap from Coral apt repo)
- `tools/dense_template_pipeline.sh` (generate -> compile -> extract -> parse -> inspect pipeline)
- `tools/conv_template_pipeline.sh` (Conv2D generate -> compile -> extract -> parse -> inspect pipeline)
- `tools/multiop_template_pipeline.sh` (Conv2D->Dense generate -> compile -> extract -> parse -> inspect pipeline)
- `scripts/m5_family_transition_map.sh` (Phase-2 M5 helper: derive EO/PC family transition map from compiled size table)
- `scripts/m5_build_family_patchspecs.sh` (Phase-2 M5 helper: build strict safe/discrete patchspec artifacts for recurrent dense families)
- `scripts/m5_crossdim_oracle_matrix.sh` (Phase-2 M5.5 helper: family-wide same-product transpose cross-dim oracle matrix on Pi)
- `scripts/m5_eo_oracle_group_probe.sh` (Phase-2 follow-up helper: coarse EO oracle block ablation on Pi)
- `scripts/m5_eo_neutral_window_crosscheck.sh` (Phase-2 follow-up helper: reverse-direction validation of candidate EO neutral windows)
- `scripts/m5_eo_window_refine_probe.sh` (Phase-2 follow-up helper: recurse only inside currently transport-critical EO windows)
- `scripts/m5_eo_rule_refine_probe.sh` (Phase-2 follow-up helper: rule-level refinement inside compact fatal EO windows)
- `scripts/phase3_conv2d_family_bootstrap.sh` (Phase-3 helper: small local 1x1 Conv2D family/size bootstrap scan)
- `scripts/phase3_conv2d_layout_matrix.sh` (Phase-3 helper: widen 1x1 Conv2D layout validation across `32/64/128` channel regimes)
- `scripts/phase3_conv2d_param_override_matrix.sh` (Phase-3 helper: exact full-stream Conv2D parameter synthesis + DUT override matrix)
- `scripts/phase3_conv2d_crossdim_oracle_matrix.sh` (Phase-3 helper: same-product spatial cross-dim oracle matrix for 1x1 Conv2D)
- `scripts/phase3_conv2d_eo_group_probe.sh` (Phase-3 follow-up helper: coarse EO oracle group ablation for same-product 1x1 Conv2D moves)
- `scripts/phase3_conv2d_eo_window_refine_probe.sh` (Phase-3 follow-up helper: recurse inside the coarse Conv2D EO body/tail windows)
- `scripts/phase3_conv2d_prefix_residual_probe.sh` (Phase-3 archival helper: stale-prefix semantic probe used before exact prefix recovery)
- `scripts/phase4_conv2d_k3_family_scout.sh` (Phase-4 scout helper: first local widening pass beyond 1x1 to single-op k=3 Conv2D)
- `scripts/phase4_conv2d_k3_crossdim_oracle_matrix.sh` (Phase-4 helper: first same-product spatial cross-dim oracle matrix for single-op k=3 Conv2D)
- `scripts/benchmark_dense_gemm_replay.sh` (benchmark helper: pure-rusb Dense replay latency / GMAC/s matrix on Pi)
- `scripts/m6_instruction_axis_probe.sh` (Phase-2 M6 helper: quant/activation/bias instruction-axis differential probe)
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
- `examples/function_gemma_decode_loop.rs` (autoregressive decode loop with Coral-backed q/k/v/o/gate/up/down and LM-head modes: cpu, coral-preload, coral-lazy)
- `examples/rusb_control_plane_probe.rs` (pure-rusb control-plane baseline: enumerate/open/claim/get-status)

Detailed workflow and caveats are documented in `docs/usb_tracing.md`.

Current reverse-engineering notes:

- `WORKLOG.md`
- `phase2-todo.md` (closed Dense Phase-2 checklist)
- `phase3-conv2d-todo.md` (closed Phase-3 Conv2D checklist)
- `docs/phase2_dense_m5_m7_2026-03-03.md`
- `docs/phase2_dense_m55_crossdim_oracle_matrix_2026-03-06.md`
- `docs/phase2_dense_eo_group_ablation_2026-03-06.md`
- `docs/phase2_dense_eo_neutral_window_crosscheck_2026-03-06.md`
- `docs/phase2_dense_eo_transport_window_refine_2026-03-06.md`
- `docs/phase2_dense_completion_2026-03-06.md`
- `docs/phase3_conv2d_kickoff_2026-03-06.md`
- `docs/phase3_conv2d_completion_2026-03-06.md`
- `docs/phase3_conv2d_eo_group_probe_2026-03-06.md`
- `docs/phase3_conv2d_eo_window_refine_2026-03-06.md`
- `docs/phase4_conv2d_k3_scout_2026-03-06.md`
- `docs/phase4_conv2d_k3_crossdim_oracle_matrix_2026-03-06.md`
- `docs/dense_gemm_replay_benchmark_2026-03-06.md`
- `docs/phase2_claims_audit_2026-03-03.md`
- `docs/usb_invoke_scaling_by_model.md`
- `docs/next_usbmon_capture_matrix.md`
- `docs/usb_register_map_candidates.md`
- `docs/usb_executable_transport_correlation.md`
- `docs/executable_opcode_diff.md`
- `docs/focus_points.md`
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
- `docs/rusb_control_plane_probe.md`
- `docs/schema/libedgetpu_executable.fbs`
- `docs/external_research_2026-02-21.md`
- `traces/re-matrix-20260221T092342Z/USBMON_PACKET_VALIDATION_20260221T1035Z.md`

Long-standing priority list (kept intentionally to four points only):

- `docs/focus_points.md`

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
