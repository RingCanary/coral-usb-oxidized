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
- `tools/bootstrap_edgetpu_compiler.sh` (local `edgetpu_compiler` bootstrap from Coral apt repo)
- `tools/dense_template_pipeline.sh` (generate -> compile -> extract -> parse -> inspect pipeline)
- `tools/strace_usb_scaling.py` (USBDEVFS submit/reap scaling fit from strace summaries)
- `tools/edgetpu_delegate_smoke.sh` (minimal delegate exercise without TensorFlow Lite C libs)
- `examples/inference_dump.rs` (single-invoke deterministic output dump for tensorizer validation)

Detailed workflow and caveats are documented in `docs/usb_tracing.md`.

Current reverse-engineering notes:

- `WORKLOG.md`
- `docs/usb_invoke_scaling_by_model.md`
- `docs/next_usbmon_capture_matrix.md`
- `docs/usb_register_map_candidates.md`
- `docs/usb_executable_transport_correlation.md`
- `docs/tensorizer_mvp.md`
- `docs/tensorizer_dense_template.md`
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
