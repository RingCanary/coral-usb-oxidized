# Coral USB Oxidized

Rust-first SDK and reverse-engineering workspace for the Google Coral USB Accelerator.

The active path in this repo is pure-`rusb` device control, native replay/materialization tooling, and bounded checked-in family assets validated on a Pi-connected Coral. Compatibility surfaces for `libedgetpu`, TensorFlow Lite, and compiler-assisted reproduction still exist, but they are not the repo front door.

## Start Here

- `docs/index.md`
- `docs/active_path.md`
- `docs/phase7_full_control_boundary_2026-03-16.md`
- `docs/phase7_conv2d_k3_h12_corridor_completion_2026-03-16.md`
- `tools/README.md`

Current published bounded status:

- Phase 4: single-op Conv2D `k=3`, same-product family rooted at `16x64`
- Phase 6: `H=8`, `EO=6496`, `p32/p64/p128`, widths `72..192`
- Phase 7: `H=12`, `EO=6512`, `p32/p64/p128`, widths `64..192`

The Phase 7 corridor is fully published through `12x192`, including the previously excluded `p32` tail widths `12x176/184/192`, validated on the Coral device.

## Raspberry Pi 5 Setup

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

### 3) Active smoke checks

```bash
cargo check --lib
cargo run --example rusb_control_plane_probe -- --verbose-configs
cargo run --example rusb_serialized_exec_replay -- --help
```

### 4) Compatibility stack, only if explicitly needed

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/coral-edgetpu.gpg
echo "deb [signed-by=/etc/apt/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list >/dev/null
sudo apt-get update
sudo apt-get install -y libedgetpu1-std libedgetpu-dev libtensorflow-lite-dev
```

See `docs/legacy_compatibility.md` for when that stack is still useful.

## Quick Entry Points

Active runtime and bounded completion:

```bash
cargo run --example rusb_control_plane_probe -- --verbose-configs
cargo run --example rusb_serialized_exec_replay -- --help
cargo run --bin conv_k3_eo_emit -- --family-spec templates/phase7_conv2d_k3_h12_corridor_6512/family.json --channels 64 --target-height 12 --target-width 192 --out-report /tmp/conv_k3_eo_emit_phase7.json
bash scripts/phase4_conv2d_k3_completion_demo.sh --family-spec templates/phase7_conv2d_k3_h12_corridor_6512/family.json --pairs p32,p64,p128
```

Compatibility examples:

```bash
cargo run --features legacy-runtime --example basic_usage
cargo run --features legacy-runtime --example cpu_vs_edgetpu_mvp -- --help
```

Offline compiled-artifact inspection:

```bash
python3 tools/extract_edgetpu_package.py extract /tmp/model_edgetpu.tflite --out /tmp/edgetpu_extract
python3 tools/parse_edgetpu_executable.py /tmp/edgetpu_extract/package_000
```

## Documentation Map

- `docs/index.md`: docs landing page
- `docs/active_path.md`: current runtime path, published families, validation entry points
- `docs/examples.md`: runnable examples, workflows, and workload deep dives
- `docs/legacy_compatibility.md`: non-active delegate/compiler/TFLite surfaces
- `docs/archive_index.md`: historical notes and archive routing
- `tools/README.md`: current tool taxonomy
- `WORKLOG.md`: chronological work log

## Device Behavior

Coral USB commonly appears as:

- Initial: `1a6e:089a`
- After init/runtime activity: `18d1:9302`

Both IDs are expected and should be covered by udev rules.
