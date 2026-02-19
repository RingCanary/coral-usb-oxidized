# AGENTS.md

## Repo intent
- `coral-usb-oxidized` is a Rust SDK/driver layer for the Google Coral USB Accelerator.
- Focus: real hardware detection, EdgeTPU delegate creation, and TensorFlow Lite C API interop.
- Current implementation is Linux-oriented and expects local EdgeTPU runtime + TFLite shared libs.

## Code map
- `src/lib.rs`: main crate; USB discovery (`rusb`), EdgeTPU FFI, TFLite wrappers, and public API (`CoralDevice`, `list_devices`, `version`).
- `src/wrapper.h`: include bridge for `tensorflow/lite/c/c_api.h` and `edgetpu_c.h`.
- `src/build.rs`: bindgen + link hints for TFLite/EdgeTPU (currently under `src/`, not root `build.rs`).
- `examples/*.rs`: runnable flows (`basic_usage`, `verify_device`, `delegate_usage`, `tflite_*`).
- `examples/archive/*.rs`: older/experimental references.
- `run_test.sh`: local helper that sets `LD_LIBRARY_PATH` and runs `tflite_test`.

## Important device behavior
- Coral typically appears as `1a6e:089a` initially, then `18d1:9302` after init/delegate/inference.
- Detection logic handles both IDs; udev rules should include both.

## Dev workflow
- Start with `cargo run --example basic_usage` and `cargo run --example verify_device`.
- Delegate path: `cargo run --example delegate_usage`.
- TFLite path: `cargo run --example tflite_test` (often needs custom `LD_LIBRARY_PATH`).
- README mentions `.cargo/config.toml` aliases, but that file is not in this repo snapshot.

## Constraints and gotchas
- Real hardware testing is the intended validation path; avoid mock-only assumptions.
- Several paths are machine-specific (for example `/home/bhavesh/...` in `src/build.rs` and `run_test.sh`).
- Some logic is intentionally simplified/placeholder (delegate options parsing, dynamic loading comments).
- Keep FFI boundaries conservative: explicit errors, pointer checks, and clear ownership.

## When extending
- Keep API behavior stable around `CoralDevice`, `EdgeTPUDelegate`, and `CoralInterpreter`.
- Add focused examples for new behavior and validate on a real Coral USB stick.
- For linking/bindgen changes, prefer env-driven paths over hardcoded local paths.
