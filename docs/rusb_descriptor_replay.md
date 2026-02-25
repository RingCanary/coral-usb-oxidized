# Pure `rusb` Descriptor Replay (edgetpuxray-aligned)

## Goal

Add a pure-Rust USB path that can:

1. parse serialized executables from compiled `*_edgetpu.tflite`,
2. frame descriptor headers (`len + tag`) like `edgetpuxray/connect.py`,
3. send payloads over bulk-out (`0x01`),
4. read completion/event from `0x82` and output from `0x81`.

## New library pieces

1. `src/control_plane.rs`
   - named CSR map and helper utilities (`split_offset`, register formatting)
   - `EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE` (52-step sequence recovered from
     `geohot/edgetpuxray`)
2. `src/usb_driver.rs`
   - `EdgeTpuUsbDriver` with:
     - device discovery/open/claim
     - vendor read/write32/64
     - setup sequence application
     - descriptor framing (`DescriptorHeader`) and chunked bulk-out
     - event (`0x82`) and interrupt (`0x83`) decode helpers
3. `src/flatbuffer.rs`
   - `extract_serialized_executables_from_tflite()`
   - executable metadata (`type`, payload, parameter-region offsets)
4. `examples/rusb_serialized_exec_replay.rs`
   - end-to-end replay attempt from compiled `.tflite`
   - supports bootstrap flow for `EXECUTION_ONLY + PARAMETER_CACHING`

## Pi5 test status (2026-02-24)

Host: `rpilm3.local`

### Build/tests

- `cargo check --example rusb_serialized_exec_replay --example rusb_control_plane_probe --example gemm_csr_perturb_probe` ✅
- `cargo test --lib` ✅ (`15 passed` including new `control_plane` and
  `usb_driver` unit tests)

### Runtime replay

Model used:
- `templates/dense_2048x2048_quant_edgetpu.tflite`

Observed extracted executables:
- `exec0`: type `EXECUTION_ONLY`, payload `16384` bytes
- `exec1`: type `PARAMETER_CACHING`, payload `4202496` bytes

Current blockers:

1. With setup enabled (write-only setup sequence), first SCU write fails:
   - `step 0 write failed at 0x0001a30c: Operation timed out`
2. With setup skipped, first descriptor bulk write fails:
   - `bulk write failed on endpoint 0x01: Input/Output Error`

Interpretation:
- transport framing and extraction are in place,
- but runtime state preconditions for this direct path are still unmet on this
  host/device state.

## Practical next debug steps

1. Run replay against clean `1a6e:089a` state before libedgetpu touches device.
2. Compare interface/config state and claim ordering vs `edgetpuxray/connect.py`
   (especially around reset/set_configuration/claim sequence).
3. Capture usbmon while `connect.py` runs on the same hardware and diff control
   and bulk ordering against this Rust path.
4. Try request-recipient variants (`Device` vs `Interface`) for SCU writes in a
   controlled probe matrix.
