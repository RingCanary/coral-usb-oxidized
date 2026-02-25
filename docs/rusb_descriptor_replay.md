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

## Pi5 test status (2026-02-25)

Host: `rpilm3.local`

### Build/tests

- `cargo check --example rusb_serialized_exec_replay --example rusb_control_plane_probe --example gemm_csr_perturb_probe` ✅
- `cargo test --lib` ✅ (`15 passed` including new `control_plane` and
  `usb_driver` unit tests)

### Runtime replay (clean-start matrix)

Model used:
- `templates/dense_2048x2048_quant_edgetpu.tflite`

Observed extracted executables:
- `exec0`: type `EXECUTION_ONLY`, payload `16384` bytes
- `exec1`: type `PARAMETER_CACHING`, payload `4202496` bytes

Clean-start procedure:
1. Power cycle Pi5 USB host ports with `uhubctl`:
   - `-l 2 off`, `-l 4 off`, wait 5s, then both on.
2. Confirm Coral is in boot state (`1a6e:089a`).
3. Run replay with `--firmware apex_latest_single_ep.bin`.

Observed baseline behavior:
1. Firmware upload and runtime transition succeed.
2. Setup sequence (write-only, 38 steps) succeeds from clean start.
3. Bootstrap instruction descriptors are accepted.
4. Parameter stream (`~4 MiB`) currently fails with bulk timeout when sent on
   expected classes (`tag 2`, also `0` and `1`).

Descriptor-tag sweep (`templates/dense_2048x2048_quant_edgetpu.tflite`,
`chunk-size=4096`):

- `--parameters-tag 2` (Parameters):
  - timeout at offset `49152`
- `--parameters-tag 0` (Instructions):
  - timeout at offset `28672`
- `--parameters-tag 1` (InputActivations):
  - timeout at offset `32768`
- `--parameters-tag 3` (OutputActivations):
  - no bulk timeout, run completes
- `--parameters-tag 4` (Interrupt0):
  - no bulk timeout, run completes

Control run:
- `--skip-param-preload` completes with event+output from clean start.

Interpretation:
- Transport and control plane are functioning.
- Current failure is narrowed to descriptor queue/class semantics for parameter
  admission, not a generic USB bulk path failure.
- Nonstandard tags (`3/4`) avoid immediate timeout but do not yet imply valid
  parameter loading.

### Extended parameter-admission probe

New replay controls now support:
- parameter stream chunk override,
- max-byte caps,
- per-chunk event/interrupt polling,
- inter-chunk pacing,
- multi-descriptor parameter segmentation.

Observed on Pi5 (clean power-cycled start each run):

1. `tag=2` fails at a stable boundary near `0xC000` bytes:
   - typically `49152`,
   - `48128` in split modes where descriptor headers consume queue budget.
2. Changing stream chunk size (`4096` vs `1024`) does not move the wall.
3. Event polling (`0x82`) and interrupt polling (`0x83`) during stream did not
   produce drains that unblock class-2 streaming.
4. Splitting parameter payload into multiple class-2 descriptors (`32K/16K/8K`)
   still fails in the same cumulative-byte regime.
5. Capping at exactly `49152` allows the write phase to end, but no bootstrap
   completion event is observed and subsequent writes time out.

### Runtime poison behavior after class-2 stall

Immediately after a `tag=2` stall (~`0xC000` bytes), control plane becomes
non-responsive:
- CSR reads (`0x1a30c`, `0x44018`) timeout,
- CSR writes (`0x44018`) timeout,
- event (`0x82`) and interrupt (`0x83`) reads timeout.

Control comparison after healthy invoke (`--skip-param-preload`) shows CSR reads
working normally in runtime mode, so this is a stall-induced poison state.

Interpretation update:
- The failure behaves like a queue-admission/backpressure stall in current
  runcontrol/runtime state.
- It is not explained by host-side chunking strategy alone.
- Next focus should be CSR/runcontrol transitions and doorbell semantics around
  class-2 descriptor consumption.

### Glitch/fuzz findings

A dedicated fuzz harness (`examples/rusb_param_glitch_fuzz.rs`) was used to
perturb class-2 streaming with controlled glitch actions.

Key observations:
1. Baseline no-glitch stall remains at `49152` bytes.
2. Deterministic one-shot runcontrol injections (`rc0`/`rc1`) at fixed chunk
   indices do not move the baseline stall.
3. High-frequency perturbation traffic does move the stall earlier:
   - `readonly` glitch mode (poll/read/sleep) reached ~`34816`,
   - `runctl` glitch mode (repeated runcontrol writes) reached ~`33792`.

Interpretation:
1. The system is sensitive to sustained control-plane activity during parameter
   ingress.
2. The class-2 wall is dynamic under control traffic, consistent with shared
   scheduler/queue pressure rather than a fixed protocol field limit.

### edgetpuxray parity note

Code inspection of `edgetpuxray/connect.py` indicates it submits multi-MB
parameter slices (descriptor class 2) as part of known-good flows. This argues
against a hard protocol limit at `0xC000` and supports the view that our replay
state machine is missing a required control transition.

## Practical next debug steps

1. Instrument runcontrol/doorbell CSR state immediately before and after each
   descriptor class submission to correlate queue-pressure and timeout offset.
2. Add endpoint drain checks (`0x82`, `0x83`) during large parameter streaming
   to detect required host-ack behavior.
3. Capture usbmon on Pi5 for:
   - known-good `libedgetpu` invoke
   - Rust replay (`tag 2` timeout case)
   and diff packet ordering/size cadence.
4. Probe descriptor scheduling permutations:
   - interleave param chunks with event reads,
   - split PARAMETER_CACHING into smaller phased submissions,
   - vary setup read/write inclusion and strictness.
