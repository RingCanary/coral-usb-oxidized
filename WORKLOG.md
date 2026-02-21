# WORKLOG

## 2026-02-21

### Objective

Bring up a modern local stack for Coral USB (`libedgetpu` + TFLite C), recover
delegate/inference functionality, and start protocol reverse-engineering from
`*_edgetpu.tflite` and USB traffic.

### Bring-up milestones

1. Implemented package extraction tooling:
   - `tools/extract_edgetpu_package.py`
   - Verified `DWN1` package extraction on
     `mobilenet_v1_1.0_224_quant_edgetpu.tflite`.
2. Added capture tooling:
   - `tools/usbmon_capture.sh` (kernel usbmon)
   - `tools/usb_syscall_trace.sh` (strace fallback)
   - `docs/usb_tracing.md`
3. Added local bootstrap tooling:
   - `tools/bootstrap_arch_stack.sh`
   - Built local libs into `~/.local/lib`.

### Key failures and fixes

1. **Missing runtime libs at link time**
   - Symptom: `unable to find library -ledgetpu` / `-ltensorflowlite_c`.
   - Fix: local-prefix build via `tools/bootstrap_arch_stack.sh`.
2. **Arch/AUR dependency drift**
   - Symptom: `flatbuffers=24.3.25` unavailable; TF/libedgetpu mismatch.
   - Fix: patched TF-generated FlatBuffers version asserts during local build.
3. **TensorFlow hermetic python mismatch**
   - Symptom: TF 2.18 requested lockfiles for Python 3.14 (unsupported).
   - Fix: forced `HERMETIC_PYTHON_VERSION` and `TF_PYTHON_VERSION` (default 3.12).
4. **Corrupted/empty embedded DFU firmware in libedgetpu build**
   - Symptom: delegate creation failed with `Invalid DFU image file`.
   - Root cause: missing `xxd`; firmware arrays generated empty.
   - Fixes:
     - installed `xxd`
     - added hard checks in `tools/bootstrap_arch_stack.sh` for `xxd`
       presence and non-empty `usb_latest_firmware.h`.
5. **`sudo` capture command lost user-local libs**
   - Symptom: `usbmon_capture.sh` run under sudo linked against `/root/.local/lib`.
   - Fix: `tools/usbmon_capture.sh` now executes traced command as `$SUDO_USER`
     by default (override via `USBMON_RUN_COMMAND_AS_ROOT=1`).

### API behavior fix

1. **Stale `CoralDevice` VID/PID reporting**
   - Symptom: `CoralDevice::new()` reported static DFU IDs even when device was
     already in initialized mode (`18d1:9302`).
   - Fix: `src/lib.rs` constructors now derive IDs from live `find_coral_devices()`.

### Verified functional outcomes

1. `cargo run --example delegate_usage`
   - delegate creation successful
   - live VID/PID reporting correct
2. `cargo run --example tflite_test`
   - TFLite + EdgeTPU integration successful
3. `cargo run --example tflite_standard_example`
   - standard TFLite interpreter path successful
4. `cargo run --example inference_benchmark -- models/mobilenet_v1_1.0_224_quant.tflite 100 10`
   - avg latency around 18 ms
5. `cargo run --example inference_benchmark -- models/mobilenet_v1_1.0_224_quant_edgetpu.tflite 30 5`
   - avg latency around 2.6-2.8 ms

### Reverse-engineering artifacts

1. Successful syscall traces:
   - `traces/usb-syscall-20260221T085116Z/strace-usb-20260221T085116Z.log`
   - `traces/usb-syscall-20260221T085310Z/strace-usb-20260221T085310Z.log`
2. Successful usbmon capture:
   - `traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.log`
   - `traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.summary.txt`

### Reverse-engineering findings (current hypotheses)

1. Capture is healthy:
   - `total_lines=690`, balanced `S/C` (`345/345`), `command_exit=0`.
2. Dominant device traffic is on USB device `005`:
   - `Bo=296`, `Bi=160`, `Co=144`, `Ci=60`, `Ii=2`.
3. Pre-inference load burst likely corresponds to program/model upload:
   - bulk-out completed bytes before first input-sized transfer:
     `4,697,104` bytes.
4. Inference loop signature appears stable and exact:
   - repeated `35` times (5 warmup + 30 measured):
     - `Bo 225824` -> `Bo 150528` -> `Bi 1008`
   - timing:
     - `Bo225824 -> Bo150528`: ~`0.345 ms` avg
     - `Bo150528 -> Bi1008`: ~`1.477 ms` avg
     - interval between `Bo150528`: ~`2.637 ms` avg
5. `Bo 150528` matches input tensor size (`224*224*3` bytes).
6. `Bi 1008` is consistent with output logits buffer (1000-class model + alignment).
7. Negative usbmon statuses are expected async URB behavior:
   - mostly `-115` (`EINPROGRESS`) and teardown/cancel `-2` (`ENOENT`).

### New analysis tool

1. Added `tools/usbmon_phase_report.py`:
   - `report` subcommand for phase timeline and cycle metrics
   - `diff` subcommand for run-to-run comparison
2. Example:
   - `python3 tools/usbmon_phase_report.py report <usbmon.log> --bus 4 --device 005`
   - `python3 tools/usbmon_phase_report.py diff <run_a.log> <run_b.log> --bus 4 --device 005`

### Open questions

1. Exact semantic meaning of the `Bo 225824` payload per inference.
2. Mapping of vendor control register addresses (`wValue`) to hardware functions.
3. Relationship between `Bi ep2 size=16` status packets and inference completion.

## 2026-02-21 (RE matrix batch)

### Matrix output directory

- `traces/re-matrix-20260221T092342Z`

### Runs executed

1. `R1_basic_usage` (`usb_syscall_trace`)
2. `R2_edgetpu_delegate_smoke` (`usb_syscall_trace`)
3. `R3_simple_delegate` (`usb_syscall_trace`)
4. `R4_tflite_standard_cpu_only` (`usb_syscall_trace`)
5. `R5_infer_plain_short` (`usb_syscall_trace`)
6. `R6_infer_plain_short_repeat` (`usb_syscall_trace`)
7. `R7_infer_plain_long` (`usb_syscall_trace`)
8. `R8_infer_edgetpu_short` (`usb_syscall_trace`)
9. `R9_infer_edgetpu_long` (`usb_syscall_trace`)
10. `R10_tflite_test` (`usb_syscall_trace`)
11. `R11b_cpu_vs_edgetpu_mvp` (`usb_syscall_trace`)

All runs exited with `command_exit=0`.

### Batch artifacts

1. `traces/re-matrix-20260221T092342Z/RE_MATRIX_SUMMARY.md`
2. `traces/re-matrix-20260221T092342Z/USBMON_BASELINE_PHASE.txt`
3. `traces/re-matrix-20260221T092342Z/USBMON_BASELINE_PHASE.json`
4. `traces/re-matrix-20260221T092342Z/USBMON_RUNS_TO_EXECUTE.txt`
5. `traces/re-matrix-20260221T092342Z/USBMON_INTERACTIVE_SUMMARY.md`
6. Per-run trace logs and summaries under each `R*/` directory.

### Batch findings

1. CPU-only baseline (`R4`) generated zero USB syscalls, confirming trace
   filtering and baseline behavior.
2. Plain quantized model runs (`R5`/`R6`/`R7`) produced near-identical USB
   ioctl counts (`SUBMITURB=105`, `REAPURBNDELAY=201`) despite run-count
   changes.
3. EdgeTPU-compiled model run (`R9`) showed significantly higher USB activity
   (`SUBMITURB=324`, `REAPURBNDELAY=565`) and much lower latency (`avg ~2.7 ms`),
   consistent with strong accelerator engagement.
4. Delegate-only path parity was observed between C and Rust:
   `R2` and `R3` both showed the same control-heavy bring-up class with
   successful delegate create/free.
5. CPU-vs-EdgeTPU harness with the plain quantized model (`R11b`) showed no
   acceleration benefit in this environment (`edgetpu_int8` p50 slower than
   `cpu_int8`), consistent with partial/off-target offload behavior for
   non-EdgeTPU-compiled graphs.
6. Interactive usbmon captures (`U1`..`U4`) confirmed packet-level behavior:
   - `U1`/`U2` delegate traces are control-heavy and bulk-free
   - `U3` plain-model inference remains bulk-free and indistinguishable from setup
   - `U4` EdgeTPU-model inference reproduces full bulk cycle signature
     (`Bo225824 -> Bo150528 -> Bi1008`, count `35`) and `4,697,104` bytes
     pre-inference upload burst.
7. Added register-map extraction pass:
   - tool: `tools/usbmon_register_map.py`
   - artifacts: `REGISTER_MAP_MATRIX.*` and per-run `*_REGISTER_REPORT.*`
   - finding: control/register address-op counts are invariant across
     `U1/U2/U3/U4/baseline`; differentiator is bulk path activation.

### Notes

1. This matrix uses `usb_syscall_trace` for automation (unprivileged).
2. Kernel usbmon remains the primary packet-level source; latest successful
   baseline capture remains:
   - `traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.log`
3. An initial `R11` attempt failed only due a CSV path typo (permission on
   `/R11_mvp_results.csv`); corrected in `R11b` with successful completion.
4. Candidate register-map write-up: `docs/usb_register_map_candidates.md`.

## 2026-02-21 (RE continuation: invoke scaling + payload signatures)

### Additional runs executed

1. `R12_infer_edgetpu_5_1` (`usb_syscall_trace`)
2. `R13_infer_edgetpu_10_0` (`usb_syscall_trace`)
3. `R14_infer_edgetpu_20_5` (`usb_syscall_trace`)
4. `R15_infer_plain_1_0` (`usb_syscall_trace`)

All runs exited with `command_exit=0`.

### New artifacts

1. `traces/re-matrix-20260221T092342Z/USB_IOCTL_SCALING.md`
2. `traces/re-matrix-20260221T092342Z/U1_BULK_SIG.{txt,json}`
3. `traces/re-matrix-20260221T092342Z/U2_BULK_SIG.{txt,json}`
4. `traces/re-matrix-20260221T092342Z/U3_BULK_SIG.{txt,json}`
5. `traces/re-matrix-20260221T092342Z/U4_BULK_SIG.{txt,json}`
6. `traces/re-matrix-20260221T092342Z/BASE_BULK_SIG.{txt,json}`
7. `traces/re-matrix-20260221T092342Z/USBMON_BULK_SIGNATURE_SUMMARY.md`
8. `traces/re-matrix-20260221T092342Z/DIFF_BASE_vs_U4_BULK_SIG.txt`

### New tooling

1. Added `tools/usbmon_bulk_signature.py`:
   - extracts top bulk payload prefix signatures from usbmon logs
   - classifies signatures by phase (`pre_loop`, `loop`, `post_loop`)
   - surfaces recurring per-invoke command/header candidates

### New findings

1. EdgeTPU model USB ioctl counts scale linearly with total invokes:
   - `SUBMITURB = 114 + 6 * total_invokes`
   - `REAPURBNDELAY = 215 + 10 * total_invokes`
2. Plain model USB ioctl counts remain flat at:
   - `SUBMITURB=105`, `REAPURBNDELAY=201`
   across `1`, `6`, and `55` total invokes.
3. `U4` and baseline usbmon captures are signature-identical for normalized
   bulk payload headers (`DIFF_BASE_vs_U4_BULK_SIG.txt` empty).
4. Strong per-invoke loop markers in `U4/baseline`:
   - `Bo 8: 20720300 00000000`
   - `Bo 8: 004c0200 01000000`
   - `Bo 225824: 800f0080 dc000000`
   - `Bo 150528: 00010203 04050607`
5. `U1/U2/U3` remain bulk-submit-free (interrupt polling only), reinforcing
   that loop markers are EdgeTPU-compiled-model specific in this matrix.

### External research snapshot

1. Added source-backed constraints summary:
   - `docs/external_research_2026-02-21.md`
2. Key conclusion:
   - practical path is constrained TFLite graph compilation (int8/static/op-set),
     not arbitrary custom-kernel USB dispatch via public APIs.

### Executable-vs-transport correlation pass

1. Added note:
   - `docs/usb_executable_transport_correlation.md`
2. New finding:
   - `U4` bulk signature markers (`20720300`, `800f0080dc...`, `501c0000`,
     `800f000c07...`) are present at specific offsets inside extracted
     serialized executables (`exec0` vs `exec1` split), strengthening the
     hypothesis that loop/preload transport headers are compiled artifact data.

### Cross-model invoke scaling extension

1. Added additional strace runs:
   - `R16_infer_edgetpu_bird_10_2`
   - `R17_infer_plain_bird_10_2`
   - `R18_infer_edgetpu_bird_1_0`
   - `R19_infer_edgetpu_bird_20_5`
   - `R20_infer_edgetpu_bird_5_1`
   - `R21_infer_edgetpu_bird_30_5`
   - `R22_infer_plain_bird_1_0`
   - `R23_infer_edgetpu_bird_10_0`
   - `R24_infer_edgetpu_bird_1_2`
   - `R25_infer_edgetpu_bird_10_2_repeat`
   - `R26_infer_inception_edgetpu_1_0`
   - `R27_infer_inception_edgetpu_10_0`
   - `R28_infer_inception_plain_10_0`
   - `R29_infer_inception_plain_1_0`
   - `R30_infer_inception_edgetpu_20_0`
   - `R31_infer_inception_plain_20_0`
2. Added summary note:
   - `docs/usb_invoke_scaling_by_model.md`
3. Added automation tool:
   - `tools/strace_usb_scaling.py`
   - emits per-model linear fits and residuals from `R*/` strace summaries
4. New finding:
   - EdgeTPU ioctl scaling differs by model:
     - model A (`mobilenet_v1..._edgetpu`): exact `+6 submit / +10 reap` per invoke
     - model B (`mobilenet_v2...inat_bird..._edgetpu`): exact
       `+8 submit / +12 reap` on primary fit, with one outlier run (`R16`)
       that did not reproduce in `R25` repeat
     - model C (`inception_v1..._edgetpu`): approximately
       `+10 submit / +15 reap` per invoke (small jitter)
   - plain-model path remains flat at setup-level USB counts in this environment.
5. Added privileged packet-capture next-step matrix:
   - `docs/next_usbmon_capture_matrix.md`

## 2026-02-21 (packet-level validation from new usbmon captures)

### New privileged captures

1. `traces/usbmon-20260221T103521Z-bus4/usbmon-bus4-20260221T103521Z.log`
   - `mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite` (`warmup=5`, `runs=20`)
2. `traces/usbmon-20260221T103552Z-bus4/usbmon-bus4-20260221T103552Z.log`
   - `inception_v1_224_quant_edgetpu.tflite` (`warmup=0`, `runs=20`)
3. `traces/usbmon-20260221T103631Z-bus4/usbmon-bus4-20260221T103631Z.log`
   - `inception_v1_224_quant.tflite` (`warmup=0`, `runs=20`)

### New analysis artifacts

1. `traces/re-matrix-20260221T092342Z/U5_*`
2. `traces/re-matrix-20260221T092342Z/U6_*`
3. `traces/re-matrix-20260221T092342Z/U7_*`
4. `traces/re-matrix-20260221T092342Z/REGISTER_MAP_MATRIX_U1_U7.{md,json}`
5. `traces/re-matrix-20260221T092342Z/USBMON_PACKET_VALIDATION_20260221T1035Z.md`

### New findings

1. Bird EdgeTPU run (`U5`) shows repeating 3-stage loop:
   - `Bo 261920 -> Bo 150528 -> Bo 10224 -> Bi 968` (`count=25`)
2. Inception EdgeTPU run (`U6`) shows a distinct 3-stage loop:
   - `Bo 254656 -> Bo 150528 -> Bo 393664 -> Bi 1008` (`count=20`)
3. Inception plain run (`U7`) remains bulk-submit-free (control/interrupt only).
4. Control/register address-op counts remain invariant across `U1..U7`; model
   differences are isolated to bulk transport choreography.
5. Packet-level data now confirms the syscall-level model classes:
   - model A: `+6/+10`
   - model B: `+8/+12`
   - model C: about `+10/+15`

## 2026-02-21 (dedicated 3-stage signature parser)

### New tooling

1. Added `tools/usbmon_three_stage_signature.py`:
   - auto-discovers repeated `Bo -> Bo -> Bo -> Bi` completion candidates
   - extracts non-overlapping cycles with per-leg timing stats
   - reports inter-stage gap-event counts to surface hidden intermediate steps
   - supports explicit pattern forcing (`--bo-1/--bo-2/--bo-3/--bi-out`)

### New analysis artifacts

1. `traces/re-matrix-20260221T092342Z/U4_STAGE3_SIG.{txt,json}`
2. `traces/re-matrix-20260221T092342Z/U5_STAGE3_SIG.{txt,json}`
3. `traces/re-matrix-20260221T092342Z/U6_STAGE3_SIG.{txt,json}`
4. `traces/re-matrix-20260221T092342Z/U6_STAGE3_SIG_254K_150K_393K.{txt,json}`
5. `traces/re-matrix-20260221T092342Z/U7_STAGE3_SIG.{txt,json}`

### New findings

1. `U5` auto-detected 3-stage signature:
   - `Bo 261920 -> Bo 150528 -> Bo 10224 -> Bi 968` (`count=25`)
2. `U6` auto-detected 3-stage tail signature:
   - `Bo 150528 -> Bo 393664 -> Bo 103200 -> Bi 1008` (`count=20`)
3. Explicit `U6` match for prior hypothesis
   (`Bo 254656 -> Bo 150528 -> Bo 393664 -> Bi 1008`) still yields `count=20`,
   but with `gap_events_3_4=2`, indicating additional completions between
   `Bo 393664` and `Bi 1008`.
4. `U7` and `U4` do not auto-select a 3-stage signature under
   `min_count=3` (expected for plain path and 2-stage baseline class).

## 2026-02-21 (schema-aware executable parser)

### New tooling

1. Added `tools/parse_edgetpu_executable.py`:
   - parses `serialized_executable_*.bin` directly as FlatBuffers
   - extracts instruction chunks, relocation metadata (`field_offsets`), layer
     metadata, parameter sizes, and DMA-hint summaries
   - scans known transport-marker byte patterns with offsets/counts
2. Added vendored schema reference:
   - `docs/schema/libedgetpu_executable.fbs`
   - provenance: `docs/schema/README.md`

### New analysis artifacts

1. `traces/re-matrix-20260221T092342Z/EXEC_PARSE_MOBILENET_V1.{txt,json}`
2. `traces/re-matrix-20260221T092342Z/EXEC_PARSE_BIRD_V2.{txt,json}`
3. `traces/re-matrix-20260221T092342Z/EXEC_PARSE_INCEPTION_V1.{txt,json}`

### New findings

1. Model A (`mobilenet_v1..._edgetpu`):
   - `EXECUTION_ONLY` chunk: `225824`
   - `PARAMETER_CACHING` chunk: `7248`
   - `PARAMETER_CACHING` params: `4464000`
2. Model B (`mobilenet_v2...bird..._edgetpu`):
   - `EXECUTION_ONLY` chunks: `261920`, `10224`
   - `PARAMETER_CACHING` chunk: `10064`
   - `PARAMETER_CACHING` params: `3947392`
3. Model C (`inception_v1..._edgetpu`):
   - `EXECUTION_ONLY` chunks: `254656`, `103200`
   - `EXECUTION_ONLY` params: `393664`
   - `PARAMETER_CACHING` chunk: `9680`
   - `PARAMETER_CACHING` params: `6581440`
4. Cross-correlation to usbmon loops:
   - model A loop stage `Bo225824` maps to exec0 chunk
   - model B loop stages `Bo261920` + `Bo10224` map to exec0 chunks
   - model C loop stages `Bo254656` + `Bo393664` + `Bo103200` map to
     exec0 chunk + exec0 params + exec0 chunk respectively, with `Bo150528`
     as input activation transfer
5. Relocation metadata confirms address patch points are explicit:
   - `BASE_ADDRESS_PARAMETER`, `BASE_ADDRESS_SCRATCH`,
     `BASE_ADDRESS_INPUT_ACTIVATION`, `BASE_ADDRESS_OUTPUT_ACTIVATION`
   appear in `EXECUTION_ONLY` executable field offsets, with layer names carried
   on input/output relocations.

## 2026-02-21 (tensorizer MVP: in-place parameter patching)

### New tooling

1. Added `tools/tensorizer_patch_edgetpu.py`:
   - `inspect` subcommand finds package/executable layout and absolute
     `Executable.parameters` byte ranges inside compiled `*_edgetpu.tflite`
   - `patch` subcommand rewrites selected parameter regions in-place while
     preserving file size and FlatBuffer offsets
   - patch modes: `zero`, `byte`, `ramp`, `xor`, `random`

### New docs

1. Added `docs/tensorizer_mvp.md`:
   - end-to-end commands and observed results for model A tensorizer sanity test

### New findings

1. Model A patch target located:
   - `exe[1]` (`PARAMETER_CACHING`) parameter region
   - absolute range `[13276, 4477276)` (`4464000` bytes)
2. Baseline inference (`runs=5`, `warmup=1`):
   - top output `index=905 score=38`
3. Patched inference (three patterns: `zero`, `ramp`, `xorff`):
   - model still runs on EdgeTPU
   - top output collapses to `index=1000 score=0`
4. Interpretation:
   - parameter payload mutation is sufficient to alter behavior while keeping
     compiled execution stream runnable, validating the tensorizer direction.

## 2026-02-21 (single-op Dense template tensorizer path)

### New tooling

1. Added `tools/bootstrap_edgetpu_compiler.sh`:
   - installs `edgetpu_compiler` from Coral apt package into local prefix
   - installs both wrapper and runtime bundle (`edgetpu_compiler_bin`)
2. Added `tools/generate_dense_quant_tflite.py`:
   - generates a single-layer Dense quantized `.tflite` with deterministic
     kernel initialization
   - supports `identity`, `permutation`, `ones`, `random_uniform`
3. Added `tools/dense_template_pipeline.sh`:
   - end-to-end `uv` flow: generate -> compile -> extract -> parse -> inspect
   - optional parameter patch generation
4. Added `examples/inference_dump.rs`:
   - single-invoke deterministic input/output dump (`zeros|ones|ramp|alt`)
   - prints full int8 output preview for tensorizer verification

### New docs

1. Added `docs/tensorizer_dense_template.md`.
2. Updated `README.md` with compiler bootstrap + dense template workflow.
3. Updated `docs/tensorizer_mvp.md` to link the single-op template track.

### Toolchain compatibility finding

1. Dense templates generated with `tensorflow-cpu==2.18.0` failed
   `edgetpu_compiler` acceptance in this environment.
2. Dense templates generated with:
   - `python 3.9`
   - `tensorflow-cpu 2.10.1`
   - `numpy 1.23.5`
   compile successfully and map `FULLY_CONNECTED` to EdgeTPU.

### New analysis artifacts

1. `traces/dense-template-20260221T120206Z/PIPELINE_SUMMARY.txt`
2. `traces/dense-template-20260221T120206Z/exec_parse.{txt,json}`
3. `traces/dense-template-20260221T120206Z/tensorizer_inspect.{txt,json}`
4. `traces/dense-template-20260221T120206Z/inference_dump_*.log`
5. `traces/dense-template-20260221T120206Z/inference_benchmark_*.log`
6. `traces/usb-syscall-20260221T120304Z/*.summary.txt`
7. `traces/usb-syscall-20260221T120317Z/*.summary.txt`

### New findings

1. Single-op compiled model (`dense_256x256_quant_edgetpu.tflite`) contains:
   - `EXECUTION_ONLY` executable with instruction chunk `4112`
   - `PARAMETER_CACHING` executable with params `65536` bytes
2. Baseline deterministic runtime output (`inference_dump`, `ramp` input):
   - output vector preserves ramp `[-128..127]` for 256 lanes
3. Parameter patch impact (same instruction stream):
   - `zero`: near-constant high saturated output (`127`-dominant)
   - `ramp`: periodic transformed pattern
   - `xorff`: descending transformed pattern
4. Latency remains stable despite payload mutation:
   - baseline avg `0.210 ms` (`runs=30`, `warmup=5`)
   - patched_zero avg `0.210 ms`
5. USB syscall profile is transport-invariant across baseline vs patched:
   - `SUBMITURB` stayed `320`
   - `REAPURBNDELAY` differed only slightly (`456` vs `466`)
