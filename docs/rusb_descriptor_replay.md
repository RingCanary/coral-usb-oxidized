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

### usbmon-parallel confirmation (tmux sessions on Pi5)

Captured with `tools/usbmon_capture.sh` while fuzz runner executed in parallel
(`traces/usbmon-fuzz-20260225T073823Z`):

1. baseline (`glitch_budget=0`):
   - `FUZZ_RESULT stall offset=49152`
   - runtime Bo complete size `1024`: `48`
2. readonly high-frequency glitches:
   - `FUZZ_RESULT stall offset=34816`
   - runtime Bo complete size `1024`: `34`
3. runctl high-frequency glitches:
   - `FUZZ_RESULT stall offset=33792`
   - runtime Bo complete size `1024`: `33`

Observed transport shift:
- as glitch traffic increases, accepted parameter Bo(1024) completions before
  failure drop proportionally.
- runctl-heavy mode also increases control transfer volume (`Co`) during stream.

### edgetpuxray parity note

Code inspection of `edgetpuxray/connect.py` indicates it submits multi-MB
parameter slices (descriptor class 2) as part of known-good flows. This argues
against a hard protocol limit at `0xC000` and supports the view that our replay
state machine is missing a required control transition.

## Side-by-side usbmon automation (2026-02-25)

To capture both replay and known-good paths in one run root, use:

```bash
./tools/usbmon_side_by_side_capture.sh --bus <BUS>
```

This runs:
1. pure-`rusb` deterministic sweep:
   - `cargo run --example rusb_serialized_exec_replay -- ... --parameters-tag <tag>`
   - default tags: `2,0,1,3,4`
2. known-good `libedgetpu` invoke:
   - `cargo run --example inference_benchmark -- <model> <runs> <warmup>`

Useful overrides:

```bash
./tools/usbmon_side_by_side_capture.sh \
  --bus <BUS> \
  --rusb-model templates/dense_2048x2048_quant_edgetpu.tflite \
  --libedgetpu-model templates/dense_2048x2048_quant_edgetpu.tflite \
  --rusb-tags 2,0,1,3,4 \
  --libedgetpu-runs 20 \
  --libedgetpu-warmup 5
```

Expected outputs:
1. root: `traces/usbmon-side-by-side-<timestamp>-bus<BUS>/`
2. replay lane:
   - `pure_rusb_deterministic_sweep/tag<descriptor_tag>/usbmon-bus<BUS>-*.log`
3. known-good lane:
   - `libedgetpu_known_good_invoke/inference/usbmon-bus<BUS>-*.log`
4. summary artifacts:
   - `capture_manifest.tsv` (exit codes + paths + commands)
   - `README.txt` (config snapshot)

Expected status pattern:
1. `tag2` often reproduces class-2 stall behavior.
2. `tag3`/`tag4` usually avoid immediate bulk timeout.
3. `libedgetpu_known_good_invoke/inference` should be exit `0` when delegate
   and invoke are healthy.

## Deterministic transition sweep (runcontrol/doorbell) vs known-good

Capture root:
- `traces/usbmon-transition-fixed-20260225T082936Z-bus4`

Matrix:
1. known-good pre invoke (`libedgetpu` path, `inference_benchmark`)
2. deterministic replay sweeps (`rusb_param_glitch_fuzz`) with:
   - `--transition-sequence resetkick` (`runctl0 -> doorbell -> runctl1`)
   - `--transition-chunks 32`, `40`, `47`
3. known-good post invoke

Command shape for deterministic cases:

```bash
sudo ./tools/usbmon_capture.sh -b 4 -o <case_dir> -- \
  cargo run --example rusb_param_glitch_fuzz -- \
    --model templates/dense_2048x2048_quant_edgetpu.tflite \
    --firmware ./apex_latest_single_ep.bin \
    --param-max-bytes 65536 \
    --param-stream-chunk-size 1024 \
    --glitch-budget 0 \
    --glitch-every-chunks 999999 \
    --glitch-mode readonly \
    --transition-chunks <chunk_idx> \
    --transition-sequence resetkick \
    --input-bytes 2048 \
    --output-bytes 2048
```

Observed deterministic outcomes:
1. `resetkick_chunk32`:
   - transition writes all `ok`
   - `FUZZ_RESULT stall offset=49152 chunk_idx=48`
2. `resetkick_chunk40`:
   - transition writes timeout (`runctl0`, `doorbell`, `runctl1`)
   - `FUZZ_RESULT stall offset=40960 chunk_idx=40`
3. `resetkick_chunk47`:
   - transition writes timeout
   - `FUZZ_RESULT stall offset=48128 chunk_idx=47`

usbmon comparison (`device=003` runtime lane):
1. known-good post:
   - `bulk_complete_sizes.Bo={"1048576":4,"9872":4,"2048":4,"8":10}`
   - `bulk_complete_sizes.Bi={"1024":8,"16":5}`
2. deterministic sweeps:
   - `chunk32`: `Bo 1024` completions `48`
   - `chunk40`: `Bo 1024` completions `40`
   - `chunk47`: `Bo 1024` completions `47`
   - no successful output-path `Bi` completions

Interpretation:
1. Early transition injection (`chunk32`) does not alter the baseline class-2 wall.
2. Near-wall transitions (`chunk40`, `chunk47`) fail at CSR write time and shift
   the admission cliff earlier.
3. This is consistent with a runcontrol/queue-state coupling issue, not a fixed
   static payload-size limit.

## Handshake candidate at the 49 KiB wall (Pi5 extraction)

A dedicated comparator was added:
- `tools/usbmon_param_handshake_probe.py`

Usage:

```bash
python3 tools/usbmon_param_handshake_probe.py <good.log> <bad.log> \
  --bus 4 \
  --threshold 49152 \
  --context 20
```

What it does:
1. finds parameter phase start from descriptor header (`Bo size=8`, `tag=2`),
2. tracks cumulative parameter payload (`Bo size>8`),
3. anchors around threshold crossing,
4. diffs transfer/status/control tuples near the anchor.

### Extracted signal from `traces/usbmon-transition-fixed-20260225T082936Z-bus4`

Good log:
- `libedgetpu_known_good/post/usbmon-bus4-20260225T083453Z.log`

Bad logs:
- `rusb_transition_sweep/resetkick_chunk32/...`
- `rusb_transition_sweep/resetkick_chunk40/...`
- `rusb_transition_sweep/resetkick_chunk47/...`

Near-anchor comparison (`good` vs `resetkick_chunk32`):
1. good transfer mix:
   - `Bi`, `Bo`, `Ci`, `Co`, `Ii`
2. bad transfer mix:
   - `Bo` only
3. control tuples present only in good:
   - `Ci:c0:01:a0d8:0001:0004`
   - `Co:40:01:a0d8:0001:0004`

Known-good sequence excerpt (device `003`):
1. `S Ci ... c0 01 a0d8 0001 0004`
2. `S Co ... 40 01 a0d8 0001 0004 = 00000080`
3. `S Bi ep2`, `S Ii ep3`, `S Bi ep1` pre-posts
4. parameter descriptor/data `Bo` stream starts.

Replay logs do not show that same pre-stream `a0d8` exchange + read-posting
pattern around the class-2 wall.

### Working hypothesis

The missing condition is likely a host-visible handshake/state prep immediately
before parameter ingress:
1. `a0d8` control exchange (`Ci/Co`),
2. posting async read endpoints (`Bi/Ii`),
3. then parameter `Bo` submission.

This is now the highest-value target for the next pure-`rusb` experiment pass.

### Negative control (important)

We explicitly tested replay variants to rule out the obvious simplification:
1. baseline write-only setup,
2. setup with read steps enabled (`--setup-include-reads`),
3. setup with reads + per-chunk event/interrupt polling
   (`--param-read-event-every 1 --param-read-interrupt-every 1`).

All variants still stalled at:
- `offset=49152 (chunk 48)`.

So the gap is not "missing any read call", but more likely:
1. pending async read-posting behavior (multiple outstanding `Bi/Ii` URBs), or
2. another tightly ordered queue-arming transition adjacent to that behavior.

### Descriptor-level pre-ingress emulation (new)

Replay now supports an explicit pre-ingress emulation path in
`rusb_serialized_exec_replay`:
1. `--param-a0d8-handshake`
2. `--param-a0d8-write-value`
3. `--param-prepost-bulk-in-reads`
4. `--param-prepost-bulk-in-size`
5. `--param-prepost-event-reads`
6. `--param-prepost-interrupt-reads`
7. `--param-prepost-timeout-ms`

Semantics:
1. before each parameter descriptor header, run optional read bursts
   (`EP 0x81/0x82/0x83`),
2. optional `a0d8` read/write pulse,
3. run optional post-read bursts.

Pi5 reboot-isolated results:
1. non-split stream (`chunk=1024`, `max=65536`, `tag=2`):
   - baseline: `offset=49152`
   - prepost-only: `offset=49152`
   - prepost+`a0d8`: `offset=49152`
2. split stream (`--param-descriptor-split-bytes 8192`):
   - split baseline: `offset=48128`
   - split prepost-only: `offset=48128`
   - split prepost+`a0d8`: handshake succeeds on descriptors `1..5`, then
     descriptor `6` `a0d8` read/write time out, followed by bulk-out timeout.

Interpretation:
1. synchronous emulation is not enough to bypass the class-2 wall,
2. descriptor-6 failure in split mode suggests control-plane degradation begins
   around `~40 KiB` cumulative ingress under this probe.

### Concurrent read lanes during stream (new)

To approximate pending reads, replay now supports concurrent lane threads:
1. `--param-async-bulk-in-lanes`
2. `--param-async-bulk-in-size`
3. `--param-async-event-lanes`
4. `--param-async-interrupt-lanes`
5. `--param-async-timeout-ms`

Observed on Pi5 (`setup-include-reads`, reboot-isolated):
1. async lanes only (`bulk=1,event=1,intr=1,timeout=250ms`):
   - lanes were active (multiple read attempts logged),
   - stall unchanged at `offset=49152` (`chunk 48`).
2. async lanes + `a0d8` handshake:
   - handshake executes but stall remains `offset=49152`.
3. split mode (`desc_split=8192`) + async lanes:
   - stall remains `offset=48128` (`chunk 47`).

Conclusion:
1. thread-level concurrent blocking reads are insufficient.
2. remaining hypothesis is stricter libusb async-transfer behavior
   (persistent pending URBs/event-loop-driven completion semantics).

### True `libusb_submit_transfer` lane path (new)

Replay now includes a true async-submit mode built on raw libusb transfer APIs:
1. `libusb_alloc_transfer` + `libusb_fill_bulk_transfer`/`libusb_fill_interrupt_transfer`
2. initial `libusb_submit_transfer` on `EP 0x81/0x82/0x83`
3. callback-driven resubmission while stream is active
4. dedicated event-loop thread using `handle_events_timeout`
5. explicit transfer cancellation and drain at stream end

Flags:
1. `--param-submit-bulk-in-lanes`
2. `--param-submit-event-lanes`
3. `--param-submit-interrupt-lanes`
4. `--param-submit-buffer-size`
5. `--param-submit-timeout-ms`
6. `--param-submit-event-poll-ms`
7. `--param-submit-log-every`

Pi5 results (reboot-isolated):
1. `tag=2`, `chunk=1024`, `max=65536`, submit lanes `1/1/1`:
   - stall unchanged at `offset=49152` (`chunk 48`)
2. same with `--param-descriptor-split-bytes 8192`:
   - stall unchanged at `offset=48128` (`chunk 47`)
3. lane callbacks were active but mostly `timed_out`; no successful data/event
   completions observed on `0x81/0x82/0x83`.

Interpretation:
1. this rules out "lack of true submitted URBs" as the sole blocker.
2. next gap is likely ordering/transition semantics around runcontrol/doorbell
   and class-2 queue admission, not just read-lane mechanics.

### Readback-coupled gate injection (offset-triggered)

Replay now supports:
1. `--param-gate-known-good-offsets LIST`

Semantics:
1. pause parameter stream at listed cumulative byte offsets,
2. inject known-good control sequence (`a0d4/a704/a33c` reads+writes,
   `a500/a600/a558/a658` writes, `a0d8` read+write),
3. resume stream.

Pi5 findings:
1. gate at `32768` succeeds fully but stall remains at `49152`.
2. gate at `33792` fails immediately (`a0d4` read timeout).
3. offsets `>= 33792` consistently fail gate read/write.

Interpretation:
1. control-plane poison starts around `32KiB..33KiB`,
2. class-2 bulk wall at `49KiB` appears to be a later consequence.

`param_queue_tail` check in known-good pre/post traces:
1. `usbmon_register_map.py sequence` + raw grep did not show `0x00048678`
   (`wValue=8678,wIndex=0004`) or `0x00048688` accesses in captured libedgetpu
   pre/post logs.
2. `0x48678` writes are present in our explicit transition-injection runs.

### Falsification add-ons

Added controls:
1. `--param-require-post-instr-event`
2. `--param-post-instr-event-timeout-ms`
3. `--param-force-full-header-len`

Pi5 outcomes:
1. requiring post-instr event after `exec1` instruction chunk fails (timeout on
   required `0x82` event) before parameter stream starts.
2. forcing full descriptor header length (`header_len=4194304`) under capped
   stream (`65536`) does not change stall; wall remains at `49152`.

### Ordered clean pair capture (known-good first, then replay)

Capture root:
1. `traces/usbmon-goodfirst-20260225T132821Z`

Ordered runs:
1. `good_libedgetpu` first:
   - `inference_benchmark templates/dense_2048x2048_quant_edgetpu.tflite 5 1`
   - success (`Delegate + interpreter ready`, `avg=0.457 ms`)
2. `replay_tag2_submit` second:
   - replay with `tag=2`, `chunk=1024`, `max=65536`, submit lanes `1/1/1`
   - fails at `offset=49152` (`chunk 48`)

Handshake probe on this exact good-vs-bad pair:
1. `tools/usbmon_param_handshake_probe.py ... --threshold 49152`
2. `tools/usbmon_param_handshake_probe.py ... --threshold 33792`

Near-anchor control tuples only in good (same as previous datasets):
1. `Ci:c0:01:a0d8:0001:0004`
2. `Co:40:01:a0d8:0001:0004`
3. `Co:40:01:a33c:0001:0004`
4. `Co:40:01:a500:0001:0004`
5. `Co:40:01:a558:0001:0004`
6. `Co:40:01:a600:0001:0004`
7. `Co:40:01:a658:0001:0004`

Interpretation:
1. clean ordering removes the earlier “poisoned known-good” ambiguity.
2. the near-anchor control cadence gap is reproduced under matched bus/device.

### `connect.py` parity probe: reset-before-claim

Replay now includes opt-in flags:
1. `--reset-before-claim`
2. `--post-reset-sleep-ms`

Pi5 outcome:
1. clean run with reset-before-claim still fails at `offset=49152`.
2. second consecutive run with this mode can fail with `DeviceNotFound` and
   trigger host-side xHCI enumerate errors (`error -62`) until reboot.

Interpretation:
1. reset-before-claim is not sufficient to unlock class-2 admission.
2. keep this mode diagnostic-only on Pi5.

### Deterministic gate-window sweep (new)

Replay now supports repeated near-wall control cadence:
1. `--param-gate-window-start-bytes`
2. `--param-gate-window-end-bytes`
3. `--param-gate-window-step-bytes`

Semantics:
1. once cumulative parameter bytes reach `start`,
2. inject the full known-good gate sequence every `step` bytes until `end`.

Pi5 reboot-isolated outcomes:
1. baseline (`no window`): stall at `49152`.
2. window `32768..49152 step 1024`:
   - gate `32768` succeeds,
   - gate `33792` fails on `a0d4` read timeout.
3. same window + `--param-write-sleep-us 100`:
   - unchanged (`33792` gate read timeout).
4. earlier window `24576..49152 step 1024`:
   - gates `24576..32768` all succeed,
   - first failure still at `33792`.

Interpretation:
1. control collapse is anchored to absolute byte offset (`33792`), not gate count.
2. additional replayed control cadence does not shift the wall.

### Window-gated usbmon parity check

Capture:
1. `traces/usbmon-window-gate-20260225T133858Z-bus4/usbmon-bus4-20260225T133858Z.log`

Probe against known-good (`threshold=33792`):
1. near-anchor control tuples are no longer missing in bad (none-only sets empty).
2. replay still fails immediately at next `a0d4` read on `33792`.

Implication:
1. tuple presence parity is insufficient.
2. remaining blocker is temporal/ordering state progression (queue/ack timing),
   not static control write content.

### Admission-wait timing gate (event/interrupt token probe)

Replay now supports timing-aware admission waits during parameter stream:
1. `--param-admission-wait-mode event|interrupt|either|both`
2. `--param-admission-wait-timeout-ms`
3. `--param-admission-wait-poll-ms`
4. `--param-admission-wait-start-bytes`
5. `--param-admission-wait-end-bytes`
6. `--param-admission-wait-every-chunks`
7. `--param-admission-wait-strict`

Semantics:
1. in the configured byte window, pause after selected chunks,
2. poll `0x82`/`0x83` until condition is satisfied or timeout.

Pi5 reboot-isolated outcomes (`mode=either`, `timeout=50ms`, window
`32768..49152`, every chunk):
1. non-strict:
   - all admission waits timed out (`event_ok=false`, `interrupt_ok=false`),
   - stream still stalled at `49152`.
2. strict:
   - first wait at `32768` timed out,
   - run failed immediately (`admission wait unsatisfied`).

Capture:
1. `traces/usbmon-admission-wait-20260225T134715Z-bus4/usbmon-bus4-20260225T134715Z.log`

Interpretation:
1. no event/interrupt admission token was observed in near-wall window.
2. timing waits on read endpoints alone do not advance class-2 admission.

### Early cadence refinement (`33024` cliff) + gate placement sweep

New deterministic Pi5 matrix (`tag=2`, `chunk=1024`, `max=65536`, submit lanes
`1/1/1`, clean USB power-cycle each run):

Artifacts:
1. `traces/replay-keepalive-cadence2-20260225T135856Z/`
2. `traces/replay-gate-placement-20260225T140440Z/`

Cadence outcomes:
1. baseline (no gates):
   - stream write fails at `49152`.
2. gate window `28672..49152 step 4096`:
   - first gate failure observed at `36864` (`a0d4` read timeout).
3. gate window `28672..49152 step 1024`:
   - first gate failure at `33792`.
4. gate window `32768..34816 step 256`:
   - first gate failure at `33024`.
5. gate window `28672..34816 step 256`:
   - first gate failure also at `33024`.

Probe-derived payload totals (`usbmon_param_handshake_probe`):
1. baseline bad phase payload: `50176`.
2. 4KiB cadence: `36864`.
3. 1KiB cadence: `33792`.
4. 256B cadence variants: `33792`.

Replay now exposes gate timing mode:
1. `--param-gate-placement before|after|both` (default `before`).

Placement sweep (window `28672..34816 step 256`) on Pi5:
1. `before`: first failing gate `33024`.
2. `after`: first failing gate `33024`.
3. `both`: first failing gate `33024`.

Interpretation:
1. refined control-plane cliff is at `33024` for gate CSR reads, with
   practical payload ceiling at `33792` under dense gate probing.
2. sparse cadence can delay when failure is *observed* (e.g. next gate at
   `36864`) but does not remove collapse.
3. gate execution timing relative to bulk writes does not change the boundary,
   reinforcing that missing state progression is deeper than tuple presence or
   simple gate ordering.

### Per-chunk CSR snapshot telemetry (queue/runcontrol set)

Replay now supports explicit near-wall CSR snapshots:
1. `--param-csr-snapshot-start-bytes`
2. `--param-csr-snapshot-end-bytes`
3. `--param-csr-snapshot-every-chunks`
4. `--param-csr-snapshot-on-error`

Captured register set:
1. `scalarCoreRunControl` (`0x44018`)
2. instruction queue:
   - `base` (`0x48590`)
   - `tail` (`0x485A8`)
   - `completed_head` (`0x485B8`)
   - `int_status` (`0x485C8`)
3. parameter queue:
   - `base` (`0x48660`)
   - `tail` (`0x48678`)
   - `completed_head` (`0x48688`)
   - `int_status` (`0x48698`)

Pi5 artifact:
1. `traces/replay-csr-snapshot-20260225T141800Z/`

Observed in both baseline and dense-gated runs:
1. at offsets `30720`, `31744`, `32768`:
   - all tracked CSRs read back `0x0`.
2. at `33792`:
   - all tracked CSR reads time out (`read64` and fallback `read32`).
3. no non-zero queue interrupt status was observed before CSR liveness loss.

Implication:
1. the wall is an abrupt control-plane non-responsiveness transition rather than
   an observed queue status bit transition in readable CSRs.
2. this further supports a deeper firmware/runtime state progression failure
   (host-visible control plane dies) rather than missing static tuple content.

## Practical next debug steps

1. Instrument runcontrol/doorbell CSR state immediately before and after each
   descriptor class submission to correlate queue-pressure and timeout offset.
2. Add endpoint drain checks (`0x82`, `0x83`) during large parameter streaming
   to detect required host-ack behavior.
3. Use `tools/usbmon_side_by_side_capture.sh` on Pi5 to collect:
   - known-good `libedgetpu` invoke
   - Rust replay sweep including `tag2` timeout case
   then diff packet ordering/size cadence between lanes.
4. Probe descriptor scheduling permutations:
   - interleave param chunks with event reads,
   - split PARAMETER_CACHING into smaller phased submissions,
   - vary setup read/write inclusion and strictness.
