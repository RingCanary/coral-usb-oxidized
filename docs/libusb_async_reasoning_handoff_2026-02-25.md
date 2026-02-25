# Coral USB RE - Async URB Handoff (2026-02-25)

## Goal

Break the class-2 (`Parameters`) ingress wall in pure-`rusb` replay so we can
stream full parameter payloads and reach stable execution without `libedgetpu`.

Observed wall in replay:
1. non-split stream: bulk-out timeout at `49152` bytes (`chunk 48`, `1024` size)
2. split descriptors (`8192`): timeout at `48128` bytes (`chunk 47`)

This is reproducible on Pi5 after clean USB power-cycle.

## What is already proven

1. Descriptor framing is correct:
   - `[len_le32][tag_u8][pad_u24]`
2. Control register setup path works from clean boot to runtime.
3. Instruction descriptors are accepted.
4. Failure is specific to class-2 data-plane admission during parameter stream.
5. Synchronous reads/polls do not fix the wall.
6. Thread-concurrent blocking reads do not fix the wall.
7. True async submitted URBs (`libusb_submit_transfer`) on `0x81/0x82/0x83`
   also do not fix the wall.

## Known-good vs replay delta (usbmon)

Near parameter ingress in known-good `libedgetpu` traces, we repeatedly observe:
1. `Ci/Co` exchange at `a0d8`
2. pre-posted `Bi`/`Ii` submits
3. then parameter `Bo` stream

Replay traces around the wall:
1. dominated by `Bo` writes
2. no equivalent successful `Bi`/`Ii` completion pattern
3. still times out at the same boundary even when async URBs are active

## New experiment implemented (this patch)

File: `examples/rusb_serialized_exec_replay.rs`

Added submit-lane mode:
1. allocate transfer per lane via `libusb_alloc_transfer`
2. `fill_bulk` for `0x81`/`0x82`, `fill_interrupt` for `0x83`
3. submit with `libusb_submit_transfer`
4. callback resubmits until stop flag
5. event-loop thread (`handle_events_timeout`) drives completions
6. cancel + drain on exit

CLI:
1. `--param-submit-bulk-in-lanes`
2. `--param-submit-event-lanes`
3. `--param-submit-interrupt-lanes`
4. `--param-submit-buffer-size`
5. `--param-submit-timeout-ms`
6. `--param-submit-event-poll-ms`
7. `--param-submit-log-every`

Validation result:
1. callbacks fire repeatedly (`~60` callbacks/lane over probe window)
2. status mostly `timed_out`, then `cancelled`
3. no movement of `49KiB`/`48KiB` wall

## Precise unresolved technical question

What **additional host-observable transition** is required for class-2 queue
admission besides:
1. correct descriptor format,
2. correct setup CSR writes,
3. pre/post control `a0d8` pulse,
4. pending async URBs on read endpoints?

Candidate classes:
1. runcontrol/doorbell ordering and exact timing edges
2. missing queue-tail/head acknowledgments via specific CSR reads/writes
3. required event/interrupt completion handshake before each descriptor window
4. endpoint/type mismatch for a control-relevant lane during parameter phase
5. hidden state dependency between runtime setup step subsets and class-2 ingest

## Falsifiable next tests

1. **Deterministic transition sweep with submit lanes enabled**:
   - inject fixed runcontrol/doorbell sequences at chunk indices `32/40/47/48`
   - compare wall movement and callback status distribution.
2. **Readback-coupled stream gating**:
   - pause parameter writes every N chunks until a specific CSR or event bit flips
   - treat this as admission token.
3. **Descriptor-window binary search**:
   - vary descriptor split sizes around first failure window while keeping total
     bytes fixed (`65536`) and submit lanes active.
4. **`a0d8` value/timing sweep**:
   - pulse at per-descriptor boundaries with value variants and inter-pulse delays.
5. **Known-good side-by-side alignment**:
   - align by cumulative parameter bytes and diff control tuple sequence exactly in
     the `40KiB..52KiB` window.

## New high-signal boundary finding (2026-02-25, second pass)

Readback-coupled gate injections at fixed byte offsets were added and tested on
Pi5 with clean power-cycle per run.

Gate sequence replayed:
1. `a0d4` read/write (`write=0x80000001`)
2. `a704` read/write (`write=0x0000007f`)
3. `a33c` read/write (`write=0x0000003f`)
4. writes `a500=1`, `a600=1`, `a558=3`, `a658=3`
5. `a0d8` read/write (`write=0x80000000`)

Observed:
1. gate at `32768` executes fully, but stream still stalls at `49152`.
2. gate at `33792` fails immediately on first gate read (`a0d4` timeout).
3. all later offsets (`36864+`) also fail immediately.

Implication:
1. control plane becomes non-responsive in `32KiB..33KiB` range,
2. bulk-out path remains writable until `49KiB`,
3. the causal failure likely begins earlier in scheduler/queue state than the
   visible bulk timeout.

This narrows the tough nut:
1. identify what state advancement libedgetpu triggers between `32KiB` and
   `49KiB` that keeps control plane alive (not just static control writes).

## Direct check: `param_queue_tail` in known-good trace

Using `tools/usbmon_register_map.py sequence` plus raw grep on:
1. `.../libedgetpu_known_good/pre/*.log`
2. `.../libedgetpu_known_good/post/*.log`

Current evidence:
1. no observed control transfers to `0x00048678` (`param_queue_tail`) or
   `0x00048688` (`param_queue_completed_head`) in those known-good pre/post logs.
2. `0x48678` control writes are visible in our own transition-injection runs.

Implication:
1. the queue-tail hypothesis is still plausible at architecture level,
2. but this specific known-good capture set does not directly confirm it via
   endpoint-0 CSR writes.

## Artifacts to inspect

1. replay submit-URB capture:
   - `traces/usbmon-20260225T121056Z-bus4/usbmon-bus4-20260225T121056Z.log`
   - `traces/usbmon-20260225T121056Z-bus4/usbmon-bus4-20260225T121056Z.summary.txt`
2. prior transition sweep root:
   - `traces/usbmon-transition-fixed-20260225T082936Z-bus4`
3. parser/diff tooling:
   - `tools/usbmon_param_handshake_probe.py`
   - `tools/usbmon_bulk_signature.py`
   - `tools/usbmon_register_map.py`
