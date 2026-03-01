# USB RE Campaign (#2-#6) - 2026-03-01

Scope: execute and close the five active tracks from the live plan:
1. `#2` URB metadata diff
2. `#3` async event-loop parity
3. `#4` deterministic fault signatures at cliff offsets
4. `#5` header grammar mining across known-good traces
5. `#6` USB descriptor/config parity (boot vs runtime)

## Strategy

1. Use one Pi5 target (`rpilm3.local`) with reboot-first runs where poisoning is likely.
2. Collect paired artifacts (known-good vs replay) for syscall-level and libusb-level views.
3. Keep tests narrow: change one dimension at a time (submit scheduling, chunk size, snapshot windows).
4. Promote only results with explicit artifact paths and reproducible commands.

## Results

### #2 URB metadata diff (syscall + interposer)

Artifacts:
- `traces/analysis/urbmeta-20260301T070810Z/` (strace)
- `traces/analysis/interposer-20260301T071059Z/` (libusb interposer)

Key observations:
1. Syscall profile diverges strongly:
   - known-good: `SUBMITURB=123`, `REAPURBNDELAY=233`, `DISCARDURB=10`
   - replay: `SUBMITURB=228`, `REAPURBNDELAY=416`, `DISCARDURB=182`
2. Replay shows heavy cancel/discard churn vs known-good.
3. Interposer confirms instruction payload SHA mismatch still exists for `len=2608` and `len=9872`.
4. Replay run in this pair used `tag2 len=131072` stream chunks and never reached known-good-style `1MiB x4` parameter payload cadence.

Takeaway:
- We have transport-side behavioral divergence (especially cancellation churn), but this run alone does not prove a single URB flag mismatch root cause.

### #3 async event-loop parity (reboot-first)

Artifacts:
- `traces/analysis/async-parity-reboot-20260301T072113Z-local/`
- `traces/analysis/async-parity-reboot-20260301T072155Z-global/`

Protocol:
1. Reboot.
2. Run submit mode (`--param-submit-*`) without global preposted lanes.
3. Reboot.
4. Run same config with `--param-submit-global-lanes`.

Outcome:
1. Both modes fail at same class-2 wall.
2. Both produce inflight bulk-out timeouts centered at `offset=49152` (`chunk 48`) with cleanup failures near `50176` and `51200`.

Takeaway:
- Keeping global submit lanes alive across the whole replay is not sufficient to cross the wall.

### #4 deterministic fault signatures (cliff table)

Artifacts:
- `traces/analysis/fault-signature-20260301T072317Z/`
- `traces/analysis/fault-signature-20260301T073000Z-submit/`

Signature A (blocking write path):
1. Config: `chunk=256`, no submit mode, probe offsets include `33024`.
2. Failure at `offset=33024`.
3. Poison probe at `33024`:
   - `usbTopInterruptStatus (0x0004c060)` read timeout
   - `scu_ctr_7 (0x0001a33c)` read timeout
4. CSR snapshot reason is `stream_write_error`.

Signature B (submit path):
1. Config: `chunk=1024`, submit mode enabled, probe around `49KiB`.
2. Poison probe at `49152` times out.
3. Inflight failures appear at `47104`, then cleanup fails at `48128` and `49152`.

Takeaway:
- Two reproducible cliff classes exist: early control-plane death near `33KiB` and bulk/inflight saturation near `49KiB`.

### #5 header grammar mining

Corpus scan on Pi5:
- selected logs: `18`
- logs with valid 8-byte Bo headers: `15`

Global tag/length counts:
1. `tag=0 len=9872` count `32`
2. `tag=1 len=2048` count `22`
3. `tag=0 len=2608` count `15`
4. `tag=2 len=65536` count `10` (replay)
5. `tag=2 len=4194304` count `5` (known-good)

Sequence pattern:
1. Known-good prefix repeatedly shows:
   - `(2608, tag0) -> (4194304, tag2) -> (9872, tag0) -> (2048, tag1) ...`
2. Replay prefix commonly shows:
   - `(9872, tag0) -> (2608, tag0) -> (65536, tag2)`

Takeaway:
- Header grammar is stable and simple; divergence is not in header encoding itself, but in runtime/topology behavior during class-2 admission.

### #6 descriptor/config parity (boot vs runtime)

Artifacts:
- Boot descriptor: `traces/analysis/descriptor-parity-20260301T084325Z/`
- Runtime descriptor: `traces/analysis/descriptor-parity-runtime-20260301T084916Z/`

Boot mode (`1a6e:089a`):
1. DFU-style interface (`Class=254`, `SubClass=1`, `Protocol=2`).
2. `bNumEndpoints=0`.

Runtime mode (`18d1:9302`) after `delegate_usage`:
1. Vendor interface (`Class=255`, `SubClass=255`, `Protocol=255`).
2. `bNumEndpoints=6`.
3. OUT: `0x01`, `0x02`, `0x03` bulk, `wMaxPacketSize=1024`, `bMaxBurst=15`.
4. IN: `0x81`, `0x82` bulk, `wMaxPacketSize=1024`, `bMaxBurst=15`.
5. IN: `0x83` interrupt, `wMaxPacketSize=64`, `bMaxBurst=0`.

Takeaway:
- Descriptor parity aligns with expected multi-endpoint runtime topology; no descriptor-level anomaly explains the wall.

## Consolidated Verdict

1. `#3` and `#6` falsify the simplest transport hypotheses:
   - global async lane lifetime alone does not unlock class-2,
   - descriptor/config mismatch is not the blocker.
2. `#4` confirms deterministic cliff signatures with device-side control path collapse under load.
3. `#5` confirms header grammar stability; the protocol words themselves are not malformed.
4. Highest-value next branch is still runtime state semantics during class-2 admission (bridge/core progress dependency), not further static descriptor/header tweaking.

## Update: Bootstrap Ordering Shim (post-campaign)

Date: 2026-03-01 (later same day)

Hypothesis:
1. Replay startup order was inverted versus known-good traces.
2. Replay did: `9872(tag0) -> 2608(tag0) -> 4194304(tag2)`.
3. Known-good did: `2608(tag0) -> 4194304(tag2) -> 9872(tag0) -> 2048(tag1)`.

Implementation:
1. Added CLI flag `--bootstrap-known-good-order` in `examples/rusb_serialized_exec_replay.rs`.
2. With this flag:
   - bootstrap sends PARAMETER_CACHING first,
   - EXECUTION_ONLY is deferred to run phase,
   - bootstrap event read is skipped until run phase.

Validation:
1. Reboot-first A/B:
   - baseline (flag off): fails at class-2 wall (`offset 0`, `actual=49152` on first 1MiB submit).
   - flag on: succeeds end-to-end.
2. Reboot-first repeatability:
   - two consecutive successful runs (`rc=0`) with identical output hash.
3. usbmon capture of successful reordered path:
   - headers observed in exact known-good order:
     - `300a0000 00000000` -> `len=2608 tag=0`
     - `00004000 02000000` -> `len=4194304 tag=2`
     - `90260000 00000000` -> `len=9872 tag=0`
     - `00080000 01000000` -> `len=2048 tag=1`

Interpretation:
1. The class-2 wall is strongly coupled to descriptor ordering/state progression.
2. This is not just transfer depth/cadence; topology order itself is load-bearing.
3. This is the first reproducible bypass of the 49KiB wall in pure replay path for this model family.
