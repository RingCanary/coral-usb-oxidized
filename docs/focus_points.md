# Primary Outcome Focus (Four Points Only)

This project now stays centered on four long-standing priorities. Every major
change should map to one (or more) of these points.

## 1. Control-Plane Map (Protocol Ground Truth)

- Goal:
  Identify exact semantic roles for the invariant setup/teardown control writes
  (`wValue`/`wIndex` register paths), including queue base, doorbell, status,
  and reset/ack flow.
- Current status:
  Partial. Core mappings are now verified for:
  - `0x44018 -> scalarCoreRunControl`
  - `0x48788 -> tileconfig0`
  - `0x1a30c -> scu_ctrl_0`
  Behavioral boundary now verified on Pi5:
  - `runcontrol=1` is safe in invoke path.
  - `runcontrol=2` reliably poisons runtime
    (`USB transfer error 2`, delegate failure while still enumerated).
  No-replug recovery sequence (`2->1`, `2->0->1`, reset-device) failed in
  latest run due post-poison control-write timeouts; reattach still required in
  worst case.
  Remaining register groups still need semantic confirmation.
- Success signal:
  A named register map where each critical control write is explained and tested
  against usbmon traces.

## 2. Pure Rust USB Driver (No `libedgetpu`)

- Goal:
  Execute model invoke lifecycle through `rusb` only:
  device transition, control sequence, bulk submit/reap, completion polling.
- Current status:
  Mid. New `EdgeTpuUsbDriver` now implements descriptor framing (`len+tag`),
  chunked bulk sends, event/interrupt decode, and serialized executable replay
  scaffolding in pure `rusb`.
  Pi5 clean-start runs now validate:
  - full VBUS software power-cycle (`uhubctl`) reliably returns device to
    `1a6e:089a`,
  - firmware upload + setup + EXECUTION_ONLY submission succeeds,
  - `--skip-param-preload` path completes with event+output.
  Remaining blocker is narrowed:
  large PARAMETER_CACHING payload admission stalls on descriptor classes
  `0/1/2` (offset-dependent timeout), while alternate classes (`3/4`) do not
  stall but are not yet confirmed as semantically correct parameter loads.
  Latest probe status:
  stream chunking, pacing, event/interrupt polling, and multi-descriptor
  segmentation do not remove the `~0xC000` cumulative ingress wall for class-2.
  This now strongly suggests missing runcontrol/doorbell queue-state transitions
  rather than transport framing bugs.
  Additional confirmation:
  after class-2 stall, control-plane CSR read/write probes time out (poisoned
  runtime), while the same probes succeed after healthy skip-param invokes.
  Ordered clean capture confirmation:
  known-good-first (`libedgetpu`) then replay (`tag=2`) reproduces the same
  near-anchor tuple gap (`a0d8/a33c/a500/a558/a600/a658` only in good) while
  replay still stalls at `49152`.
  New deterministic gate-window sweep:
  replay can now inject repeated known-good control gates every `1 KiB` in a
  byte window, and this closes tuple-level near-anchor diff; however, failure
  remains pinned at `33792` (`a0d4` read timeout). This narrows the blocker to
  timing/state progression semantics, not missing tuple content.
  Admission-wait timing probe:
  replay can pause in `32768..49152` and poll `0x82/0x83` for explicit
  admission tokens; no tokens were observed (`event_ok=false`,
  `interrupt_ok=false`), and class-2 still stalls.
  Refined cadence boundary:
  dense gate windows (`step=256`) expose first gate read timeout at `33024`
  (`a0d4`), with practical stream ceiling `33792`; sparse cadence can defer
  observed failure to next gate point (e.g. `36864` at `step=4096`).
  Gate placement invariance:
  `--param-gate-placement before|after|both` does not move the `33024` cliff.
  CSR snapshot telemetry:
  per-chunk queue/runcontrol snapshots (`0x44018`, `0x48590/85a8/85b8/85c8`,
  `0x48660/8678/8688/8698`) read `0x0` through `32768`, then all reads time out
  at `33792`; no non-zero queue int-status was observed before collapse.
  Parity reset probe (`--reset-before-claim`) did not move the wall and is
  unstable across consecutive Pi5 runs (can trigger xHCI enumerate `error -62`);
  keep diagnostic-only.
  Fuzz addendum:
  sustained control-plane perturbation during class-2 streaming moves the stall
  boundary earlier (~33-40 KB vs 49 KB baseline), while one-shot deterministic
  runcontrol injections do not. This indicates dynamic queue/scheduler coupling.
  usbmon-parallel confirmation:
  the shifted boundaries correlate directly with fewer accepted Bo(1024)
  completions in runtime traces (baseline 48, readonly 34, runctl 33).
- Success signal:
  A Rust example that runs one known template end-to-end and returns valid output
  without `libedgetpu`.

## 3. EXECUTION_ONLY Semantics (ISA / Descriptor RE)

- Goal:
  Move from opaque chunk hashes to structural understanding of instruction words
  (which regions are relocations vs real op encoding, and which deltas correlate
  with op/shape changes).
- Current status:
  Early-mid. Schema-aware executable parsing is done; dedicated chunk diffing is
  now available, but instruction meaning is not decoded.
- Success signal:
  Repeatable opcode/field hypotheses with evidence across model families
  (Dense/Conv/Depthwise and shape sweeps).

## 4. Graph Lowering Path (Compiler Independence)

- Goal:
  Replace template-only injection with a lightweight lowering path from model
  tensors/ops to executable command streams and USB submission.
- Current status:
  Not started in implementation; requires progress on points 1-3.
- Success signal:
  Running a non-template dynamic graph segment without `edgetpu_compiler`.

## Operating Rule

If a proposed task does not clearly improve one of these four points, it is
deferred.
