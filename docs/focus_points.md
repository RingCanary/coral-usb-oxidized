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
  Early-mid. Probe now supports vendor CSR read/write plus event (`0x82`) and
  interrupt (`0x83`) decoding, but full invoke replay still depends on
  `libedgetpu`.
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
