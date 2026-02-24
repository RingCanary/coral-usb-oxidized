# Primary Outcome Focus (Four Points Only)

This project now stays centered on four long-standing priorities. Every major
change should map to one (or more) of these points.

## 1. Control-Plane Map (Protocol Ground Truth)

- Goal:
  Identify exact semantic roles for the invariant setup/teardown control writes
  (`wValue`/`wIndex` register paths), including queue base, doorbell, status,
  and reset/ack flow.
- Current status:
  Partial. Address clusters are known and stable (`a30c`, `4018`, `8788`, ...),
  but symbolic meaning is still hypothesis-level.
- Success signal:
  A named register map where each critical control write is explained and tested
  against usbmon traces.

## 2. Pure Rust USB Driver (No `libedgetpu`)

- Goal:
  Execute model invoke lifecycle through `rusb` only:
  device transition, control sequence, bulk submit/reap, completion polling.
- Current status:
  Early. Baseline control-plane probe exists, but full invoke replay still
  depends on `libedgetpu`.
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
