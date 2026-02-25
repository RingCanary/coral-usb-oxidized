# Descriptor-Ring Contract Handoff (Perturbation Campaign)

## 1) Why this handoff exists

This note expands the "tough nut" in
`docs/coral_edgetpu_re_analysis_2026-02-24.md` with concrete on-device evidence.
It is meant for reasoning/web agents that cannot run commands.

Core problem: we still do not know the exact host->TPU descriptor/queue
contract, especially the semantics of `scalarCoreRunControl` transitions.

---

## 2) Test setup and method

Host:
- Raspberry Pi 5 (`rpilm3.local`)
- Coral USB in runtime mode (`18d1:9302`)

Probe binary:
- `examples/gemm_csr_perturb_probe.rs`

Method:
1. Build + run one known-good bundled GEMM invoke (`2048`, identity, `runs=1`).
2. Before invoke, mutate exactly one CSR write (`write32`/`write64`), read back.
3. Run invoke and classify result.
4. Restore original CSR value and validate restore readback.

Automation artifacts:
- Script: `tools/csr_perturbation_matrix.sh`
- Benign matrix results:
  - `traces/csr-perturb-benign-20260224T191025Z/summary.tsv`

Poison capture artifact:
- `traces/usbmon-20260224T191133Z-bus4/usbmon-bus4-20260224T191133Z.log`
- `traces/usbmon-20260224T191133Z-bus4/register-report.json`
- `traces/usbmon-20260224T191133Z-bus4/register-seq-known-writes.json`
- `traces/usbmon-20260224T191133Z-bus4/bulk-signature.json`

---

## 3) Stable perturbation results (non-poison cases)

From `traces/csr-perturb-benign-20260224T191025Z/summary.tsv`:

1. All tested cases completed GEMM successfully (`status=ok`), including:
   - `runcontrol_0` (`0x44018 <- 0`)
   - `tileconfig` variants (`0`, `1`, `0x7f`)
   - `scu_ctr_7` low/high masks
   - `rambist_ctrl_1` low/high variants
   - `slv_abm_en` and `mst_abm_en` toggles
2. Output prefix remained identical to baseline:
   - `output_head=0,1,2,...,15`
3. Restore succeeded in all benign cases.

Interpretation:
- Most single-register changes tested here are either tolerated, overwritten
  internally, or not critical at this exact invoke boundary.

---

## 4) Poison boundary: `scalarCoreRunControl = 2`

Target case:
- `0x00044018 (scalarCoreRunControl) <- 0x2` before invoke.

Observed runtime failure:
- Process abort during invoke:
  - `transfer on tag 2 failed. Abort. Deadline exceeded: USB transfer error 2 [LibUsbDataOutCallback]`
- After this, subsequent `delegate_usage` attempts fail with:
  - `Failed to create EdgeTPU delegate`
- Device remains enumerated as `18d1:9302` but enters repeated reset behavior in
  kernel log (`usb 4-1: reset SuperSpeed USB device ...`).

Important empirical clue from user observation:
- During this poison case, onboard white LED appeared brighter/blinking.
- This suggests transition `runcontrol=2` may enter a fault/active test state or
  altered power/clock domain state.

---

## 5) Poison trace evidence (control sequence)

From `register-seq-known-writes.json` in poison run:

- `tileconfig0 <- 0x7f`
- `scalarCoreRunControl <- 0x1`
- `tileconfig0 <- 0x7f`
- `scalarCoreRunControl <- 0x1`
- `scalarCoreRunControl <- 0x0`
- `scalarCoreRunControl <- 0x2`  <-- poison pivot
- `tileconfig0 <- 0x7f`
- later again: `scalarCoreRunControl <- 0x2`

From `register-report.json`:
- `vendor_control=153`
- top ops include:
  - `00044018::read64` (7)
  - `00044018::write64` (6)
  - `00048788::write64` (5)

Interpretation:
- `0x44018` is not a static config register; it is an actively driven control
  state machine input used multiple times during setup and failure handling.

---

## 6) Current hypothesis set (for reasoning agents)

H1. `scalarCoreRunControl` encodes run-state transitions, not a persistent mode
bit.
- Readback often remains `0` even after writes of `1`/`2`, implying command-like
or edge-triggered semantics.

H2. Value `2` is a privileged transition requiring preconditions.
- If descriptor queues or completion rings are not in expected state, `2`
  triggers fatal transport path (`tag 2` failure), after which delegate init
  cannot complete.

H3. `tileconfig0` is a necessary companion gate but not sufficient.
- Varying `tileconfig0` alone did not poison.
- Poison requires `runcontrol=2` path.

H4. LED brightness/blink during poison is likely a side effect of altered
power/clock/interrupt state, not a dedicated LED register path.

---

## 7) Questions to answer via reasoning/web research

1. In open-source libedgetpu / DarwiNN docs, what exact state transitions are
   associated with `scalarCoreRunControl` values `0/1/2`?
2. Is there a documented or inferred recovery sequence from a failed
   `runcontrol=2` transition without physical replug?
3. Which CSRs are coupled with runcontrol transitions (queue head/tail pointers,
   completion heads, fabric enables)?
4. Is USB transfer `tag 2` mapped to a specific descriptor queue class in public
   code/history?
5. Could repeated `runcontrol=2` writes correspond to reset/quit-reset phases
   that assume specific SCU state (`scu_ctrl_2/3`)?

---

## 8) Suggested reasoning-agent prompt seed

Use this as a direct prompt:

1. "Given repeated writes to `scalarCoreRunControl (0x44018)` with values
   `{1,0,2}` and poison on `2`, infer likely state machine semantics from
   libedgetpu source and Beagle/DarwiNN register helpers."
2. "Map USB error `transfer on tag 2 failed` to queue/descriptor classes in
   EdgeTPU runtime."
3. "Propose a minimal, no-replug recovery control sequence and justify each CSR
   write."

---

## 9) Practical next experiment (execution agent side)

After research hypotheses are returned:
1. Test candidate recovery sequence immediately after poison.
2. Validate with:
   - `cargo run --example delegate_usage`
   - one GEMM invoke (`gemm_csr_perturb_probe none ...`)
3. Capture usbmon for both failed and recovered paths.

---

## 10) Addendum: Clean A/B run (`runcontrol=1` vs `runcontrol=2`)

Date: 2026-02-24 (Pi5, clean reattach state)

Artifact:
- `traces/led-poison-ab-20260224T194024Z/`

Results:
1. Baseline (`none`) was healthy:
   - `RESULT status=ok`
   - `output_head=0..15`
2. `runcontrol=1` remained healthy:
   - `RESULT status=ok`
   - `output_head=0..15` (no output drift)
3. `runcontrol=2` triggered immediate fatal path:
   - `transfer on tag 2 failed ... USB transfer error 2`
4. Post-poison, delegate creation failed while device still enumerated
   (`18d1:9302`).

This confirms the contract requirement:
- device presence (`18d1:9302`) is not equivalent to runtime readiness.
- readiness must include a delegate-create/invoke health probe.

### 10.1 State model update

Observed staged states:
1. `Healthy` (delegate + invoke succeed)
2. `Enumerated-But-Unusable` (`18d1:9302` visible, delegate create fails)
3. `Off-Bus/Recovering` (device disappears; later reattach restores health)

User-visible indicator:
- white LED blinking/brighter starts near poison trigger (`runcontrol=2`), then
  later stops before off-bus phase.

### 10.2 No-replug recovery attempts (new evidence)

Artifact:
- `traces/runcontrol-recovery-20260224T194932Z`

Key result:
1. Starting from healthy state, poison was reproduced once.
2. Immediate recovery attempts failed:
   - `runcontrol=1` write -> timeout
   - `runcontrol=0` then `runcontrol=1` -> timeout
   - `--reset-device` did not restore delegate readiness
3. Delegate remained broken post-poison, and later no device was detected by
   delegate probe.

Interpretation update:
- A deeper poisoned state can disable effective host control writes to
  `scalarCoreRunControl`, not just inference submission.
- Recovery likely requires transport-level or physical re-enumeration once this
  point is reached.
