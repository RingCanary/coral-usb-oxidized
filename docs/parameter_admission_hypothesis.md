# Parameter Admission Protocol Hypothesis

## Core Premise
The class-2 (Parameter) ingress wall at ~33KiB (dense control cliff) to ~49KiB (bulk timeout cliff) is not caused by a missing setup control transfer, nor by a missing `param_queue_tail` write.
Instead, it is a **hardware execution backpressure stall**.

### The Mechanism
1. The USB bridge hardware automatically parses the 8-byte `[length][tag]` bulk-out headers and routes payloads to the respective SRAM queues (Instruction, Parameter, Input).
2. The bridge *hardware* automatically increments the queue tail pointers (e.g., `param_queue_tail` 0x48678) as data lands in the FIFOs. This is why `libedgetpu` never writes to them.
3. The parameter stream acts as a ring buffer/FIFO into the TPU's matrix MAC units.
4. **The Cliff:** The TPU core only drains the parameter queue when an *Instruction* tells it to execute a layer or cache a weight.
5. In the `PARAMETER_CACHING` phase (executable type 1), if the host blasts 4MiB of parameters *without interleaving or co-executing the corresponding instruction bitstreams that consume them*, the 32KiB parameter SRAM fills up.
6. Once the 32KiB SRAM is full, the 16KiB USB endpoint FIFO fills up (total ~48-49KiB).
7. When the USB FIFO is full, the entire AXI/SCU interconnect backpressures. The control plane (`0x0001a0d4` etc.) shares this interconnect and violently locks up (the 33KiB control cliff).

## 1. Prove/Disprove Claim
> “parameter queue drain only advances after a specific EP0 control transition tied to runtime state, not just bulk headers.”

**Investigation Ongoing.** While all tested static handshake variants (including deterministic single/multi-gate insertions and pre/post payload polling) have been insufficient to advance the queue, this does *not* confirm that no host-visible ACK/credit exists. Current replay evidence leaves open the possibility of a missing state progression or a dynamically formatted host-ack behavior that we have not yet reproduced.

## 2. Minimal State Machine for Ingress Near the Cliff
The actual state machine is a consumer-producer loop:
`Host [Bulk-Out Bo(tag=0)] -> TPU Inst Queue` -> `Host [Bulk-out Bo(tag=2)] -> TPU Param Queue`
`TPU Core executes Inst -> TPU pops Param Queue -> frees SRAM -> Host can send more Bo(tag=2)`

**Guard Conditions:**
* `Bo(tag=2) bytes_sent - TPU_params_consumed < 32768 (SRAM size)`
If this condition is violated, the interconnect hangs.

## 3. Reverse-Mapping Key Control Tuples
Based on the `REGISTER_MAP` and standard ASIC design:
* `0x0001a0d4` (`omc0_d4`): Likely an interconnect/memory-controller status or error register. Reading it checks for AXI stalls. It timing out at 33KiB means the AXI bus is wedged.
* `0x0001a33c` (`scu_ctr_7`): System Control Unit (SCU) status. Checking if the TPU core hit a fault.
* `0x0001a704` (`rambist_ctrl_1`): RAM Built-In Self Test... or generic SRAM status.
* `0x0001a0d8` (`omc0_d8`): Often used as a flush or cache-invalidate token (`write 0x80000000`) before starting new descriptor sequences.

## 4. PARAMETER_CACHING Instruction Dependency
> Determine whether PARAMETER_CACHING requires concurrent instruction progress tokens (or run-control edges) to consume params...

**Yes, but "instructions-before-params" mapping alone is insufficient.** Replay traces demonstrate that instruction chunks *are* sent before parameters in both preload and bootstrap paths ([examples/rusb_serialized_exec_replay.rs](file:///home/bhav/Documents/experiments/rngcnr-gh/coral-usb-oxidized/examples/rusb_serialized_exec_replay.rs)), yet the stall still occurs. This indicates that simply queuing instructions into the Instruction SRAM is not enough to induce parameter consumption. The missing condition is likely the actual instruction *progress/completion* semantics (i.e. runtime state boundaries, execution triggering) rather than just the structural presence of the instruction bitstreams.

---

## 5. ranked, falsifiable transaction scripts to run on Pi5

Here are 3 scripts to empirically prove the TPU backpressure and instruction-dependency hypothesis.

### Script 1: The Instruction Interleave (The "Fix")
**Hypothesis:** If we send valid Instructions (tag 0) *interleaved* with or specifically chunked ahead of the Parameters (tag 2) using precise boundary conditions/credits, the parameter queue will drain, and we will cross the 49KiB wall.
**Action:**
1. Send an exact mapped instruction slice (tag 0) that functionally maps to the initial parameter bytes.
2. Send exactly 32KiB of parameters (tag 2).
3. Attempt to poll for completion (`0x82` Event or `0x83` Interrupt) tied to the instruction execution.
4. Send the next chunk series.
**Expected Result:** The bulk-out stream successfully passes the 49KiB wall, demonstrating that parameter admission requires precise instruction execution progress/interleaving semantics, not just a front-loaded dump.

### Script 2: Queue Status Telemetry at the Cliff (The "Probe")
**Hypothesis:** Right before the control plane dies (e.g., at 32000 bytes of parameter payload), the queue status registers might reveal consumption state, although these registers are known to read zero pre-cliff and timeout at the cliff. Tracking their state up to the edge validates whether they are banked/unreadable or simply stationary.
**Action:** Use `--param-gate-known-good-offsets 32000` but replace the gate injection with CSR reads of:
*   `0x00048660` (param_queue_base)
*   `0x00048678` (param_queue_tail)
*   `0x00048688` (param_queue_completed_head)
*   `0x00048698` (param_queue_int_status)
**Expected Result:** The script assesses register observability. If `completed_head` is zero while `tail` is zero, these registers are banked/unobservable. If `tail` increments but `completed_head` remains zero before the timeout, it directly evidences a queue stall independent of observability.

### Script 3: The Interconnect/Handler Halt Differentiator (The "Poison")
**Hypothesis:** The control plane timeout at 33024 bytes is caused by deep queue backpressure, but this backpressure could be either a low-level AXI bus lockup (hardware) OR a firmware/endpoint handler deadlock (software).
**Action:**
1. Send 33024 bytes of Parameters (tag 2) to trigger the poison state.
2. Attempt to read an entirely unrelated, harmless USB bridge register (e.g., `usbTopInterruptStatus` 0x0004c060) and contrast it against an SCU region register (e.g. `0x0001a33c`).
**Expected Result:** If both the bridge register and the SCU register time out, the root cause is likely a firmware/USB-endpoint handler deadlock. If the bridge register succeeds but the SCU register times out, the root cause is an AXI bus wedge.
