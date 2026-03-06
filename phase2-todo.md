# Phase 2 TODO (M5-M8)

- [x] M5: Generate family transition map from size table and refresh on every new dim compile.
- [x] M5: Cover recurrent families with >=3 members where missing:
  - [x] EO/PC `9872/2608`: compiled + profiled 3rd member (`3072x2048`).
  - [x] EO/PC `7056/1840`: compiled + profiled 3rd member (`640x1280`, rectangular).
- [x] M5: For each recurrent family, produce:
  - [x] `safe_core` patchspec
  - [x] tiered/profile JSON
  - [x] DUT matrix summary with strict signature gates
- [x] M5: Cross-family table classifying offsets into hardware-constant / family-specific / dim-scaling.
- [x] M5 Exit: 4 recurrent family profiles validated at anchor dims + documented transition function.
- [x] M5.5: Cross-dim deployment audit complete: no family reached target-equivalent replay without EO target bytes (`7056/7952/8976` overflow; `9872` hash drift).

- [x] M6: Controlled instruction probes on fixed dims (single-axis changes only).
  - [x] Quantization-axis probe (rep-range / rep-offset variants).
  - [x] Activation-axis probe (none vs ReLU vs ReLU6).
  - [x] Bias-axis probe (with vs without bias).
  - [x] Input-axis refresh (same-family sweeps for `7056/1840`, `7952/2096`, `9872/2608`).
- [x] M6: Emit per-byte semantic labels with evidence counts.
- [x] M6 Exit: semantic class map for one recurrent family (`8976/2352`, fixed `1792x1792`).

- [x] M7: Quantify derivable vs opaque instruction bytes.
- [x] M7: EO oracle ablation on `7056/8976/9872` (8/16-way; representative 32-way on `7056/9872`) to separate transport-critical vs transport-safe blocks.
- [x] M7: Reverse-direction cross-check of candidate neutral EO windows (`f7056`,`f9872`) — most are context-local; only `f9872:2862..3206` survived.
- [x] M7: Decide path:
  - [ ] Parametric instruction generator (not selected yet)
  - [x] Minimal instruction-template path (selected now; EO remains opaque/toxic)
- [x] M7 Exit: residual compiler-dependency report written.

- [ ] M8: Start Conv2D family RE only after M5-M7 stabilize for Dense.

## Execution guardrails
- [x] Always run `--check-profile` before DUT replay.
- [x] Keep no-reboot recovery workflow (`--reset-before-claim`) for wedge recovery.
- [x] Log every checkpoint in `WORKLOG.md` and commit immediately after each milestone step.
