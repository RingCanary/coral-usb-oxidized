# Phase 2 TODO (M5-M8)

- [ ] M5: Generate family transition map from size table and refresh on every new dim compile.
- [ ] M5: Cover recurrent families with >=3 members where missing:
  - [ ] EO/PC `9872/2608`: compile + profile a 3rd dim member.
  - [ ] EO/PC `7056/1840`: compile + profile a 3rd dim member.
- [ ] M5: For each recurrent family, produce:
  - [ ] `safe_core` patchspec
  - [ ] tiered profile JSON (`generic` + optional `overlays`)
  - [ ] DUT matrix summary with strict signature gates
- [ ] M5: Cross-family table classifying offsets into hardware-constant / family-specific / dim-scaling.
- [ ] M5 Exit: 4 validated recurrent family profiles + documented transition function.

- [ ] M6: Controlled instruction probes on fixed dims (single-axis changes only).
  - [ ] Quantization-axis probe (scale/zero-point variants).
  - [ ] Activation-axis probe (none vs ReLU vs ReLU6).
  - [ ] Bias-axis probe (with vs without bias).
  - [ ] Input-axis refresh (existing dim sweep, same-family only).
- [ ] M6: Emit per-byte semantic labels with evidence counts.
- [ ] M6 Exit: semantic class map for at least one recurrent family.

- [ ] M7: Quantify derivable vs opaque instruction bytes.
- [ ] M7: Decide path:
  - [ ] Parametric instruction generator (if derivable coverage is high), or
  - [ ] Minimal instruction-template format (if opaque core remains).
- [ ] M7 Exit: precise residual compiler-dependency report.

- [ ] M8: Start Conv2D family RE only after M5-M7 stabilize for Dense.

## Execution guardrails
- [ ] Always run `--check-profile` before DUT replay.
- [ ] Keep no-reboot recovery workflow (`--reset-before-claim`) for wedge recovery.
- [ ] Log every checkpoint in `WORKLOG.md` and commit immediately after each milestone step.
