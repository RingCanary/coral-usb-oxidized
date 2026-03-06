# Phase 3 Conv2D TODO

- [x] C0: Kick off a single-op Conv2D-1x1 path (`stride=1`, `padding=same`, `bias=off`) instead of widening directly to general Conv2D.
- [x] C0: Establish local anchor artifact for `32x32x64 -> 64` and verify executable/package extraction.
- [x] C0: Establish first DUT baseline for the anchor model with strict signature capture.

- [x] C1: Bootstrap a first Conv2D-1x1 family map over a small local scan.
- [x] C1: Verify whether EO/PC sizes are stable across spatial-only changes at fixed channels.
- [x] C1: Verify whether parameter bytes scale independently from EO/PC for initial 1x1 cases.

- [x] C2: Revalidate the old 1x1 Conv2D layout clue with the current toolchain.
- [ ] C2: Expand layout recovery beyond `64x64` channels and test threshold changes (`32/64/128`).
- [ ] C2: Decide whether 1x1 Conv2D parameter packing is “Dense-like + prefix” or needs family-specific rules.

- [ ] C3: Add compilerless 1x1 Conv2D parameter packing in Rust.
- [ ] C3: Prove local byte-equivalence against compiled parameter streams.
- [ ] C3: Prove DUT hash-equivalence with parameter override at anchor dims.

- [ ] C4: Add minimal Conv2D family-profile glue / `--check-profile` support.
- [ ] C4: Validate anchor-dim Conv2D profile replay on DUT.

- [ ] C5: Run a same-family cross-dim oracle matrix for 1x1 Conv2D.
- [ ] C5: Determine whether cross-dim target-equivalent replay again requires EO target bytes.

- [ ] C6: Only if C5 says EO is the blocker, localize Conv2D EO transport-critical windows conservatively.
- [ ] Phase 3 Exit: write a precise Conv2D residual-dependency boundary and keep multi-op / k>1 / depthwise deferred unless justified.

## Guardrails
- [x] Keep no-reboot recovery workflow (`--reset-before-claim`); do not power-cycle the hub.
- [x] Prefer single-op 1x1 Conv2D first; keep multi-op Conv2D->Dense out of the critical path.
- [x] Log checkpoints in `WORKLOG.md` and keep claims conservative.
