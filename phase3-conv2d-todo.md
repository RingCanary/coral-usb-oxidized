# Phase 3 Conv2D TODO

- [x] C0: Kick off a single-op Conv2D-1x1 path (`stride=1`, `padding=same`, `bias=off`) instead of widening directly to general Conv2D.
- [x] C0: Establish local anchor artifact for `32x32x64 -> 64` and verify executable/package extraction.
- [x] C0: Establish first DUT baseline for the anchor model with strict signature capture.

- [x] C1: Bootstrap a first Conv2D-1x1 family map over a small local scan.
- [x] C1: Verify whether EO/PC sizes are stable across spatial-only changes at fixed channels.
- [x] C1: Verify whether parameter bytes scale independently from EO/PC for initial 1x1 cases.

- [x] C2: Revalidate the old 1x1 Conv2D layout clue with the current toolchain.
- [x] C2: Expand layout recovery beyond `64x64` channels and test threshold changes (`32/64/128`).
- [x] C2: Decide the current tested rule: blockwise output-channel packing + per-block prefix, not a single global prefix.

- [x] C3: Add compilerless 1x1 Conv2D parameter packing in Rust.
- [x] C3: Decode the Conv2D prefix as blockwise `f32 effective_scale[out]` + `u32 stored_zero_point[out]`, where `effective_scale = (input_scale * weight_scale) * f32(1/output_scale)`.
- [x] C3: Prove full local byte-equivalence against compiled parameter streams.
- [x] C3: Prove DUT hash-equivalence with full-stream parameter override at anchor dims.

- [x] C4: Critically decide whether Conv2D family-profile glue is needed for Phase-3 boundary-setting.
- [x] C4: Defer profile glue; not required to state the residual dependency precisely for single-op `1x1` Conv2D.

- [x] C5: Run a same-family cross-dim oracle matrix for 1x1 Conv2D.
- [x] C5: Determine the tested cross-dim blocker: EO target bytes only (`PC=0-rule`, params already equal).

- [x] C6: Re-check whether deeper EO localization is necessary for this phase.
- [x] C6: Defer EO ablation; the Phase-3 residual boundary is already precise enough without it.
- [x] Phase 3 Exit: write a precise Conv2D residual-dependency boundary and keep multi-op / k>1 / depthwise deferred.

## Guardrails
- [x] Keep no-reboot recovery workflow (`--reset-before-claim`); do not power-cycle the hub.
- [x] Prefer single-op 1x1 Conv2D first; keep multi-op Conv2D->Dense out of the critical path.
- [x] Log checkpoints in `WORKLOG.md` and keep claims conservative.
