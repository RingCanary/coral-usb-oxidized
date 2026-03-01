# Word-Patch Holdout (Family 8976/2352) — 2026-03-01

## Goal
Run a true endpoint holdout test without target leakage:
- low/high endpoints: `896` and `2688`
- target holdout: `1792`
- family: `EO=8976`, `PC=2352`

## Artifacts
- Analysis JSON:
  - `docs/artifacts/instruction-word-field-20260301-family8976/eo_8976_896_1792_2688_wordfield.json`
  - `docs/artifacts/instruction-word-field-20260301-family8976/pc_2352_896_1792_2688_wordfield.json`
- Emitted holdout specs:
  - `traces/analysis/holdout-family8976-20260301T142200Z/eo_endpoint.patchspec`
  - `traces/analysis/holdout-family8976-20260301T142200Z/pc_endpoint.patchspec`
- Pi5 reboot-first DUT logs:
  - `traces/analysis/holdout-family8976-reboot-matrix-20260301T142227Z/`

## Endpoint Holdout Mismatch
- EO: `mismatch_vs_target=167`
- PC: `mismatch_vs_target=114`

## Reboot-First DUT Results (Pi5 + Coral)
- Baseline (`1792x1792`): **PASS**
  - `Output: bytes=1792 fnv1a64=0x67709fedfd103a2d`
- PC-only patch (`114` bytes): **FAIL at bootstrap class-2 offset 0**
  - `bootstrap param: parameter stream write failed at offset 0 ... Operation timed out`
- EO-only patch (`167` bytes): **RUN phase reached + event observed, then output timeout**
  - `Event: tag=4 ...`
  - `Error: UsbError(Timeout)`
- Both patch (`281` bytes): **same as PC failure**
  - fails at bootstrap class-2 offset 0

## Conclusion
The admission/output split reproduces under true holdout conditions:
- PC errors are admission-critical.
- EO errors are execution/output-critical.

This validates that endpoint interpolation still misses semantically important fields in both planes when predicting an unseen target dimension.
