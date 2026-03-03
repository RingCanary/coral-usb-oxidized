# M3 P4→P2 Sequencing: Coupling Probe First + Unseen Prediction (2026-03-03)

## Objective
Apply the corrected sequencing:
1. test EO/PC coupling first,
2. then attempt unseen-dimension prediction with explicit family-boundary checks.

## 1) Coupling probe (DUT first)
Script:
- `scripts/m3_instruction_coupling_probe.sh`

Run:
- `traces/analysis/specv3-m3-instr-coupling-probe-20260303T162721Z/`

Cases:
- baseline
- pc_safe14
- eo_toxic4
- eo_toxic4_plus_pc_safe14
- eo_nontoxic6
- pc_res39
- pc_res39_plus_eo_nontoxic6

Results (`SUMMARY.txt`):
- `baseline`: PASS hash `0x67709fedfd103a2d`
- `pc_safe14`: PASS hash `0x67709fedfd103a2d`
- `eo_toxic4`: FAIL `UsbError(Timeout)`
- `eo_toxic4_plus_pc_safe14`: FAIL `UsbError(Timeout)`
- `eo_nontoxic6`: PASS hash `0xf790ee9e92c4c4f1`
- `pc_res39`: FAIL `UsbError(Timeout)`
- `pc_res39_plus_eo_nontoxic6`: FAIL `UsbError(Timeout)`

Coupling verdicts:
- `eo_toxic4` vs `eo_toxic4+pc_safe14`: **no coupling evidence**
- `pc_res39` vs `pc_res39+eo_nontoxic6`: **no coupling evidence**

Interpretation:
- In these probes, EO toxic and PC toxic behaviors remain independently toxic under the tested cross-plane context.

## 2) Unseen prediction run
Script:
- `scripts/m3_instruction_predict_unseen_rect1792.sh`

Run:
- `traces/analysis/m3-unseen-predict-rect1792-20260303T163516Z/`

Setup:
- Fixed output dim `1792`
- Training inputs `{896, 1792, 2688}`
- Unseen target input `1344`

Intermediate analysis artifact:
- `traces/analysis/m3-rect1792-wordfield-20260303T163317Z/`
  - `pc_2352_inputvar_out1792_wordfield.json`
  - `eo_8976_inputvar_out1792_wordfield.json`

### PC unseen prediction (same family, payload_len=2352)
From `SUMMARY.txt`:
- endpoint: baseline `231` -> v2 `29`
- best: baseline `33` -> v2 `27`
- strict: baseline `156` -> v2 `105`
- threepoint: baseline `47` -> v2 `27`

Best observed modes:
- `best` and `threepoint` tie at `v2_mismatch=27` bytes.

Mismatch breakdown artifact:
- `traces/analysis/m3-unseen-predict-rect1792-20260303T163516Z/pc_best.mismatch_breakdown.txt`

Breakdown summary:
- total mismatches: `27`
- covered by `discrete_flags`: `26`
- covered by `safe_core`: `0`
- uncovered: `1` (offset `1683`, latent boundary transition)

Byte-residue histogram:
- `{39:14, 18:6, 19:3, 57:2, 2:1, 4:1}`

Notable overlap:
- full overlap with known `res39` cluster (`14/14` bytes)
- additional concentration in the `res18` band (+ neighboring high byte residue `19`)

### EO unseen boundary outcome
EO target executable for `1344x1792` is not in the `8976` family:
- base/training EO chunk size: `8976`
- target EO chunk size: `7904`

Tool reports:
- `base/target chunk size mismatch: base=8976 target=7904`

Interpretation:
- This is a family boundary crossing; direct same-family EO patch prediction is invalid for this unseen target.

## Practical takeaway
- Coupling-first check indicates no immediate EO/PC rescue coupling for the tested toxic subsets.
- Unseen prediction is viable only when executable family shape (payload length) is preserved.
- For target `1344x1792`, PC remained in-family and was predicted to small residual mismatch; EO crossed family boundary and must be treated as a new family (new anchor/synthesis track).
