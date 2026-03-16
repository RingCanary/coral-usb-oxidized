# Frontier 00 Trace Contract Stub

Date: 2026-03-16

## Purpose

Establish a repo-local normalization surface for historical Conv Phase 4-7 and
Dense artifacts without widening any current native-control claim.

## Source roots

- Conv March 16 corpus:
  `../arsenal/coral-rusb-replay-src/traces/analysis`
- Dense historical corpus:
  `../arsenal/coral-usb-oxidized-lab/traces`

## New repo-local surface

- `captures/index.tsv`
- `captures/contract_ledger.tsv`
- `captures/inventory_phase4_7_dense.csv`
- `captures/schema/*`
- `captures/importers/import_phase4_7_and_dense.py`

## Batch 1 policy

- historical-only
- explicit stale-reference handling
- no silent pass/fail inference for completion-demo rows that lack stored root
  summaries
- large binaries remain in place and are referenced by absolute path plus sha256

## First imported rows

- Conv structural row:
  `phase4-conv2d-k3-completion-demo-20260316T102326Z/p32/h8_w128`
- Conv structural historical failure row:
  `phase4-conv2d-k3-completion-demo-20260316T114632Z/p32/h12_w176`
- Conv recovery row:
  `phase7-p32-tail-dut-proof-20260316T145105Z/cases/h12_w176`
- Dense stock source:
  `dense-template-1024x1024-20260222T062017Z`
- Dense baseline/gated usbmon scenarios:
  `replay-csr-snapshot-20260225T141800Z/baseline_snapshot`
  `replay-csr-snapshot-20260225T141800Z/dense_gate_snapshot`

## Batch 1 result

- inventory rows: `399`
- explicit stale-reference rows: `2`
- normalized captures: `6`
- direct verified `hash_eq_target=true` rows kept by rule: `1`

The stored March 16 completion-demo artifacts remain structurally useful but do
not carry colocated root summaries in this workspace copy, so their imported
status remains bounded to structural normalization. Claim-map rows
`f00_conv_phase6_structural` and `f00_conv_phase7_overwide_structural` freeze
that boundary.

Claim-map row `f00_conv_phase7_tail_w176_verified` is the only kept direct DUT
success because it is the only imported exact-match tail proof that also has a
same-shape structural counterexample in the current normalized corpus. The
omitted exact-match siblings `h12_w184` and `h12_w192` are frozen as discard
rows `f00_tail_h12_w184_excluded` and `f00_tail_h12_w192_excluded` until the
same rule can be satisfied machine-readably.

Claim-map rows `f00_dense_stock_structural`,
`f00_dense_csr_baseline_failure`, and `f00_dense_gate_failure` keep the Dense
artifacts bounded to structural or exact-failure evidence only. This stub does
not widen to full control, firmware ownership, general executable generation,
or live reproducibility.
