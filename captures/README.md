# Frontier 00 Captures

This tree is the repo-local normalization surface for Frontier 00.

Batch 1 scope:

- import historical Conv Phase 4-7 artifacts from
  `../arsenal/coral-rusb-replay-src/traces/analysis`
- import historical Dense artifacts from
  `../arsenal/coral-usb-oxidized-lab/traces`
- keep large binaries in place and reference them by absolute path plus sha256
- copy only lightweight text, JSON, markdown, and patchspec evidence into
  `captures/<capture_id>/evidence/`

Primary files:

- `inventory_phase4_7_dense.csv`: pre-normalization inventory
- `index.tsv`: normalized capture index
- `contract_ledger.tsv`: frontier experiment ledger
- `frontier00_claim_map.tsv`: keep/discard map for the current Frontier 00 stub
- `schema/`: column definitions and metadata schema
- `importers/import_phase4_7_and_dense.py`: inventory and sample import driver
- `importers/validate_frontier00_contract.py`: frozen Frontier 00 contract validator
- `run_inventory_phase4_7_dense.sh`: inventory wrapper
- `run_import_sample_batch.sh`: sample import wrapper

Batch 1 is historical-only. Missing or stale references must be recorded
explicitly; they are never auto-healed.

Validator invariants:

- the normalized corpus remains a 6-row `index.tsv` to `contract_ledger.tsv` join
- `contract_ledger.tsv firmware_hash` must match `index.tsv firmware_sha256`
- `transport_admission` to `admission_ok` is frozen to `pass -> true`,
  `unknown -> unknown`, and `command_exit_1 -> command_exit_1`
- kept verified-success claims require an imported same-shape structural
  counterexample in the current corpus
- omitted phase7 tail siblings must stay explicitly excluded in the claim map
  until they are imported with the same rule satisfied
