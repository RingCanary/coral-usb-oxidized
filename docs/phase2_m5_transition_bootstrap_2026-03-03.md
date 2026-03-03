# Phase 2 Bootstrap (M5): Family Transition Map (2026-03-03)

## Objective
Start Phase 2 by operationalizing the family transition-function artifact from the dense instruction size table.

## Added tooling
- Rust binary: `src/bin/family_transition_map.rs`
- Runner script: `scripts/m5_family_transition_map.sh`

Inputs:
- `docs/artifacts/instruction-dim-field-20260301/dense_instruction_size_table.tsv`

Outputs:
- `transition_map.json` (machine-readable family map)
- `transition_map.md` (human summary)
- `SUMMARY.txt`

## Run
- `traces/analysis/m5-family-transition-map-20260303T182605Z/`

Key results (`SUMMARY.txt`):
- `family_count=10` paired EO/PC families observed in sampled dims.
- `recurrent_family_count=4` (families with >=2 sampled dims):
  - `eo7056_pc1840` dims `[640,1280]`
  - `eo7952_pc2096` dims `[768,1536,2304]`
  - `eo8976_pc2352` dims `[896,1792,2688]`
  - `eo9872_pc2608` dims `[1024,2048]`

## Relevance to M5
This establishes a reproducible baseline transition map and explicitly highlights the 4 recurrent Dense families targeted for full Phase-2 coverage/profiling.
