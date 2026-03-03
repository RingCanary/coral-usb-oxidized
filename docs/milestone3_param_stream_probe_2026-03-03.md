# Milestone 3 Progress: Parameter Stream Differential Probe (2026-03-03)

## Scope
Executed Priority-1 work from Milestone 3 strategy: controlled differential analysis of `PARAMETER_CACHING` parameter streams (`tag 2`) using generated dense models.

## What was added

### 1) Rust analysis binary
- `src/bin/param_stream_diff.rs`

Capabilities:
- loads one or more EdgeTPU-compiled `.tflite` files,
- extracts parameter stream bytes from serialized executables via existing Rust flatbuffer path,
- emits per-model stream metadata (len, executable mapping, FNV hash, instruction chunk lens),
- computes pairwise byte-level diffs:
  - overlap changed/equal counts,
  - prefix/suffix equality lengths,
  - chunk-level equality (4 KiB),
  - histogram distance,
  - sampled byte transitions,
- computes multi-model invariant offset fraction over common prefix.

### 2) Repro script
- `scripts/m3_param_stream_probe.sh`

Workflow:
1. generate quantized dense models (`896`, `1792`) for `init_mode=zero` and `init_mode=ones`,
2. compile each with `edgetpu_compiler`,
3. run `param_stream_diff` across key comparisons,
4. write:
   - `param_stream_diff.report.json`
   - `SUMMARY.txt`

## Probe run artifacts
- `traces/analysis/m3-param-stream-probe-r2-20260303T143830Z/`
  - `d896_zero/*`
  - `d896_ones/*`
  - `d1792_zero/*`
  - `d1792_ones/*`
  - `param_stream_diff.report.json`
  - `SUMMARY.txt`

## Key observations

### A) Same dimension, different content (`zero` vs `ones`)
- `1792`: changed bytes = `3211264 / 3211264` (100%)
- `896`: changed bytes = `802816 / 802816` (100%)
- dominant transition is uniform `0x80 -> 0xff` for every byte.

Interpretation:
- for these controlled extreme-weight models, parameter payload bytes are content-driven and globally rewritten by quantized weight value; no fixed metadata subregion remained unchanged.

### B) Same content class, different dimensions (`896` vs `1792`)
- `d896_ones` is an exact prefix of `d1792_ones` over full `802816` bytes.
- `d896_zero` is an exact prefix of `d1792_zero` over full `802816` bytes.
- extra bytes in 1792 stream: `2408448`.

Interpretation:
- within this family and for uniform content, dimension scaling appears as deterministic stream-length extension with prefix preservation rather than re-scrambling of the existing prefix.

### C) Multi-model invariants (4-way common prefix)
- invariant bytes across `{d896_zero,d896_ones,d1792_zero,d1792_ones}` over common length `802816`: `0`.

Interpretation:
- with extreme opposing payloads (all 0x80 vs all 0xff), common-byte invariants vanish as expected.

## Milestone-3 Priority-1 status
- **Progress:** strong initial evidence gathered.
- **Exit criterion (bijective reorder or not):** **not fully closed yet**.

Reason:
- this run uses degenerate uniform payloads (`zero` / `ones`), so it proves deterministic structure and scaling behavior, but does not yet distinguish all possible layout transforms on non-uniform weights.

## Immediate next probe (recommended)
1. Repeat with sparse/non-uniform payloads at `896` and `1792` (e.g., `single_hot`, checkerboard, low-density random) to test whether 896 remains an exact prefix of 1792 under structured variation.
2. Add permutation-check diagnostics over non-uniform payloads (not only histogram equality).
3. If prefix property holds broadly, prioritize deriving direct index mapping from logical weight index -> stream offset.
