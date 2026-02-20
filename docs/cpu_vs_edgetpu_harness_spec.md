# CPU vs EdgeTPU Matrix Benchmark Harness Spec

## Status
- Draft v0.2
- Target repo: `coral-usb-oxidized`
- Test platform: Raspberry Pi 5 with Coral USB Accelerator

## 1. Purpose
Define an implementation-ready benchmark harness for fair CPU vs EdgeTPU comparisons on matrix-like ML inference workloads.

This spec is for capability benchmarking only. It is explicitly separate from microgpt runtime integration.

## 2. Scope
- Build a harness that executes comparable CPU and EdgeTPU inference scenarios.
- Focus on static-shape TFLite models with INT8 as the primary comparison path.
- Capture stable metrics (`invoke` and `end_to_end`) with repeatable conditions.
- Emit machine-readable artifacts and a final summary table for reports.

## 3. Baseline Constraints From Current Repo
- Existing benchmark entry point: `examples/inference_benchmark.rs`.
- Existing device/delegate APIs: `src/lib.rs`.
- Existing docs and known caveats: `README.md`.
- Current example path is delegate-first and does not provide a clean CPU-only baseline mode.
- Some `_edgetpu.tflite` models may fail under runtime/library mismatch; the harness must isolate and report per-scenario failures.

## 4. Feasibility Summary
- High feasibility for CPU INT8 vs EdgeTPU INT8 when compile mapping is strong and runtime is stable.
- Medium feasibility for custom matrix-like workloads due to op support and compiler partitioning limits.
- Low feasibility for broad claims about arbitrary GEMM acceleration.

## 5. Primary Benchmark Questions
1. For valid comparable workloads, what are the p50 and p95 latency differences between CPU and EdgeTPU?
2. How stable are those differences across repeated runs under controlled thermal/governor conditions?
3. What fraction of scenarios fail or become non-comparable due to delegate/runtime/compiler constraints?

## 6. Test Matrix Definition

### 6.1 Matrix Dimensions
- Backend:
1. `cpu_int8`
2. `edgetpu_int8`
3. `cpu_fp32` (reference-only, not part of primary speedup claim)

- Workload family:
1. `sanity_model` (known stable quantized model)
2. `fc_matrix_small`
3. `fc_matrix_medium`
4. `fc_matrix_large`
5. `fc_matrix_batched`

- Shape class:
1. Small: `M=1, K=128, N=128`
2. Medium: `M=1, K=256, N=256`
3. Large: `M=1, K=512, N=512`
4. Batched: `M=4, K=256, N=256`

- Repeat id: `1..R`

### 6.2 Pairing Rule for Fair Comparison
- CPU and EdgeTPU scenarios are comparable only if they share:
1. Same source model id from model manifest
2. Same input/output tensor metadata (dtype, shape, quant params)
3. Same preprocessing/postprocessing policy

- For EdgeTPU-compiled variants, record:
1. Source model hash
2. Compiled model hash
3. Compiler version and compile options
4. Compiler report path and hash

### 6.3 Model Manifest (Required)
Create `bench/models.toml` as source of truth for benchmark workloads.

Each manifest entry must include:
- `workload_id`
- `family`
- `source_model_path`
- `source_model_sha256`
- `compiled_model_path` (if applicable)
- `compiled_model_sha256` (if applicable)
- `input_dtype`
- `input_shape`
- `output_dtype`
- `output_shape`
- `expected_delegate_mode` (`full_delegate|unknown_delegate_coverage|cpu_only`)
- `notes`

### 6.4 Execution Classification
Each scenario must be classified as one of:
- `full_delegate`
- `unknown_delegate_coverage`
- `cpu_only`
- `failed`

Only `full_delegate` vs `cpu_only` pairs are eligible for primary speedup claims.

## 7. Environment and Fairness Controls
- Set CPU governor to `performance` and record pre/post values.
- Record CPU affinity policy used for benchmark process.
- Disable USB autosuspend for benchmark session and record resulting state.
- Record CPU temperature every second during each scenario.
- Enforce a start temperature window before each repeat (or apply cooldown and log it).
- Keep background load low and avoid concurrent high-traffic USB operations.
- Fix thread count for CPU scenarios and include it in outputs.
- Use fixed seed and deterministic input generation for every comparable pair.
- Always report both:
1. `invoke_ms`
2. `end_to_end_ms`

## 8. Experiment Protocol
1. Session setup:
1. Load model manifest.
2. Capture environment manifest.
3. Validate toolchain/runtime availability.

2. Scenario preflight:
1. Verify model files and hashes.
2. Validate backend/model compatibility.
3. For EdgeTPU: verify device availability and delegate creation.

3. Scenario execution:
1. Run in its own subprocess.
2. Warmup for `W` iterations.
3. Execute `N` measured iterations.
4. Record per-iteration `invoke_ms` and `end_to_end_ms`.
5. Capture stderr, exit code, and failure reason if unsuccessful.

4. Matrix orchestration:
1. Randomize scenario order in blocked rounds.
2. Run `R` independent repeats.
3. Continue on scenario failure and mark status.

5. Post-processing:
1. Aggregate repeat-level metrics first.
2. Compute speedups only for valid comparable pairs.
3. Emit final report table and artifact set.

## 9. Statistical Requirements
- Warmup: `W >= 10`.
- Measured iterations: `N >= 100` per repeat.
- Repeats: `R >= 5` per scenario.

Report per scenario:
- `n_total`
- `min_ms`
- `p50_ms`
- `p95_ms`
- `mean_ms`
- `stddev_ms`
- `max_ms`

Confidence intervals:
- Compute 95% CI with bootstrap (`10,000` resamples) on repeat-level aggregates.
- If CI cannot be computed, emit `null` plus an explicit `ci_reason`.

Outlier policy:
- Do not remove outliers silently.
- If filtering is used, report both raw and filtered metrics plus rule used.

## 10. Harness Functional Requirements

### 10.1 Runner vs Matrix Orchestrator
Implement two binaries (or two modes):
- `harness-runner`: executes one scenario.
- `harness-matrix`: reads manifest, plans matrix, randomizes order, launches runner subprocesses, aggregates results.

### 10.2 Runner CLI (Single Scenario)
- `--backend` (`cpu_int8|edgetpu_int8|cpu_fp32`)
- `--model <path>`
- `--workload-id <id>`
- `--runs <N>`
- `--warmup <N>`
- `--threads <N>`
- `--seed <u64>`
- `--output-dir <path>`

### 10.3 Matrix CLI
- `--manifest <path>`
- `--repeats <R>`
- `--random-seed <u64>`
- `--output-dir <path>`
- `--fail-fast` (default false)

### 10.4 Guardrails
- Reject invalid backend/model combinations before running.
- Never silently label CPU fallback as EdgeTPU success.
- Mandatory execution class tagging for every scenario.
- Speedup computation must reject non-comparable pairs.

### 10.5 Output Contract
Under `--output-dir`, write:
- `session_manifest.json`
- `scenario_summary.jsonl`
- `latency_samples.jsonl`
- `final_report.csv`

Also print one deterministic summary line per scenario:
- `RESULT workload=<id> backend=<backend> class=<execution_class> p50_ms=<v> p95_ms=<v> status=<ok|failed>`

## 11. Data Schemas

### 11.1 Session Manifest Fields
- `timestamp_utc`
- `git_sha`
- `platform` (Raspberry Pi 5 hardware descriptor)
- `kernel_version`
- `os_release`
- `libedgetpu_version`
- `tflite_version`
- `compiler_version`
- `cpu_governor`
- `cpu_affinity`
- `usb_topology`
- `usb_autosuspend_state`

### 11.2 Scenario Summary Fields
- `scenario_id`
- `repeat_id`
- `workload_id`
- `backend`
- `execution_class`
- `source_model_sha256`
- `compiled_model_sha256`
- `compiler_report_sha256`
- `threads`
- `seed`
- `warmup`
- `runs`
- `invoke_min_ms`
- `invoke_p50_ms`
- `invoke_p95_ms`
- `invoke_mean_ms`
- `invoke_stddev_ms`
- `invoke_max_ms`
- `e2e_min_ms`
- `e2e_p50_ms`
- `e2e_p95_ms`
- `e2e_mean_ms`
- `status`
- `error_code`
- `error_message`

### 11.3 Latency Sample Fields
- `scenario_id`
- `repeat_id`
- `iteration`
- `invoke_ms`
- `end_to_end_ms`

### 11.4 Final Report Table Columns
- `workload_id`
- `backend`
- `execution_class`
- `invoke_p50_ms`
- `invoke_p95_ms`
- `e2e_p50_ms`
- `success_rate`
- `speedup_vs_cpu_invoke_p50`
- `speedup_vs_cpu_e2e_p50`
- `ci_95_low`
- `ci_95_high`
- `notes`

`success_rate` denominator is total planned scenario executions for that `(workload_id, backend)` pair.

## 12. Acceptance Criteria
1. Correctness:
- Deterministic input generation policy is identical across comparable scenarios.
- For INT8 comparable runs: top-1 index match required; if not classification-based, tensor delta must be within per-workload tolerance defined in manifest.

2. Stability:
- No harness-level crash aborting the full matrix.
- Scenario failures are isolated and fully reported.

3. Reproducibility:
- Repeat-level median drift <= 10% for stable scenarios.
- Required metadata fields are complete.

4. Reporting:
- All contract files are emitted and parseable.
- Final table is generated without manual edits.

## 13. Claim Policy
- Do not claim general GEMM acceleration from this benchmark.
- Do not claim EdgeTPU superiority for scenarios labeled `unknown_delegate_coverage`.
- Do not claim energy efficiency without external power data.

## 14. Risks and Mitigations
- Runtime/library mismatch causing faults.
- Mitigation: subprocess isolation, explicit environment capture, strict failure tagging.

- Thermal drift bias.
- Mitigation: governor control, temperature sampling, cooldown window, randomized order.

- Misleading mixed execution comparisons.
- Mitigation: mandatory execution class and strict pairing gates.

## 15. Rollout Plan
1. Phase A: Harness MVP
- Add `harness-runner` with CPU/EdgeTPU modes and output contract.
- Add `harness-matrix` with repeats and randomized execution.

2. Phase B: Matrix Expansion
- Add full manifest-driven workload matrix including batched cases.
- Add bootstrap CI and speedup pair validation.

3. Phase C: Hardening
- Add baseline snapshot diffing and regression thresholds.
- Add optional CI/manual performance gate script.

## 16. Deliverables Before Build Start
- Approved version of this spec.
- Initial `bench/models.toml` manifest.
- Agreed primary headline metric (`invoke_p50_ms`) and pass/fail thresholds.
