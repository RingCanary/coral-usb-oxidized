# CPU vs EdgeTPU MVP Spec (Hypothesis Test)

## Goal
Quickly test whether EdgeTPU provides a latency win over CPU for two representative workloads.

## Hypothesis
For supported INT8 models, `edgetpu_int8` has lower p50 latency than `cpu_int8`.

## Scope (MVP Only)
- Backends: `cpu_int8`, `edgetpu_int8`
- Workloads:
1. One known stable quantized model (`sanity_model`)
2. One matrix-style model (`matrix_model`)
- Platform: Raspberry Pi 5 + Coral USB Accelerator

## Run Plan
- Warmup: `10`
- Measured iterations: `100`
- Repeats: `3` per `(workload, backend)`
- Input generation: deterministic with fixed seed
- Metric capture:
1. `invoke_ms` per iteration
2. `end_to_end_ms` per iteration

## Execution Rules
- Use identical input/output tensor setup across backends for each workload.
- If EdgeTPU delegate creation fails, mark scenario as `failed` and continue.
- Do not label CPU fallback as EdgeTPU success.
- Run each scenario independently so one failure does not stop all runs.

## Outputs
- Console summary per scenario:
`RESULT workload=<id> backend=<b> repeat=<r> invoke_p50_ms=<v> invoke_p95_ms=<v> e2e_p50_ms=<v> status=<ok|failed>`
- One CSV file with:
`workload,backend,repeat,invoke_p50_ms,invoke_p95_ms,invoke_mean_ms,e2e_p50_ms,e2e_p95_ms,e2e_mean_ms,status,error`

## MVP Success Criteria
- All 12 planned scenarios complete or fail with explicit status
  (`2 workloads x 2 backends x 3 repeats`).
- CSV is complete and parseable.
- At least one valid backend pair exists for each workload.

## Deferred to Full Plan
- CI thresholds, bootstrap CIs, large matrix expansion, and regression gating remain in the full spec.
