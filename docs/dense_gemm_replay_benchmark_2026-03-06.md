# Dense GEMM replay benchmark (2026-03-06)

## Goal
Get a first reproducible benchmark of Coral USB as a large-square Dense GEMM / GEMV-style device using the now-validated pure-rusb replay path.

This is intentionally a **runtime replay benchmark**, not a libedgetpu benchmark.
It measures the device under the same control surface used for the reverse-engineering work.

## Helper and artifact
- New helper:
  - `scripts/benchmark_dense_gemm_replay.sh`
- Artifact:
  - `traces/analysis/benchmark-dense-gemm-replay-20260306T141045Z/`

Replay path:
- `examples/rusb_serialized_exec_replay/main.rs`
  - now prints per-run timing:
    - `Run timing: run_ms=...`
    - `Run timing summary: ...`

DUT host:
- `rpc@10.76.127.205`

All runs used:
- `--bootstrap-known-good-order`
- `--reset-before-claim`
- `--post-reset-sleep-ms 1200`
- bundled compiled templates from `templates/`
- 10 measured runs per case in one replay process

## Cases
- `2048 x 2048`
- `2304 x 2304`
- `2688 x 2688`

Input/output bytes were set to the Dense dimension (`N`).
This is the current large-square single-vector replay regime.

## Results

| Case | avg_ms | min_ms | p50_ms | p95_ms | max_ms | GMAC/s | hash stable |
|---|---:|---:|---:|---:|---:|---:|---|
| `2048x2048` | `0.293` | `0.289` | `0.290` | `0.310` | `0.310` | `14.315` | yes |
| `2304x2304` | `0.312` | `0.308` | `0.3095` | `0.334` | `0.334` | `17.014` | yes |
| `2688x2688` | `0.301` | `0.294` | `0.2995` | `0.321` | `0.321` | `24.004` | yes |

Hashes were stable across all 10 measured runs in every case.

Representative hashes:
- `2048x2048`: `0x3ce2a859ce7ed025`
- `2304x2304`: `0x84600c709be258c4`
- `2688x2688`: `0xfbbb82020e13d160`

## Interpretation
Within this replay regime, the Coral behaves like a very fast, stable large-square Dense engine.

Two practical observations stand out:
1. Latency stays close to ~`0.3 ms` across these three large-square templates.
2. Effective throughput rises with dimension in this set, reaching ~`24 GMAC/s` for `2688x2688`.

This is consistent with fixed replay/control overhead being amortized better at larger square sizes.

## Important caveat
This is **not** yet a universal “Coral GEMM benchmark”.
It is specifically:
- pure-rusb replay
- bundled square Dense templates
- one-vector input regime
- current runtime setup / replay order
- current Pi + Coral host path

So these numbers should be read as:
> benchmark evidence for the current controlled replay stack,
not a universal chip maximum.

## Why this benchmark matters for the RE effort
The device is no longer just a black-box inference target.
Under the current replay stack we can now treat it as a reproducible Dense linear engine and talk about:
- stable per-run latency,
- effective throughput by template size,
- replay overhead amortization,
- future comparisons against tiled-row and multi-stage pipelines.

## Best next benchmark follow-ups
1. add a tiled-row replay benchmark so throughput can be compared against larger logical GEMMs,
2. compare square vs rectangular dense families under the same replay timing surface,
3. separate cold-start / preload cost from steady-state run cost more explicitly,
4. compare pure-rusb replay timings against the legacy delegate path only if the environments are truly comparable.
