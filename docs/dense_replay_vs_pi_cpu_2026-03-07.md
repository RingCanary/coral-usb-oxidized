# Dense replay vs Pi CPU on Raspberry Pi 5 (2026-03-07)

## Question
Estimate the effective host↔device service rate of the current pure-rusb Dense replay path, compare it against CPU GEMV/GEMM on the Pi, and add a tiled-row benchmark that approximates larger logical GEMM offload throughput.

## Scope and method

### 1. Device baseline

Baseline replay data comes from:

- `traces/analysis/benchmark-dense-gemm-replay-20260306T141045Z/`
- `docs/dense_gemm_replay_benchmark_2026-03-06.md`

Those runs use:

- Pi 5 + Coral USB
- pure-rusb replay
- bundled square Dense templates
- one-vector input regime
- `10` measured steady-state runs per case

The extracted run executable for these Dense templates is:

- `EXECUTION_ONLY`
- instruction payload: `16384` bytes
- run-phase parameter stream: `0` bytes

The `PARAMETER_CACHING` executable still exists, but it is preloaded before the measured run loop. So the measured run-phase transport footprint is:

- host→device: `16384 + N` bytes per run
- device→host: `N` bytes per run
- total bulk payload per steady-state run: `16384 + 2N`

This report therefore estimates a **steady-state service rate**, not a cold-preload bandwidth.

### 2. Pi CPU baseline

Added:

- `tools/pi_cpu_int8_bench.c`
- `scripts/benchmark_pi_cpu_int8_linear.sh`

Method:

- compile on Pi with `gcc -O3 -mcpu=cortex-a76+dotprod -fopenmp`
- `OMP_NUM_THREADS=4`
- int8 x int8 dot-product kernel with int32 accumulation
- warmup before measurement
- report median and mean latency / GMAC/s

Artifact:

- `traces/analysis/benchmark-pi-cpu-int8-linear-20260306T184750Z/`

### 3. Tiled logical GEMM approximation

Added:

- `scripts/benchmark_dense_tiled_gemm_replay.sh`

Method:

- reuse `templates/dense_2688x2688_quant_edgetpu.tflite`
- replay the same `2688x2688` run executable many times in one process
- group repeated steady-state runs into one logical GEMM:
  - `device_runs_per_logical = ceil(M / 2688) * N`
- interpret logical throughput as:
  - `GMAC/s = (M * K * N) / logical_time`

This approximates a larger logical GEMM implemented as repeated GEMV offloads plus row tiling. It does **not** claim a new compiled multi-column graph on device.

Artifact:

- `traces/analysis/benchmark-dense-tiled-gemm-replay-20260306T184907Z/`

## Results

### A. Steady-state device service-rate estimate

Using the existing single-vector replay benchmark:

| Case | avg run ms | run payload bytes | total MB/s | host→device MB/s | device→host MB/s |
|---|---:|---:|---:|---:|---:|
| `2048x2048` | `0.293` | `20480` | `69.90` | `62.91` | `6.99` |
| `2304x2304` | `0.312` | `20992` | `67.28` | `59.90` | `7.38` |
| `2688x2688` | `0.301` | `21760` | `72.29` | `63.36` | `8.93` |

From the longer tiled replay sweep:

| Case | avg device-run ms | total MB/s |
|---|---:|---:|
| `2688x2688x2688` logical GEMM | `0.3239` | `67.19` |
| `5376x2688x2688` logical GEMM | `0.3248` | `66.99` |
| `8064x2688x2688` logical GEMM | `0.3256` | `66.83` |

So the current pure-rusb replay stack behaves like a roughly **`67–72 MB/s` steady-state host↔device service path** during the measured run phase.

### B. Coral replay vs Pi CPU GEMV

| Case | Coral replay GMAC/s | Pi CPU int8 GEMV GMAC/s | Coral / CPU |
|---|---:|---:|---:|
| `2048x2048` | `14.315` | `4.954` | `2.89x` |
| `2304x2304` | `17.014` | `5.642` | `3.02x` |
| `2688x2688` | `24.004` | `6.118` | `3.92x` |

Pi CPU artifact source:

- `traces/analysis/benchmark-pi-cpu-int8-linear-20260306T184750Z/SUMMARY.txt`

### C. Logical GEMM offload vs Pi CPU GEMM

| Shape | Coral tiled logical GEMM GMAC/s | Pi CPU int8 GEMM GMAC/s | Coral / CPU |
|---|---:|---:|---:|
| `2688x2688x2688` | `22.309` | `13.677` | `1.63x` |
| `5376x2688x2688` | `22.243` | `14.064` | `1.58x` |
| `8064x2688x2688` | `22.192` | `14.031` | `1.58x` |

Logical GEMM replay hashes stayed stable across all grouped runs.

## Interpretation

### 1. The measured replay path is not preload-bandwidth-bound

Because the measured run loop does not resend the multi-megabyte parameter stream, the steady-state run is dominated by:

- one `16 KiB` instruction payload
- one `N`-byte input payload
- one `N`-byte output read

That is why the implied steady-state service rate is only about `67–72 MB/s`, even though the one-time parameter preload for the `2688x2688` template is about `7.23 MiB`.

### 2. Coral wins clearly for the current GEMV regime

On the Pi 5, a 4-thread A76 int8 GEMV kernel reaches about `5.0–6.1 GMAC/s` over the tested square shapes.

The current pure-rusb replay path reaches about `14.3–24.0 GMAC/s`, so in this regime the Coral is about `2.9x–3.9x` faster than CPU.

### 3. Coral still wins for the row-tiled logical GEMM approximation, but by less

The replay-native tiled logical GEMM stays near `22.2 GMAC/s` across `1x/2x/3x` row-tiling depth. That stability suggests the main steady-state cost is per-device-run and scales linearly with the number of vector offloads.

The Pi CPU int8 GEMM kernel lands near `14.0 GMAC/s` on the matching `2688`-wide shapes.

So the current row-tiled offload approximation is still faster than CPU GEMM, but only by about `1.58x–1.63x`, much less dramatic than the single-vector GEMV advantage.

### 4. Long steady-state replay is slightly slower than the short 10-run benchmark

The earlier `2688x2688` short-run replay result was `24.0 GMAC/s`.

The grouped logical GEMM sweep implies about `22.3 GMAC/s`, which corresponds to `~0.324 ms` per device run instead of `0.301 ms`.

That is a real but modest drop, and it is exactly why the tiled benchmark is useful: long-run throughput is the more relevant number for practical offload planning than a short 10-run microbenchmark.

## Caveats

- CPU numbers are from a custom int8 A76-targeted kernel, not from OpenBLAS. That is intentional: the Pi’s available NumPy/BLAS stack on this host resolves to the basic BLAS implementation and is not a fair high-performance baseline for this workload.
- The logical GEMM replay benchmark is an approximation built from repeated GEMV offloads. It does not yet include a richer host scheduler, overlapping transfers, or a genuinely compiled multi-column Dense graph.
- The service-rate estimate is for the measured steady-state run phase only. It should not be misread as full cold-start bandwidth for parameter admission.

## Repro

```bash
./scripts/benchmark_pi_cpu_int8_linear.sh
./scripts/benchmark_dense_tiled_gemm_replay.sh
```
