# GEMM Weight-Loading Bridge (f32 -> int8 -> Coral)

This workflow bridges synthetic matrix modes to model-style weight loading:

1. start from `f32` weights and inputs
2. quantize both to symmetric `int8`
3. patch the quantized weight matrix into bundled `2304x2304` template bytes
4. execute on Coral with `PreparedDenseGemm`
5. verify output against a CPU quantized reference matmul

Because bundled templates carry fixed TFLite quantization parameters, the
example reports two calibration modes from CPU int32 accumulators to EdgeTPU
output:

- global affine (`q_out ~= alpha * accum + beta`)
- per-output affine (one `alpha/beta` pair per output channel)

Holdout error is reported on rows not used for calibration.
Use at least `calibration_rows=8` for stable per-output fits.

## Run

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
cargo run --example gemm_weight_load_verify -- 8 3 1 2
```

Arguments:

- `seq_len` (default `8`)
- `runs` (default `3`)
- `warmup` (default `1`)
- `calibration_rows` (default `2`)

Optional:

- `--seed N` deterministic synthetic data seed
- `--input-qmax N` target symmetric quant max for inputs (default `32`)
- `--weight-qmax N` target symmetric quant max for weights (default `16`)
- `--weights-f32-le PATH` raw little-endian `f32` file of length `2304*2304`
- `--inputs-f32-le PATH` raw little-endian `f32` file of length `seq_len*2304`

## Output

The example prints:

- EdgeTPU latency and throughput (`gmac_per_s`)
- CPU reference accumulation time
- calibrated affine map (`alpha`, `beta`)
- holdout/all-point verification metrics:
  - `mae`, `rmse`, `max_abs_delta`
  - mismatch counts at `|delta| > 2` and `|delta| > 4`
  - correlation
