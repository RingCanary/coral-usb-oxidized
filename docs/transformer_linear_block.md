# Transformer Linear Block Benchmark (2304)

This benchmark isolates Coral integration behavior for a transformer-like block
using six same-dimension GEMMs:

- `q_proj`: `2304 x 2304`
- `k_proj`: `2304 x 2304`
- `v_proj`: `2304 x 2304`
- `o_proj`: `2304 x 2304`
- `mlp_up`: `2304 x 2304` (ratio `1x` milestone)
- `mlp_down`: `2304 x 2304`

Each stage uses a patched `DenseGemmTemplate::from_bundled_2304()` and executes
through `PreparedDenseGemm`.

## Why this shape

- `2304 x 2304` fits below the on-chip parameter caching cliff observed in this
  repo's prior runs.
- It avoids MLP-down partial-sum tiling issues while still testing realistic
  multi-stage model switching and prefill behavior.

## Example run

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
cargo run --example transformer_linear_block -- 8 5 1
```

Arguments:

- `seq_len` (default `8`)
- `runs` (default `5`)
- `warmup` (default `1`)
- `--no-attention` to disable CPU single-head attention and focus on linear path

## Reported metrics

The example prints:

- per-stage setup timing (`prepare_ms`, `first_invoke_ms`)
- per-stage average latency (`q`, `k`, `v`, `o`, `up`, `down`)
- CPU attention time (`attn_cpu`)
- `linear_only_ms` and `total_ms`
- `same_stage6_ms` baseline (`q_proj` executed six times) to estimate model-switch overhead
- derived throughput:
  - `linear_gmac_per_s`
  - `end_to_end_gmac_per_s`

This is intended as the milestone harness before moving to expanded MLP ratios,
multi-head splitting, and larger prefill sequence benchmarking.

## Initial measurements (2026-02-22)

Hardware run snapshots from this repo:

- `seq_len=4`, `runs=1`, attention enabled:
  - `linear_only_ms=10.621`
  - `total_ms=11.370`
  - `linear_gmac_per_s=11.995`
- `seq_len=16`, `runs=3`, `--no-attention`:
  - `linear_only_ms=30.757`
  - `same_stage6_ms=30.936`
  - `linear_gmac_per_s=16.569`
- `seq_len=16`, `runs=3`, attention enabled:
  - `linear_only_ms=33.195`
  - `attn_cpu=10.903`
  - `total_ms=44.099`
  - `end_to_end_gmac_per_s=11.556`
