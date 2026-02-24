# Function Gemma Decode Loop (Coral)

This example runs an autoregressive Function-Gemma decode loop in Rust with:

- Coral-backed linear stages for every decoder layer (`q/k/v/o/gate/up/down`)
- KV-cache + single-token GQA attention on CPU
- RMSNorm + SwiGLU on CPU
- LM head selectable as:
  - `cpu`: tied embedding projection on CPU
  - `coral-preload` (`coral` alias): preload all LM tiles once
  - `coral-lazy`: on-demand LM tile prep with LRU cache

## Command

```bash
cargo run --example function_gemma_decode_loop -- \
  <model.safetensors> \
  <templates_dir> \
  <prompt_token_ids_csv> \
  --steps 8 \
  --rounds 2 \
  --weight-quant per-channel \
  --lm-head coral-preload \
  --lm-template <dense_640x2624_quant_edgetpu.tflite>
```

## Required template files

`<templates_dir>` should contain the stage templates used by Function-Gemma 270M
shape classes:

- `dense_640x1024_quant_edgetpu.tflite` (`q`)
- `dense_640x256_quant_edgetpu.tflite` (`k` / `v`)
- `dense_1024x640_quant_edgetpu.tflite` (`o`)
- `dense_640x2048_quant_edgetpu.tflite` (`gate` / `up`)
- `dense_2048x640_quant_edgetpu.tflite` (`down`)

For `--lm-head coral-preload` / `--lm-head coral-lazy`, provide a tiled
LM-head template (default tile output size is `2624`, so an
`input=640, output=2624` template is expected).

## Notes

- Prompt is token IDs CSV (for example `2,2516,29901`).
- The example prints per-step token predictions and top-k logits.
- `--max-layers` can be used for faster bring-up before full-depth runs.
- `--prefill-logits` is disabled by default to avoid wasting LM-head work on
  prompt tokens whose logits are discarded.
- `--rounds N` reuses one prepared process (delegate/stages/lm cache) across
  repeated decode rounds to amortize setup.
- `--weight-quant per-channel` is supported for stage and LM-tile quantization
  and is the recommended mode for better decode quality.
- `--lm-head coral-lazy` uses an LRU tile cache (`--lm-cache-capacity`) to
  reduce setup/memory spikes compared to eager full-vocab preload.
- `--lm-head coral-lazy` with `cache_capacity < tile_count` performs exact
  full-vocab top-k with repeated tile evictions, which can be significantly
  slower than `coral-preload`.
- `--lm-shortlist-tiles N` enables approximate decode for `coral-lazy` by
  evaluating only `N` LM tiles per step (`0` keeps exact full-vocab behavior).
- shortlist mode is intended for low-cache Pi runs where exact lazy mode
  thrashes; it trades output quality for speed.
- On Pi5, use runtime env exports before running:

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
```

## Pi5 benchmark snapshot

Pi5 runtime matrix (`prompt=2,2516,29901`, Function-Gemma 270M, default tile
size `2624`):

1. `--max-layers 1 --lm-head cpu --steps 1`
   - `setup_ms ~= 5953`
   - `decode_ms_per_token ~= 16670`
2. `--max-layers 1 --lm-head coral-preload --steps 1`
   - `setup_ms ~= 56737`
   - `decode_ms_per_token ~= 642`
3. `--max-layers 1 --lm-head coral-lazy --lm-cache-capacity 32 --steps 1`
   - `setup_ms ~= 4571`
   - `decode_ms_per_token ~= 51883`
   - cache stats showed strong churn (`misses >> hits`, high evictions)
4. `--max-layers 1 --lm-head coral-lazy --lm-cache-capacity 32 --lm-shortlist-tiles 16 --steps 1`
   - `setup_ms ~= 4399`
   - `decode_ms_per_token ~= 7973`
   - cache stats: `misses=16`, `evictions=0`, `avg_eval_tiles=16`

Observations:

- `coral-preload` gives the best steady-state decode throughput.
- `coral-lazy` can reduce setup cost but is only practical for exact full-vocab
  decoding when cache capacity is close to tile count.
- `coral-lazy` + shortlist (`--lm-shortlist-tiles 16`) reduced this probe from
  `~48.0 s/token` to `~8.0 s/token` (~`6x` faster) versus exact lazy mode.
- shortlist is approximate: next-token predictions can differ from exact
  full-vocab top-k.
- `--prefill-logits` is expensive and should stay disabled unless explicitly
  needed:
  - prefill off: `~47.6 ms`
  - prefill on: `~103810 ms`

## Practical guidance

1. Use `--lm-head cpu` only for bring-up/debug.
2. Use `--lm-head coral-preload` for exact full-vocab decode throughput.
3. Use `--lm-head coral-lazy` only when constrained by setup memory/time and
   with sufficient cache capacity.
4. For low cache capacities, add `--lm-shortlist-tiles` to bound per-step tile
   work and avoid full-vocab lazy thrash.
5. Expect higher one-time setup time with Coral LM-head:
   - per-layer stage preparation
   - LM vocab tile preparation (`640x2624`, `100` tiles for vocab `262146`).
6. For long decode runs, setup amortizes quickly and Coral LM-head gives the
  better steady-state path.
