# Function Gemma Decode Loop (Coral)

This example runs an autoregressive Function-Gemma decode loop in Rust with:

- Coral-backed linear stages for every decoder layer (`q/k/v/o/gate/up/down`)
- KV-cache + single-token GQA attention on CPU
- RMSNorm + SwiGLU on CPU
- LM head selectable as:
  - `cpu`: tied embedding projection on CPU
  - `coral`: tiled Coral LM-head projection (`hidden -> vocab`)

## Command

```bash
cargo run --example function_gemma_decode_loop -- \
  <model.safetensors> \
  <templates_dir> \
  <prompt_token_ids_csv> \
  --steps 8 \
  --lm-head coral \
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

For `--lm-head coral`, provide a tiled LM-head template (default tile output
size is `2624`, so an `input=640, output=2624` template is expected).

## Notes

- Prompt is token IDs CSV (for example `2,2516,29901`).
- The example prints per-step token predictions and top-k logits.
- `--max-layers` can be used for faster bring-up before full-depth runs.
- On Pi5, use runtime env exports before running:

```bash
eval "$(./tools/bootstrap_arch_stack.sh print-env)"
```

## Pi5 benchmark snapshot

Using the same prompt/config on Pi5, switching LM-head from CPU to Coral-tiled
changes decode latency substantially:

1. `--max-layers 18 --lm-head cpu --steps 1`:
   - `ms_per_token ~= 17226`
2. `--max-layers 18 --lm-head coral --steps 2`:
   - `ms_per_token ~= 978`

Observed improvement:

- ~`17.6x` faster token decode with Coral-tiled LM-head for the same model and
  decode path.

## Practical guidance

1. Use `--lm-head cpu` only for bring-up/debug (simpler, lower setup).
2. Use `--lm-head coral` for real decode throughput.
3. Expect higher one-time setup time with Coral LM-head:
   - per-layer stage preparation
   - LM vocab tile preparation (`640x2624`, `100` tiles for vocab `262146`).
4. For long decode runs, setup amortizes quickly and Coral LM-head gives the
   better steady-state path.
