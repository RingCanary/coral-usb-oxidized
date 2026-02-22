# Function Gemma Embedding + LM-Head Sanity

This example validates the non-TPU path first:

- token id -> embedding row lookup (`model.embed_tokens.weight`)
- tied LM head logits (`embedding @ embed_tokens^T`)
- top-k token ids from logits

This is a checkpoint-ingestion sanity pass before stacking full decoder blocks.

## Command

```bash
cargo run --example function_gemma_lm_head_sanity -- \
  <model.safetensors> \
  <token_id> \
  [topk]
```

Defaults:

- `topk=10`

## Notes

- Works with `F32`, `F16`, or `BF16` embedding tensors.
- For Function-Gemma derivatives, embeddings are typically BF16.
- This example is CPU-only and does not require TPU templates.
