# CLIP ViT SafeTensors Report

This example is the first integration step for real-model ingestion: parse a
CLIP checkpoint in SafeTensors format, validate ViT encoder layer tensor names
and shapes, and run weight quantization preflight for Coral template loading.

## Command

```bash
cargo run --example clip_vit_safetensors_report -- <model.safetensors> [layer_idx] [qmax]
```

Defaults:

- `layer_idx=0`
- `qmax=127`

Example:

```bash
cargo run --example clip_vit_safetensors_report -- models/clip-vit-base-patch32/model.safetensors 0 127
```

## What it validates

For a target layer `N`, the example checks these tensors exist with `f32` dtype
and expected ViT-B/32 dimensions:

- `vision_model.encoder.layers.N.self_attn.q_proj.weight` (`768x768`)
- `vision_model.encoder.layers.N.self_attn.k_proj.weight` (`768x768`)
- `vision_model.encoder.layers.N.self_attn.v_proj.weight` (`768x768`)
- `vision_model.encoder.layers.N.self_attn.out_proj.weight` (`768x768`)
- `vision_model.encoder.layers.N.mlp.fc1.weight` (`3072x768`)
- `vision_model.encoder.layers.N.mlp.fc2.weight` (`768x3072`)

It then quantizes:

- `q_proj` (`768x768`)
- `mlp.fc1` (`768x3072` after transpose to row-major input-by-output layout)
- `mlp.fc2` (`3072x768` after transpose to row-major input-by-output layout)

Output includes quantized byte count and scale metadata, which is the direct
input to `DenseGemmTemplate::set_weights_from_slice`.
