# Phase 4 Conv2D `k=3` native parameter materialization (2026-03-07)

## Goal
Attempt `P4-M2` immediately after `P4-M1` instead of stopping at explanation.

The question was:

> can the bounded-family `k=3` target parameter stream now be emitted natively in Rust, with no `edgetpu_compiler` in the active materialization loop?

## Helpers and artifact
- Native extractor/materializer:
  - `src/bin/conv_k_param_materialize.rs`
  - `src/bin/conv_k3_param_anatomy.rs`
- Wrapper scripts:
  - `scripts/phase4_conv2d_k3_param_region_probe.sh`
  - `scripts/phase4_conv2d_k3_native_param_materialize_probe.sh`
- Input artifact:
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/`
- Derived outputs:
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/p32/NATIVE_PARAM_MATERIALIZE.txt`
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/p64/NATIVE_PARAM_MATERIALIZE.txt`
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/p128/NATIVE_PARAM_MATERIALIZE.txt`
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/p32/target_param_stream.native.bin`
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/p64/target_param_stream.native.bin`
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/p128/target_param_stream.native.bin`

## Method
For each target model in the existing Phase 4 oracle artifact:

1. read the uncompiled quantized `.tflite`,
2. extract Conv2D quantization metadata and raw weight bytes in Rust,
3. canonicalize stored weights to `[out_channel][kernel_y][kernel_x][in_channel]`,
4. synthesize the parameter stream natively using:
   - blockwise prefix = `f32 effective_scale[out]` then `u32 stored_zero_point[out]`,
   - output blocks of up to `64` channels,
   - recovered `k=3` weight packing law:
     - block-local group order is `(in_channel_group, kernel_y, kernel_x)`,
     - inner order is `(out_channel_local, in_channel_mod_4)`,
5. compare the native output against the compiler-produced `target_param_stream.bin`.

The effective-scale arithmetic used the already-proven Conv rule:

> `effective_scale = (input_scale * weight_scale) * f32(1 / output_scale)`

## Result
All three bounded-family targets now reproduce the compiler-produced parameter stream exactly:

- `p32`
  - `stream_len = 9472`
  - `byte_equal = true`
  - `mismatch_count = 0`
- `p64`
  - `stream_len = 37376`
  - `byte_equal = true`
  - `mismatch_count = 0`
- `p128`
  - `stream_len = 148480`
  - `byte_equal = true`
  - `mismatch_count = 0`

The first attempt failed with near-total weight mismatches while the prefix was already exact. That failure was scientifically useful: it isolated the remaining unknown to the `k=3` weight traversal law, not the quantization arithmetic. The decisive correction was changing the block-local weight group order from naive flat `(kernel_y, kernel_x, in_channel_group)` to:

> `(in_channel_group, kernel_y, kernel_x)`

Once that was fixed, all three target streams matched byte-for-byte.

## What is now proven
For the bounded Phase 4 family:

- single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- same-product spatial moves

the parameter stream is now a native Rust materialization problem that is solved locally.

More precisely:

- native Rust can extract the needed Conv quant metadata from the uncompiled `.tflite`,
- native Rust can recover the exact target blockwise prefix bytes,
- native Rust can recover the exact `k=3` weight packing order,
- native Rust can emit the full target parameter stream byte-for-byte without `edgetpu_compiler`.

## Phase 4 consequence
`P4-M2` is now complete for the bounded family.

The remaining completion blocker is no longer parameters. It is:

> EO target-state localization and native synthesis (`P4-M3`)

That sharply changes the frontier. Phase 4 is no longer “params + EO unsolved”; it is now “params solved, EO unsolved.”
