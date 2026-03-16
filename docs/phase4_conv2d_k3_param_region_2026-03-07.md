# Phase 4 Conv2D `k=3` parameter-region anatomy (2026-03-07)

## Goal
Close `P4-M1` from the Phase 4 completion/control plan by explaining exactly why same-product spatial moves change the parameter stream for single-op Conv2D `k=3`, `stride=1`, `padding=same`, `bias=off`.

The key question was:

> does the `k=3` parameter delta reach stored weights, or is it confined to a small metadata/prefix region that a native materializer can plausibly synthesize?

## Helper and artifact
- Helper:
  - `src/bin/conv_k3_param_anatomy.rs`
  - `scripts/phase4_conv2d_k3_param_region_probe.sh`
- Input artifact:
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/`
- Derived outputs:
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/PARAM_ANATOMY_SUMMARY.txt`
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/p32/PARAM_ANATOMY.txt`
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/p64/PARAM_ANATOMY.txt`
  - `traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z/p128/PARAM_ANATOMY.txt`

## Method
For each previously validated same-product pair (`p32`, `p64`, `p128`):

1. load anchor/target metadata and anchor/target parameter streams,
2. infer the stream layout as repeated output blocks of at most `64` output channels,
3. split each block into:
   - `f32 effective_scale[out]`
   - `u32 stored_zero_point[out]`
   - quantized weight bytes for that output block,
4. count byte deltas by subregion.

The blockwise layout assumption is not arbitrary; it is the already-validated Conv2D packing rule from Phase 3, extended from `1x1` to the observed `k=3` family sizes.

## Result
Across all three tested same-product `k=3` pairs:

- `kernel_sha256_equal = true`
- `input_scale_equal = true`
- `output_scale_equal = false`
- all parameter-stream byte deltas are confined to the blockwise `effective_scale` prefix bytes
- stored zero-point bytes are unchanged
- weight bytes are unchanged

Per pair:

- `p32`
  - `param_len = 9472`
  - `expected_weight_bytes = 9216`
  - `diff_byte_count = 96`
  - single output block: `scale 0..127`, `zp 128..255`, `weight 256..9471`
  - `diff_scale = 96`, `diff_zp = 0`, `diff_weight = 0`
- `p64`
  - `param_len = 37376`
  - `expected_weight_bytes = 36864`
  - `diff_byte_count = 192`
  - single output block: `scale 0..255`, `zp 256..511`, `weight 512..37375`
  - `diff_scale = 192`, `diff_zp = 0`, `diff_weight = 0`
- `p128`
  - `param_len = 148480`
  - `expected_weight_bytes = 147456`
  - `diff_byte_count = 383`
  - two output blocks:
    - block 0: `scale 0..255`, `zp 256..511`, `weight 512..74239`, `diff_scale = 192`
    - block 1: `scale 74240..74495`, `zp 74496..74751`, `weight 74752..148479`, `diff_scale = 191`

The anatomy tool conclusion is the same in all three cases:

> Parameter delta is confined to blockwise effective-scale prefix bytes; stored zero-point bytes and weight region are unchanged.

## Interpretation
This resolves the Phase 4 `k=3` parameter mystery structurally.

The same-product spatial move does **not** require new weights, new weight layout, or new stored zero-point bytes. The changed parameter state is a compact prefix-only target state, consistent with recomputed per-output effective scales for the target model quantization.

What is proven:

- the `k=3` parameter delta is structurally simple
- it lives in a native-synthesis-friendly prefix region
- the Phase 4 param blocker is much narrower than “rebuild the whole parameter stream”

What is still inference rather than direct proof:

- the exact arithmetic law for the changed `effective_scale` bytes
- whether that law is exactly the Phase 3 `1x1` rule reused under `k=3`, or a nearby variant

The evidence strongly suggests the next step should be native scale-prefix materialization, not more broad binary diffing.

## Phase 4 consequence
`P4-M1` is complete for the bounded family:

- single-op Conv2D
- `kernel_size=3`
- `stride=1`
- `padding=same`
- `bias=off`
- same-product spatial moves

The next critical-path milestone is now `P4-M2`: emit the target `k=3` parameter stream natively by preserving the byte-identical weight region and synthesizing only the target blockwise scale-prefix bytes.
