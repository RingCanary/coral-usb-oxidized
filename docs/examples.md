# Examples And Workflows

Use this page for runnable entry points and deeper workload notes. For the current repo status and bounded-family scope, go back to `docs/index.md` and `docs/active_path.md`.

## Active Native Runtime

- `cargo run --example rusb_control_plane_probe -- --verbose-configs`
- `cargo run --example rusb_serialized_exec_replay -- --help`
- `cargo run --example rusb_param_glitch_fuzz -- --help`
- `cargo run --bin conv_k3_eo_emit -- --family-spec templates/phase7_conv2d_k3_h12_corridor_6512/family.json --channels 64 --target-height 12 --target-width 192 --out-report /tmp/conv_k3_eo_emit_phase7.json`
- `bash scripts/phase4_conv2d_k3_completion_demo.sh --family-spec templates/phase7_conv2d_k3_h12_corridor_6512/family.json --pairs p32,p64,p128`

## Compatibility Examples

- `cargo run --features legacy-runtime --example basic_usage`
- `cargo run --features legacy-runtime --example verify_device`
- `cargo run --features legacy-runtime --example delegate_usage`
- `cargo run --features legacy-runtime --example simple_delegate`
- `cargo run --features legacy-runtime --example tflite_standard_example`
- `cargo run --features legacy-runtime --example cpu_vs_edgetpu_mvp -- --help`

Compatibility context lives in `docs/legacy_compatibility.md`.

## Artifact Inspection

- `python3 tools/extract_edgetpu_package.py extract /tmp/model_edgetpu.tflite --out /tmp/edgetpu_extract`
- `python3 tools/parse_edgetpu_executable.py /tmp/edgetpu_extract/package_000`
- `python3 tools/tensorizer_patch_edgetpu.py inspect /tmp/model_edgetpu.tflite`

## Workload Deep Dives

- `docs/function_gemma_decode_loop.md`
- `docs/function_gemma_layer_tpu_probe.md`
- `docs/function_gemma_lm_head_sanity.md`
- `docs/transformer_linear_block.md`
- `docs/clip_vit_block_tpu_pipeline.md`
- `docs/clip_vit_layer_tpu_probe.md`
- `docs/clip_vit_full_forward.md`
- `docs/gemm_weight_load_verify.md`

## Historical Workflow Notes

Older workflow writeups, layout probes, and compiler-assisted research notes are archived. Start from `docs/archive_index.md`.
