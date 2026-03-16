# Tools

This directory is split between the active native RE path and compatibility/archive surfaces.

## Active Native Path

These are on the live path for pure-rusb replay, native RE, or bounded family materialization work.

Core USB workflow:

- `usbmon_capture.sh`
- `usbmon_bulk_signature.py`
- `usbmon_three_stage_signature.py`
- `usbmon_phase_report.py`
- `usb_syscall_trace.sh`
- `libusb_trace_diff.py`
- `pyusb_parity_harness.py`
- `test_usb_command_dispatch.sh`
- `usb_wedge_diag.sh`

## Transitional Compiler-Assisted Tooling

These are still referenced by active research scripts, but they are not the target end-state for native completion/control.

- `generate_dense_quant_tflite.py`
- `dense_template_pipeline.sh`
- `extract_edgetpu_package.py`
- `parse_edgetpu_executable.py`
- `instruction_word_field_analysis.py`
- `generate_conv2d_quant_tflite.py`
- `generate_dense_conv_quant_tflite.py`
- `tensorizer_patch_edgetpu.py`
- `dump_tflite_conv1x1_weights.py`
- `patch_tflite_conv1x1_weight_pattern.py`
- `edgetpu_delegate_smoke.sh`
- `edgetpu_delegate_smoke.c`

## Archive-Only Tooling

These paths are retained only under `tools/archive/`. They should not be treated as the active repo surface.

- `tools/archive/bootstrap_arch_stack.sh`
- `tools/archive/bootstrap_edgetpu_compiler.sh`
- `tools/archive/clip_hf_reference.py`
- `tools/archive/conv_layout_probe.py`
- `tools/archive/conv_template_pipeline.sh`
- `tools/archive/csr_perturbation_matrix.sh`
- `tools/archive/dense_layout_probe.py`
- `tools/archive/dense_quant_value_probe.py`
- `tools/archive/dense_template_matrix_patch.py`
- `tools/archive/emit_word_field_patch_spec.py`
- `tools/archive/exec_chunk_diff.py`
- `tools/archive/instruction_dim_field_analysis.py`
- `tools/archive/multiop_template_pipeline.sh`
- `tools/archive/patch_tflite_dense_weight_pattern.py`
- `tools/archive/re_capture_decode_lm_compare.sh`
- `tools/archive/strace_usb_scaling.py`
- `tools/archive/synthesize_instruction_patch_spec.py`
- `tools/archive/usbmon_param_handshake_probe.py`
- `tools/archive/usbmon_register_map.py`

## Prune safety check

Run this before deleting or moving tools:

```bash
./tools/check_prune_safety.sh
```

The check scans `README.md`, `docs/`, `examples/`, `src/`, `scripts/`, `tools/*.sh`, and `tools/archive/*.sh` for `tools/...` paths and fails if any referenced path is missing.
