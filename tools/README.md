# Tools

This directory is split between the active native RE path and compatibility/archive surfaces.

## Active Phase 4 path

These are still on the live path for pure-rusb replay, native RE, or bounded
Phase 4 materialization work.

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

## Transitional compiler-assisted tooling

These are still referenced by active research scripts, but they are not the
target end-state for Phase 4 completion/control.

- `generate_dense_quant_tflite.py`
- `dense_template_pipeline.sh`
- `extract_edgetpu_package.py`
- `parse_edgetpu_executable.py`
- `instruction_word_field_analysis.py`
- `generate_conv2d_quant_tflite.py`
- `generate_dense_conv_quant_tflite.py`
- `tensorizer_patch_edgetpu.py`
- `edgetpu_delegate_smoke.sh`
- `edgetpu_delegate_smoke.c`

## Compatibility/archive shims

These top-level paths remain only as compatibility forwards into `tools/archive/`.
They should not be treated as the active repo surface.

The following top-level tool paths are compatibility shims forwarding to `tools/archive/`:

- `bootstrap_arch_stack.sh`
- `bootstrap_edgetpu_compiler.sh`
- `build_instruction_patch_spec.py`
- `clip_hf_reference.py`
- `conv_layout_probe.py`
- `conv_template_pipeline.sh`
- `csr_perturbation_matrix.sh`
- `dense_layout_probe.py`
- `dense_quant_value_probe.py`
- `dense_template_matrix_patch.py`
- `emit_word_field_patch_spec.py`
- `exec_chunk_diff.py`
- `instruction_dim_field_analysis.py`
- `multiop_template_pipeline.sh`
- `patch_tflite_dense_weight_pattern.py`
- `re_capture_decode_lm_compare.sh`
- `strace_usb_scaling.py`
- `synthesize_instruction_patch_spec.py`
- `usbmon_param_handshake_probe.py`
- `usbmon_register_map.py`
- `word_field_holdout_validate.py`

## Prune safety check

Run this before deleting or moving tools:

```bash
./tools/check_prune_safety.sh
```

The check scans `src/`, `scripts/`, `tools/*.sh`, and `tools/archive/*.sh` for
`tools/...` paths and fails if any referenced path is missing.
