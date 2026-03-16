# Phase 7 `p32` `H=12` Tail Param Switch

Date: 2026-03-16

This note is retained only as a historical diagnosis record. It is superseded by:

- `docs/phase7_conv2d_k3_h12_corridor_completion_2026-03-16.md`
- `docs/phase7_full_control_boundary_2026-03-16.md`

## Historical Result

The excluded Phase 7 `p32` tail widths `12x176`, `12x184`, and `12x192` were traced to a narrow weight-group ordering switch in native parameter materialization:

- prior/native order: `group_index = ic_group * 9 + kernel_pos`
- compiler tail order: `group_index = kernel_pos * 8 + ic_group`

That recovery was real and became the final fix used to close the published Phase 7 `H=12` corridor through `12x192`.

## Status

Do not treat this note as the current published boundary statement. The current published Phase 7 state is the fully closed `12x64..12x192` corridor validated on-device.
