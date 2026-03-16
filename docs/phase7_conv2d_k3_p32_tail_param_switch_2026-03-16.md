## Phase 7 `p32` `H=12` Tail Param Switch

Date: 2026-03-16

## Scope

This note records the local diagnosis and native materialization fix for the
excluded `p32` Phase 7 tail widths:

- `12x176`
- `12x184`
- `12x192`

The active published Phase 7 family boundary remains `12x64..12x168` until a
clean DUT rerun is completed for the widened tail through the current remote
access path.

## Diagnostic Result

Probe helper:

- `scripts/phase7_conv2d_k3_p32_tail_param_probe.sh`

Probe artifact:

- `traces/analysis/phase7-conv2d-k3-p32-tail-param-probe-20260316T121528Z/`

For all three excluded widths, the probe shows:

- EO patchspec reconstruction is exact: `eo_mismatch_count=0`
- native parameter materialization mismatches the compiler stream at
  `8931` bytes
- the mismatch is explained by a pure weight-group permutation

Observed tail permutation law:

- prior/native order:
  `group_index = ic_group * 9 + kernel_pos`
- compiler tail order:
  `group_index = kernel_pos * 8 + ic_group`

The probe also reports `mapping_matches_expected=True` for
`12x176/184/192`, so the tail behavior is not random drift; it matches a
deterministic loop-order switch.

## Native Fix

`src/bin/conv_k_param_materialize.rs` now parses `height` and `width` from the
existing metadata JSON and applies an alternate weight-group order only for:

- `kernel_size=3`
- `in_channels=32`
- `out_channels=32`
- `height=12`
- `width >= 176`

The fix is intentionally narrow:

- effective-scale prefix unchanged
- stored-zero-point prefix unchanged
- only the weight-group ordering changes

## Local Validation

Local checks passed:

- `cargo check --bin conv_k_param_materialize`
- `bash -n scripts/phase7_conv2d_k3_p32_tail_param_probe.sh`

Byte-equal native-vs-compiler param verification now passes for:

- `p32 h12_w128`
- `p32 h12_w160`
- `p32 h12_w176`
- `p32 h12_w184`
- `p32 h12_w192`

## Remote Status

The updated repo tree was synced to the remote DUT source directory through the
jump host:

- jump host: `cc@5.223.60.37:22022`
- DUT: `rpc@192.168.29.216`

The first widened completion rerun through that path failed for an operational
reason, not a model/runtime regression: `scripts/phase4_conv2d_k3_completion_demo.sh`
assumes it is launched from a control machine that can SSH directly into the
DUT, while the current remote access path reaches the DUT only through the jump
host's key material. Because that rerun was not completed cleanly, this note
does not widen the checked-in Phase 7 family yet.
