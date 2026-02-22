# Conv2D Layout Probe (1x1, 64x64 channels)

Date: 2026-02-22

Goal: recover how compiled Conv2D weights map into the EdgeTPU parameter payload,
analogous to Dense layout recovery.

## Run

```bash
./tools/conv_layout_probe.py \
  --height 32 --width 32 \
  --in-channels 64 --out-channels 64 \
  --kernel-size 1 --rep-samples 32
```

Artifacts:

- `traces/conv-layout-probe-20260222T071933Z/layout_probe.json`
- `traces/conv-layout-probe-20260222T071933Z/layout_probe.txt`

## Key observations

1. Selected executable parameter region size is `4608` bytes.
2. For 1x1 Conv2D with `in=64`, `out=64`, candidate offsets from single-hot
   probes fit:

```text
offset = 512 + ((ic // 4) * 256) + (oc * 4) + (ic % 4)
```

Where:

- `ic`: input channel index
- `oc`: output channel index

Example matches:

- `(ic=0, oc=1) -> 516`
- `(ic=1, oc=0) -> 513`
- `(ic=31, oc=31) -> 2431`
- `(ic=63, oc=63) -> 4607`

Interpretation:

- There is a `512`-byte prefix before the channel-mixing weight payload.
- The `64x64` channel kernel uses the same 4-lane inner ordering pattern already
  recovered in Dense templates.

## Practical implication

For `1x1` Conv channel-mixer templates, parameter patching can reuse the Dense
inner-lane mapping logic after accounting for the fixed prefix offset.
