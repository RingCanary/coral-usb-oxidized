# USB Register Map Candidates (EdgeTPU Coral USB)

This note captures current protocol findings from usbmon control-transfer
analysis and cross-reference against public `libedgetpu`/DarwiNN register maps.

## Data sources

Primary captures:

- `traces/re-matrix-20260221T092342Z/U1_delegate_smoke_usbmon/usbmon-bus4-20260221T093358Z.log`
- `traces/re-matrix-20260221T092342Z/U2_simple_delegate_usbmon/usbmon-bus4-20260221T093432Z.log`
- `traces/re-matrix-20260221T092342Z/U3_infer_plain_usbmon/usbmon-bus4-20260221T093512Z.log`
- `traces/re-matrix-20260221T092342Z/U4_infer_edgetpu_usbmon/usbmon-bus4-20260221T093544Z.log`
- `traces/usbmon-20260221T103521Z-bus4/usbmon-bus4-20260221T103521Z.log` (`U5`, bird edgetpu)
- `traces/usbmon-20260221T103552Z-bus4/usbmon-bus4-20260221T103552Z.log` (`U6`, inception edgetpu)
- `traces/usbmon-20260221T103631Z-bus4/usbmon-bus4-20260221T103631Z.log` (`U7`, inception plain)
- baseline: `traces/usbmon-20260221T090004Z-bus4/usbmon-bus4-20260221T090004Z.log`

Generated artifacts:

- `traces/re-matrix-20260221T092342Z/REGISTER_MAP_MATRIX.md`
- `traces/re-matrix-20260221T092342Z/REGISTER_MAP_MATRIX.json`
- `traces/re-matrix-20260221T092342Z/U4_REGISTER_REPORT.txt`
- `traces/re-matrix-20260221T092342Z/U4_BULK_SIG.txt`
- `traces/re-matrix-20260221T092342Z/USBMON_BULK_SIGNATURE_SUMMARY.md`
- `traces/re-matrix-20260221T092342Z/REGISTER_MAP_MATRIX_U1_U7.md`
- `traces/re-matrix-20260221T092342Z/USBMON_PACKET_VALIDATION_20260221T1035Z.md`

## Operation-class inference

From setup fields (`bmRequestType`, `bRequest`, `wValue`, `wIndex`, `wLength`):

- `40 00 xxxx 0004 0008` -> candidate `write64`
- `c0 00 xxxx 0004 0008` -> candidate `read64`
- `40 01 xxxx 0001 0004` -> candidate `write32`
- `c0 01 xxxx 0001 0004` -> candidate `read32`

For vendor register requests, full 32-bit CSR offset is:

- `full_offset = (wIndex << 16) | wValue`

## Cross-run invariants

1. `U1`, `U2`, `U3`, `U4`, and baseline share the same control/register
   address-op counts for setup/teardown.
2. The major behavioral difference appears in bulk transfers:
   - `U1/U2/U3`: no `Bo` inference payload path
   - `U4/baseline`: full bulk inference signature
3. `U4` and baseline are signature-identical for bulk payload headers (after
   normalization), including per-invoke loop command prefixes.
4. `U5`/`U6` add model-specific bulk loop signatures while preserving identical
   control/register address-op counts seen in `U1..U4`.

## Phase behavior (U4/baseline)

Loop window identified by bulk cycle (`Bo150528` first to `Bi1008` last):

- `first_bo_b_ts=2762713111`
- `last_bi_out_ts=2762807813`

Control ops by phase:

- pre-loop:
  - `read32=11`, `read64=3`, `write32=11`, `write64=27`
- loop:
  - no control-register ops detected (bulk-only data path)
- post-loop:
  - `read32=6`, `read64=1`, `write32=11`, `write64=21`

## Candidate register groups

These labels are hypotheses, based on repetition and phase placement.

### Group A: core status/control (mostly 32-bit)

- `a30c`, `a314`, `a318`, `a33c`, `a0d4`, `a0d8`, `a704` (`read32`/`write32`)
- `a500`, `a558`, `a600`, `a658` (`write32`)
- `907c` (`write32`, post-loop only in U4)

Known mappings now confirmed for observed offsets:

- `a30c` + `wIndex=0001` -> `0x1a30c` -> `scu_ctrl_0`

Hypothesis (remaining): lifecycle/status bits and shutdown/ack paths for other
members of this group.

### Group B: fabric/queue/channel config (mostly 64-bit writes)

- `4018` (`write64` and one `read64`)
- `4158`, `4198`, `41d8`, `4218` (`write64`)
- `00c0`, `0110`, `0150`, `0190`, `01d0`, `0210`, `0250`, `0298`, `02e0`, `0328` (`write64`)
- `c058`, `c060`, `c070`, `c080`, `c090`, `c0a0`, `c148`, `c160` (`write64`)

Known mappings now confirmed for observed offsets:

- `4018` + `wIndex=0004` -> `0x44018` -> `scalarCoreRunControl`

Hypothesis (remaining): queue/ring/channel enables and teardown/reset
sequencing for other members of this group.

### Group C: repeated handshake knob

- `8788` (`write64` and `read64`, both pre-loop and post-loop)

Known mapping now confirmed:

- `8788` + `wIndex=0004` -> `0x48788` -> `tileconfig0`

## Bulk-loop signature markers (U4/baseline)

Recurring per-invoke submit signatures:

- `Bo size=8 sig=20720300 00000000` (count 35)
- `Bo size=8 sig=004c0200 01000000` (count 35)
- `Bo size=225824 sig=800f0080 dc000000` (count 35)
- `Bo size=150528 sig=00010203 04050607` (count 35)

Completion signatures:

- `Bi size=1008 sig=00000000 00000000` (count 35, loop)
- `Bi size=16 sig=00000000 00000000` (count 36, status path)

`Bo size=8` signatures decode as USB ML packet headers:

- first 4 bytes (little-endian): payload length
- byte 4: descriptor tag (`0=Instructions`, `1=InputActivations`,
  `2=Parameters`, `3=OutputActivations`, `4..7=Interrupt0..3`)
- bytes 5..7: padding

Example:

- `20720300 00000000` -> `length=225824`, `tag=0 (Instructions)`
- `004c0200 01000000` -> `length=150528`, `tag=1 (InputActivations)`

## Standard USB setup signatures

Observed each run during enumeration/setup:

- `80 06 0300 0000 00ff` (string descriptor)
- `80 06 0100 0000 0008/0012` (device descriptor)
- `80 06 0200 0000 0060` (configuration descriptor)
- `80 06 0f00 0000 0005/0016` (BOS/capability descriptors)
- `00 31 0028 0000 0000`
- `00 09 0001 0000 0000` (set configuration)

## Confidence and gaps

- High confidence:
  - operation-class inference (`read32/write32/read64/write64`)
  - phase split: control-heavy setup/teardown vs bulk-only inference loop
- Medium confidence:
  - grouping addresses by subsystem role
- Low confidence:
  - exact semantic meaning of still-unnamed register addresses beyond the
    confirmed set above

## Reproduction

```bash
python3 tools/usbmon_register_map.py report <usbmon.log> --bus 4 --device 005
python3 tools/usbmon_register_map.py matrix --run U1=<log1> --run U2=<log2> --run U3=<log3> --run U4=<log4> --bus 4 --device 005
python3 tools/usbmon_register_map.py sequence <usbmon.log> --bus 4 --device 005 --phase pre_loop --json > /tmp/vendor-seq-preloop.json
python3 tools/usbmon_register_map.py sequence <usbmon.log> --bus 4 --device 005 --phase all --writes-only --json > /tmp/vendor-writes-all.json
python3 tools/usbmon_register_map.py sequence <usbmon.log> --bus 4 --device 005 --phase all --writes-only --known-only --json > /tmp/vendor-writes-known.json
python3 tools/usbmon_bulk_signature.py <usbmon.log> --bus 4 --device 005 --prefix-words 2
python3 tools/usbmon_three_stage_signature.py <usbmon.log> --bus 4 --device 005
```
