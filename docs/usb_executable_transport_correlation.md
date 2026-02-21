# EdgeTPU Executable vs USB Transport Correlation

Date: 2026-02-21

This note correlates extracted `DWN1` serialized executable blobs with observed
usbmon bulk signature markers.

## Inputs

- Model: `models/mobilenet_v1_1.0_224_quant_edgetpu.tflite`
- Extracted package metadata: `/tmp/edgetpu_extract/metadata.json`
- Executables:
  - `/tmp/edgetpu_extract/package_000/serialized_executable_000.bin`
  - `/tmp/edgetpu_extract/package_000/serialized_executable_001.bin`
  - `/tmp/edgetpu_extract/package_000/serialized_multi_executable.bin`
- USB signatures:
  - `traces/re-matrix-20260221T092342Z/U4_BULK_SIG.txt`
  - `traces/re-matrix-20260221T092342Z/BASE_BULK_SIG.txt`

## Size context

From metadata:

- `serialized_executable_000.bin`: `233472`
- `serialized_executable_001.bin`: `4476928`
- `serialized_multi_executable.bin`: `4722688`

From usbmon phase report (`U4`):

- `pre_first_bo_b_bulk_out_bytes`: `4697104`

## Marker hits in extracted executables

Searched for loop/burst marker words observed in usbmon payload signatures.

### Markers tied to `Bo 225824`

usbmon loop marker:

- `20720300 00000000` (8-byte command)
- `800f0080 dc000000` (start of `Bo 225824` payload signature)

Found in `serialized_executable_000.bin`:

- offset `0x00001dd8`: `20720300`
- offset `0x00001ddc`: `800f0080 dc000000`

This pairing appears exactly once in `serialized_executable_000.bin`.

### Markers tied to `Bo 7248`

usbmon preload marker:

- `501c0000 00000000` (8-byte command)
- `800f000c 07000000` (start of `Bo 7248` payload signature)

Found in `serialized_executable_001.bin` near the tail:

- offset `0x004433a8`: `501c0000`
- offset `0x004433ac`: `800f000c 07000000`

This pairing appears exactly once in `serialized_executable_001.bin`.

### Marker distribution summary

- `serialized_executable_000.bin`:
  - contains `20720300` and `800f0080dc000000`
  - does not contain `501c0000` or `800f000c07000000`
- `serialized_executable_001.bin`:
  - contains `501c0000` and `800f000c07000000`
  - does not contain `20720300` or `800f0080dc000000`

## Interpretation (current confidence: medium-high)

1. The usbmon loop/preload headers are not random runtime-only values; they are
   embedded in serialized executable payloads.
2. `serialized_executable_000.bin` likely carries descriptors/data tied to the
   recurring `Bo 225824` per-invoke stream.
3. `serialized_executable_001.bin` likely carries descriptors/data tied to the
   preload `Bo 7248` sequence and other setup payloads.
4. The stable 8-byte command word `004c0200 01000000` from loop traffic was not
   matched as a unique contiguous 8-byte pattern in these three blobs (the
   leading dword appears in multiple metadata-like contexts with different
   trailing dwords).

## Cross-model variant probe (`mobilenet_v2_1.0_224_inat_bird_quant_edgetpu`)

A second EdgeTPU model was extracted (`/tmp/edgetpu_extract_bird`) and scanned
with the same marker set.

Observed:

1. The exact model-A marker pairs above are absent.
2. New structured marker pairs appear:
   - `f0270000` + `800f00f4 09000000` in `serialized_executable_000.bin`
     (around `0x000017a0`)
   - `50270000` + `800f00cc 09000000` in `serialized_executable_001.bin`
     (around `0x003c58a8`)

Interpretation:

- Header words embedded in serialized executables are model-specific, consistent
  with the observed model-dependent USB syscall scaling.
- Packet-level confirmation for these new markers still needs usbmon capture on
  the second model.

## Reproduction commands

```bash
# Marker count scan
for f in \
  /tmp/edgetpu_extract/package_000/serialized_executable_000.bin \
  /tmp/edgetpu_extract/package_000/serialized_executable_001.bin \
  /tmp/edgetpu_extract/package_000/serialized_multi_executable.bin; do
  echo "$f"
  LC_ALL=C grep -aobP '\x20\x72\x03\x00' "$f" | wc -l
  LC_ALL=C grep -aobP '\x50\x1c\x00\x00' "$f" | wc -l
  LC_ALL=C grep -aobP '\x80\x0f\x00\x80\xdc\x00\x00\x00' "$f" | wc -l
  LC_ALL=C grep -aobP '\x80\x0f\x00\x0c\x07\x00\x00\x00' "$f" | wc -l
done

# Offset context dump
xxd -g 4 -s 7600 -l 96 /tmp/edgetpu_extract/package_000/serialized_executable_000.bin
xxd -g 4 -s 4469640 -l 120 /tmp/edgetpu_extract/package_000/serialized_executable_001.bin
```
