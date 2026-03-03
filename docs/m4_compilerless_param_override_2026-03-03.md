# M4 Bring-up: Compilerless Dense Parameter Stream Override Proof (2026-03-03)

## Objective
Demonstrate end-to-end usage of compilerless-generated parameter streams in replay, replacing model-extracted parameter bytes on DUT while preserving execution/output signatures.

## Code added

### 1) Dense parameter packing API (Rust lib)
- `src/param_pack.rs`
- exported from `src/lib.rs`

Public APIs:
- `dense_param_stream_len(rows, cols)`
- `dense_param_stream_offset(rows, cols, row, col)`
- `pack_dense_row_major_u8_to_stream(rows, cols, src)`
- `pack_dense_row_major_i8_to_stream(rows, cols, src)`
- `unpack_dense_stream_to_row_major_u8(rows, cols, stream)`
- `unpack_dense_stream_to_row_major_i8(rows, cols, stream)`

Includes unit tests for known offsets, bijection checks, and roundtrip checks.

### 2) Packing CLI
- `src/bin/dense_param_pack.rs`

Used here with deterministic pattern source:
- `--pattern-index-mod --modulus 251 --signed-reinterpret`

### 3) Replay override path
- `examples/rusb_serialized_exec_replay.rs`

New flag:
- `--param-stream-override-file PATH`

Behavior:
- replaces extracted non-empty parameter stream bytes with file payload,
- enforces exact length match per executable,
- logs file/before/after FNV hashes.

## Local equivalence artifacts
Run root:
- `traces/analysis/m4-compilerless-param-override-20260303T153955Z/`

Generated streams:
- `stream_mod251_r896_c896.bin`
- `stream_mod251_r1792_c1792.bin`
- `stream_mod251_r1792_c896.bin`

Byte-equivalence report:
- `stream_vs_compiled_compare.json`

All three cases are exact byte matches vs compiler-produced parameter streams:
- `896x896`: equal=true
- `1792x1792`: equal=true
- `rect (stored 1792x896)`: equal=true

## DUT matrix (Pi5 + Coral)
Run root:
- `traces/analysis/specv3-m4-compilerless-override-matrix-20260303T154219Z/`

Cases and output hashes:

1. `base_896` -> `0x8d7854bd1eb9c1e2`
2. `override_896` -> `0x8d7854bd1eb9c1e2`

3. `base_1792` -> `0x394aa8758535e7e9`
4. `override_1792` -> `0x394aa8758535e7e9`

5. `base_rect` -> `0xe0f607a60893b844`
6. `override_rect` -> `0xe0f607a60893b844`

Result: for all tested shapes, compilerless-generated param stream override reproduces base model output hash exactly.

## Significance
This is a direct operational proof that dense parameter stream generation no longer depends on `edgetpu_compiler` output bytes at replay time, provided row-major source bytes are available and packed via recovered formula over stored tensor shape.
