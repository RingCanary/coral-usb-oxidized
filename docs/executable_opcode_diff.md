# EXECUTION_ONLY Chunk Diff Workflow

`tools/exec_chunk_diff.py` compares instruction chunks from
`serialized_executable_*.bin` files to support Phase-B ISA work.

## Why

`parse_edgetpu_executable.py` already provides schema-aware summaries. This tool
adds direct chunk-byte diffing, including relocation-byte overlap and
instruction-word change counts.

## Usage

Compare two extracted executable directories:

```bash
python3 tools/exec_chunk_diff.py \
  traces/dense-template-256x256-20260222T062154Z/extract/package_000 \
  traces/dense-template-512x512-20260222T062006Z/extract/package_000 \
  --only-exec-index 0 \
  --out-dir traces/phase-b-diff-256-vs-512 \
  --dump-chunks
```

Compare multiple inputs (default: consecutive pairing):

```bash
python3 tools/exec_chunk_diff.py \
  traces/dense-template-1024x1024-20260222T062017Z/extract/package_000 \
  traces/dense-template-2048x2048-20260222T062027Z/extract/package_000 \
  traces/dense-template-2304x2304-20260222T062229Z/extract/package_000 \
  --only-exec-index 0 \
  --out-dir traces/phase-b-diff-size-sweep \
  --instruction-width 8 \
  --dump-chunks
```

Use explicit pair selection:

```bash
python3 tools/exec_chunk_diff.py \
  <pathA/serialized_executable_000.bin> \
  <pathB/serialized_executable_000.bin> \
  <pathC/serialized_executable_000.bin> \
  --pair <pathA/serialized_executable_000.bin>:<pathC/serialized_executable_000.bin>
```

## Output

- Human summary on stdout:
  - changed bytes per chunk
  - changed instruction words (configurable width)
  - relocation-overlap vs non-relocation changes
- JSON report:
  - `<out-dir>/exec_chunk_diff_report.json`
- Optional chunk dumps:
  - `<out-dir>/chunks/<label>/chunk_XXX.bin`
  - `<out-dir>/chunks/<label>/chunk_XXX.json`
  - `<out-dir>/chunk_dump_manifest.json`

## Interpretation Notes

- Large non-relocation delta with fixed shape often suggests true instruction
  encoding differences, not just address patching.
- Relocation-overlap-only deltas suggest pointer/base updates with otherwise
  stable code.
- If instruction-word change counts are clustered in small regions, those
  windows are strong candidates for opcode field analysis.
