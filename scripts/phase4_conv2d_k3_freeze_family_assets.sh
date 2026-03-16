#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ARTIFACT_REL="${SOURCE_ARTIFACT_REL:-traces/analysis/phase4-conv2d-k3-crossdim-oracle-matrix-20260306T143012Z}"
OUT_DIR_REL="${OUT_DIR_REL:-templates/phase4_conv2d_k3_sameprod_6512}"
TMP_RUN_ID="phase4-conv2d-k3-freeze-family-$(date -u +%Y%m%dT%H%M%SZ)"
TMP_DIR="$REPO_ROOT/traces/analysis/$TMP_RUN_ID"
SOURCE_ARTIFACT="$REPO_ROOT/$SOURCE_ARTIFACT_REL"
OUT_DIR="$REPO_ROOT/$OUT_DIR_REL"
SAME_PRODUCT=1024
SEED="${SEED:-1337}"

pairs=(
  "p32 32"
  "p64 64"
  "p128 128"
)

target_heights=(64 128)

copy_required_file() {
  local src="$1"
  local dst="$2"
  [[ -f "$src" ]] || {
    echo "missing file: $src" >&2
    exit 1
  }
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 1
  }
}

need_cmd cargo
need_cmd python3

[[ -d "$SOURCE_ARTIFACT" ]] || {
  echo "source artifact not found: $SOURCE_ARTIFACT" >&2
  exit 1
}

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR" "$TMP_DIR"

for spec in "${pairs[@]}"; do
  read -r pair channels <<<"$spec"
  pair_src="$SOURCE_ARTIFACT/$pair"
  pair_out="$OUT_DIR/$pair"
  anchor_dir="$pair_out/anchor"
  target32_dir="$pair_out/targets/h32_w32"
  tmp_pair_dir="$TMP_DIR/$pair"

  mkdir -p "$anchor_dir" "$target32_dir" "$tmp_pair_dir"

  anchor_quant=$(find "$pair_src/anchor" -maxdepth 1 -name '*.tflite' ! -name '*_edgetpu.tflite' | head -n 1)
  anchor_compiled=$(find "$pair_src/anchor" -maxdepth 1 -name '*_edgetpu.tflite' | head -n 1)
  anchor_meta=$(find "$pair_src/anchor" -maxdepth 1 -name '*.metadata.json' | head -n 1)
  target32_quant=$(find "$pair_src/target" -maxdepth 1 -name '*.tflite' ! -name '*_edgetpu.tflite' | head -n 1)
  target32_compiled=$(find "$pair_src/target" -maxdepth 1 -name '*_edgetpu.tflite' | head -n 1)
  target32_meta=$(find "$pair_src/target" -maxdepth 1 -name '*.metadata.json' | head -n 1)

  copy_required_file "$anchor_quant" "$anchor_dir/$(basename "$anchor_quant")"
  copy_required_file "$anchor_compiled" "$anchor_dir/$(basename "$anchor_compiled")"
  copy_required_file "$anchor_meta" "$anchor_dir/$(basename "$anchor_meta")"
  copy_required_file "$target32_quant" "$target32_dir/$(basename "$target32_quant")"
  copy_required_file "$target32_compiled" "$target32_dir/$(basename "$target32_compiled")"
  copy_required_file "$target32_meta" "$target32_dir/$(basename "$target32_meta")"
  copy_required_file "$pair_src/eo_oracle.patchspec" "$tmp_pair_dir/eo_h32_w32.patchspec"

  anchor_exec="$pair_src/anchor/extract/package_000/serialized_executable_000.bin"
  [[ -f "$anchor_exec" ]] || {
    echo "missing anchor executable chunk: $anchor_exec" >&2
    exit 1
  }

  for target_height in "${target_heights[@]}"; do
    target_width=$((SAME_PRODUCT / target_height))
    case_dir="$tmp_pair_dir/h${target_height}_w${target_width}"
    "$REPO_ROOT/tools/conv_template_pipeline.sh" \
      --out-dir "$case_dir" \
      --height "$target_height" \
      --width "$target_width" \
      --in-channels "$channels" \
      --out-channels "$channels" \
      --kernel-size 3 \
      --stride 1 \
      --padding same \
      --init-mode random_uniform \
      --seed "$SEED" >/dev/null

    target_exec="$case_dir/extract/package_000/serialized_executable_000.bin"
    patch_out="$tmp_pair_dir/eo_h${target_height}_w${target_width}.patchspec"
    cargo run --quiet --bin instruction_chunk_patchspec -- \
      --base-exec "$anchor_exec" \
      --target-exec "$target_exec" \
      --out-patchspec "$patch_out" >/dev/null

    target_out_dir="$pair_out/targets/h${target_height}_w${target_width}"
    mkdir -p "$target_out_dir"
    target_quant=$(find "$case_dir" -maxdepth 1 -name '*.tflite' ! -name '*_edgetpu.tflite' | head -n 1)
    target_compiled=$(find "$case_dir" -maxdepth 1 -name '*_edgetpu.tflite' | head -n 1)
    target_meta=$(find "$case_dir" -maxdepth 1 -name '*.metadata.json' | head -n 1)
    copy_required_file "$target_quant" "$target_out_dir/$(basename "$target_quant")"
    copy_required_file "$target_compiled" "$target_out_dir/$(basename "$target_compiled")"
    copy_required_file "$target_meta" "$target_out_dir/$(basename "$target_meta")"
  done
done

python3 - <<'PY' "$OUT_DIR" "$TMP_DIR" "$SOURCE_ARTIFACT_REL" "$OUT_DIR_REL"
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
tmp_dir = pathlib.Path(sys.argv[2])
source_artifact_rel = sys.argv[3]
out_dir_rel = sys.argv[4]
payload_len_expected = 6512

regimes = [
    ("p32", 32),
    ("p64", 64),
    ("p128", 128),
]

targets = [
    (16, 64, "noop"),
    (32, 32, "table"),
    (64, 16, "table"),
    (128, 8, "table"),
]


def rel(path: pathlib.Path) -> str:
    return str(path.relative_to(out_dir))


def find_single(base: pathlib.Path, pattern: str) -> pathlib.Path:
    matches = list(base.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one match for {pattern} under {base}, got {matches}")
    return matches[0]


def find_single_quant_model(base: pathlib.Path) -> pathlib.Path:
    matches = [p for p in base.glob("*.tflite") if not p.name.endswith("_edgetpu.tflite")]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one uncompiled .tflite under {base}, got {matches}")
    return matches[0]


def read_patch(path: pathlib.Path):
    if not path.exists():
        return []
    rules = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        clean = raw.split("#", 1)[0].replace(",", " ").strip()
        if not clean:
            continue
        fields = clean.split()
        if len(fields) != 3:
            raise SystemExit(f"invalid patch line in {path}: {raw}")
        payload_len = int(fields[0], 0)
        if payload_len != payload_len_expected:
            raise SystemExit(
                f"unexpected payload_len in {path}: {payload_len} != {payload_len_expected}"
            )
        rules.append([int(fields[1], 0), int(fields[2], 0)])
    return rules


spec = {
    "schema_version": 1,
    "family_id": "phase4_conv2d_k3_sameprod_6512_v1",
    "same_product": 1024,
    "anchor_height": 16,
    "anchor_width": 64,
    "eo_payload_len": payload_len_expected,
    "kernel_size": 3,
    "stride": 1,
    "padding": "same",
    "bias": False,
    "source_artifact": source_artifact_rel,
    "asset_root": out_dir_rel,
    "regimes": [],
}

summary_lines = [
    f"family_id={spec['family_id']}",
    f"source_artifact={source_artifact_rel}",
    f"asset_root={out_dir_rel}",
]

for pair, channels in regimes:
    pair_dir = out_dir / pair
    anchor_dir = pair_dir / "anchor"
    anchor_quant = find_single_quant_model(anchor_dir)
    anchor_compiled = find_single(anchor_dir, "*_edgetpu.tflite")
    anchor_meta = find_single(anchor_dir, "*.metadata.json")
    regime_entry = {
        "name": pair,
        "channels": channels,
        "anchor_compiled_model": rel(anchor_compiled),
        "anchor_uncompiled_model": rel(anchor_quant),
        "anchor_metadata": rel(anchor_meta),
        "targets": [],
    }
    summary_lines.append(f"[{pair}] channels={channels}")
    for height, width, source_kind in targets:
        if height == 16:
            target_quant = anchor_quant
            target_compiled = anchor_compiled
            target_meta = anchor_meta
            rules = []
        else:
            target_dir = pair_dir / "targets" / f"h{height}_w{width}"
            target_quant = find_single_quant_model(target_dir)
            target_compiled = find_single(target_dir, "*_edgetpu.tflite")
            target_meta = find_single(target_dir, "*.metadata.json")
            patch_path = tmp_dir / pair / f"eo_h{height}_w{width}.patchspec"
            rules = read_patch(patch_path)
        regime_entry["targets"].append(
            {
                "height": height,
                "width": width,
                "target_model": rel(target_quant),
                "target_compiled_model": rel(target_compiled),
                "target_metadata": rel(target_meta),
                "source_kind": source_kind,
                "rules": rules,
            }
        )
        summary_lines.append(
            f"  h{height}_w{width}: source={source_kind} rule_count={len(rules)} model={rel(target_quant)}"
        )
    spec["regimes"].append(regime_entry)

(out_dir / "family.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
(out_dir / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print(out_dir / "family.json")
print(out_dir / "SUMMARY.txt")
PY

echo "family assets frozen under: $OUT_DIR"
echo "temporary build artifact kept under: $TMP_DIR"
