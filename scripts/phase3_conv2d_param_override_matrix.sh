#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_HOST="${PI_HOST:-rpc@rpilm3.local}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_glmpitwo}"
REMOTE_REPO="${REMOTE_REPO:-/home/rpc/coral-usb-oxidized}"
FIRMWARE_REMOTE="${FIRMWARE_REMOTE:-/home/rpc/coral-usb-oxidized-lab/apex_latest_single_ep.bin}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/tmp/coral-m5-eo-probe-src}"
RUN_ID="phase3-conv2d-param-override-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/traces/analysis/$RUN_ID"
DUT_DIR="$REPO_ROOT/traces/analysis/specv3-$RUN_ID-dut"
mkdir -p "$OUT_DIR" "$DUT_DIR"

echo "run_id=$RUN_ID"
echo "pi_host=$PI_HOST"

SSH_OPTS=( -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" )
ssh_run() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }
rsync_pi() { rsync -av -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$@"; }

cases=(
  "32 32 64 64"
  "32 32 64 128"
  "32 32 128 64"
)

for spec in "${cases[@]}"; do
  read -r H W IC OC <<<"$spec"
  CASE_ID="h${H}_w${W}_ic${IC}_oc${OC}"
  CASE_DIR="$OUT_DIR/$CASE_ID"
  mkdir -p "$CASE_DIR"
  QUANT="$CASE_DIR/${CASE_ID}_quant.tflite"
  COMPILED="$CASE_DIR/${CASE_ID}_quant_edgetpu.tflite"
  ROW_MAJOR="$CASE_DIR/${CASE_ID}.row_major_i8.bin"
  PACKED="$CASE_DIR/${CASE_ID}.packed.bin"
  COMPILED_STREAM="$CASE_DIR/${CASE_ID}.compiled_param.bin"
  COMPARE_JSON="$CASE_DIR/${CASE_ID}.stream_vs_compiled_compare.json"

  uv run --python 3.9 --with tensorflow-cpu==2.10.1 --with numpy==1.23.5 \
    "$REPO_ROOT/tools/generate_conv2d_quant_tflite.py" \
    --output "$QUANT" \
    --metadata-out "$CASE_DIR/${CASE_ID}.quant_meta.json" \
    --height "$H" --width "$W" --in-channels "$IC" --out-channels "$OC" \
    --kernel-size 1 --stride 1 --padding same --init-mode random_uniform >/dev/null

  uv run --python 3.9 --with tensorflow-cpu==2.10.1 --with numpy==1.23.5 \
    "$REPO_ROOT/tools/dump_tflite_conv1x1_weights.py" \
    --input "$QUANT" --output "$ROW_MAJOR" \
    --in-channels "$IC" --out-channels "$OC" \
    --metadata-out "$CASE_DIR/${CASE_ID}.weight_dump_meta.json" >/dev/null

  edgetpu_compiler -s -o "$CASE_DIR" "$QUANT" > "$CASE_DIR/edgetpu_compile.log" 2>&1

  cargo run --quiet --bin model_param_stream_dump -- --model "$COMPILED" --out "$COMPILED_STREAM" --metadata-out "$CASE_DIR/${CASE_ID}.compiled_param_meta.json" >/dev/null
  cargo run --quiet --bin conv1x1_param_pack -- --in-channels "$IC" --out-channels "$OC" --stored-i8 "$ROW_MAJOR" --quant-model "$QUANT" --out "$PACKED" >/dev/null

  python3 - <<'PY' "$COMPILED_STREAM" "$PACKED" "$COMPARE_JSON" "$IC" "$OC"
import hashlib, json, pathlib, sys
compiled_path, packed_path, out_path, ic_s, oc_s = sys.argv[1:6]
ic = int(ic_s); oc = int(oc_s)
compiled = pathlib.Path(compiled_path).read_bytes()
packed = pathlib.Path(packed_path).read_bytes()

def block_widths(out_channels):
    rem = out_channels
    out = []
    while rem > 64:
        out.append(64)
        rem -= 64
    out.append(rem)
    return out
prefix_offsets = set()
block_start = 0
for bw in block_widths(oc):
    prefix_len = bw * 8
    prefix_offsets.update(range(block_start, block_start + prefix_len))
    block_start += bw * (8 + ic)
weight_mismatches = [i for i,(a,b) in enumerate(zip(compiled, packed)) if i not in prefix_offsets and a != b]
prefix_mismatches = [i for i,(a,b) in enumerate(zip(compiled, packed)) if i in prefix_offsets and a != b]
report = {
  'in_channels': ic,
  'out_channels': oc,
  'compiled_len': len(compiled),
  'packed_len': len(packed),
  'compiled_sha256': hashlib.sha256(compiled).hexdigest(),
  'packed_sha256': hashlib.sha256(packed).hexdigest(),
  'byte_equal': compiled == packed,
  'first_mismatch': next((i for i,(a,b) in enumerate(zip(compiled, packed)) if a != b), None),
  'prefix_offset_count': len(prefix_offsets),
  'prefix_mismatch_count': len(prefix_mismatches),
  'weight_mismatch_count': len(weight_mismatches),
  'weight_region_byte_equal': len(weight_mismatches) == 0,
}
pathlib.Path(out_path).write_text(json.dumps(report, indent=2) + '\n')
PY

done

ssh_run "mkdir -p '$REMOTE_SRC_DIR' '$REMOTE_REPO/traces/analysis/$RUN_ID' '$REMOTE_REPO/traces/analysis/specv3-$RUN_ID-dut'"
rsync -av --delete --exclude '.git' --exclude 'target' --exclude 'traces' -e "ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $SSH_KEY" "$REPO_ROOT/" "$PI_HOST:$REMOTE_SRC_DIR/" >/dev/null
ssh_run "cd '$REMOTE_SRC_DIR' && cargo build --example rusb_serialized_exec_replay >/dev/null"

for spec in "${cases[@]}"; do
  read -r H W IC OC <<<"$spec"
  CASE_ID="h${H}_w${W}_ic${IC}_oc${OC}"
  CASE_DIR="$OUT_DIR/$CASE_ID"
  MODEL_REL="traces/analysis/$RUN_ID/$CASE_ID/${CASE_ID}_quant_edgetpu.tflite"
  OVERRIDE_REL="traces/analysis/$RUN_ID/$CASE_ID/${CASE_ID}.packed.bin"
  REMOTE_MODEL="$REMOTE_REPO/$MODEL_REL"
  REMOTE_OVERRIDE="$REMOTE_REPO/$OVERRIDE_REL"
  rsync_pi "$CASE_DIR/" "$PI_HOST:$REMOTE_REPO/traces/analysis/$RUN_ID/$CASE_ID/" >/dev/null
  INPUT_BYTES=$((H * W * IC))
  OUTPUT_BYTES=$((H * W * OC))
  for mode in baseline override; do
    LOG="$DUT_DIR/${CASE_ID}_${mode}.log"
    CMD="cd '$REMOTE_SRC_DIR' && sudo target/debug/examples/rusb_serialized_exec_replay --model '$REMOTE_MODEL' --firmware '$FIRMWARE_REMOTE' --input-bytes '$INPUT_BYTES' --output-bytes '$OUTPUT_BYTES' --bootstrap-known-good-order --reset-before-claim --post-reset-sleep-ms 1200"
    if [[ "$mode" == "override" ]]; then
      CMD+=" --param-stream-override-file '$REMOTE_OVERRIDE'"
    fi
    ssh_run "$CMD" > "$LOG" 2>&1 || true
  done
done

python3 - <<'PY' "$OUT_DIR" "$DUT_DIR"
import json, pathlib, re, sys
out_dir = pathlib.Path(sys.argv[1]); dut_dir = pathlib.Path(sys.argv[2])
lines = [f"run_id={out_dir.name}"]
for case_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
    case = case_dir.name
    cmp = json.loads((case_dir / f"{case}.stream_vs_compiled_compare.json").read_text())
    lines.append(f"[{case}] local_byte_equal={cmp['byte_equal']} weight_region_byte_equal={cmp['weight_region_byte_equal']} prefix_mismatch_count={cmp['prefix_mismatch_count']} weight_mismatch_count={cmp['weight_mismatch_count']}")
    for mode in ['baseline','override']:
        log = (dut_dir / f"{case}_{mode}.log").read_text(errors='ignore')
        out = re.findall(r'Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)', log)
        err = re.findall(r'Error: (.+)', log)
        if out:
            lines.append(f"  {mode}: pass=True bytes={out[-1][0]} hash={out[-1][1]}")
        else:
            lines.append(f"  {mode}: pass=False error={err[-1] if err else None}")
    base_log = (dut_dir / f"{case}_baseline.log").read_text(errors='ignore')
    over_log = (dut_dir / f"{case}_override.log").read_text(errors='ignore')
    base = re.findall(r'Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)', base_log)
    over = re.findall(r'Output: bytes=([0-9]+) fnv1a64=(0x[0-9a-fA-F]+)', over_log)
    lines.append(f"  dut_hash_equal={bool(base and over and base[-1][1] == over[-1][1])}")
(out_dir / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n')
print(out_dir / 'SUMMARY.txt')
PY

echo "done: $OUT_DIR"
