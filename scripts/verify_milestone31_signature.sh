#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_ROOT/docs/milestone_manifest_8976_2352_2026-03-01.json"
REQUIRE_COUNT=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/verify_milestone31_signature.sh <run_dir> [--manifest <path>] [--require-count <n>]

Checks full DUT signature for each *.log in run_dir:
  - required patterns present (event + expected output hash)
  - forbidden error patterns absent
  - output bytes/hash exactly match manifest expectation
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

RUN_DIR="$1"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --require-count)
      REQUIRE_COUNT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

python - "$MANIFEST" "$RUN_DIR" "$REQUIRE_COUNT" <<'PY'
import glob
import json
import os
import re
import sys

manifest_path, run_dir, require_count_raw = sys.argv[1:4]

if not os.path.isfile(manifest_path):
    raise SystemExit(f"manifest not found: {manifest_path}")
if not os.path.isdir(run_dir):
    raise SystemExit(f"run_dir not found: {run_dir}")

manifest = json.load(open(manifest_path))
exp = manifest["expected_signature"]
required = list(exp.get("required_patterns", []))
forbidden = list(exp.get("forbidden_patterns", []))
expected_hash = str(exp["output_hash_fnv1a64"]).lower()
expected_bytes = int(exp["output_bytes"])
expected_event_tag = int(exp["event_tag"])

logs = sorted(glob.glob(os.path.join(run_dir, "*.log")))
if not logs:
    raise SystemExit(f"no .log files found in {run_dir}")

if require_count_raw:
    req = int(require_count_raw)
    if len(logs) != req:
        raise SystemExit(f"expected {req} logs, found {len(logs)} in {run_dir}")

pat_out = re.compile(r"Output: bytes=(\d+) fnv1a64=(0x[0-9a-f]+)")
pat_event = re.compile(r"Event: tag=(\d+)")

rows = []
failed = False
for path in logs:
    name = os.path.basename(path)
    text = open(path, "r", encoding="utf-8", errors="replace").read()

    reasons = []

    for rp in required:
        if rp not in text:
            reasons.append(f"missing_required:{rp}")

    for fp in forbidden:
        if fp in text:
            reasons.append(f"forbidden_seen:{fp}")

    # explicit event check
    event_tags = [int(m.group(1)) for m in pat_event.finditer(text)]
    if expected_event_tag not in event_tags:
        reasons.append(f"event_tag_missing:{expected_event_tag}")

    outs = pat_out.findall(text)
    out_hash = "-"
    if not outs:
        reasons.append("output_missing")
    else:
        b, h = outs[-1]
        out_hash = h.lower()
        if int(b) != expected_bytes:
            reasons.append(f"output_bytes_mismatch:{b}!={expected_bytes}")
        if out_hash != expected_hash:
            reasons.append(f"hash_mismatch:{out_hash}!={expected_hash}")

    status = "PASS" if not reasons else "FAIL"
    if status == "FAIL":
        failed = True
    rows.append((name, status, out_hash, ";".join(reasons) if reasons else "-"))

print("case\tstatus\thash\treason")
for row in rows:
    print("\t".join(row))

if failed:
    raise SystemExit(1)

print(f"PASS: verified {len(rows)} logs in {run_dir}")
PY
