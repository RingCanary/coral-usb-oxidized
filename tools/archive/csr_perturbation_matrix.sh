#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Run a controlled CSR perturbation matrix against bundled GEMM invoke.

Usage:
  tools/csr_perturbation_matrix.sh [runs]

Args:
  runs   Number of GEMM executes per case (default: 1)

Outputs:
  traces/csr-perturb-<timestamp>/
    - *.log per case
    - summary.tsv
USAGE
  exit 0
fi

RUNS="${1:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="traces/csr-perturb-${STAMP}"
mkdir -p "${OUT_DIR}"

run_case() {
  local name="$1"
  shift
  local log="${OUT_DIR}/${name}.log"
  local rc=0
  echo "[case:${name}] cargo run --example gemm_csr_perturb_probe -- $*" | tee "${log}"
  set +e
  timeout 90s cargo run --example gemm_csr_perturb_probe -- "$@" >>"${log}" 2>&1
  rc=$?
  set -e
  local result_line
  result_line="$( (rg -n '^RESULT ' "${log}" -N || true) | tail -n 1 | sed 's/^[0-9]*://')"
  local meta_line
  meta_line="$( (rg -n '^RESULT_META ' "${log}" -N || true) | tail -n 1 | sed 's/^[0-9]*://')"
  local status="missing"
  if [[ -n "${result_line}" ]]; then
    status="$(echo "${result_line}" | sed -n 's/.*status=\([^ ]*\).*/\1/p')"
  elif rg -q "DelegateCreationFailed" "${log}"; then
    status="delegate_failed"
  elif rg -q "Abort" "${log}"; then
    status="runtime_abort"
  elif [[ "${rc}" -eq 124 ]]; then
    status="timeout"
  elif [[ "${rc}" -ne 0 ]]; then
    status="command_failed"
  fi
  echo -e "${name}\t${rc}\t${status}\t${result_line}\t${meta_line}" >> "${OUT_DIR}/summary.tsv"
}

{
  echo -e "case\trc\tstatus\tresult\tmeta"
} > "${OUT_DIR}/summary.tsv"

# Baseline control
run_case "baseline" "none" "2048" "identity" "${RUNS}" "1"

# Core run control and tile controls
run_case "runcontrol_0" "0x00044018" "64" "0x0" "2048" "identity" "${RUNS}" "1"
run_case "runcontrol_2" "0x00044018" "64" "0x2" "2048" "identity" "${RUNS}" "1"
run_case "tileconfig_0" "0x00048788" "64" "0x0" "2048" "identity" "${RUNS}" "1"
run_case "tileconfig_1" "0x00048788" "64" "0x1" "2048" "identity" "${RUNS}" "1"
run_case "tileconfig_7f" "0x00048788" "64" "0x7f" "2048" "identity" "${RUNS}" "1"

# Group-A control candidates observed in runtime traces
run_case "scu7_lowmask" "0x0001a33c" "32" "0x0000003f" "2048" "identity" "${RUNS}" "1"
run_case "scu7_highmask" "0x0001a33c" "32" "0x000c003f" "2048" "identity" "${RUNS}" "1"
run_case "rambist_low" "0x0001a704" "32" "0x0000007f" "2048" "identity" "${RUNS}" "1"
run_case "rambist_high" "0x0001a704" "32" "0x0070007f" "2048" "identity" "${RUNS}" "1"
run_case "slv_abm_off" "0x0001a500" "32" "0x00000000" "2048" "identity" "${RUNS}" "1"
run_case "slv_abm_on" "0x0001a500" "32" "0x00000001" "2048" "identity" "${RUNS}" "1"
run_case "mst_abm_off" "0x0001a600" "32" "0x00000000" "2048" "identity" "${RUNS}" "1"
run_case "mst_abm_on" "0x0001a600" "32" "0x00000001" "2048" "identity" "${RUNS}" "1"

echo "Wrote ${OUT_DIR}/summary.tsv"
