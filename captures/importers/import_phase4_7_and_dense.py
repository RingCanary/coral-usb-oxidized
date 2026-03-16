#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import (
    CAPTURES_ROOT,
    REPO_ROOT,
    WORKSPACE_ROOT,
    copy_text_evidence,
    now_utc,
    read_json,
    read_key_value_file,
    relativize_to_repo,
    repo_commit,
    sha256_file,
    write_csv,
    write_json,
    write_table,
)


CONV_ROOT = WORKSPACE_ROOT / "arsenal" / "coral-rusb-replay-src" / "traces" / "analysis"
DENSE_ROOT = WORKSPACE_ROOT / "arsenal" / "coral-usb-oxidized-lab" / "traces"
INVENTORY_PATH = CAPTURES_ROOT / "inventory_phase4_7_dense.csv"
INDEX_PATH = CAPTURES_ROOT / "index.tsv"
LEDGER_PATH = CAPTURES_ROOT / "contract_ledger.tsv"

FRONTIER = "00_trace_contract"

INVENTORY_COLUMNS = [
    "corpus",
    "row_kind",
    "frontier",
    "run_id",
    "run_path",
    "artifact_family",
    "phase",
    "date_utc",
    "scenario",
    "regime",
    "shape",
    "source_status",
    "summary_path",
    "metadata_path",
    "tflite_path",
    "compiled_tflite_path",
    "exec_path",
    "param_path",
    "patchspec_path",
    "usbmon_log_path",
    "usbmon_summary_path",
    "dut_log_path",
    "admission_outcome",
    "hash_eq_target",
    "failure_mode",
    "firmware_path",
    "anchor_ref",
    "target_ref",
    "notes",
]

INDEX_COLUMNS = (CAPTURES_ROOT / "schema" / "index.columns.tsv").read_text(encoding="utf-8").splitlines()
LEDGER_COLUMNS = (CAPTURES_ROOT / "schema" / "contract_ledger.columns.tsv").read_text(encoding="utf-8").splitlines()


def parse_run_date(name: str) -> str:
    match = re.search(r"(20\d{2})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z", name)
    if not match:
        return ""
    y, m, d, hh, mm, ss = match.groups()
    return f"{y}-{m}-{d}T{hh}:{mm}:{ss}Z"


def first_match(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def inventory_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for run_dir in sorted(CONV_ROOT.glob("phase4-conv2d-k3-completion-demo-20260316T*")):
        rows.append(
            {
                "corpus": "conv_phase4_7",
                "row_kind": "run",
                "frontier": FRONTIER,
                "run_id": run_dir.name,
                "run_path": str(run_dir),
                "artifact_family": "conv_completion_demo",
                "phase": infer_conv_phase(run_dir.name),
                "date_utc": parse_run_date(run_dir.name),
                "source_status": "present",
                "notes": "completion demo run root",
            }
        )
        for regime_dir in sorted(run_dir.glob("p*")):
            for case_dir in sorted(regime_dir.glob("h*")):
                eo_report = case_dir / "eo_report.json"
                report = read_json(eo_report) if eo_report.exists() else {}
                rows.append(
                    {
                        "corpus": "conv_phase4_7",
                        "row_kind": "case",
                        "frontier": FRONTIER,
                        "run_id": run_dir.name,
                        "run_path": str(case_dir),
                        "artifact_family": str(report.get("family_id", "conv_case")),
                        "phase": infer_conv_phase(run_dir.name),
                        "date_utc": parse_run_date(run_dir.name),
                        "scenario": "historical_case",
                        "regime": regime_dir.name,
                        "shape": case_dir.name,
                        "source_status": "present",
                        "summary_path": "",
                        "metadata_path": str(Path(str(report.get("target_metadata", "")))) if report else "",
                        "tflite_path": str(Path(str(report.get("target_model", "")))) if report else "",
                        "compiled_tflite_path": str(Path(str(report.get("target_compiled_model", "")))) if report else "",
                        "exec_path": "",
                        "param_path": str(case_dir / "target_param_stream.bin"),
                        "patchspec_path": str(case_dir / "eo.patchspec"),
                        "admission_outcome": "unknown",
                        "hash_eq_target": "unknown",
                        "failure_mode": "summary_missing",
                        "anchor_ref": str(report.get("anchor_metadata", "")) if report else "",
                        "target_ref": str(report.get("target_metadata", "")) if report else "",
                        "notes": "completion-demo case with per-case EO artifacts only",
                    }
                )

    for run_dir in sorted(CONV_ROOT.glob("phase7-p32-tail-dut-proof-20260316T*")):
        summary = read_key_value_file(run_dir / "SUMMARY.txt")
        rows.append(
            {
                "corpus": "conv_phase4_7",
                "row_kind": "run",
                "frontier": FRONTIER,
                "run_id": run_dir.name,
                "run_path": str(run_dir),
                "artifact_family": "phase7_tail_proof",
                "phase": "phase7_tail",
                "date_utc": parse_run_date(run_dir.name),
                "source_status": "present",
                "summary_path": str(run_dir / "SUMMARY.txt"),
                "firmware_path": summary.get("firmware", ""),
                "admission_outcome": "pass" if summary.get("all_hash_eq_target") == "true" else "unknown",
                "hash_eq_target": summary.get("all_hash_eq_target", "unknown"),
                "notes": "tail proof run root",
            }
        )
        for case_dir in sorted((run_dir / "cases").glob("h*")):
            shape = case_dir.name
            case_summary = summary_for_tail_shape(summary, shape)
            rows.append(
                {
                    "corpus": "conv_phase4_7",
                    "row_kind": "case",
                    "frontier": FRONTIER,
                    "run_id": run_dir.name,
                    "run_path": str(case_dir),
                    "artifact_family": "phase7_tail_proof_case",
                    "phase": "phase7_tail",
                    "date_utc": parse_run_date(run_dir.name),
                    "scenario": "tail_recovery",
                    "regime": "p32",
                    "shape": shape,
                    "source_status": "present",
                    "summary_path": str(run_dir / "SUMMARY.txt"),
                    "param_path": str(case_dir / "native_param.bin"),
                    "patchspec_path": str(case_dir / "eo.patchspec"),
                    "dut_log_path": str(run_dir / "dut" / f"{shape}_native_completion.log"),
                    "admission_outcome": "pass",
                    "hash_eq_target": case_summary.get("hash_eq_target", "unknown"),
                    "failure_mode": "",
                    "firmware_path": summary.get("firmware", ""),
                    "notes": "tail proof case with DUT and param parity logs",
                }
            )

    for run_dir in sorted(DENSE_ROOT.glob("dense-template-*")):
        metadata = first_match(run_dir, "*.metadata.json")
        model = first_match(run_dir, "*_quant.tflite")
        compiled = first_match(run_dir, "*_quant_edgetpu.tflite")
        rows.append(
            {
                "corpus": "dense_template",
                "row_kind": "run",
                "frontier": FRONTIER,
                "run_id": run_dir.name,
                "run_path": str(run_dir),
                "artifact_family": "dense_template",
                "phase": "phase2_dense",
                "date_utc": parse_run_date(run_dir.name),
                "source_status": "present",
                "summary_path": str(run_dir / "PIPELINE_SUMMARY.txt"),
                "metadata_path": str(metadata) if metadata else "",
                "tflite_path": str(model) if model else "",
                "compiled_tflite_path": str(compiled) if compiled else "",
                "exec_path": str(run_dir / "extract" / "package_000" / "serialized_executable_000.bin"),
                "shape": infer_dense_shape(run_dir.name),
                "notes": "dense template pipeline artifact",
            }
        )

    for run_dir in sorted(DENSE_ROOT.glob("dense-layout-probe-*")):
        rows.append(
            {
                "corpus": "dense_layout_probe",
                "row_kind": "run",
                "frontier": FRONTIER,
                "run_id": run_dir.name,
                "run_path": str(run_dir),
                "artifact_family": "dense_layout_probe",
                "phase": "phase2_dense",
                "date_utc": parse_run_date(run_dir.name),
                "source_status": "present",
                "summary_path": str(run_dir / "layout_probe.txt"),
                "metadata_path": str(run_dir / "layout_probe.json"),
                "shape": "",
                "notes": "dense layout probe run root",
            }
        )

    for run_dir in sorted(DENSE_ROOT.glob("replay-csr-snapshot-*")):
        for scenario_dir in sorted(run_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            summary = first_match(scenario_dir, "*.summary.txt")
            log = first_match(scenario_dir, "*.log")
            rows.append(
                {
                    "corpus": "dense_usbmon",
                    "row_kind": "scenario",
                    "frontier": FRONTIER,
                    "run_id": run_dir.name,
                    "run_path": str(scenario_dir),
                    "artifact_family": "dense_usbmon_snapshot",
                    "phase": "phase2_dense",
                    "date_utc": parse_run_date(run_dir.name),
                    "scenario": scenario_dir.name,
                    "source_status": "present",
                    "summary_path": str(summary) if summary else "",
                    "usbmon_log_path": str(log) if log else "",
                    "usbmon_summary_path": str(summary) if summary else "",
                    "admission_outcome": "command_exit_1",
                    "hash_eq_target": "unknown",
                    "failure_mode": "command_exit_1",
                    "firmware_path": "./apex_latest_single_ep.bin",
                    "notes": "usbmon snapshot scenario",
                }
            )

    for run_dir in sorted(DENSE_ROOT.glob("replay-gate-placement-*")):
        rows.append(
            {
                "corpus": "dense_usbmon",
                "row_kind": "run",
                "frontier": FRONTIER,
                "run_id": run_dir.name,
                "run_path": str(run_dir),
                "artifact_family": "dense_gate_placement",
                "phase": "phase2_dense",
                "date_utc": parse_run_date(run_dir.name),
                "source_status": "present",
                "notes": "gate placement run root",
            }
        )

    for run_dir in sorted(DENSE_ROOT.glob("replay-keepalive-cadence*")):
        rows.append(
            {
                "corpus": "dense_usbmon",
                "row_kind": "run",
                "frontier": FRONTIER,
                "run_id": run_dir.name,
                "run_path": str(run_dir),
                "artifact_family": "dense_keepalive_cadence",
                "phase": "phase2_dense",
                "date_utc": parse_run_date(run_dir.name),
                "source_status": "present",
                "notes": "keepalive cadence run root",
            }
        )

    for run_dir in sorted(DENSE_ROOT.glob("phase-b-diff-*")):
        rows.append(
            {
                "corpus": "dense_exec_diff",
                "row_kind": "run",
                "frontier": FRONTIER,
                "run_id": run_dir.name,
                "run_path": str(run_dir),
                "artifact_family": "dense_exec_diff",
                "phase": "phase2_dense",
                "date_utc": parse_run_date(run_dir.name),
                "source_status": "present",
                "notes": "dense executable diff artifact",
            }
        )

    for missing_run in (
        "phase4-conv2d-k3-family-scout-20260316T114108Z",
        "phase7-conv2d-k3-p32-tail-param-probe-20260316T121528Z",
    ):
        rows.append(
            {
                "corpus": "conv_phase4_7",
                "row_kind": "referenced_missing",
                "frontier": FRONTIER,
                "run_id": missing_run,
                "run_path": str(CONV_ROOT / missing_run),
                "artifact_family": "referenced_missing",
                "phase": "phase4_7_reference",
                "date_utc": parse_run_date(missing_run),
                "source_status": "stale_reference",
                "failure_mode": "stale_reference",
                "notes": "referenced by docs/template metadata but not present in workspace",
            }
        )

    return rows


def infer_conv_phase(run_name: str) -> str:
    mapping = {
        "065217": "phase4",
        "092300": "phase5",
        "102326": "phase6",
        "114632": "phase7_overwide",
        "115420": "phase7",
    }
    for marker, phase in mapping.items():
        if marker in run_name:
            return phase
    return "phase4_7"


def infer_dense_shape(run_name: str) -> str:
    match = re.search(r"dense-template-(\d+x\d+)-", run_name)
    return match.group(1) if match else ""


def summary_for_tail_shape(summary: dict[str, str], shape: str) -> dict[str, str]:
    result = {"hash_eq_target": "unknown"}
    prefix = f"{shape} "
    for key, value in summary.items():
        if not key.startswith(prefix):
            continue
        result["hash_eq_target"] = value.split("hash_eq_target=")[-1] if "hash_eq_target=" in value else "unknown"
    line = next((v for k, v in summary.items() if k == shape), "")
    if line:
        result["hash_eq_target"] = line.split("hash_eq_target=")[-1]
    for raw_key, raw_value in summary.items():
        if raw_key.startswith(shape):
            result["hash_eq_target"] = raw_value.split("hash_eq_target=")[-1]
    return result


def inventory_command() -> None:
    rows = inventory_rows()
    write_csv(INVENTORY_PATH, INVENTORY_COLUMNS, rows)


SELECTED_CAPTURES = [
    {
        "capture_id": "conv_phase6_p32_h8_w128_structural",
        "kind": "conv_case",
        "source_path": CONV_ROOT / "phase4-conv2d-k3-completion-demo-20260316T102326Z" / "p32" / "h8_w128",
        "run_id": "phase4-conv2d-k3-completion-demo-20260316T102326Z",
        "baseline_capture_id": "",
        "nearest_counterexample_id": "conv_phase7_overwide_p32_h12_w176_structural",
        "comparison_class": "historical_structural_pair",
    },
    {
        "capture_id": "conv_phase7_overwide_p32_h12_w176_structural",
        "kind": "conv_case",
        "source_path": CONV_ROOT / "phase4-conv2d-k3-completion-demo-20260316T114632Z" / "p32" / "h12_w176",
        "run_id": "phase4-conv2d-k3-completion-demo-20260316T114632Z",
        "baseline_capture_id": "conv_phase6_p32_h8_w128_structural",
        "nearest_counterexample_id": "conv_phase7_tail_p32_h12_w176_recovered",
        "comparison_class": "historical_structural_pair",
    },
    {
        "capture_id": "conv_phase7_tail_p32_h12_w176_recovered",
        "kind": "tail_case",
        "source_path": CONV_ROOT / "phase7-p32-tail-dut-proof-20260316T145105Z" / "cases" / "h12_w176",
        "run_id": "phase7-p32-tail-dut-proof-20260316T145105Z",
        "baseline_capture_id": "conv_phase7_overwide_p32_h12_w176_structural",
        "nearest_counterexample_id": "conv_phase7_overwide_p32_h12_w176_structural",
        "comparison_class": "recovery_pair",
    },
    {
        "capture_id": "dense_template_1024x1024_stock",
        "kind": "dense_template",
        "source_path": DENSE_ROOT / "dense-template-1024x1024-20260222T062017Z",
        "run_id": "dense-template-1024x1024-20260222T062017Z",
        "baseline_capture_id": "",
        "nearest_counterexample_id": "dense_gate_snapshot_20260225_before",
        "comparison_class": "dense_stock_vs_perturbed",
    },
    {
        "capture_id": "dense_csr_baseline_snapshot",
        "kind": "dense_usbmon",
        "source_path": DENSE_ROOT / "replay-csr-snapshot-20260225T141800Z" / "baseline_snapshot",
        "run_id": "replay-csr-snapshot-20260225T141800Z",
        "baseline_capture_id": "dense_template_1024x1024_stock",
        "nearest_counterexample_id": "dense_gate_snapshot_20260225_before",
        "comparison_class": "dense_stock_vs_perturbed",
    },
    {
        "capture_id": "dense_gate_snapshot_20260225_before",
        "kind": "dense_usbmon",
        "source_path": DENSE_ROOT / "replay-csr-snapshot-20260225T141800Z" / "dense_gate_snapshot",
        "run_id": "replay-csr-snapshot-20260225T141800Z",
        "baseline_capture_id": "dense_csr_baseline_snapshot",
        "nearest_counterexample_id": "dense_csr_baseline_snapshot",
        "comparison_class": "dense_stock_vs_perturbed",
    },
]


def import_sample_command() -> None:
    imported_rows = []
    ledger_rows = []
    for spec in SELECTED_CAPTURES:
        capture_dir = CAPTURES_ROOT / spec["capture_id"]
        evidence_dir = capture_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "conv_case":
            row, ledger = import_conv_case(spec, capture_dir, evidence_dir)
        elif spec["kind"] == "tail_case":
            row, ledger = import_tail_case(spec, capture_dir, evidence_dir)
        elif spec["kind"] == "dense_template":
            row, ledger = import_dense_template(spec, capture_dir, evidence_dir)
        elif spec["kind"] == "dense_usbmon":
            row, ledger = import_dense_usbmon(spec, capture_dir, evidence_dir)
        else:
            raise SystemExit(f"unsupported capture kind: {spec['kind']}")
        imported_rows.append(row)
        ledger_rows.append(ledger)

    write_table(INDEX_PATH, INDEX_COLUMNS, imported_rows)
    write_table(LEDGER_PATH, LEDGER_COLUMNS, ledger_rows)


def import_conv_case(spec: dict[str, str], capture_dir: Path, evidence_dir: Path):
    report = read_json(spec["source_path"] / "eo_report.json")
    eo_patch = spec["source_path"] / "eo.patchspec"
    eo_txt = spec["source_path"] / "eo_report.txt"
    eo_json = spec["source_path"] / "eo_report.json"
    param_stream = spec["source_path"] / "target_param_stream.bin"
    copy_text_evidence(eo_patch, evidence_dir)
    copy_text_evidence(eo_txt, evidence_dir)
    copy_text_evidence(eo_json, evidence_dir)

    model_path = normalize_legacy_path(str(report.get("target_model", "")))
    compiled_model = normalize_legacy_path(str(report.get("target_compiled_model", "")))
    metadata_path = normalize_legacy_path(str(report.get("target_metadata", "")))
    anchor_path = normalize_legacy_path(str(report.get("anchor_metadata", "")))

    metadata = {
        "capture_id": spec["capture_id"],
        "frontier": FRONTIER,
        "capture_version": 1,
        "source": {
            "root": str(spec["source_path"]),
            "run_id": spec["run_id"],
            "kind": "historical_conv_case",
            "imported_at_utc": now_utc(),
        },
        "provenance": {
            "repo_path": str(REPO_ROOT),
            "repo_commit": repo_commit(),
            "workspace_root": str(WORKSPACE_ROOT),
        },
        "firmware": {
            "path": "",
            "sha256": "",
            "provenance_note": "historical completion-demo case has no colocated firmware summary",
        },
        "artifacts": {
            "model": artifact_ref(model_path),
            "metadata_json": artifact_ref(metadata_path),
            "serialized_executable": artifact_ref(None),
            "param_stream": artifact_ref(param_stream),
            "eo_patchspec": artifact_ref(eo_patch),
            "dut_logs": [],
            "usbmon_logs": [],
        },
        "identity": {
            "family": str(report.get("family_id", "")),
            "regime": str(report.get("regime_name", "")),
            "shape": f"h{report.get('target_height', '')}_w{report.get('target_width', '')}",
            "operator": "conv2d_k3_same_bias_off",
        },
        "outcome": {
            "transport_admission": "unknown",
            "dut_hash": "",
            "hash_eq_target": "unknown",
            "exact_failure_mode": "summary_missing",
        },
        "comparison": {
            "baseline_capture_id": spec["baseline_capture_id"],
            "nearest_counterexample_id": spec["nearest_counterexample_id"],
            "comparison_class": spec["comparison_class"],
        },
        "decision": {
            "keep_discard": "keep",
            "reason": "structural historical import; direct run summary absent",
        },
        "notes": {
            "anchor_ref": str(anchor_path) if anchor_path else "",
            "compiled_model": str(compiled_model) if compiled_model else "",
        },
    }
    write_json(capture_dir / "metadata.json", metadata)
    summary_lines = [
        f"# {spec['capture_id']}",
        "",
        f"- source: `{spec['source_path']}`",
        "- imported evidence: `eo.patchspec`, `eo_report.json`, `eo_report.txt`",
        f"- referenced model: `{model_path}`",
        f"- referenced param stream: `{param_stream}`",
        "- outcome: `summary_missing`; no direct DUT summary in stored run copy",
        f"- nearest counterexample: `{spec['nearest_counterexample_id']}`",
    ]
    (capture_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    row = {
        "capture_id": spec["capture_id"],
        "source_root": str(spec["source_path"]),
        "source_run_id": spec["run_id"],
        "source_kind": "historical_conv_case",
        "frontier": FRONTIER,
        "family": str(report.get("family_id", "")),
        "regime": str(report.get("regime_name", "")),
        "shape": f"h{report.get('target_height', '')}_w{report.get('target_width', '')}",
        "artifact_date": parse_run_date(spec["run_id"]),
        "firmware_path": "",
        "firmware_sha256": "",
        "model_id": model_path.name if model_path else "",
        "model_sha256": sha256_file(model_path),
        "metadata_path": relativize_to_repo(capture_dir / "metadata.json"),
        "exec_id": "",
        "exec_path": "",
        "params_id": param_stream.name,
        "params_path": str(param_stream),
        "trace_kind": "artifact_import",
        "transport_admission": "unknown",
        "dut_hash": "",
        "hash_eq_target": "unknown",
        "nearest_counterexample_id": spec["nearest_counterexample_id"],
        "summary_path": relativize_to_repo(capture_dir / "SUMMARY.md"),
        "notes": "completion-demo structural import; direct outcome missing",
    }
    ledger = ledger_row(
        spec["capture_id"],
        spec["run_id"],
        FRONTIER,
        "Normalize historical Conv case without inventing missing DUT outcome",
        "",
        row["model_sha256"],
        "unknown",
        "unknown",
        "imported_structural",
        "trace_contract_stub",
        "summary_missing",
        relativize_to_repo(capture_dir / "SUMMARY.md"),
        spec["baseline_capture_id"],
        spec["nearest_counterexample_id"],
    )
    return row, ledger


def import_tail_case(spec: dict[str, str], capture_dir: Path, evidence_dir: Path):
    run_root = spec["source_path"].parents[1]
    summary_path = run_root / "SUMMARY.txt"
    summary = summary_path.read_text(encoding="utf-8").splitlines()
    case_shape = spec["source_path"].name
    summary_line = next((line for line in summary if line.startswith(f"{case_shape} ")), "")
    values = parse_tail_summary_line(summary_line)
    firmware_path = Path(next((line.split("=", 1)[1] for line in summary if line.startswith("firmware=")), ""))
    eo_patch = spec["source_path"] / "eo.patchspec"
    param_verify = spec["source_path"] / "param_verify.log"
    dut_native = run_root / "dut" / f"{case_shape}_native_completion.log"
    dut_baseline = run_root / "dut" / f"{case_shape}_target_baseline.log"
    for path in (summary_path, eo_patch, param_verify, dut_native, dut_baseline):
        copy_text_evidence(path, evidence_dir)

    param_verify_values = read_key_value_file(param_verify)
    metadata = {
        "capture_id": spec["capture_id"],
        "frontier": FRONTIER,
        "capture_version": 1,
        "source": {
            "root": str(spec["source_path"]),
            "run_id": spec["run_id"],
            "kind": "historical_phase7_tail_case",
            "imported_at_utc": now_utc(),
        },
        "provenance": {
            "repo_path": str(REPO_ROOT),
            "repo_commit": repo_commit(),
            "workspace_root": str(WORKSPACE_ROOT),
        },
        "firmware": {
            "path": str(firmware_path),
            "sha256": sha256_file(firmware_path),
            "provenance_note": "directly recorded in tail-proof summary",
        },
        "artifacts": {
            "model": artifact_ref(None),
            "metadata_json": artifact_ref(None),
            "serialized_executable": artifact_ref(None),
            "param_stream": artifact_ref(spec["source_path"] / "native_param.bin"),
            "eo_patchspec": artifact_ref(eo_patch),
            "dut_logs": [artifact_ref(dut_native), artifact_ref(dut_baseline)],
            "usbmon_logs": [],
        },
        "identity": {
            "family": "phase7_tail_proof",
            "regime": "p32",
            "shape": case_shape,
            "operator": "conv2d_k3_same_bias_off",
        },
        "outcome": {
            "transport_admission": "pass",
            "dut_hash": values.get("native_hash", ""),
            "hash_eq_target": values.get("hash_eq_target", "unknown"),
            "exact_failure_mode": "",
        },
        "comparison": {
            "baseline_capture_id": spec["baseline_capture_id"],
            "nearest_counterexample_id": spec["nearest_counterexample_id"],
            "comparison_class": spec["comparison_class"],
        },
        "decision": {
            "keep_discard": "keep",
            "reason": "direct tail recovery proof with summary and param parity",
        },
        "notes": {
            "param_verify": param_verify_values,
        },
    }
    write_json(capture_dir / "metadata.json", metadata)
    summary_lines = [
        f"# {spec['capture_id']}",
        "",
        f"- source: `{spec['source_path']}`",
        "- imported evidence: `SUMMARY.txt`, `eo.patchspec`, `param_verify.log`, DUT logs",
        f"- baseline hash: `{values.get('baseline_hash', '')}`",
        f"- native hash: `{values.get('native_hash', '')}`",
        f"- hash_eq_target: `{values.get('hash_eq_target', 'unknown')}`",
        f"- nearest counterexample: `{spec['nearest_counterexample_id']}`",
    ]
    (capture_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    row = {
        "capture_id": spec["capture_id"],
        "source_root": str(spec["source_path"]),
        "source_run_id": spec["run_id"],
        "source_kind": "historical_phase7_tail_case",
        "frontier": FRONTIER,
        "family": "phase7_tail_proof",
        "regime": "p32",
        "shape": case_shape,
        "artifact_date": parse_run_date(spec["run_id"]),
        "firmware_path": str(firmware_path),
        "firmware_sha256": sha256_file(firmware_path),
        "model_id": "",
        "model_sha256": "",
        "metadata_path": relativize_to_repo(capture_dir / "metadata.json"),
        "exec_id": "",
        "exec_path": "",
        "params_id": "native_param.bin",
        "params_path": str(spec["source_path"] / "native_param.bin"),
        "trace_kind": "artifact_import",
        "transport_admission": "pass",
        "dut_hash": values.get("native_hash", ""),
        "hash_eq_target": values.get("hash_eq_target", "unknown"),
        "nearest_counterexample_id": spec["nearest_counterexample_id"],
        "summary_path": relativize_to_repo(capture_dir / "SUMMARY.md"),
        "notes": "tail proof import with direct DUT outcome",
    }
    ledger = ledger_row(
        spec["capture_id"],
        spec["run_id"],
        FRONTIER,
        "Normalize direct tail recovery proof and preserve DUT outcome",
        row["firmware_sha256"],
        "",
        "true",
        row["hash_eq_target"],
        "imported_verified",
        "trace_contract_recovery",
        "",
        relativize_to_repo(capture_dir / "SUMMARY.md"),
        spec["baseline_capture_id"],
        spec["nearest_counterexample_id"],
    )
    return row, ledger


def import_dense_template(spec: dict[str, str], capture_dir: Path, evidence_dir: Path):
    metadata_path = first_match(spec["source_path"], "*.metadata.json")
    summary_path = spec["source_path"] / "PIPELINE_SUMMARY.txt"
    exec_json = spec["source_path"] / "exec_parse.json"
    extract_metadata = spec["source_path"] / "extract" / "metadata.json"
    for path in (metadata_path, summary_path, exec_json, extract_metadata):
        if path:
            copy_text_evidence(path, evidence_dir)
    metadata_json = read_json(metadata_path) if metadata_path else {}
    model_path = first_match(spec["source_path"], "*_quant.tflite")
    exec_path = spec["source_path"] / "extract" / "package_000" / "serialized_executable_000.bin"
    metadata = {
        "capture_id": spec["capture_id"],
        "frontier": FRONTIER,
        "capture_version": 1,
        "source": {
            "root": str(spec["source_path"]),
            "run_id": spec["run_id"],
            "kind": "historical_dense_template",
            "imported_at_utc": now_utc(),
        },
        "provenance": {
            "repo_path": str(REPO_ROOT),
            "repo_commit": repo_commit(),
            "workspace_root": str(WORKSPACE_ROOT),
        },
        "firmware": {
            "path": "",
            "sha256": "",
            "provenance_note": "template pipeline artifact has no colocated firmware record",
        },
        "artifacts": {
            "model": artifact_ref(model_path),
            "metadata_json": artifact_ref(metadata_path),
            "serialized_executable": artifact_ref(exec_path),
            "param_stream": artifact_ref(None),
            "eo_patchspec": artifact_ref(None),
            "dut_logs": [],
            "usbmon_logs": [],
        },
        "identity": {
            "family": "dense_template",
            "regime": "",
            "shape": infer_dense_shape(spec["run_id"]),
            "operator": "dense",
        },
        "outcome": {
            "transport_admission": "unknown",
            "dut_hash": "",
            "hash_eq_target": "unknown",
            "exact_failure_mode": "",
        },
        "comparison": {
            "baseline_capture_id": spec["baseline_capture_id"],
            "nearest_counterexample_id": spec["nearest_counterexample_id"],
            "comparison_class": spec["comparison_class"],
        },
        "decision": {
            "keep_discard": "keep",
            "reason": "historical dense template source for stock/native comparison",
        },
    }
    write_json(capture_dir / "metadata.json", metadata)
    summary_lines = [
        f"# {spec['capture_id']}",
        "",
        f"- source: `{spec['source_path']}`",
        "- imported evidence: `PIPELINE_SUMMARY.txt`, `*.metadata.json`, `exec_parse.json`, `extract/metadata.json`",
        f"- referenced model: `{model_path}`",
        f"- referenced executable: `{exec_path}`",
        "- outcome: historical template source; no DUT hash stored here",
        f"- nearest counterexample: `{spec['nearest_counterexample_id']}`",
    ]
    (capture_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    row = {
        "capture_id": spec["capture_id"],
        "source_root": str(spec["source_path"]),
        "source_run_id": spec["run_id"],
        "source_kind": "historical_dense_template",
        "frontier": FRONTIER,
        "family": "dense_template",
        "regime": "",
        "shape": infer_dense_shape(spec["run_id"]),
        "artifact_date": parse_run_date(spec["run_id"]),
        "firmware_path": "",
        "firmware_sha256": "",
        "model_id": model_path.name if model_path else "",
        "model_sha256": sha256_file(model_path),
        "metadata_path": relativize_to_repo(capture_dir / "metadata.json"),
        "exec_id": exec_path.name,
        "exec_path": str(exec_path),
        "params_id": "",
        "params_path": "",
        "trace_kind": "artifact_import",
        "transport_admission": "unknown",
        "dut_hash": "",
        "hash_eq_target": "unknown",
        "nearest_counterexample_id": spec["nearest_counterexample_id"],
        "summary_path": relativize_to_repo(capture_dir / "SUMMARY.md"),
        "notes": "dense template historical stock source",
    }
    ledger = ledger_row(
        spec["capture_id"],
        spec["run_id"],
        FRONTIER,
        "Normalize dense stock source artifact for later baseline comparison",
        "",
        row["model_sha256"],
        "unknown",
        "unknown",
        "imported_structural",
        "trace_contract_dense_stock",
        "",
        relativize_to_repo(capture_dir / "SUMMARY.md"),
        spec["baseline_capture_id"],
        spec["nearest_counterexample_id"],
    )
    return row, ledger


def import_dense_usbmon(spec: dict[str, str], capture_dir: Path, evidence_dir: Path):
    summary_path = first_match(spec["source_path"], "*.summary.txt")
    log_path = first_match(spec["source_path"], "*.log")
    if summary_path:
        copy_text_evidence(summary_path, evidence_dir)
    summary_values = read_key_value_file(summary_path) if summary_path else {}
    metadata = {
        "capture_id": spec["capture_id"],
        "frontier": FRONTIER,
        "capture_version": 1,
        "source": {
            "root": str(spec["source_path"]),
            "run_id": spec["run_id"],
            "kind": "historical_dense_usbmon",
            "imported_at_utc": now_utc(),
        },
        "provenance": {
            "repo_path": str(REPO_ROOT),
            "repo_commit": repo_commit(),
            "workspace_root": str(WORKSPACE_ROOT),
        },
        "firmware": {
            "path": summary_values.get("command", ""),
            "sha256": "",
            "provenance_note": "firmware appears in recorded command line, not as a separate hashed artifact",
        },
        "artifacts": {
            "model": artifact_ref(None),
            "metadata_json": artifact_ref(None),
            "serialized_executable": artifact_ref(None),
            "param_stream": artifact_ref(None),
            "eo_patchspec": artifact_ref(None),
            "dut_logs": [],
            "usbmon_logs": [artifact_ref(log_path)],
        },
        "identity": {
            "family": "dense_usbmon_snapshot",
            "regime": "",
            "shape": "",
            "operator": "dense",
        },
        "outcome": {
            "transport_admission": command_exit_label(summary_values),
            "dut_hash": "",
            "hash_eq_target": "unknown",
            "exact_failure_mode": command_exit_label(summary_values),
        },
        "comparison": {
            "baseline_capture_id": spec["baseline_capture_id"],
            "nearest_counterexample_id": spec["nearest_counterexample_id"],
            "comparison_class": spec["comparison_class"],
        },
        "decision": {
            "keep_discard": "keep",
            "reason": "historical usbmon scenario for baseline vs gated comparison",
        },
        "notes": {
            "usbmon_summary": summary_values,
        },
    }
    write_json(capture_dir / "metadata.json", metadata)
    summary_lines = [
        f"# {spec['capture_id']}",
        "",
        f"- source: `{spec['source_path']}`",
        f"- imported evidence: `{summary_path.name if summary_path else ''}`",
        f"- referenced usbmon log: `{log_path}`",
        f"- command_exit: `{summary_values.get('command_exit', 'unknown')}`",
        f"- nearest counterexample: `{spec['nearest_counterexample_id']}`",
    ]
    (capture_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    row = {
        "capture_id": spec["capture_id"],
        "source_root": str(spec["source_path"]),
        "source_run_id": spec["run_id"],
        "source_kind": "historical_dense_usbmon",
        "frontier": FRONTIER,
        "family": "dense_usbmon_snapshot",
        "regime": "",
        "shape": "",
        "artifact_date": parse_run_date(spec["run_id"]),
        "firmware_path": "",
        "firmware_sha256": "",
        "model_id": "",
        "model_sha256": "",
        "metadata_path": relativize_to_repo(capture_dir / "metadata.json"),
        "exec_id": "",
        "exec_path": "",
        "params_id": "",
        "params_path": "",
        "trace_kind": "artifact_import",
        "transport_admission": command_exit_label(summary_values),
        "dut_hash": "",
        "hash_eq_target": "unknown",
        "nearest_counterexample_id": spec["nearest_counterexample_id"],
        "summary_path": relativize_to_repo(capture_dir / "SUMMARY.md"),
        "notes": "dense usbmon scenario import",
    }
    ledger = ledger_row(
        spec["capture_id"],
        spec["run_id"],
        FRONTIER,
        "Normalize dense usbmon scenario while preserving baseline vs gated distinction",
        "",
        "",
        command_exit_label(summary_values),
        "unknown",
        "imported_structural",
        "trace_contract_dense_usbmon",
        command_exit_label(summary_values),
        relativize_to_repo(capture_dir / "SUMMARY.md"),
        spec["baseline_capture_id"],
        spec["nearest_counterexample_id"],
    )
    return row, ledger


def artifact_ref(path: Path | None) -> dict[str, str]:
    if path is None:
        return {"path": "", "sha256": "", "copied": "false"}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "copied": "false",
    }


def parse_tail_summary_line(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


def normalize_legacy_path(raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    marker = "coral-usb-oxidized/"
    text = raw.replace("\\", "/")
    if marker in text:
        suffix = text.split(marker, 1)[1]
        candidate = REPO_ROOT / suffix
        if candidate.exists():
            return candidate
    return path


def resolve_source_relative_path(raw: object, source_root: Path) -> Path | None:
    text = str(raw or "")
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path if path.exists() else normalize_legacy_path(text)
    candidate = source_root / path
    if candidate.exists():
        return candidate
    basename_candidate = source_root / path.name
    if basename_candidate.exists():
        return basename_candidate
    workspace_candidate = WORKSPACE_ROOT / path
    if workspace_candidate.exists():
        return workspace_candidate
    return path


def command_exit_label(summary_values: dict[str, str]) -> str:
    value = summary_values.get("command_exit", "")
    if not value:
        return "unknown"
    return f"command_exit_{value}"


def ledger_row(
    capture_id: str,
    run_id: str,
    frontier: str,
    hypothesis: str,
    firmware_hash: str,
    tflite_id: str,
    admission_ok: str,
    hash_eq_target: str,
    status: str,
    claim_delta: str,
    failure_mode: str,
    summary_path: str,
    baseline_capture_id: str,
    nearest_counterexample_id: str,
) -> dict[str, str]:
    return {
        "timestamp": now_utc(),
        "frontier": frontier,
        "experiment_id": capture_id,
        "hypothesis": hypothesis,
        "input_ids": run_id,
        "firmware_hash": firmware_hash,
        "tflite_id": tflite_id,
        "executable_id": "",
        "admission_ok": admission_ok,
        "hash_eq_target": hash_eq_target,
        "holdout_ok": "",
        "counterexample_ok": "",
        "status": status,
        "claim_delta": claim_delta,
        "notes": "",
        "capture_id": capture_id,
        "baseline_capture_id": baseline_capture_id,
        "counterexample_capture_id": nearest_counterexample_id,
        "failure_mode": failure_mode,
        "summary_path": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    sub.add_parser("import-sample")
    args = parser.parse_args()

    if args.command == "inventory":
        inventory_command()
    elif args.command == "import-sample":
        import_sample_command()


if __name__ == "__main__":
    main()
