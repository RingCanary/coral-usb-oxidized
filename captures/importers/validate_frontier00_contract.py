#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURES_ROOT = REPO_ROOT / "captures"
DOCS_ROOT = REPO_ROOT / "docs"

INDEX_PATH = CAPTURES_ROOT / "index.tsv"
LEDGER_PATH = CAPTURES_ROOT / "contract_ledger.tsv"
CLAIM_MAP_PATH = CAPTURES_ROOT / "frontier00_claim_map.tsv"
CLAIM_MAP_COLUMNS_PATH = CAPTURES_ROOT / "schema" / "frontier00_claim_map.columns.tsv"
STUB_PATH = DOCS_ROOT / "frontier00_trace_contract_stub_2026-03-16.md"

EXPECTED_CAPTURE_COUNT = 6
ALLOWED_TRANSPORT = {"pass", "unknown", "command_exit_1"}
ALLOWED_ADMISSION = {"true", "unknown", "command_exit_1"}
ALLOWED_FAILURE = {"", "summary_missing", "command_exit_1"}
EXPECTED_ADMISSION_MAP = {
    "pass": "true",
    "unknown": "unknown",
    "command_exit_1": "command_exit_1",
}
REQUIRED_EXCLUDED_SOURCES = {
    "phase7-p32-tail-dut-proof-20260316T145105Z/cases/h12_w184",
    "phase7-p32-tail-dut-proof-20260316T145105Z/cases/h12_w192",
}
REQUIRED_SELECTION_RULE = "require_same_shape_counterexample_in_corpus"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    index_rows = read_tsv(INDEX_PATH)
    ledger_rows = read_tsv(LEDGER_PATH)
    claim_rows = read_tsv(CLAIM_MAP_PATH)

    expected_claim_columns = CLAIM_MAP_COLUMNS_PATH.read_text(encoding="utf-8").splitlines()
    expect(claim_rows, "claim map must contain at least one row")
    expect(list(claim_rows[0].keys()) == expected_claim_columns, "claim map columns do not match schema")

    expect(len(index_rows) == EXPECTED_CAPTURE_COUNT, f"expected {EXPECTED_CAPTURE_COUNT} index rows")
    expect(len(ledger_rows) == EXPECTED_CAPTURE_COUNT, f"expected {EXPECTED_CAPTURE_COUNT} ledger rows")

    index_by_id = {row["capture_id"]: row for row in index_rows}
    ledger_by_id = {row["capture_id"]: row for row in ledger_rows}

    expect(len(index_by_id) == EXPECTED_CAPTURE_COUNT, "index capture_id values must be unique")
    expect(len(ledger_by_id) == EXPECTED_CAPTURE_COUNT, "ledger capture_id values must be unique")
    expect(set(index_by_id) == set(ledger_by_id), "index and ledger capture_id sets must match")

    for capture_id, index_row in index_by_id.items():
        ledger_row = ledger_by_id[capture_id]
        expect(ledger_row["experiment_id"] == capture_id, f"{capture_id}: experiment_id must equal capture_id")
        expect(index_row["transport_admission"] in ALLOWED_TRANSPORT, f"{capture_id}: unexpected transport_admission")
        expect(ledger_row["admission_ok"] in ALLOWED_ADMISSION, f"{capture_id}: unexpected admission_ok")
        expect(ledger_row["failure_mode"] in ALLOWED_FAILURE, f"{capture_id}: unexpected failure_mode")
        expect(
            EXPECTED_ADMISSION_MAP[index_row["transport_admission"]] == ledger_row["admission_ok"],
            f"{capture_id}: transport/admission mapping mismatch",
        )
        expect(
            ledger_row["firmware_hash"] == index_row["firmware_sha256"],
            f"{capture_id}: ledger firmware_hash must match index firmware_sha256",
        )

    keep_claim_ids: list[str] = []
    excluded_sources: set[str] = set()
    for row in claim_rows:
        expect(row["frontier"] == "00_trace_contract", f"{row['claim_id']}: unexpected frontier")
        expect(row["decision"] in {"keep", "discard"}, f"{row['claim_id']}: unexpected decision")
        expect(row["selection_rule"], f"{row['claim_id']}: selection_rule is required")
        capture_ref = row["capture_ref"]
        if row["decision"] == "keep":
            keep_claim_ids.append(row["claim_id"])
            expect(capture_ref in index_by_id, f"{row['claim_id']}: capture_ref missing from index")
            expect(capture_ref in ledger_by_id, f"{row['claim_id']}: capture_ref missing from ledger")
            expect(row["nearest_counterexample_ref"] in index_by_id, f"{row['claim_id']}: nearest counterexample missing")
            index_row = index_by_id[capture_ref]
            ledger_row = ledger_by_id[capture_ref]
            expect(row["transport_outcome"] == index_row["transport_admission"], f"{row['claim_id']}: transport mismatch")
            expect(row["firmware_hash"] == ledger_row["firmware_hash"], f"{row['claim_id']}: firmware mismatch")
            if row["claim_class"] == "verified_success":
                expect(index_row["transport_admission"] == "pass", f"{row['claim_id']}: verified success requires pass")
                expect(ledger_row["admission_ok"] == "true", f"{row['claim_id']}: verified success requires admission_ok=true")
                expect(ledger_row["hash_eq_target"] == "true", f"{row['claim_id']}: verified success requires hash_eq_target=true")
                expect(index_row["dut_hash"], f"{row['claim_id']}: verified success requires dut_hash")
                expect(row["selection_rule"] == REQUIRED_SELECTION_RULE, f"{row['claim_id']}: verified success uses wrong rule")
        else:
            expect(not capture_ref, f"{row['claim_id']}: discard rows must not point at imported captures")
            expect(row["source_input_ids"] in REQUIRED_EXCLUDED_SOURCES, f"{row['claim_id']}: unexpected discard source")
            expect(row["selection_rule"] == REQUIRED_SELECTION_RULE, f"{row['claim_id']}: discard rows use wrong rule")
            expect(
                row["nearest_counterexample_result"] == "missing_same_shape_structural_counterexample",
                f"{row['claim_id']}: discard rows must record missing same-shape counterexample",
            )
            excluded_sources.add(row["source_input_ids"])

    stub_text = STUB_PATH.read_text(encoding="utf-8")
    for claim_id in keep_claim_ids:
        expect(claim_id in stub_text, f"stub note missing claim id {claim_id}")
    expect(REQUIRED_EXCLUDED_SOURCES == excluded_sources, "claim map must explicitly exclude the two omitted tail siblings")


if __name__ == "__main__":
    main()
