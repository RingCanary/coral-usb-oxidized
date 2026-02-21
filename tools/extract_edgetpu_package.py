#!/usr/bin/env python3
"""Offline extractor for EdgeTPU DWN1 packages embedded in *_edgetpu.tflite files.

The tool scans an input binary for FlatBuffer roots with file identifier "DWN1",
validates Package/MultiExecutable table sanity, extracts key package fields, and
writes serialized executable blobs plus metadata JSON.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DWN1 = b"DWN1"
DEFAULT_SELF_TEST_INPUT = Path("/tmp/mobilenet_v1_1.0_224_quant_edgetpu.tflite")
DEFAULT_SELF_TEST_OUT = Path("/tmp/edgetpu_extract_selftest")


class ExtractorError(Exception):
    """Raised for parse/validation/extraction failures."""


@dataclass
class FlatTable:
    data: bytes
    table_offset: int
    vtable_offset: int
    vtable_len: int
    object_len: int

    def field_offset(self, field_id: int) -> Optional[int]:
        entry = self.vtable_offset + 4 + (field_id * 2)
        if entry + 2 > self.vtable_offset + self.vtable_len:
            return None
        rel = _u16(self.data, entry)
        if rel == 0:
            return None
        abs_off = self.table_offset + rel
        if abs_off + 4 > len(self.data):
            raise ExtractorError(
                f"field {field_id} offset {abs_off} is out of bounds for table"
            )
        return abs_off


@dataclass
class ParsedPackage:
    root_offset: int
    identifier_offset: int
    min_runtime_version: int
    keypair_version: int
    compiler_version: str
    virtual_chip_id: int
    model_identifier: Optional[str]
    signature_size: int
    serialized_multi_executable: bytes
    serialized_executables: List[bytes]
    multi_chip_package_count: int


def _u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _i32(data: bytes, off: int) -> int:
    return struct.unpack_from("<i", data, off)[0]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iso_utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ExtractorError(msg)


def _parse_root_table(data: bytes, root_offset: int, file_identifier: Optional[bytes]) -> FlatTable:
    _require(root_offset >= 0, f"root offset {root_offset} is negative")
    _require(root_offset + 4 <= len(data), f"root offset {root_offset} is out of range")

    if file_identifier is not None:
        _require(
            root_offset + 8 <= len(data),
            f"identifier check at root {root_offset} exceeds buffer",
        )
        ident = data[root_offset + 4 : root_offset + 8]
        _require(
            ident == file_identifier,
            f"identifier mismatch at {root_offset + 4}: expected {file_identifier!r}, got {ident!r}",
        )

    table_rel = _u32(data, root_offset)
    table_offset = root_offset + table_rel
    _require(table_offset + 4 <= len(data), "table pointer is out of range")

    vtable_rel = _i32(data, table_offset)
    _require(vtable_rel > 0, f"invalid vtable relative offset: {vtable_rel}")

    vtable_offset = table_offset - vtable_rel
    _require(vtable_offset >= 0, f"vtable offset underflow: {vtable_offset}")
    _require(vtable_offset + 4 <= len(data), "vtable header out of range")

    vtable_len = _u16(data, vtable_offset)
    object_len = _u16(data, vtable_offset + 2)
    _require(vtable_len >= 4 and (vtable_len % 2 == 0), f"invalid vtable length: {vtable_len}")
    _require(object_len >= 4, f"invalid object length: {object_len}")
    _require(vtable_offset + vtable_len <= len(data), "vtable overruns buffer")
    _require(table_offset + object_len <= len(data), "table object overruns buffer")

    return FlatTable(
        data=data,
        table_offset=table_offset,
        vtable_offset=vtable_offset,
        vtable_len=vtable_len,
        object_len=object_len,
    )


def _read_i32_field(table: FlatTable, field_id: int, *, default: Optional[int] = None) -> Optional[int]:
    off = table.field_offset(field_id)
    if off is None:
        return default
    return _i32(table.data, off)


def _read_offset_object(table: FlatTable, field_id: int) -> Optional[int]:
    off = table.field_offset(field_id)
    if off is None:
        return None
    rel = _u32(table.data, off)
    target = off + rel
    _require(target + 4 <= len(table.data), f"offset field {field_id} points outside buffer")
    return target


def _read_string_field(
    table: FlatTable, field_id: int, *, default: Optional[str] = None
) -> Optional[str]:
    target = _read_offset_object(table, field_id)
    if target is None:
        return default
    slen = _u32(table.data, target)
    sstart = target + 4
    send = sstart + slen
    _require(send <= len(table.data), f"string field {field_id} overruns buffer")
    raw = table.data[sstart:send]
    return raw.decode("utf-8", errors="replace")


def _read_vector_bytes_field(table: FlatTable, field_id: int) -> Optional[bytes]:
    target = _read_offset_object(table, field_id)
    if target is None:
        return None
    vlen = _u32(table.data, target)
    vstart = target + 4
    vend = vstart + vlen
    _require(vend <= len(table.data), f"vector field {field_id} overruns buffer")
    return bytes(table.data[vstart:vend])


def _read_vector_length(table: FlatTable, field_id: int, *, default: int = 0) -> int:
    target = _read_offset_object(table, field_id)
    if target is None:
        return default
    return _u32(table.data, target)


def _read_vector_of_strings(table: FlatTable, field_id: int) -> List[bytes]:
    target = _read_offset_object(table, field_id)
    _require(target is not None, f"vector<string> field {field_id} is missing")

    length = _u32(table.data, target)
    vec_start = target + 4
    vec_end = vec_start + (length * 4)
    _require(vec_end <= len(table.data), "vector<string> entries exceed buffer")

    out: List[bytes] = []
    for i in range(length):
        elem_slot = vec_start + (i * 4)
        elem_rel = _u32(table.data, elem_slot)
        elem_off = elem_slot + elem_rel
        _require(elem_off + 4 <= len(table.data), f"string[{i}] header out of range")

        slen = _u32(table.data, elem_off)
        sstart = elem_off + 4
        send = sstart + slen
        _require(send <= len(table.data), f"string[{i}] data out of range")
        out.append(bytes(table.data[sstart:send]))

    return out


def _parse_multi_executable(data: bytes) -> List[bytes]:
    _require(len(data) >= 8, "serialized_multi_executable is too small")
    table = _parse_root_table(data, 0, file_identifier=None)
    executables = _read_vector_of_strings(table, 0)
    _require(executables, "serialized_multi_executable has no serialized executables")
    return executables


def _parse_package_at(data: bytes, root_offset: int) -> ParsedPackage:
    package_table = _parse_root_table(data, root_offset, file_identifier=DWN1)

    min_runtime_version = _read_i32_field(package_table, 0)
    keypair_version = _read_i32_field(package_table, 3)
    compiler_version = _read_string_field(package_table, 4)
    _require(
        min_runtime_version is not None,
        "Package.min_runtime_version is missing",
    )
    _require(
        keypair_version is not None,
        "Package.keypair_version is missing",
    )
    _require(
        compiler_version is not None,
        "Package.compiler_version is missing",
    )
    virtual_chip_id = _read_i32_field(package_table, 5, default=0)

    serialized_multi_executable = _read_vector_bytes_field(package_table, 1)
    _require(
        serialized_multi_executable is not None,
        "Package.serialized_multi_executable is missing",
    )
    serialized_executables = _parse_multi_executable(serialized_multi_executable)

    signature = _read_vector_bytes_field(package_table, 2) or b""
    model_identifier = _read_string_field(package_table, 7)
    multi_chip_package_count = _read_vector_length(package_table, 6, default=0)

    return ParsedPackage(
        root_offset=root_offset,
        identifier_offset=root_offset + 4,
        min_runtime_version=min_runtime_version,
        keypair_version=keypair_version,
        compiler_version=compiler_version,
        virtual_chip_id=virtual_chip_id,
        model_identifier=model_identifier,
        signature_size=len(signature),
        serialized_multi_executable=serialized_multi_executable,
        serialized_executables=serialized_executables,
        multi_chip_package_count=multi_chip_package_count,
    )


def _scan_dwn1_candidates(data: bytes) -> List[int]:
    offsets: List[int] = []
    seen = set()
    start = 0
    while True:
        hit = data.find(DWN1, start)
        if hit < 0:
            break
        if hit >= 4:
            candidate = hit - 4
            if candidate not in seen:
                seen.add(candidate)
                offsets.append(candidate)
        start = hit + 1
    return offsets


def _safe_prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and overwrite:
        resolved = out_dir.resolve()
        _require(resolved != Path("/"), "refusing to remove root directory")
        shutil.rmtree(out_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    if any(out_dir.iterdir()):
        raise ExtractorError(
            f"output directory '{out_dir}' is not empty (use --overwrite to replace it)"
        )


def extract_file(input_path: Path, out_dir: Path, overwrite: bool = False) -> Dict[str, Any]:
    if not input_path.exists():
        raise ExtractorError(f"input file not found: {input_path}")
    if not input_path.is_file():
        raise ExtractorError(f"input path is not a file: {input_path}")

    blob = input_path.read_bytes()
    candidates = _scan_dwn1_candidates(blob)

    packages: List[ParsedPackage] = []
    rejected: List[Dict[str, Any]] = []
    for root_offset in candidates:
        try:
            packages.append(_parse_package_at(blob, root_offset))
        except ExtractorError as exc:
            rejected.append({"root_offset": root_offset, "error": str(exc)})

    if not candidates:
        raise ExtractorError("no DWN1 identifiers found")
    if not packages:
        details = "; ".join(
            f"{item['root_offset']}: {item['error']}" for item in rejected[:3]
        )
        raise ExtractorError(f"found {len(candidates)} DWN1 candidates but none were valid: {details}")

    _safe_prepare_output_dir(out_dir, overwrite=overwrite)

    metadata_packages: List[Dict[str, Any]] = []
    for pkg_index, package in enumerate(packages):
        pkg_dir = out_dir / f"package_{pkg_index:03d}"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        multi_path = pkg_dir / "serialized_multi_executable.bin"
        multi_path.write_bytes(package.serialized_multi_executable)

        executable_entries = []
        for exe_index, exe_blob in enumerate(package.serialized_executables):
            exe_name = f"serialized_executable_{exe_index:03d}.bin"
            exe_path = pkg_dir / exe_name
            exe_path.write_bytes(exe_blob)
            executable_entries.append(
                {
                    "index": exe_index,
                    "size": len(exe_blob),
                    "sha256": _sha256_bytes(exe_blob),
                    "path": str(exe_path.relative_to(out_dir)),
                }
            )

        metadata_packages.append(
            {
                "index": pkg_index,
                "root_offset": package.root_offset,
                "identifier_offset": package.identifier_offset,
                "min_runtime_version": package.min_runtime_version,
                "keypair_version": package.keypair_version,
                "compiler_version": package.compiler_version,
                "virtual_chip_id": package.virtual_chip_id,
                "model_identifier": package.model_identifier,
                "signature_size": package.signature_size,
                "multi_chip_package_count": package.multi_chip_package_count,
                "serialized_multi_executable_size": len(package.serialized_multi_executable),
                "serialized_multi_executable_sha256": _sha256_bytes(
                    package.serialized_multi_executable
                ),
                "serialized_multi_executable_path": str(multi_path.relative_to(out_dir)),
                "serialized_executables": executable_entries,
            }
        )

    metadata = {
        "tool": "extract_edgetpu_package.py",
        "generated_at_utc": _iso_utc_now(),
        "input": {
            "path": str(input_path),
            "size": len(blob),
            "sha256": _sha256_bytes(blob),
        },
        "scan": {
            "dwn1_candidate_root_offsets": candidates,
            "candidate_count": len(candidates),
            "valid_package_count": len(metadata_packages),
            "rejected_candidates": rejected,
        },
        "packages": metadata_packages,
    }

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    return {
        "metadata_path": metadata_path,
        "candidate_count": len(candidates),
        "valid_package_count": len(metadata_packages),
        "packages": metadata_packages,
        "rejected_candidates": rejected,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_edgetpu_package.py",
        description=(
            "Extract EdgeTPU DWN1 package data embedded in *_edgetpu.tflite files, "
            "including Package metadata and serialized executable blobs."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="scan one file and extract valid DWN1 package content",
    )
    extract_parser.add_argument("input", type=Path, help="input .tflite (or binary) path")
    extract_parser.add_argument(
        "--out",
        "-o",
        type=Path,
        required=True,
        help="output directory for extracted blobs + metadata.json",
    )
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove and recreate the output directory if it already exists",
    )

    self_test_parser = subparsers.add_parser(
        "self-test",
        help="quick extraction test against /tmp mobilenet file when available",
    )
    self_test_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SELF_TEST_INPUT,
        help=f"self-test input file (default: {DEFAULT_SELF_TEST_INPUT})",
    )
    self_test_parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SELF_TEST_OUT,
        help=f"self-test output directory (default: {DEFAULT_SELF_TEST_OUT})",
    )
    self_test_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove and recreate self-test output directory if it already exists",
    )
    self_test_parser.add_argument(
        "--require-input",
        action="store_true",
        help="fail if the default self-test file is missing instead of skipping",
    )

    return parser


def _run_extract(args: argparse.Namespace) -> int:
    result = extract_file(args.input, args.out, overwrite=args.overwrite)

    print(
        "Extraction complete: "
        f"{result['valid_package_count']} valid package(s) "
        f"from {result['candidate_count']} DWN1 candidate(s)."
    )
    print(f"Metadata: {result['metadata_path']}")

    for pkg in result["packages"]:
        print(
            "  package_{:03d}: min_runtime_version={} keypair_version={} "
            "compiler_version={} virtual_chip_id={} executables={}".format(
                pkg["index"],
                pkg["min_runtime_version"],
                pkg["keypair_version"],
                pkg["compiler_version"],
                pkg["virtual_chip_id"],
                len(pkg["serialized_executables"]),
            )
        )

    return 0


def _run_self_test(args: argparse.Namespace) -> int:
    if not args.input.exists():
        msg = f"Self-test skipped: input file not found: {args.input}"
        if args.require_input:
            raise ExtractorError(msg)
        print(msg)
        return 0

    print(f"Running self-test on: {args.input}")
    result = extract_file(args.input, args.out, overwrite=args.overwrite)

    print(
        "Self-test passed: "
        f"{result['valid_package_count']} valid package(s), "
        f"metadata at {result['metadata_path']}"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "extract":
            return _run_extract(args)
        if args.command == "self-test":
            return _run_self_test(args)
        parser.error(f"unknown command: {args.command}")
    except ExtractorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"io error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
