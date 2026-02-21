#!/usr/bin/env python3
"""Patch EdgeTPU executable parameter blobs in compiled *_edgetpu.tflite files.

This tool targets the embedded DarwiNN Package (file identifier DWN1) and
modifies `Executable.parameters` vectors in-place while preserving file size and
FlatBuffer offsets.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DWN1 = b"DWN1"

EXECUTABLE_TYPE = {
    0: "STAND_ALONE",
    1: "PARAMETER_CACHING",
    2: "EXECUTION_ONLY",
}


class TensorizerError(Exception):
    """Raised for parse/validation/patch errors."""


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
        if abs_off > len(self.data):
            raise TensorizerError(f"field {field_id} offset {abs_off} out of bounds")
        return abs_off


@dataclass
class Region:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class ExecutableView:
    index: int
    type_value: int
    type_name: str
    abs_start: int
    abs_end: int
    sha256: str
    parameter_region: Optional[Region]


@dataclass
class PackageView:
    index: int
    root_offset: int
    multi_region: Region
    executables: List[ExecutableView]


def _u8(data: bytes, off: int) -> int:
    return data[off]


def _u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def _i16(data: bytes, off: int) -> int:
    return struct.unpack_from("<h", data, off)[0]


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
        raise TensorizerError(msg)


def _parse_root_table(data: bytes, root_offset: int = 0, file_identifier: Optional[bytes] = None) -> FlatTable:
    _require(root_offset >= 0, f"root offset {root_offset} negative")
    _require(root_offset + 4 <= len(data), f"root offset {root_offset} out of range")

    if file_identifier is not None:
        _require(root_offset + 8 <= len(data), "identifier check out of range")
        got = data[root_offset + 4 : root_offset + 8]
        _require(got == file_identifier, f"identifier mismatch at {root_offset + 4}: expected {file_identifier!r}, got {got!r}")

    table_rel = _u32(data, root_offset)
    table_offset = root_offset + table_rel
    _require(table_offset + 4 <= len(data), "table pointer out of range")

    vtable_rel = _i32(data, table_offset)
    _require(vtable_rel != 0, f"invalid vtable relative offset: {vtable_rel}")
    vtable_offset = table_offset - vtable_rel
    _require(vtable_offset >= 0, f"vtable underflow: {vtable_offset}")
    _require(vtable_offset + 4 <= len(data), "vtable header out of range")

    vtable_len = _u16(data, vtable_offset)
    object_len = _u16(data, vtable_offset + 2)
    _require(vtable_len >= 4 and (vtable_len % 2 == 0), f"invalid vtable length: {vtable_len}")
    _require(object_len >= 4, f"invalid object length: {object_len}")
    _require(vtable_offset + vtable_len <= len(data), "vtable overruns buffer")
    _require(table_offset + object_len <= len(data), "table overruns buffer")

    return FlatTable(
        data=data,
        table_offset=table_offset,
        vtable_offset=vtable_offset,
        vtable_len=vtable_len,
        object_len=object_len,
    )


def _read_offset_object(table: FlatTable, field_id: int) -> Optional[int]:
    off = table.field_offset(field_id)
    if off is None:
        return None
    rel = _u32(table.data, off)
    if rel == 0:
        return None
    target = off + rel
    _require(target + 4 <= len(table.data), f"offset field {field_id} out of range")
    return target


def _read_vector_region(table: FlatTable, field_id: int) -> Optional[Region]:
    target = _read_offset_object(table, field_id)
    if target is None:
        return None
    vlen = _u32(table.data, target)
    start = target + 4
    end = start + vlen
    _require(end <= len(table.data), f"vector field {field_id} out of range")
    return Region(start=start, end=end)


def _read_i16_field(table: FlatTable, field_id: int, default: int = 0) -> int:
    off = table.field_offset(field_id)
    if off is None:
        return default
    return _i16(table.data, off)


def _scan_dwn1_candidates(data: bytes) -> List[int]:
    out: List[int] = []
    seen = set()
    pos = 0
    while True:
        hit = data.find(DWN1, pos)
        if hit < 0:
            break
        if hit >= 4:
            root = hit - 4
            if root not in seen:
                seen.add(root)
                out.append(root)
        pos = hit + 1
    return out


def _parse_multi_executable_layout(multi_bytes: bytes) -> List[Region]:
    table = _parse_root_table(multi_bytes, 0, file_identifier=None)
    vec_target = _read_offset_object(table, 0)
    _require(vec_target is not None, "MultiExecutable.serialized_executables missing")

    length = _u32(multi_bytes, vec_target)
    vec_start = vec_target + 4
    vec_end = vec_start + (length * 4)
    _require(vec_end <= len(multi_bytes), "serialized_executables vector out of range")

    regions: List[Region] = []
    for i in range(length):
        slot = vec_start + (i * 4)
        rel = _u32(multi_bytes, slot)
        s_off = slot + rel
        _require(s_off + 4 <= len(multi_bytes), f"string[{i}] header out of range")
        s_len = _u32(multi_bytes, s_off)
        s_start = s_off + 4
        s_end = s_start + s_len
        _require(s_end <= len(multi_bytes), f"string[{i}] data out of range")
        regions.append(Region(start=s_start, end=s_end))
    return regions


def _inspect_packages(blob: bytes) -> List[PackageView]:
    packages: List[PackageView] = []
    candidates = _scan_dwn1_candidates(blob)

    for root_offset in candidates:
        try:
            pkg_table = _parse_root_table(blob, root_offset, file_identifier=DWN1)
            multi_region = _read_vector_region(pkg_table, 1)
            if multi_region is None:
                continue

            multi_bytes = blob[multi_region.start : multi_region.end]
            exe_regions_rel = _parse_multi_executable_layout(multi_bytes)

            executables: List[ExecutableView] = []
            for idx, reg in enumerate(exe_regions_rel):
                abs_start = multi_region.start + reg.start
                abs_end = multi_region.start + reg.end
                exe_blob = blob[abs_start:abs_end]

                exe_table = _parse_root_table(exe_blob, 0, file_identifier=None)
                tval = _read_i16_field(exe_table, 13, default=0)
                preg = _read_vector_region(exe_table, 6)
                abs_preg = None
                if preg is not None:
                    abs_preg = Region(start=abs_start + preg.start, end=abs_start + preg.end)

                executables.append(
                    ExecutableView(
                        index=idx,
                        type_value=tval,
                        type_name=EXECUTABLE_TYPE.get(tval, f"UNKNOWN_{tval}"),
                        abs_start=abs_start,
                        abs_end=abs_end,
                        sha256=_sha256_bytes(exe_blob),
                        parameter_region=abs_preg,
                    )
                )

            packages.append(
                PackageView(
                    index=len(packages),
                    root_offset=root_offset,
                    multi_region=multi_region,
                    executables=executables,
                )
            )
        except TensorizerError:
            continue

    return packages


def _filter_executables(
    package: PackageView,
    exec_indices: Optional[Sequence[int]],
    exec_type: str,
    require_parameters: bool,
) -> List[ExecutableView]:
    selected = list(package.executables)

    if exec_indices is not None and len(exec_indices) > 0:
        allowed = set(exec_indices)
        selected = [e for e in selected if e.index in allowed]

    if exec_type == "parameter_caching":
        selected = [e for e in selected if e.type_name == "PARAMETER_CACHING"]
    elif exec_type == "execution_only":
        selected = [e for e in selected if e.type_name == "EXECUTION_ONLY"]
    elif exec_type == "stand_alone":
        selected = [e for e in selected if e.type_name == "STAND_ALONE"]

    if require_parameters:
        selected = [e for e in selected if e.parameter_region is not None and e.parameter_region.size > 0]

    return selected


def _make_payload(
    mode: str,
    size: int,
    source: bytes,
    *,
    byte_value: int,
    seed: int,
) -> bytes:
    if mode == "zero":
        return bytes(size)
    if mode == "byte":
        return bytes([byte_value & 0xFF]) * size
    if mode == "ramp":
        return bytes((i & 0xFF) for i in range(size))
    if mode == "xor":
        mask = byte_value & 0xFF
        return bytes((b ^ mask) for b in source)
    if mode == "random":
        rng = random.Random(seed)
        return bytes(rng.randrange(0, 256) for _ in range(size))
    raise TensorizerError(f"unknown patch mode: {mode}")


def _render_inspect_text(path: Path, packages: List[PackageView]) -> str:
    lines = [
        f"input={path}",
        f"size={path.stat().st_size}",
        f"packages={len(packages)}",
    ]
    for p in packages:
        lines.append(
            f"package[{p.index}] root_offset={p.root_offset} "
            f"multi_region=[{p.multi_region.start},{p.multi_region.end}) size={p.multi_region.size} "
            f"executables={len(p.executables)}"
        )
        for e in p.executables:
            pr = e.parameter_region
            if pr is None:
                pinfo = "parameters=none"
            else:
                pinfo = f"parameters=[{pr.start},{pr.end}) size={pr.size}"
            lines.append(
                f"  exe[{e.index}] type={e.type_name} "
                f"region=[{e.abs_start},{e.abs_end}) size={e.abs_end - e.abs_start} "
                f"{pinfo}"
            )
    return "\n".join(lines)


def _inspect_json(path: Path, packages: List[PackageView]) -> Dict[str, object]:
    return {
        "tool": "tensorizer_patch_edgetpu.py",
        "generated_at_utc": _iso_utc_now(),
        "input": {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256_bytes(path.read_bytes()),
        },
        "packages": [
            {
                "index": p.index,
                "root_offset": p.root_offset,
                "multi_region": {
                    "start": p.multi_region.start,
                    "end": p.multi_region.end,
                    "size": p.multi_region.size,
                },
                "executables": [
                    {
                        "index": e.index,
                        "type_value": e.type_value,
                        "type_name": e.type_name,
                        "region": {
                            "start": e.abs_start,
                            "end": e.abs_end,
                            "size": e.abs_end - e.abs_start,
                        },
                        "sha256": e.sha256,
                        "parameter_region": None
                        if e.parameter_region is None
                        else {
                            "start": e.parameter_region.start,
                            "end": e.parameter_region.end,
                            "size": e.parameter_region.size,
                        },
                    }
                    for e in p.executables
                ],
            }
            for p in packages
        ],
    }


def _run_inspect(args: argparse.Namespace) -> int:
    blob = args.input.read_bytes()
    packages = _inspect_packages(blob)
    if not packages:
        raise TensorizerError("no valid DWN1 packages found")

    if args.json:
        print(json.dumps(_inspect_json(args.input, packages), indent=2, sort_keys=False))
    else:
        print(_render_inspect_text(args.input, packages))
    return 0


def _run_patch(args: argparse.Namespace) -> int:
    _require(args.input.exists() and args.input.is_file(), f"input not found: {args.input}")
    _require(args.output is not None, "--output is required for patch command")
    if args.output.exists() and not args.overwrite:
        raise TensorizerError(f"output exists: {args.output} (pass --overwrite)")

    blob = bytearray(args.input.read_bytes())
    packages = _inspect_packages(bytes(blob))
    if not packages:
        raise TensorizerError("no valid DWN1 packages found")

    _require(0 <= args.package_index < len(packages), f"package-index out of range: {args.package_index}")
    package = packages[args.package_index]
    selected = _filter_executables(
        package=package,
        exec_indices=args.exec_index,
        exec_type=args.exec_type,
        require_parameters=True,
    )
    if not selected:
        raise TensorizerError("no executables matched selection with non-empty parameter regions")

    patch_log: List[Dict[str, object]] = []
    for exe in selected:
        preg = exe.parameter_region
        _require(preg is not None, "internal: missing parameter region")
        old_bytes = bytes(blob[preg.start : preg.end])
        new_bytes = _make_payload(
            mode=args.mode,
            size=preg.size,
            source=old_bytes,
            byte_value=args.byte_value,
            seed=args.seed + exe.index,
        )
        _require(len(new_bytes) == preg.size, "internal: patch payload length mismatch")

        blob[preg.start : preg.end] = new_bytes
        patch_log.append(
            {
                "executable_index": exe.index,
                "type_name": exe.type_name,
                "parameter_region": {"start": preg.start, "end": preg.end, "size": preg.size},
                "old_sha256": _sha256_bytes(old_bytes),
                "new_sha256": _sha256_bytes(new_bytes),
                "mode": args.mode,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(blob))

    out_meta = {
        "tool": "tensorizer_patch_edgetpu.py",
        "generated_at_utc": _iso_utc_now(),
        "input_path": str(args.input),
        "input_sha256": _sha256_bytes(args.input.read_bytes()),
        "output_path": str(args.output),
        "output_sha256": _sha256_bytes(args.output.read_bytes()),
        "package_index": args.package_index,
        "exec_type_filter": args.exec_type,
        "exec_index_filter": args.exec_index or [],
        "mode": args.mode,
        "byte_value": args.byte_value,
        "seed": args.seed,
        "patched_executables": patch_log,
    }

    if args.metadata_out is not None:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(out_meta, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(out_meta, indent=2, sort_keys=False))
    else:
        print(f"Patched file written: {args.output}")
        print(f"Patched executables: {len(patch_log)}")
        for item in patch_log:
            pr = item["parameter_region"]
            print(
                "  exe[{i}] type={t} params=[{s},{e}) size={n} old={o} new={nn}".format(
                    i=item["executable_index"],
                    t=item["type_name"],
                    s=pr["start"],
                    e=pr["end"],
                    n=pr["size"],
                    o=item["old_sha256"][:12],
                    nn=item["new_sha256"][:12],
                )
            )
        if args.metadata_out is not None:
            print(f"Patch metadata: {args.metadata_out}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorizer_patch_edgetpu.py",
        description="Inspect and patch EdgeTPU executable parameters inside compiled *_edgetpu.tflite models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    insp = sub.add_parser("inspect", help="Inspect embedded DWN1 package and executable parameter regions.")
    insp.add_argument("input", type=Path, help="Path to compiled *_edgetpu.tflite file.")
    insp.add_argument("--json", action="store_true", help="Emit JSON output.")

    pat = sub.add_parser("patch", help="Patch executable parameter bytes in-place and write a new model file.")
    pat.add_argument("input", type=Path, help="Path to compiled *_edgetpu.tflite file.")
    pat.add_argument("--output", "-o", type=Path, required=True, help="Output patched model path.")
    pat.add_argument("--overwrite", action="store_true", help="Allow overwriting output path.")
    pat.add_argument("--package-index", type=int, default=0, help="Package index (from inspect output).")
    pat.add_argument(
        "--exec-type",
        choices=["all", "parameter_caching", "execution_only", "stand_alone"],
        default="parameter_caching",
        help="Executable type filter.",
    )
    pat.add_argument(
        "--exec-index",
        type=int,
        action="append",
        help="Executable index filter (repeatable). If omitted, all matching execs are patched.",
    )
    pat.add_argument(
        "--mode",
        choices=["zero", "byte", "ramp", "xor", "random"],
        default="zero",
        help="Patch mode for parameter bytes.",
    )
    pat.add_argument("--byte-value", type=int, default=0, help="Byte value for mode=byte/xor (0-255).")
    pat.add_argument("--seed", type=int, default=1337, help="Seed for mode=random.")
    pat.add_argument("--metadata-out", type=Path, help="Optional patch metadata JSON path.")
    pat.add_argument("--json", action="store_true", help="Emit patch metadata as JSON on stdout.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "patch":
            return _run_patch(args)
        parser.error(f"unknown command: {args.command}")
    except TensorizerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"io error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
