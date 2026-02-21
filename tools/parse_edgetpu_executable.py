#!/usr/bin/env python3
"""Parse EdgeTPU serialized executable FlatBuffers into RE-friendly summaries.

Targets blobs extracted by:
  tools/extract_edgetpu_package.py extract <model> --out <dir>

Primary focus:
- instruction bitstream chunks
- relocation metadata (field_offsets + Meta)
- parameter payload sizing
- input/output layer metadata
- dma_hints structure
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ParseError(Exception):
    """Raised for parsing failures."""


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
            raise ParseError(f"field {field_id} offset {abs_off} out of bounds")
        return abs_off


DESCRIPTION = {
    0: "BASE_ADDRESS_OUTPUT_ACTIVATION",
    1: "BASE_ADDRESS_INPUT_ACTIVATION",
    2: "BASE_ADDRESS_PARAMETER",
    3: "BASE_ADDRESS_SCRATCH",
}

POSITION = {
    0: "LOWER_32BIT",
    1: "UPPER_32BIT",
}

EXECUTABLE_TYPE = {
    0: "STAND_ALONE",
    1: "PARAMETER_CACHING",
    2: "EXECUTION_ONLY",
}

DATA_TYPE = {
    0: "FIXED_POINT8",
    1: "FIXED_POINT16",
    2: "SIGNED_FIXED_POINT32",
    3: "BFLOAT",
    4: "HALF",
    5: "SINGLE",
    8: "SIGNED_FIXED_POINT8",
    9: "SIGNED_FIXED_POINT16",
}

ANY_LAYER = {
    0: "NONE",
    1: "OutputLayer",
    2: "InputLayer",
}

DMA_DIRECTION = {
    0: "INFEED",
    1: "OUTFEED",
}

ANY_HINT = {
    0: "NONE",
    1: "DmaDescriptorHint",
    2: "InstructionHint",
    3: "InterruptHint",
    4: "FenceHint",
}

INTERRUPT_TYPE = {
    0: "SCALAR_CORE_INT_0",
    1: "SCALAR_CORE_INT_1",
    2: "SCALAR_CORE_INT_2",
    3: "SCALAR_CORE_INT_3",
}

DEFAULT_MARKERS = {
    "cmd_20720300": "20720300",
    "cmd_004c0200_01000000": "004c020001000000",
    "hdr_800f0080dc000000": "800f0080dc000000",
    "hdr_800f00f409000000": "800f00f409000000",
    "cmd_50270000": "50270000",
    "hdr_800f00cc09000000": "800f00cc09000000",
}


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


def _u64(data: bytes, off: int) -> int:
    return struct.unpack_from("<Q", data, off)[0]


def _i64(data: bytes, off: int) -> int:
    return struct.unpack_from("<q", data, off)[0]


def _f32(data: bytes, off: int) -> float:
    return struct.unpack_from("<f", data, off)[0]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex_preview(data: bytes, n: int = 16) -> str:
    return data[:n].hex()


def _iso_utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ParseError(msg)


def _parse_root_table(data: bytes, root_offset: int = 0, file_identifier: Optional[bytes] = None) -> FlatTable:
    _require(root_offset >= 0, f"root offset {root_offset} is negative")
    _require(root_offset + 4 <= len(data), f"root offset {root_offset} out of range")

    if file_identifier is not None:
        _require(root_offset + 8 <= len(data), "identifier check out of range")
        got = data[root_offset + 4 : root_offset + 8]
        _require(got == file_identifier, f"identifier mismatch: expected {file_identifier!r}, got {got!r}")

    table_rel = _u32(data, root_offset)
    table_offset = root_offset + table_rel
    _require(table_offset + 4 <= len(data), "table pointer out of range")

    vtable_rel = _i32(data, table_offset)
    _require(vtable_rel != 0, f"invalid vtable relative offset: {vtable_rel}")
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


def _parse_table_at(data: bytes, table_offset: int) -> FlatTable:
    _require(table_offset + 4 <= len(data), "table out of range")
    vtable_rel = _i32(data, table_offset)
    _require(vtable_rel != 0, f"invalid nested vtable relative offset: {vtable_rel}")
    vtable_offset = table_offset - vtable_rel
    _require(vtable_offset >= 0, "nested vtable underflow")
    _require(vtable_offset + 4 <= len(data), "nested vtable header out of range")
    vtable_len = _u16(data, vtable_offset)
    object_len = _u16(data, vtable_offset + 2)
    _require(vtable_offset + vtable_len <= len(data), "nested vtable overruns buffer")
    _require(table_offset + object_len <= len(data), "nested table overruns buffer")
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
    _require(target + 4 <= len(table.data), f"offset field {field_id} out of bounds")
    return target


def _read_i32_field(table: FlatTable, field_id: int, *, default: Optional[int] = None) -> Optional[int]:
    off = table.field_offset(field_id)
    if off is None:
        return default
    return _i32(table.data, off)


def _read_i64_field(table: FlatTable, field_id: int, *, default: Optional[int] = None) -> Optional[int]:
    off = table.field_offset(field_id)
    if off is None:
        return default
    return _i64(table.data, off)


def _read_u64_field(table: FlatTable, field_id: int, *, default: Optional[int] = None) -> Optional[int]:
    off = table.field_offset(field_id)
    if off is None:
        return default
    return _u64(table.data, off)


def _read_i16_field(table: FlatTable, field_id: int, *, default: Optional[int] = None) -> Optional[int]:
    off = table.field_offset(field_id)
    if off is None:
        return default
    return _i16(table.data, off)


def _read_u8_field(table: FlatTable, field_id: int, *, default: Optional[int] = None) -> Optional[int]:
    off = table.field_offset(field_id)
    if off is None:
        return default
    return _u8(table.data, off)


def _read_bool_field(table: FlatTable, field_id: int, *, default: bool = False) -> bool:
    off = table.field_offset(field_id)
    if off is None:
        return default
    return _u8(table.data, off) != 0


def _read_f32_field(table: FlatTable, field_id: int, *, default: Optional[float] = None) -> Optional[float]:
    off = table.field_offset(field_id)
    if off is None:
        return default
    return _f32(table.data, off)


def _read_string_field(table: FlatTable, field_id: int, *, default: Optional[str] = None) -> Optional[str]:
    target = _read_offset_object(table, field_id)
    if target is None:
        return default
    slen = _u32(table.data, target)
    sstart = target + 4
    send = sstart + slen
    _require(send <= len(table.data), f"string field {field_id} overruns buffer")
    return table.data[sstart:send].decode("utf-8", errors="replace")


def _read_vector_len(table: FlatTable, field_id: int) -> int:
    target = _read_offset_object(table, field_id)
    if target is None:
        return 0
    return _u32(table.data, target)


def _read_vector_i32_field(table: FlatTable, field_id: int) -> List[int]:
    target = _read_offset_object(table, field_id)
    if target is None:
        return []
    length = _u32(table.data, target)
    vec_start = target + 4
    vec_end = vec_start + (length * 4)
    _require(vec_end <= len(table.data), f"vector<int> field {field_id} out of range")
    return [_i32(table.data, vec_start + (i * 4)) for i in range(length)]


def _read_vector_bytes_field(table: FlatTable, field_id: int) -> bytes:
    target = _read_offset_object(table, field_id)
    if target is None:
        return b""
    length = _u32(table.data, target)
    start = target + 4
    end = start + length
    _require(end <= len(table.data), f"vector<byte> field {field_id} out of range")
    return bytes(table.data[start:end])


def _read_table_field(table: FlatTable, field_id: int) -> Optional[FlatTable]:
    target = _read_offset_object(table, field_id)
    if target is None:
        return None
    return _parse_table_at(table.data, target)


def _read_vector_table_field(table: FlatTable, field_id: int) -> List[FlatTable]:
    target = _read_offset_object(table, field_id)
    if target is None:
        return []
    length = _u32(table.data, target)
    vec_start = target + 4
    vec_end = vec_start + (length * 4)
    _require(vec_end <= len(table.data), f"vector<table> field {field_id} out of range")

    out: List[FlatTable] = []
    for i in range(length):
        slot = vec_start + (i * 4)
        rel = _u32(table.data, slot)
        t_off = slot + rel
        out.append(_parse_table_at(table.data, t_off))
    return out


def _read_vector_of_strings(table: FlatTable, field_id: int) -> List[bytes]:
    target = _read_offset_object(table, field_id)
    if target is None:
        return []
    length = _u32(table.data, target)
    vec_start = target + 4
    vec_end = vec_start + (length * 4)
    _require(vec_end <= len(table.data), f"vector<string> field {field_id} out of range")
    out: List[bytes] = []
    for i in range(length):
        slot = vec_start + (i * 4)
        rel = _u32(table.data, slot)
        s_off = slot + rel
        _require(s_off + 4 <= len(table.data), f"string[{i}] header out of range")
        slen = _u32(table.data, s_off)
        s_start = s_off + 4
        s_end = s_start + slen
        _require(s_end <= len(table.data), f"string[{i}] out of range")
        out.append(bytes(table.data[s_start:s_end]))
    return out


def _read_tensor_shape(table: Optional[FlatTable]) -> Dict[str, Any]:
    if table is None:
        return {"rank": 0, "ranges": [], "extents": []}
    target = _read_offset_object(table, 0)
    if target is None:
        return {"rank": 0, "ranges": [], "extents": []}
    length = _u32(table.data, target)
    vec_start = target + 4
    struct_size = 8  # Range { int start; int end; }
    vec_end = vec_start + (length * struct_size)
    _require(vec_end <= len(table.data), "TensorShape.dimension out of range")

    ranges: List[Dict[str, int]] = []
    extents: List[int] = []
    for i in range(length):
        off = vec_start + (i * struct_size)
        start = _i32(table.data, off)
        end = _i32(table.data, off + 4)
        ranges.append({"start": start, "end": end})
        extents.append((end - start + 1) if end >= start else 0)
    return {"rank": length, "ranges": ranges, "extents": extents}


def _parse_meta(table: Optional[FlatTable]) -> Dict[str, Any]:
    if table is None:
        return {}
    desc = _read_i16_field(table, 0, default=0)
    batch = _read_i32_field(table, 1, default=0) or 0
    name = _read_string_field(table, 2)
    position = _read_i16_field(table, 3, default=0)
    return {
        "desc": desc,
        "desc_name": DESCRIPTION.get(desc, f"UNKNOWN_{desc}"),
        "batch": batch,
        "name": name,
        "position": position,
        "position_name": POSITION.get(position, f"UNKNOWN_{position}"),
    }


def _parse_field_offset(table: FlatTable) -> Dict[str, Any]:
    meta = _parse_meta(_read_table_field(table, 0))
    offset_bit = _read_i32_field(table, 1, default=0) or 0
    return {"offset_bit": offset_bit, "meta": meta}


def _parse_instruction_bitstream(table: FlatTable, max_offsets: int) -> Dict[str, Any]:
    bitstream = _read_vector_bytes_field(table, 0)
    field_tables = _read_vector_table_field(table, 1)
    offsets = [_parse_field_offset(t) for t in field_tables]

    by_desc = Counter(o["meta"].get("desc_name", "UNKNOWN") for o in offsets)
    by_position = Counter(o["meta"].get("position_name", "UNKNOWN") for o in offsets)
    named_refs = Counter(o["meta"].get("name") for o in offsets if o.get("meta", {}).get("name"))

    return {
        "bitstream_size": len(bitstream),
        "bitstream_sha256": _sha256_bytes(bitstream),
        "field_offset_count": len(offsets),
        "field_offsets_preview": offsets[:max_offsets],
        "field_offsets_omitted": max(0, len(offsets) - max_offsets),
        "field_offsets_by_desc": dict(sorted(by_desc.items())),
        "field_offsets_by_position": dict(sorted(by_position.items())),
        "field_offsets_by_name": dict(sorted(named_refs.items())),
        "bitstream_prefix16": _hex_preview(bitstream, n=16),
    }


def _parse_numerics(table: Optional[FlatTable]) -> Dict[str, Any]:
    if table is None:
        return {}
    return {
        "zero_point": _read_i32_field(table, 0),
        "dequantization_factor": _read_f32_field(table, 1),
    }


def _parse_output_layout(table: Optional[FlatTable]) -> Dict[str, Any]:
    if table is None:
        return {}
    return {
        "y_to_tile_len": _read_vector_len(table, 0),
        "x_to_tile_len": _read_vector_len(table, 1),
        "tile_byte_offset_len": _read_vector_len(table, 2),
        "x_local_byte_offset_len": _read_vector_len(table, 3),
        "y_local_offset_len": _read_vector_len(table, 4),
        "x_local_row_size_len": _read_vector_len(table, 5),
    }


def _parse_tensor_layout(table: FlatTable) -> Dict[str, Any]:
    return {
        "shape": _read_tensor_shape(_read_table_field(table, 0)),
        "stride_len": _read_vector_len(table, 1),
    }


def _parse_output_shape_info(table: Optional[FlatTable]) -> Dict[str, Any]:
    if table is None:
        return {}
    layouts = _read_vector_table_field(table, 0)
    return {
        "slice_layout_count": len(layouts),
        "slice_layout_preview": [_parse_tensor_layout(t) for t in layouts[:4]],
        "slice_layout_omitted": max(0, len(layouts) - 4),
        "slice_offset_len": _read_vector_len(table, 1),
    }


def _parse_layer(table: FlatTable) -> Dict[str, Any]:
    name = _read_string_field(table, 0)
    size_bytes = _read_i32_field(table, 1, default=0) or 0
    y_dim = _read_i32_field(table, 2, default=0) or 0
    x_dim = _read_i32_field(table, 3, default=0) or 0
    z_dim = _read_i32_field(table, 4, default=0) or 0
    numerics = _parse_numerics(_read_table_field(table, 5))
    data_type = _read_i16_field(table, 6)
    any_layer_type = _read_u8_field(table, 7, default=0) or 0
    any_layer_obj = _read_table_field(table, 8)
    exec_count = _read_i32_field(table, 9, default=1) or 1
    cache_on_dram = _read_bool_field(table, 10, default=False)
    shape = _read_tensor_shape(_read_table_field(table, 11))

    any_layer_info: Dict[str, Any] = {}
    if any_layer_type == 1:  # OutputLayer
        any_layer_info = {
            "layout": _parse_output_layout(_read_table_field(any_layer_obj, 0) if any_layer_obj else None),
            "shape_info": _parse_output_shape_info(
                _read_table_field(any_layer_obj, 2) if any_layer_obj else None
            ),
        }
    elif any_layer_type == 2:  # InputLayer
        any_layer_info = {}

    return {
        "name": name,
        "size_bytes": size_bytes,
        "dims_yxz": [y_dim, x_dim, z_dim],
        "numerics": numerics,
        "data_type": data_type,
        "data_type_name": DATA_TYPE.get(data_type, f"UNKNOWN_{data_type}"),
        "any_layer_type": any_layer_type,
        "any_layer_type_name": ANY_LAYER.get(any_layer_type, f"UNKNOWN_{any_layer_type}"),
        "any_layer": any_layer_info,
        "execution_count_per_inference": exec_count,
        "cache_on_dram": cache_on_dram,
        "shape": shape,
    }


def _parse_dma_hint(table: FlatTable) -> Dict[str, Any]:
    any_hint_type = _read_u8_field(table, 0, default=0) or 0
    any_hint_table = _read_table_field(table, 1)
    direction = _read_i16_field(table, 2, default=0) or 0

    detail: Dict[str, Any] = {}
    if any_hint_type == 1 and any_hint_table is not None:  # DmaDescriptorHint
        detail = {
            "meta": _parse_meta(_read_table_field(any_hint_table, 0)),
            "offset_in_bytes": _read_i32_field(any_hint_table, 1),
            "size_in_bytes": _read_i32_field(any_hint_table, 2),
        }
    elif any_hint_type == 2 and any_hint_table is not None:  # InstructionHint
        detail = {"instruction_chunk_index": _read_i32_field(any_hint_table, 0)}
    elif any_hint_type == 3 and any_hint_table is not None:  # InterruptHint
        t = _read_i16_field(any_hint_table, 0)
        detail = {"interrupt_type": t, "interrupt_type_name": INTERRUPT_TYPE.get(t, f"UNKNOWN_{t}")}

    return {
        "any_hint_type": any_hint_type,
        "any_hint_type_name": ANY_HINT.get(any_hint_type, f"UNKNOWN_{any_hint_type}"),
        "direction": direction,
        "direction_name": DMA_DIRECTION.get(direction, f"UNKNOWN_{direction}"),
        "detail": detail,
    }


def _parse_dma_hints(table: Optional[FlatTable], max_hints: int) -> Dict[str, Any]:
    if table is None:
        return {"present": False}
    hints = [_parse_dma_hint(t) for t in _read_vector_table_field(table, 0)]
    fully_deterministic = _read_bool_field(table, 1, default=False)

    by_dir = Counter(h["direction_name"] for h in hints)
    by_type = Counter(h["any_hint_type_name"] for h in hints)

    descriptor_sizes = [
        h["detail"]["size_in_bytes"]
        for h in hints
        if h["any_hint_type_name"] == "DmaDescriptorHint" and h.get("detail", {}).get("size_in_bytes") is not None
    ]
    descriptor_offsets = [
        h["detail"]["offset_in_bytes"]
        for h in hints
        if h["any_hint_type_name"] == "DmaDescriptorHint" and h.get("detail", {}).get("offset_in_bytes") is not None
    ]

    return {
        "present": True,
        "fully_deterministic": fully_deterministic,
        "hint_count": len(hints),
        "by_direction": dict(sorted(by_dir.items())),
        "by_any_hint_type": dict(sorted(by_type.items())),
        "dma_descriptor_count": len(descriptor_sizes),
        "dma_descriptor_total_bytes": int(sum(descriptor_sizes)),
        "dma_descriptor_size_min": int(min(descriptor_sizes)) if descriptor_sizes else None,
        "dma_descriptor_size_max": int(max(descriptor_sizes)) if descriptor_sizes else None,
        "dma_descriptor_offset_min": int(min(descriptor_offsets)) if descriptor_offsets else None,
        "dma_descriptor_offset_max": int(max(descriptor_offsets)) if descriptor_offsets else None,
        "hints_preview": hints[:max_hints],
        "hints_omitted": max(0, len(hints) - max_hints),
    }


def _find_markers(data: bytes, markers: Dict[str, bytes], max_hits: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, pat in markers.items():
        hits: List[int] = []
        if not pat:
            out[name] = {"pattern_hex": "", "count": 0, "offsets": []}
            continue
        start = 0
        while True:
            pos = data.find(pat, start)
            if pos < 0:
                break
            hits.append(pos)
            if len(hits) >= max_hits:
                start = pos + len(pat)
                break
            start = pos + 1

        # Continue counting without storing offsets.
        total = len(hits)
        if len(hits) >= max_hits:
            while True:
                pos = data.find(pat, start)
                if pos < 0:
                    break
                total += 1
                start = pos + 1

        out[name] = {"pattern_hex": pat.hex(), "count": total, "offsets": hits}
    return out


def _stats(values: Iterable[int]) -> Dict[str, Optional[float]]:
    vals = list(values)
    if not vals:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "avg": float(sum(vals) / len(vals)),
    }


def parse_executable_blob(
    path: Path,
    max_field_offsets: int,
    max_hints: int,
    marker_patterns: Optional[Dict[str, bytes]],
    max_marker_hits: int,
) -> Dict[str, Any]:
    blob = path.read_bytes()
    root = _parse_root_table(blob, 0, file_identifier=None)

    version = _read_i32_field(root, 0, default=0) or 0
    name = _read_string_field(root, 1)
    serialized_model = _read_vector_bytes_field(root, 2)
    batch_size = _read_i32_field(root, 3, default=1) or 1
    scratch_size_bytes = _read_i32_field(root, 4, default=0) or 0
    instruction_tables = _read_vector_table_field(root, 5)
    parameters = _read_vector_bytes_field(root, 6)
    dma_hints = _parse_dma_hints(_read_table_field(root, 7), max_hints=max_hints)
    input_layer_tables = _read_vector_table_field(root, 8)
    output_layer_tables = _read_vector_table_field(root, 9)
    chip = _read_string_field(root, 10)
    estimated_cycles = _read_i32_field(root, 11)
    used_narrow_memory_bytes_per_tile = _read_i32_field(root, 12)
    exe_type = _read_i16_field(root, 13, default=0) or 0
    parameter_caching_token = _read_u64_field(root, 14)
    use_tpu_dram_for_parameters = _read_bool_field(root, 15, default=False)
    estimated_cycles_64bit = _read_i64_field(root, 16)

    chunks = [_parse_instruction_bitstream(t, max_offsets=max_field_offsets) for t in instruction_tables]
    input_layers = [_parse_layer(t) for t in input_layer_tables]
    output_layers = [_parse_layer(t) for t in output_layer_tables]

    desc_counts = Counter()
    position_counts = Counter()
    named_refs = Counter()
    for c in chunks:
        desc_counts.update(c["field_offsets_by_desc"])
        position_counts.update(c["field_offsets_by_position"])
        named_refs.update(c.get("field_offsets_by_name", {}))

    instruction_sizes = [c["bitstream_size"] for c in chunks]
    layer_dims_in = [l["dims_yxz"] for l in input_layers]
    layer_dims_out = [l["dims_yxz"] for l in output_layers]

    inferred_role = "unknown"
    if parameters and not chunks:
        inferred_role = "parameter_only"
    elif chunks and not parameters:
        inferred_role = "instruction_only"
    elif parameters and chunks:
        ratio = (len(parameters) / max(1, sum(instruction_sizes)))
        if ratio > 8.0:
            inferred_role = "parameter_heavy"
        elif ratio < 0.5:
            inferred_role = "instruction_heavy"
        else:
            inferred_role = "mixed"

    report: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": len(blob),
        "sha256": _sha256_bytes(blob),
        "root_table_offset": root.table_offset,
        "root_vtable_len": root.vtable_len,
        "version": version,
        "name": name,
        "chip": chip,
        "batch_size": batch_size,
        "scratch_size_bytes": scratch_size_bytes,
        "estimated_cycles": estimated_cycles,
        "estimated_cycles_64bit": estimated_cycles_64bit,
        "used_narrow_memory_bytes_per_tile": used_narrow_memory_bytes_per_tile,
        "type": exe_type,
        "type_name": EXECUTABLE_TYPE.get(exe_type, f"UNKNOWN_{exe_type}"),
        "parameter_caching_token": parameter_caching_token,
        "use_tpu_dram_for_parameters": use_tpu_dram_for_parameters,
        "serialized_model": {
            "size_bytes": len(serialized_model),
            "sha256": _sha256_bytes(serialized_model) if serialized_model else None,
            "prefix16": _hex_preview(serialized_model, n=16) if serialized_model else "",
        },
        "instruction_bitstreams": {
            "count": len(chunks),
            "total_bytes": int(sum(instruction_sizes)),
            "size_stats": _stats(instruction_sizes),
            "chunks": chunks,
        },
        "relocations": {
            "total_field_offsets": int(sum(c["field_offset_count"] for c in chunks)),
            "by_desc": dict(sorted(desc_counts.items())),
            "by_position": dict(sorted(position_counts.items())),
            "named_ref_preview": dict(named_refs.most_common(20)),
        },
        "parameters": {
            "size_bytes": len(parameters),
            "sha256": _sha256_bytes(parameters) if parameters else None,
            "prefix16": _hex_preview(parameters, n=16) if parameters else "",
        },
        "dma_hints": dma_hints,
        "input_layers": {
            "count": len(input_layers),
            "dims_yxz": layer_dims_in,
            "layers": input_layers,
        },
        "output_layers": {
            "count": len(output_layers),
            "dims_yxz": layer_dims_out,
            "layers": output_layers,
        },
        "inferred_role": inferred_role,
    }

    if marker_patterns is not None:
        report["marker_hits"] = _find_markers(blob, marker_patterns, max_hits=max_marker_hits)

    return report


def parse_multi_executable_blob(path: Path) -> Dict[str, Any]:
    blob = path.read_bytes()
    table = _parse_root_table(blob, 0, file_identifier=None)
    entries = _read_vector_of_strings(table, 0)
    return {
        "path": str(path),
        "size_bytes": len(blob),
        "sha256": _sha256_bytes(blob),
        "serialized_executable_count": len(entries),
        "serialized_executable_sizes": [len(e) for e in entries],
        "serialized_executable_sha256": [_sha256_bytes(e) for e in entries],
        "serialized_executable_prefix16": [_hex_preview(e, n=16) for e in entries],
    }


def collect_executable_files(paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    files: List[Path] = []
    dirs: List[Path] = []
    for p in paths:
        if p.is_dir():
            dirs.append(p)
            matches = sorted(p.glob("serialized_executable_*.bin"))
            if not matches:
                raise ParseError(f"directory has no serialized_executable_*.bin files: {p}")
            files.extend(matches)
        elif p.is_file():
            files.append(p)
        else:
            raise ParseError(f"path not found: {p}")
    return files, dirs


def build_report(
    paths: List[Path],
    max_field_offsets: int,
    max_hints: int,
    markers: bool,
    max_marker_hits: int,
) -> Dict[str, Any]:
    files, dirs = collect_executable_files(paths)

    marker_patterns: Optional[Dict[str, bytes]] = None
    if markers:
        marker_patterns = {name: bytes.fromhex(hex_s) for name, hex_s in DEFAULT_MARKERS.items()}

    reports = [
        parse_executable_blob(
            path=f,
            max_field_offsets=max_field_offsets,
            max_hints=max_hints,
            marker_patterns=marker_patterns,
            max_marker_hits=max_marker_hits,
        )
        for f in files
    ]

    by_dir: Dict[str, Dict[str, Any]] = {}
    for d in dirs:
        entry: Dict[str, Any] = {"path": str(d)}
        multi_path = d / "serialized_multi_executable.bin"
        if multi_path.exists():
            try:
                entry["multi_executable"] = parse_multi_executable_blob(multi_path)
            except Exception as exc:  # noqa: BLE001
                entry["multi_executable_error"] = str(exc)

        local_reports = [r for r in reports if Path(r["path"]).parent == d]
        entry["executables"] = [
            {
                "path": r["path"],
                "size_bytes": r["size_bytes"],
                "type_name": r["type_name"],
                "instruction_total_bytes": r["instruction_bitstreams"]["total_bytes"],
                "parameters_size_bytes": r["parameters"]["size_bytes"],
                "inferred_role": r["inferred_role"],
            }
            for r in local_reports
        ]
        if "multi_executable" in entry:
            multi_sizes = entry["multi_executable"]["serialized_executable_sizes"]
            file_sizes = [e["size_bytes"] for e in entry["executables"]]
            entry["size_correlation"] = {
                "multi_sizes": multi_sizes,
                "file_sizes": file_sizes,
                "exact_match": multi_sizes == file_sizes,
                "sum_file_sizes": int(sum(file_sizes)),
                "sum_multi_sizes": int(sum(multi_sizes)),
            }
        by_dir[str(d)] = entry

    return {
        "tool": "parse_edgetpu_executable.py",
        "generated_at_utc": _iso_utc_now(),
        "input_paths": [str(p) for p in paths],
        "report_count": len(reports),
        "reports": reports,
        "directories": by_dir,
    }


def render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"tool={report['tool']} generated_at={report['generated_at_utc']}")
    lines.append(f"report_count={report['report_count']}")
    lines.append("")

    for idx, r in enumerate(report["reports"], start=1):
        lines.append(f"[executable {idx}] {r['path']}")
        lines.append(
            "  size={} type={} batch={} chip={} inferred_role={}".format(
                r["size_bytes"], r["type_name"], r["batch_size"], r.get("chip"), r["inferred_role"]
            )
        )
        lines.append(
            "  instruction_chunks={} instruction_total_bytes={} relocations={}".format(
                r["instruction_bitstreams"]["count"],
                r["instruction_bitstreams"]["total_bytes"],
                r["relocations"]["total_field_offsets"],
            )
        )
        lines.append(
            "  parameters_size={} scratch_size={} estimated_cycles_64bit={}".format(
                r["parameters"]["size_bytes"], r["scratch_size_bytes"], r["estimated_cycles_64bit"]
            )
        )
        lines.append(
            "  input_layers={} output_layers={} input_dims={} output_dims={}".format(
                r["input_layers"]["count"],
                r["output_layers"]["count"],
                r["input_layers"]["dims_yxz"],
                r["output_layers"]["dims_yxz"],
            )
        )
        lines.append("  relocations.by_desc=" + json.dumps(r["relocations"]["by_desc"], sort_keys=True))
        if r.get("marker_hits") is not None:
            marker_counts = {k: v["count"] for k, v in r["marker_hits"].items()}
            lines.append("  marker_counts=" + json.dumps(marker_counts, sort_keys=True))
        lines.append("")

    if report["directories"]:
        lines.append("[directories]")
        for d, meta in report["directories"].items():
            lines.append(f"  dir={d}")
            if "multi_executable" in meta:
                me = meta["multi_executable"]
                lines.append(
                    "    multi_executable size={} entries={} sizes={}".format(
                        me["size_bytes"], me["serialized_executable_count"], me["serialized_executable_sizes"]
                    )
                )
            if "size_correlation" in meta:
                corr = meta["size_correlation"]
                lines.append(
                    "    size_match exact={} sum_files={} sum_multi={}".format(
                        corr["exact_match"], corr["sum_file_sizes"], corr["sum_multi_sizes"]
                    )
                )
            for e in meta["executables"]:
                lines.append(
                    "    exe {} type={} instr={} params={} role={}".format(
                        e["path"],
                        e["type_name"],
                        e["instruction_total_bytes"],
                        e["parameters_size_bytes"],
                        e["inferred_role"],
                    )
                )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse EdgeTPU serialized executable FlatBuffers into structured summaries."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="serialized_executable_*.bin files and/or extracted package directories",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument("--max-field-offsets", type=int, default=40, help="Preview count per instruction chunk.")
    parser.add_argument("--max-hints", type=int, default=40, help="Preview count for dma hints.")
    parser.add_argument("--no-markers", action="store_true", help="Disable default marker offset scan.")
    parser.add_argument("--max-marker-hits", type=int, default=32, help="Max stored offsets per marker.")
    args = parser.parse_args()

    try:
        report = build_report(
            paths=args.paths,
            max_field_offsets=max(0, args.max_field_offsets),
            max_hints=max(0, args.max_hints),
            markers=not args.no_markers,
            max_marker_hits=max(0, args.max_marker_hits),
        )
    except ParseError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
