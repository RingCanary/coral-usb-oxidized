#!/usr/bin/env python3
"""PyUSB transport parity harness for Coral USB class-2 parameter ingress.

Goal:
- mirror the same bootstrap path as `examples/rusb_serialized_exec_replay.rs`
- emit explicit per-write `actual_length` behavior from PyUSB/libusb
- isolate whether the 49,152-byte wall is transport-wrapper specific
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import usb.core
import usb.util


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import parse_edgetpu_executable as pe  # noqa: E402


RUNTIME_VID = 0x18D1
RUNTIME_PID = 0x9302
BOOT_VID = 0x1A6E
BOOT_PID = 0x089A

EP_BULK_OUT = 0x01

TAG_INSTRUCTIONS = 0
TAG_INPUT = 1
TAG_PARAMETERS = 2

# Mirror the write-only setup path used by our Rust replay default.
EDGETPUXRAY_SETUP_38: Sequence[Tuple[int, int, int]] = (
    (32, 0x0001_A30C, 0x000F0059),
    (32, 0x0001_A318, 0x5085025C),
    (64, 0x0004_A000, 0x00000001),
    (64, 0x0004_8788, 0x0000007F),
    (64, 0x0004_0020, 0x00001E02),
    (32, 0x0001_A314, 0x00150000),
    (64, 0x0004_C148, 0x000000F0),
    (64, 0x0004_C160, 0x00000000),
    (64, 0x0004_C058, 0x00000080),
    (64, 0x0004_4018, 0x00000001),
    (64, 0x0004_4158, 0x00000001),
    (64, 0x0004_4198, 0x00000001),
    (64, 0x0004_41D8, 0x00000001),
    (64, 0x0004_4218, 0x00000001),
    (64, 0x0004_8788, 0x0000007F),
    (64, 0x0004_00C0, 0x00000001),
    (64, 0x0004_0150, 0x00000001),
    (64, 0x0004_0110, 0x00000001),
    (64, 0x0004_0250, 0x00000001),
    (64, 0x0004_0298, 0x00000001),
    (64, 0x0004_02E0, 0x00000001),
    (64, 0x0004_0328, 0x00000001),
    (64, 0x0004_0190, 0x00000001),
    (64, 0x0004_01D0, 0x00000001),
    (64, 0x0004_0210, 0x00000001),
    (64, 0x0004_C060, 0x00000001),
    (64, 0x0004_C070, 0x00000001),
    (64, 0x0004_C080, 0x00000001),
    (64, 0x0004_C090, 0x00000001),
    (64, 0x0004_C0A0, 0x00000001),
    (32, 0x0001_A0D4, 0x80000001),
    (32, 0x0001_A704, 0x0000007F),
    (32, 0x0001_A33C, 0x0000003F),
    (32, 0x0001_A500, 0x00000001),
    (32, 0x0001_A600, 0x00000001),
    (32, 0x0001_A558, 0x00000003),
    (32, 0x0001_A658, 0x00000003),
    (32, 0x0001_A0D8, 0x80000000),
)


@dataclass
class ExecutablePayload:
    path: Path
    executable_type: int
    instruction_bitstreams: List[bytes]
    parameters_stream: bytes


def _read_executable(path: Path) -> ExecutablePayload:
    blob = path.read_bytes()
    root = pe._parse_root_table(blob, 0, file_identifier=None)
    executable_type = pe._read_i16_field(root, 13, default=0) or 0
    instruction_tables = pe._read_vector_table_field(root, 5)
    instruction_bitstreams = [pe._read_vector_bytes_field(table, 0) for table in instruction_tables]
    parameters_stream = pe._read_vector_bytes_field(root, 6)
    return ExecutablePayload(
        path=path,
        executable_type=executable_type,
        instruction_bitstreams=instruction_bitstreams,
        parameters_stream=parameters_stream,
    )


def _extract_executables(model: Path, out_dir: Path) -> List[ExecutablePayload]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "extract_edgetpu_package.py"),
        "extract",
        str(model),
        "--out",
        str(out_dir),
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)
    paths = sorted(out_dir.rglob("serialized_executable_*.bin"))
    if not paths:
        raise RuntimeError(f"no serialized_executable_*.bin found under {out_dir}")
    return [_read_executable(path) for path in paths]


def _select_executables(executables: Iterable[ExecutablePayload]) -> Tuple[ExecutablePayload, ExecutablePayload]:
    run_exe: Optional[ExecutablePayload] = None
    param_exe: Optional[ExecutablePayload] = None
    for exe in executables:
        if run_exe is None and exe.executable_type == 2 and exe.instruction_bitstreams:
            run_exe = exe
        if param_exe is None and exe.executable_type == 1 and exe.parameters_stream:
            param_exe = exe
    if run_exe is None:
        raise RuntimeError("no EXECUTION_ONLY executable with instruction chunks found")
    if param_exe is None:
        raise RuntimeError("no PARAMETER_CACHING executable with parameter stream found")
    return run_exe, param_exe


def _split_offset(offset: int) -> Tuple[int, int]:
    return offset & 0xFFFF, (offset >> 16) & 0xFFFF


def _vendor_write(dev: usb.core.Device, width: int, offset: int, value: int, timeout_ms: int) -> None:
    req = 0x01 if width == 32 else 0x00
    w_value, w_index = _split_offset(offset)
    if width == 32:
        payload = struct.pack("<I", value & 0xFFFFFFFF)
    else:
        payload = struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF)
    ret = dev.ctrl_transfer(0x40, req, w_value, w_index, payload, timeout=timeout_ms)
    expect = len(payload)
    if int(ret) != expect:
        raise RuntimeError(
            f"vendor write failed width={width} offset=0x{offset:08x}: ret={ret} expect={expect}"
        )


def _apply_setup(dev: usb.core.Device, timeout_ms: int) -> None:
    print(f"Applying setup sequence: {len(EDGETPUXRAY_SETUP_38)} writes")
    for idx, (width, offset, value) in enumerate(EDGETPUXRAY_SETUP_38):
        _vendor_write(dev, width=width, offset=offset, value=value, timeout_ms=timeout_ms)
        print(f"  setup[{idx:02d}] write{width} 0x{offset:08x} = 0x{value:x}")


def _claim_runtime_device(dev: usb.core.Device) -> usb.core.Device:
    try:
        dev.set_configuration(1)
    except usb.core.USBError:
        dev.set_configuration()
    cfg = dev.get_active_configuration()
    intf_num = cfg[(0, 0)].bInterfaceNumber
    try:
        if dev.is_kernel_driver_active(intf_num):
            dev.detach_kernel_driver(intf_num)
    except (NotImplementedError, usb.core.USBError):
        pass
    usb.util.claim_interface(dev, intf_num)
    return dev


def _upload_firmware_if_needed(firmware: Path, timeout_ms: int) -> usb.core.Device:
    runtime = usb.core.find(idVendor=RUNTIME_VID, idProduct=RUNTIME_PID)
    if runtime is not None:
        return runtime

    boot = usb.core.find(idVendor=BOOT_VID, idProduct=BOOT_PID)
    if boot is None:
        raise RuntimeError("no runtime or boot-mode Coral device found")

    fw = firmware.read_bytes()
    print(f"Boot-mode device detected; uploading firmware {firmware} ({len(fw)} bytes)")
    chunk_idx = 0
    for off in range(0, len(fw), 0x100):
        chunk = fw[off : off + 0x100]
        ret = boot.ctrl_transfer(0x21, 1, chunk_idx, 0, chunk, timeout=timeout_ms)
        if int(ret) != len(chunk):
            raise RuntimeError(f"firmware chunk {chunk_idx} short write: ret={ret} len={len(chunk)}")
        _ = boot.ctrl_transfer(0xA1, 3, 0, 0, 6, timeout=timeout_ms)
        chunk_idx += 1
    _ = boot.ctrl_transfer(0x21, 1, chunk_idx, 0, b"", timeout=timeout_ms)
    _ = boot.ctrl_transfer(0xA1, 3, 0, 0, 6, timeout=timeout_ms)
    for i in range(0x81):
        _ = boot.ctrl_transfer(0xA1, 2, i, 0, 0x100, timeout=timeout_ms)

    try:
        boot.reset()
    except usb.core.USBError:
        pass
    time.sleep(0.8)
    runtime = usb.core.find(idVendor=RUNTIME_VID, idProduct=RUNTIME_PID)
    if runtime is None:
        raise RuntimeError("runtime device did not appear after firmware upload")
    return runtime


def _bulk_write_stream(
    dev: usb.core.Device,
    payload: bytes,
    *,
    timeout_ms: int,
    chunk_size: int,
    phase_label: str,
) -> int:
    offset = 0
    total = len(payload)
    while offset < total:
        # Keep a stable chunk window so a partial write retries the remaining
        # bytes in the same logical chunk (matches the Rust replay semantics).
        window_end = min(offset + chunk_size, total)
        while offset < window_end:
            chunk = payload[offset:window_end]
            t0 = time.monotonic()
            try:
                ret = dev.write(EP_BULK_OUT, chunk, timeout=timeout_ms)
                wrote = int(ret)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                print(
                    f"  {phase_label}: off={offset} req={len(chunk)} wrote={wrote} elapsed_ms={elapsed_ms:.3f}"
                )
            except usb.core.USBError as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                errno = getattr(exc, "errno", None)
                be = getattr(exc, "backend_error_code", None)
                print(
                    f"  {phase_label}: off={offset} req={len(chunk)} USBError errno={errno} backend={be} elapsed_ms={elapsed_ms:.3f} err={exc}",
                    file=sys.stderr,
                )
                raise
            if wrote <= 0:
                raise RuntimeError(f"{phase_label}: zero-progress write at offset {offset}")
            offset += wrote
    return offset


def _send_descriptor(
    dev: usb.core.Device,
    *,
    tag: int,
    payload: bytes,
    header_len: Optional[int],
    timeout_ms: int,
    chunk_size: int,
    phase_label: str,
) -> int:
    use_header_len = len(payload) if header_len is None else int(header_len)
    header = struct.pack("<II", use_header_len, tag)
    ret = int(dev.write(EP_BULK_OUT, header, timeout=timeout_ms))
    if ret != len(header):
        raise RuntimeError(f"{phase_label}: header short write ret={ret} expect=8")
    print(
        f"{phase_label}: header tag={tag} payload_len={len(payload)} header_len={use_header_len} wrote={ret}"
    )
    return _bulk_write_stream(
        dev,
        payload,
        timeout_ms=timeout_ms,
        chunk_size=chunk_size,
        phase_label=phase_label,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyUSB parity harness for Coral parameter ingress")
    p.add_argument("--model", required=True, help="compiled *_edgetpu.tflite model")
    p.add_argument("--firmware", default="apex_latest_single_ep.bin", help="firmware blob for boot-mode device")
    p.add_argument("--timeout-ms", type=int, default=12000, help="USB control/bulk timeout (ms)")
    p.add_argument("--param-chunk-size", type=int, default=1024 * 1024, help="parameter stream chunk size")
    p.add_argument(
        "--param-force-full-header-len",
        action="store_true",
        help="set class-2 descriptor header length to full parameter payload length",
    )
    p.add_argument("--param-max-bytes", type=int, default=None, help="cap parameter bytes streamed")
    p.add_argument(
        "--extract-dir",
        default=None,
        help="optional extraction directory (default: temporary directory)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    model = Path(args.model).expanduser().resolve()
    firmware = Path(args.firmware).expanduser().resolve()
    if not model.exists():
        raise SystemExit(f"model not found: {model}")
    if not firmware.exists():
        raise SystemExit(f"firmware not found: {firmware}")
    if args.param_chunk_size <= 0:
        raise SystemExit("--param-chunk-size must be > 0")
    if args.timeout_ms <= 0:
        raise SystemExit("--timeout-ms must be > 0")

    if args.extract_dir:
        extract_dir = Path(args.extract_dir).expanduser().resolve()
        extract_dir.mkdir(parents=True, exist_ok=True)
        cleanup_tmp = None
    else:
        cleanup_tmp = tempfile.TemporaryDirectory(prefix="pyusb-parity-")
        extract_dir = Path(cleanup_tmp.name)

    print(f"model={model}")
    print(f"extract_dir={extract_dir}")
    executables = _extract_executables(model, extract_dir)
    run_exe, param_exe = _select_executables(executables)
    print(
        "selected run_exe={} type={} instr_chunks={} | param_exe={} type={} instr_chunks={} param_bytes={}".format(
            run_exe.path.name,
            run_exe.executable_type,
            len(run_exe.instruction_bitstreams),
            param_exe.path.name,
            param_exe.executable_type,
            len(param_exe.instruction_bitstreams),
            len(param_exe.parameters_stream),
        )
    )

    runtime = _upload_firmware_if_needed(firmware, timeout_ms=args.timeout_ms)
    try:
        runtime.reset()
    except usb.core.USBError:
        pass
    time.sleep(0.6)
    runtime = usb.core.find(idVendor=RUNTIME_VID, idProduct=RUNTIME_PID)
    if runtime is None:
        raise RuntimeError("runtime device missing after reset")
    dev = _claim_runtime_device(runtime)

    _apply_setup(dev, timeout_ms=args.timeout_ms)

    for idx, chunk in enumerate(run_exe.instruction_bitstreams):
        _send_descriptor(
            dev,
            tag=TAG_INSTRUCTIONS,
            payload=chunk,
            header_len=None,
            timeout_ms=args.timeout_ms,
            chunk_size=max(len(chunk), 1),
            phase_label=f"run_exe instr[{idx}]",
        )

    for idx, chunk in enumerate(param_exe.instruction_bitstreams):
        _send_descriptor(
            dev,
            tag=TAG_INSTRUCTIONS,
            payload=chunk,
            header_len=None,
            timeout_ms=args.timeout_ms,
            chunk_size=max(len(chunk), 1),
            phase_label=f"param_exe instr[{idx}]",
        )

    param_payload = param_exe.parameters_stream
    if args.param_max_bytes is not None:
        if args.param_max_bytes <= 0:
            raise SystemExit("--param-max-bytes must be > 0")
        param_payload = param_payload[: args.param_max_bytes]
    header_len = len(param_exe.parameters_stream) if args.param_force_full_header_len else len(param_payload)
    _send_descriptor(
        dev,
        tag=TAG_PARAMETERS,
        payload=param_payload,
        header_len=header_len,
        timeout_ms=args.timeout_ms,
        chunk_size=args.param_chunk_size,
        phase_label="param_exe params",
    )
    print("parameter stream completed without exception")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
