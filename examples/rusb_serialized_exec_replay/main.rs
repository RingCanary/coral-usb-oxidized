use coral_usb_oxidized::{
    dense_param_stream_len, executable_type_name, extract_serialized_executables_from_tflite,
    pack_dense_row_major_i8_to_stream, pack_dense_row_major_u8_to_stream, CoralError,
    DenseFamilyProfile, DescriptorTag, EdgeTpuUsbDriver, SerializedExecutableBlob, VendorDirection,
    EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE, LIBEDGETPU_KNOWN_GOOD_SETUP_SEQUENCE,
};
use rusb::ffi as libusb;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::error::Error;
use std::ffi::CStr;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

mod cli;
mod patch;
mod profile;
mod scripts;
mod stream;

use cli::{parse_args, Config, ParamAdmissionWaitMode};
use patch::{
    descriptor_tag_name, load_instruction_patch_spec, merge_instruction_patch_specs,
    validate_instruction_patch_spec_against_executables, InstructionPatchSpec,
};
use profile::resolve_family_profile;
use scripts::apply_script_defaults;
use stream::header_and_stream_len;

fn choose_execution_executable<'a>(
    executables: &'a [SerializedExecutableBlob],
    forced_index: Option<usize>,
) -> Result<&'a SerializedExecutableBlob, Box<dyn Error>> {
    if let Some(idx) = forced_index {
        return executables
            .iter()
            .find(|exe| exe.executable_index == idx)
            .ok_or_else(|| format!("--exec-index {} not found", idx).into());
    }

    if let Some(exe) = executables.iter().find(|exe| exe.executable_type == 2) {
        return Ok(exe);
    }
    if let Some(exe) = executables.iter().find(|exe| exe.executable_type == 0) {
        return Ok(exe);
    }

    Err("no EXECUTION_ONLY or STAND_ALONE executable found".into())
}

fn load_inputs(config: &Config) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    if let Some(path) = &config.input_batch_file {
        let data = std::fs::read(path)?;
        let expected_len = config
            .runs
            .checked_mul(config.input_bytes)
            .ok_or("input batch size overflow")?;
        if data.len() != expected_len {
            return Err(format!(
                "input batch file size mismatch: expected {} bytes (= runs {} * input_bytes {}), got {} ({})",
                expected_len,
                config.runs,
                config.input_bytes,
                data.len(),
                path
            )
            .into());
        }
        let mut batches = Vec::with_capacity(config.runs);
        for chunk in data.chunks_exact(config.input_bytes) {
            batches.push(chunk.to_vec());
        }
        return Ok(batches);
    }

    if let Some(path) = &config.input_file {
        let data = std::fs::read(path)?;
        if data.len() != config.input_bytes {
            return Err(format!(
                "input file size mismatch: expected {} bytes, got {}",
                config.input_bytes,
                data.len()
            )
            .into());
        }
        return Ok((0..config.runs).map(|_| data.clone()).collect());
    }

    let mut data = vec![0u8; config.input_bytes];
    for (idx, byte) in data.iter_mut().enumerate() {
        *byte = (idx % 251) as u8;
    }
    Ok((0..config.runs).map(|_| data.clone()).collect())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a64(bytes))
}

fn apply_param_stream_override_bytes(
    executables: &mut [SerializedExecutableBlob],
    source_label: &str,
    override_bytes: &[u8],
) -> Result<(), Box<dyn Error>> {
    let override_hash = fnv1a64_hex(override_bytes);
    let mut replaced = 0usize;
    for exe in executables
        .iter_mut()
        .filter(|exe| !exe.parameters_stream.is_empty())
    {
        if exe.parameters_stream.len() != override_bytes.len() {
            return Err(format!(
                "parameter stream override length mismatch for executable idx={}: expected {} bytes, got {} bytes from {}",
                exe.executable_index,
                exe.parameters_stream.len(),
                override_bytes.len(),
                source_label
            )
            .into());
        }
        let before_hash = fnv1a64_hex(&exe.parameters_stream);
        exe.parameters_stream = override_bytes.to_vec();
        let after_hash = fnv1a64_hex(&exe.parameters_stream);
        println!(
            "Parameter stream override: source={} exec_idx={} len={} source_fnv={} before_fnv={} after_fnv={}",
            source_label,
            exe.executable_index,
            exe.parameters_stream.len(),
            override_hash,
            before_hash,
            after_hash
        );
        replaced += 1;
    }
    if replaced == 0 {
        return Err(format!(
            "parameter stream override source had no target executable: {source_label}"
        )
        .into());
    }
    Ok(())
}

fn maybe_generate_param_stream_from_weights(
    config: &Config,
    profile: &DenseFamilyProfile,
) -> Result<Option<(Vec<u8>, String)>, Box<dyn Error>> {
    let rows = profile.stored_weight_rows();
    let cols = profile.stored_weight_cols();
    let expected_len = dense_param_stream_len(rows, cols)?;

    if let Some(path) = config.weights_row_major_u8_file.as_ref() {
        let row_major_u8 = std::fs::read(path)?;
        if row_major_u8.len() != expected_len {
            return Err(format!(
                "--weights-row-major-u8-file length mismatch: expected {} bytes from profile shape [{}x{}], got {} bytes ({})",
                expected_len,
                rows,
                cols,
                row_major_u8.len(),
                path
            )
            .into());
        }
        let stream = pack_dense_row_major_u8_to_stream(rows, cols, &row_major_u8)?;
        println!(
            "Generated parameter stream from u8 row-major weights: source={} shape={}x{} row_major_fnv=0x{} stream_fnv=0x{}",
            path,
            rows,
            cols,
            fnv1a64_hex(&row_major_u8),
            fnv1a64_hex(&stream)
        );
        return Ok(Some((
            stream,
            format!("generated:u8:{}:{}x{}", path, rows, cols),
        )));
    }

    if let Some(path) = config.weights_row_major_i8_file.as_ref() {
        let row_major_i8_bytes = std::fs::read(path)?;
        if row_major_i8_bytes.len() != expected_len {
            return Err(format!(
                "--weights-row-major-i8-file length mismatch: expected {} bytes from profile shape [{}x{}], got {} bytes ({})",
                expected_len,
                rows,
                cols,
                row_major_i8_bytes.len(),
                path
            )
            .into());
        }
        let row_major_i8: Vec<i8> = row_major_i8_bytes.iter().map(|&v| v as i8).collect();
        let stream = pack_dense_row_major_i8_to_stream(rows, cols, &row_major_i8)?;
        println!(
            "Generated parameter stream from i8 row-major weights: source={} shape={}x{} row_major_fnv=0x{} stream_fnv=0x{}",
            path,
            rows,
            cols,
            fnv1a64_hex(&row_major_i8_bytes),
            fnv1a64_hex(&stream)
        );
        return Ok(Some((
            stream,
            format!("generated:i8:{}:{}x{}", path, rows, cols),
        )));
    }

    if config.weights_pattern_index_mod {
        let row_major_u8: Vec<u8> = (0..expected_len)
            .map(|idx| {
                let v = (idx % config.weights_pattern_modulus) as i16;
                if config.weights_pattern_signed_reinterpret {
                    ((v - 128).rem_euclid(256)) as u8
                } else {
                    (v % 256) as u8
                }
            })
            .collect();
        let stream = pack_dense_row_major_u8_to_stream(rows, cols, &row_major_u8)?;
        println!(
            "Generated parameter stream from synthetic pattern: mode=index_mod modulus={} signed_reinterpret={} shape={}x{} row_major_fnv=0x{} stream_fnv=0x{}",
            config.weights_pattern_modulus,
            config.weights_pattern_signed_reinterpret,
            rows,
            cols,
            fnv1a64_hex(&row_major_u8),
            fnv1a64_hex(&stream)
        );
        return Ok(Some((
            stream,
            format!(
                "generated:pattern:index_mod:{}:signed={}:{}x{}",
                config.weights_pattern_modulus,
                config.weights_pattern_signed_reinterpret,
                rows,
                cols
            ),
        )));
    }

    Ok(None)
}

fn apply_instruction_patch_if_enabled(
    instruction_patch_spec: Option<&InstructionPatchSpec>,
    payload: &[u8],
    phase_label: &str,
    chunk_idx: usize,
) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
    let Some(spec) = instruction_patch_spec else {
        return Ok(None);
    };
    let Some(entries) = spec.by_payload_len.get(&payload.len()) else {
        return Ok(None);
    };

    let mut patched = payload.to_vec();
    let mut changed = 0usize;
    for (offset, value) in entries {
        if *offset >= patched.len() {
            return Err(format!(
                "instruction patch out-of-range (phase='{}' chunk={} len={} offset={}): spec mismatch",
                phase_label,
                chunk_idx,
                patched.len(),
                offset
            )
            .into());
        }
        if patched[*offset] != *value {
            patched[*offset] = *value;
            changed += 1;
        }
    }

    let before_hash = fnv1a64_hex(payload);
    let after_hash = fnv1a64_hex(&patched);
    println!(
        "  {}: instr patch chunk={} len={} rules={} changed={} fnv_before={} fnv_after={}",
        phase_label,
        chunk_idx,
        payload.len(),
        entries.len(),
        changed,
        before_hash,
        after_hash
    );
    Ok(Some(patched))
}

fn send_instruction_chunk(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    instruction_patch_spec: Option<&InstructionPatchSpec>,
    phase_label: &str,
    chunk_idx: usize,
    payload: &[u8],
) -> Result<(), Box<dyn Error>> {
    if let Some(patched) =
        apply_instruction_patch_if_enabled(instruction_patch_spec, payload, phase_label, chunk_idx)?
    {
        driver.send_descriptor_payload_raw(config.instructions_tag, &patched)?;
    } else {
        driver.send_descriptor_payload_raw(config.instructions_tag, payload)?;
    }
    Ok(())
}

fn run_param_prepost_reads(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    timeout: Duration,
    phase_label: &str,
    descriptor_idx: usize,
    stage: &str,
) {
    for attempt in 0..config.param_prepost_bulk_in_reads {
        match driver.read_output_bytes_with_timeout(config.param_prepost_bulk_in_size, timeout) {
            Ok(bytes) => {
                let head_len = bytes.len().min(16);
                println!(
                    "      {}: {} descriptor {} bulk-in read {} => {} bytes head={:02x?}",
                    phase_label,
                    stage,
                    descriptor_idx,
                    attempt + 1,
                    bytes.len(),
                    &bytes[..head_len]
                );
            }
            Err(CoralError::UsbError(rusb::Error::Timeout)) => println!(
                "      {}: {} descriptor {} bulk-in read {} => timeout",
                phase_label,
                stage,
                descriptor_idx,
                attempt + 1
            ),
            Err(err) => println!(
                "      {}: {} descriptor {} bulk-in read {} => error: {}",
                phase_label,
                stage,
                descriptor_idx,
                attempt + 1,
                err
            ),
        }
    }

    for attempt in 0..config.param_prepost_event_reads {
        match driver.read_event_packet_with_timeout(timeout) {
            Ok(event) => println!(
                "      {}: {} descriptor {} event read {} => tag={} offset=0x{:016x} length={}",
                phase_label,
                stage,
                descriptor_idx,
                attempt + 1,
                event.tag,
                event.offset,
                event.length
            ),
            Err(CoralError::UsbError(rusb::Error::Timeout)) => println!(
                "      {}: {} descriptor {} event read {} => timeout",
                phase_label,
                stage,
                descriptor_idx,
                attempt + 1
            ),
            Err(err) => println!(
                "      {}: {} descriptor {} event read {} => error: {}",
                phase_label,
                stage,
                descriptor_idx,
                attempt + 1,
                err
            ),
        }
    }

    for attempt in 0..config.param_prepost_interrupt_reads {
        match driver.read_interrupt_packet_with_timeout(timeout) {
            Ok(pkt) => println!(
                "      {}: {} descriptor {} interrupt read {} => raw=0x{:08x} fatal={} top_level_mask=0x{:08x}",
                phase_label,
                stage,
                descriptor_idx,
                attempt + 1,
                pkt.raw,
                pkt.fatal,
                pkt.top_level_mask
            ),
            Err(CoralError::UsbError(rusb::Error::Timeout)) => println!(
                "      {}: {} descriptor {} interrupt read {} => timeout",
                phase_label,
                stage,
                descriptor_idx,
                attempt + 1
            ),
            Err(err) => println!(
                "      {}: {} descriptor {} interrupt read {} => error: {}",
                phase_label,
                stage,
                descriptor_idx,
                attempt + 1,
                err
            ),
        }
    }
}

fn run_param_pre_ingress_handshake(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    phase_label: &str,
    descriptor_idx: usize,
) {
    let has_prepost_reads = config.param_prepost_bulk_in_reads > 0
        || config.param_prepost_event_reads > 0
        || config.param_prepost_interrupt_reads > 0;
    if !config.param_a0d8_handshake && !has_prepost_reads {
        return;
    }

    let timeout = Duration::from_millis(config.param_prepost_timeout_ms);
    println!(
        "      {}: pre-ingress descriptor {} handshake={} a0d8_write=0x{:08x} bulk_reads={} bulk_size={} event_reads={} intr_reads={} timeout_ms={}",
        phase_label,
        descriptor_idx,
        config.param_a0d8_handshake,
        config.param_a0d8_write_value,
        config.param_prepost_bulk_in_reads,
        config.param_prepost_bulk_in_size,
        config.param_prepost_event_reads,
        config.param_prepost_interrupt_reads,
        config.param_prepost_timeout_ms
    );

    run_param_prepost_reads(driver, config, timeout, phase_label, descriptor_idx, "pre");

    if config.param_a0d8_handshake {
        match driver.vendor_read32(0x0001_a0d8) {
            Ok(value) => println!(
                "      {}: handshake descriptor {} read 0x0001a0d8 => 0x{:08x}",
                phase_label, descriptor_idx, value
            ),
            Err(err) => println!(
                "      {}: handshake descriptor {} read 0x0001a0d8 error: {}",
                phase_label, descriptor_idx, err
            ),
        }
        match driver.vendor_write32(0x0001_a0d8, config.param_a0d8_write_value) {
            Ok(()) => println!(
                "      {}: handshake descriptor {} write 0x0001a0d8 <= 0x{:08x}",
                phase_label, descriptor_idx, config.param_a0d8_write_value
            ),
            Err(err) => println!(
                "      {}: handshake descriptor {} write 0x0001a0d8 error: {}",
                phase_label, descriptor_idx, err
            ),
        }
    }

    run_param_prepost_reads(driver, config, timeout, phase_label, descriptor_idx, "post");
}

fn run_known_good_param_gate(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    phase_label: &str,
    gate_idx: usize,
    gate_offset: usize,
) -> Result<(), Box<dyn Error>> {
    println!(
        "      {}: known-good gate #{} at offset={} bytes (a0d4/a704/a33c reads+writes + a500/a600/a558/a658 + a0d8 read/write)",
        phase_label, gate_idx, gate_offset
    );

    let gate_read_write_pairs = [
        (0x0001_a0d4_u32, 0x8000_0001_u32),
        (0x0001_a704_u32, 0x0000_007f_u32),
        (0x0001_a33c_u32, 0x0000_003f_u32),
    ];

    for (offset, value) in gate_read_write_pairs {
        let readback = driver
            .vendor_read32(offset)
            .map_err(|err| format!("{}: gate read 0x{offset:08x} failed: {}", phase_label, err))?;
        println!(
            "      {}: known-good gate #{} read 0x{:08x} => 0x{:08x}",
            phase_label, gate_idx, offset, readback
        );

        driver.vendor_write32(offset, value).map_err(|err| {
            format!(
                "{}: gate write 0x{offset:08x}=0x{value:08x} failed: {}",
                phase_label, err
            )
        })?;
    }

    let gate_writes = [
        (0x0001_a500_u32, 0x0000_0001_u32),
        (0x0001_a600_u32, 0x0000_0001_u32),
        (0x0001_a558_u32, 0x0000_0003_u32),
        (0x0001_a658_u32, 0x0000_0003_u32),
    ];
    for (offset, value) in gate_writes {
        driver.vendor_write32(offset, value).map_err(|err| {
            format!(
                "{}: gate write 0x{offset:08x}=0x{value:08x} failed: {}",
                phase_label, err
            )
        })?;
    }

    let readback = driver
        .vendor_read32(0x0001_a0d8)
        .map_err(|err| format!("{}: gate read 0x0001a0d8 failed: {}", phase_label, err))?;
    println!(
        "      {}: known-good gate #{} read 0x0001a0d8 => 0x{:08x}",
        phase_label, gate_idx, readback
    );

    driver
        .vendor_write32(0x0001_a0d8, config.param_a0d8_write_value)
        .map_err(|err| {
            format!(
                "{}: gate write 0x0001a0d8=0x{:08x} failed: {}",
                phase_label, config.param_a0d8_write_value, err
            )
        })?;

    println!(
        "      {}: known-good gate #{} wrote 0x0001a0d8 <= 0x{:08x}",
        phase_label, gate_idx, config.param_a0d8_write_value
    );

    Ok(())
}

const PARAM_SNAPSHOT_CSRS: [(&str, u32); 9] = [
    ("scalarCoreRunControl", 0x0004_4018),
    ("instruction_queue_base", 0x0004_8590),
    ("instruction_queue_tail", 0x0004_85a8),
    ("instruction_queue_completed_head", 0x0004_85b8),
    ("instruction_queue_int_status", 0x0004_85c8),
    ("param_queue_base", 0x0004_8660),
    ("param_queue_tail", 0x0004_8678),
    ("param_queue_completed_head", 0x0004_8688),
    ("param_queue_int_status", 0x0004_8698),
];

const PARAM_POISON_PROBE_CSRS: [(&str, u32); 2] = [
    ("usbTopInterruptStatus", 0x0004_c060),
    ("scu_ctr_7", 0x0001_a33c),
];

fn has_param_csr_snapshot(config: &Config) -> bool {
    config.param_csr_snapshot_start_bytes.is_some() && config.param_csr_snapshot_end_bytes.is_some()
}

fn dump_param_csr_snapshot(
    driver: &EdgeTpuUsbDriver,
    phase_label: &str,
    reason: &str,
    chunk_idx: usize,
    param_bytes_written: usize,
) {
    println!(
        "      {}: csr snapshot reason={} chunk={} offset={}",
        phase_label, reason, chunk_idx, param_bytes_written
    );
    for (name, offset) in PARAM_SNAPSHOT_CSRS {
        match driver.vendor_read64(offset) {
            Ok(value) => println!(
                "      {}: csr {:<32} 0x{:08x} => 0x{:016x} (read64)",
                phase_label, name, offset, value
            ),
            Err(err64) => match driver.vendor_read32(offset) {
                Ok(value32) => println!(
                    "      {}: csr {:<32} 0x{:08x} => 0x{:08x} (read32 fallback)",
                    phase_label, name, offset, value32
                ),
                Err(err32) => println!(
                    "      {}: csr {:<32} 0x{:08x} read64_err='{}' read32_err='{}'",
                    phase_label, name, offset, err64, err32
                ),
            },
        }
    }
}

fn capture_param_csr_snapshot(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    phase_label: &str,
    reason: &str,
    chunk_idx: usize,
    param_bytes_written: usize,
) {
    if !has_param_csr_snapshot(config) {
        return;
    }
    dump_param_csr_snapshot(driver, phase_label, reason, chunk_idx, param_bytes_written);
}

fn should_capture_param_csr_snapshot(
    config: &Config,
    chunk_idx: usize,
    param_bytes_written: usize,
) -> bool {
    if !has_param_csr_snapshot(config) {
        return false;
    }
    let start = config.param_csr_snapshot_start_bytes.unwrap_or(0);
    let end = config.param_csr_snapshot_end_bytes.unwrap_or(0);
    if param_bytes_written < start || param_bytes_written >= end {
        return false;
    }
    chunk_idx % config.param_csr_snapshot_every_chunks == 0
}

fn run_pending_param_csr_probes(
    driver: &EdgeTpuUsbDriver,
    phase_label: &str,
    chunk_idx: usize,
    param_bytes_written: usize,
    probe_offsets: &[usize],
    probe_cursor: &mut usize,
) {
    while *probe_cursor < probe_offsets.len() && param_bytes_written >= probe_offsets[*probe_cursor]
    {
        dump_param_csr_snapshot(
            driver,
            phase_label,
            "probe_offset",
            chunk_idx,
            param_bytes_written,
        );
        *probe_cursor += 1;
    }
}

fn run_param_poison_probe(
    driver: &EdgeTpuUsbDriver,
    phase_label: &str,
    chunk_idx: usize,
    param_bytes_written: usize,
    poison_probe_offset: Option<usize>,
    poison_probe_done: &mut bool,
) {
    if *poison_probe_done {
        return;
    }
    let Some(offset) = poison_probe_offset else {
        return;
    };
    if param_bytes_written < offset {
        return;
    }

    println!(
        "      {}: poison probe chunk={} offset={} threshold={}",
        phase_label, chunk_idx, param_bytes_written, offset
    );
    for (name, reg) in PARAM_POISON_PROBE_CSRS {
        match driver.vendor_read32(reg) {
            Ok(value) => println!(
                "      {}: poison csr {:<22} 0x{:08x} => 0x{:08x} (ok)",
                phase_label, name, reg, value
            ),
            Err(err) => println!(
                "      {}: poison csr {:<22} 0x{:08x} => err='{}'",
                phase_label, name, reg, err
            ),
        }
    }
    *poison_probe_done = true;
}

fn run_pending_known_good_gates(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    phase_label: &str,
    param_bytes_written: usize,
    gate_offsets: &[usize],
    gate_cursor: &mut usize,
    window_gate: Option<(usize, usize, usize)>,
    next_window_gate_offset: &mut Option<usize>,
    window_gate_count: &mut usize,
) -> Result<(), Box<dyn Error>> {
    while *gate_cursor < gate_offsets.len() && param_bytes_written >= gate_offsets[*gate_cursor] {
        if let Err(err) = run_known_good_param_gate(
            driver,
            config,
            phase_label,
            *gate_cursor + 1,
            gate_offsets[*gate_cursor],
        ) {
            if config.param_csr_snapshot_on_error {
                capture_param_csr_snapshot(
                    driver,
                    config,
                    phase_label,
                    "gate_error",
                    *gate_cursor,
                    param_bytes_written,
                );
            }
            return Err(err);
        }
        *gate_cursor += 1;
    }

    while let (Some((_, window_end, window_step)), Some(gate_offset)) =
        (window_gate, *next_window_gate_offset)
    {
        if param_bytes_written < gate_offset {
            break;
        }
        if let Err(err) = run_known_good_param_gate(
            driver,
            config,
            phase_label,
            gate_offsets.len() + *window_gate_count + 1,
            gate_offset,
        ) {
            if config.param_csr_snapshot_on_error {
                capture_param_csr_snapshot(
                    driver,
                    config,
                    phase_label,
                    "gate_error",
                    gate_offsets.len() + *window_gate_count,
                    param_bytes_written,
                );
            }
            return Err(err);
        }
        *window_gate_count += 1;
        let next = gate_offset.saturating_add(window_step);
        if next >= window_end {
            *next_window_gate_offset = None;
        } else {
            *next_window_gate_offset = Some(next);
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum ParamAsyncLaneKind {
    BulkIn,
    Event,
    Interrupt,
}

impl ParamAsyncLaneKind {
    fn as_str(self) -> &'static str {
        match self {
            ParamAsyncLaneKind::BulkIn => "bulk_in",
            ParamAsyncLaneKind::Event => "event",
            ParamAsyncLaneKind::Interrupt => "interrupt",
        }
    }
}

#[derive(Debug, Clone)]
struct ParamAsyncLaneStats {
    kind: ParamAsyncLaneKind,
    lane_idx: usize,
    reads: usize,
    ok: usize,
    timeouts: usize,
    errors: usize,
}

impl ParamAsyncLaneStats {
    fn new(kind: ParamAsyncLaneKind, lane_idx: usize) -> Self {
        Self {
            kind,
            lane_idx,
            reads: 0,
            ok: 0,
            timeouts: 0,
            errors: 0,
        }
    }
}

fn has_param_async_lanes(config: &Config) -> bool {
    config.param_async_bulk_in_lanes > 0
        || config.param_async_event_lanes > 0
        || config.param_async_interrupt_lanes > 0
}

fn has_param_known_good_gates(config: &Config) -> bool {
    !config.param_gate_known_good_offsets.is_empty()
        || (config.param_gate_window_start_bytes.is_some()
            && config.param_gate_window_end_bytes.is_some()
            && config.param_gate_window_step_bytes.is_some())
}

fn has_param_csr_probes(config: &Config) -> bool {
    !config.param_csr_probe_offsets.is_empty()
}

fn has_param_poison_probe(config: &Config) -> bool {
    config.param_poison_probe_offset.is_some()
}

fn has_param_admission_wait(config: &Config) -> bool {
    config.param_admission_wait_mode.is_some()
}

fn has_param_submit_lanes(config: &Config) -> bool {
    config.param_submit_bulk_out
        || config.param_submit_bulk_in_lanes > 0
        || config.param_submit_event_lanes > 0
        || config.param_submit_interrupt_lanes > 0
}

#[derive(Debug, Clone, Copy)]
enum ParamSubmitLaneKind {
    BulkOut,
    BulkIn,
    Event,
    Interrupt,
}

impl ParamSubmitLaneKind {
    fn as_str(self) -> &'static str {
        match self {
            ParamSubmitLaneKind::BulkOut => "bulk_out",
            ParamSubmitLaneKind::BulkIn => "bulk_in",
            ParamSubmitLaneKind::Event => "event",
            ParamSubmitLaneKind::Interrupt => "interrupt",
        }
    }

    fn endpoint(self) -> u8 {
        match self {
            ParamSubmitLaneKind::BulkOut => 0x01,
            ParamSubmitLaneKind::BulkIn => 0x81,
            ParamSubmitLaneKind::Event => 0x82,
            ParamSubmitLaneKind::Interrupt => 0x83,
        }
    }
}

#[derive(Default)]
struct ParamSubmitLaneCounters {
    callbacks: AtomicUsize,
    bytes: AtomicUsize,
    completed: AtomicUsize,
    timed_out: AtomicUsize,
    cancelled: AtomicUsize,
    stalled: AtomicUsize,
    no_device: AtomicUsize,
    overflow: AtomicUsize,
    errors: AtomicUsize,
    submit_errors: AtomicUsize,
}

struct ParamSubmitLaneUserData {
    kind: ParamSubmitLaneKind,
    lane_idx: usize,
    stop: Arc<AtomicBool>,
    resubmit: bool,
    submitted: Arc<AtomicBool>,
    counters: Arc<ParamSubmitLaneCounters>,
    log_every: usize,
}

struct ParamSubmitLane {
    kind: ParamSubmitLaneKind,
    lane_idx: usize,
    transfer: *mut libusb::libusb_transfer,
    user_data: *mut ParamSubmitLaneUserData,
    submitted: Arc<AtomicBool>,
    counters: Arc<ParamSubmitLaneCounters>,
    _buffer: Vec<u8>,
}

impl Drop for ParamSubmitLane {
    fn drop(&mut self) {
        if self.submitted.load(Ordering::Relaxed) {
            println!(
                "      async submit lane {}#{} still submitted during drop; leaking transfer for safety",
                self.kind.as_str(),
                self.lane_idx
            );
            return;
        }
        unsafe {
            if !self.transfer.is_null() {
                libusb::libusb_free_transfer(self.transfer);
            }
            if !self.user_data.is_null() {
                let _ = Box::from_raw(self.user_data);
            }
        }
    }
}

fn libusb_error_name(code: c_int) -> String {
    unsafe {
        let ptr = libusb::libusb_error_name(code);
        if ptr.is_null() {
            return format!("code {}", code);
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

fn libusb_transfer_status_name(status: c_int) -> &'static str {
    match status {
        libusb::constants::LIBUSB_TRANSFER_COMPLETED => "completed",
        libusb::constants::LIBUSB_TRANSFER_ERROR => "error",
        libusb::constants::LIBUSB_TRANSFER_TIMED_OUT => "timed_out",
        libusb::constants::LIBUSB_TRANSFER_CANCELLED => "cancelled",
        libusb::constants::LIBUSB_TRANSFER_STALL => "stall",
        libusb::constants::LIBUSB_TRANSFER_NO_DEVICE => "no_device",
        libusb::constants::LIBUSB_TRANSFER_OVERFLOW => "overflow",
        _ => "unknown",
    }
}

extern "system" fn param_submit_transfer_callback(transfer: *mut libusb::libusb_transfer) {
    unsafe {
        if transfer.is_null() {
            return;
        }
        let user_data_ptr = (*transfer).user_data as *mut ParamSubmitLaneUserData;
        if user_data_ptr.is_null() {
            return;
        }
        let user_data = &*user_data_ptr;
        let counters = &user_data.counters;
        let callback_idx = counters.callbacks.fetch_add(1, Ordering::Relaxed) + 1;
        let status = (*transfer).status;
        let actual_length = (*transfer).actual_length.max(0) as usize;
        if actual_length > 0 {
            counters.bytes.fetch_add(actual_length, Ordering::Relaxed);
        }

        match status {
            libusb::constants::LIBUSB_TRANSFER_COMPLETED => {
                counters.completed.fetch_add(1, Ordering::Relaxed);
            }
            libusb::constants::LIBUSB_TRANSFER_TIMED_OUT => {
                counters.timed_out.fetch_add(1, Ordering::Relaxed);
            }
            libusb::constants::LIBUSB_TRANSFER_CANCELLED => {
                counters.cancelled.fetch_add(1, Ordering::Relaxed);
            }
            libusb::constants::LIBUSB_TRANSFER_STALL => {
                counters.stalled.fetch_add(1, Ordering::Relaxed);
            }
            libusb::constants::LIBUSB_TRANSFER_NO_DEVICE => {
                counters.no_device.fetch_add(1, Ordering::Relaxed);
            }
            libusb::constants::LIBUSB_TRANSFER_OVERFLOW => {
                counters.overflow.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                counters.errors.fetch_add(1, Ordering::Relaxed);
            }
        }

        if user_data.log_every > 0 && (callback_idx % user_data.log_every == 0) {
            println!(
                "      async submit lane {}#{} callback={} status={} actual_length={}",
                user_data.kind.as_str(),
                user_data.lane_idx,
                callback_idx,
                libusb_transfer_status_name(status),
                (*transfer).actual_length
            );
        }

        if !user_data.resubmit || user_data.stop.load(Ordering::Relaxed) {
            user_data.submitted.store(false, Ordering::Relaxed);
            return;
        }

        let rc = libusb::libusb_submit_transfer(transfer);
        if rc == 0 {
            user_data.submitted.store(true, Ordering::Relaxed);
            return;
        }

        counters.submit_errors.fetch_add(1, Ordering::Relaxed);
        user_data.submitted.store(false, Ordering::Relaxed);
        println!(
            "      async submit lane {}#{} resubmit failed: {} ({})",
            user_data.kind.as_str(),
            user_data.lane_idx,
            rc,
            libusb_error_name(rc)
        );
    }
}

fn create_param_submit_lane(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    stop: Arc<AtomicBool>,
    kind: ParamSubmitLaneKind,
    lane_idx: usize,
) -> Result<ParamSubmitLane, Box<dyn Error>> {
    let transfer = unsafe { libusb::libusb_alloc_transfer(0) };
    if transfer.is_null() {
        return Err(format!(
            "failed to allocate libusb transfer for lane {}#{}",
            kind.as_str(),
            lane_idx
        )
        .into());
    }

    let lane_buffer_size = match kind {
        ParamSubmitLaneKind::BulkIn => config.param_submit_buffer_size,
        ParamSubmitLaneKind::Event => 16,
        ParamSubmitLaneKind::Interrupt => 4,
        ParamSubmitLaneKind::BulkOut => config.param_submit_buffer_size,
    };
    let mut buffer = vec![0u8; lane_buffer_size];
    let submitted = Arc::new(AtomicBool::new(false));
    let counters = Arc::new(ParamSubmitLaneCounters::default());
    let user_data = Box::new(ParamSubmitLaneUserData {
        kind,
        lane_idx,
        stop,
        resubmit: true,
        submitted: Arc::clone(&submitted),
        counters: Arc::clone(&counters),
        log_every: config.param_submit_log_every,
    });
    let user_data_ptr = Box::into_raw(user_data);

    unsafe {
        match kind {
            ParamSubmitLaneKind::Interrupt => libusb::libusb_fill_interrupt_transfer(
                transfer,
                driver.raw_libusb_handle(),
                kind.endpoint(),
                buffer.as_mut_ptr(),
                buffer.len() as c_int,
                param_submit_transfer_callback,
                user_data_ptr as *mut _,
                config.param_submit_timeout_ms.min(u32::MAX as u64) as u32,
            ),
            ParamSubmitLaneKind::BulkOut
            | ParamSubmitLaneKind::BulkIn
            | ParamSubmitLaneKind::Event => libusb::libusb_fill_bulk_transfer(
                transfer,
                driver.raw_libusb_handle(),
                kind.endpoint(),
                buffer.as_mut_ptr(),
                buffer.len() as c_int,
                param_submit_transfer_callback,
                user_data_ptr as *mut _,
                config.param_submit_timeout_ms.min(u32::MAX as u64) as u32,
            ),
        }

        let rc = libusb::libusb_submit_transfer(transfer);
        if rc != 0 {
            let _ = Box::from_raw(user_data_ptr);
            libusb::libusb_free_transfer(transfer);
            return Err(format!(
                "failed to submit async lane {}#{}: {} ({})",
                kind.as_str(),
                lane_idx,
                rc,
                libusb_error_name(rc)
            )
            .into());
        }
    }
    submitted.store(true, Ordering::Relaxed);

    Ok(ParamSubmitLane {
        kind,
        lane_idx,
        transfer,
        user_data: user_data_ptr,
        submitted,
        counters,
        _buffer: buffer,
    })
}

struct ParamBulkOutAttemptStats {
    bytes: usize,
    completed: usize,
    timed_out: usize,
    cancelled: usize,
    stalled: usize,
    no_device: usize,
    overflow: usize,
    errors: usize,
    submit_errors: usize,
}

impl ParamBulkOutAttemptStats {
    fn summary(&self) -> String {
        format!(
            "actual={} completed={} timed_out={} cancelled={} stall={} no_device={} overflow={} errors={} submit_errors={}",
            self.bytes,
            self.completed,
            self.timed_out,
            self.cancelled,
            self.stalled,
            self.no_device,
            self.overflow,
            self.errors,
            self.submit_errors
        )
    }
}

fn submit_param_bulk_out_once(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    phase_label: &str,
    lane_idx: usize,
    stream_offset: usize,
    chunk: &[u8],
) -> Result<ParamBulkOutAttemptStats, Box<dyn Error>> {
    let transfer = unsafe { libusb::libusb_alloc_transfer(0) };
    if transfer.is_null() {
        return Err("failed to allocate libusb transfer for bulk_out chunk".into());
    }

    let mut buffer = chunk.to_vec();
    let submitted = Arc::new(AtomicBool::new(false));
    let counters = Arc::new(ParamSubmitLaneCounters::default());
    let stop = Arc::new(AtomicBool::new(false));
    let user_data = Box::new(ParamSubmitLaneUserData {
        kind: ParamSubmitLaneKind::BulkOut,
        lane_idx,
        stop,
        resubmit: false,
        submitted: Arc::clone(&submitted),
        counters: Arc::clone(&counters),
        log_every: config.param_submit_log_every,
    });
    let user_data_ptr = Box::into_raw(user_data);

    unsafe {
        libusb::libusb_fill_bulk_transfer(
            transfer,
            driver.raw_libusb_handle(),
            ParamSubmitLaneKind::BulkOut.endpoint(),
            buffer.as_mut_ptr(),
            buffer.len() as c_int,
            param_submit_transfer_callback,
            user_data_ptr as *mut _,
            config.param_submit_timeout_ms.min(u32::MAX as u64) as u32,
        );
    }

    let submit_rc = unsafe { libusb::libusb_submit_transfer(transfer) };
    if submit_rc != 0 {
        unsafe {
            let _ = Box::from_raw(user_data_ptr);
            libusb::libusb_free_transfer(transfer);
        }
        return Err(format!(
            "{}: failed to submit bulk_out lane={} at offset {} ({} bytes): {} ({})",
            phase_label,
            lane_idx,
            stream_offset,
            chunk.len(),
            submit_rc,
            libusb_error_name(submit_rc)
        )
        .into());
    }
    submitted.store(true, Ordering::Relaxed);

    let settle_deadline = Instant::now()
        + Duration::from_millis(config.param_submit_timeout_ms.saturating_mul(4).max(200));
    while submitted.load(Ordering::Relaxed) && Instant::now() < settle_deadline {
        std::thread::sleep(Duration::from_micros(200));
    }
    if submitted.load(Ordering::Relaxed) {
        let cancel_rc = unsafe { libusb::libusb_cancel_transfer(transfer) };
        let cancel_deadline = Instant::now()
            + Duration::from_millis(config.param_submit_timeout_ms.saturating_mul(2).max(100));
        while submitted.load(Ordering::Relaxed) && Instant::now() < cancel_deadline {
            std::thread::sleep(Duration::from_micros(200));
        }
        if submitted.load(Ordering::Relaxed) {
            unsafe {
                libusb::libusb_free_transfer(transfer);
                let _ = Box::from_raw(user_data_ptr);
            }
            return Err(format!(
                "{}: bulk_out lane={} stuck after cancel at offset {} ({} bytes), cancel_rc={} ({})",
                phase_label,
                lane_idx,
                stream_offset,
                chunk.len(),
                cancel_rc,
                libusb_error_name(cancel_rc)
            )
            .into());
        }
    }

    let stats = ParamBulkOutAttemptStats {
        bytes: counters.bytes.load(Ordering::Relaxed),
        completed: counters.completed.load(Ordering::Relaxed),
        timed_out: counters.timed_out.load(Ordering::Relaxed),
        cancelled: counters.cancelled.load(Ordering::Relaxed),
        stalled: counters.stalled.load(Ordering::Relaxed),
        no_device: counters.no_device.load(Ordering::Relaxed),
        overflow: counters.overflow.load(Ordering::Relaxed),
        errors: counters.errors.load(Ordering::Relaxed),
        submit_errors: counters.submit_errors.load(Ordering::Relaxed),
    };

    unsafe {
        libusb::libusb_free_transfer(transfer);
        let _ = Box::from_raw(user_data_ptr);
    }

    Ok(stats)
}

fn submit_param_bulk_out_chunk(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    phase_label: &str,
    chunk_idx: usize,
    stream_offset: usize,
    chunk: &[u8],
) -> Result<(), Box<dyn Error>> {
    if chunk.is_empty() {
        return Ok(());
    }

    let mut sent = 0usize;
    let mut attempts = 0usize;
    let mut retries_left = config.param_submit_bulk_out_max_retries;
    while sent < chunk.len() {
        attempts += 1;
        let lane_idx = chunk_idx + attempts;
        let attempt_offset = stream_offset + sent;
        let remaining = &chunk[sent..];
        let stats = submit_param_bulk_out_once(
            driver,
            config,
            phase_label,
            lane_idx,
            attempt_offset,
            remaining,
        )?;
        let progressed = stats.bytes.min(remaining.len());

        if progressed == remaining.len() {
            sent += progressed;
            continue;
        }

        if progressed > 0 && config.param_submit_bulk_out_accept_partial {
            sent += progressed;
            println!(
                "      {}: bulk_out chunk {} partial progress at offset {} progressed={} remaining={} ({})",
                phase_label,
                chunk_idx,
                attempt_offset,
                progressed,
                chunk.len() - sent,
                stats.summary()
            );
            continue;
        }

        if progressed == 0 && retries_left > 0 {
            retries_left -= 1;
            println!(
                "      {}: bulk_out chunk {} zero-progress retry at offset {} retries_left={} ({})",
                phase_label,
                chunk_idx,
                attempt_offset,
                retries_left,
                stats.summary()
            );
            continue;
        }

        return Err(format!(
            "{}: bulk_out chunk {} failed at offset {} expected_remaining={} accept_partial={} retries_left={} ({})",
            phase_label,
            chunk_idx,
            attempt_offset,
            remaining.len(),
            config.param_submit_bulk_out_accept_partial,
            retries_left,
            stats.summary()
        )
        .into());
    }

    if attempts > 1 {
        println!(
            "      {}: bulk_out chunk {} converged via {} attempts total_bytes={}",
            phase_label,
            chunk_idx,
            attempts,
            chunk.len()
        );
    }

    Ok(())
}

struct ParamBulkOutInFlight {
    transfer: *mut libusb::libusb_transfer,
    user_data: *mut ParamSubmitLaneUserData,
    submitted: Arc<AtomicBool>,
    counters: Arc<ParamSubmitLaneCounters>,
    expected_len: usize,
    chunk_idx: usize,
    stream_offset: usize,
    _buffer: Vec<u8>,
}

impl Drop for ParamBulkOutInFlight {
    fn drop(&mut self) {
        if self.submitted.load(Ordering::Relaxed) {
            println!(
                "      bulk_out inflight chunk {} still submitted during drop; leaking transfer for safety",
                self.chunk_idx
            );
            return;
        }
        unsafe {
            if !self.transfer.is_null() {
                libusb::libusb_free_transfer(self.transfer);
            }
            if !self.user_data.is_null() {
                let _ = Box::from_raw(self.user_data);
            }
        }
    }
}

fn submit_param_bulk_out_inflight(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    phase_label: &str,
    chunk_idx: usize,
    stream_offset: usize,
    chunk: &[u8],
) -> Result<ParamBulkOutInFlight, Box<dyn Error>> {
    let transfer = unsafe { libusb::libusb_alloc_transfer(0) };
    if transfer.is_null() {
        return Err("failed to allocate libusb transfer for inflight bulk_out chunk".into());
    }

    let mut buffer = chunk.to_vec();
    let submitted = Arc::new(AtomicBool::new(false));
    let counters = Arc::new(ParamSubmitLaneCounters::default());
    let stop = Arc::new(AtomicBool::new(false));
    let user_data = Box::new(ParamSubmitLaneUserData {
        kind: ParamSubmitLaneKind::BulkOut,
        lane_idx: chunk_idx + 1,
        stop,
        resubmit: false,
        submitted: Arc::clone(&submitted),
        counters: Arc::clone(&counters),
        log_every: config.param_submit_log_every,
    });
    let user_data_ptr = Box::into_raw(user_data);

    unsafe {
        libusb::libusb_fill_bulk_transfer(
            transfer,
            driver.raw_libusb_handle(),
            ParamSubmitLaneKind::BulkOut.endpoint(),
            buffer.as_mut_ptr(),
            buffer.len() as c_int,
            param_submit_transfer_callback,
            user_data_ptr as *mut _,
            config.param_submit_timeout_ms.min(u32::MAX as u64) as u32,
        );
    }

    let submit_rc = unsafe { libusb::libusb_submit_transfer(transfer) };
    if submit_rc != 0 {
        unsafe {
            let _ = Box::from_raw(user_data_ptr);
            libusb::libusb_free_transfer(transfer);
        }
        return Err(format!(
            "{}: failed to submit inflight bulk_out chunk {} at offset {} ({} bytes): {} ({})",
            phase_label,
            chunk_idx,
            stream_offset,
            chunk.len(),
            submit_rc,
            libusb_error_name(submit_rc)
        )
        .into());
    }
    submitted.store(true, Ordering::Relaxed);

    Ok(ParamBulkOutInFlight {
        transfer,
        user_data: user_data_ptr,
        submitted,
        counters,
        expected_len: chunk.len(),
        chunk_idx,
        stream_offset,
        _buffer: buffer,
    })
}

fn finalize_param_bulk_out_inflight(
    inflight: &ParamBulkOutInFlight,
    config: &Config,
    phase_label: &str,
) -> Result<(), Box<dyn Error>> {
    let settle_deadline = Instant::now()
        + Duration::from_millis(config.param_submit_timeout_ms.saturating_mul(4).max(200));
    while inflight.submitted.load(Ordering::Relaxed) && Instant::now() < settle_deadline {
        std::thread::sleep(Duration::from_micros(200));
    }
    if inflight.submitted.load(Ordering::Relaxed) {
        let cancel_rc = unsafe { libusb::libusb_cancel_transfer(inflight.transfer) };
        let cancel_deadline = Instant::now()
            + Duration::from_millis(config.param_submit_timeout_ms.saturating_mul(2).max(100));
        while inflight.submitted.load(Ordering::Relaxed) && Instant::now() < cancel_deadline {
            std::thread::sleep(Duration::from_micros(200));
        }
        if inflight.submitted.load(Ordering::Relaxed) {
            return Err(format!(
                "{}: inflight bulk_out chunk {} stuck after cancel at offset {} expected={} cancel_rc={} ({})",
                phase_label,
                inflight.chunk_idx,
                inflight.stream_offset,
                inflight.expected_len,
                cancel_rc,
                libusb_error_name(cancel_rc)
            )
            .into());
        }
    }

    let stats = ParamBulkOutAttemptStats {
        bytes: inflight.counters.bytes.load(Ordering::Relaxed),
        completed: inflight.counters.completed.load(Ordering::Relaxed),
        timed_out: inflight.counters.timed_out.load(Ordering::Relaxed),
        cancelled: inflight.counters.cancelled.load(Ordering::Relaxed),
        stalled: inflight.counters.stalled.load(Ordering::Relaxed),
        no_device: inflight.counters.no_device.load(Ordering::Relaxed),
        overflow: inflight.counters.overflow.load(Ordering::Relaxed),
        errors: inflight.counters.errors.load(Ordering::Relaxed),
        submit_errors: inflight.counters.submit_errors.load(Ordering::Relaxed),
    };
    let progressed = stats.bytes.min(inflight.expected_len);
    if progressed == inflight.expected_len && stats.completed > 0 {
        return Ok(());
    }

    Err(format!(
        "{}: inflight bulk_out chunk {} failed at offset {} expected={} ({})",
        phase_label,
        inflight.chunk_idx,
        inflight.stream_offset,
        inflight.expected_len,
        stats.summary()
    )
    .into())
}

fn cleanup_inflight_bulk_out_queue(
    inflight_bulk_out: &mut VecDeque<ParamBulkOutInFlight>,
    config: &Config,
    phase_label: &str,
) {
    while let Some(inflight) = inflight_bulk_out.pop_front() {
        if let Err(err) = finalize_param_bulk_out_inflight(&inflight, config, phase_label) {
            println!(
                "      {}: cleanup inflight bulk_out chunk {} offset {} failed: {}",
                phase_label, inflight.chunk_idx, inflight.stream_offset, err
            );
        }
    }
}

fn cancel_param_submit_lanes(lanes: &[ParamSubmitLane], phase_label: &str) {
    for lane in lanes {
        if !lane.submitted.load(Ordering::Relaxed) {
            continue;
        }
        let rc = unsafe { libusb::libusb_cancel_transfer(lane.transfer) };
        if rc == 0 {
            continue;
        }
        if rc == libusb::constants::LIBUSB_ERROR_NOT_FOUND
            || rc == libusb::constants::LIBUSB_ERROR_NO_DEVICE
        {
            continue;
        }
        println!(
            "      {}: async submit lane {}#{} cancel rc={} ({})",
            phase_label,
            lane.kind.as_str(),
            lane.lane_idx,
            rc,
            libusb_error_name(rc)
        );
    }
}

fn print_param_submit_lane_stats(lanes: &[ParamSubmitLane], phase_label: &str) {
    for lane in lanes {
        let counters = &lane.counters;
        println!(
            "      {}: async submit lane {}#{} callbacks={} bytes={} completed={} timed_out={} cancelled={} stall={} no_device={} overflow={} errors={} submit_errors={} submitted={}",
            phase_label,
            lane.kind.as_str(),
            lane.lane_idx,
            counters.callbacks.load(Ordering::Relaxed),
            counters.bytes.load(Ordering::Relaxed),
            counters.completed.load(Ordering::Relaxed),
            counters.timed_out.load(Ordering::Relaxed),
            counters.cancelled.load(Ordering::Relaxed),
            counters.stalled.load(Ordering::Relaxed),
            counters.no_device.load(Ordering::Relaxed),
            counters.overflow.load(Ordering::Relaxed),
            counters.errors.load(Ordering::Relaxed),
            counters.submit_errors.load(Ordering::Relaxed),
            lane.submitted.load(Ordering::Relaxed)
        );
    }
}

fn stream_parameter_chunks_with_submit_lanes(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    payload: &[u8],
    phase_label: &str,
    header_total_len: usize,
    stream_len: usize,
    stream_chunk_size: usize,
    descriptor_split_size: usize,
    poll_timeout: Duration,
    interrupt_poll_timeout: Duration,
) -> Result<(), Box<dyn Error>> {
    println!(
        "      {}: libusb submit lanes bulk_out={} bulk_out_accept_partial={} bulk_out_max_retries={} bulk_out_depth={} bulk_in={} event={} interrupt={} buffer_size={} transfer_timeout_ms={} event_poll_ms={} log_every={}",
        phase_label,
        config.param_submit_bulk_out,
        config.param_submit_bulk_out_accept_partial,
        config.param_submit_bulk_out_max_retries,
        config.param_submit_bulk_out_depth,
        config.param_submit_bulk_in_lanes,
        config.param_submit_event_lanes,
        config.param_submit_interrupt_lanes,
        config.param_submit_buffer_size,
        config.param_submit_timeout_ms,
        config.param_submit_event_poll_ms,
        config.param_submit_log_every
    );

    let stream_stop = Arc::new(AtomicBool::new(false));
    let event_loop_stop = Arc::new(AtomicBool::new(false));
    let event_loop_polls = Arc::new(AtomicUsize::new(0));
    let event_loop_errors = Arc::new(AtomicUsize::new(0));
    let mut lanes = Vec::new();

    for lane_idx in 0..config.param_submit_bulk_in_lanes {
        lanes.push(create_param_submit_lane(
            driver,
            config,
            Arc::clone(&stream_stop),
            ParamSubmitLaneKind::BulkIn,
            lane_idx + 1,
        )?);
    }
    for lane_idx in 0..config.param_submit_event_lanes {
        lanes.push(create_param_submit_lane(
            driver,
            config,
            Arc::clone(&stream_stop),
            ParamSubmitLaneKind::Event,
            lane_idx + 1,
        )?);
    }
    for lane_idx in 0..config.param_submit_interrupt_lanes {
        lanes.push(create_param_submit_lane(
            driver,
            config,
            Arc::clone(&stream_stop),
            ParamSubmitLaneKind::Interrupt,
            lane_idx + 1,
        )?);
    }

    let event_poll_timeout = Duration::from_millis(config.param_submit_event_poll_ms);
    let result = std::thread::scope(|scope| -> Result<(), Box<dyn Error>> {
        let event_loop_stop_for_thread = Arc::clone(&event_loop_stop);
        let event_loop_polls_for_thread = Arc::clone(&event_loop_polls);
        let event_loop_errors_for_thread = Arc::clone(&event_loop_errors);
        let event_thread = scope.spawn(move || {
            while !event_loop_stop_for_thread.load(Ordering::Relaxed) {
                event_loop_polls_for_thread.fetch_add(1, Ordering::Relaxed);
                if let Err(err) = driver.handle_events_timeout(Some(event_poll_timeout)) {
                    event_loop_errors_for_thread.fetch_add(1, Ordering::Relaxed);
                    if !matches!(err, CoralError::UsbError(rusb::Error::Timeout)) {
                        println!("      async submit event loop error: {}", err);
                    }
                }
            }
        });

        let stream_result = stream_parameter_chunks(
            driver,
            config,
            payload,
            phase_label,
            header_total_len,
            stream_len,
            stream_chunk_size,
            descriptor_split_size,
            poll_timeout,
            interrupt_poll_timeout,
        );

        stream_stop.store(true, Ordering::Relaxed);
        cancel_param_submit_lanes(&lanes, phase_label);

        let settle_deadline = Instant::now()
            + Duration::from_millis(config.param_submit_timeout_ms.saturating_mul(4).max(200));
        while Instant::now() < settle_deadline {
            if lanes
                .iter()
                .all(|lane| !lane.submitted.load(Ordering::Relaxed))
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }

        event_loop_stop.store(true, Ordering::Relaxed);
        let _ = event_thread.join();
        stream_result
    });

    println!(
        "      {}: async submit event loop polls={} errors={}",
        phase_label,
        event_loop_polls.load(Ordering::Relaxed),
        event_loop_errors.load(Ordering::Relaxed)
    );
    print_param_submit_lane_stats(&lanes, phase_label);
    result
}

fn run_with_global_submit_lanes<F>(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    phase_label: &str,
    body: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce() -> Result<(), Box<dyn Error>>,
{
    println!(
        "Enabling global submit lanes: bulk_in={} event={} interrupt={} buffer_size={} transfer_timeout_ms={} event_poll_ms={} log_every={}",
        config.param_submit_bulk_in_lanes,
        config.param_submit_event_lanes,
        config.param_submit_interrupt_lanes,
        config.param_submit_buffer_size,
        config.param_submit_timeout_ms,
        config.param_submit_event_poll_ms,
        config.param_submit_log_every
    );

    let stream_stop = Arc::new(AtomicBool::new(false));
    let event_loop_stop = Arc::new(AtomicBool::new(false));
    let event_loop_polls = Arc::new(AtomicUsize::new(0));
    let event_loop_errors = Arc::new(AtomicUsize::new(0));
    let mut lanes = Vec::new();

    for lane_idx in 0..config.param_submit_bulk_in_lanes {
        lanes.push(create_param_submit_lane(
            driver,
            config,
            Arc::clone(&stream_stop),
            ParamSubmitLaneKind::BulkIn,
            lane_idx + 1,
        )?);
    }
    for lane_idx in 0..config.param_submit_event_lanes {
        lanes.push(create_param_submit_lane(
            driver,
            config,
            Arc::clone(&stream_stop),
            ParamSubmitLaneKind::Event,
            lane_idx + 1,
        )?);
    }
    for lane_idx in 0..config.param_submit_interrupt_lanes {
        lanes.push(create_param_submit_lane(
            driver,
            config,
            Arc::clone(&stream_stop),
            ParamSubmitLaneKind::Interrupt,
            lane_idx + 1,
        )?);
    }

    let event_poll_timeout = Duration::from_millis(config.param_submit_event_poll_ms);
    let result = std::thread::scope(|scope| -> Result<(), Box<dyn Error>> {
        let event_loop_stop_for_thread = Arc::clone(&event_loop_stop);
        let event_loop_polls_for_thread = Arc::clone(&event_loop_polls);
        let event_loop_errors_for_thread = Arc::clone(&event_loop_errors);
        let event_thread = scope.spawn(move || {
            while !event_loop_stop_for_thread.load(Ordering::Relaxed) {
                event_loop_polls_for_thread.fetch_add(1, Ordering::Relaxed);
                if let Err(err) = driver.handle_events_timeout(Some(event_poll_timeout)) {
                    event_loop_errors_for_thread.fetch_add(1, Ordering::Relaxed);
                    if !matches!(err, CoralError::UsbError(rusb::Error::Timeout)) {
                        println!("      global submit event loop error: {}", err);
                    }
                }
            }
        });

        let body_result = body();

        stream_stop.store(true, Ordering::Relaxed);
        cancel_param_submit_lanes(&lanes, phase_label);

        let settle_deadline = Instant::now()
            + Duration::from_millis(config.param_submit_timeout_ms.saturating_mul(4).max(200));
        while Instant::now() < settle_deadline {
            if lanes
                .iter()
                .all(|lane| !lane.submitted.load(Ordering::Relaxed))
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }

        event_loop_stop.store(true, Ordering::Relaxed);
        let _ = event_thread.join();
        body_result
    });

    println!(
        "Global submit event loop polls={} errors={}",
        event_loop_polls.load(Ordering::Relaxed),
        event_loop_errors.load(Ordering::Relaxed)
    );
    print_param_submit_lane_stats(&lanes, phase_label);
    result
}

fn run_param_async_lane_loop(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    stop: &AtomicBool,
    kind: ParamAsyncLaneKind,
    lane_idx: usize,
) -> ParamAsyncLaneStats {
    let timeout = Duration::from_millis(config.param_async_timeout_ms);
    let mut stats = ParamAsyncLaneStats::new(kind, lane_idx);
    while !stop.load(Ordering::Relaxed) {
        stats.reads += 1;
        match kind {
            ParamAsyncLaneKind::BulkIn => {
                match driver
                    .read_output_bytes_with_timeout(config.param_async_bulk_in_size, timeout)
                {
                    Ok(_) => stats.ok += 1,
                    Err(CoralError::UsbError(rusb::Error::Timeout)) => stats.timeouts += 1,
                    Err(_) => stats.errors += 1,
                }
            }
            ParamAsyncLaneKind::Event => match driver.read_event_packet_with_timeout(timeout) {
                Ok(_) => stats.ok += 1,
                Err(CoralError::UsbError(rusb::Error::Timeout)) => stats.timeouts += 1,
                Err(_) => stats.errors += 1,
            },
            ParamAsyncLaneKind::Interrupt => {
                match driver.read_interrupt_packet_with_timeout(timeout) {
                    Ok(_) => stats.ok += 1,
                    Err(CoralError::UsbError(rusb::Error::Timeout)) => stats.timeouts += 1,
                    Err(_) => stats.errors += 1,
                }
            }
        }
    }
    stats
}

fn wait_for_param_admission(
    driver: &EdgeTpuUsbDriver,
    phase_label: &str,
    mode: ParamAdmissionWaitMode,
    chunk_idx: usize,
    param_bytes_written: usize,
    timeout: Duration,
    poll_timeout: Duration,
) -> Result<bool, Box<dyn Error>> {
    let start = Instant::now();
    let mut event_ok = false;
    let mut interrupt_ok = false;
    let mut polls = 0usize;

    loop {
        if mode.needs_event() && !event_ok {
            polls += 1;
            match driver.read_event_packet_with_timeout(poll_timeout) {
                Ok(event) => {
                    event_ok = true;
                    println!(
                        "      {}: admission event ok chunk={} offset={} tag={} event_offset=0x{:016x} length={}",
                        phase_label, chunk_idx, param_bytes_written, event.tag, event.offset, event.length
                    );
                }
                Err(CoralError::UsbError(rusb::Error::Timeout)) => {}
                Err(err) => println!(
                    "      {}: admission event read error chunk={} offset={}: {}",
                    phase_label, chunk_idx, param_bytes_written, err
                ),
            }
        }

        if mode.needs_interrupt() && !interrupt_ok {
            polls += 1;
            match driver.read_interrupt_packet_with_timeout(poll_timeout) {
                Ok(pkt) => {
                    interrupt_ok = true;
                    println!(
                        "      {}: admission interrupt ok chunk={} offset={} raw=0x{:08x} fatal={} top_level_mask=0x{:08x}",
                        phase_label,
                        chunk_idx,
                        param_bytes_written,
                        pkt.raw,
                        pkt.fatal,
                        pkt.top_level_mask
                    );
                }
                Err(CoralError::UsbError(rusb::Error::Timeout)) => {}
                Err(err) => println!(
                    "      {}: admission interrupt read error chunk={} offset={}: {}",
                    phase_label, chunk_idx, param_bytes_written, err
                ),
            }
        }

        let satisfied = match mode {
            ParamAdmissionWaitMode::Event => event_ok,
            ParamAdmissionWaitMode::Interrupt => interrupt_ok,
            ParamAdmissionWaitMode::Either => event_ok || interrupt_ok,
            ParamAdmissionWaitMode::Both => event_ok && interrupt_ok,
        };
        if satisfied {
            let elapsed_ms = start.elapsed().as_millis();
            println!(
                "      {}: admission wait satisfied mode={} chunk={} offset={} polls={} elapsed_ms={}",
                phase_label,
                mode.as_str(),
                chunk_idx,
                param_bytes_written,
                polls,
                elapsed_ms
            );
            return Ok(true);
        }

        if start.elapsed() >= timeout {
            let elapsed_ms = start.elapsed().as_millis();
            println!(
                "      {}: admission wait timeout mode={} chunk={} offset={} polls={} elapsed_ms={} event_ok={} interrupt_ok={}",
                phase_label,
                mode.as_str(),
                chunk_idx,
                param_bytes_written,
                polls,
                elapsed_ms,
                event_ok,
                interrupt_ok
            );
            return Ok(false);
        }
    }
}

fn stream_parameter_chunks(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    payload: &[u8],
    phase_label: &str,
    header_total_len: usize,
    stream_len: usize,
    stream_chunk_size: usize,
    descriptor_split_size: usize,
    poll_timeout: Duration,
    interrupt_poll_timeout: Duration,
) -> Result<(), Box<dyn Error>> {
    let mut global_offset = 0usize;
    let mut global_chunk_idx = 0usize;
    let mut descriptor_idx = 0usize;
    let mut param_bytes_written = 0usize;
    let mut gate_cursor = 0usize;
    let gate_offsets = &config.param_gate_known_good_offsets;
    let mut csr_probe_cursor = 0usize;
    let csr_probe_offsets = &config.param_csr_probe_offsets;
    let mut poison_probe_done = false;
    let window_gate = match (
        config.param_gate_window_start_bytes,
        config.param_gate_window_end_bytes,
        config.param_gate_window_step_bytes,
    ) {
        (Some(start), Some(end), Some(step)) if start < end && step > 0 => Some((start, end, step)),
        _ => None,
    };
    let mut next_window_gate_offset = window_gate.map(|(start, _, _)| start);
    let mut window_gate_count = 0usize;
    let admission_mode = config.param_admission_wait_mode;
    let admission_wait_timeout = Duration::from_millis(config.param_admission_wait_timeout_ms);
    let admission_wait_poll = Duration::from_millis(config.param_admission_wait_poll_ms);
    let admission_wait_every_chunks = config.param_admission_wait_every_chunks;
    let admission_wait_start = config.param_admission_wait_start_bytes.unwrap_or(0);
    let admission_wait_end = config.param_admission_wait_end_bytes.unwrap_or(usize::MAX);
    let submit_bulk_out_depth = config.param_submit_bulk_out_depth.max(1);
    let mut inflight_bulk_out: VecDeque<ParamBulkOutInFlight> = VecDeque::new();
    while global_offset < stream_len {
        let descriptor_end = (global_offset + descriptor_split_size).min(stream_len);
        let descriptor_len = if descriptor_idx == 0 {
            header_total_len
        } else {
            descriptor_end - global_offset
        };
        run_param_pre_ingress_handshake(driver, config, phase_label, descriptor_idx + 1);
        driver.send_descriptor_header_raw(config.parameters_tag, descriptor_len)?;

        let mut descriptor_offset = global_offset;
        while descriptor_offset < descriptor_end {
            if should_capture_param_csr_snapshot(config, global_chunk_idx, param_bytes_written) {
                capture_param_csr_snapshot(
                    driver,
                    config,
                    phase_label,
                    "pre_chunk",
                    global_chunk_idx,
                    param_bytes_written,
                );
            }
            run_pending_param_csr_probes(
                driver,
                phase_label,
                global_chunk_idx,
                param_bytes_written,
                csr_probe_offsets,
                &mut csr_probe_cursor,
            );
            run_param_poison_probe(
                driver,
                phase_label,
                global_chunk_idx,
                param_bytes_written,
                config.param_poison_probe_offset,
                &mut poison_probe_done,
            );
            if config.param_gate_placement.run_before() {
                if let Err(err) = run_pending_known_good_gates(
                    driver,
                    config,
                    phase_label,
                    param_bytes_written,
                    gate_offsets,
                    &mut gate_cursor,
                    window_gate,
                    &mut next_window_gate_offset,
                    &mut window_gate_count,
                ) {
                    cleanup_inflight_bulk_out_queue(&mut inflight_bulk_out, config, phase_label);
                    return Err(err);
                }
            }
            let end = (descriptor_offset + stream_chunk_size).min(descriptor_end);
            let chunk = &payload[descriptor_offset..end];
            let write_result = if config.param_submit_bulk_out && submit_bulk_out_depth > 1 {
                while inflight_bulk_out.len() >= submit_bulk_out_depth {
                    let completed = inflight_bulk_out
                        .pop_front()
                        .ok_or("inflight bulk_out queue bookkeeping failure")?;
                    if let Err(err) =
                        finalize_param_bulk_out_inflight(&completed, config, phase_label)
                    {
                        cleanup_inflight_bulk_out_queue(
                            &mut inflight_bulk_out,
                            config,
                            phase_label,
                        );
                        return Err(err);
                    }
                }
                let inflight = submit_param_bulk_out_inflight(
                    driver,
                    config,
                    phase_label,
                    global_chunk_idx,
                    descriptor_offset,
                    chunk,
                )?;
                inflight_bulk_out.push_back(inflight);
                Ok(())
            } else if config.param_submit_bulk_out {
                submit_param_bulk_out_chunk(
                    driver,
                    config,
                    phase_label,
                    global_chunk_idx,
                    descriptor_offset,
                    chunk,
                )
            } else {
                driver
                    .write_bulk_out_chunk(chunk)
                    .map_err(|err| -> Box<dyn Error> { Box::new(err) })
            };
            if let Err(err) = write_result {
                if config.param_csr_snapshot_on_error {
                    capture_param_csr_snapshot(
                        driver,
                        config,
                        phase_label,
                        "stream_write_error",
                        global_chunk_idx,
                        param_bytes_written,
                    );
                }
                cleanup_inflight_bulk_out_queue(&mut inflight_bulk_out, config, phase_label);
                return Err(format!(
                    "{}: parameter stream write failed at offset {} of {} bytes (chunk {}): {}",
                    phase_label, descriptor_offset, stream_len, global_chunk_idx, err
                )
                .into());
            }
            param_bytes_written += chunk.len();
            descriptor_offset = end;
            global_chunk_idx += 1;

            run_pending_param_csr_probes(
                driver,
                phase_label,
                global_chunk_idx,
                param_bytes_written,
                csr_probe_offsets,
                &mut csr_probe_cursor,
            );
            run_param_poison_probe(
                driver,
                phase_label,
                global_chunk_idx,
                param_bytes_written,
                config.param_poison_probe_offset,
                &mut poison_probe_done,
            );

            if config.param_gate_placement.run_after() {
                if let Err(err) = run_pending_known_good_gates(
                    driver,
                    config,
                    phase_label,
                    param_bytes_written,
                    gate_offsets,
                    &mut gate_cursor,
                    window_gate,
                    &mut next_window_gate_offset,
                    &mut window_gate_count,
                ) {
                    cleanup_inflight_bulk_out_queue(&mut inflight_bulk_out, config, phase_label);
                    return Err(err);
                }
            }

            if let Some(mode) = admission_mode {
                let in_admission_window = param_bytes_written >= admission_wait_start
                    && param_bytes_written < admission_wait_end;
                if in_admission_window && (global_chunk_idx % admission_wait_every_chunks == 0) {
                    let admission_ok = match wait_for_param_admission(
                        driver,
                        phase_label,
                        mode,
                        global_chunk_idx,
                        param_bytes_written,
                        admission_wait_timeout,
                        admission_wait_poll,
                    ) {
                        Ok(ok) => ok,
                        Err(err) => {
                            cleanup_inflight_bulk_out_queue(
                                &mut inflight_bulk_out,
                                config,
                                phase_label,
                            );
                            return Err(format!(
                                "{}: admission wait call failed at offset {} (chunk {}): {}",
                                phase_label, param_bytes_written, global_chunk_idx, err
                            )
                            .into());
                        }
                    };
                    if !admission_ok {
                        if config.param_admission_wait_strict {
                            cleanup_inflight_bulk_out_queue(
                                &mut inflight_bulk_out,
                                config,
                                phase_label,
                            );
                            return Err(format!(
                                "{}: admission wait unsatisfied at offset {} (chunk {})",
                                phase_label, param_bytes_written, global_chunk_idx
                            )
                            .into());
                        }
                        println!(
                            "      {}: admission wait unsatisfied at offset {} (chunk {}), continuing (strict=false)",
                            phase_label, param_bytes_written, global_chunk_idx
                        );
                    }
                }
            }

            if config.param_read_event_every > 0
                && (global_chunk_idx % config.param_read_event_every == 0)
            {
                match driver.read_event_packet_with_timeout(poll_timeout) {
                    Ok(event) => println!(
                        "      stream poll event: tag={} offset=0x{:016x} length={}",
                        event.tag, event.offset, event.length
                    ),
                    Err(CoralError::UsbError(rusb::Error::Timeout)) => {}
                    Err(err) => println!("      stream poll event error: {}", err),
                }
            }
            if config.param_read_interrupt_every > 0
                && (global_chunk_idx % config.param_read_interrupt_every == 0)
            {
                match driver.read_interrupt_packet_with_timeout(interrupt_poll_timeout) {
                    Ok(pkt) => println!(
                        "      stream poll interrupt: raw=0x{:08x} fatal={} top_level_mask=0x{:08x}",
                        pkt.raw, pkt.fatal, pkt.top_level_mask
                    ),
                    Err(CoralError::UsbError(rusb::Error::Timeout)) => {}
                    Err(err) => println!("      stream poll interrupt error: {}", err),
                }
            }

            if config.param_write_sleep_us > 0 {
                std::thread::sleep(Duration::from_micros(config.param_write_sleep_us));
            }
        }

        descriptor_idx += 1;
        global_offset = descriptor_end;
        if config.param_drain_event_every_descriptors > 0
            && (descriptor_idx % config.param_drain_event_every_descriptors == 0)
        {
            match driver.read_event_packet_with_timeout(poll_timeout) {
                Ok(event) => println!(
                    "      descriptor drain event: tag={} offset=0x{:016x} length={}",
                    event.tag, event.offset, event.length
                ),
                Err(CoralError::UsbError(rusb::Error::Timeout)) => {}
                Err(err) => println!("      descriptor drain event error: {}", err),
            }
        }
    }

    while let Some(completed) = inflight_bulk_out.pop_front() {
        if let Err(err) = finalize_param_bulk_out_inflight(&completed, config, phase_label) {
            cleanup_inflight_bulk_out_queue(&mut inflight_bulk_out, config, phase_label);
            return Err(format!(
                "{}: pending inflight bulk_out finalize failed after stream completion: {}",
                phase_label, err
            )
            .into());
        }
    }

    Ok(())
}

fn stream_parameter_chunks_with_optional_async_lanes(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    payload: &[u8],
    phase_label: &str,
    header_total_len: usize,
    stream_len: usize,
    stream_chunk_size: usize,
    descriptor_split_size: usize,
    poll_timeout: Duration,
    interrupt_poll_timeout: Duration,
) -> Result<(), Box<dyn Error>> {
    if has_param_submit_lanes(config) {
        return stream_parameter_chunks_with_submit_lanes(
            driver,
            config,
            payload,
            phase_label,
            header_total_len,
            stream_len,
            stream_chunk_size,
            descriptor_split_size,
            poll_timeout,
            interrupt_poll_timeout,
        );
    }

    if !has_param_async_lanes(config) {
        return stream_parameter_chunks(
            driver,
            config,
            payload,
            phase_label,
            header_total_len,
            stream_len,
            stream_chunk_size,
            descriptor_split_size,
            poll_timeout,
            interrupt_poll_timeout,
        );
    }

    println!(
        "      {}: async read lanes bulk_in={} (size={}) event={} interrupt={} timeout_ms={}",
        phase_label,
        config.param_async_bulk_in_lanes,
        config.param_async_bulk_in_size,
        config.param_async_event_lanes,
        config.param_async_interrupt_lanes,
        config.param_async_timeout_ms
    );

    let stop = Arc::new(AtomicBool::new(false));
    let mut lane_stats: Vec<ParamAsyncLaneStats> = Vec::new();
    let result = std::thread::scope(|scope| -> Result<(), Box<dyn Error>> {
        let mut handles = Vec::new();
        for lane_idx in 0..config.param_async_bulk_in_lanes {
            let stop = Arc::clone(&stop);
            handles.push(scope.spawn(move || {
                run_param_async_lane_loop(
                    driver,
                    config,
                    stop.as_ref(),
                    ParamAsyncLaneKind::BulkIn,
                    lane_idx + 1,
                )
            }));
        }
        for lane_idx in 0..config.param_async_event_lanes {
            let stop = Arc::clone(&stop);
            handles.push(scope.spawn(move || {
                run_param_async_lane_loop(
                    driver,
                    config,
                    stop.as_ref(),
                    ParamAsyncLaneKind::Event,
                    lane_idx + 1,
                )
            }));
        }
        for lane_idx in 0..config.param_async_interrupt_lanes {
            let stop = Arc::clone(&stop);
            handles.push(scope.spawn(move || {
                run_param_async_lane_loop(
                    driver,
                    config,
                    stop.as_ref(),
                    ParamAsyncLaneKind::Interrupt,
                    lane_idx + 1,
                )
            }));
        }

        let stream_result = stream_parameter_chunks(
            driver,
            config,
            payload,
            phase_label,
            header_total_len,
            stream_len,
            stream_chunk_size,
            descriptor_split_size,
            poll_timeout,
            interrupt_poll_timeout,
        );

        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            match handle.join() {
                Ok(stats) => lane_stats.push(stats),
                Err(_) => println!("      {}: async lane thread panicked", phase_label),
            }
        }

        stream_result
    });

    for stats in lane_stats {
        println!(
            "      {}: async lane {}#{} reads={} ok={} timeouts={} errors={}",
            phase_label,
            stats.kind.as_str(),
            stats.lane_idx,
            stats.reads,
            stats.ok,
            stats.timeouts,
            stats.errors
        );
    }

    result
}

fn send_parameter_payload_with_instruction_interleave(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    instruction_patch_spec: Option<&InstructionPatchSpec>,
    payload: &[u8],
    phase_label: &str,
    interleave_window_bytes: usize,
    instruction_chunks: &[Vec<u8>],
) -> Result<(), Box<dyn Error>> {
    if payload.is_empty() {
        return Ok(());
    }
    if instruction_chunks.is_empty() {
        return Err(format!(
            "{}: interleave requested but no instruction chunks are available",
            phase_label
        )
        .into());
    }

    let mut segment_start = 0usize;
    let mut segment_idx = 0usize;
    while segment_start < payload.len() {
        let segment_end = (segment_start + interleave_window_bytes).min(payload.len());
        let segment_label = format!(
            "{} interleave-seg{} [{}..{})",
            phase_label,
            segment_idx + 1,
            segment_start,
            segment_end
        );
        send_parameter_payload(
            driver,
            config,
            &payload[segment_start..segment_end],
            &segment_label,
        )?;
        segment_start = segment_end;
        segment_idx += 1;

        if segment_start >= payload.len() {
            break;
        }

        let instr_idx = (segment_idx - 1) % instruction_chunks.len();
        let instr_chunk = &instruction_chunks[instr_idx];
        println!(
            "    {}: interleave instruction chunk {} ({} bytes) after param offset={}",
            phase_label,
            instr_idx,
            instr_chunk.len(),
            segment_start
        );
        send_instruction_chunk(
            driver,
            config,
            instruction_patch_spec,
            &format!("{phase_label} interleave"),
            instr_idx,
            instr_chunk,
        )?;

        if config.param_interleave_require_event {
            let timeout = Duration::from_millis(config.param_interleave_event_timeout_ms);
            match driver.read_event_packet_with_timeout(timeout) {
                Ok(event) => println!(
                    "    {}: interleave post-instr event: tag={} offset=0x{:016x} length={}",
                    phase_label, event.tag, event.offset, event.length
                ),
                Err(err) => {
                    return Err(format!(
                        "{}: interleave required event failed after segment {}: {}",
                        phase_label, segment_idx, err
                    )
                    .into())
                }
            }
        }
    }

    Ok(())
}

fn send_parameter_payload(
    driver: &EdgeTpuUsbDriver,
    config: &Config,
    payload: &[u8],
    phase_label: &str,
) -> Result<(), Box<dyn Error>> {
    if payload.is_empty() {
        return Ok(());
    }

    let use_custom_stream = config.param_stream_chunk_size.is_some()
        || config.param_stream_max_bytes.is_some()
        || config.param_read_event_every > 0
        || config.param_read_interrupt_every > 0
        || config.param_write_sleep_us > 0
        || config.param_descriptor_split_bytes.is_some()
        || config.param_drain_event_every_descriptors > 0
        || config.param_a0d8_handshake
        || config.param_prepost_bulk_in_reads > 0
        || config.param_prepost_event_reads > 0
        || config.param_prepost_interrupt_reads > 0
        || has_param_async_lanes(config)
        || has_param_submit_lanes(config)
        || has_param_known_good_gates(config)
        || has_param_csr_snapshot(config)
        || has_param_csr_probes(config)
        || has_param_poison_probe(config)
        || has_param_admission_wait(config);
    if !use_custom_stream {
        driver.send_descriptor_payload_raw(config.parameters_tag, payload)?;
        return Ok(());
    }

    let (header_total_len, stream_len) = header_and_stream_len(
        payload.len(),
        config.param_stream_max_bytes,
        config.param_force_full_header_len,
    );
    let stream_chunk_size = config.param_stream_chunk_size.unwrap_or(config.chunk_size);
    let descriptor_split_size = config
        .param_descriptor_split_bytes
        .unwrap_or(stream_len.max(1));
    let poll_timeout = Duration::from_millis(config.param_event_timeout_ms);
    let interrupt_poll_timeout = Duration::from_millis(config.param_interrupt_timeout_ms);

    println!(
        "    {}: streaming parameters len={} header_len={} chunk={} desc_split={} event_poll_every={} intr_poll_every={} drain_desc_every={} sleep_us={} tag={} handshake={} a0d8_write=0x{:08x} gate_offsets={:?} gate_window_start={:?} gate_window_end={:?} gate_window_step={:?} gate_placement={} csr_snapshot_start={:?} csr_snapshot_end={:?} csr_snapshot_every_chunks={} csr_snapshot_on_error={} admission_mode={} admission_timeout_ms={} admission_poll_ms={} admission_start={:?} admission_end={:?} admission_every_chunks={} admission_strict={} prepost_bulk_reads={} prepost_bulk_size={} prepost_event_reads={} prepost_intr_reads={} prepost_timeout_ms={} async_bulk_lanes={} async_bulk_size={} async_event_lanes={} async_intr_lanes={} async_timeout_ms={} submit_bulk_out={} submit_bulk_out_depth={} submit_bulk_lanes={} submit_event_lanes={} submit_intr_lanes={} submit_buf_size={} submit_timeout_ms={} submit_event_poll_ms={} submit_log_every={} interleave_window={:?} interleave_require_event={} interleave_event_timeout_ms={} csr_probe_offsets={:?} poison_probe_offset={:?}",
        phase_label,
        stream_len,
        header_total_len,
        stream_chunk_size,
        descriptor_split_size,
        config.param_read_event_every,
        config.param_read_interrupt_every,
        config.param_drain_event_every_descriptors,
        config.param_write_sleep_us,
        config.parameters_tag,
        config.param_a0d8_handshake,
        config.param_a0d8_write_value,
        config.param_gate_known_good_offsets,
        config.param_gate_window_start_bytes,
        config.param_gate_window_end_bytes,
        config.param_gate_window_step_bytes,
        config.param_gate_placement.as_str(),
        config.param_csr_snapshot_start_bytes,
        config.param_csr_snapshot_end_bytes,
        config.param_csr_snapshot_every_chunks,
        config.param_csr_snapshot_on_error,
        config
            .param_admission_wait_mode
            .map(|mode| mode.as_str())
            .unwrap_or("off"),
        config.param_admission_wait_timeout_ms,
        config.param_admission_wait_poll_ms,
        config.param_admission_wait_start_bytes,
        config.param_admission_wait_end_bytes,
        config.param_admission_wait_every_chunks,
        config.param_admission_wait_strict,
        config.param_prepost_bulk_in_reads,
        config.param_prepost_bulk_in_size,
        config.param_prepost_event_reads,
        config.param_prepost_interrupt_reads,
        config.param_prepost_timeout_ms,
        config.param_async_bulk_in_lanes,
        config.param_async_bulk_in_size,
        config.param_async_event_lanes,
        config.param_async_interrupt_lanes,
        config.param_async_timeout_ms,
        config.param_submit_bulk_out,
        config.param_submit_bulk_out_depth,
        config.param_submit_bulk_in_lanes,
        config.param_submit_event_lanes,
        config.param_submit_interrupt_lanes,
        config.param_submit_buffer_size,
        config.param_submit_timeout_ms,
        config.param_submit_event_poll_ms,
        config.param_submit_log_every,
        config.param_interleave_window_bytes,
        config.param_interleave_require_event,
        config.param_interleave_event_timeout_ms,
        config.param_csr_probe_offsets,
        config.param_poison_probe_offset
    );

    stream_parameter_chunks_with_optional_async_lanes(
        driver,
        config,
        payload,
        phase_label,
        header_total_len,
        stream_len,
        stream_chunk_size,
        descriptor_split_size,
        poll_timeout,
        interrupt_poll_timeout,
    )
}

fn describe_executable(exe: &SerializedExecutableBlob) -> String {
    let param = match exe.parameter_region {
        Some((start, end)) => format!("param_region={}..{} ({} bytes)", start, end, end - start),
        None => "param_region=none".to_string(),
    };
    format!(
        "pkg={} exec={} type={}({}) payload={} bytes instr_chunks={} param_stream={} {}",
        exe.package_index,
        exe.executable_index,
        exe.executable_type,
        executable_type_name(exe.executable_type),
        exe.payload.len(),
        exe.instruction_bitstreams.len(),
        exe.parameters_stream.len(),
        param
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut config = parse_args()?;
    apply_script_defaults(&mut config);

    let resolved_profile = resolve_family_profile(&mut config)?;
    let family_profile_loaded = resolved_profile.loaded;
    let profile_instruction_patch_paths = resolved_profile.instruction_patch_paths;

    if config.model_path.is_empty() {
        return Err(
            "model path unresolved: provide --model or --family-profile with anchor_model".into(),
        );
    }

    let model_path = PathBuf::from(&config.model_path);
    if !model_path.is_file() {
        return Err(format!(
            "model path does not exist or is not a file: {}",
            model_path.display()
        )
        .into());
    }

    if let Some(spec_path) = config.instruction_patch_spec.as_ref() {
        if !Path::new(spec_path).is_file() {
            return Err(format!(
                "instruction patch spec path does not exist or is not a file: {}",
                spec_path
            )
            .into());
        }
    }
    for spec_path in &profile_instruction_patch_paths {
        if !Path::new(spec_path).is_file() {
            return Err(format!(
                "family-profile instruction patch path does not exist or is not a file: {}",
                spec_path
            )
            .into());
        }
    }

    let model_bytes = std::fs::read(&model_path)?;
    let mut executables = extract_serialized_executables_from_tflite(&model_bytes)?;

    println!("Model: {}", model_path.display());
    println!(
        "Descriptor tags: instr={}({}), params={}({}), input={}({})",
        config.instructions_tag,
        descriptor_tag_name(config.instructions_tag),
        config.parameters_tag,
        descriptor_tag_name(config.parameters_tag),
        config.input_activations_tag,
        descriptor_tag_name(config.input_activations_tag)
    );

    let generated_param_override = if let Some((_, profile)) = family_profile_loaded.as_ref() {
        maybe_generate_param_stream_from_weights(&config, profile)?
    } else {
        None
    };

    if let Some(path) = config.param_stream_override_file.as_ref() {
        let override_bytes = std::fs::read(path)?;
        apply_param_stream_override_bytes(
            &mut executables,
            &format!("file:{}", path),
            &override_bytes,
        )?;
    } else if let Some((override_bytes, source_label)) = generated_param_override {
        apply_param_stream_override_bytes(&mut executables, &source_label, &override_bytes)?;
    }
    if config.param_stream_chunk_size.is_some()
        || config.param_stream_max_bytes.is_some()
        || config.param_read_event_every > 0
        || config.param_read_interrupt_every > 0
        || config.param_write_sleep_us > 0
        || config.param_descriptor_split_bytes.is_some()
        || config.param_drain_event_every_descriptors > 0
        || config.param_a0d8_handshake
        || config.param_prepost_bulk_in_reads > 0
        || config.param_prepost_event_reads > 0
        || config.param_prepost_interrupt_reads > 0
        || has_param_async_lanes(&config)
        || has_param_submit_lanes(&config)
        || has_param_known_good_gates(&config)
        || has_param_csr_snapshot(&config)
        || has_param_csr_probes(&config)
        || has_param_poison_probe(&config)
        || has_param_admission_wait(&config)
    {
        println!(
            "Parameter stream controls: chunk={:?} max_bytes={:?} force_full_header_len={} desc_split={:?} event_poll_every={} intr_poll_every={} drain_desc_every={} event_timeout_ms={} intr_timeout_ms={} sleep_us={} handshake={} a0d8_write=0x{:08x} gate_offsets={:?} gate_window_start={:?} gate_window_end={:?} gate_window_step={:?} gate_placement={} csr_snapshot_start={:?} csr_snapshot_end={:?} csr_snapshot_every_chunks={} csr_snapshot_on_error={} admission_mode={} admission_timeout_ms={} admission_poll_ms={} admission_start={:?} admission_end={:?} admission_every_chunks={} admission_strict={} prepost_bulk_reads={} prepost_bulk_size={} prepost_event_reads={} prepost_intr_reads={} prepost_timeout_ms={} async_bulk_lanes={} async_bulk_size={} async_event_lanes={} async_intr_lanes={} async_timeout_ms={} submit_bulk_out={} submit_bulk_out_depth={} submit_bulk_lanes={} submit_event_lanes={} submit_intr_lanes={} submit_buf_size={} submit_timeout_ms={} submit_event_poll_ms={} submit_log_every={} submit_global_lanes={} bootstrap_known_good_order={} require_post_instr_event={} post_instr_event_timeout_ms={} interleave_window={:?} interleave_require_event={} interleave_event_timeout_ms={} csr_probe_offsets={:?} poison_probe_offset={:?} script1={} script2={} script3={}",
            config.param_stream_chunk_size,
            config.param_stream_max_bytes,
            config.param_force_full_header_len,
            config.param_descriptor_split_bytes,
            config.param_read_event_every,
            config.param_read_interrupt_every,
            config.param_drain_event_every_descriptors,
            config.param_event_timeout_ms,
            config.param_interrupt_timeout_ms,
            config.param_write_sleep_us,
            config.param_a0d8_handshake,
            config.param_a0d8_write_value,
            config.param_gate_known_good_offsets,
            config.param_gate_window_start_bytes,
            config.param_gate_window_end_bytes,
            config.param_gate_window_step_bytes,
            config.param_gate_placement.as_str(),
            config.param_csr_snapshot_start_bytes,
            config.param_csr_snapshot_end_bytes,
            config.param_csr_snapshot_every_chunks,
            config.param_csr_snapshot_on_error,
            config
                .param_admission_wait_mode
                .map(|mode| mode.as_str())
                .unwrap_or("off"),
            config.param_admission_wait_timeout_ms,
            config.param_admission_wait_poll_ms,
            config.param_admission_wait_start_bytes,
            config.param_admission_wait_end_bytes,
            config.param_admission_wait_every_chunks,
            config.param_admission_wait_strict,
            config.param_prepost_bulk_in_reads,
            config.param_prepost_bulk_in_size,
            config.param_prepost_event_reads,
            config.param_prepost_interrupt_reads,
            config.param_prepost_timeout_ms,
            config.param_async_bulk_in_lanes,
            config.param_async_bulk_in_size,
            config.param_async_event_lanes,
            config.param_async_interrupt_lanes,
            config.param_async_timeout_ms,
            config.param_submit_bulk_out,
            config.param_submit_bulk_out_depth,
            config.param_submit_bulk_in_lanes,
            config.param_submit_event_lanes,
            config.param_submit_interrupt_lanes,
            config.param_submit_buffer_size,
            config.param_submit_timeout_ms,
            config.param_submit_event_poll_ms,
            config.param_submit_log_every,
            config.param_submit_global_lanes,
            config.bootstrap_known_good_order,
            config.param_require_post_instr_event,
            config.param_post_instr_event_timeout_ms,
            config.param_interleave_window_bytes,
            config.param_interleave_require_event,
            config.param_interleave_event_timeout_ms,
            config.param_csr_probe_offsets,
            config.param_poison_probe_offset,
            config.script1_interleave,
            config.script2_queue_probe,
            config.script3_poison_diff
        );
    }
    println!("Extracted executables: {}", executables.len());
    for exe in &executables {
        println!("  {}", describe_executable(exe));
    }

    let param_executables: Vec<&SerializedExecutableBlob> = executables
        .iter()
        .filter(|exe| exe.executable_type == 1)
        .collect();
    let run_exe = choose_execution_executable(&executables, config.exec_index)?;

    let input_batches = load_inputs(&config)?;

    let mut instruction_patch_sources: Vec<String> = Vec::new();
    if let Some(spec_path) = config.instruction_patch_spec.as_ref() {
        instruction_patch_sources.push(spec_path.clone());
    }
    instruction_patch_sources.extend(profile_instruction_patch_paths.clone());

    let instruction_patch_spec = if !instruction_patch_sources.is_empty() {
        for spec_path in &instruction_patch_sources {
            let spec = load_instruction_patch_spec(spec_path)?;
            println!(
                "Instruction patch source: path={} payload_lens={} rule_count={}",
                spec_path,
                spec.by_payload_len.len(),
                spec.rule_count()
            );
            for (payload_len, entries) in &spec.by_payload_len {
                let first = entries.first().copied();
                let last = entries.last().copied();
                println!(
                    "  source patch len={} entries={} first={:?} last={:?}",
                    payload_len,
                    entries.len(),
                    first,
                    last
                );
            }
        }

        let merged = merge_instruction_patch_specs(&instruction_patch_sources)?;
        println!(
            "Instruction patch merged: sources={} payload_lens={} rule_count={}",
            instruction_patch_sources.len(),
            merged.by_payload_len.len(),
            merged.rule_count()
        );
        for (payload_len, entries) in &merged.by_payload_len {
            let first = entries.first().copied();
            let last = entries.last().copied();
            println!(
                "  merged patch len={} entries={} first={:?} last={:?}",
                payload_len,
                entries.len(),
                first,
                last
            );
        }
        validate_instruction_patch_spec_against_executables(&merged, &executables)?;
        println!(
            "Instruction patch compatibility: payload lengths validated against extracted executable chunks"
        );
        Some(merged)
    } else {
        None
    };

    if config.check_profile {
        println!("Profile check mode: PASS (no USB operations executed; --check-profile)");
        return Ok(());
    }

    let mut driver =
        EdgeTpuUsbDriver::open_first_prefer_runtime(Duration::from_millis(config.timeout_ms))?;
    driver.set_descriptor_chunk_size(config.chunk_size)?;

    let mut info = driver.device_info();
    println!(
        "Using USB device: bus={} addr={} id={:04x}:{:04x} runtime={}",
        info.bus,
        info.address,
        info.vendor_id,
        info.product_id,
        info.is_runtime()
    );

    if !info.is_runtime() {
        let firmware_path = config
            .firmware_path
            .as_ref()
            .ok_or("device is in boot mode (1a6e:089a); provide --firmware PATH")?;
        let firmware = std::fs::read(firmware_path)?;
        println!(
            "Boot-mode device detected; uploading firmware from {} ({} bytes)",
            firmware_path,
            firmware.len()
        );
        driver.upload_firmware_single_ep(&firmware)?;
        let _ = driver.reset_device();
        drop(driver);
        std::thread::sleep(Duration::from_secs(3));
        driver =
            EdgeTpuUsbDriver::open_first_prefer_runtime(Duration::from_millis(config.timeout_ms))?;
        driver.set_descriptor_chunk_size(config.chunk_size)?;
        info = driver.device_info();
        println!(
            "Post-firmware device: bus={} addr={} id={:04x}:{:04x} runtime={}",
            info.bus,
            info.address,
            info.vendor_id,
            info.product_id,
            info.is_runtime()
        );
    }

    if config.reset_before_claim {
        println!(
            "Reset-before-claim enabled; issuing reset, sleeping {} ms, and reopening",
            config.post_reset_sleep_ms
        );
        let _ = driver.reset_device();
        drop(driver);
        std::thread::sleep(Duration::from_millis(config.post_reset_sleep_ms));
        driver =
            EdgeTpuUsbDriver::open_first_prefer_runtime(Duration::from_millis(config.timeout_ms))?;
        driver.set_descriptor_chunk_size(config.chunk_size)?;
        info = driver.device_info();
        println!(
            "Post-reset device: bus={} addr={} id={:04x}:{:04x} runtime={}",
            info.bus,
            info.address,
            info.vendor_id,
            info.product_id,
            info.is_runtime()
        );
    }

    let _ = driver.set_configuration_1();
    driver.claim_interface0()?;

    if config.setup {
        let base_sequence = if config.setup_libedgetpu {
            LIBEDGETPU_KNOWN_GOOD_SETUP_SEQUENCE
        } else {
            EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE
        };
        let setup_steps: Vec<_> = if config.setup_include_reads {
            base_sequence.to_vec()
        } else {
            base_sequence
                .iter()
                .copied()
                .filter(|step| step.direction == VendorDirection::Out)
                .collect()
        };
        let seq_name = if config.setup_libedgetpu {
            "libedgetpu known-good"
        } else {
            "edgetpuxray"
        };
        println!(
            "Applying {} runtime setup sequence ({} steps, include_reads={}, verify_reads={})",
            seq_name,
            setup_steps.len(),
            config.setup_include_reads,
            config.verify_setup_reads
        );
        driver.apply_vendor_steps(&setup_steps, config.verify_setup_reads)?;
    } else {
        println!("Skipping setup sequence (--skip-setup)");
    }

    let mut replay_config = config.clone();
    if config.param_submit_global_lanes {
        // Global lane mode preposts EP1/EP2/EP3 submit lanes before any Bo write.
        // Per-parameter streaming still controls Bo submit behavior, so disable
        // per-parameter read lane startup to avoid duplicate readers.
        replay_config.param_submit_bulk_in_lanes = 0;
        replay_config.param_submit_event_lanes = 0;
        replay_config.param_submit_interrupt_lanes = 0;
    }

    let run_replay = || -> Result<(), Box<dyn Error>> {
        let config = &replay_config;
        if !config.skip_param_preload {
            if !param_executables.is_empty() && run_exe.executable_type == 2 {
                if config.bootstrap_known_good_order {
                    println!(
                        "Bootstrap phase (known-good order): send PARAMETER_CACHING streams before EXECUTION_ONLY run chunks"
                    );
                } else {
                    println!(
                        "Bootstrap phase: send EXECUTION_ONLY instruction chunks, then PARAMETER_CACHING streams"
                    );
                    for (idx, chunk) in run_exe.instruction_bitstreams.iter().enumerate() {
                        println!("  EXECUTION_ONLY chunk {} ({} bytes)", idx, chunk.len());
                        send_instruction_chunk(
                            &driver,
                            config,
                            instruction_patch_spec.as_ref(),
                            "bootstrap exec_only",
                            idx,
                            chunk,
                        )?;
                    }
                }
                for exe in &param_executables {
                    println!(
                        "  PARAMETER_CACHING executable idx={} chunks={} params={} bytes",
                        exe.executable_index,
                        exe.instruction_bitstreams.len(),
                        exe.parameters_stream.len()
                    );
                    for (idx, chunk) in exe.instruction_bitstreams.iter().enumerate() {
                        println!("    instr chunk {} ({} bytes)", idx, chunk.len());
                        send_instruction_chunk(
                            &driver,
                            config,
                            instruction_patch_spec.as_ref(),
                            "bootstrap param_caching",
                            idx,
                            chunk,
                        )?;
                    }
                    if config.param_require_post_instr_event {
                        let timeout =
                            Duration::from_millis(config.param_post_instr_event_timeout_ms);
                        match driver.read_event_packet_with_timeout(timeout) {
                            Ok(event) => println!(
                                "    post-instr event (required): tag={} offset=0x{:016x} length={}",
                                event.tag, event.offset, event.length
                            ),
                            Err(err) => {
                                return Err(format!(
                                    "required post-instr event failed before parameter stream (exe idx={}): {}",
                                    exe.executable_index, err
                                )
                                .into())
                            }
                        }
                    }
                    if let Some(window_bytes) = config.param_interleave_window_bytes {
                        let interleave_chunks: &[Vec<u8>] =
                            if !exe.instruction_bitstreams.is_empty() {
                                &exe.instruction_bitstreams
                            } else {
                                &run_exe.instruction_bitstreams
                            };
                        send_parameter_payload_with_instruction_interleave(
                            &driver,
                            config,
                            instruction_patch_spec.as_ref(),
                            &exe.parameters_stream,
                            "bootstrap param",
                            window_bytes,
                            interleave_chunks,
                        )?;
                    } else {
                        send_parameter_payload(
                            &driver,
                            config,
                            &exe.parameters_stream,
                            "bootstrap param",
                        )?;
                    }
                }
                if config.bootstrap_known_good_order {
                    println!(
                        "  Bootstrap event read skipped (known-good order defers EXECUTION_ONLY+input to run phase)"
                    );
                } else {
                    match driver.read_event_packet() {
                        Ok(event) => println!(
                            "  Bootstrap event: tag={} offset=0x{:016x} length={}",
                            event.tag, event.offset, event.length
                        ),
                        Err(err) => println!("  Bootstrap event read failed: {}", err),
                    }
                }
            } else {
                for exe in &param_executables {
                    println!(
                        "Preload PARAMETER_CACHING executable idx={} chunks={} params={} bytes",
                        exe.executable_index,
                        exe.instruction_bitstreams.len(),
                        exe.parameters_stream.len()
                    );
                    for (idx, chunk) in exe.instruction_bitstreams.iter().enumerate() {
                        send_instruction_chunk(
                            &driver,
                            config,
                            instruction_patch_spec.as_ref(),
                            "preload param_caching",
                            idx,
                            chunk,
                        )?;
                    }
                    if config.param_require_post_instr_event {
                        let timeout =
                            Duration::from_millis(config.param_post_instr_event_timeout_ms);
                        match driver.read_event_packet_with_timeout(timeout) {
                            Ok(event) => println!(
                                "  post-instr event (required): tag={} offset=0x{:016x} length={}",
                                event.tag, event.offset, event.length
                            ),
                            Err(err) => {
                                return Err(format!(
                                    "required post-instr event failed before parameter stream (exe idx={}): {}",
                                    exe.executable_index, err
                                )
                                .into())
                            }
                        }
                    }
                    if let Some(window_bytes) = config.param_interleave_window_bytes {
                        let interleave_chunks: &[Vec<u8>] =
                            if !exe.instruction_bitstreams.is_empty() {
                                &exe.instruction_bitstreams
                            } else {
                                &run_exe.instruction_bitstreams
                            };
                        send_parameter_payload_with_instruction_interleave(
                            &driver,
                            config,
                            instruction_patch_spec.as_ref(),
                            &exe.parameters_stream,
                            "preload param",
                            window_bytes,
                            interleave_chunks,
                        )?;
                    } else {
                        send_parameter_payload(
                            &driver,
                            config,
                            &exe.parameters_stream,
                            "preload param",
                        )?;
                    }
                }
            }
        } else {
            println!("Skipping parameter preload (--skip-param-preload)");
        }

        println!(
            "Execution executable: idx={} type={}({}) payload={} bytes",
            run_exe.executable_index,
            run_exe.executable_type,
            executable_type_name(run_exe.executable_type),
            run_exe.payload.len()
        );

        let mut run_ms_values = Vec::with_capacity(config.runs);
        for run in 0..config.runs {
            println!("RUN {}", run + 1);
            let run_started = Instant::now();
            for (idx, chunk) in run_exe.instruction_bitstreams.iter().enumerate() {
                println!("  run instr chunk {} ({} bytes)", idx, chunk.len());
                send_instruction_chunk(
                    &driver,
                    config,
                    instruction_patch_spec.as_ref(),
                    "run",
                    idx,
                    chunk,
                )?;
            }
            if !run_exe.parameters_stream.is_empty() {
                send_parameter_payload(&driver, config, &run_exe.parameters_stream, "run param")?;
            }
            let input_bytes = &input_batches[run];
            let input_hash = fnv1a64(input_bytes);
            println!(
                "  Input: bytes={} fnv1a64=0x{:016x} head={:02x?}",
                input_bytes.len(),
                input_hash,
                &input_bytes[..input_bytes.len().min(16)]
            );
            driver.send_descriptor_payload_raw(config.input_activations_tag, input_bytes)?;

            match driver.read_event_packet() {
                Ok(event) => println!(
                    "  Event: tag={} offset=0x{:016x} length={}",
                    event.tag, event.offset, event.length
                ),
                Err(err) => println!("  Event read failed: {}", err),
            }

            let output = driver.read_output_bytes(config.output_bytes)?;
            let head_len = output.len().min(16);
            let output_hash = fnv1a64(&output);
            println!(
                "  Output: bytes={} fnv1a64=0x{:016x} head={:02x?}",
                output.len(),
                output_hash,
                &output[..head_len]
            );

            if config.read_interrupt {
                match driver.read_interrupt_packet() {
                    Ok(pkt) => println!(
                        "  Interrupt: raw=0x{:08x} fatal={} top_level_mask=0x{:08x}",
                        pkt.raw, pkt.fatal, pkt.top_level_mask
                    ),
                    Err(err) => println!("  Interrupt read failed: {}", err),
                }
            }
            let run_ms = run_started.elapsed().as_secs_f64() * 1000.0;
            run_ms_values.push(run_ms);
            println!("  Run timing: run_ms={:.3}", run_ms);
        }
        if !run_ms_values.is_empty() {
            let sum: f64 = run_ms_values.iter().sum();
            let avg = sum / run_ms_values.len() as f64;
            let min = run_ms_values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = run_ms_values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            println!(
                "Run timing summary: runs={} avg_ms={:.3} min_ms={:.3} max_ms={:.3}",
                run_ms_values.len(),
                avg,
                min,
                max
            );
        }

        Ok(())
    };

    if config.param_submit_global_lanes {
        run_with_global_submit_lanes(&driver, &config, "global submit", run_replay)?;
    } else {
        run_replay()?;
    }

    Ok(())
}
