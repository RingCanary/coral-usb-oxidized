use coral_usb_oxidized::{version, CoralDevice, DenseGemmTemplate};
use rusb::{Context, Device, DeviceDescriptor, Direction, Recipient, RequestType, UsbContext};
use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

const CORAL_INITIAL_VID: u16 = 0x1a6e;
const CORAL_INITIAL_PID: u16 = 0x089a;
const CORAL_READY_VID: u16 = 0x18d1;
const CORAL_READY_PID: u16 = 0x9302;

#[derive(Clone, Copy, Debug)]
enum MatrixMode {
    Identity,
    ShiftPlus1,
    ShiftMinus1,
}

#[derive(Clone, Copy, Debug)]
enum AccessWidth {
    W32,
    W64,
}

#[derive(Clone, Copy, Debug)]
struct PerturbConfig {
    offset: u32,
    width: AccessWidth,
    value: u64,
}

fn usage(program: &str) {
    println!(
        "Usage: {program} <offset_hex|none> [32|64] [value_hex] [dim] [mode] [runs] [restore]"
    );
    println!("  mode: identity|shift_plus1|shift_minus1 (default: identity)");
    println!("  restore: 1|0 (default: 1)");
    println!(
        "Example: {program} 0x00048788 64 0x0000000000000000 2048 identity 1 1"
    );
    println!("Example: {program} none");
}

fn parse_u64_auto(s: &str) -> Result<u64, Box<dyn Error>> {
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return Ok(u64::from_str_radix(rest, 16)?);
    }
    Ok(s.parse::<u64>()?)
}

fn parse_u32_auto(s: &str) -> Result<u32, Box<dyn Error>> {
    let v = parse_u64_auto(s)?;
    if v > u32::MAX as u64 {
        return Err(format!("u32 overflow: {s}").into());
    }
    Ok(v as u32)
}

fn parse_mode(value: &str) -> Result<MatrixMode, Box<dyn Error>> {
    match value {
        "identity" => Ok(MatrixMode::Identity),
        "shift_plus1" => Ok(MatrixMode::ShiftPlus1),
        "shift_minus1" => Ok(MatrixMode::ShiftMinus1),
        _ => Err(
            format!("unknown mode: {value} (expected identity|shift_plus1|shift_minus1)").into(),
        ),
    }
}

fn parse_width(value: &str) -> Result<AccessWidth, Box<dyn Error>> {
    match value {
        "32" => Ok(AccessWidth::W32),
        "64" => Ok(AccessWidth::W64),
        _ => Err(format!("unknown width: {value} (expected 32|64)").into()),
    }
}

fn load_bundled_template(dim: usize) -> Result<DenseGemmTemplate, Box<dyn Error>> {
    let template = match dim {
        2048 => DenseGemmTemplate::from_bundled_2048()?,
        2304 => DenseGemmTemplate::from_bundled_2304()?,
        2688 => DenseGemmTemplate::from_bundled_2688()?,
        _ => return Err(format!("unsupported bundled dimension: {dim}").into()),
    };
    Ok(template)
}

fn build_ramp(dim: usize) -> Vec<i8> {
    let mut out = vec![0i8; dim];
    for (idx, v) in out.iter_mut().enumerate() {
        *v = idx as i8;
    }
    out
}

fn is_coral(desc: &DeviceDescriptor) -> bool {
    (desc.vendor_id() == CORAL_INITIAL_VID && desc.product_id() == CORAL_INITIAL_PID)
        || (desc.vendor_id() == CORAL_READY_VID && desc.product_id() == CORAL_READY_PID)
}

fn find_coral_device<T: UsbContext>(
    devices: &[Device<T>],
) -> Result<(Device<T>, DeviceDescriptor), Box<dyn Error>> {
    let mut runtime_match: Option<(Device<T>, DeviceDescriptor)> = None;
    let mut boot_match: Option<(Device<T>, DeviceDescriptor)> = None;

    for d in devices {
        let desc = d.device_descriptor()?;
        if !is_coral(&desc) {
            continue;
        }
        if desc.vendor_id() == CORAL_READY_VID && desc.product_id() == CORAL_READY_PID {
            runtime_match = Some((d.clone(), desc));
            break;
        }
        if boot_match.is_none() {
            boot_match = Some((d.clone(), desc));
        }
    }

    runtime_match
        .or(boot_match)
        .ok_or_else(|| "no Coral USB device found".into())
}

fn split_offset(offset: u32) -> (u16, u16) {
    let w_value = (offset & 0xffff) as u16;
    let w_index = ((offset >> 16) & 0xffff) as u16;
    (w_value, w_index)
}

fn vendor_req_type(direction: Direction) -> u8 {
    rusb::request_type(direction, RequestType::Vendor, Recipient::Device)
}

fn vendor_read32(
    handle: &rusb::DeviceHandle<Context>,
    offset: u32,
    timeout: Duration,
) -> rusb::Result<u32> {
    let (w_value, w_index) = split_offset(offset);
    let mut buf = [0u8; 4];
    let read = handle.read_control(
        vendor_req_type(Direction::In),
        0x01,
        w_value,
        w_index,
        &mut buf,
        timeout,
    )?;
    if read != 4 {
        return Err(rusb::Error::Other);
    }
    Ok(u32::from_le_bytes(buf))
}

fn vendor_read64(
    handle: &rusb::DeviceHandle<Context>,
    offset: u32,
    timeout: Duration,
) -> rusb::Result<u64> {
    let (w_value, w_index) = split_offset(offset);
    let mut buf = [0u8; 8];
    let read = handle.read_control(
        vendor_req_type(Direction::In),
        0x00,
        w_value,
        w_index,
        &mut buf,
        timeout,
    )?;
    if read != 8 {
        return Err(rusb::Error::Other);
    }
    Ok(u64::from_le_bytes(buf))
}

fn vendor_write32(
    handle: &rusb::DeviceHandle<Context>,
    offset: u32,
    value: u32,
    timeout: Duration,
) -> rusb::Result<()> {
    let (w_value, w_index) = split_offset(offset);
    let bytes = value.to_le_bytes();
    let written = handle.write_control(
        vendor_req_type(Direction::Out),
        0x01,
        w_value,
        w_index,
        &bytes,
        timeout,
    )?;
    if written != 4 {
        return Err(rusb::Error::Other);
    }
    Ok(())
}

fn vendor_write64(
    handle: &rusb::DeviceHandle<Context>,
    offset: u32,
    value: u64,
    timeout: Duration,
) -> rusb::Result<()> {
    let (w_value, w_index) = split_offset(offset);
    let bytes = value.to_le_bytes();
    let written = handle.write_control(
        vendor_req_type(Direction::Out),
        0x00,
        w_value,
        w_index,
        &bytes,
        timeout,
    )?;
    if written != 8 {
        return Err(rusb::Error::Other);
    }
    Ok(())
}

fn read_value(
    handle: &rusb::DeviceHandle<Context>,
    cfg: PerturbConfig,
    timeout: Duration,
) -> Result<u64, Box<dyn Error>> {
    let out = match cfg.width {
        AccessWidth::W32 => vendor_read32(handle, cfg.offset, timeout)? as u64,
        AccessWidth::W64 => vendor_read64(handle, cfg.offset, timeout)?,
    };
    Ok(out)
}

fn write_value(
    handle: &rusb::DeviceHandle<Context>,
    cfg: PerturbConfig,
    value: u64,
    timeout: Duration,
) -> Result<(), Box<dyn Error>> {
    match cfg.width {
        AccessWidth::W32 => {
            if value > u32::MAX as u64 {
                return Err(format!("value out of u32 range: 0x{value:016x}").into());
            }
            vendor_write32(handle, cfg.offset, value as u32, timeout)?;
        }
        AccessWidth::W64 => {
            vendor_write64(handle, cfg.offset, value, timeout)?;
        }
    }
    Ok(())
}

fn preview_first(data: &[i8], count: usize) -> String {
    data.iter()
        .take(count)
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        let program = args
            .first()
            .cloned()
            .unwrap_or_else(|| "gemm_csr_perturb_probe".to_string());
        usage(&program);
        return Ok(());
    }

    let perturb = if args.get(1).map(String::as_str).unwrap_or("none") == "none" {
        None
    } else {
        let offset = parse_u32_auto(args.get(1).ok_or("missing offset")?)?;
        let width = parse_width(args.get(2).ok_or("missing width (32|64)")?)?;
        let value = parse_u64_auto(args.get(3).ok_or("missing value")?)?;
        Some(PerturbConfig {
            offset,
            width,
            value,
        })
    };

    let arg_base = if perturb.is_some() { 4 } else { 2 };
    let dim = args
        .get(arg_base)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048);
    let mode = parse_mode(
        args.get(arg_base + 1)
            .map(String::as_str)
            .unwrap_or("identity"),
    )?;
    let runs = args
        .get(arg_base + 2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let restore = args
        .get(arg_base + 3)
        .and_then(|s| s.parse::<u32>().ok())
        .map(|v| v != 0)
        .unwrap_or(true);

    if runs == 0 {
        return Err("runs must be >= 1".into());
    }

    let mut template = load_bundled_template(dim)?;
    match mode {
        MatrixMode::Identity => template.set_identity(127)?,
        MatrixMode::ShiftPlus1 => template.set_shift_plus1(127)?,
        MatrixMode::ShiftMinus1 => template.set_shift_minus1(127)?,
    }
    let input = build_ramp(dim);

    println!("EdgeTPU version: {}", version());
    println!("dim={} mode={:?} runs={runs} restore={restore}", dim, mode);
    if let Some(p) = perturb {
        println!(
            "PERTURB offset=0x{:08x} width={:?} value=0x{:016x}",
            p.offset, p.width, p.value
        );
    } else {
        println!("PERTURB none");
    }

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;
    let prepared = template.prepare(&delegate)?;

    let timeout = Duration::from_millis(400);
    let mut original_value: Option<u64> = None;
    let mut perturbed_readback: Option<u64> = None;
    let mut restore_ok = false;

    let mut maybe_handle: Option<(rusb::DeviceHandle<Context>, PerturbConfig)> = None;
    if let Some(cfg) = perturb {
        let context = Context::new()?;
        let devices: Vec<Device<Context>> = context.devices()?.iter().collect();
        let (usb_dev, desc) = find_coral_device(&devices)?;
        println!(
            "USB handle target: bus={} addr={} id={:04x}:{:04x}",
            usb_dev.bus_number(),
            usb_dev.address(),
            desc.vendor_id(),
            desc.product_id()
        );

        let handle = usb_dev.open()?;
        let orig = read_value(&handle, cfg, timeout)?;
        original_value = Some(orig);
        write_value(&handle, cfg, cfg.value, timeout)?;
        let rb = read_value(&handle, cfg, timeout)?;
        perturbed_readback = Some(rb);
        println!(
            "PERTURB_APPLY original=0x{:016x} readback_after_write=0x{:016x}",
            orig, rb
        );
        maybe_handle = Some((handle, cfg));
    }

    let mut output = vec![0i8; dim];
    let started = Instant::now();
    let mut exec_err: Option<String> = None;
    for run_idx in 0..runs {
        match prepared.execute(&input) {
            Ok(current) => {
                if run_idx + 1 == runs {
                    output = current;
                }
            }
            Err(err) => {
                exec_err = Some(err.to_string());
                break;
            }
        }
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    if let Some((handle, cfg)) = &maybe_handle {
        if restore {
            if let Some(orig) = original_value {
                let w = write_value(handle, *cfg, orig, timeout);
                let r = read_value(handle, *cfg, timeout);
                let read_back = r.ok();
                restore_ok = w.is_ok() && read_back == Some(orig);
                if restore_ok {
                    println!("PERTURB_RESTORE ok value=0x{:016x}", orig);
                } else {
                    println!(
                        "PERTURB_RESTORE failed write_ok={} read_back={:?} expected=0x{:016x}",
                        w.is_ok(),
                        read_back.map(|v| format!("0x{v:016x}")),
                        orig
                    );
                }
            }
        } else {
            restore_ok = true;
        }
    } else {
        restore_ok = true;
    }

    let status = if exec_err.is_none() { "ok" } else { "exec_failed" };
    println!(
        "RESULT status={} elapsed_ms={:.3} output_head={}",
        status,
        elapsed_ms,
        preview_first(&output, 16)
    );
    if let Some(err) = exec_err {
        println!("RESULT_ERROR {err}");
    }
    println!(
        "RESULT_META original={} perturbed_readback={} restore_ok={}",
        original_value
            .map(|v| format!("0x{v:016x}"))
            .unwrap_or_else(|| "none".to_string()),
        perturbed_readback
            .map(|v| format!("0x{v:016x}"))
            .unwrap_or_else(|| "none".to_string()),
        restore_ok
    );

    if status != "ok" {
        return Err("execute failed".into());
    }
    if !restore_ok {
        return Err("restore failed".into());
    }
    Ok(())
}
