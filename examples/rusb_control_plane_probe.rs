use rusb::{Context, Device, DeviceDescriptor, Direction, Recipient, RequestType, UsbContext};
use std::env;
use std::error::Error;
use std::time::Duration;

const CORAL_INITIAL_VID: u16 = 0x1a6e;
const CORAL_INITIAL_PID: u16 = 0x089a;
const CORAL_READY_VID: u16 = 0x18d1;
const CORAL_READY_PID: u16 = 0x9302;
const EP_EVENT_IN: u8 = 0x82;
const EP_INTERRUPT_IN: u8 = 0x83;

const TAG_NAMES: [&str; 8] = [
    "Instructions",
    "InputActivations",
    "Parameters",
    "OutputActivations",
    "Interrupt0",
    "Interrupt1",
    "Interrupt2",
    "Interrupt3",
];

#[derive(Debug, Clone)]
struct Config {
    claim_interface: bool,
    reset_device: bool,
    get_status: bool,
    verbose_configs: bool,
    vendor_read32: Vec<u32>,
    vendor_read64: Vec<u32>,
    vendor_write32: Vec<(u32, u32)>,
    vendor_write64: Vec<(u32, u64)>,
    read_event_count: usize,
    read_interrupt_count: usize,
    timeout_ms: u64,
}

fn usage(program: &str) {
    println!("Usage: {program} [options]");
    println!("Options:");
    println!("  --claim-interface   Claim interface 0 on the first Coral device");
    println!("  --get-status        Issue standard GET_STATUS control read");
    println!("  --reset-device      Call libusb reset on the first Coral device");
    println!("  --verbose-configs   Print interface/endpoint topology");
    println!("  --timeout-ms N      Timeout for reads/writes in ms (default: 500)");
    println!("  --vendor-read32 OFF Read vendor register32 at full CSR offset");
    println!("  --vendor-read64 OFF Read vendor register64 at full CSR offset");
    println!("  --vendor-write32 O=V Write vendor register32 (offset=value)");
    println!("  --vendor-write64 O=V Write vendor register64 (offset=value)");
    println!("  --read-event N      Read N event packets from endpoint 0x82");
    println!("  --read-interrupt N  Read N interrupt packets from endpoint 0x83");
    println!("Examples:");
    println!("  {program} --claim-interface --vendor-read64 0x00044018");
    println!("  {program} --claim-interface --vendor-read64 0x00048788 --read-interrupt 1");
    println!("  {program} --claim-interface --vendor-write32 0x0001a30c=0x00003800");
}

fn parse_u64_auto(value: &str) -> Result<u64, Box<dyn Error>> {
    if let Some(rest) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        return Ok(u64::from_str_radix(rest, 16)?);
    }
    Ok(value.parse::<u64>()?)
}

fn parse_u32_auto(value: &str) -> Result<u32, Box<dyn Error>> {
    let v = parse_u64_auto(value)?;
    if v > u32::MAX as u64 {
        return Err(format!("value out of u32 range: {}", value).into());
    }
    Ok(v as u32)
}

fn parse_write32_pair(value: &str) -> Result<(u32, u32), Box<dyn Error>> {
    let (offset_s, reg_s) = value
        .split_once('=')
        .ok_or("expected OFFSET=VALUE for --vendor-write32")?;
    Ok((parse_u32_auto(offset_s)?, parse_u32_auto(reg_s)?))
}

fn parse_write64_pair(value: &str) -> Result<(u32, u64), Box<dyn Error>> {
    let (offset_s, reg_s) = value
        .split_once('=')
        .ok_or("expected OFFSET=VALUE for --vendor-write64")?;
    Ok((parse_u32_auto(offset_s)?, parse_u64_auto(reg_s)?))
}

fn known_reg_name(offset: u32) -> Option<&'static str> {
    match offset {
        0x0004_4018 => Some("scalarCoreRunControl"),
        0x0004_8788 => Some("tileconfig0"),
        0x0001_a30c => Some("scu_ctrl_0"),
        0x0004_8528 => Some("output_actv_queue_base"),
        0x0004_8540 => Some("output_actv_queue_tail"),
        0x0004_8550 => Some("output_actv_queue_completed_head"),
        0x0004_8590 => Some("instruction_queue_base"),
        0x0004_85a8 => Some("instruction_queue_tail"),
        0x0004_85b0 => Some("instruction_queue_fetched_head"),
        0x0004_85b8 => Some("instruction_queue_completed_head"),
        0x0004_85f8 => Some("input_actv_queue_base"),
        0x0004_8610 => Some("input_actv_queue_tail"),
        0x0004_8620 => Some("input_actv_queue_completed_head"),
        0x0004_8660 => Some("param_queue_base"),
        0x0004_8678 => Some("param_queue_tail"),
        0x0004_8688 => Some("param_queue_completed_head"),
        _ => None,
    }
}

fn format_reg(offset: u32) -> String {
    if let Some(name) = known_reg_name(offset) {
        format!("0x{offset:08x} ({name})")
    } else {
        format!("0x{offset:08x}")
    }
}

fn split_offset(offset: u32) -> (u16, u16) {
    let w_value = (offset & 0xffff) as u16;
    let w_index = ((offset >> 16) & 0xffff) as u16;
    (w_value, w_index)
}

fn vendor_req_type(direction: Direction) -> u8 {
    rusb::request_type(direction, RequestType::Vendor, Recipient::Device)
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "rusb_control_plane_probe".to_string());

    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage(&program);
        std::process::exit(0);
    }

    let mut config = Config {
        claim_interface: false,
        reset_device: false,
        get_status: false,
        verbose_configs: false,
        vendor_read32: Vec::new(),
        vendor_read64: Vec::new(),
        vendor_write32: Vec::new(),
        vendor_write64: Vec::new(),
        read_event_count: 0,
        read_interrupt_count: 0,
        timeout_ms: 500,
    };

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--claim-interface" => config.claim_interface = true,
            "--reset-device" => config.reset_device = true,
            "--get-status" => config.get_status = true,
            "--verbose-configs" => config.verbose_configs = true,
            "--timeout-ms" => {
                idx += 1;
                config.timeout_ms = args
                    .get(idx)
                    .ok_or("--timeout-ms requires value")?
                    .parse()?;
            }
            "--vendor-read32" => {
                idx += 1;
                let offset =
                    parse_u32_auto(args.get(idx).ok_or("--vendor-read32 requires value")?)?;
                config.vendor_read32.push(offset);
            }
            "--vendor-read64" => {
                idx += 1;
                let offset =
                    parse_u32_auto(args.get(idx).ok_or("--vendor-read64 requires value")?)?;
                config.vendor_read64.push(offset);
            }
            "--vendor-write32" => {
                idx += 1;
                let pair =
                    parse_write32_pair(args.get(idx).ok_or("--vendor-write32 requires value")?)?;
                config.vendor_write32.push(pair);
            }
            "--vendor-write64" => {
                idx += 1;
                let pair =
                    parse_write64_pair(args.get(idx).ok_or("--vendor-write64 requires value")?)?;
                config.vendor_write64.push(pair);
            }
            "--read-event" => {
                idx += 1;
                config.read_event_count = args
                    .get(idx)
                    .ok_or("--read-event requires value")?
                    .parse()?;
            }
            "--read-interrupt" => {
                idx += 1;
                config.read_interrupt_count = args
                    .get(idx)
                    .ok_or("--read-interrupt requires value")?
                    .parse()?;
            }
            other => return Err(format!("unknown argument: {}", other).into()),
        }
        idx += 1;
    }

    if config.timeout_ms == 0 {
        return Err("--timeout-ms must be >= 1".into());
    }

    Ok(config)
}

fn is_coral(desc: &DeviceDescriptor) -> bool {
    (desc.vendor_id() == CORAL_INITIAL_VID && desc.product_id() == CORAL_INITIAL_PID)
        || (desc.vendor_id() == CORAL_READY_VID && desc.product_id() == CORAL_READY_PID)
}

fn state_label(desc: &DeviceDescriptor) -> &'static str {
    match (desc.vendor_id(), desc.product_id()) {
        (CORAL_INITIAL_VID, CORAL_INITIAL_PID) => "boot/uninitialized",
        (CORAL_READY_VID, CORAL_READY_PID) => "runtime/initialized",
        _ => "unknown",
    }
}

fn print_device_topology<T: UsbContext>(device: &Device<T>) -> Result<(), Box<dyn Error>> {
    let config_count = device.device_descriptor()?.num_configurations();
    for cfg_idx in 0..config_count {
        let config = device.config_descriptor(cfg_idx)?;
        println!(
            "  config {}: interfaces={} max_power={}mA",
            cfg_idx,
            config.num_interfaces(),
            config.max_power()
        );
        for interface in config.interfaces() {
            for descriptor in interface.descriptors() {
                println!(
                    "    if={} alt={} class=0x{:02x} subclass=0x{:02x} proto=0x{:02x}",
                    descriptor.interface_number(),
                    descriptor.setting_number(),
                    descriptor.class_code(),
                    descriptor.sub_class_code(),
                    descriptor.protocol_code()
                );
                for endpoint in descriptor.endpoint_descriptors() {
                    println!(
                        "      ep=0x{:02x} dir={:?} transfer={:?} max_packet={} interval={}",
                        endpoint.address(),
                        endpoint.direction(),
                        endpoint.transfer_type(),
                        endpoint.max_packet_size(),
                        endpoint.interval()
                    );
                }
            }
        }
    }
    Ok(())
}

fn vendor_read32(
    handle: &rusb::DeviceHandle<Context>,
    offset: u32,
    timeout: Duration,
) -> Result<u32, Box<dyn Error>> {
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
        return Err(format!("vendor_read32 short read: expected 4, got {}", read).into());
    }
    Ok(u32::from_le_bytes(buf))
}

fn vendor_read64(
    handle: &rusb::DeviceHandle<Context>,
    offset: u32,
    timeout: Duration,
) -> Result<u64, Box<dyn Error>> {
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
        return Err(format!("vendor_read64 short read: expected 8, got {}", read).into());
    }
    Ok(u64::from_le_bytes(buf))
}

fn vendor_write32(
    handle: &rusb::DeviceHandle<Context>,
    offset: u32,
    value: u32,
    timeout: Duration,
) -> Result<(), Box<dyn Error>> {
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
        return Err(format!("vendor_write32 short write: expected 4, got {}", written).into());
    }
    Ok(())
}

fn vendor_write64(
    handle: &rusb::DeviceHandle<Context>,
    offset: u32,
    value: u64,
    timeout: Duration,
) -> Result<(), Box<dyn Error>> {
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
        return Err(format!("vendor_write64 short write: expected 8, got {}", written).into());
    }
    Ok(())
}

fn decode_event_packet(buf: &[u8]) -> Option<(u64, u32, u8)> {
    if buf.len() < 16 {
        return None;
    }
    let offset = u64::from_le_bytes(buf[0..8].try_into().ok()?);
    let length = u32::from_le_bytes(buf[8..12].try_into().ok()?);
    let tag = buf[12] & 0x0f;
    Some((offset, length, tag))
}

fn decode_interrupt_packet(buf: &[u8]) -> Option<u32> {
    if buf.len() < 4 {
        return None;
    }
    Some(u32::from_le_bytes(buf[0..4].try_into().ok()?))
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    let context = Context::new()?;
    let devices = context.devices()?;

    let mut coral_devices: Vec<(Device<Context>, DeviceDescriptor)> = Vec::new();
    for device in devices.iter() {
        let desc = device.device_descriptor()?;
        if is_coral(&desc) {
            coral_devices.push((device, desc));
        }
    }

    println!("Found {} Coral USB device(s)", coral_devices.len());
    for (idx, (device, desc)) in coral_devices.iter().enumerate() {
        println!(
            "  [{}] bus={} addr={} id={:04x}:{:04x} state={}",
            idx,
            device.bus_number(),
            device.address(),
            desc.vendor_id(),
            desc.product_id(),
            state_label(desc)
        );
        if config.verbose_configs {
            print_device_topology(device)?;
        }
    }

    if coral_devices.is_empty() {
        return Ok(());
    }

    let has_vendor_ops = !config.vendor_read32.is_empty()
        || !config.vendor_read64.is_empty()
        || !config.vendor_write32.is_empty()
        || !config.vendor_write64.is_empty();
    let needs_io = has_vendor_ops || config.read_event_count > 0 || config.read_interrupt_count > 0;
    let need_open = config.claim_interface || config.reset_device || config.get_status || needs_io;

    if need_open {
        let (device, desc) = &coral_devices[0];
        println!(
            "Opening first Coral device: bus={} addr={} id={:04x}:{:04x}",
            device.bus_number(),
            device.address(),
            desc.vendor_id(),
            desc.product_id()
        );
        let handle = device.open()?;
        let _ = handle.set_auto_detach_kernel_driver(true);
        let timeout = Duration::from_millis(config.timeout_ms);
        let should_claim = config.claim_interface || needs_io;
        let mut claimed = false;

        if should_claim {
            handle.claim_interface(0)?;
            claimed = true;
            if config.claim_interface {
                println!("Claimed interface 0");
            } else {
                println!("Claimed interface 0 (required for requested vendor/endpoint ops)");
            }
        }

        if config.get_status {
            let mut status = [0u8; 2];
            let req_ty =
                rusb::request_type(Direction::In, RequestType::Standard, Recipient::Device);
            let read = handle.read_control(req_ty, 0x00, 0, 0, &mut status, timeout)?;
            println!(
                "GET_STATUS: read={} bytes value=0x{:02x}{:02x}",
                read, status[1], status[0]
            );
        }

        for offset in &config.vendor_read32 {
            let val = vendor_read32(&handle, *offset, timeout)?;
            println!(
                "VENDOR_READ32 {} => 0x{val:08x} ({val})",
                format_reg(*offset)
            );
        }
        for offset in &config.vendor_read64 {
            let val = vendor_read64(&handle, *offset, timeout)?;
            println!(
                "VENDOR_READ64 {} => 0x{val:016x} ({val})",
                format_reg(*offset)
            );
        }
        for (offset, value) in &config.vendor_write32 {
            vendor_write32(&handle, *offset, *value, timeout)?;
            println!(
                "VENDOR_WRITE32 {} <= 0x{value:08x} ({value})",
                format_reg(*offset)
            );
        }
        for (offset, value) in &config.vendor_write64 {
            vendor_write64(&handle, *offset, *value, timeout)?;
            println!(
                "VENDOR_WRITE64 {} <= 0x{value:016x} ({value})",
                format_reg(*offset)
            );
        }

        if config.read_event_count > 0 {
            println!(
                "Reading {} event packet(s) from endpoint 0x{EP_EVENT_IN:02x}",
                config.read_event_count
            );
            for idx in 0..config.read_event_count {
                let mut buf = [0u8; 16];
                match handle.read_bulk(EP_EVENT_IN, &mut buf, timeout) {
                    Ok(read) => {
                        if let Some((offset, length, tag)) = decode_event_packet(&buf[..read]) {
                            let tag_name =
                                TAG_NAMES.get(tag as usize).copied().unwrap_or("UnknownTag");
                            println!(
                                "EVENT[{idx}] read={read} tag={}({}) offset=0x{offset:016x} length={length}",
                                tag, tag_name
                            );
                        } else {
                            println!("EVENT[{idx}] read={read} raw={:02x?}", &buf[..read]);
                        }
                    }
                    Err(rusb::Error::Timeout) => {
                        println!("EVENT[{idx}] timeout after {} ms", config.timeout_ms);
                    }
                    Err(err) => return Err(format!("event read failed: {}", err).into()),
                }
            }
        }

        if config.read_interrupt_count > 0 {
            println!(
                "Reading {} interrupt packet(s) from endpoint 0x{EP_INTERRUPT_IN:02x}",
                config.read_interrupt_count
            );
            for idx in 0..config.read_interrupt_count {
                let mut buf = [0u8; 4];
                match handle.read_interrupt(EP_INTERRUPT_IN, &mut buf, timeout) {
                    Ok(read) => {
                        if let Some(raw) = decode_interrupt_packet(&buf[..read]) {
                            let fatal = (raw & 1) != 0;
                            let top_level = raw >> 1;
                            println!(
                                "INTERRUPT[{idx}] read={read} raw=0x{raw:08x} fatal={} top_level_mask=0x{top_level:08x}",
                                fatal
                            );
                        } else {
                            println!("INTERRUPT[{idx}] read={read} raw={:02x?}", &buf[..read]);
                        }
                    }
                    Err(rusb::Error::Timeout) => {
                        println!("INTERRUPT[{idx}] timeout after {} ms", config.timeout_ms);
                    }
                    Err(err) => return Err(format!("interrupt read failed: {}", err).into()),
                }
            }
        }

        if config.reset_device {
            handle.reset()?;
            println!("Issued libusb reset() on device");
        }

        if claimed {
            let _ = handle.release_interface(0);
        }
    }

    Ok(())
}
