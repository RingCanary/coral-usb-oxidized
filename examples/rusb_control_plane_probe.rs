use rusb::{Context, Device, DeviceDescriptor, Direction, Recipient, RequestType, UsbContext};
use std::env;
use std::error::Error;
use std::time::Duration;

const CORAL_INITIAL_VID: u16 = 0x1a6e;
const CORAL_INITIAL_PID: u16 = 0x089a;
const CORAL_READY_VID: u16 = 0x18d1;
const CORAL_READY_PID: u16 = 0x9302;

#[derive(Debug, Clone)]
struct Config {
    claim_interface: bool,
    reset_device: bool,
    get_status: bool,
    verbose_configs: bool,
}

fn usage(program: &str) {
    println!("Usage: {program} [options]");
    println!("Options:");
    println!("  --claim-interface   Claim interface 0 on the first Coral device");
    println!("  --get-status        Issue standard GET_STATUS control read");
    println!("  --reset-device      Call libusb reset on the first Coral device");
    println!("  --verbose-configs   Print interface/endpoint topology");
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
    };

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "--claim-interface" => config.claim_interface = true,
            "--reset-device" => config.reset_device = true,
            "--get-status" => config.get_status = true,
            "--verbose-configs" => config.verbose_configs = true,
            other => return Err(format!("unknown argument: {}", other).into()),
        }
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

    if config.claim_interface || config.reset_device || config.get_status {
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

        if config.claim_interface {
            handle.claim_interface(0)?;
            println!("Claimed interface 0");
        }

        if config.get_status {
            let mut status = [0u8; 2];
            let req_ty =
                rusb::request_type(Direction::In, RequestType::Standard, Recipient::Device);
            let read =
                handle.read_control(req_ty, 0x00, 0, 0, &mut status, Duration::from_secs(1))?;
            println!(
                "GET_STATUS: read={} bytes value=0x{:02x}{:02x}",
                read, status[1], status[0]
            );
        }

        if config.reset_device {
            handle.reset()?;
            println!("Issued libusb reset() on device");
        }

        if config.claim_interface {
            let _ = handle.release_interface(0);
        }
    }

    Ok(())
}
