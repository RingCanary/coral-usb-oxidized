use coral_usb_oxidized::{
    executable_type_name, extract_serialized_executables_from_tflite, CoralError, DescriptorTag,
    EdgeTpuUsbDriver, SerializedExecutableBlob, VendorDirection,
    EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE,
};
use std::env;
use std::error::Error;
use std::time::Duration;

#[derive(Debug, Clone)]
struct Config {
    model_path: String,
    input_bytes: usize,
    output_bytes: usize,
    runs: usize,
    timeout_ms: u64,
    chunk_size: usize,
    setup: bool,
    verify_setup_reads: bool,
    setup_include_reads: bool,
    firmware_path: Option<String>,
    input_file: Option<String>,
    exec_index: Option<usize>,
    skip_param_preload: bool,
    read_interrupt: bool,
    instructions_tag: u32,
    parameters_tag: u32,
    input_activations_tag: u32,
    param_stream_chunk_size: Option<usize>,
    param_stream_max_bytes: Option<usize>,
    param_read_event_every: usize,
    param_event_timeout_ms: u64,
    param_write_sleep_us: u64,
    param_descriptor_split_bytes: Option<usize>,
    param_drain_event_every_descriptors: usize,
    param_read_interrupt_every: usize,
    param_interrupt_timeout_ms: u64,
}

fn usage(program: &str) {
    println!("Usage: {program} --model PATH [options]");
    println!("Options:");
    println!("  --model PATH              Compiled *_edgetpu.tflite model");
    println!("  --input-bytes N           Input activation bytes (default: 150528)");
    println!("  --output-bytes N          Output bytes to read from EP 0x81 (default: 1001)");
    println!("  --runs N                  Number of invoke attempts (default: 1)");
    println!("  --timeout-ms N            USB timeout ms (default: 6000)");
    println!("  --chunk-size N            Descriptor chunk size (default: 1048576)");
    println!("  --input-file PATH         Use raw input bytes from file instead of ramp");
    println!("  --exec-index N            Force executable index from extracted list");
    println!("  --skip-setup              Skip edgetpuxray runtime setup sequence");
    println!("  --setup-include-reads     Include setup read steps (default: write-only)");
    println!("  --verify-setup-reads      Enforce exact readback match for setup sequence");
    println!("  --firmware PATH           apex_latest_single_ep.bin for boot-mode devices");
    println!("  --skip-param-preload      Do not send PARAMETER_CACHING executables");
    println!("  --read-interrupt          Read one interrupt packet after run");
    println!(
        "  --instructions-tag N      Descriptor tag for instructions (default: {})",
        DescriptorTag::Instructions.as_u32()
    );
    println!(
        "  --parameters-tag N        Descriptor tag for parameters (default: {})",
        DescriptorTag::Parameters.as_u32()
    );
    println!(
        "  --input-tag N             Descriptor tag for input activations (default: {})",
        DescriptorTag::InputActivations.as_u32()
    );
    println!("  --param-stream-chunk-size N  Override parameter stream chunk size");
    println!("  --param-stream-max-bytes N   Limit parameter stream bytes for probing");
    println!("  --param-descriptor-split-bytes N  Split parameter stream across descriptor headers");
    println!(
        "  --param-read-event-every N  Poll EP 0x82 every N parameter chunks (0=off)"
    );
    println!(
        "  --param-read-interrupt-every N  Poll EP 0x83 every N parameter chunks (0=off)"
    );
    println!(
        "  --param-drain-event-every-descriptors N  Poll EP 0x82 after every N parameter descriptors (0=off)"
    );
    println!(
        "  --param-event-timeout-ms N  Timeout for poll reads during parameter stream (default: 1)"
    );
    println!(
        "  --param-interrupt-timeout-ms N  Timeout for interrupt polls during parameter stream (default: 1)"
    );
    println!(
        "  --param-write-sleep-us N    Sleep between parameter chunks to pace stream (default: 0)"
    );
    println!("Examples:");
    println!(
        "  {program} --model models/mobilenet_v1_1.0_224_quant_edgetpu.tflite --input-bytes 150528 --output-bytes 1001"
    );
}

fn parse_u64_auto(value: &str) -> Result<u64, Box<dyn Error>> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        return Ok(u64::from_str_radix(hex, 16)?);
    }
    Ok(value.parse::<u64>()?)
}

fn parse_usize_auto(value: &str) -> Result<usize, Box<dyn Error>> {
    Ok(parse_u64_auto(value)? as usize)
}

fn parse_tag_u32(value: &str, flag_name: &str) -> Result<u32, Box<dyn Error>> {
    let parsed = parse_u64_auto(value)?;
    if parsed > u32::MAX as u64 {
        return Err(format!(
            "{} value {} exceeds u32 max {}",
            flag_name,
            parsed,
            u32::MAX
        )
        .into());
    }
    Ok(parsed as u32)
}

fn descriptor_tag_name(tag: u32) -> &'static str {
    match tag {
        0 => "Instructions",
        1 => "InputActivations",
        2 => "Parameters",
        3 => "OutputActivations",
        4 => "Interrupt0",
        5 => "Interrupt1",
        6 => "Interrupt2",
        7 => "Interrupt3",
        _ => "Custom",
    }
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "rusb_serialized_exec_replay".to_string());

    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage(&program);
        std::process::exit(0);
    }

    let mut config = Config {
        model_path: String::new(),
        input_bytes: 150_528,
        output_bytes: 1001,
        runs: 1,
        timeout_ms: 6000,
        chunk_size: 0x100000,
        setup: true,
        verify_setup_reads: false,
        setup_include_reads: false,
        firmware_path: None,
        input_file: None,
        exec_index: None,
        skip_param_preload: false,
        read_interrupt: false,
        instructions_tag: DescriptorTag::Instructions.as_u32(),
        parameters_tag: DescriptorTag::Parameters.as_u32(),
        input_activations_tag: DescriptorTag::InputActivations.as_u32(),
        param_stream_chunk_size: None,
        param_stream_max_bytes: None,
        param_read_event_every: 0,
        param_event_timeout_ms: 1,
        param_write_sleep_us: 0,
        param_descriptor_split_bytes: None,
        param_drain_event_every_descriptors: 0,
        param_read_interrupt_every: 0,
        param_interrupt_timeout_ms: 1,
    };

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                config.model_path = args.get(i).ok_or("--model requires value")?.to_string();
            }
            "--input-bytes" => {
                i += 1;
                config.input_bytes =
                    parse_usize_auto(args.get(i).ok_or("--input-bytes requires value")?)?;
            }
            "--output-bytes" => {
                i += 1;
                config.output_bytes =
                    parse_usize_auto(args.get(i).ok_or("--output-bytes requires value")?)?;
            }
            "--runs" => {
                i += 1;
                config.runs = parse_usize_auto(args.get(i).ok_or("--runs requires value")?)?;
            }
            "--timeout-ms" => {
                i += 1;
                config.timeout_ms =
                    parse_u64_auto(args.get(i).ok_or("--timeout-ms requires value")?)?;
            }
            "--chunk-size" => {
                i += 1;
                config.chunk_size =
                    parse_usize_auto(args.get(i).ok_or("--chunk-size requires value")?)?;
            }
            "--input-file" => {
                i += 1;
                config.input_file = Some(
                    args.get(i)
                        .ok_or("--input-file requires value")?
                        .to_string(),
                );
            }
            "--exec-index" => {
                i += 1;
                config.exec_index = Some(parse_usize_auto(
                    args.get(i).ok_or("--exec-index requires value")?,
                )?);
            }
            "--skip-setup" => config.setup = false,
            "--setup-include-reads" => config.setup_include_reads = true,
            "--verify-setup-reads" => config.verify_setup_reads = true,
            "--firmware" => {
                i += 1;
                config.firmware_path =
                    Some(args.get(i).ok_or("--firmware requires value")?.to_string());
            }
            "--skip-param-preload" => config.skip_param_preload = true,
            "--read-interrupt" => config.read_interrupt = true,
            "--instructions-tag" => {
                i += 1;
                config.instructions_tag = parse_tag_u32(
                    args.get(i).ok_or("--instructions-tag requires value")?,
                    "--instructions-tag",
                )?;
            }
            "--parameters-tag" => {
                i += 1;
                config.parameters_tag = parse_tag_u32(
                    args.get(i).ok_or("--parameters-tag requires value")?,
                    "--parameters-tag",
                )?;
            }
            "--input-tag" => {
                i += 1;
                config.input_activations_tag =
                    parse_tag_u32(args.get(i).ok_or("--input-tag requires value")?, "--input-tag")?;
            }
            "--param-stream-chunk-size" => {
                i += 1;
                config.param_stream_chunk_size = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-stream-chunk-size requires value")?,
                )?);
            }
            "--param-stream-max-bytes" => {
                i += 1;
                config.param_stream_max_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-stream-max-bytes requires value")?,
                )?);
            }
            "--param-descriptor-split-bytes" => {
                i += 1;
                config.param_descriptor_split_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-descriptor-split-bytes requires value")?,
                )?);
            }
            "--param-read-event-every" => {
                i += 1;
                config.param_read_event_every = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-read-event-every requires value")?,
                )?;
            }
            "--param-read-interrupt-every" => {
                i += 1;
                config.param_read_interrupt_every = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-read-interrupt-every requires value")?,
                )?;
            }
            "--param-drain-event-every-descriptors" => {
                i += 1;
                config.param_drain_event_every_descriptors = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-drain-event-every-descriptors requires value")?,
                )?;
            }
            "--param-event-timeout-ms" => {
                i += 1;
                config.param_event_timeout_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-event-timeout-ms requires value")?,
                )?;
            }
            "--param-interrupt-timeout-ms" => {
                i += 1;
                config.param_interrupt_timeout_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-interrupt-timeout-ms requires value")?,
                )?;
            }
            "--param-write-sleep-us" => {
                i += 1;
                config.param_write_sleep_us = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-write-sleep-us requires value")?,
                )?;
            }
            other => return Err(format!("unknown argument: {}", other).into()),
        }
        i += 1;
    }

    if config.model_path.is_empty() {
        return Err("--model is required".into());
    }
    if config.runs == 0 {
        return Err("--runs must be >= 1".into());
    }
    if config.input_bytes == 0 {
        return Err("--input-bytes must be >= 1".into());
    }
    if config.output_bytes == 0 {
        return Err("--output-bytes must be >= 1".into());
    }
    if config.chunk_size == 0 {
        return Err("--chunk-size must be >= 1".into());
    }
    if matches!(config.param_stream_chunk_size, Some(0)) {
        return Err("--param-stream-chunk-size must be >= 1".into());
    }
    if matches!(config.param_descriptor_split_bytes, Some(0)) {
        return Err("--param-descriptor-split-bytes must be >= 1".into());
    }

    Ok(config)
}

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

fn load_input(config: &Config) -> Result<Vec<u8>, Box<dyn Error>> {
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
        return Ok(data);
    }

    let mut data = vec![0u8; config.input_bytes];
    for (idx, byte) in data.iter_mut().enumerate() {
        *byte = (idx % 251) as u8;
    }
    Ok(data)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
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
        || config.param_drain_event_every_descriptors > 0;
    if !use_custom_stream {
        driver.send_descriptor_payload_raw(config.parameters_tag, payload)?;
        return Ok(());
    }

    let stream_len = config
        .param_stream_max_bytes
        .map(|max| max.min(payload.len()))
        .unwrap_or(payload.len());
    let stream_chunk_size = config.param_stream_chunk_size.unwrap_or(config.chunk_size);
    let descriptor_split_size = config.param_descriptor_split_bytes.unwrap_or(stream_len.max(1));
    let poll_timeout = Duration::from_millis(config.param_event_timeout_ms);
    let interrupt_poll_timeout = Duration::from_millis(config.param_interrupt_timeout_ms);

    println!(
        "    {}: streaming parameters len={} chunk={} desc_split={} event_poll_every={} intr_poll_every={} drain_desc_every={} sleep_us={} tag={}",
        phase_label,
        stream_len,
        stream_chunk_size,
        descriptor_split_size,
        config.param_read_event_every,
        config.param_read_interrupt_every,
        config.param_drain_event_every_descriptors,
        config.param_write_sleep_us,
        config.parameters_tag
    );

    let mut global_offset = 0usize;
    let mut global_chunk_idx = 0usize;
    let mut descriptor_idx = 0usize;
    while global_offset < stream_len {
        let descriptor_end = (global_offset + descriptor_split_size).min(stream_len);
        let descriptor_len = descriptor_end - global_offset;
        driver.send_descriptor_header_raw(config.parameters_tag, descriptor_len)?;

        let mut descriptor_offset = global_offset;
        while descriptor_offset < descriptor_end {
            let end = (descriptor_offset + stream_chunk_size).min(descriptor_end);
            let chunk = &payload[descriptor_offset..end];
            if let Err(err) = driver.write_bulk_out_chunk(chunk) {
                return Err(format!(
                    "{}: parameter stream write failed at offset {} of {} bytes (chunk {}): {}",
                    phase_label, descriptor_offset, stream_len, global_chunk_idx, err
                )
                .into());
            }
            descriptor_offset = end;
            global_chunk_idx += 1;

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

    Ok(())
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
    let config = parse_args()?;
    let model_bytes = std::fs::read(&config.model_path)?;
    let executables = extract_serialized_executables_from_tflite(&model_bytes)?;

    println!("Model: {}", config.model_path);
    println!(
        "Descriptor tags: instr={}({}), params={}({}), input={}({})",
        config.instructions_tag,
        descriptor_tag_name(config.instructions_tag),
        config.parameters_tag,
        descriptor_tag_name(config.parameters_tag),
        config.input_activations_tag,
        descriptor_tag_name(config.input_activations_tag)
    );
    if config.param_stream_chunk_size.is_some()
        || config.param_stream_max_bytes.is_some()
        || config.param_read_event_every > 0
        || config.param_read_interrupt_every > 0
        || config.param_write_sleep_us > 0
        || config.param_descriptor_split_bytes.is_some()
        || config.param_drain_event_every_descriptors > 0
    {
        println!(
            "Parameter stream controls: chunk={:?} max_bytes={:?} desc_split={:?} event_poll_every={} intr_poll_every={} drain_desc_every={} event_timeout_ms={} intr_timeout_ms={} sleep_us={}",
            config.param_stream_chunk_size,
            config.param_stream_max_bytes,
            config.param_descriptor_split_bytes,
            config.param_read_event_every,
            config.param_read_interrupt_every,
            config.param_drain_event_every_descriptors,
            config.param_event_timeout_ms,
            config.param_interrupt_timeout_ms,
            config.param_write_sleep_us
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

    let input_bytes = load_input(&config)?;

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

    let _ = driver.set_configuration_1();
    driver.claim_interface0()?;

    if config.setup {
        let setup_steps: Vec<_> = if config.setup_include_reads {
            EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE.to_vec()
        } else {
            EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE
                .iter()
                .copied()
                .filter(|step| step.direction == VendorDirection::Out)
                .collect()
        };
        println!(
            "Applying edgetpuxray runtime setup sequence ({} steps, include_reads={}, verify_reads={})",
            setup_steps.len(),
            config.setup_include_reads,
            config.verify_setup_reads
        );
        driver.apply_vendor_steps(&setup_steps, config.verify_setup_reads)?;
    } else {
        println!("Skipping setup sequence (--skip-setup)");
    }

    if !config.skip_param_preload {
        if !param_executables.is_empty() && run_exe.executable_type == 2 {
            println!(
                "Bootstrap phase: send EXECUTION_ONLY instruction chunks, then PARAMETER_CACHING streams"
            );
            for (idx, chunk) in run_exe.instruction_bitstreams.iter().enumerate() {
                println!("  EXECUTION_ONLY chunk {} ({} bytes)", idx, chunk.len());
                driver.send_descriptor_payload_raw(config.instructions_tag, chunk)?;
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
                    driver.send_descriptor_payload_raw(config.instructions_tag, chunk)?;
                }
                send_parameter_payload(&driver, &config, &exe.parameters_stream, "bootstrap param")?;
            }
            match driver.read_event_packet() {
                Ok(event) => println!(
                    "  Bootstrap event: tag={} offset=0x{:016x} length={}",
                    event.tag, event.offset, event.length
                ),
                Err(err) => println!("  Bootstrap event read failed: {}", err),
            }
        } else {
            for exe in &param_executables {
                println!(
                    "Preload PARAMETER_CACHING executable idx={} chunks={} params={} bytes",
                    exe.executable_index,
                    exe.instruction_bitstreams.len(),
                    exe.parameters_stream.len()
                );
                for chunk in &exe.instruction_bitstreams {
                    driver.send_descriptor_payload_raw(config.instructions_tag, chunk)?;
                }
                send_parameter_payload(&driver, &config, &exe.parameters_stream, "preload param")?;
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

    for run in 0..config.runs {
        println!("RUN {}", run + 1);
        for (idx, chunk) in run_exe.instruction_bitstreams.iter().enumerate() {
            println!("  run instr chunk {} ({} bytes)", idx, chunk.len());
            driver.send_descriptor_payload_raw(config.instructions_tag, chunk)?;
        }
        if !run_exe.parameters_stream.is_empty() {
            send_parameter_payload(&driver, &config, &run_exe.parameters_stream, "run param")?;
        }
        driver.send_descriptor_payload_raw(config.input_activations_tag, &input_bytes)?;

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
    }

    Ok(())
}
