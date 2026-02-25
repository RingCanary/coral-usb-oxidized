use coral_usb_oxidized::{
    extract_serialized_executables_from_tflite, EdgeTpuUsbDriver, SerializedExecutableBlob,
    VendorDirection, EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE,
};
use std::env;
use std::error::Error;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct Config {
    model_path: String,
    firmware_path: Option<String>,
    timeout_ms: u64,
    descriptor_chunk_size: usize,
    param_stream_chunk_size: usize,
    param_max_bytes: usize,
    seed: u64,
    glitch_every_chunks: usize,
    glitch_budget: usize,
    aggressive: bool,
    input_bytes: usize,
    output_bytes: usize,
    skip_setup: bool,
    inject_runcontrol0_chunks: Vec<usize>,
    inject_runcontrol1_chunks: Vec<usize>,
    glitch_mode: GlitchMode,
}

#[derive(Debug, Clone, Copy)]
enum GlitchAction {
    PollEvent,
    PollInterrupt,
    ReadRunControl,
    WriteRunControlOne,
    ReadScuCtrl,
    SleepJitter,
    WriteRunControlZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlitchMode {
    Mixed,
    ReadOnly,
    RunControlOnly,
}

#[derive(Debug, Clone, Copy)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        let seeded = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state: seeded }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_range(&mut self, upper_exclusive: u32) -> u32 {
        if upper_exclusive <= 1 {
            return 0;
        }
        self.next_u32() % upper_exclusive
    }
}

fn usage(program: &str) {
    println!("Usage: {program} --model PATH [options]");
    println!("Options:");
    println!("  --model PATH                 Compiled *_edgetpu.tflite model");
    println!("  --firmware PATH              apex_latest_single_ep.bin for boot-mode devices");
    println!("  --timeout-ms N               USB timeout ms (default: 6000)");
    println!("  --descriptor-chunk-size N    Descriptor payload chunk size (default: 4096)");
    println!("  --param-stream-chunk-size N  Chunk size for fuzzed parameter stream (default: 1024)");
    println!("  --param-max-bytes N          Max parameter bytes to stream (default: 65536)");
    println!("  --seed N                     RNG seed (default: current unix nanos)");
    println!("  --glitch-every-chunks N      Inject one glitch every N parameter chunks (default: 4)");
    println!("  --glitch-budget N            Max glitches in this run (default: 32)");
    println!("  --aggressive                 Include runcontrol=0 glitch action");
    println!("  --input-bytes N              Input bytes for final invoke (default: 2048)");
    println!("  --output-bytes N             Output bytes to read (default: 2048)");
    println!("  --skip-setup                 Skip runtime setup sequence");
    println!("  --inject-runcontrol0-at LIST Comma-separated chunk indices for rc=0 writes");
    println!("  --inject-runcontrol1-at LIST Comma-separated chunk indices for rc=1 writes");
    println!("  --glitch-mode MODE           mixed|readonly|runctl (default: mixed)");
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

fn parse_usize_list(value: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    if value.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in value.split(',') {
        out.push(parse_usize_auto(part.trim())?);
    }
    Ok(out)
}

fn parse_glitch_mode(value: &str) -> Result<GlitchMode, Box<dyn Error>> {
    match value {
        "mixed" => Ok(GlitchMode::Mixed),
        "readonly" => Ok(GlitchMode::ReadOnly),
        "runctl" => Ok(GlitchMode::RunControlOnly),
        other => Err(
            format!("invalid --glitch-mode: {other} (expected mixed|readonly|runctl)").into(),
        ),
    }
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "rusb_param_glitch_fuzz".to_string());

    if args.iter().any(|a| a == "--help" || a == "-h") {
        usage(&program);
        std::process::exit(0);
    }

    let default_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_nanos() as u64;

    let mut cfg = Config {
        model_path: String::new(),
        firmware_path: None,
        timeout_ms: 6000,
        descriptor_chunk_size: 4096,
        param_stream_chunk_size: 1024,
        param_max_bytes: 65536,
        seed: default_seed,
        glitch_every_chunks: 4,
        glitch_budget: 32,
        aggressive: false,
        input_bytes: 2048,
        output_bytes: 2048,
        skip_setup: false,
        inject_runcontrol0_chunks: Vec::new(),
        inject_runcontrol1_chunks: Vec::new(),
        glitch_mode: GlitchMode::Mixed,
    };

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                cfg.model_path = args.get(i).ok_or("--model requires value")?.to_string();
            }
            "--firmware" => {
                i += 1;
                cfg.firmware_path = Some(args.get(i).ok_or("--firmware requires value")?.to_string());
            }
            "--timeout-ms" => {
                i += 1;
                cfg.timeout_ms = parse_u64_auto(args.get(i).ok_or("--timeout-ms requires value")?)?;
            }
            "--descriptor-chunk-size" => {
                i += 1;
                cfg.descriptor_chunk_size =
                    parse_usize_auto(args.get(i).ok_or("--descriptor-chunk-size requires value")?)?;
            }
            "--param-stream-chunk-size" => {
                i += 1;
                cfg.param_stream_chunk_size = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-stream-chunk-size requires value")?,
                )?;
            }
            "--param-max-bytes" => {
                i += 1;
                cfg.param_max_bytes =
                    parse_usize_auto(args.get(i).ok_or("--param-max-bytes requires value")?)?;
            }
            "--seed" => {
                i += 1;
                cfg.seed = parse_u64_auto(args.get(i).ok_or("--seed requires value")?)?;
            }
            "--glitch-every-chunks" => {
                i += 1;
                cfg.glitch_every_chunks = parse_usize_auto(
                    args.get(i)
                        .ok_or("--glitch-every-chunks requires value")?,
                )?;
            }
            "--glitch-budget" => {
                i += 1;
                cfg.glitch_budget =
                    parse_usize_auto(args.get(i).ok_or("--glitch-budget requires value")?)?;
            }
            "--aggressive" => cfg.aggressive = true,
            "--input-bytes" => {
                i += 1;
                cfg.input_bytes = parse_usize_auto(args.get(i).ok_or("--input-bytes requires value")?)?;
            }
            "--output-bytes" => {
                i += 1;
                cfg.output_bytes = parse_usize_auto(args.get(i).ok_or("--output-bytes requires value")?)?;
            }
            "--skip-setup" => cfg.skip_setup = true,
            "--inject-runcontrol0-at" => {
                i += 1;
                cfg.inject_runcontrol0_chunks =
                    parse_usize_list(args.get(i).ok_or("--inject-runcontrol0-at requires value")?)?;
            }
            "--inject-runcontrol1-at" => {
                i += 1;
                cfg.inject_runcontrol1_chunks =
                    parse_usize_list(args.get(i).ok_or("--inject-runcontrol1-at requires value")?)?;
            }
            "--glitch-mode" => {
                i += 1;
                cfg.glitch_mode =
                    parse_glitch_mode(args.get(i).ok_or("--glitch-mode requires value")?)?;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    if cfg.model_path.is_empty() {
        return Err("--model is required".into());
    }
    if cfg.descriptor_chunk_size == 0
        || cfg.param_stream_chunk_size == 0
        || cfg.param_max_bytes == 0
        || cfg.glitch_every_chunks == 0
    {
        return Err("chunk/bytes/glitch values must be non-zero".into());
    }
    Ok(cfg)
}

fn choose_execution_executable(
    executables: &[SerializedExecutableBlob],
) -> Result<&SerializedExecutableBlob, Box<dyn Error>> {
    if let Some(exe) = executables.iter().find(|exe| exe.executable_type == 2) {
        return Ok(exe);
    }
    if let Some(exe) = executables.iter().find(|exe| exe.executable_type == 0) {
        return Ok(exe);
    }
    Err("no EXECUTION_ONLY or STAND_ALONE executable found".into())
}

fn choose_param_executable(
    executables: &[SerializedExecutableBlob],
) -> Result<&SerializedExecutableBlob, Box<dyn Error>> {
    executables
        .iter()
        .find(|exe| exe.executable_type == 1 && !exe.parameters_stream.is_empty())
        .ok_or_else(|| "no PARAMETER_CACHING executable with parameters_stream found".into())
}

fn choose_glitch_action(rng: &mut Lcg, aggressive: bool) -> GlitchAction {
    if aggressive {
        match rng.next_range(7) {
            0 => GlitchAction::PollEvent,
            1 => GlitchAction::PollInterrupt,
            2 => GlitchAction::ReadRunControl,
            3 => GlitchAction::WriteRunControlOne,
            4 => GlitchAction::ReadScuCtrl,
            5 => GlitchAction::SleepJitter,
            _ => GlitchAction::WriteRunControlZero,
        }
    } else {
        match rng.next_range(6) {
            0 => GlitchAction::PollEvent,
            1 => GlitchAction::PollInterrupt,
            2 => GlitchAction::ReadRunControl,
            3 => GlitchAction::WriteRunControlOne,
            4 => GlitchAction::ReadScuCtrl,
            _ => GlitchAction::SleepJitter,
        }
    }
}

fn apply_glitch(
    driver: &EdgeTpuUsbDriver,
    rng: &mut Lcg,
    aggressive: bool,
    mode: GlitchMode,
    idx: usize,
) {
    let action = match mode {
        GlitchMode::Mixed => choose_glitch_action(rng, aggressive),
        GlitchMode::ReadOnly => match rng.next_range(5) {
            0 => GlitchAction::PollEvent,
            1 => GlitchAction::PollInterrupt,
            2 => GlitchAction::ReadRunControl,
            3 => GlitchAction::ReadScuCtrl,
            _ => GlitchAction::SleepJitter,
        },
        GlitchMode::RunControlOnly => {
            if aggressive && rng.next_range(2) == 0 {
                GlitchAction::WriteRunControlZero
            } else {
                GlitchAction::WriteRunControlOne
            }
        }
    };
    match action {
        GlitchAction::PollEvent => {
            let r = driver.read_event_packet_with_timeout(Duration::from_millis(1));
            println!(
                "  glitch[{idx}] PollEvent => {}",
                if r.is_ok() { "ok" } else { "timeout/err" }
            );
        }
        GlitchAction::PollInterrupt => {
            let r = driver.read_interrupt_packet_with_timeout(Duration::from_millis(1));
            println!(
                "  glitch[{idx}] PollInterrupt => {}",
                if r.is_ok() { "ok" } else { "timeout/err" }
            );
        }
        GlitchAction::ReadRunControl => match driver.vendor_read64(0x00044018) {
            Ok(v) => println!("  glitch[{idx}] ReadRunControl => 0x{v:016x}"),
            Err(e) => println!("  glitch[{idx}] ReadRunControl => err: {e}"),
        },
        GlitchAction::WriteRunControlOne => match driver.vendor_write64(0x00044018, 1) {
            Ok(_) => println!("  glitch[{idx}] WriteRunControlOne => ok"),
            Err(e) => println!("  glitch[{idx}] WriteRunControlOne => err: {e}"),
        },
        GlitchAction::ReadScuCtrl => match driver.vendor_read32(0x0001a30c) {
            Ok(v) => println!("  glitch[{idx}] ReadScuCtrl => 0x{v:08x}"),
            Err(e) => println!("  glitch[{idx}] ReadScuCtrl => err: {e}"),
        },
        GlitchAction::SleepJitter => {
            let us = 25 + (rng.next_range(976) as u64);
            std::thread::sleep(Duration::from_micros(us));
            println!("  glitch[{idx}] SleepJitter => {us}us");
        }
        GlitchAction::WriteRunControlZero => match driver.vendor_write64(0x00044018, 0) {
            Ok(_) => println!("  glitch[{idx}] WriteRunControlZero => ok"),
            Err(e) => println!("  glitch[{idx}] WriteRunControlZero => err: {e}"),
        },
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = parse_args()?;
    println!(
        "Fuzz config: seed={} aggressive={} glitch_mode={:?} glitch_every_chunks={} glitch_budget={} param_max_bytes={} param_stream_chunk_size={} inject_rc0={:?} inject_rc1={:?}",
        cfg.seed,
        cfg.aggressive,
        cfg.glitch_mode,
        cfg.glitch_every_chunks,
        cfg.glitch_budget,
        cfg.param_max_bytes,
        cfg.param_stream_chunk_size,
        cfg.inject_runcontrol0_chunks,
        cfg.inject_runcontrol1_chunks
    );

    let model_bytes = std::fs::read(&cfg.model_path)?;
    let executables = extract_serialized_executables_from_tflite(&model_bytes)?;
    let run_exe = choose_execution_executable(&executables)?;
    let param_exe = choose_param_executable(&executables)?;
    let stream_len = cfg.param_max_bytes.min(param_exe.parameters_stream.len());

    println!(
        "Model {} => exec_type={} exec_instr_chunks={} param_bytes={} stream_len={}",
        cfg.model_path,
        run_exe.executable_type,
        run_exe.instruction_bitstreams.len(),
        param_exe.parameters_stream.len(),
        stream_len
    );

    let mut driver =
        EdgeTpuUsbDriver::open_first_prefer_runtime(Duration::from_millis(cfg.timeout_ms))?;
    driver.set_descriptor_chunk_size(cfg.descriptor_chunk_size)?;
    let mut info = driver.device_info();
    println!(
        "Device: bus={} addr={} id={:04x}:{:04x} runtime={}",
        info.bus,
        info.address,
        info.vendor_id,
        info.product_id,
        info.is_runtime()
    );

    if !info.is_runtime() {
        let firmware_path = cfg
            .firmware_path
            .as_ref()
            .ok_or("device is in boot mode; pass --firmware PATH")?;
        let firmware = std::fs::read(firmware_path)?;
        println!("Uploading firmware: {} bytes", firmware.len());
        driver.upload_firmware_single_ep(&firmware)?;
        let _ = driver.reset_device();
        drop(driver);
        std::thread::sleep(Duration::from_secs(3));
        driver =
            EdgeTpuUsbDriver::open_first_prefer_runtime(Duration::from_millis(cfg.timeout_ms))?;
        driver.set_descriptor_chunk_size(cfg.descriptor_chunk_size)?;
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

    if !cfg.skip_setup {
        let setup_steps: Vec<_> = EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE
            .iter()
            .copied()
            .filter(|step| step.direction == VendorDirection::Out)
            .collect();
        println!("Applying setup steps: {}", setup_steps.len());
        driver.apply_vendor_steps(&setup_steps, false)?;
    } else {
        println!("Skipping setup sequence");
    }

    for (idx, chunk) in run_exe.instruction_bitstreams.iter().enumerate() {
        println!("Bootstrap EXEC instr chunk {idx} ({} bytes)", chunk.len());
        driver.send_descriptor_payload_raw(0, chunk)?;
    }
    for (idx, chunk) in param_exe.instruction_bitstreams.iter().enumerate() {
        println!("Bootstrap PARAM instr chunk {idx} ({} bytes)", chunk.len());
        driver.send_descriptor_payload_raw(0, chunk)?;
    }

    driver.send_descriptor_header_raw(2, stream_len)?;
    let mut rng = Lcg::new(cfg.seed);
    let mut offset = 0usize;
    let mut chunk_idx = 0usize;
    let mut glitches = 0usize;
    while offset < stream_len {
        if cfg.inject_runcontrol0_chunks.contains(&chunk_idx) {
            match driver.vendor_write64(0x00044018, 0) {
                Ok(_) => println!("  inject chunk={chunk_idx} rc0 => ok"),
                Err(e) => println!("  inject chunk={chunk_idx} rc0 => err: {e}"),
            }
        }
        if cfg.inject_runcontrol1_chunks.contains(&chunk_idx) {
            match driver.vendor_write64(0x00044018, 1) {
                Ok(_) => println!("  inject chunk={chunk_idx} rc1 => ok"),
                Err(e) => println!("  inject chunk={chunk_idx} rc1 => err: {e}"),
            }
        }

        if glitches < cfg.glitch_budget && chunk_idx % cfg.glitch_every_chunks == 0 {
            apply_glitch(
                &driver,
                &mut rng,
                cfg.aggressive,
                cfg.glitch_mode,
                glitches,
            );
            glitches += 1;
        }

        let end = (offset + cfg.param_stream_chunk_size).min(stream_len);
        let chunk = &param_exe.parameters_stream[offset..end];
        if let Err(err) = driver.write_bulk_out_chunk(chunk) {
            println!(
                "FUZZ_RESULT stall offset={} chunk_idx={} glitches={} err={}",
                offset, chunk_idx, glitches, err
            );
            return Err("parameter stream stalled".into());
        }
        offset = end;
        chunk_idx += 1;
    }
    println!(
        "FUZZ_RESULT streamed_ok bytes={} chunks={} glitches={}",
        stream_len, chunk_idx, glitches
    );

    match driver.read_event_packet() {
        Ok(event) => println!(
            "Post-stream event: tag={} offset=0x{:016x} length={}",
            event.tag, event.offset, event.length
        ),
        Err(err) => println!("Post-stream event read failed: {err}"),
    }

    println!("Final invoke phase");
    for (idx, chunk) in run_exe.instruction_bitstreams.iter().enumerate() {
        println!("Run instr chunk {idx} ({} bytes)", chunk.len());
        driver.send_descriptor_payload_raw(0, chunk)?;
    }

    let mut input = vec![0u8; cfg.input_bytes];
    for (i, b) in input.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    driver.send_descriptor_payload_raw(1, &input)?;

    match driver.read_event_packet() {
        Ok(event) => println!(
            "Run event: tag={} offset=0x{:016x} length={}",
            event.tag, event.offset, event.length
        ),
        Err(err) => println!("Run event read failed: {err}"),
    }

    match driver.read_output_bytes(cfg.output_bytes) {
        Ok(output) => {
            let head_len = output.len().min(16);
            println!("Run output bytes={} head={:02x?}", output.len(), &output[..head_len]);
        }
        Err(err) => println!("Run output read failed: {err}"),
    }

    Ok(())
}
