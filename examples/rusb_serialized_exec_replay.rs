use coral_usb_oxidized::{
    executable_type_name, extract_serialized_executables_from_tflite, CoralError, DescriptorTag,
    EdgeTpuUsbDriver, SerializedExecutableBlob, VendorDirection,
    EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE,
};
use rusb::ffi as libusb;
use std::env;
use std::error::Error;
use std::ffi::CStr;
use std::os::raw::c_int;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamAdmissionWaitMode {
    Event,
    Interrupt,
    Either,
    Both,
}

impl ParamAdmissionWaitMode {
    fn as_str(self) -> &'static str {
        match self {
            ParamAdmissionWaitMode::Event => "event",
            ParamAdmissionWaitMode::Interrupt => "interrupt",
            ParamAdmissionWaitMode::Either => "either",
            ParamAdmissionWaitMode::Both => "both",
        }
    }

    fn needs_event(self) -> bool {
        matches!(self, ParamAdmissionWaitMode::Event | ParamAdmissionWaitMode::Either | ParamAdmissionWaitMode::Both)
    }

    fn needs_interrupt(self) -> bool {
        matches!(
            self,
            ParamAdmissionWaitMode::Interrupt
                | ParamAdmissionWaitMode::Either
                | ParamAdmissionWaitMode::Both
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamGatePlacement {
    Before,
    After,
    Both,
}

impl ParamGatePlacement {
    fn as_str(self) -> &'static str {
        match self {
            ParamGatePlacement::Before => "before",
            ParamGatePlacement::After => "after",
            ParamGatePlacement::Both => "both",
        }
    }

    fn run_before(self) -> bool {
        matches!(self, ParamGatePlacement::Before | ParamGatePlacement::Both)
    }

    fn run_after(self) -> bool {
        matches!(self, ParamGatePlacement::After | ParamGatePlacement::Both)
    }
}

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
    reset_before_claim: bool,
    post_reset_sleep_ms: u64,
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
    param_force_full_header_len: bool,
    param_read_event_every: usize,
    param_event_timeout_ms: u64,
    param_write_sleep_us: u64,
    param_descriptor_split_bytes: Option<usize>,
    param_drain_event_every_descriptors: usize,
    param_read_interrupt_every: usize,
    param_interrupt_timeout_ms: u64,
    param_a0d8_handshake: bool,
    param_a0d8_write_value: u32,
    param_prepost_bulk_in_reads: usize,
    param_prepost_bulk_in_size: usize,
    param_prepost_event_reads: usize,
    param_prepost_interrupt_reads: usize,
    param_prepost_timeout_ms: u64,
    param_async_bulk_in_lanes: usize,
    param_async_bulk_in_size: usize,
    param_async_event_lanes: usize,
    param_async_interrupt_lanes: usize,
    param_async_timeout_ms: u64,
    param_gate_known_good_offsets: Vec<usize>,
    param_gate_window_start_bytes: Option<usize>,
    param_gate_window_end_bytes: Option<usize>,
    param_gate_window_step_bytes: Option<usize>,
    param_gate_placement: ParamGatePlacement,
    param_csr_snapshot_start_bytes: Option<usize>,
    param_csr_snapshot_end_bytes: Option<usize>,
    param_csr_snapshot_every_chunks: usize,
    param_csr_snapshot_on_error: bool,
    param_admission_wait_mode: Option<ParamAdmissionWaitMode>,
    param_admission_wait_timeout_ms: u64,
    param_admission_wait_poll_ms: u64,
    param_admission_wait_start_bytes: Option<usize>,
    param_admission_wait_end_bytes: Option<usize>,
    param_admission_wait_every_chunks: usize,
    param_admission_wait_strict: bool,
    param_submit_bulk_in_lanes: usize,
    param_submit_event_lanes: usize,
    param_submit_interrupt_lanes: usize,
    param_submit_buffer_size: usize,
    param_submit_timeout_ms: u64,
    param_submit_event_poll_ms: u64,
    param_submit_log_every: usize,
    param_require_post_instr_event: bool,
    param_post_instr_event_timeout_ms: u64,
    param_interleave_window_bytes: Option<usize>,
    param_interleave_require_event: bool,
    param_interleave_event_timeout_ms: u64,
    param_csr_probe_offsets: Vec<usize>,
    param_poison_probe_offset: Option<usize>,
    script1_interleave: bool,
    script2_queue_probe: bool,
    script3_poison_diff: bool,
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
    println!("  --reset-before-claim      Reset USB device, reopen, then claim/setup (pyusb parity probe)");
    println!("  --post-reset-sleep-ms N   Sleep after reset-before-claim before reopen (default: 600)");
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
    println!(
        "  --param-force-full-header-len  Keep descriptor header length at full parameter payload even when stream is capped"
    );
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
        "  --param-a0d8-handshake      Enable pre-ingress handshake before each parameter descriptor"
    );
    println!(
        "  --param-a0d8-write-value N  Vendor write32 value for 0x0001a0d8 (default: 0x80000000)"
    );
    println!(
        "  --param-prepost-bulk-in-reads N  Number of pre/post EP 0x81 read attempts per descriptor (default: 0)"
    );
    println!(
        "  --param-prepost-bulk-in-size N   EP 0x81 pre/post read buffer size (default: 1024)"
    );
    println!(
        "  --param-prepost-event-reads N    Number of pre/post EP 0x82 read attempts per descriptor (default: 0)"
    );
    println!(
        "  --param-prepost-interrupt-reads N  Number of pre/post EP 0x83 read attempts per descriptor (default: 0)"
    );
    println!(
        "  --param-prepost-timeout-ms N  Timeout for pre/post reads (default: 1)"
    );
    println!(
        "  --param-async-bulk-in-lanes N  Concurrent EP 0x81 read lanes during param stream (default: 0)"
    );
    println!(
        "  --param-async-bulk-in-size N   EP 0x81 async read buffer size (default: 1024)"
    );
    println!(
        "  --param-async-event-lanes N    Concurrent EP 0x82 read lanes during param stream (default: 0)"
    );
    println!(
        "  --param-async-interrupt-lanes N  Concurrent EP 0x83 read lanes during param stream (default: 0)"
    );
    println!(
        "  --param-async-timeout-ms N     Timeout per async lane read attempt (default: 100)"
    );
    println!(
        "  --param-gate-known-good-offsets LIST  Pause stream at byte offsets and inject known-good control gate (comma-separated, e.g. 32768,40960)"
    );
    println!(
        "  --param-gate-window-start-bytes N  Start byte offset for repeated known-good gate injection"
    );
    println!(
        "  --param-gate-window-end-bytes N    End byte offset (exclusive) for repeated known-good gate injection"
    );
    println!(
        "  --param-gate-window-step-bytes N   Step bytes between repeated known-good gate injections"
    );
    println!(
        "  --param-gate-placement MODE   Run known-good gates before|after|both around each chunk (default: before)"
    );
    println!(
        "  --param-csr-snapshot-start-bytes N  Start byte offset (inclusive) for queue/runcontrol CSR snapshots"
    );
    println!(
        "  --param-csr-snapshot-end-bytes N    End byte offset (exclusive) for queue/runcontrol CSR snapshots"
    );
    println!(
        "  --param-csr-snapshot-every-chunks N  Snapshot cadence in chunks while in snapshot window (default: 1)"
    );
    println!(
        "  --param-csr-snapshot-on-error    Snapshot queue/runcontrol CSRs immediately before returning stream/gate errors"
    );
    println!(
        "  --param-admission-wait-mode MODE   Wait mode for stream admission checks: event|interrupt|either|both"
    );
    println!(
        "  --param-admission-wait-timeout-ms N  Max wait per admission check (default: 0/off)"
    );
    println!(
        "  --param-admission-wait-poll-ms N   Poll timeout per event/interrupt read during admission wait (default: 1)"
    );
    println!(
        "  --param-admission-wait-start-bytes N  Start byte offset (inclusive) for admission waits"
    );
    println!(
        "  --param-admission-wait-end-bytes N    End byte offset (exclusive) for admission waits"
    );
    println!(
        "  --param-admission-wait-every-chunks N  Trigger admission wait every N chunks in window (default: 1)"
    );
    println!(
        "  --param-admission-wait-strict     Fail immediately if admission wait is not satisfied"
    );
    println!(
        "  --param-submit-bulk-in-lanes N  libusb_submit_transfer lanes on EP 0x81 (default: 0)"
    );
    println!(
        "  --param-submit-event-lanes N    libusb_submit_transfer lanes on EP 0x82 (default: 0)"
    );
    println!(
        "  --param-submit-interrupt-lanes N  libusb_submit_transfer lanes on EP 0x83 (default: 0)"
    );
    println!(
        "  --param-submit-buffer-size N    Buffer size for submit lanes (default: 1024)"
    );
    println!(
        "  --param-submit-timeout-ms N     Per-transfer timeout for submit lanes (default: 100)"
    );
    println!(
        "  --param-submit-event-poll-ms N  Event loop poll timeout for submit lanes (default: 1)"
    );
    println!(
        "  --param-submit-log-every N      Callback log cadence per lane (0=off, default: 0)"
    );
    println!(
        "  --param-require-post-instr-event  Require EP 0x82 event after PARAMETER_CACHING instr chunks before param stream"
    );
    println!(
        "  --param-post-instr-event-timeout-ms N  Timeout for required post-instr event (default: 100)"
    );
    println!(
        "  --param-interleave-window-bytes N  Script1: split param payload into windows and inject instruction chunk between windows"
    );
    println!(
        "  --param-interleave-require-event   Script1: require EP 0x82 event after each injected interleave instruction chunk"
    );
    println!(
        "  --param-interleave-event-timeout-ms N  Timeout for Script1 interleave required event (default: 100)"
    );
    println!(
        "  --param-csr-probe-offsets LIST  Script2: capture queue/runcontrol CSR snapshot when param bytes cross offsets"
    );
    println!(
        "  --param-poison-probe-offset N   Script3: probe bridge/scu CSR liveness when param bytes cross offset"
    );
    println!(
        "  --script1-interleave            Enable Script1 profile defaults (interleave window=32768)"
    );
    println!(
        "  --script2-queue-probe          Enable Script2 profile defaults (csr probe=32000, chunk=256)"
    );
    println!(
        "  --script3-poison-diff          Enable Script3 profile defaults (poison probe offset=33024, chunk=256)"
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

fn parse_usize_list_auto(value: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut out = Vec::new();
    for raw in value.split(',') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(parse_usize_auto(trimmed)?);
    }
    if out.is_empty() {
        return Err("expected at least one numeric offset".into());
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
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

fn parse_param_admission_wait_mode(value: &str) -> Result<ParamAdmissionWaitMode, Box<dyn Error>> {
    match value {
        "event" => Ok(ParamAdmissionWaitMode::Event),
        "interrupt" => Ok(ParamAdmissionWaitMode::Interrupt),
        "either" => Ok(ParamAdmissionWaitMode::Either),
        "both" => Ok(ParamAdmissionWaitMode::Both),
        _ => Err(format!(
            "invalid --param-admission-wait-mode: {value} (expected event|interrupt|either|both)"
        )
        .into()),
    }
}

fn parse_param_gate_placement(value: &str) -> Result<ParamGatePlacement, Box<dyn Error>> {
    match value {
        "before" => Ok(ParamGatePlacement::Before),
        "after" => Ok(ParamGatePlacement::After),
        "both" => Ok(ParamGatePlacement::Both),
        _ => Err(
            format!("invalid --param-gate-placement: {value} (expected before|after|both)").into(),
        ),
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
        reset_before_claim: false,
        post_reset_sleep_ms: 600,
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
        param_force_full_header_len: false,
        param_read_event_every: 0,
        param_event_timeout_ms: 1,
        param_write_sleep_us: 0,
        param_descriptor_split_bytes: None,
        param_drain_event_every_descriptors: 0,
        param_read_interrupt_every: 0,
        param_interrupt_timeout_ms: 1,
        param_a0d8_handshake: false,
        param_a0d8_write_value: 0x8000_0000,
        param_prepost_bulk_in_reads: 0,
        param_prepost_bulk_in_size: 1024,
        param_prepost_event_reads: 0,
        param_prepost_interrupt_reads: 0,
        param_prepost_timeout_ms: 1,
        param_async_bulk_in_lanes: 0,
        param_async_bulk_in_size: 1024,
        param_async_event_lanes: 0,
        param_async_interrupt_lanes: 0,
        param_async_timeout_ms: 100,
        param_gate_known_good_offsets: Vec::new(),
        param_gate_window_start_bytes: None,
        param_gate_window_end_bytes: None,
        param_gate_window_step_bytes: None,
        param_gate_placement: ParamGatePlacement::Before,
        param_csr_snapshot_start_bytes: None,
        param_csr_snapshot_end_bytes: None,
        param_csr_snapshot_every_chunks: 1,
        param_csr_snapshot_on_error: false,
        param_admission_wait_mode: None,
        param_admission_wait_timeout_ms: 0,
        param_admission_wait_poll_ms: 1,
        param_admission_wait_start_bytes: None,
        param_admission_wait_end_bytes: None,
        param_admission_wait_every_chunks: 1,
        param_admission_wait_strict: false,
        param_submit_bulk_in_lanes: 0,
        param_submit_event_lanes: 0,
        param_submit_interrupt_lanes: 0,
        param_submit_buffer_size: 1024,
        param_submit_timeout_ms: 100,
        param_submit_event_poll_ms: 1,
        param_submit_log_every: 0,
        param_require_post_instr_event: false,
        param_post_instr_event_timeout_ms: 100,
        param_interleave_window_bytes: None,
        param_interleave_require_event: false,
        param_interleave_event_timeout_ms: 100,
        param_csr_probe_offsets: Vec::new(),
        param_poison_probe_offset: None,
        script1_interleave: false,
        script2_queue_probe: false,
        script3_poison_diff: false,
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
            "--reset-before-claim" => config.reset_before_claim = true,
            "--post-reset-sleep-ms" => {
                i += 1;
                config.post_reset_sleep_ms = parse_u64_auto(
                    args.get(i).ok_or("--post-reset-sleep-ms requires value")?,
                )?;
            }
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
            "--param-force-full-header-len" => {
                config.param_force_full_header_len = true;
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
            "--param-a0d8-handshake" => config.param_a0d8_handshake = true,
            "--param-a0d8-write-value" => {
                i += 1;
                config.param_a0d8_write_value = parse_tag_u32(
                    args.get(i)
                        .ok_or("--param-a0d8-write-value requires value")?,
                    "--param-a0d8-write-value",
                )?;
            }
            "--param-prepost-bulk-in-reads" => {
                i += 1;
                config.param_prepost_bulk_in_reads = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-prepost-bulk-in-reads requires value")?,
                )?;
            }
            "--param-prepost-bulk-in-size" => {
                i += 1;
                config.param_prepost_bulk_in_size = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-prepost-bulk-in-size requires value")?,
                )?;
            }
            "--param-prepost-event-reads" => {
                i += 1;
                config.param_prepost_event_reads = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-prepost-event-reads requires value")?,
                )?;
            }
            "--param-prepost-interrupt-reads" => {
                i += 1;
                config.param_prepost_interrupt_reads = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-prepost-interrupt-reads requires value")?,
                )?;
            }
            "--param-prepost-timeout-ms" => {
                i += 1;
                config.param_prepost_timeout_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-prepost-timeout-ms requires value")?,
                )?;
            }
            "--param-async-bulk-in-lanes" => {
                i += 1;
                config.param_async_bulk_in_lanes = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-async-bulk-in-lanes requires value")?,
                )?;
            }
            "--param-async-bulk-in-size" => {
                i += 1;
                config.param_async_bulk_in_size = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-async-bulk-in-size requires value")?,
                )?;
            }
            "--param-async-event-lanes" => {
                i += 1;
                config.param_async_event_lanes = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-async-event-lanes requires value")?,
                )?;
            }
            "--param-async-interrupt-lanes" => {
                i += 1;
                config.param_async_interrupt_lanes = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-async-interrupt-lanes requires value")?,
                )?;
            }
            "--param-async-timeout-ms" => {
                i += 1;
                config.param_async_timeout_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-async-timeout-ms requires value")?,
                )?;
            }
            "--param-gate-known-good-offsets" => {
                i += 1;
                config.param_gate_known_good_offsets = parse_usize_list_auto(
                    args.get(i)
                        .ok_or("--param-gate-known-good-offsets requires value")?,
                )?;
            }
            "--param-gate-window-start-bytes" => {
                i += 1;
                config.param_gate_window_start_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-gate-window-start-bytes requires value")?,
                )?);
            }
            "--param-gate-window-end-bytes" => {
                i += 1;
                config.param_gate_window_end_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-gate-window-end-bytes requires value")?,
                )?);
            }
            "--param-gate-window-step-bytes" => {
                i += 1;
                config.param_gate_window_step_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-gate-window-step-bytes requires value")?,
                )?);
            }
            "--param-gate-placement" => {
                i += 1;
                config.param_gate_placement = parse_param_gate_placement(
                    args.get(i).ok_or("--param-gate-placement requires value")?,
                )?;
            }
            "--param-csr-snapshot-start-bytes" => {
                i += 1;
                config.param_csr_snapshot_start_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-csr-snapshot-start-bytes requires value")?,
                )?);
            }
            "--param-csr-snapshot-end-bytes" => {
                i += 1;
                config.param_csr_snapshot_end_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-csr-snapshot-end-bytes requires value")?,
                )?);
            }
            "--param-csr-snapshot-every-chunks" => {
                i += 1;
                config.param_csr_snapshot_every_chunks = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-csr-snapshot-every-chunks requires value")?,
                )?;
            }
            "--param-csr-snapshot-on-error" => {
                config.param_csr_snapshot_on_error = true;
            }
            "--param-admission-wait-mode" => {
                i += 1;
                config.param_admission_wait_mode = Some(parse_param_admission_wait_mode(
                    args.get(i)
                        .ok_or("--param-admission-wait-mode requires value")?,
                )?);
            }
            "--param-admission-wait-timeout-ms" => {
                i += 1;
                config.param_admission_wait_timeout_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-admission-wait-timeout-ms requires value")?,
                )?;
            }
            "--param-admission-wait-poll-ms" => {
                i += 1;
                config.param_admission_wait_poll_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-admission-wait-poll-ms requires value")?,
                )?;
            }
            "--param-admission-wait-start-bytes" => {
                i += 1;
                config.param_admission_wait_start_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-admission-wait-start-bytes requires value")?,
                )?);
            }
            "--param-admission-wait-end-bytes" => {
                i += 1;
                config.param_admission_wait_end_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-admission-wait-end-bytes requires value")?,
                )?);
            }
            "--param-admission-wait-every-chunks" => {
                i += 1;
                config.param_admission_wait_every_chunks = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-admission-wait-every-chunks requires value")?,
                )?;
            }
            "--param-admission-wait-strict" => {
                config.param_admission_wait_strict = true;
            }
            "--param-submit-bulk-in-lanes" => {
                i += 1;
                config.param_submit_bulk_in_lanes = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-submit-bulk-in-lanes requires value")?,
                )?;
            }
            "--param-submit-event-lanes" => {
                i += 1;
                config.param_submit_event_lanes = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-submit-event-lanes requires value")?,
                )?;
            }
            "--param-submit-interrupt-lanes" => {
                i += 1;
                config.param_submit_interrupt_lanes = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-submit-interrupt-lanes requires value")?,
                )?;
            }
            "--param-submit-buffer-size" => {
                i += 1;
                config.param_submit_buffer_size = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-submit-buffer-size requires value")?,
                )?;
            }
            "--param-submit-timeout-ms" => {
                i += 1;
                config.param_submit_timeout_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-submit-timeout-ms requires value")?,
                )?;
            }
            "--param-submit-event-poll-ms" => {
                i += 1;
                config.param_submit_event_poll_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-submit-event-poll-ms requires value")?,
                )?;
            }
            "--param-submit-log-every" => {
                i += 1;
                config.param_submit_log_every = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-submit-log-every requires value")?,
                )?;
            }
            "--param-require-post-instr-event" => {
                config.param_require_post_instr_event = true;
            }
            "--param-post-instr-event-timeout-ms" => {
                i += 1;
                config.param_post_instr_event_timeout_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-post-instr-event-timeout-ms requires value")?,
                )?;
            }
            "--param-interleave-window-bytes" => {
                i += 1;
                config.param_interleave_window_bytes = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-interleave-window-bytes requires value")?,
                )?);
            }
            "--param-interleave-require-event" => {
                config.param_interleave_require_event = true;
            }
            "--param-interleave-event-timeout-ms" => {
                i += 1;
                config.param_interleave_event_timeout_ms = parse_u64_auto(
                    args.get(i)
                        .ok_or("--param-interleave-event-timeout-ms requires value")?,
                )?;
            }
            "--param-csr-probe-offsets" => {
                i += 1;
                config.param_csr_probe_offsets = parse_usize_list_auto(
                    args.get(i)
                        .ok_or("--param-csr-probe-offsets requires value")?,
                )?;
            }
            "--param-poison-probe-offset" => {
                i += 1;
                config.param_poison_probe_offset = Some(parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-poison-probe-offset requires value")?,
                )?);
            }
            "--script1-interleave" => {
                config.script1_interleave = true;
            }
            "--script2-queue-probe" => {
                config.script2_queue_probe = true;
            }
            "--script3-poison-diff" => {
                config.script3_poison_diff = true;
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

    if config.script1_interleave && config.param_interleave_window_bytes.is_none() {
        config.param_interleave_window_bytes = Some(32_768);
    }
    if config.script2_queue_probe {
        if config.param_csr_probe_offsets.is_empty() {
            config.param_csr_probe_offsets = vec![32_000];
        }
        if config.param_stream_chunk_size.is_none() {
            config.param_stream_chunk_size = Some(256);
        }
        if config.param_csr_snapshot_start_bytes.is_none() {
            config.param_csr_snapshot_start_bytes = Some(30_720);
        }
        if config.param_csr_snapshot_end_bytes.is_none() {
            config.param_csr_snapshot_end_bytes = Some(33_792);
        }
        config.param_csr_snapshot_on_error = true;
    }
    if config.script3_poison_diff {
        if config.param_poison_probe_offset.is_none() {
            config.param_poison_probe_offset = Some(33_024);
        }
        if config.param_stream_chunk_size.is_none() {
            config.param_stream_chunk_size = Some(256);
        }
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
    if matches!(config.param_interleave_window_bytes, Some(0)) {
        return Err("--param-interleave-window-bytes must be >= 1".into());
    }
    if config.param_interleave_require_event && config.param_interleave_window_bytes.is_none() {
        return Err(
            "set --param-interleave-window-bytes (or --script1-interleave) when using --param-interleave-require-event"
                .into(),
        );
    }
    if matches!(config.param_descriptor_split_bytes, Some(0)) {
        return Err("--param-descriptor-split-bytes must be >= 1".into());
    }
    if config.param_force_full_header_len && config.param_descriptor_split_bytes.is_some() {
        return Err(
            "--param-force-full-header-len currently requires no --param-descriptor-split-bytes"
                .into(),
        );
    }
    if config.param_prepost_bulk_in_size == 0 {
        return Err("--param-prepost-bulk-in-size must be >= 1".into());
    }
    if config.param_async_bulk_in_size == 0 {
        return Err("--param-async-bulk-in-size must be >= 1".into());
    }
    if config.param_submit_buffer_size == 0 {
        return Err("--param-submit-buffer-size must be >= 1".into());
    }
    if config.param_submit_buffer_size > c_int::MAX as usize {
        return Err(format!(
            "--param-submit-buffer-size exceeds i32 max: {} > {}",
            config.param_submit_buffer_size,
            c_int::MAX
        )
        .into());
    }
    if config.param_submit_event_poll_ms == 0 {
        return Err("--param-submit-event-poll-ms must be >= 1".into());
    }
    let has_window_gate_config = config.param_gate_window_start_bytes.is_some()
        || config.param_gate_window_end_bytes.is_some()
        || config.param_gate_window_step_bytes.is_some();
    if has_window_gate_config {
        let start = config.param_gate_window_start_bytes.ok_or(
            "--param-gate-window-start-bytes is required when using param-gate-window controls",
        )?;
        let end = config.param_gate_window_end_bytes.ok_or(
            "--param-gate-window-end-bytes is required when using param-gate-window controls",
        )?;
        let step = config.param_gate_window_step_bytes.ok_or(
            "--param-gate-window-step-bytes is required when using param-gate-window controls",
        )?;
        if start >= end {
            return Err(format!(
                "--param-gate-window-start-bytes ({start}) must be < --param-gate-window-end-bytes ({end})"
            )
            .into());
        }
        if step == 0 {
            return Err("--param-gate-window-step-bytes must be >= 1".into());
        }
    }
    let has_csr_snapshot_config = config.param_csr_snapshot_start_bytes.is_some()
        || config.param_csr_snapshot_end_bytes.is_some()
        || config.param_csr_snapshot_every_chunks != 1
        || config.param_csr_snapshot_on_error;
    if has_csr_snapshot_config {
        let start = config.param_csr_snapshot_start_bytes.ok_or(
            "--param-csr-snapshot-start-bytes is required when using CSR snapshot controls",
        )?;
        let end = config.param_csr_snapshot_end_bytes.ok_or(
            "--param-csr-snapshot-end-bytes is required when using CSR snapshot controls",
        )?;
        if start >= end {
            return Err(format!(
                "--param-csr-snapshot-start-bytes ({start}) must be < --param-csr-snapshot-end-bytes ({end})"
            )
            .into());
        }
        if config.param_csr_snapshot_every_chunks == 0 {
            return Err("--param-csr-snapshot-every-chunks must be >= 1".into());
        }
    }
    if config.param_admission_wait_poll_ms == 0 {
        return Err("--param-admission-wait-poll-ms must be >= 1".into());
    }
    let has_admission_window_config = config.param_admission_wait_start_bytes.is_some()
        || config.param_admission_wait_end_bytes.is_some();
    if config.param_admission_wait_mode.is_some() {
        if config.param_admission_wait_timeout_ms == 0 {
            return Err(
                "--param-admission-wait-timeout-ms must be >= 1 when --param-admission-wait-mode is set"
                    .into(),
            );
        }
        if config.param_admission_wait_every_chunks == 0 {
            return Err("--param-admission-wait-every-chunks must be >= 1".into());
        }
        if has_admission_window_config {
            let start = config.param_admission_wait_start_bytes.ok_or(
                "--param-admission-wait-start-bytes is required when using admission wait window",
            )?;
            let end = config.param_admission_wait_end_bytes.ok_or(
                "--param-admission-wait-end-bytes is required when using admission wait window",
            )?;
            if start >= end {
                return Err(format!(
                    "--param-admission-wait-start-bytes ({start}) must be < --param-admission-wait-end-bytes ({end})"
                )
                .into());
            }
        }
    } else if config.param_admission_wait_timeout_ms > 0
        || has_admission_window_config
        || config.param_admission_wait_every_chunks != 1
        || config.param_admission_wait_strict
    {
        return Err(
            "set --param-admission-wait-mode when using --param-admission-wait-* controls".into(),
        );
    }
    if has_param_async_lanes(&config) && has_param_submit_lanes(&config) {
        return Err(
            "choose either thread-blocking async lanes (--param-async-*) or libusb submit lanes (--param-submit-*), not both"
                .into(),
        );
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

        driver
            .vendor_write32(offset, value)
            .map_err(|err| format!("{}: gate write 0x{offset:08x}=0x{value:08x} failed: {}", phase_label, err))?;
    }

    let gate_writes = [
        (0x0001_a500_u32, 0x0000_0001_u32),
        (0x0001_a600_u32, 0x0000_0001_u32),
        (0x0001_a558_u32, 0x0000_0003_u32),
        (0x0001_a658_u32, 0x0000_0003_u32),
    ];
    for (offset, value) in gate_writes {
        driver
            .vendor_write32(offset, value)
            .map_err(|err| format!("{}: gate write 0x{offset:08x}=0x{value:08x} failed: {}", phase_label, err))?;
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
        .map_err(|err| format!("{}: gate write 0x0001a0d8=0x{:08x} failed: {}", phase_label, config.param_a0d8_write_value, err))?;

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
    config.param_csr_snapshot_start_bytes.is_some()
        && config.param_csr_snapshot_end_bytes.is_some()
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
    while *probe_cursor < probe_offsets.len() && param_bytes_written >= probe_offsets[*probe_cursor] {
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
    config.param_submit_bulk_in_lanes > 0
        || config.param_submit_event_lanes > 0
        || config.param_submit_interrupt_lanes > 0
}

#[derive(Debug, Clone, Copy)]
enum ParamSubmitLaneKind {
    BulkIn,
    Event,
    Interrupt,
}

impl ParamSubmitLaneKind {
    fn as_str(self) -> &'static str {
        match self {
            ParamSubmitLaneKind::BulkIn => "bulk_in",
            ParamSubmitLaneKind::Event => "event",
            ParamSubmitLaneKind::Interrupt => "interrupt",
        }
    }

    fn endpoint(self) -> u8 {
        match self {
            ParamSubmitLaneKind::BulkIn => 0x81,
            ParamSubmitLaneKind::Event => 0x82,
            ParamSubmitLaneKind::Interrupt => 0x83,
        }
    }
}

#[derive(Default)]
struct ParamSubmitLaneCounters {
    callbacks: AtomicUsize,
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

        if user_data.stop.load(Ordering::Relaxed) {
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

    let mut buffer = vec![0u8; config.param_submit_buffer_size];
    let submitted = Arc::new(AtomicBool::new(false));
    let counters = Arc::new(ParamSubmitLaneCounters::default());
    let user_data = Box::new(ParamSubmitLaneUserData {
        kind,
        lane_idx,
        stop,
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
            ParamSubmitLaneKind::BulkIn | ParamSubmitLaneKind::Event => {
                libusb::libusb_fill_bulk_transfer(
                    transfer,
                    driver.raw_libusb_handle(),
                    kind.endpoint(),
                    buffer.as_mut_ptr(),
                    buffer.len() as c_int,
                    param_submit_transfer_callback,
                    user_data_ptr as *mut _,
                    config.param_submit_timeout_ms.min(u32::MAX as u64) as u32,
                )
            }
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
            "      {}: async submit lane {}#{} callbacks={} completed={} timed_out={} cancelled={} stall={} no_device={} overflow={} errors={} submit_errors={} submitted={}",
            phase_label,
            lane.kind.as_str(),
            lane.lane_idx,
            counters.callbacks.load(Ordering::Relaxed),
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
        "      {}: libusb submit lanes bulk_in={} event={} interrupt={} buffer_size={} transfer_timeout_ms={} event_poll_ms={} log_every={}",
        phase_label,
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
            if lanes.iter().all(|lane| !lane.submitted.load(Ordering::Relaxed)) {
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
                match driver.read_output_bytes_with_timeout(config.param_async_bulk_in_size, timeout) {
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
                run_pending_known_good_gates(
                    driver,
                    config,
                    phase_label,
                    param_bytes_written,
                    gate_offsets,
                    &mut gate_cursor,
                    window_gate,
                    &mut next_window_gate_offset,
                    &mut window_gate_count,
                )?;
            }
            let end = (descriptor_offset + stream_chunk_size).min(descriptor_end);
            let chunk = &payload[descriptor_offset..end];
            if let Err(err) = driver.write_bulk_out_chunk(chunk) {
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
                run_pending_known_good_gates(
                    driver,
                    config,
                    phase_label,
                    param_bytes_written,
                    gate_offsets,
                    &mut gate_cursor,
                    window_gate,
                    &mut next_window_gate_offset,
                    &mut window_gate_count,
                )?;
            }

            if let Some(mode) = admission_mode {
                let in_admission_window =
                    param_bytes_written >= admission_wait_start && param_bytes_written < admission_wait_end;
                if in_admission_window && (global_chunk_idx % admission_wait_every_chunks == 0) {
                    let admission_ok = wait_for_param_admission(
                        driver,
                        phase_label,
                        mode,
                        global_chunk_idx,
                        param_bytes_written,
                        admission_wait_timeout,
                        admission_wait_poll,
                    )?;
                    if !admission_ok {
                        if config.param_admission_wait_strict {
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
            phase_label, segment_idx + 1, segment_start, segment_end
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
        driver.send_descriptor_payload_raw(config.instructions_tag, instr_chunk)?;

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

    let stream_len = config
        .param_stream_max_bytes
        .map(|max| max.min(payload.len()))
        .unwrap_or(payload.len());
    let header_total_len = if config.param_force_full_header_len {
        payload.len()
    } else {
        stream_len
    };
    let stream_chunk_size = config.param_stream_chunk_size.unwrap_or(config.chunk_size);
    let descriptor_split_size = config.param_descriptor_split_bytes.unwrap_or(stream_len.max(1));
    let poll_timeout = Duration::from_millis(config.param_event_timeout_ms);
    let interrupt_poll_timeout = Duration::from_millis(config.param_interrupt_timeout_ms);

    println!(
        "    {}: streaming parameters len={} header_len={} chunk={} desc_split={} event_poll_every={} intr_poll_every={} drain_desc_every={} sleep_us={} tag={} handshake={} a0d8_write=0x{:08x} gate_offsets={:?} gate_window_start={:?} gate_window_end={:?} gate_window_step={:?} gate_placement={} csr_snapshot_start={:?} csr_snapshot_end={:?} csr_snapshot_every_chunks={} csr_snapshot_on_error={} admission_mode={} admission_timeout_ms={} admission_poll_ms={} admission_start={:?} admission_end={:?} admission_every_chunks={} admission_strict={} prepost_bulk_reads={} prepost_bulk_size={} prepost_event_reads={} prepost_intr_reads={} prepost_timeout_ms={} async_bulk_lanes={} async_bulk_size={} async_event_lanes={} async_intr_lanes={} async_timeout_ms={} submit_bulk_lanes={} submit_event_lanes={} submit_intr_lanes={} submit_buf_size={} submit_timeout_ms={} submit_event_poll_ms={} submit_log_every={} interleave_window={:?} interleave_require_event={} interleave_event_timeout_ms={} csr_probe_offsets={:?} poison_probe_offset={:?}",
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
            "Parameter stream controls: chunk={:?} max_bytes={:?} force_full_header_len={} desc_split={:?} event_poll_every={} intr_poll_every={} drain_desc_every={} event_timeout_ms={} intr_timeout_ms={} sleep_us={} handshake={} a0d8_write=0x{:08x} gate_offsets={:?} gate_window_start={:?} gate_window_end={:?} gate_window_step={:?} gate_placement={} csr_snapshot_start={:?} csr_snapshot_end={:?} csr_snapshot_every_chunks={} csr_snapshot_on_error={} admission_mode={} admission_timeout_ms={} admission_poll_ms={} admission_start={:?} admission_end={:?} admission_every_chunks={} admission_strict={} prepost_bulk_reads={} prepost_bulk_size={} prepost_event_reads={} prepost_intr_reads={} prepost_timeout_ms={} async_bulk_lanes={} async_bulk_size={} async_event_lanes={} async_intr_lanes={} async_timeout_ms={} submit_bulk_lanes={} submit_event_lanes={} submit_intr_lanes={} submit_buf_size={} submit_timeout_ms={} submit_event_poll_ms={} submit_log_every={} require_post_instr_event={} post_instr_event_timeout_ms={} interleave_window={:?} interleave_require_event={} interleave_event_timeout_ms={} csr_probe_offsets={:?} poison_probe_offset={:?} script1={} script2={} script3={}",
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
            config.param_submit_bulk_in_lanes,
            config.param_submit_event_lanes,
            config.param_submit_interrupt_lanes,
            config.param_submit_buffer_size,
            config.param_submit_timeout_ms,
            config.param_submit_event_poll_ms,
            config.param_submit_log_every,
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
                    let interleave_chunks: &[Vec<u8>] = if !exe.instruction_bitstreams.is_empty() {
                        &exe.instruction_bitstreams
                    } else {
                        &run_exe.instruction_bitstreams
                    };
                    send_parameter_payload_with_instruction_interleave(
                        &driver,
                        &config,
                        &exe.parameters_stream,
                        "bootstrap param",
                        window_bytes,
                        interleave_chunks,
                    )?;
                } else {
                    send_parameter_payload(&driver, &config, &exe.parameters_stream, "bootstrap param")?;
                }
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
                    let interleave_chunks: &[Vec<u8>] = if !exe.instruction_bitstreams.is_empty() {
                        &exe.instruction_bitstreams
                    } else {
                        &run_exe.instruction_bitstreams
                    };
                    send_parameter_payload_with_instruction_interleave(
                        &driver,
                        &config,
                        &exe.parameters_stream,
                        "preload param",
                        window_bytes,
                        interleave_chunks,
                    )?;
                } else {
                    send_parameter_payload(&driver, &config, &exe.parameters_stream, "preload param")?;
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
