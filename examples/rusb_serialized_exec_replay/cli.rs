use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParamAdmissionWaitMode {
    Event,
    Interrupt,
    Either,
    Both,
}

impl ParamAdmissionWaitMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ParamAdmissionWaitMode::Event => "event",
            ParamAdmissionWaitMode::Interrupt => "interrupt",
            ParamAdmissionWaitMode::Either => "either",
            ParamAdmissionWaitMode::Both => "both",
        }
    }

    pub(crate) fn needs_event(self) -> bool {
        matches!(
            self,
            ParamAdmissionWaitMode::Event
                | ParamAdmissionWaitMode::Either
                | ParamAdmissionWaitMode::Both
        )
    }

    pub(crate) fn needs_interrupt(self) -> bool {
        matches!(
            self,
            ParamAdmissionWaitMode::Interrupt
                | ParamAdmissionWaitMode::Either
                | ParamAdmissionWaitMode::Both
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParamGatePlacement {
    Before,
    After,
    Both,
}

impl ParamGatePlacement {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ParamGatePlacement::Before => "before",
            ParamGatePlacement::After => "after",
            ParamGatePlacement::Both => "both",
        }
    }

    pub(crate) fn run_before(self) -> bool {
        matches!(self, ParamGatePlacement::Before | ParamGatePlacement::Both)
    }

    pub(crate) fn run_after(self) -> bool {
        matches!(self, ParamGatePlacement::After | ParamGatePlacement::Both)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub(crate) model_path: String,
    pub(crate) family_profile: Option<String>,
    pub(crate) check_profile: bool,
    pub(crate) input_bytes: usize,
    pub(crate) output_bytes: usize,
    pub(crate) runs: usize,
    pub(crate) timeout_ms: u64,
    pub(crate) chunk_size: usize,
    pub(crate) setup: bool,
    pub(crate) setup_libedgetpu: bool,
    pub(crate) verify_setup_reads: bool,
    pub(crate) setup_include_reads: bool,
    pub(crate) reset_before_claim: bool,
    pub(crate) post_reset_sleep_ms: u64,
    pub(crate) firmware_path: Option<String>,
    pub(crate) input_file: Option<String>,
    pub(crate) weights_row_major_u8_file: Option<String>,
    pub(crate) weights_row_major_i8_file: Option<String>,
    pub(crate) weights_pattern_index_mod: bool,
    pub(crate) weights_pattern_modulus: usize,
    pub(crate) weights_pattern_signed_reinterpret: bool,
    pub(crate) exec_index: Option<usize>,
    pub(crate) skip_param_preload: bool,
    pub(crate) read_interrupt: bool,
    pub(crate) instructions_tag: u32,
    pub(crate) instruction_patch_spec: Option<String>,
    pub(crate) parameters_tag: u32,
    pub(crate) input_activations_tag: u32,
    pub(crate) param_stream_chunk_size: Option<usize>,
    pub(crate) param_stream_max_bytes: Option<usize>,
    pub(crate) param_stream_override_file: Option<String>,
    pub(crate) param_force_full_header_len: bool,
    pub(crate) param_read_event_every: usize,
    pub(crate) param_event_timeout_ms: u64,
    pub(crate) param_write_sleep_us: u64,
    pub(crate) param_descriptor_split_bytes: Option<usize>,
    pub(crate) param_drain_event_every_descriptors: usize,
    pub(crate) param_read_interrupt_every: usize,
    pub(crate) param_interrupt_timeout_ms: u64,
    pub(crate) param_a0d8_handshake: bool,
    pub(crate) param_a0d8_write_value: u32,
    pub(crate) param_prepost_bulk_in_reads: usize,
    pub(crate) param_prepost_bulk_in_size: usize,
    pub(crate) param_prepost_event_reads: usize,
    pub(crate) param_prepost_interrupt_reads: usize,
    pub(crate) param_prepost_timeout_ms: u64,
    pub(crate) param_async_bulk_in_lanes: usize,
    pub(crate) param_async_bulk_in_size: usize,
    pub(crate) param_async_event_lanes: usize,
    pub(crate) param_async_interrupt_lanes: usize,
    pub(crate) param_async_timeout_ms: u64,
    pub(crate) param_gate_known_good_offsets: Vec<usize>,
    pub(crate) param_gate_window_start_bytes: Option<usize>,
    pub(crate) param_gate_window_end_bytes: Option<usize>,
    pub(crate) param_gate_window_step_bytes: Option<usize>,
    pub(crate) param_gate_placement: ParamGatePlacement,
    pub(crate) param_csr_snapshot_start_bytes: Option<usize>,
    pub(crate) param_csr_snapshot_end_bytes: Option<usize>,
    pub(crate) param_csr_snapshot_every_chunks: usize,
    pub(crate) param_csr_snapshot_on_error: bool,
    pub(crate) param_admission_wait_mode: Option<ParamAdmissionWaitMode>,
    pub(crate) param_admission_wait_timeout_ms: u64,
    pub(crate) param_admission_wait_poll_ms: u64,
    pub(crate) param_admission_wait_start_bytes: Option<usize>,
    pub(crate) param_admission_wait_end_bytes: Option<usize>,
    pub(crate) param_admission_wait_every_chunks: usize,
    pub(crate) param_admission_wait_strict: bool,
    pub(crate) param_submit_bulk_in_lanes: usize,
    pub(crate) param_submit_bulk_out: bool,
    pub(crate) param_submit_bulk_out_accept_partial: bool,
    pub(crate) param_submit_bulk_out_max_retries: usize,
    pub(crate) param_submit_bulk_out_depth: usize,
    pub(crate) param_submit_event_lanes: usize,
    pub(crate) param_submit_interrupt_lanes: usize,
    pub(crate) param_submit_buffer_size: usize,
    pub(crate) param_submit_timeout_ms: u64,
    pub(crate) param_submit_event_poll_ms: u64,
    pub(crate) param_submit_log_every: usize,
    pub(crate) param_submit_global_lanes: bool,
    pub(crate) bootstrap_known_good_order: bool,
    pub(crate) param_require_post_instr_event: bool,
    pub(crate) param_post_instr_event_timeout_ms: u64,
    pub(crate) param_interleave_window_bytes: Option<usize>,
    pub(crate) param_interleave_require_event: bool,
    pub(crate) param_interleave_event_timeout_ms: u64,
    pub(crate) param_csr_probe_offsets: Vec<usize>,
    pub(crate) param_poison_probe_offset: Option<usize>,
    pub(crate) script1_interleave: bool,
    pub(crate) script2_queue_probe: bool,
    pub(crate) script3_poison_diff: bool,
}

pub(crate) fn usage(program: &str) {
    println!("Usage: {program} (--model PATH | --family-profile PATH) [options]");
    println!("Options:");
    println!("  --model PATH              Compiled *_edgetpu.tflite model (or anchor from --family-profile)");
    println!("  --family-profile PATH     Dense family profile JSON (anchor model + optional tiered instruction patches + stored weight shape)");
    println!("  --check-profile           Validate resolved family-profile plan (model + patch sources + overlays) and exit before USB");
    println!("  --input-bytes N           Input activation bytes (default: 150528)");
    println!("  --output-bytes N          Output bytes to read from EP 0x81 (default: 1001)");
    println!("  --runs N                  Number of invoke attempts (default: 1)");
    println!("  --timeout-ms N            USB timeout ms (default: 6000)");
    println!("  --chunk-size N            Descriptor chunk size (default: 1048576)");
    println!("  --input-file PATH         Use raw input bytes from file instead of ramp");
    println!("  --weights-row-major-u8-file PATH  Build parameter stream from row-major u8 weights using family profile shape");
    println!("  --weights-row-major-i8-file PATH  Build parameter stream from row-major i8 weights using family profile shape");
    println!("  --weights-pattern-index-mod       Build row-major synthetic weights value=i%modulus (requires --family-profile)");
    println!("  --weights-pattern-modulus N       Modulus for --weights-pattern-index-mod (default: 251)");
    println!("  --weights-pattern-signed-reinterpret  Use signed reinterpret pattern ((i%modulus)-128) mod 256");
    println!("  --exec-index N            Force executable index from extracted list");
    println!("  --skip-setup              Skip edgetpuxray runtime setup sequence");
    println!(
        "  --setup-libedgetpu        Use libedgetpu's 95-step setup (default: edgetpuxray 52-step)"
    );
    println!("  --setup-include-reads     Include setup read steps (default: write-only)");
    println!("  --verify-setup-reads      Enforce exact readback match for setup sequence");
    println!("  --reset-before-claim      Reset USB device, reopen, then claim/setup (pyusb parity probe)");
    println!(
        "  --post-reset-sleep-ms N   Sleep after reset-before-claim before reopen (default: 600)"
    );
    println!("  --firmware PATH           apex_latest_single_ep.bin for boot-mode devices");
    println!("  --skip-param-preload      Do not send PARAMETER_CACHING executables");
    println!("  --read-interrupt          Read one interrupt packet after run");
    println!(
        "  --instructions-tag N      Descriptor tag for instructions (default: {})",
        DescriptorTag::Instructions.as_u32()
    );
    println!(
        "  --instruction-patch-spec PATH  Patch instruction payload bytes from spec file (<len> <offset> <value>)"
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
        "  --param-stream-override-file PATH  Replace extracted parameter stream bytes with raw file payload (length must match)"
    );
    println!(
        "  --param-force-full-header-len  Keep descriptor header length at full parameter payload even when stream is capped"
    );
    println!(
        "  --param-descriptor-split-bytes N  Split parameter stream across descriptor headers"
    );
    println!("  --param-read-event-every N  Poll EP 0x82 every N parameter chunks (0=off)");
    println!("  --param-read-interrupt-every N  Poll EP 0x83 every N parameter chunks (0=off)");
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
    println!("  --param-prepost-timeout-ms N  Timeout for pre/post reads (default: 1)");
    println!(
        "  --param-async-bulk-in-lanes N  Concurrent EP 0x81 read lanes during param stream (default: 0)"
    );
    println!("  --param-async-bulk-in-size N   EP 0x81 async read buffer size (default: 1024)");
    println!(
        "  --param-async-event-lanes N    Concurrent EP 0x82 read lanes during param stream (default: 0)"
    );
    println!(
        "  --param-async-interrupt-lanes N  Concurrent EP 0x83 read lanes during param stream (default: 0)"
    );
    println!("  --param-async-timeout-ms N     Timeout per async lane read attempt (default: 100)");
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
        "  --param-submit-bulk-out      Send EP 0x01 parameter chunks via one-shot libusb_submit_transfer"
    );
    println!(
        "  --param-submit-bulk-out-accept-partial  Advance stream offset by actual_length on short/timeout submit completions"
    );
    println!(
        "  --param-submit-bulk-out-max-retries N   Retries for zero-progress bulk_out submit attempts (default: 0)"
    );
    println!(
        "  --param-submit-bulk-out-depth N   Max in-flight EP 0x01 submit transfers (default: 1)"
    );
    println!(
        "  --param-submit-event-lanes N    libusb_submit_transfer lanes on EP 0x82 (default: 0)"
    );
    println!(
        "  --param-submit-interrupt-lanes N  libusb_submit_transfer lanes on EP 0x83 (default: 0)"
    );
    println!("  --param-submit-buffer-size N    Buffer size for submit lanes (default: 1024)");
    println!(
        "  --param-submit-timeout-ms N     Per-transfer timeout for submit lanes (default: 100)"
    );
    println!(
        "  --param-submit-event-poll-ms N  Event loop poll timeout for submit lanes (default: 1)"
    );
    println!("  --param-submit-log-every N      Callback log cadence per lane (0=off, default: 0)");
    println!(
        "  --param-submit-global-lanes   Start submit read lanes + event loop before first bulk-out and keep active across whole replay"
    );
    println!(
        "  --bootstrap-known-good-order  Preload PARAMETER_CACHING before EXECUTION_ONLY bootstrap chunks"
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

pub(crate) fn parse_u64_auto(value: &str) -> Result<u64, Box<dyn Error>> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        return Ok(u64::from_str_radix(hex, 16)?);
    }
    Ok(value.parse::<u64>()?)
}

pub(crate) fn parse_usize_auto(value: &str) -> Result<usize, Box<dyn Error>> {
    Ok(parse_u64_auto(value)? as usize)
}

pub(crate) fn parse_u8_auto(value: &str, flag_name: &str) -> Result<u8, Box<dyn Error>> {
    let parsed = parse_u64_auto(value)?;
    if parsed > u8::MAX as u64 {
        return Err(format!("{} value {} exceeds u8 max {}", flag_name, parsed, u8::MAX).into());
    }
    Ok(parsed as u8)
}

pub(crate) fn parse_usize_list_auto(value: &str) -> Result<Vec<usize>, Box<dyn Error>> {
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

pub(crate) fn parse_tag_u32(value: &str, flag_name: &str) -> Result<u32, Box<dyn Error>> {
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

pub(crate) fn parse_param_admission_wait_mode(
    value: &str,
) -> Result<ParamAdmissionWaitMode, Box<dyn Error>> {
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

pub(crate) fn parse_param_gate_placement(
    value: &str,
) -> Result<ParamGatePlacement, Box<dyn Error>> {
    match value {
        "before" => Ok(ParamGatePlacement::Before),
        "after" => Ok(ParamGatePlacement::After),
        "both" => Ok(ParamGatePlacement::Both),
        _ => Err(
            format!("invalid --param-gate-placement: {value} (expected before|after|both)").into(),
        ),
    }
}

pub(crate) fn parse_args() -> Result<Config, Box<dyn Error>> {
    parse_args_from(env::args().collect())
}

pub(crate) fn parse_args_from(args: Vec<String>) -> Result<Config, Box<dyn Error>> {
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
        family_profile: None,
        check_profile: false,
        input_bytes: 150_528,
        output_bytes: 1001,
        runs: 1,
        timeout_ms: 6000,
        chunk_size: 0x100000,
        setup: true,
        setup_libedgetpu: false,
        verify_setup_reads: false,
        setup_include_reads: false,
        reset_before_claim: false,
        post_reset_sleep_ms: 600,
        firmware_path: None,
        input_file: None,
        weights_row_major_u8_file: None,
        weights_row_major_i8_file: None,
        weights_pattern_index_mod: false,
        weights_pattern_modulus: 251,
        weights_pattern_signed_reinterpret: false,
        exec_index: None,
        skip_param_preload: false,
        read_interrupt: false,
        instructions_tag: DescriptorTag::Instructions.as_u32(),
        instruction_patch_spec: None,
        parameters_tag: DescriptorTag::Parameters.as_u32(),
        input_activations_tag: DescriptorTag::InputActivations.as_u32(),
        param_stream_chunk_size: None,
        param_stream_max_bytes: None,
        param_stream_override_file: None,
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
        param_submit_bulk_out: false,
        param_submit_bulk_out_accept_partial: false,
        param_submit_bulk_out_max_retries: 0,
        param_submit_bulk_out_depth: 1,
        param_submit_event_lanes: 0,
        param_submit_interrupt_lanes: 0,
        param_submit_buffer_size: 1024,
        param_submit_timeout_ms: 100,
        param_submit_event_poll_ms: 1,
        param_submit_log_every: 0,
        param_submit_global_lanes: false,
        bootstrap_known_good_order: false,
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
            "--family-profile" => {
                i += 1;
                config.family_profile = Some(
                    args.get(i)
                        .ok_or("--family-profile requires value")?
                        .to_string(),
                );
            }
            "--check-profile" => {
                config.check_profile = true;
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
            "--weights-row-major-u8-file" => {
                i += 1;
                config.weights_row_major_u8_file = Some(
                    args.get(i)
                        .ok_or("--weights-row-major-u8-file requires value")?
                        .to_string(),
                );
            }
            "--weights-row-major-i8-file" => {
                i += 1;
                config.weights_row_major_i8_file = Some(
                    args.get(i)
                        .ok_or("--weights-row-major-i8-file requires value")?
                        .to_string(),
                );
            }
            "--weights-pattern-index-mod" => {
                config.weights_pattern_index_mod = true;
            }
            "--weights-pattern-modulus" => {
                i += 1;
                config.weights_pattern_modulus = parse_usize_auto(
                    args.get(i)
                        .ok_or("--weights-pattern-modulus requires value")?,
                )?;
            }
            "--weights-pattern-signed-reinterpret" => {
                config.weights_pattern_signed_reinterpret = true;
            }
            "--exec-index" => {
                i += 1;
                config.exec_index = Some(parse_usize_auto(
                    args.get(i).ok_or("--exec-index requires value")?,
                )?);
            }
            "--skip-setup" => config.setup = false,
            "--setup-libedgetpu" => config.setup_libedgetpu = true,
            "--setup-include-reads" => config.setup_include_reads = true,
            "--verify-setup-reads" => config.verify_setup_reads = true,
            "--reset-before-claim" => config.reset_before_claim = true,
            "--post-reset-sleep-ms" => {
                i += 1;
                config.post_reset_sleep_ms =
                    parse_u64_auto(args.get(i).ok_or("--post-reset-sleep-ms requires value")?)?;
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
            "--instruction-patch-spec" => {
                i += 1;
                config.instruction_patch_spec = Some(
                    args.get(i)
                        .ok_or("--instruction-patch-spec requires value")?
                        .to_string(),
                );
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
                config.input_activations_tag = parse_tag_u32(
                    args.get(i).ok_or("--input-tag requires value")?,
                    "--input-tag",
                )?;
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
            "--param-stream-override-file" => {
                i += 1;
                config.param_stream_override_file = Some(
                    args.get(i)
                        .ok_or("--param-stream-override-file requires value")?
                        .to_string(),
                );
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
            "--param-submit-bulk-out" => {
                config.param_submit_bulk_out = true;
            }
            "--param-submit-bulk-out-accept-partial" => {
                config.param_submit_bulk_out_accept_partial = true;
            }
            "--param-submit-bulk-out-max-retries" => {
                i += 1;
                config.param_submit_bulk_out_max_retries = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-submit-bulk-out-max-retries requires value")?,
                )?;
            }
            "--param-submit-bulk-out-depth" => {
                i += 1;
                config.param_submit_bulk_out_depth = parse_usize_auto(
                    args.get(i)
                        .ok_or("--param-submit-bulk-out-depth requires value")?,
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
            "--param-submit-global-lanes" => {
                config.param_submit_global_lanes = true;
            }
            "--bootstrap-known-good-order" => {
                config.bootstrap_known_good_order = true;
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
                config.param_write_sleep_us =
                    parse_u64_auto(args.get(i).ok_or("--param-write-sleep-us requires value")?)?;
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

    if config.model_path.is_empty() && config.family_profile.is_none() {
        return Err("--model is required unless --family-profile provides anchor_model".into());
    }
    if config.check_profile && config.family_profile.is_none() {
        return Err("--check-profile requires --family-profile".into());
    }
    if let Some(path) = config.family_profile.as_ref() {
        if !Path::new(path).is_file() {
            return Err(format!(
                "--family-profile path does not exist or is not a file: {}",
                path
            )
            .into());
        }
    }
    if let Some(spec_path) = config.instruction_patch_spec.as_ref() {
        if !Path::new(spec_path).is_file() {
            return Err(format!(
                "--instruction-patch-spec path does not exist or is not a file: {}",
                spec_path
            )
            .into());
        }
    }
    if let Some(path) = config.param_stream_override_file.as_ref() {
        if !Path::new(path).is_file() {
            return Err(format!(
                "--param-stream-override-file path does not exist or is not a file: {}",
                path
            )
            .into());
        }
    }
    if let Some(path) = config.weights_row_major_u8_file.as_ref() {
        if !Path::new(path).is_file() {
            return Err(format!(
                "--weights-row-major-u8-file path does not exist or is not a file: {}",
                path
            )
            .into());
        }
    }
    if let Some(path) = config.weights_row_major_i8_file.as_ref() {
        if !Path::new(path).is_file() {
            return Err(format!(
                "--weights-row-major-i8-file path does not exist or is not a file: {}",
                path
            )
            .into());
        }
    }

    let weight_source_count = (config.weights_row_major_u8_file.is_some() as usize)
        + (config.weights_row_major_i8_file.is_some() as usize)
        + (config.weights_pattern_index_mod as usize);
    if weight_source_count > 1 {
        return Err("choose only one weight source: --weights-row-major-u8-file | --weights-row-major-i8-file | --weights-pattern-index-mod".into());
    }
    if weight_source_count > 0 && config.family_profile.is_none() {
        return Err("weight-source driven parameter generation requires --family-profile".into());
    }
    if weight_source_count > 0 && config.param_stream_override_file.is_some() {
        return Err("cannot combine generated weight-source parameter stream with --param-stream-override-file".into());
    }
    if config.weights_pattern_modulus == 0 || config.weights_pattern_modulus > 256 {
        return Err("--weights-pattern-modulus must be in [1,256]".into());
    }
    if (config.weights_pattern_signed_reinterpret || config.weights_pattern_modulus != 251)
        && !config.weights_pattern_index_mod
    {
        return Err("--weights-pattern-modulus/--weights-pattern-signed-reinterpret require --weights-pattern-index-mod".into());
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
    if config.param_submit_bulk_out_depth == 0 {
        return Err("--param-submit-bulk-out-depth must be >= 1".into());
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
        let end = config
            .param_csr_snapshot_end_bytes
            .ok_or("--param-csr-snapshot-end-bytes is required when using CSR snapshot controls")?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_snapshot() {
        let cfg =
            parse_args_from(vec!["prog".into(), "--model".into(), "m.tflite".into()]).unwrap();
        assert_eq!(cfg.input_bytes, 150_528);
        assert_eq!(cfg.output_bytes, 1001);
        assert_eq!(cfg.chunk_size, 0x100000);
        assert!(cfg.setup);
        assert_eq!(cfg.param_stream_max_bytes, None);
    }

    #[test]
    fn parse_representative_flags_snapshot() {
        let cfg = parse_args_from(vec![
            "prog".into(),
            "--model".into(),
            "m.tflite".into(),
            "--param-stream-max-bytes".into(),
            "4096".into(),
            "--param-force-full-header-len".into(),
            "--param-interleave-window-bytes".into(),
            "32768".into(),
            "--param-admission-wait-mode".into(),
            "either".into(),
            "--param-admission-wait-timeout-ms".into(),
            "10".into(),
        ])
        .unwrap();
        assert_eq!(cfg.param_stream_max_bytes, Some(4096));
        assert!(cfg.param_force_full_header_len);
        assert_eq!(cfg.param_interleave_window_bytes, Some(32768));
        assert_eq!(
            cfg.param_admission_wait_mode,
            Some(ParamAdmissionWaitMode::Either)
        );
    }
}
