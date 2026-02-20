use coral_usb_oxidized::{CoralDevice, EdgeTPUDelegate};
use std::env;
use std::ffi::CString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::ptr;
use std::time::Instant;

pub enum TfLiteModel {}
pub enum TfLiteInterpreter {}
pub enum TfLiteInterpreterOptions {}
pub enum TfLiteTensor {}
pub enum TfLiteDelegate {}

extern "C" {
    fn TfLiteModelCreateFromFile(model_path: *const c_char) -> *mut TfLiteModel;
    fn TfLiteModelDelete(model: *mut TfLiteModel);

    fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);
    fn TfLiteInterpreterOptionsAddDelegate(
        options: *mut TfLiteInterpreterOptions,
        delegate: *mut TfLiteDelegate,
    );

    fn TfLiteInterpreterCreate(
        model: *mut TfLiteModel,
        options: *mut TfLiteInterpreterOptions,
    ) -> *mut TfLiteInterpreter;
    fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);
    fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterInvoke(interpreter: *mut TfLiteInterpreter) -> i32;

    fn TfLiteInterpreterGetInputTensor(
        interpreter: *mut TfLiteInterpreter,
        input_index: i32,
    ) -> *mut TfLiteTensor;
    fn TfLiteInterpreterGetOutputTensor(
        interpreter: *mut TfLiteInterpreter,
        output_index: i32,
    ) -> *mut TfLiteTensor;

    fn TfLiteTensorByteSize(tensor: *mut TfLiteTensor) -> usize;
    fn TfLiteTensorCopyFromBuffer(
        tensor: *mut TfLiteTensor,
        input_data: *const c_void,
        input_data_size: usize,
    ) -> i32;
    fn TfLiteTensorCopyToBuffer(
        tensor: *mut TfLiteTensor,
        output_data: *mut c_void,
        output_data_size: usize,
    ) -> i32;
}

#[derive(Clone, Copy)]
enum Backend {
    CpuInt8,
    EdgeTpuInt8,
}

impl Backend {
    fn as_str(self) -> &'static str {
        match self {
            Backend::CpuInt8 => "cpu_int8",
            Backend::EdgeTpuInt8 => "edgetpu_int8",
        }
    }
}

struct CliConfig {
    sanity_model: String,
    matrix_model: String,
    runs: usize,
    warmup: usize,
    repeats: usize,
    csv_path: String,
    seed: u64,
}

struct Workload<'a> {
    id: &'static str,
    model_path: &'a str,
}

struct ScenarioStats {
    invoke_p50_ms: f64,
    invoke_p95_ms: f64,
    invoke_mean_ms: f64,
    e2e_p50_ms: f64,
    e2e_p95_ms: f64,
    e2e_mean_ms: f64,
}

struct CsvRow {
    workload: String,
    backend: String,
    repeat: usize,
    invoke_p50_ms: f64,
    invoke_p95_ms: f64,
    invoke_mean_ms: f64,
    e2e_p50_ms: f64,
    e2e_p95_ms: f64,
    e2e_mean_ms: f64,
    status: String,
    error: String,
}

struct RawInterpreter {
    model: *mut TfLiteModel,
    interpreter: *mut TfLiteInterpreter,
}

impl RawInterpreter {
    fn new(model_path: &str, delegate: *mut TfLiteDelegate) -> Result<Self, String> {
        let c_model_path = CString::new(model_path)
            .map_err(|_| format!("model path contains embedded NUL byte: {}", model_path))?;

        let model = unsafe { TfLiteModelCreateFromFile(c_model_path.as_ptr()) };
        if model.is_null() {
            return Err(format!(
                "TfLiteModelCreateFromFile failed for {}",
                model_path
            ));
        }

        let options = unsafe { TfLiteInterpreterOptionsCreate() };
        if options.is_null() {
            unsafe {
                TfLiteModelDelete(model);
            }
            return Err("TfLiteInterpreterOptionsCreate failed".to_string());
        }

        if !delegate.is_null() {
            unsafe {
                TfLiteInterpreterOptionsAddDelegate(options, delegate);
            }
        }

        let interpreter = unsafe { TfLiteInterpreterCreate(model, options) };
        unsafe {
            TfLiteInterpreterOptionsDelete(options);
        }

        if interpreter.is_null() {
            unsafe {
                TfLiteModelDelete(model);
            }
            return Err("TfLiteInterpreterCreate failed".to_string());
        }

        let status = unsafe { TfLiteInterpreterAllocateTensors(interpreter) };
        if status != 0 {
            unsafe {
                TfLiteInterpreterDelete(interpreter);
                TfLiteModelDelete(model);
            }
            return Err(format!(
                "TfLiteInterpreterAllocateTensors failed: {}",
                status
            ));
        }

        Ok(Self { model, interpreter })
    }

    fn input_tensor(&self, input_index: i32) -> Result<*mut TfLiteTensor, String> {
        let tensor = unsafe { TfLiteInterpreterGetInputTensor(self.interpreter, input_index) };
        if tensor.is_null() {
            return Err(format!("input tensor {} not found", input_index));
        }
        Ok(tensor)
    }

    fn output_tensor(&self, output_index: i32) -> Result<*mut TfLiteTensor, String> {
        let tensor = unsafe { TfLiteInterpreterGetOutputTensor(self.interpreter, output_index) };
        if tensor.is_null() {
            return Err(format!("output tensor {} not found", output_index));
        }
        Ok(tensor)
    }

    fn invoke(&self) -> Result<(), String> {
        let status = unsafe { TfLiteInterpreterInvoke(self.interpreter) };
        if status != 0 {
            return Err(format!("TfLiteInterpreterInvoke failed: {}", status));
        }
        Ok(())
    }
}

impl Drop for RawInterpreter {
    fn drop(&mut self) {
        if !self.interpreter.is_null() {
            unsafe {
                TfLiteInterpreterDelete(self.interpreter);
            }
            self.interpreter = ptr::null_mut();
        }

        if !self.model.is_null() {
            unsafe {
                TfLiteModelDelete(self.model);
            }
            self.model = ptr::null_mut();
        }
    }
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} --sanity-model <path> --matrix-model <path> [--runs <n>] [--warmup <n>] [--repeats <n>] [--csv <path>] [--seed <u64>]"
    )
}

fn arg_value<'a>(args: &'a [String], idx: usize, flag: &str) -> Result<&'a str, String> {
    args.get(idx)
        .map(|v| v.as_str())
        .ok_or_else(|| format!("missing value for {}", flag))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid value for {}: {}", flag, value))
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("invalid value for {}: {}", flag, value))
}

fn parse_args() -> Result<CliConfig, String> {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "cpu_vs_edgetpu_mvp".to_string());

    let mut sanity_model: Option<String> = None;
    let mut matrix_model: Option<String> = None;
    let mut runs: usize = 100;
    let mut warmup: usize = 10;
    let mut repeats: usize = 3;
    let mut csv_path = "mvp_results.csv".to_string();
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sanity-model" => {
                i += 1;
                sanity_model = Some(arg_value(&args, i, "--sanity-model")?.to_string());
            }
            "--matrix-model" => {
                i += 1;
                matrix_model = Some(arg_value(&args, i, "--matrix-model")?.to_string());
            }
            "--runs" => {
                i += 1;
                runs = parse_usize(arg_value(&args, i, "--runs")?, "--runs")?;
            }
            "--warmup" => {
                i += 1;
                warmup = parse_usize(arg_value(&args, i, "--warmup")?, "--warmup")?;
            }
            "--repeats" => {
                i += 1;
                repeats = parse_usize(arg_value(&args, i, "--repeats")?, "--repeats")?;
            }
            "--csv" => {
                i += 1;
                csv_path = arg_value(&args, i, "--csv")?.to_string();
            }
            "--seed" => {
                i += 1;
                seed = parse_u64(arg_value(&args, i, "--seed")?, "--seed")?;
            }
            "--help" | "-h" => {
                return Err(usage(&program));
            }
            unknown => {
                return Err(format!(
                    "unknown argument: {}\n{}",
                    unknown,
                    usage(&program)
                ));
            }
        }

        i += 1;
    }

    let sanity_model = match sanity_model {
        Some(path) => path,
        None => {
            return Err(format!(
                "missing required --sanity-model\n{}",
                usage(&program)
            ))
        }
    };

    let matrix_model = match matrix_model {
        Some(path) => path,
        None => {
            return Err(format!(
                "missing required --matrix-model\n{}",
                usage(&program)
            ))
        }
    };

    Ok(CliConfig {
        sanity_model,
        matrix_model,
        runs,
        warmup,
        repeats,
        csv_path,
        seed,
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

fn deterministic_input(seed: u64, len: usize) -> Vec<u8> {
    let mut value = seed ^ (len as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = vec![0u8; len];
    for byte in &mut out {
        value = value.wrapping_mul(6364136223846793005).wrapping_add(1);
        *byte = (value >> 56) as u8;
    }
    out
}

fn run_iteration(
    interpreter: &RawInterpreter,
    input_tensor: *mut TfLiteTensor,
    output_tensor: *mut TfLiteTensor,
    input_data: &[u8],
    output_data: &mut [u8],
) -> Result<(f64, f64), String> {
    let e2e_start = Instant::now();

    let copy_in_status = unsafe {
        TfLiteTensorCopyFromBuffer(
            input_tensor,
            input_data.as_ptr() as *const c_void,
            input_data.len(),
        )
    };
    if copy_in_status != 0 {
        return Err(format!(
            "TfLiteTensorCopyFromBuffer failed: {}",
            copy_in_status
        ));
    }

    let invoke_start = Instant::now();
    interpreter.invoke()?;
    let invoke_ms = invoke_start.elapsed().as_secs_f64() * 1000.0;

    let copy_out_status = unsafe {
        TfLiteTensorCopyToBuffer(
            output_tensor,
            output_data.as_mut_ptr() as *mut c_void,
            output_data.len(),
        )
    };
    if copy_out_status != 0 {
        return Err(format!(
            "TfLiteTensorCopyToBuffer failed: {}",
            copy_out_status
        ));
    }

    let e2e_ms = e2e_start.elapsed().as_secs_f64() * 1000.0;
    Ok((invoke_ms, e2e_ms))
}

fn run_scenario(
    backend: Backend,
    model_path: &str,
    warmup: usize,
    runs: usize,
    seed: u64,
) -> Result<ScenarioStats, String> {
    if !Path::new(model_path).exists() {
        return Err(format!("model file not found: {}", model_path));
    }

    let mut delegate_ptr: *mut TfLiteDelegate = ptr::null_mut();
    let mut _delegate_guard: Option<EdgeTPUDelegate> = None;

    if let Backend::EdgeTpuInt8 = backend {
        let device = CoralDevice::new().map_err(|e| format!("CoralDevice::new failed: {}", e))?;
        let delegate = device
            .create_delegate()
            .map_err(|e| format!("create_delegate failed: {}", e))?;
        delegate_ptr = delegate.as_ptr() as *mut TfLiteDelegate;
        _delegate_guard = Some(delegate);
    }

    let interpreter = RawInterpreter::new(model_path, delegate_ptr)?;

    let input_tensor = interpreter.input_tensor(0)?;
    let output_tensor = interpreter.output_tensor(0)?;

    let input_len = unsafe { TfLiteTensorByteSize(input_tensor) };
    if input_len == 0 {
        return Err("input tensor byte size is zero".to_string());
    }

    let output_len = unsafe { TfLiteTensorByteSize(output_tensor) };
    if output_len == 0 {
        return Err("output tensor byte size is zero".to_string());
    }

    let input_data = deterministic_input(seed, input_len);
    let mut output_data = vec![0u8; output_len];

    for _ in 0..warmup {
        let _ = run_iteration(
            &interpreter,
            input_tensor,
            output_tensor,
            &input_data,
            &mut output_data,
        )?;
    }

    let mut invoke_samples = Vec::with_capacity(runs);
    let mut e2e_samples = Vec::with_capacity(runs);

    for _ in 0..runs {
        let (invoke_ms, e2e_ms) = run_iteration(
            &interpreter,
            input_tensor,
            output_tensor,
            &input_data,
            &mut output_data,
        )?;
        invoke_samples.push(invoke_ms);
        e2e_samples.push(e2e_ms);
    }

    let mut invoke_sorted = invoke_samples.clone();
    invoke_sorted.sort_by(|a, b| a.total_cmp(b));

    let mut e2e_sorted = e2e_samples.clone();
    e2e_sorted.sort_by(|a, b| a.total_cmp(b));

    Ok(ScenarioStats {
        invoke_p50_ms: percentile(&invoke_sorted, 0.50),
        invoke_p95_ms: percentile(&invoke_sorted, 0.95),
        invoke_mean_ms: mean(&invoke_samples),
        e2e_p50_ms: percentile(&e2e_sorted, 0.50),
        e2e_p95_ms: percentile(&e2e_sorted, 0.95),
        e2e_mean_ms: mean(&e2e_samples),
    })
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn write_csv(path: &str, rows: &[CsvRow]) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("failed to create CSV {}: {}", path, e))?;
    let mut writer = BufWriter::new(file);

    writer
        .write_all(
            b"workload,backend,repeat,invoke_p50_ms,invoke_p95_ms,invoke_mean_ms,e2e_p50_ms,e2e_p95_ms,e2e_mean_ms,status,error\n",
        )
        .map_err(|e| format!("failed writing CSV header: {}", e))?;

    for row in rows {
        let line = format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
            csv_escape(&row.workload),
            csv_escape(&row.backend),
            row.repeat,
            row.invoke_p50_ms,
            row.invoke_p95_ms,
            row.invoke_mean_ms,
            row.e2e_p50_ms,
            row.e2e_p95_ms,
            row.e2e_mean_ms,
            csv_escape(&row.status),
            csv_escape(&row.error),
        );
        writer
            .write_all(line.as_bytes())
            .map_err(|e| format!("failed writing CSV row: {}", e))?;
    }

    writer
        .flush()
        .map_err(|e| format!("failed flushing CSV {}: {}", path, e))
}

fn main() {
    let config = match parse_args() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("{}", err);
            std::process::exit(2);
        }
    };

    let workloads = [
        Workload {
            id: "sanity_model",
            model_path: &config.sanity_model,
        },
        Workload {
            id: "matrix_model",
            model_path: &config.matrix_model,
        },
    ];

    let backends = [Backend::CpuInt8, Backend::EdgeTpuInt8];
    let mut csv_rows = Vec::new();

    for workload in workloads {
        for backend in backends {
            for repeat in 1..=config.repeats {
                let result = run_scenario(
                    backend,
                    workload.model_path,
                    config.warmup,
                    config.runs,
                    config.seed,
                );

                match result {
                    Ok(stats) => {
                        println!(
                            "RESULT workload={} backend={} repeat={} invoke_p50_ms={:.3} invoke_p95_ms={:.3} e2e_p50_ms={:.3} status=ok",
                            workload.id,
                            backend.as_str(),
                            repeat,
                            stats.invoke_p50_ms,
                            stats.invoke_p95_ms,
                            stats.e2e_p50_ms,
                        );

                        csv_rows.push(CsvRow {
                            workload: workload.id.to_string(),
                            backend: backend.as_str().to_string(),
                            repeat,
                            invoke_p50_ms: stats.invoke_p50_ms,
                            invoke_p95_ms: stats.invoke_p95_ms,
                            invoke_mean_ms: stats.invoke_mean_ms,
                            e2e_p50_ms: stats.e2e_p50_ms,
                            e2e_p95_ms: stats.e2e_p95_ms,
                            e2e_mean_ms: stats.e2e_mean_ms,
                            status: "ok".to_string(),
                            error: String::new(),
                        });
                    }
                    Err(err) => {
                        println!(
                            "RESULT workload={} backend={} repeat={} invoke_p50_ms={:.3} invoke_p95_ms={:.3} e2e_p50_ms={:.3} status=failed",
                            workload.id,
                            backend.as_str(),
                            repeat,
                            0.0,
                            0.0,
                            0.0,
                        );

                        csv_rows.push(CsvRow {
                            workload: workload.id.to_string(),
                            backend: backend.as_str().to_string(),
                            repeat,
                            invoke_p50_ms: 0.0,
                            invoke_p95_ms: 0.0,
                            invoke_mean_ms: 0.0,
                            e2e_p50_ms: 0.0,
                            e2e_p95_ms: 0.0,
                            e2e_mean_ms: 0.0,
                            status: "failed".to_string(),
                            error: err,
                        });
                    }
                }
            }
        }
    }

    if let Err(err) = write_csv(&config.csv_path, &csv_rows) {
        eprintln!("{}", err);
        std::process::exit(1);
    }
}
