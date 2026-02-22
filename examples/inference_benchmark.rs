use coral_usb_oxidized::{list_devices, version, CoralDevice};
use std::env;
use std::error::Error;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::path::Path;
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
    fn TfLiteTensorNumDims(tensor: *mut TfLiteTensor) -> i32;
    fn TfLiteTensorDim(tensor: *mut TfLiteTensor, dim_index: i32) -> i32;
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

fn dims_to_string(dims: &[i32]) -> String {
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

fn tensor_dims(tensor: *mut TfLiteTensor) -> Vec<i32> {
    let num_dims = unsafe { TfLiteTensorNumDims(tensor) };
    let mut dims = Vec::with_capacity(num_dims.max(0) as usize);
    for i in 0..num_dims {
        dims.push(unsafe { TfLiteTensorDim(tensor, i) });
    }
    dims
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: cargo run --example inference_benchmark -- <model.tflite> [runs] [warmup]"
        );
        std::process::exit(2);
    }

    let model_path = &args[1];
    let runs = args
        .get(2)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50);
    let warmup = args
        .get(3)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5);

    if !Path::new(model_path).exists() {
        return Err(format!("Model not found: {}", model_path).into());
    }

    println!("EdgeTPU version: {}", version());
    let devices = list_devices()?;
    println!("Detected devices: {}", devices.len());
    for (idx, device) in devices.iter().enumerate() {
        println!("  {}. {}", idx + 1, device);
    }

    println!("STEP create_device");
    let device = CoralDevice::new()?;
    println!("STEP create_delegate");
    let delegate = device.create_delegate()?;
    println!("STEP create_interpreter");

    let c_model_path = CString::new(model_path.as_str())?;

    let (interpreter, model) = unsafe {
        let model = TfLiteModelCreateFromFile(c_model_path.as_ptr());
        if model.is_null() {
            return Err("TfLiteModelCreateFromFile returned null".into());
        }

        let options = TfLiteInterpreterOptionsCreate();
        if options.is_null() {
            TfLiteModelDelete(model);
            return Err("TfLiteInterpreterOptionsCreate returned null".into());
        }

        TfLiteInterpreterOptionsAddDelegate(options, delegate.as_ptr() as *mut TfLiteDelegate);

        let interpreter = TfLiteInterpreterCreate(model, options);
        TfLiteInterpreterOptionsDelete(options);
        if interpreter.is_null() {
            TfLiteModelDelete(model);
            return Err("TfLiteInterpreterCreate returned null".into());
        }

        let alloc_status = TfLiteInterpreterAllocateTensors(interpreter);
        if alloc_status != 0 {
            TfLiteInterpreterDelete(interpreter);
            TfLiteModelDelete(model);
            return Err(
                format!("TfLiteInterpreterAllocateTensors failed: {}", alloc_status).into(),
            );
        }

        (interpreter, model)
    };

    println!("STEP interpreter_ready");

    let input_tensor = unsafe { TfLiteInterpreterGetInputTensor(interpreter, 0) };
    let output_tensor = unsafe { TfLiteInterpreterGetOutputTensor(interpreter, 0) };
    if input_tensor.is_null() || output_tensor.is_null() {
        unsafe {
            TfLiteInterpreterDelete(interpreter);
            TfLiteModelDelete(model);
        }
        return Err("Failed to resolve input/output tensors".into());
    }

    let input_dims = tensor_dims(input_tensor);
    let output_dims = tensor_dims(output_tensor);
    let input_bytes = unsafe { TfLiteTensorByteSize(input_tensor) };
    let output_bytes = unsafe { TfLiteTensorByteSize(output_tensor) };

    println!("Model: {}", model_path);
    println!("Input dims: {}", dims_to_string(&input_dims));
    println!("Output dims: {}", dims_to_string(&output_dims));
    println!("Input bytes: {}", input_bytes);
    println!("Output bytes: {}", output_bytes);
    println!("Warmup: {} runs", warmup);
    println!("Measured: {} runs", runs);

    let mut input_data = vec![0u8; input_bytes];
    for (idx, byte) in input_data.iter_mut().enumerate() {
        *byte = (idx % 251) as u8;
    }
    let mut output_data = vec![0u8; output_bytes];

    for _ in 0..warmup {
        let copy_status = unsafe {
            TfLiteTensorCopyFromBuffer(
                input_tensor,
                input_data.as_ptr().cast::<c_void>(),
                input_data.len(),
            )
        };
        if copy_status != 0 {
            unsafe {
                TfLiteInterpreterDelete(interpreter);
                TfLiteModelDelete(model);
            }
            return Err(format!("TfLiteTensorCopyFromBuffer failed: {}", copy_status).into());
        }
        let invoke_status = unsafe { TfLiteInterpreterInvoke(interpreter) };
        if invoke_status != 0 {
            unsafe {
                TfLiteInterpreterDelete(interpreter);
                TfLiteModelDelete(model);
            }
            return Err(format!("TfLiteInterpreterInvoke failed: {}", invoke_status).into());
        }
    }

    let mut latencies_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let copy_status = unsafe {
            TfLiteTensorCopyFromBuffer(
                input_tensor,
                input_data.as_ptr().cast::<c_void>(),
                input_data.len(),
            )
        };
        if copy_status != 0 {
            unsafe {
                TfLiteInterpreterDelete(interpreter);
                TfLiteModelDelete(model);
            }
            return Err(format!("TfLiteTensorCopyFromBuffer failed: {}", copy_status).into());
        }

        let start = Instant::now();
        let invoke_status = unsafe { TfLiteInterpreterInvoke(interpreter) };
        if invoke_status != 0 {
            unsafe {
                TfLiteInterpreterDelete(interpreter);
                TfLiteModelDelete(model);
            }
            return Err(format!("TfLiteInterpreterInvoke failed: {}", invoke_status).into());
        }
        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let copy_out_status = unsafe {
        TfLiteTensorCopyToBuffer(
            output_tensor,
            output_data.as_mut_ptr().cast::<c_void>(),
            output_data.len(),
        )
    };
    if copy_out_status != 0 {
        unsafe {
            TfLiteInterpreterDelete(interpreter);
            TfLiteModelDelete(model);
        }
        return Err(format!("TfLiteTensorCopyToBuffer failed: {}", copy_out_status).into());
    }

    unsafe {
        TfLiteInterpreterDelete(interpreter);
        TfLiteModelDelete(model);
    }

    let mut sorted = latencies_ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let sum: f64 = latencies_ms.iter().sum();
    let avg = sum / latencies_ms.len() as f64;
    let min = *sorted.first().unwrap_or(&0.0);
    let max = *sorted.last().unwrap_or(&0.0);
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);

    let (top_idx, top_score) = output_data
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .map(|(i, v)| (i, *v))
        .unwrap_or((0, 0));

    println!(
        "Latency ms: min={:.3} p50={:.3} p95={:.3} avg={:.3} max={:.3}",
        min, p50, p95, avg, max
    );
    println!("Top output: index={} score={}", top_idx, top_score);
    println!(
        "RESULT model={} runs={} warmup={} input_dims={} output_dims={} input_bytes={} output_bytes={} min_ms={:.3} p50_ms={:.3} p95_ms={:.3} avg_ms={:.3} max_ms={:.3} top_index={} top_score={}",
        model_path,
        runs,
        warmup,
        dims_to_string(&input_dims),
        dims_to_string(&output_dims),
        input_bytes,
        output_bytes,
        min,
        p50,
        p95,
        avg,
        max,
        top_idx,
        top_score
    );

    Ok(())
}
