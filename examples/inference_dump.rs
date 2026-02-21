use coral_usb_oxidized::{version, CoralDevice};
use std::env;
use std::error::Error;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::path::Path;

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

fn fill_input(mode: &str, input: &mut [u8]) -> Result<(), Box<dyn Error>> {
    match mode {
        "zeros" => {
            input.fill(0);
        }
        "ones" => {
            input.fill(1u8);
        }
        "ramp" => {
            for (idx, value) in input.iter_mut().enumerate() {
                let signed = ((idx % 256) as i16) - 128;
                *value = (signed as i8) as u8;
            }
        }
        "alt" => {
            for (idx, value) in input.iter_mut().enumerate() {
                *value = if idx % 2 == 0 { 0x7f } else { 0x80 };
            }
        }
        _ => {
            return Err(format!("unknown input mode: {mode} (expected: zeros|ones|ramp|alt)").into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example inference_dump -- <model.tflite> [input_mode]");
        std::process::exit(2);
    }

    let model_path = &args[1];
    let input_mode = args.get(2).map(String::as_str).unwrap_or("ramp");

    if !Path::new(model_path).exists() {
        return Err(format!("Model not found: {}", model_path).into());
    }

    println!("EdgeTPU version: {}", version());
    println!("Model: {}", model_path);
    println!("Input mode: {}", input_mode);

    let device = CoralDevice::new()?;
    let delegate = device.create_delegate()?;
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
            return Err(format!("TfLiteInterpreterAllocateTensors failed: {alloc_status}").into());
        }

        (interpreter, model)
    };

    let input_tensor = unsafe { TfLiteInterpreterGetInputTensor(interpreter, 0) };
    let output_tensor = unsafe { TfLiteInterpreterGetOutputTensor(interpreter, 0) };
    if input_tensor.is_null() || output_tensor.is_null() {
        unsafe {
            TfLiteInterpreterDelete(interpreter);
            TfLiteModelDelete(model);
        }
        return Err("failed to resolve input/output tensors".into());
    }

    let input_dims = tensor_dims(input_tensor);
    let output_dims = tensor_dims(output_tensor);
    let input_bytes = unsafe { TfLiteTensorByteSize(input_tensor) };
    let output_bytes = unsafe { TfLiteTensorByteSize(output_tensor) };

    println!("Input dims: {}", dims_to_string(&input_dims));
    println!("Output dims: {}", dims_to_string(&output_dims));
    println!("Input bytes: {}", input_bytes);
    println!("Output bytes: {}", output_bytes);

    let mut input_data = vec![0u8; input_bytes];
    fill_input(input_mode, &mut input_data)?;
    let mut output_data = vec![0u8; output_bytes];

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
        return Err(format!("TfLiteTensorCopyFromBuffer failed: {copy_status}").into());
    }

    let invoke_status = unsafe { TfLiteInterpreterInvoke(interpreter) };
    if invoke_status != 0 {
        unsafe {
            TfLiteInterpreterDelete(interpreter);
            TfLiteModelDelete(model);
        }
        return Err(format!("TfLiteInterpreterInvoke failed: {invoke_status}").into());
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
        return Err(format!("TfLiteTensorCopyToBuffer failed: {copy_out_status}").into());
    }

    unsafe {
        TfLiteInterpreterDelete(interpreter);
        TfLiteModelDelete(model);
    }

    let output_i8: Vec<i8> = output_data.iter().map(|b| *b as i8).collect();
    let (top_idx, top_score) = output_i8
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| **v)
        .map(|(i, v)| (i, *v))
        .unwrap_or((0, 0));

    let preview_len = output_i8.len().min(256);
    let preview = output_i8
        .iter()
        .take(preview_len)
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("Output int8 len={}", output_i8.len());
    println!("Output preview (first {}): {}", preview_len, preview);
    println!("Top output: index={} score={}", top_idx, top_score);

    Ok(())
}
