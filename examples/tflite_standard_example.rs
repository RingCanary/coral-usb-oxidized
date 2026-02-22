use coral_usb_oxidized::version;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::Path;

// Define TensorFlow Lite C API types
pub enum TfLiteModel {}
pub enum TfLiteInterpreter {}
pub enum TfLiteInterpreterOptions {}
pub enum TfLiteTensor {}

// FFI declarations for TensorFlow Lite C API
#[link(name = "tensorflowlite_c")]
extern "C" {
    fn TfLiteModelCreateFromFile(model_path: *const c_char) -> *mut TfLiteModel;
    fn TfLiteModelDelete(model: *mut TfLiteModel);

    fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);
    fn TfLiteInterpreterOptionsSetNumThreads(
        options: *mut TfLiteInterpreterOptions,
        num_threads: c_int,
    );

    fn TfLiteInterpreterCreate(
        model: *mut TfLiteModel,
        options: *mut TfLiteInterpreterOptions,
    ) -> *mut TfLiteInterpreter;
    fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);

    fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> c_int;
    fn TfLiteInterpreterGetInputTensorCount(interpreter: *mut TfLiteInterpreter) -> c_int;
    fn TfLiteInterpreterGetOutputTensorCount(interpreter: *mut TfLiteInterpreter) -> c_int;
    fn TfLiteInterpreterGetInputTensor(
        interpreter: *mut TfLiteInterpreter,
        input_index: c_int,
    ) -> *mut TfLiteTensor;
    fn TfLiteInterpreterGetOutputTensor(
        interpreter: *mut TfLiteInterpreter,
        output_index: c_int,
    ) -> *mut TfLiteTensor;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Print library version information
    let version_str = version();
    println!("EdgeTPU Library Version: {}", version_str);

    // Path to the standard TensorFlow Lite model
    let model_path = "models/mobilenet_v1_1.0_224_quant.tflite";

    // Check if model file exists
    if !Path::new(model_path).exists() {
        return Err(format!("Model file not found: {}", model_path).into());
    }
    println!("Model file found: {}", model_path);

    unsafe {
        // Convert path to C string
        let c_model_path = CString::new(model_path)?;

        // Load the model
        println!("Loading TensorFlow Lite model...");
        let model = TfLiteModelCreateFromFile(c_model_path.as_ptr());
        if model.is_null() {
            return Err("Failed to load model".into());
        }
        println!("Model loaded successfully");

        // Create interpreter options
        println!("Creating interpreter options...");
        let options = TfLiteInterpreterOptionsCreate();
        if options.is_null() {
            TfLiteModelDelete(model);
            return Err("Failed to create interpreter options".into());
        }

        // Set number of threads
        TfLiteInterpreterOptionsSetNumThreads(options, 1);
        println!("Set number of threads to 1");

        // Create interpreter
        println!("Creating interpreter...");
        let interpreter = TfLiteInterpreterCreate(model, options);
        if interpreter.is_null() {
            TfLiteInterpreterOptionsDelete(options);
            TfLiteModelDelete(model);
            return Err("Failed to create interpreter".into());
        }
        println!("Interpreter created successfully");

        // Allocate tensors
        println!("Allocating tensors...");
        let status = TfLiteInterpreterAllocateTensors(interpreter);
        if status != 0 {
            TfLiteInterpreterDelete(interpreter);
            TfLiteInterpreterOptionsDelete(options);
            TfLiteModelDelete(model);
            return Err(format!("Failed to allocate tensors: {}", status).into());
        }
        println!("Tensors allocated successfully");

        // Get tensor counts
        let input_count = TfLiteInterpreterGetInputTensorCount(interpreter);
        let output_count = TfLiteInterpreterGetOutputTensorCount(interpreter);
        println!("Input tensor count: {}", input_count);
        println!("Output tensor count: {}", output_count);

        // Get input tensor
        if input_count > 0 {
            let input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
            if !input_tensor.is_null() {
                println!("Successfully retrieved input tensor");
            } else {
                println!("Failed to get input tensor");
            }
        }

        // Get output tensor
        if output_count > 0 {
            let output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
            if !output_tensor.is_null() {
                println!("Successfully retrieved output tensor");
            } else {
                println!("Failed to get output tensor");
            }
        }

        // Clean up
        println!("Cleaning up resources...");
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        println!("Cleanup complete");
    }

    println!("\nTensorFlow Lite standard example completed successfully");
    Ok(())
}
