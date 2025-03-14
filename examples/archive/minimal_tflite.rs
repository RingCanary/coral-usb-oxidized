use std::path::Path;
use std::ffi::CString;
use std::os::raw::c_void;

// Define TensorFlow Lite C API types
pub enum TfLiteModel {}
pub enum TfLiteInterpreter {}
pub enum TfLiteInterpreterOptions {}

// FFI declarations for TensorFlow Lite C API
#[link(name = "tensorflowlite_c")]
extern "C" {
    fn TfLiteModelCreateFromFile(model_path: *const std::os::raw::c_char) -> *mut TfLiteModel;
    fn TfLiteModelDelete(model: *mut TfLiteModel);
    
    fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);
    
    fn TfLiteInterpreterCreate(model: *mut TfLiteModel, options: *mut TfLiteInterpreterOptions) -> *mut TfLiteInterpreter;
    fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);
    
    fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetInputTensorCount(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetOutputTensorCount(interpreter: *mut TfLiteInterpreter) -> i32;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to the model file
    let model_path = "models/mobilenet_v2_1.0_224_quant_edgetpu.tflite";

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
        println!("Model pointer: {:p}", model);
        
        // Create interpreter options
        println!("Creating interpreter options...");
        let options = TfLiteInterpreterOptionsCreate();
        if options.is_null() {
            TfLiteModelDelete(model);
            return Err("Failed to create interpreter options".into());
        }
        println!("Interpreter options created successfully");
        println!("Options pointer: {:p}", options);
        
        // Create interpreter
        println!("Creating interpreter...");
        let interpreter = TfLiteInterpreterCreate(model, options);
        if interpreter.is_null() {
            TfLiteInterpreterOptionsDelete(options);
            TfLiteModelDelete(model);
            return Err("Failed to create interpreter".into());
        }
        println!("Interpreter created successfully");
        println!("Interpreter pointer: {:p}", interpreter);
        
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
        
        // Clean up
        println!("Cleaning up...");
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        println!("Cleanup complete");
    }
    
    println!("Minimal TensorFlow Lite test completed successfully!");
    Ok(())
}
