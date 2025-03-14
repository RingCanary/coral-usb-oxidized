use std::path::Path;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};

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
    fn TfLiteInterpreterOptionsSetNumThreads(options: *mut TfLiteInterpreterOptions, num_threads: i32);
    
    fn TfLiteInterpreterCreate(model: *mut TfLiteModel, options: *mut TfLiteInterpreterOptions) -> *mut TfLiteInterpreter;
    fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);
    
    fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetInputTensorCount(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetOutputTensorCount(interpreter: *mut TfLiteInterpreter) -> i32;
    
    fn TfLiteInterpreterGetInputTensor(interpreter: *mut TfLiteInterpreter, input_index: i32) -> *mut TfLiteTensor;
    fn TfLiteInterpreterGetOutputTensor(interpreter: *mut TfLiteInterpreter, output_index: i32) -> *mut TfLiteTensor;
    
    fn TfLiteTensorName(tensor: *mut TfLiteTensor) -> *const c_char;
    fn TfLiteTensorType(tensor: *mut TfLiteTensor) -> i32;
    fn TfLiteTensorNumDims(tensor: *mut TfLiteTensor) -> i32;
    fn TfLiteTensorDim(tensor: *mut TfLiteTensor, dim_index: i32) -> i32;
    fn TfLiteTensorByteSize(tensor: *mut TfLiteTensor) -> usize;
    fn TfLiteTensorData(tensor: *mut TfLiteTensor) -> *mut c_void;
    
    fn TfLiteInterpreterInvoke(interpreter: *mut TfLiteInterpreter) -> i32;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to a standard TensorFlow Lite model (not EdgeTPU optimized)
    let model_path = "models/standard/mobilenet_v1_1.0_224_quant.tflite";

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
        
        // Set number of threads
        TfLiteInterpreterOptionsSetNumThreads(options, 1);
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
        
        // Get input tensor information
        println!("\nInput tensor information:");
        for i in 0..input_count {
            let tensor = TfLiteInterpreterGetInputTensor(interpreter, i);
            if !tensor.is_null() {
                let name_ptr = TfLiteTensorName(tensor);
                let name = if !name_ptr.is_null() {
                    std::ffi::CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
                } else {
                    "Unknown".to_string()
                };
                
                let tensor_type = TfLiteTensorType(tensor);
                let num_dims = TfLiteTensorNumDims(tensor);
                
                let mut dims = Vec::new();
                for d in 0..num_dims {
                    dims.push(TfLiteTensorDim(tensor, d));
                }
                
                let byte_size = TfLiteTensorByteSize(tensor);
                
                println!("  Input[{}]:", i);
                println!("    Name: {}", name);
                println!("    Type: {}", tensor_type);
                println!("    Dimensions: {:?}", dims);
                println!("    Byte size: {}", byte_size);
            }
        }
        
        // Get output tensor information
        println!("\nOutput tensor information:");
        for i in 0..output_count {
            let tensor = TfLiteInterpreterGetOutputTensor(interpreter, i);
            if !tensor.is_null() {
                let name_ptr = TfLiteTensorName(tensor);
                let name = if !name_ptr.is_null() {
                    std::ffi::CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
                } else {
                    "Unknown".to_string()
                };
                
                let tensor_type = TfLiteTensorType(tensor);
                let num_dims = TfLiteTensorNumDims(tensor);
                
                let mut dims = Vec::new();
                for d in 0..num_dims {
                    dims.push(TfLiteTensorDim(tensor, d));
                }
                
                let byte_size = TfLiteTensorByteSize(tensor);
                
                println!("  Output[{}]:", i);
                println!("    Name: {}", name);
                println!("    Type: {}", tensor_type);
                println!("    Dimensions: {:?}", dims);
                println!("    Byte size: {}", byte_size);
            }
        }
        
        // Clean up
        println!("\nCleaning up...");
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        println!("Cleanup complete");
    }
    
    println!("\nStandard TensorFlow Lite test completed successfully!");
    Ok(())
}
