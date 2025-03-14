use coral_usb_oxidized::{CoralDevice, version};
use std::path::Path;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};

// Define TensorFlow Lite C API types
pub enum TfLiteModel {}
pub enum TfLiteInterpreter {}
pub enum TfLiteInterpreterOptions {}
pub enum TfLiteTensor {}
pub enum TfLiteDelegate {}

// FFI declarations for TensorFlow Lite C API
#[link(name = "tensorflowlite_c")]
extern "C" {
    fn TfLiteModelCreateFromFile(model_path: *const c_char) -> *mut TfLiteModel;
    fn TfLiteModelDelete(model: *mut TfLiteModel);
    
    fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);
    fn TfLiteInterpreterOptionsAddDelegate(options: *mut TfLiteInterpreterOptions, delegate: *mut TfLiteDelegate);
    
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
    // Print library version information
    let version_str = version();
    println!("EdgeTPU Library Version: {}", version_str);

    // Path to the model file
    let model_path = "models/mobilenet_v2_1.0_224_quant_edgetpu.tflite";

    // Check if model file exists
    if !Path::new(model_path).exists() {
        return Err(format!("Model file not found: {}", model_path).into());
    }
    println!("Model file found: {}", model_path);

    // Create a Coral device
    println!("Creating Coral device...");
    let device = match CoralDevice::new() {
        Ok(dev) => {
            println!("Coral device created successfully!");
            println!("  Valid: {}", dev.is_valid());
            println!("  Vendor ID: 0x{:x}", dev.vendor_id());
            println!("  Product ID: 0x{:x}", dev.product_id());
            println!("  Name: {:?}", dev.name());
            dev
        },
        Err(e) => {
            return Err(format!("Failed to create Coral device: {}", e).into());
        }
    };

    // Create an EdgeTPU delegate
    println!("Creating EdgeTPU delegate...");
    let delegate = match device.create_delegate() {
        Ok(del) => {
            println!("EdgeTPU delegate created successfully!");
            println!("  Delegate is valid: {}", del.is_valid());
            println!("  Delegate pointer: {:p}", del.as_ptr());
            del
        },
        Err(e) => {
            return Err(format!("Failed to create EdgeTPU delegate: {}", e).into());
        }
    };

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
        
        // Add EdgeTPU delegate to options
        println!("Adding EdgeTPU delegate to interpreter options...");
        TfLiteInterpreterOptionsAddDelegate(options, delegate.as_ptr() as *mut TfLiteDelegate);
        println!("EdgeTPU delegate added to interpreter options");
        
        // Create interpreter
        println!("Creating interpreter with EdgeTPU delegate...");
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
        
        // Create dummy input data (all 128s)
        println!("\nPreparing dummy input data...");
        let input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
        if !input_tensor.is_null() {
            let input_size = TfLiteTensorByteSize(input_tensor);
            println!("Input tensor size: {} bytes", input_size);
            
            // Create a vector with all elements set to 128
            let input_data = vec![128u8; input_size];
            
            // Get pointer to the input tensor data
            let tensor_data_ptr = TfLiteTensorData(input_tensor) as *mut u8;
            
            // Copy input data to tensor
            if !tensor_data_ptr.is_null() {
                std::ptr::copy_nonoverlapping(input_data.as_ptr(), tensor_data_ptr, input_size);
                println!("Input data copied to tensor");
            } else {
                println!("Warning: Input tensor data pointer is null");
            }
        }
        
        // Run inference
        println!("\nRunning inference...");
        let invoke_status = TfLiteInterpreterInvoke(interpreter);
        if invoke_status != 0 {
            TfLiteInterpreterDelete(interpreter);
            TfLiteInterpreterOptionsDelete(options);
            TfLiteModelDelete(model);
            return Err(format!("Failed to run inference: {}", invoke_status).into());
        }
        println!("Inference completed successfully");
        
        // Get output data
        println!("\nRetrieving output data...");
        let output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
        if !output_tensor.is_null() {
            let output_size = TfLiteTensorByteSize(output_tensor);
            println!("Output tensor size: {} bytes", output_size);
            
            // Get pointer to the output tensor data
            let tensor_data_ptr = TfLiteTensorData(output_tensor) as *const u8;
            
            if !tensor_data_ptr.is_null() {
                // Copy output data from tensor
                let mut output_data = vec![0u8; output_size];
                std::ptr::copy_nonoverlapping(tensor_data_ptr, output_data.as_mut_ptr(), output_size);
                
                // Find the top 5 scores
                let mut indices: Vec<usize> = (0..output_data.len()).collect();
                indices.sort_unstable_by(|&a, &b| output_data[b].cmp(&output_data[a]));
                
                println!("\nTop 5 scores (with dummy input):");
                for i in 0..5.min(indices.len()) {
                    let idx = indices[i];
                    let score = output_data[idx] as f32 / 255.0; // Convert from uint8 to float
                    println!("  {}. Index {} - {:.2}%", i + 1, idx, score * 100.0);
                }
            } else {
                println!("Warning: Output tensor data pointer is null");
            }
        }
        
        // Clean up
        println!("\nCleaning up...");
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        println!("Cleanup complete");
    }
    
    println!("\nEdgeTPU delegate test completed successfully!");
    Ok(())
}
