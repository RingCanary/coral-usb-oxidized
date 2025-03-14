use coral_usb_oxidized::{CoralDevice, version};
use std::path::Path;
use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_void};
use std::io::Read;
use std::fs::File;

// Define TensorFlow Lite C API types
pub enum TfLiteModel {}
pub enum TfLiteInterpreter {}
pub enum TfLiteInterpreterOptions {}
pub enum TfLiteTensor {}
pub enum TfLiteDelegate {}

// TensorFlow Lite tensor types
const TFLITE_TYPE_FLOAT32: i32 = 0;
const TFLITE_TYPE_INT32: i32 = 2;
const TFLITE_TYPE_UINT8: i32 = 3;
const TFLITE_TYPE_INT64: i32 = 4;

// FFI declarations for TensorFlow Lite C API
#[link(name = "tensorflowlite_c")]
extern "C" {
    fn TfLiteModelCreateFromFile(model_path: *const c_char) -> *mut TfLiteModel;
    fn TfLiteModelDelete(model: *mut TfLiteModel);
    
    fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);
    fn TfLiteInterpreterOptionsSetNumThreads(options: *mut TfLiteInterpreterOptions, num_threads: i32);
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

// Helper function to get tensor type name
fn get_tensor_type_name(tensor_type: i32) -> &'static str {
    match tensor_type {
        TFLITE_TYPE_FLOAT32 => "FLOAT32",
        TFLITE_TYPE_INT32 => "INT32",
        TFLITE_TYPE_UINT8 => "UINT8",
        TFLITE_TYPE_INT64 => "INT64",
        _ => "UNKNOWN",
    }
}

// Helper function to load and preprocess an image
fn load_image(image_path: &str, width: i32, height: i32) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    println!("Loading image: {}", image_path);
    
    // For simplicity, we'll just create a dummy image with all pixels set to 128
    let image_size = (width * height * 3) as usize;
    let image_data = vec![128u8; image_size];
    
    println!("Created dummy image data of size: {}", image_size);
    Ok(image_data)
}

// Helper function to load labels
fn load_labels(label_path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    println!("Loading labels: {}", label_path);
    
    let mut file = File::open(label_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    let labels: Vec<String> = contents.lines().map(|s| s.to_string()).collect();
    println!("Loaded {} labels", labels.len());
    
    Ok(labels)
}

// Function to run inference with a standard TensorFlow Lite model
fn run_standard_model(model_path: &str, image_path: &str, label_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Running Standard TensorFlow Lite Model ===");
    
    // Check if model file exists
    if !Path::new(model_path).exists() {
        return Err(format!("Model file not found: {}", model_path).into());
    }
    println!("Model file found: {}", model_path);
    
    // Check if label file exists
    if !Path::new(label_path).exists() {
        return Err(format!("Label file not found: {}", label_path).into());
    }
    
    // Load labels
    let labels = load_labels(label_path)?;
    
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
        println!("Interpreter options created successfully");
        
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
        
        // Get input tensor information
        println!("\nInput tensor information:");
        let mut input_width = 0;
        let mut input_height = 0;
        let mut input_channels = 0;
        
        for i in 0..input_count {
            let tensor = TfLiteInterpreterGetInputTensor(interpreter, i);
            if !tensor.is_null() {
                let name_ptr = TfLiteTensorName(tensor);
                let name = if !name_ptr.is_null() {
                    CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
                } else {
                    "Unknown".to_string()
                };
                
                let tensor_type = TfLiteTensorType(tensor);
                let num_dims = TfLiteTensorNumDims(tensor);
                
                let mut dims = Vec::new();
                for d in 0..num_dims {
                    dims.push(TfLiteTensorDim(tensor, d));
                }
                
                // Extract dimensions for image preprocessing
                if num_dims == 4 {
                    input_height = dims[1];
                    input_width = dims[2];
                    input_channels = dims[3];
                }
                
                let byte_size = TfLiteTensorByteSize(tensor);
                
                println!("  Input[{}]:", i);
                println!("    Name: {}", name);
                println!("    Type: {} ({})", tensor_type, get_tensor_type_name(tensor_type));
                println!("    Dimensions: {:?}", dims);
                println!("    Byte size: {}", byte_size);
            }
        }
        
        // Load and preprocess image
        if input_width > 0 && input_height > 0 && input_channels > 0 {
            println!("\nPreprocessing image...");
            println!("Input dimensions: {}x{}x{}", input_width, input_height, input_channels);
            
            let image_data = load_image(image_path, input_width, input_height)?;
            
            // Copy image data to input tensor
            let input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
            if !input_tensor.is_null() {
                let tensor_data_ptr = TfLiteTensorData(input_tensor) as *mut u8;
                if !tensor_data_ptr.is_null() {
                    std::ptr::copy_nonoverlapping(image_data.as_ptr(), tensor_data_ptr, image_data.len());
                    println!("Image data copied to input tensor");
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
            println!("\nOutput tensor information and results:");
            for i in 0..output_count {
                let tensor = TfLiteInterpreterGetOutputTensor(interpreter, i);
                if !tensor.is_null() {
                    let name_ptr = TfLiteTensorName(tensor);
                    let name = if !name_ptr.is_null() {
                        CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
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
                    println!("    Type: {} ({})", tensor_type, get_tensor_type_name(tensor_type));
                    println!("    Dimensions: {:?}", dims);
                    println!("    Byte size: {}", byte_size);
                    
                    // Get output data
                    let tensor_data_ptr = TfLiteTensorData(tensor) as *const u8;
                    if !tensor_data_ptr.is_null() {
                        // Copy output data from tensor
                        let mut output_data = vec![0u8; byte_size];
                        std::ptr::copy_nonoverlapping(tensor_data_ptr, output_data.as_mut_ptr(), byte_size);
                        
                        // Find the top 5 scores
                        let mut indices: Vec<usize> = (0..output_data.len()).collect();
                        indices.sort_unstable_by(|&a, &b| output_data[b].cmp(&output_data[a]));
                        
                        println!("\nTop 5 predictions:");
                        for i in 0..5.min(indices.len()) {
                            let idx = indices[i];
                            let score = output_data[idx] as f32 / 255.0; // Convert from uint8 to float
                            
                            let label = if idx < labels.len() {
                                &labels[idx]
                            } else {
                                "Unknown"
                            };
                            
                            println!("  {}. {} - {:.2}%", i + 1, label, score * 100.0);
                        }
                    } else {
                        println!("Warning: Output tensor data pointer is null");
                    }
                }
            }
        } else {
            println!("Warning: Could not determine input dimensions");
        }
        
        // Clean up
        println!("\nCleaning up...");
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        println!("Cleanup complete");
    }
    
    println!("Standard TensorFlow Lite inference completed successfully!");
    Ok(())
}

// Function to run inference with an EdgeTPU-optimized model
fn run_edgetpu_model(model_path: &str, image_path: &str, label_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Running EdgeTPU-Optimized TensorFlow Lite Model ===");
    
    // Print library version information
    let version_str = version();
    println!("EdgeTPU Library Version: {}", version_str);
    
    // Check if model file exists
    if !Path::new(model_path).exists() {
        return Err(format!("Model file not found: {}", model_path).into());
    }
    println!("Model file found: {}", model_path);
    
    // Check if label file exists
    if !Path::new(label_path).exists() {
        return Err(format!("Label file not found: {}", label_path).into());
    }
    
    // Load labels
    let labels = load_labels(label_path)?;
    
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
        
        // Create interpreter options
        println!("Creating interpreter options...");
        let options = TfLiteInterpreterOptionsCreate();
        if options.is_null() {
            TfLiteModelDelete(model);
            return Err("Failed to create interpreter options".into());
        }
        
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
        let mut input_width = 0;
        let mut input_height = 0;
        let mut input_channels = 0;
        
        for i in 0..input_count {
            let tensor = TfLiteInterpreterGetInputTensor(interpreter, i);
            if !tensor.is_null() {
                let name_ptr = TfLiteTensorName(tensor);
                let name = if !name_ptr.is_null() {
                    CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
                } else {
                    "Unknown".to_string()
                };
                
                let tensor_type = TfLiteTensorType(tensor);
                let num_dims = TfLiteTensorNumDims(tensor);
                
                let mut dims = Vec::new();
                for d in 0..num_dims {
                    dims.push(TfLiteTensorDim(tensor, d));
                }
                
                // Extract dimensions for image preprocessing
                if num_dims == 4 {
                    input_height = dims[1];
                    input_width = dims[2];
                    input_channels = dims[3];
                }
                
                let byte_size = TfLiteTensorByteSize(tensor);
                
                println!("  Input[{}]:", i);
                println!("    Name: {}", name);
                println!("    Type: {} ({})", tensor_type, get_tensor_type_name(tensor_type));
                println!("    Dimensions: {:?}", dims);
                println!("    Byte size: {}", byte_size);
            }
        }
        
        // Load and preprocess image
        if input_width > 0 && input_height > 0 && input_channels > 0 {
            println!("\nPreprocessing image...");
            println!("Input dimensions: {}x{}x{}", input_width, input_height, input_channels);
            
            let image_data = load_image(image_path, input_width, input_height)?;
            
            // Copy image data to input tensor
            let input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
            if !input_tensor.is_null() {
                let tensor_data_ptr = TfLiteTensorData(input_tensor) as *mut u8;
                if !tensor_data_ptr.is_null() {
                    std::ptr::copy_nonoverlapping(image_data.as_ptr(), tensor_data_ptr, image_data.len());
                    println!("Image data copied to input tensor");
                } else {
                    println!("Warning: Input tensor data pointer is null");
                }
            }
            
            // Run inference
            println!("\nRunning inference with EdgeTPU...");
            let invoke_status = TfLiteInterpreterInvoke(interpreter);
            if invoke_status != 0 {
                TfLiteInterpreterDelete(interpreter);
                TfLiteInterpreterOptionsDelete(options);
                TfLiteModelDelete(model);
                return Err(format!("Failed to run inference: {}", invoke_status).into());
            }
            println!("Inference completed successfully");
            
            // Get output data
            println!("\nOutput tensor information and results:");
            for i in 0..output_count {
                let tensor = TfLiteInterpreterGetOutputTensor(interpreter, i);
                if !tensor.is_null() {
                    let name_ptr = TfLiteTensorName(tensor);
                    let name = if !name_ptr.is_null() {
                        CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
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
                    println!("    Type: {} ({})", tensor_type, get_tensor_type_name(tensor_type));
                    println!("    Dimensions: {:?}", dims);
                    println!("    Byte size: {}", byte_size);
                    
                    // Get output data
                    let tensor_data_ptr = TfLiteTensorData(tensor) as *const u8;
                    if !tensor_data_ptr.is_null() {
                        // Copy output data from tensor
                        let mut output_data = vec![0u8; byte_size];
                        std::ptr::copy_nonoverlapping(tensor_data_ptr, output_data.as_mut_ptr(), byte_size);
                        
                        // Find the top 5 scores
                        let mut indices: Vec<usize> = (0..output_data.len()).collect();
                        indices.sort_unstable_by(|&a, &b| output_data[b].cmp(&output_data[a]));
                        
                        println!("\nTop 5 predictions:");
                        for i in 0..5.min(indices.len()) {
                            let idx = indices[i];
                            let score = output_data[idx] as f32 / 255.0; // Convert from uint8 to float
                            
                            let label = if idx < labels.len() {
                                &labels[idx]
                            } else {
                                "Unknown"
                            };
                            
                            println!("  {}. {} - {:.2}%", i + 1, label, score * 100.0);
                        }
                    } else {
                        println!("Warning: Output tensor data pointer is null");
                    }
                }
            }
        } else {
            println!("Warning: Could not determine input dimensions");
        }
        
        // Clean up
        println!("\nCleaning up...");
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        println!("Cleanup complete");
    }
    
    println!("EdgeTPU TensorFlow Lite inference completed successfully!");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Paths to model files
    let standard_model_path = "models/standard/mobilenet_v1_1.0_224_quant.tflite";
    let edgetpu_model_path = "models/mobilenet_v2_1.0_224_quant_edgetpu.tflite";
    
    // Paths to label files
    let standard_label_path = "models/standard/temp/labels_mobilenet_quant_v1_224.txt";
    let edgetpu_label_path = "models/imagenet_labels.txt";
    
    // Path to image file
    let image_path = "models/grace_hopper.bmp";
    
    // Run inference with standard model
    match run_standard_model(standard_model_path, image_path, standard_label_path) {
        Ok(_) => println!("Standard model inference completed successfully"),
        Err(e) => println!("Error running standard model inference: {}", e),
    }
    
    // Run inference with EdgeTPU model
    match run_edgetpu_model(edgetpu_model_path, image_path, edgetpu_label_path) {
        Ok(_) => println!("EdgeTPU model inference completed successfully"),
        Err(e) => println!("Error running EdgeTPU model inference: {}", e),
    }
    
    Ok(())
}
