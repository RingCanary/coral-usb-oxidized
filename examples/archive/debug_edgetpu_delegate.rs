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
    fn TfLiteInterpreterOptionsSetNumThreads(options: *mut TfLiteInterpreterOptions, num_threads: i32);
    fn TfLiteInterpreterOptionsAddDelegate(options: *mut TfLiteInterpreterOptions, delegate: *mut TfLiteDelegate);
    
    fn TfLiteInterpreterCreate(model: *mut TfLiteModel, options: *mut TfLiteInterpreterOptions) -> *mut TfLiteInterpreter;
    fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);
    
    fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetInputTensorCount(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetOutputTensorCount(interpreter: *mut TfLiteInterpreter) -> i32;
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
        
        // Set number of threads
        TfLiteInterpreterOptionsSetNumThreads(options, 1);
        println!("Set number of threads to 1");
        
        // Create interpreter without delegate first
        println!("Creating interpreter without EdgeTPU delegate...");
        let interpreter = TfLiteInterpreterCreate(model, options);
        if interpreter.is_null() {
            TfLiteInterpreterOptionsDelete(options);
            TfLiteModelDelete(model);
            return Err("Failed to create interpreter".into());
        }
        println!("Interpreter created successfully");
        println!("Interpreter pointer: {:p}", interpreter);
        
        // Try to allocate tensors (this should fail for EdgeTPU model without delegate)
        println!("Attempting to allocate tensors without EdgeTPU delegate (expected to fail)...");
        let status = TfLiteInterpreterAllocateTensors(interpreter);
        if status != 0 {
            println!("Failed to allocate tensors without EdgeTPU delegate as expected: {}", status);
        } else {
            println!("Unexpectedly succeeded in allocating tensors without EdgeTPU delegate");
        }
        
        // Clean up the first interpreter
        println!("Cleaning up first interpreter...");
        TfLiteInterpreterDelete(interpreter);
        println!("First interpreter deleted");
        
        // Create new options for EdgeTPU delegate
        println!("Creating new interpreter options for EdgeTPU delegate...");
        let options_with_delegate = TfLiteInterpreterOptionsCreate();
        if options_with_delegate.is_null() {
            TfLiteModelDelete(model);
            return Err("Failed to create interpreter options for EdgeTPU delegate".into());
        }
        println!("New interpreter options created successfully");
        println!("Options with delegate pointer: {:p}", options_with_delegate);
        
        // Set number of threads
        TfLiteInterpreterOptionsSetNumThreads(options_with_delegate, 1);
        println!("Set number of threads to 1");
        
        // Add EdgeTPU delegate to options
        println!("Adding EdgeTPU delegate to interpreter options...");
        TfLiteInterpreterOptionsAddDelegate(options_with_delegate, delegate.as_ptr() as *mut TfLiteDelegate);
        println!("EdgeTPU delegate added to interpreter options");
        
        // Create interpreter with EdgeTPU delegate
        println!("Creating interpreter with EdgeTPU delegate...");
        println!("  Model pointer: {:p}", model);
        println!("  Options with delegate pointer: {:p}", options_with_delegate);
        println!("  Delegate pointer: {:p}", delegate.as_ptr());
        
        // This is the line that causes the segmentation fault
        // We'll add a try/catch block in Rust to handle the panic
        let result = std::panic::catch_unwind(|| {
            TfLiteInterpreterCreate(model, options_with_delegate)
        });
        
        match result {
            Ok(interpreter_with_delegate) => {
                if interpreter_with_delegate.is_null() {
                    println!("Failed to create interpreter with EdgeTPU delegate (returned null)");
                } else {
                    println!("Interpreter with EdgeTPU delegate created successfully");
                    println!("Interpreter with delegate pointer: {:p}", interpreter_with_delegate);
                    
                    // Try to allocate tensors
                    println!("Attempting to allocate tensors with EdgeTPU delegate...");
                    let status = TfLiteInterpreterAllocateTensors(interpreter_with_delegate);
                    if status != 0 {
                        println!("Failed to allocate tensors with EdgeTPU delegate: {}", status);
                    } else {
                        println!("Tensors allocated successfully with EdgeTPU delegate");
                        
                        // Get tensor counts
                        let input_count = TfLiteInterpreterGetInputTensorCount(interpreter_with_delegate);
                        let output_count = TfLiteInterpreterGetOutputTensorCount(interpreter_with_delegate);
                        println!("Input tensor count: {}", input_count);
                        println!("Output tensor count: {}", output_count);
                    }
                    
                    // Clean up the second interpreter
                    println!("Cleaning up second interpreter...");
                    TfLiteInterpreterDelete(interpreter_with_delegate);
                    println!("Second interpreter deleted");
                }
            },
            Err(_) => {
                println!("PANIC: TfLiteInterpreterCreate with EdgeTPU delegate caused a segmentation fault");
            }
        }
        
        // Clean up options and model
        println!("Cleaning up remaining resources...");
        TfLiteInterpreterOptionsDelete(options_with_delegate);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        println!("Cleanup complete");
    }
    
    println!("\nEdgeTPU delegate debug test completed");
    Ok(())
}
