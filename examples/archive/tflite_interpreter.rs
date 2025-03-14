use coral_usb_oxidized::{CoralDevice, TfLiteModelWrapper, version};
use std::path::Path;
use std::time::Instant;
use std::ffi::CString;

// FFI declarations for TensorFlow Lite C API
#[link(name = "tensorflowlite_c")]
extern "C" {
    fn TfLiteInterpreterCreate(model: *mut coral_usb_oxidized::TfLiteModel, options: *mut std::ffi::c_void) -> *mut coral_usb_oxidized::TfLiteInterpreter;
    fn TfLiteInterpreterDelete(interpreter: *mut coral_usb_oxidized::TfLiteInterpreter);
    fn TfLiteInterpreterAllocateTensors(interpreter: *mut coral_usb_oxidized::TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetInputTensorCount(interpreter: *mut coral_usb_oxidized::TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetOutputTensorCount(interpreter: *mut coral_usb_oxidized::TfLiteInterpreter) -> i32;
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

    // Try to load the model
    println!("Loading TensorFlow Lite model...");
    let start = Instant::now();
    
    let model = match TfLiteModelWrapper::new_from_file(model_path) {
        Ok(model) => {
            println!("Model loaded successfully in {:?}", start.elapsed());
            println!("Model pointer: {:p}", model.as_ptr());
            model
        },
        Err(e) => {
            return Err(format!("Failed to load model: {}", e).into());
        }
    };

    // Create a TensorFlow Lite interpreter without EdgeTPU delegate
    println!("\nCreating TensorFlow Lite interpreter without EdgeTPU delegate...");
    let start = Instant::now();
    
    unsafe {
        let interpreter = TfLiteInterpreterCreate(model.as_ptr(), std::ptr::null_mut());
        if interpreter.is_null() {
            return Err("Failed to create interpreter".into());
        }
        
        println!("Interpreter created successfully in {:?}", start.elapsed());
        println!("Interpreter pointer: {:p}", interpreter);
        
        // Allocate tensors
        println!("Allocating tensors...");
        let status = TfLiteInterpreterAllocateTensors(interpreter);
        if status != 0 {
            TfLiteInterpreterDelete(interpreter);
            return Err(format!("Failed to allocate tensors: {}", status).into());
        }
        
        // Get input and output tensor counts
        let input_count = TfLiteInterpreterGetInputTensorCount(interpreter);
        let output_count = TfLiteInterpreterGetOutputTensorCount(interpreter);
        
        println!("Tensors allocated successfully");
        println!("Input tensor count: {}", input_count);
        println!("Output tensor count: {}", output_count);
        
        // Clean up
        TfLiteInterpreterDelete(interpreter);
    }
    
    // Now try with EdgeTPU delegate
    println!("\nCreating Coral device...");
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

    // Create a TensorFlow Lite interpreter options with the EdgeTPU delegate
    println!("\nCreating TensorFlow Lite interpreter with EdgeTPU delegate...");
    println!("This step is skipped for now as it requires additional FFI bindings");
    println!("Interpreter test completed successfully!");

    Ok(())
}
