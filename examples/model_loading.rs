use coral_usb_oxidized::{CoralDevice, CoralInterpreter, is_device_connected, version};
use std::path::Path;
use std::time::Instant;

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

    // Check if a Coral USB Accelerator is connected
    if !is_device_connected() {
        return Err("No Coral USB Accelerator detected. Please connect the device and try again.".into());
    }
    println!("Coral USB Accelerator detected!");

    // Create a new Coral device
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

    // Create a TensorFlow Lite interpreter with the EdgeTPU delegate
    println!("Loading model and creating interpreter...");
    println!("Model path: {}", model_path);
    let start = Instant::now();
    
    let interpreter = match CoralInterpreter::new(model_path, &delegate) {
        Ok(interp) => {
            println!("Model loaded in {:?}", start.elapsed());
            interp
        },
        Err(e) => {
            return Err(format!("Failed to create interpreter: {}", e).into());
        }
    };

    // Get input tensor information
    let input_count = interpreter.input_tensor_count();
    println!("Input tensor count: {}", input_count);

    for i in 0..input_count {
        match interpreter.input_tensor_dims(i) {
            Ok(dims) => {
                match interpreter.input_tensor_name(i) {
                    Ok(name) => println!("Input tensor {}: name={}, dims={:?}", i, name, dims),
                    Err(e) => println!("Error getting input tensor {} name: {}", i, e)
                }
            },
            Err(e) => println!("Error getting input tensor {} dims: {}", i, e)
        }
    }

    // Get output tensor information
    let output_count = interpreter.output_tensor_count();
    println!("Output tensor count: {}", output_count);

    for i in 0..output_count {
        match interpreter.output_tensor_dims(i) {
            Ok(dims) => {
                match interpreter.output_tensor_name(i) {
                    Ok(name) => println!("Output tensor {}: name={}, dims={:?}", i, name, dims),
                    Err(e) => println!("Error getting output tensor {} name: {}", i, e)
                }
            },
            Err(e) => println!("Error getting output tensor {} dims: {}", i, e)
        }
    }

    println!("\nModel loaded successfully!");
    Ok(())
}
