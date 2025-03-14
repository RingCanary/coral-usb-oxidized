use coral_usb_oxidized::{TfLiteModelWrapper, version};
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

    // Try to load the model
    println!("Loading TensorFlow Lite model...");
    let start = Instant::now();
    
    match TfLiteModelWrapper::new_from_file(model_path) {
        Ok(model) => {
            println!("Model loaded successfully in {:?}", start.elapsed());
            println!("Model pointer: {:p}", model.as_ptr());
            println!("Model loading test completed successfully!");
        },
        Err(e) => {
            return Err(format!("Failed to load model: {}", e).into());
        }
    }

    Ok(())
}
