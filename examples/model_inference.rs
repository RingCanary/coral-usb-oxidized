use coral_usb_oxidized::{
    EdgeTPUDelegate, 
    CoralInterpreter, 
    TfLiteError
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to your TensorFlow Lite model file
    let model_path = "efficientnet-edgetpu-L_quant_edgetpu.tflite";
    
    // Check if the model file exists
    if !Path::new(model_path).exists() {
        println!("Model file not found: {}", model_path);
        println!("Please provide a valid TensorFlow Lite model file.");
        return Ok(());
    }
    
    println!("Creating EdgeTPU delegate...");
    // Create an EdgeTPU delegate
    let delegate = EdgeTPUDelegate::new()
        .map_err(|e| format!("Failed to create EdgeTPU delegate: {}", e))?;
    
    println!("Loading model and creating interpreter...");
    // Create a TensorFlow Lite interpreter with the EdgeTPU delegate
    let interpreter = CoralInterpreter::new(model_path, &delegate)
        .map_err(|e| format!("Failed to create interpreter: {}", e))?;
    
    // Set the number of threads to use for inference
    interpreter.set_num_threads(4)
        .map_err(|e| format!("Failed to set number of threads: {}", e))?;
    
    // Get input tensor information
    let input_count = interpreter.input_tensor_count();
    println!("Input tensor count: {}", input_count);
    
    for i in 0..input_count {
        let dims = interpreter.input_tensor_dims(i)
            .map_err(|e| format!("Failed to get input tensor dimensions: {}", e))?;
        
        let name = interpreter.input_tensor_name(i)
            .map_err(|e| format!("Failed to get input tensor name: {}", e))?;
        
        println!("Input tensor {}: name={}, dims={:?}", i, name, dims);
    }
    
    // Get output tensor information
    let output_count = interpreter.output_tensor_count();
    println!("Output tensor count: {}", output_count);
    
    for i in 0..output_count {
        let dims = interpreter.output_tensor_dims(i)
            .map_err(|e| format!("Failed to get output tensor dimensions: {}", e))?;
        
        let name = interpreter.output_tensor_name(i)
            .map_err(|e| format!("Failed to get output tensor name: {}", e))?;
        
        println!("Output tensor {}: name={}, dims={:?}", i, name, dims);
    }
    
    // Example: Prepare input data (assuming a single input tensor with float32 values)
    if input_count > 0 {
        // Get the dimensions of the first input tensor
        let dims = interpreter.input_tensor_dims(0)
            .map_err(|e| format!("Failed to get input tensor dimensions: {}", e))?;
        
        // Calculate the total number of elements
        let total_elements: i32 = dims.iter().product();
        
        // Create a dummy input tensor with all zeros
        // Note: In a real application, you would fill this with your actual input data
        let input_data = vec![0u8; (total_elements as usize) * 4]; // Assuming float32 (4 bytes per element)
        
        println!("Copying data to input tensor...");
        // Copy the input data to the input tensor
        interpreter.copy_to_input_tensor(0, &input_data)
            .map_err(|e| format!("Failed to copy data to input tensor: {}", e))?;
        
        println!("Running inference...");
        // Run inference
        interpreter.run()
            .map_err(|e| format!("Failed to run inference: {}", e))?;
        
        // Process the output (assuming a single output tensor with float32 values)
        if output_count > 0 {
            // Get the dimensions of the first output tensor
            let dims = interpreter.output_tensor_dims(0)
                .map_err(|e| format!("Failed to get output tensor dimensions: {}", e))?;
            
            // Calculate the total number of elements
            let total_elements: i32 = dims.iter().product();
            
            // Create a buffer to hold the output data
            let mut output_data = vec![0u8; (total_elements as usize) * 4]; // Assuming float32 (4 bytes per element)
            
            // Copy the output data from the output tensor
            interpreter.copy_from_output_tensor(0, &mut output_data)
                .map_err(|e| format!("Failed to copy data from output tensor: {}", e))?;
            
            println!("Inference completed successfully!");
            println!("Output data size: {} bytes", output_data.len());
            
            // In a real application, you would process the output data here
            // For example, convert the bytes back to float32 values and find the top predictions
            
            // Example: Print the first few bytes of the output data
            let max_bytes_to_print = std::cmp::min(output_data.len(), 16);
            println!("First {} bytes of output data: {:?}", max_bytes_to_print, &output_data[..max_bytes_to_print]);
        }
    }
    
    println!("Example completed successfully!");
    Ok(())
}
