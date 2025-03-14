use coral_usb_oxidized::{CoralDevice, CoralInterpreter, is_device_connected, version};
use std::path::Path;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Print library version information
    let version_str = version();
    println!("EdgeTPU Library Version: {}", version_str);

    // Path to the model file
    let model_path = "models/mobilenet_v2_1.0_224_quant_edgetpu.tflite";
    let labels_path = "models/imagenet_labels.txt";

    // Check if files exist
    if !Path::new(model_path).exists() {
        return Err(format!("Model file not found: {}", model_path).into());
    }
    if !Path::new(labels_path).exists() {
        return Err(format!("Labels file not found: {}", labels_path).into());
    }

    // Load labels
    println!("Loading labels from {}", labels_path);
    let labels = load_labels(labels_path)?;
    println!("Loaded {} labels", labels.len());

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

    // Create dummy input data (all zeros)
    // For MobileNet, the input is typically [1, 224, 224, 3] with uint8 values
    let input_dims = match interpreter.input_tensor_dims(0) {
        Ok(dims) => dims,
        Err(e) => return Err(format!("Failed to get input tensor dimensions: {}", e).into())
    };
    
    println!("Input dimensions: {:?}", input_dims);
    let total_input_elements: i32 = input_dims.iter().product();
    println!("Total input elements: {}", total_input_elements);
    
    let input_data = vec![128u8; total_input_elements as usize]; // Using 128 (middle value) instead of 0
    
    println!("Copying dummy data to input tensor...");
    match interpreter.copy_to_input_tensor(0, &input_data) {
        Ok(_) => println!("Successfully copied data to input tensor"),
        Err(e) => return Err(format!("Failed to copy data to input tensor: {}", e).into())
    }

    // Run inference
    println!("Running inference...");
    let start = Instant::now();
    match interpreter.run() {
        Ok(_) => {
            let inference_time = start.elapsed();
            println!("Inference completed in {:?}", inference_time);
        },
        Err(e) => return Err(format!("Failed to run inference: {}", e).into())
    }

    // Get the output data
    let output_dims = match interpreter.output_tensor_dims(0) {
        Ok(dims) => dims,
        Err(e) => return Err(format!("Failed to get output tensor dimensions: {}", e).into())
    };
    
    println!("Output dimensions: {:?}", output_dims);
    let total_output_elements: i32 = output_dims.iter().product();
    println!("Total output elements: {}", total_output_elements);
    
    let mut output_data = vec![0u8; total_output_elements as usize];
    match interpreter.copy_from_output_tensor(0, &mut output_data) {
        Ok(_) => println!("Successfully copied data from output tensor"),
        Err(e) => return Err(format!("Failed to copy data from output tensor: {}", e).into())
    }

    // Find the top 5 scores
    let top_indices = find_top_k_indices(&output_data, 5);
    
    // Print the top results
    println!("\nTop 5 classifications (with dummy input):");
    for (i, &idx) in top_indices.iter().enumerate() {
        let score = output_data[idx] as f32 / 255.0; // Convert from uint8 to float
        let label = if idx < labels.len() { &labels[idx] } else { "Unknown" };
        println!("  {}. {} - {:.2}%", i + 1, label, score * 100.0);
    }

    println!("\nInference test completed successfully!");
    Ok(())
}

// Load labels from a file
fn load_labels(filename: &str) -> io::Result<Vec<String>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut labels = Vec::new();

    for line in reader.lines() {
        labels.push(line?);
    }

    Ok(labels)
}

// Find the indices of the top k values in a slice
fn find_top_k_indices(data: &[u8], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    
    // Sort indices by corresponding values in descending order
    indices.sort_unstable_by(|&a, &b| data[b].cmp(&data[a]));
    
    // Take the top k indices
    indices.truncate(k);
    
    indices
}
