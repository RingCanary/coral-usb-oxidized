use coral_usb_oxidized::{version, CoralDevice};
#[path = "../common/tflite_c.rs"]
mod tflite_c;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use tflite_c::{Interpreter, InterpreterOptions, Model, TfLiteDelegate};

const TFLITE_TYPE_FLOAT32: i32 = 0;
const TFLITE_TYPE_INT32: i32 = 2;
const TFLITE_TYPE_UINT8: i32 = 3;
const TFLITE_TYPE_INT64: i32 = 4;

fn get_tensor_type_name(tensor_type: i32) -> &'static str {
    match tensor_type {
        TFLITE_TYPE_FLOAT32 => "FLOAT32",
        TFLITE_TYPE_INT32 => "INT32",
        TFLITE_TYPE_UINT8 => "UINT8",
        TFLITE_TYPE_INT64 => "INT64",
        _ => "UNKNOWN",
    }
}

fn load_image(
    image_path: &str,
    width: i32,
    height: i32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    println!("Loading image: {}", image_path);

    let image_size = (width * height * 3) as usize;
    let image_data = vec![128u8; image_size];

    println!("Created dummy image data of size: {}", image_size);
    Ok(image_data)
}

fn load_labels(label_path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    println!("Loading labels: {}", label_path);

    let mut file = File::open(label_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let labels: Vec<String> = contents.lines().map(|s| s.to_string()).collect();
    println!("Loaded {} labels", labels.len());

    Ok(labels)
}

fn run_standard_model(
    model_path: &str,
    image_path: &str,
    label_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Running Standard TensorFlow Lite Model ===");

    if !Path::new(model_path).exists() {
        return Err(format!("Model file not found: {}", model_path).into());
    }
    println!("Model file found: {}", model_path);

    if !Path::new(label_path).exists() {
        return Err(format!("Label file not found: {}", label_path).into());
    }

    let labels = load_labels(label_path)?;

    println!("Loading TensorFlow Lite model...");
    let model = Model::from_file(model_path).map_err(|_| "Failed to load model")?;
    println!("Model loaded successfully");

    println!("Creating interpreter options...");
    let options = InterpreterOptions::new().map_err(|_| "Failed to create interpreter options")?;
    options.set_num_threads(1);
    println!("Interpreter options created successfully");

    println!("Creating interpreter...");
    let interpreter = Interpreter::new(&model, &options).map_err(|_| "Failed to create interpreter")?;
    println!("Interpreter created successfully");

    println!("Allocating tensors...");
    interpreter
        .allocate_tensors()
        .map_err(|status| format!("Failed to allocate tensors: {}", status))?;
    println!("Tensors allocated successfully");

    let input_count = interpreter.input_tensor_count();
    let output_count = interpreter.output_tensor_count();
    println!("Input tensor count: {}", input_count);
    println!("Output tensor count: {}", output_count);

    println!("\nInput tensor information:");
    let mut input_width = 0;
    let mut input_height = 0;
    let mut input_channels = 0;

    for i in 0..input_count {
        if let Some(tensor) = interpreter.input_tensor(i) {
            let name = tensor.name().unwrap_or_else(|| "Unknown".to_string());
            let tensor_type = tensor.tensor_type();
            let num_dims = tensor.num_dims();

            let mut dims = Vec::new();
            for d in 0..num_dims {
                dims.push(tensor.dim(d));
            }

            if num_dims == 4 {
                input_height = dims[1];
                input_width = dims[2];
                input_channels = dims[3];
            }

            let byte_size = tensor.byte_size();

            println!("  Input[{}]:", i);
            println!("    Name: {}", name);
            println!(
                "    Type: {} ({})",
                tensor_type,
                get_tensor_type_name(tensor_type)
            );
            println!("    Dimensions: {:?}", dims);
            println!("    Byte size: {}", byte_size);
        }
    }

    if input_width > 0 && input_height > 0 && input_channels > 0 {
        println!("\nPreprocessing image...");
        println!(
            "Input dimensions: {}x{}x{}",
            input_width, input_height, input_channels
        );

        let image_data = load_image(image_path, input_width, input_height)?;

        if let Some(input_tensor) = interpreter.input_tensor(0) {
            input_tensor
                .copy_from(&image_data)
                .map_err(|status| format!("Failed to copy input tensor: {}", status))?;
            println!("Image data copied to input tensor");
        }

        println!("\nRunning inference...");
        interpreter
            .invoke()
            .map_err(|status| format!("Failed to run inference: {}", status))?;
        println!("Inference completed successfully");

        println!("\nOutput tensor information and results:");
        for i in 0..output_count {
            if let Some(tensor) = interpreter.output_tensor(i) {
                let name = tensor.name().unwrap_or_else(|| "Unknown".to_string());
                let tensor_type = tensor.tensor_type();
                let num_dims = tensor.num_dims();

                let mut dims = Vec::new();
                for d in 0..num_dims {
                    dims.push(tensor.dim(d));
                }

                let byte_size = tensor.byte_size();

                println!("  Output[{}]:", i);
                println!("    Name: {}", name);
                println!(
                    "    Type: {} ({})",
                    tensor_type,
                    get_tensor_type_name(tensor_type)
                );
                println!("    Dimensions: {:?}", dims);
                println!("    Byte size: {}", byte_size);

                let mut output_data = vec![0u8; byte_size];
                tensor
                    .copy_to(&mut output_data)
                    .map_err(|status| format!("Failed to copy output tensor: {}", status))?;

                let mut indices: Vec<usize> = (0..output_data.len()).collect();
                indices.sort_unstable_by(|&a, &b| output_data[b].cmp(&output_data[a]));

                println!("\nTop 5 predictions:");
                for i in 0..5.min(indices.len()) {
                    let idx = indices[i];
                    let score = output_data[idx] as f32 / 255.0;

                    let label = if idx < labels.len() {
                        &labels[idx]
                    } else {
                        "Unknown"
                    };

                    println!("  {}. {} - {:.2}%", i + 1, label, score * 100.0);
                }
            }
        }
    } else {
        println!("Warning: Could not determine input dimensions");
    }

    println!("\nCleaning up...");
    drop(interpreter);
    drop(options);
    drop(model);
    println!("Cleanup complete");

    println!("Standard TensorFlow Lite inference completed successfully!");
    Ok(())
}

fn run_edgetpu_model(
    model_path: &str,
    image_path: &str,
    label_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Running EdgeTPU-Optimized TensorFlow Lite Model ===");

    let version_str = version();
    println!("EdgeTPU Library Version: {}", version_str);

    if !Path::new(model_path).exists() {
        return Err(format!("Model file not found: {}", model_path).into());
    }
    println!("Model file found: {}", model_path);

    if !Path::new(label_path).exists() {
        return Err(format!("Label file not found: {}", label_path).into());
    }

    let labels = load_labels(label_path)?;

    println!("Creating Coral device...");
    let device = match CoralDevice::new() {
        Ok(dev) => {
            println!("Coral device created successfully!");
            println!("  Valid: {}", dev.is_valid());
            println!("  Vendor ID: 0x{:x}", dev.vendor_id());
            println!("  Product ID: 0x{:x}", dev.product_id());
            println!("  Name: {:?}", dev.name());
            dev
        }
        Err(e) => {
            return Err(format!("Failed to create Coral device: {}", e).into());
        }
    };

    println!("Creating EdgeTPU delegate...");
    let delegate = match device.create_delegate() {
        Ok(del) => {
            println!("EdgeTPU delegate created successfully!");
            println!("  Delegate is valid: {}", del.is_valid());
            println!("  Delegate pointer: {:p}", del.as_ptr());
            del
        }
        Err(e) => {
            return Err(format!("Failed to create EdgeTPU delegate: {}", e).into());
        }
    };

    println!("Loading TensorFlow Lite model...");
    let model = Model::from_file(model_path).map_err(|_| "Failed to load model")?;
    println!("Model loaded successfully");

    println!("Creating interpreter options...");
    let options = InterpreterOptions::new().map_err(|_| "Failed to create interpreter options")?;

    println!("Adding EdgeTPU delegate to interpreter options...");
    options.add_delegate(delegate.as_ptr() as *mut TfLiteDelegate);
    println!("EdgeTPU delegate added to interpreter options");

    println!("Creating interpreter with EdgeTPU delegate...");
    let interpreter = Interpreter::new(&model, &options).map_err(|_| "Failed to create interpreter")?;
    println!("Interpreter created successfully");

    println!("Allocating tensors...");
    interpreter
        .allocate_tensors()
        .map_err(|status| format!("Failed to allocate tensors: {}", status))?;
    println!("Tensors allocated successfully");

    let input_count = interpreter.input_tensor_count();
    let output_count = interpreter.output_tensor_count();
    println!("Input tensor count: {}", input_count);
    println!("Output tensor count: {}", output_count);

    println!("\nInput tensor information:");
    let mut input_width = 0;
    let mut input_height = 0;
    let mut input_channels = 0;

    for i in 0..input_count {
        if let Some(tensor) = interpreter.input_tensor(i) {
            let name = tensor.name().unwrap_or_else(|| "Unknown".to_string());
            let tensor_type = tensor.tensor_type();
            let num_dims = tensor.num_dims();

            let mut dims = Vec::new();
            for d in 0..num_dims {
                dims.push(tensor.dim(d));
            }

            if num_dims == 4 {
                input_height = dims[1];
                input_width = dims[2];
                input_channels = dims[3];
            }

            let byte_size = tensor.byte_size();

            println!("  Input[{}]:", i);
            println!("    Name: {}", name);
            println!(
                "    Type: {} ({})",
                tensor_type,
                get_tensor_type_name(tensor_type)
            );
            println!("    Dimensions: {:?}", dims);
            println!("    Byte size: {}", byte_size);
        }
    }

    if input_width > 0 && input_height > 0 && input_channels > 0 {
        println!("\nPreprocessing image...");
        println!(
            "Input dimensions: {}x{}x{}",
            input_width, input_height, input_channels
        );

        let image_data = load_image(image_path, input_width, input_height)?;

        if let Some(input_tensor) = interpreter.input_tensor(0) {
            input_tensor
                .copy_from(&image_data)
                .map_err(|status| format!("Failed to copy input tensor: {}", status))?;
            println!("Image data copied to input tensor");
        }

        println!("\nRunning inference with EdgeTPU...");
        interpreter
            .invoke()
            .map_err(|status| format!("Failed to run inference: {}", status))?;
        println!("Inference completed successfully");

        println!("\nOutput tensor information and results:");
        for i in 0..output_count {
            if let Some(tensor) = interpreter.output_tensor(i) {
                let name = tensor.name().unwrap_or_else(|| "Unknown".to_string());
                let tensor_type = tensor.tensor_type();
                let num_dims = tensor.num_dims();

                let mut dims = Vec::new();
                for d in 0..num_dims {
                    dims.push(tensor.dim(d));
                }

                let byte_size = tensor.byte_size();

                println!("  Output[{}]:", i);
                println!("    Name: {}", name);
                println!(
                    "    Type: {} ({})",
                    tensor_type,
                    get_tensor_type_name(tensor_type)
                );
                println!("    Dimensions: {:?}", dims);
                println!("    Byte size: {}", byte_size);

                let mut output_data = vec![0u8; byte_size];
                tensor
                    .copy_to(&mut output_data)
                    .map_err(|status| format!("Failed to copy output tensor: {}", status))?;

                let mut indices: Vec<usize> = (0..output_data.len()).collect();
                indices.sort_unstable_by(|&a, &b| output_data[b].cmp(&output_data[a]));

                println!("\nTop 5 predictions:");
                for i in 0..5.min(indices.len()) {
                    let idx = indices[i];
                    let score = output_data[idx] as f32 / 255.0;

                    let label = if idx < labels.len() {
                        &labels[idx]
                    } else {
                        "Unknown"
                    };

                    println!("  {}. {} - {:.2}%", i + 1, label, score * 100.0);
                }
            }
        }
    } else {
        println!("Warning: Could not determine input dimensions");
    }

    println!("\nCleaning up...");
    drop(interpreter);
    drop(options);
    drop(model);
    println!("Cleanup complete");

    println!("EdgeTPU TensorFlow Lite inference completed successfully!");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let standard_model_path = "models/standard/mobilenet_v1_1.0_224_quant.tflite";
    let edgetpu_model_path = "models/mobilenet_v2_1.0_224_quant_edgetpu.tflite";

    let standard_label_path = "models/standard/temp/labels_mobilenet_quant_v1_224.txt";
    let edgetpu_label_path = "models/imagenet_labels.txt";

    let image_path = "models/grace_hopper.bmp";

    match run_standard_model(standard_model_path, image_path, standard_label_path) {
        Ok(_) => println!("Standard model inference completed successfully"),
        Err(e) => println!("Error running standard model inference: {}", e),
    }

    match run_edgetpu_model(edgetpu_model_path, image_path, edgetpu_label_path) {
        Ok(_) => println!("EdgeTPU model inference completed successfully"),
        Err(e) => println!("Error running EdgeTPU model inference: {}", e),
    }

    Ok(())
}
