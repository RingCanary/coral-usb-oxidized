# Coral USB Oxidized

A Rust SDK for interacting with Google Coral USB Accelerator hardware.

## Features

- Device detection and management
- USB device information retrieval
- Error handling for device operations
- EdgeTPU delegate creation for TensorFlow Lite acceleration
- Mock mode for testing without actual hardware

## Device Information

The Google Coral USB Accelerator is identified with:
- Vendor ID: `0x1a6e` (Global Unichip Corp.)
- Product ID: `0x089a`

## Important Note on Device ID Change

The Coral USB Accelerator exhibits an interesting behavior where its USB device ID changes after initialization:

- **Initial state**: When first connected, the device appears as:
  - Vendor ID: `0x1a6e` (Global Unichip Corp.)
  - Product ID: `0x089a`

- **After initialization**: After creating an EdgeTPU delegate or running the first inference, the device ID changes to:
  - Vendor ID: `0x18d1` (Google Inc.)
  - Product ID: `0x9302`

This is expected behavior according to Google and is handled automatically by this library. However, if you're setting up udev rules or other system configurations, you'll need to account for both device IDs:

```bash
# Example udev rules for Coral USB Accelerator
echo 'SUBSYSTEMS=="usb", ATTRS{idVendor}=="1a6e", ATTRS{idProduct}=="089a", MODE="0664", TAG+="uaccess"' | sudo tee -a /etc/udev/rules.d/71-edgetpu.rules > /dev/null
echo 'SUBSYSTEMS=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", MODE="0664", TAG+="uaccess"' | sudo tee -a /etc/udev/rules.d/71-edgetpu.rules > /dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Usage

### Basic Example

```rust
use coral_usb_oxidized::{CoralDevice, list_devices, version, is_device_connected};

fn main() {
    // Print the EdgeTPU library version
    println!("EdgeTPU Library Version: {}", version());
    
    // Check if a Coral USB Accelerator is connected
    println!("Coral USB Accelerator connected: {}", is_device_connected());
    
    // List available devices
    match list_devices() {
        Ok(devices) => {
            println!("Found {} device(s):", devices.len());
            for (i, device) in devices.iter().enumerate() {
                println!("  {}. {}", i + 1, device);
            }
        },
        Err(e) => {
            println!("Error listing devices: {}", e);
        }
    }
    
    // Create a new Coral device
    match CoralDevice::new() {
        Ok(device) => {
            println!("\nSuccessfully created Coral device:");
            println!("  Valid: {}", device.is_valid());
            println!("  Vendor ID: 0x{:04x}", device.vendor_id());
            println!("  Product ID: 0x{:04x}", device.product_id());
            println!("  Name: {:?}", device.name());
            
            // Device will be automatically freed when it goes out of scope
            println!("\nDevice will be freed when it goes out of scope");
        },
        Err(e) => {
            println!("Error creating device: {}", e);
        }
    }
}
```

### Running the Example

```bash
cargo run --example basic_usage
```

### EdgeTPU Delegate Usage

The SDK provides functionality to create EdgeTPU delegates for accelerating TensorFlow Lite models:

```rust
use coral_usb_oxidized::{CoralDevice, is_device_connected};

fn main() {
    // Check if a Coral USB Accelerator is connected
    if !is_device_connected() {
        println!("No Coral USB Accelerator detected. Please connect the device and try again.");
        return;
    }
    
    // Create a new Coral device
    match CoralDevice::new() {
        Ok(device) => {
            // Create an EdgeTPU delegate
            match device.create_delegate() {
                Ok(delegate) => {
                    println!("Successfully created EdgeTPU delegate!");
                    
                    // Use the delegate with TensorFlow Lite
                    // For example (pseudo-code):
                    // 
                    // let model_path = "path/to/your/model.tflite";
                    // let interpreter = tflite::Interpreter::new_with_delegate(model_path, delegate);
                    // interpreter.run();
                    
                    // The delegate will be automatically freed when it goes out of scope
                },
                Err(e) => {
                    println!("Error creating EdgeTPU delegate: {}", e);
                }
            }
        },
        Err(e) => {
            println!("Error creating device: {}", e);
        }
    }
}
```

### Running the Delegate Example

```bash
cargo run --example delegate_usage
```

### TensorFlow Lite Integration

This SDK provides direct integration with TensorFlow Lite through FFI bindings to the TensorFlow Lite C API. This allows you to load TensorFlow Lite models, create interpreters, and run inference with the EdgeTPU delegate for hardware acceleration:

```rust
use coral_usb_oxidized::{EdgeTPUDelegate, CoralInterpreter};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to your TensorFlow Lite model file
    let model_path = "model.tflite";
    
    // Create an EdgeTPU delegate
    let delegate = EdgeTPUDelegate::new()?;
    
    // Create a TensorFlow Lite interpreter with the EdgeTPU delegate
    let interpreter = CoralInterpreter::new(model_path, &delegate)?;
    
    // Set the number of threads to use for inference
    interpreter.set_num_threads(4)?;
    
    // Get input tensor information
    let input_count = interpreter.input_tensor_count();
    println!("Input tensor count: {}", input_count);
    
    // Prepare input data (example with zeros)
    let dims = interpreter.input_tensor_dims(0)?;
    let total_elements: i32 = dims.iter().product();
    let input_data = vec![0u8; (total_elements as usize) * 4]; // Assuming float32
    
    // Copy data to input tensor
    interpreter.copy_to_input_tensor(0, &input_data)?;
    
    // Run inference
    interpreter.run()?;
    
    // Get output data
    let dims = interpreter.output_tensor_dims(0)?;
    let total_elements: i32 = dims.iter().product();
    let mut output_data = vec![0u8; (total_elements as usize) * 4]; // Assuming float32
    interpreter.copy_from_output_tensor(0, &mut output_data)?;
    
    // Process results...
    
    Ok(())
}
```

### Running the Model Inference Example

```bash
cargo run --example model_inference
```

### Dependencies

To use the TensorFlow Lite integration, you need to have the TensorFlow Lite and EdgeTPU libraries installed on your system:

```bash
# Install TensorFlow Lite
# Follow the instructions at: https://www.tensorflow.org/lite/guide/build_cmake

# Install EdgeTPU runtime
# For Debian/Ubuntu:
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

### Mock Mode for Testing

The SDK includes a mock mode for testing without actual hardware. This is useful for development and testing in environments where the Coral USB Accelerator is not available:

```rust
use coral_usb_oxidized::{CoralDevice, enable_mock_mode, is_device_connected};

fn main() {
    // Enable mock mode for testing without actual hardware
    enable_mock_mode(true);
    
    // Now all calls will use mock implementations
    if is_device_connected() {
        match CoralDevice::new() {
            Ok(device) => {
                // Create a mock EdgeTPU delegate
                match device.create_delegate() {
                    Ok(delegate) => {
                        println!("Successfully created mock EdgeTPU delegate!");
                        // Use the mock delegate for testing
                    },
                    Err(e) => {
                        println!("Error creating mock EdgeTPU delegate: {}", e);
                    }
                }
            },
            Err(e) => {
                println!("Error creating mock device: {}", e);
            }
        }
    }
}
```

To run an example with mock mode enabled:

```bash
cargo run --example delegate_usage --features mock
```

## TensorFlow Lite and EdgeTPU Integration

This project requires the TensorFlow Lite C API and EdgeTPU libraries to be installed on your system.

### Installing TensorFlow Lite C API

The TensorFlow Lite C API can be built from source:

```bash
# Clone the TensorFlow repository
git clone https://github.com/tensorflow/tensorflow.git tensorflow-source
cd tensorflow-source

# Configure and build the TensorFlow Lite C API
./configure
bazel build --config=opt //tensorflow/lite/c:tensorflowlite_c

# The built library will be available at:
# bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so
```

### Installing EdgeTPU Library

The EdgeTPU library can be installed following the official Google Coral documentation:

```bash
# For Debian-based systems
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

### Running the Example

After installing the required libraries, you can run the example test program:

```bash
# Set the library path to include the TensorFlow Lite C API library
export LD_LIBRARY_PATH=/path/to/tensorflow-source/bazel-bin/tensorflow/lite/c:$LD_LIBRARY_PATH

# Run the test program
cargo run --example tflite_test
```

Alternatively, you can use the provided script:

```bash
./run_test.sh
```

### Verifying Installation

The test program will check if:
1. The TensorFlow Lite C API library is properly installed and linked
2. The EdgeTPU library is properly installed and linked
3. A Coral USB Accelerator is connected to the system

If no Coral USB Accelerator is connected, the test will still verify that the libraries are installed correctly.

## Device Verification

The SDK provides robust methods to verify that a connected device is a genuine Coral USB Accelerator:

### Basic Verification

The simplest way to verify a device is to use the `is_device_connected()` function:

```rust
use coral_usb_oxidized::is_device_connected;

if is_device_connected() {
    println!("Coral USB Accelerator detected");
} else {
    println!("No Coral USB Accelerator found");
}
```

### Comprehensive Verification

For more thorough verification, use the `CoralDevice::new()` method which attempts to create a device instance:

```rust
use coral_usb_oxidized::CoralDevice;

match CoralDevice::new() {
    Ok(device) => {
        if device.is_valid() {
            println!("Coral USB Accelerator verified");
        } else {
            println!("Device found but validation failed");
        }
    },
    Err(e) => {
        println!("Error: {}", e);
    }
}
```

## Development Status

This SDK is currently in development. The current implementation includes:

- [x] USB device detection
- [x] Device information retrieval
- [x] Basic error handling
- [x] Device verification
- [x] EdgeTPU delegate creation
- [x] TensorFlow Lite model inference
- [ ] Performance optimization

## Dependencies

- `rusb`: For USB device communication
- `libc`: For C library integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.
