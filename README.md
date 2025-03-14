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

This SDK provides direct integration with TensorFlow Lite through FFI bindings to the TensorFlow Lite C API. This allows you to load TensorFlow Lite models, create interpreters, and run inference with the EdgeTPU delegate for hardware acceleration.

#### Current Status and Known Issues

We have implemented several examples demonstrating TensorFlow Lite integration with the EdgeTPU:

1. **Basic Model Loading**: Successfully loading TensorFlow Lite models
2. **Standard TensorFlow Lite Inference**: Running inference with standard (non-EdgeTPU) models works correctly
3. **EdgeTPU Integration**: We've encountered some challenges when integrating the EdgeTPU delegate with TensorFlow Lite:
   - Standard TensorFlow Lite models load and run successfully
   - EdgeTPU-optimized models can be loaded but require the EdgeTPU delegate for inference
   - When attempting to create a TensorFlow Lite interpreter with the EdgeTPU delegate, a segmentation fault occurs

#### Example Usage

```rust
use coral_usb_oxidized::{CoralDevice, version};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Print the EdgeTPU library version
    println!("EdgeTPU Library Version: {}", version());
    
    // Create a Coral device
    let device = CoralDevice::new()?;
    println!("Coral device created successfully!");
    
    // Create an EdgeTPU delegate
    let delegate = device.create_delegate()?;
    println!("EdgeTPU delegate created successfully!");
    
    // The delegate can now be used with TensorFlow Lite
    // Note: Direct integration with TensorFlow Lite interpreter is still being developed
    
    Ok(())
}
```

#### Working with Standard TensorFlow Lite Models

While we continue to develop the EdgeTPU integration, you can use standard TensorFlow Lite models with this SDK:

```rust
use std::path::Path;
use std::ffi::CString;
use std::os::raw::c_char;

// TensorFlow Lite C API types
pub enum TfLiteModel {}
pub enum TfLiteInterpreter {}
pub enum TfLiteInterpreterOptions {}

// FFI declarations for TensorFlow Lite C API
#[link(name = "tensorflowlite_c")]
extern "C" {
    fn TfLiteModelCreateFromFile(model_path: *const c_char) -> *mut TfLiteModel;
    fn TfLiteModelDelete(model: *mut TfLiteModel);
    
    fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);
    
    fn TfLiteInterpreterCreate(model: *mut TfLiteModel, options: *mut TfLiteInterpreterOptions) -> *mut TfLiteInterpreter;
    fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);
    
    fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> i32;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to a standard TensorFlow Lite model
    let model_path = "path/to/model.tflite";
    
    unsafe {
        // Convert path to C string
        let c_model_path = CString::new(model_path)?;
        
        // Load the model
        let model = TfLiteModelCreateFromFile(c_model_path.as_ptr());
        if model.is_null() {
            return Err("Failed to load model".into());
        }
        
        // Create interpreter options
        let options = TfLiteInterpreterOptionsCreate();
        if options.is_null() {
            TfLiteModelDelete(model);
            return Err("Failed to create interpreter options".into());
        }
        
        // Create interpreter
        let interpreter = TfLiteInterpreterCreate(model, options);
        if interpreter.is_null() {
            TfLiteInterpreterOptionsDelete(options);
            TfLiteModelDelete(model);
            return Err("Failed to create interpreter".into());
        }
        
        // Allocate tensors
        let status = TfLiteInterpreterAllocateTensors(interpreter);
        if status != 0 {
            TfLiteInterpreterDelete(interpreter);
            TfLiteInterpreterOptionsDelete(options);
            TfLiteModelDelete(model);
            return Err(format!("Failed to allocate tensors: {}", status).into());
        }
        
        // Run inference and process results...
        
        // Clean up
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
    }
    
    Ok(())
}
```

#### Next Steps

We are actively working on resolving the segmentation fault issue when creating a TensorFlow Lite interpreter with the EdgeTPU delegate. Future updates will include:

1. Full integration with the TensorFlow Lite C API
2. High-level Rust abstractions for TensorFlow Lite operations
3. Comprehensive examples for image classification and other ML tasks
4. Improved error handling and debugging capabilities

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
