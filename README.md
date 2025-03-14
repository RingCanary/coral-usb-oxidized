# Coral USB Oxidized

A Rust SDK for interacting with Google Coral USB Accelerator hardware.

## Features

- Device detection and management
- USB device information retrieval
- Error handling for device operations
- EdgeTPU delegate creation for TensorFlow Lite acceleration
- Real hardware testing with no mock code

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

### Running the Examples

We provide several examples that demonstrate different aspects of the SDK:

```bash
# Basic device detection and information
cargo run --example basic_usage

# Verify device connection and monitor connection status
cargo run --example verify_device

# Create and use EdgeTPU delegate
cargo run --example delegate_usage

# Simple delegate creation example
cargo run --example simple_delegate

# Test TensorFlow Lite integration
cargo run --example tflite_test

# Standard TensorFlow Lite example
cargo run --example tflite_standard_example
```

You can also use the custom commands defined in `.cargo/config.toml`:

```bash
cargo test-basic-usage
cargo test-verify-device
cargo test-delegate-usage
cargo test-simple-delegate
cargo test-tflite-test
cargo test-tflite-standard
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

## TensorFlow Lite Integration

This SDK provides integration with TensorFlow Lite through FFI bindings to the TensorFlow Lite C API. This allows you to load TensorFlow Lite models, create interpreters, and run inference with the EdgeTPU delegate for hardware acceleration.

### Current Status

The SDK successfully supports:
- Device detection and management
- EdgeTPU delegate creation
- Basic TensorFlow Lite integration

## Development and Testing

This project requires real Coral USB Accelerator hardware for testing. No mock code is used, ensuring that all functionality is tested against actual hardware.

### Running Tests

To run the unit tests:

```bash
cargo test-lib
```

## Prerequisites

### EdgeTPU Runtime

You'll need to install the EdgeTPU runtime:

```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

### TensorFlow Lite

For TensorFlow Lite integration, you'll need to build the TensorFlow Lite C API from source:

```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure
bazel build //tensorflow/lite/c:libtensorflowlite_c.so
```

Then, update the build.rs file to point to your TensorFlow Lite C API library path.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Coral team for the EdgeTPU hardware and software
- TensorFlow team for TensorFlow Lite
- Rust community for the excellent ecosystem
