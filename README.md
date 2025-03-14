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

fn is_coral_device_present() -> bool {
    match CoralDevice::new() {
        Ok(device) => device.is_valid(), // Returns true if the device is valid
        Err(_) => false,                 // Returns false if creation failed
    }
}
```

This approach goes beyond just checking USB IDs - it verifies that the device can be properly initialized, which only a genuine Coral USB Accelerator can do.

### Advanced Verification

The SDK includes a comprehensive example (`examples/verify_device.rs`) that demonstrates:

1. **Multi-level verification**:
   - USB ID validation
   - Device name validation
   - EdgeTPU initialization check

2. **Detailed error reporting**:
   - Distinguishes between different error types (device not found, permission issues, initialization failures)
   - Provides specific feedback to help troubleshoot issues

3. **Continuous monitoring**:
   - Detects when devices are connected or disconnected
   - Includes retry mechanism to handle transient issues

Run the advanced verification example:

```bash
cargo run --example verify_device
```

### Error Handling

The SDK provides granular error types to help diagnose issues:

- `CoralError::DeviceNotFound`: No device with Coral USB IDs was found
- `CoralError::InitializationFailed`: Device was found but EdgeTPU initialization failed (possible fake device)
- `CoralError::PermissionDenied`: Permission issues when accessing the USB device
- `CoralError::InvalidDeviceName`: Invalid device name provided
- `CoralError::DeviceCreationFailed`: General failure during device creation
- `CoralError::DeviceListFailed`: Failed to list available devices
- `CoralError::UsbError`: Other USB-related errors

## Development Status

This SDK is currently in development. The current implementation includes:

- [x] USB device detection
- [x] Device information retrieval
- [x] Basic error handling
- [x] Device verification
- [x] EdgeTPU delegate creation
- [ ] TensorFlow Lite model inference
- [ ] Performance optimization

## Dependencies

- `rusb`: For USB device communication
- `libc`: For C library integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.
