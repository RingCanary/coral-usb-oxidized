# Coral USB Oxidized

A Rust SDK for interacting with Google Coral USB Accelerator hardware.

## Features

- Device detection and management
- USB device information retrieval
- Error handling for device operations

## Device Information

The Google Coral USB Accelerator is identified with:
- Vendor ID: `0x1a6e` (Global Unichip Corp.)
- Product ID: `0x089a`

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
- [ ] EdgeTPU delegate creation
- [ ] TensorFlow Lite model inference
- [ ] Performance optimization

## Dependencies

- `rusb`: For USB device communication
- `libc`: For C library integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.
