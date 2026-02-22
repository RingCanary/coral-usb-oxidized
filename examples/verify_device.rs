use chrono;
use coral_usb_oxidized::{list_devices, version, CoralDevice, CoralError};
use std::thread;
use std::time::Duration;

// Expected device name for validation
const EXPECTED_DEVICE_NAME: &str = "Coral USB Accelerator";

/// Comprehensive function to verify if a connected device is a genuine Coral USB Accelerator
/// Returns a tuple with (is_present, detailed_message)
fn verify_coral_device() -> (bool, String) {
    // Step 1: Try to create a device instance
    // This is the key verification step - only a real Edge TPU can successfully initialize
    let device_result = CoralDevice::new();

    match device_result {
        Ok(device) => {
            if !device.is_valid() {
                return (false, "Device created but reported as invalid".to_string());
            }

            // Step 2: Validate device name if available
            if let Some(name) = device.name() {
                if name != EXPECTED_DEVICE_NAME {
                    return (
                        false,
                        format!(
                            "Device name mismatch: expected '{}', got '{}'",
                            EXPECTED_DEVICE_NAME, name
                        ),
                    );
                }
            }

            // Step 3: Get device details for additional verification
            let mut details = format!(
                "Verified Coral USB Accelerator:\n\
                 - Vendor ID: 0x{:04x}\n\
                 - Product ID: 0x{:04x}",
                device.vendor_id(),
                device.product_id()
            );

            if let Some(name) = device.name() {
                details.push_str(&format!("\n- Device Name: {}", name));
            }

            // Step 4: Get additional device information from USB descriptors
            if let Ok(device_list) = list_devices() {
                if !device_list.is_empty() {
                    details.push_str("\n- USB Descriptor Info: ");
                    details.push_str(&device_list[0]);
                }
            }

            (true, details)
        }
        Err(CoralError::DeviceNotFound) => (false, "No Coral USB device found".to_string()),
        Err(CoralError::InitializationFailed) => (
            false,
            "Device found but Edge TPU initialization failed - possible fake device".to_string(),
        ),
        Err(CoralError::PermissionDenied) => (
            false,
            "Permission denied - check USB access rights".to_string(),
        ),
        Err(e) => (false, format!("Unexpected error: {}", e)),
    }
}

fn main() {
    println!("EdgeTPU Library Version: {}", version());
    println!("Verifying Coral USB Accelerator...");

    // First verification
    let (is_genuine, details) = verify_coral_device();

    if is_genuine {
        println!("✅ VERIFICATION SUCCESSFUL");
        println!("{}", details);
    } else {
        println!("❌ VERIFICATION FAILED");
        println!("{}", details);
        println!("\nPossible reasons:");
        println!("  1. No USB device with Coral IDs is connected");
        println!("  2. A device with matching IDs is connected but isn't a genuine Coral USB Accelerator");
        println!("  3. The Edge TPU runtime failed to initialize");
        println!("  4. Insufficient permissions to access the USB device");

        // Exit with error code
        std::process::exit(1);
    }

    // Demonstrate continuous monitoring (optional)
    println!("\nContinuously monitoring device presence (press Ctrl+C to stop)...");
    let mut last_state = is_genuine;

    loop {
        // Implement retry mechanism for transient issues
        let mut retries = 3;
        let mut current_state = false;
        let mut current_details = String::new();

        // Try up to 3 times with short delays between attempts
        while retries > 0 {
            let (state, details) = verify_coral_device();
            current_state = state;
            current_details = details;

            // If state changed or we're on the last retry, break out
            if current_state != last_state || retries == 1 {
                break;
            }

            retries -= 1;
            thread::sleep(Duration::from_millis(500));
        }

        // Only report state changes
        if current_state != last_state {
            if current_state {
                println!(
                    "✅ Device CONNECTED at {}",
                    chrono::Local::now().format("%H:%M:%S")
                );
                println!("{}", current_details);
            } else {
                println!(
                    "❌ Device DISCONNECTED at {}",
                    chrono::Local::now().format("%H:%M:%S")
                );
                println!("Reason: {}", current_details);
            }
            last_state = current_state;
        }

        // Wait before next check
        thread::sleep(Duration::from_secs(2));
    }
}
