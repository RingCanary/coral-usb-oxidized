use coral_usb_oxidized::{version, CoralDevice};

// Simple function to verify the Coral USB Accelerator
// This goes beyond just checking USB IDs by attempting to create a device instance
// Only a real Edge TPU device will successfully initialize
fn is_coral_device_present() -> bool {
    match CoralDevice::new() {
        Ok(device) => device.is_valid(), // Returns true if the delegate was created and is valid
        Err(_) => false, // Returns false if creation failed (e.g., no real Edge TPU)
    }
}

fn main() {
    // Print the EdgeTPU library version
    println!("EdgeTPU Library Version: {}", version());

    // Check if a Coral USB Accelerator is present and valid
    // This is the key verification step that confirms we have a real device
    let is_present = is_coral_device_present();
    println!("Coral USB Accelerator present and valid: {}", is_present);

    if is_present {
        // Create a device instance for detailed info
        match CoralDevice::new() {
            Ok(device) => {
                println!("\nCoral Device Details:");
                println!("  Valid: {}", device.is_valid());
                println!("  Vendor ID: 0x{:04x}", device.vendor_id());
                println!("  Product ID: 0x{:04x}", device.product_id());
                println!("  Name: {:?}", device.name());

                // The device will be automatically freed when it goes out of scope
                println!("\nDevice will be freed when it goes out of scope");
            }
            Err(e) => println!("Error creating device: {}", e),
        }
    } else {
        println!("No valid Coral USB Accelerator detected.");
        println!("This could mean either:");
        println!("  1. No device with Coral USB IDs is connected");
        println!("  2. A device with matching IDs is connected but isn't a genuine Coral USB Accelerator");
    }
}
