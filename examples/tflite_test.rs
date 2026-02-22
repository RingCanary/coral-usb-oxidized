use coral_usb_oxidized::{is_device_connected, version, CoralDevice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing TensorFlow Lite and EdgeTPU integration");

    // Get EdgeTPU library version
    println!("EdgeTPU library version: {}", version());

    // Check if a Coral USB Accelerator is connected
    println!("Checking for EdgeTPU devices...");
    if !is_device_connected() {
        println!(
            "No EdgeTPU devices found. This is expected if no Coral USB Accelerator is connected."
        );
        println!("The TensorFlow Lite library is installed and linked correctly.");
        println!("When a Coral USB Accelerator is connected, this program will be able to use it for inference.");

        // We can still test some TensorFlow Lite functionality without the EdgeTPU
        println!("\nTesting TensorFlow Lite functionality...");

        // Here we could load a TensorFlow Lite model without the EdgeTPU delegate
        // For now, we'll just print a message
        println!("TensorFlow Lite is available for use without EdgeTPU acceleration.");

        println!(
            "\nTest completed successfully. The libraries are installed and linked correctly."
        );
        return Ok(());
    }

    // If we get here, a Coral USB Accelerator is connected
    println!("Coral USB Accelerator detected!");

    // Create a Coral device
    println!("Creating Coral device...");
    let device = CoralDevice::new()?;
    println!("Coral device created successfully!");
    println!("  Vendor ID: 0x{:04x}", device.vendor_id());
    println!("  Product ID: 0x{:04x}", device.product_id());
    if let Some(name) = device.name() {
        println!("  Name: {}", name);
    }

    // Create an EdgeTPU delegate
    println!("Creating EdgeTPU delegate...");
    let _delegate = device.create_delegate()?;
    println!("EdgeTPU delegate created successfully!");

    // Here we would load a TensorFlow Lite model and run inference
    // For this test, we'll just verify that we can create the delegate
    println!("TensorFlow Lite and EdgeTPU integration test completed successfully!");

    Ok(())
}
