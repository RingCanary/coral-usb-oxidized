use coral_usb_oxidized::{is_device_connected, version, CoralDevice};
use std::thread;
use std::time::Duration;

fn print_device_snapshot(stage: &str) {
    println!("\nDevice information {}:", stage);
    match CoralDevice::new() {
        Ok(device) => {
            println!("  Vendor ID: 0x{:04x}", device.vendor_id());
            println!("  Product ID: 0x{:04x}", device.product_id());
            if let Some(name) = device.name() {
                println!("  Name: {}", name);
            }
        }
        Err(e) => println!("  Unable to read device info: {}", e),
    }
}

fn main() {
    // Print the EdgeTPU library version
    println!("EdgeTPU Library Version: {}", version());

    // Check if a Coral USB Accelerator is connected
    if is_device_connected() {
        println!("Coral USB Accelerator detected!");

        print_device_snapshot("before initialization");
    } else {
        println!("No Coral USB Accelerator detected.");
        println!("Please connect a Coral USB Accelerator and try again.");
        return;
    }

    // Create a new Coral device
    let device = match CoralDevice::new() {
        Ok(device) => device,
        Err(e) => {
            println!("Error creating Coral device: {}", e);
            return;
        }
    };

    // Print device information
    println!("\nSuccessfully created Coral device:");
    println!("  Valid: {}", device.is_valid());
    println!("  Vendor ID: 0x{:04x}", device.vendor_id());
    println!("  Product ID: 0x{:04x}", device.product_id());

    // Remove the mock mode check as it's not needed

    // Try to create an EdgeTPU delegate
    println!("\nCreating EdgeTPU delegate...");

    // First try with no options
    match device.create_delegate() {
        Ok(delegate) => {
            println!("Successfully created EdgeTPU delegate with no options!");
            println!("Delegate is valid: {}", delegate.is_valid());
            println!("Delegate pointer: {:?}", delegate.as_ptr());

            // Wait a moment for the device ID to change
            println!("\nWaiting for device ID to change...");
            thread::sleep(Duration::from_secs(1));

            print_device_snapshot("after initialization");

            println!(
                "\nNote: The Coral USB Accelerator changes its device ID after initialization."
            );
            println!("Initial ID: 1a6e:089a (Global Unichip Corp.)");
            println!("After initialization: 18d1:9302 (Google Inc.)");
            println!("This is expected behavior according to Google.");

            return;
        }
        Err(e) => {
            println!("Error creating EdgeTPU delegate with no options: {}", e);
        }
    }

    // Try with different options to diagnose the issue
    let options = [
        "{\"device\":\"USB\"}",
        "{\"device\":\"usb\"}",
        "{\"device\":\"\"}",
    ];

    for option in options.iter() {
        println!("\nTrying with options: {}", option);
        match device.create_delegate_with_options(option) {
            Ok(delegate) => {
                println!("Successfully created EdgeTPU delegate!");
                println!("Delegate is valid: {}", delegate.is_valid());
                println!("Delegate pointer: {:?}", delegate.as_ptr());

                // Wait a moment for the device ID to change
                println!("\nWaiting for device ID to change...");
                thread::sleep(Duration::from_secs(1));

                print_device_snapshot("after initialization");

                println!(
                    "\nNote: The Coral USB Accelerator changes its device ID after initialization."
                );
                println!("Initial ID: 1a6e:089a (Global Unichip Corp.)");
                println!("After initialization: 18d1:9302 (Google Inc.)");
                println!("This is expected behavior according to Google.");

                return;
            }
            Err(e) => {
                println!(
                    "Error creating EdgeTPU delegate with options '{}': {}",
                    option, e
                );
            }
        }
    }

    println!("\nFailed to create EdgeTPU delegate with all options.");
    println!("This could mean:");
    println!("1. The device is not properly connected or recognized");
    println!("2. The libedgetpu library is not properly installed");
    println!("3. There are permission issues accessing the device");
    println!("4. The device may need to be reset or the system rebooted");

    println!("\nDevice will be freed when it goes out of scope");
}
