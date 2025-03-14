use coral_usb_oxidized::{CoralDevice, EdgeTPUDelegate, version, is_device_connected};

fn main() {
    // Print the EdgeTPU library version
    println!("EdgeTPU Library Version: {}", version());
    
    // Check if a Coral USB Accelerator is connected
    if !is_device_connected() {
        println!("No Coral USB Accelerator detected. Please connect the device and try again.");
        return;
    }
    
    println!("Coral USB Accelerator detected!");
    
    // Create a new Coral device
    match CoralDevice::new() {
        Ok(device) => {
            println!("\nSuccessfully created Coral device:");
            println!("  Valid: {}", device.is_valid());
            println!("  Vendor ID: 0x{:04x}", device.vendor_id());
            println!("  Product ID: 0x{:04x}", device.product_id());
            println!("  Name: {:?}", device.name());
            
            // Create an EdgeTPU delegate
            match device.create_delegate() {
                Ok(delegate) => {
                    println!("\nSuccessfully created EdgeTPU delegate!");
                    println!("  Delegate is valid: {}", delegate.is_valid());
                    println!("  Delegate pointer: {:p}", delegate.as_ptr());
                    
                    // In a real application, you would use the delegate with TensorFlow Lite
                    // For example:
                    // let model_path = "path/to/your/model.tflite";
                    // let interpreter = tflite::Interpreter::new_with_delegate(model_path, delegate);
                    // interpreter.run();
                    
                    println!("\nIn a real application, you would now:");
                    println!("1. Load a TensorFlow Lite model");
                    println!("2. Create an interpreter with the delegate");
                    println!("3. Run inference on input data");
                    println!("4. Process the results");
                    
                    // The delegate will be automatically freed when it goes out of scope
                    println!("\nDelegate will be freed when it goes out of scope");
                },
                Err(e) => {
                    println!("Error creating EdgeTPU delegate: {}", e);
                }
            }
            
            // The device will be automatically freed when it goes out of scope
            println!("Device will be freed when it goes out of scope");
        },
        Err(e) => {
            println!("Error creating device: {}", e);
        }
    }
}
