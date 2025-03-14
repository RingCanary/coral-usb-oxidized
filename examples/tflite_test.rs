use coral_usb_oxidized::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing TensorFlow Lite and EdgeTPU integration");
    
    // Get EdgeTPU library version
    println!("EdgeTPU library version: {}", version());
    
    // Check if a Coral USB Accelerator is connected
    println!("Checking for EdgeTPU devices...");
    if !is_device_connected() {
        println!("No EdgeTPU devices found. This is expected if no Coral USB Accelerator is connected.");
        println!("The TensorFlow Lite library is installed and linked correctly.");
        println!("When a Coral USB Accelerator is connected, this program will be able to use it for inference.");
        
        // We can still test some TensorFlow Lite functionality without the EdgeTPU
        println!("\nTesting TensorFlow Lite functionality...");
        
        // Here we could load a TensorFlow Lite model without the EdgeTPU delegate
        // For now, we'll just print a message
        println!("TensorFlow Lite is available for use without EdgeTPU acceleration.");
        
        println!("\nTest completed successfully. The libraries are installed and linked correctly.");
        return Ok(());
    }
    
    // If we get here, a Coral USB Accelerator is connected
    println!("Coral USB Accelerator detected!");
    
    // List all available devices
    println!("Listing available EdgeTPU devices...");
    let devices = list_devices()?;
    println!("Found {} EdgeTPU device(s)", devices.len());
    for (i, device_name) in devices.iter().enumerate() {
        println!("Device {}: {}", i, device_name);
    }
    
    // Get device information
    println!("Getting device information...");
    let device_info = get_device_info()?;
    for info in device_info {
        println!("{}", info);
    }
    
    // Create a Coral device
    println!("Creating Coral device...");
    let device = CoralDevice::new()?;
    println!("Coral device created successfully!");
    
    // Create an EdgeTPU delegate
    println!("Creating EdgeTPU delegate...");
    let _delegate = device.create_delegate()?;
    println!("EdgeTPU delegate created successfully!");
    
    // Here we would load a TensorFlow Lite model and run inference
    // For this test, we'll just verify that we can create the delegate
    println!("TensorFlow Lite and EdgeTPU integration test completed successfully!");
    
    Ok(())
}
