use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use rusb::{Context, Device, DeviceDescriptor, UsbContext};

// Coral USB Accelerator device information
pub const CORAL_USB_VENDOR_ID: u16 = 0x1a6e;  // Global Unichip Corp.
pub const CORAL_USB_PRODUCT_ID: u16 = 0x089a; // Coral USB Accelerator

#[derive(Debug)]
pub enum CoralError {
    DeviceCreationFailed,
    DeviceListFailed,
    InvalidDeviceName,
    DeviceNotFound,
    InitializationFailed,
    PermissionDenied,
    UsbError(rusb::Error),
}

impl fmt::Display for CoralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoralError::DeviceCreationFailed => write!(f, "Failed to create EdgeTPU device"),
            CoralError::DeviceListFailed => write!(f, "Failed to list EdgeTPU devices"),
            CoralError::InvalidDeviceName => write!(f, "Invalid device name"),
            CoralError::DeviceNotFound => write!(f, "No Coral USB Accelerator found"),
            CoralError::InitializationFailed => write!(f, "EdgeTPU initialization failed - possible fake device"),
            CoralError::PermissionDenied => write!(f, "Permission denied - check USB access rights"),
            CoralError::UsbError(e) => write!(f, "USB error: {}", e),
        }
    }
}

impl std::error::Error for CoralError {}

impl From<rusb::Error> for CoralError {
    fn from(error: rusb::Error) -> Self {
        match error {
            rusb::Error::Access => CoralError::PermissionDenied,
            rusb::Error::NoDevice => CoralError::DeviceNotFound,
            rusb::Error::NotFound => CoralError::DeviceNotFound,
            rusb::Error::Io => CoralError::InitializationFailed,
            rusb::Error::Pipe => CoralError::InitializationFailed,
            rusb::Error::InvalidParam => CoralError::InitializationFailed,
            _ => CoralError::UsbError(error),
        }
    }
}

pub struct CoralDevice {
    is_valid: bool,
    name: Option<String>,
    vendor_id: u16,
    product_id: u16,
    // We could store a device handle here in a real implementation
}

impl CoralDevice {
    /// Create a new Coral device using the default device
    pub fn new() -> Result<Self, CoralError> {
        // Check if the device is actually connected
        if !is_device_connected() {
            return Err(CoralError::DeviceNotFound);
        }
        
        Ok(CoralDevice { 
            is_valid: true,
            name: None,
            vendor_id: CORAL_USB_VENDOR_ID,
            product_id: CORAL_USB_PRODUCT_ID,
        })
    }
    
    /// Create a new Coral device with a specific device name
    pub fn with_device_name(device_name: &str) -> Result<Self, CoralError> {
        if device_name.is_empty() {
            return Err(CoralError::InvalidDeviceName);
        }
        
        // Check if the device is actually connected
        if !is_device_connected() {
            return Err(CoralError::DeviceNotFound);
        }
        
        Ok(CoralDevice { 
            is_valid: true,
            name: Some(device_name.to_string()),
            vendor_id: CORAL_USB_VENDOR_ID,
            product_id: CORAL_USB_PRODUCT_ID,
        })
    }
    
    /// Check if the device is valid
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }
    
    /// Get the device name if available
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    /// Get the vendor ID of the device
    pub fn vendor_id(&self) -> u16 {
        self.vendor_id
    }
    
    /// Get the product ID of the device
    pub fn product_id(&self) -> u16 {
        self.product_id
    }
    
    // For testing: simulate device failure
    #[cfg(test)]
    pub fn set_mock_device_available(available: bool) {
        MOCK_DEVICE_AVAILABLE.store(available, Ordering::SeqCst);
    }
}

impl Drop for CoralDevice {
    fn drop(&mut self) {
        // In a real implementation, this would free the device resources
        self.is_valid = false;
    }
}

// Global flag to simulate device availability (only used for tests)
static MOCK_DEVICE_AVAILABLE: AtomicBool = AtomicBool::new(true);

/// Check if a Coral USB Accelerator is connected to the system
pub fn is_device_connected() -> bool {
    // In test mode, use the mock flag
    if cfg!(test) {
        return MOCK_DEVICE_AVAILABLE.load(Ordering::SeqCst);
    }
    
    // In normal mode, actually check for the device
    match find_coral_devices() {
        Ok(devices) => !devices.is_empty(),
        Err(_) => false,
    }
}

/// Find all Coral USB devices connected to the system
fn find_coral_devices() -> Result<Vec<(Device<Context>, DeviceDescriptor)>, CoralError> {
    let context = Context::new()?;
    
    let devices = context.devices()?;
    let mut result = Vec::new();
    
    for device in devices.iter() {
        let device_desc = device.device_descriptor()?;
        
        if device_desc.vendor_id() == CORAL_USB_VENDOR_ID && 
           device_desc.product_id() == CORAL_USB_PRODUCT_ID {
            result.push((device, device_desc));
        }
    }
    
    Ok(result)
}

/// List all available Coral USB devices
pub fn list_devices() -> Result<Vec<String>, CoralError> {
    // In test mode, use the mock implementation
    if cfg!(test) {
        if MOCK_DEVICE_AVAILABLE.load(Ordering::SeqCst) {
            return Ok(vec![format!("Coral USB Accelerator (VID: {:04x}, PID: {:04x})", 
                                  CORAL_USB_VENDOR_ID, 
                                  CORAL_USB_PRODUCT_ID)]);
        } else {
            return Err(CoralError::DeviceListFailed);
        }
    }
    
    // In normal mode, actually list the devices
    let devices = find_coral_devices()?;
    
    if devices.is_empty() {
        return Ok(Vec::new());
    }
    
    let mut result = Vec::new();
    let timeout = Duration::from_secs(1); // 1 second timeout for USB operations
    
    for (device, desc) in devices {
        let mut device_info = format!("Coral USB Accelerator (VID: {:04x}, PID: {:04x})", 
                                     desc.vendor_id(), 
                                     desc.product_id());
        
        // Try to get more information about the device
        if let Ok(handle) = device.open() {
            if let Ok(languages) = handle.read_languages(timeout) {
                if !languages.is_empty() {
                    let language = languages[0];
                    
                    // Try to get manufacturer string
                    if let Ok(manufacturer) = handle.read_manufacturer_string(language, &desc, timeout) {
                        device_info.push_str(&format!(", Manufacturer: {}", manufacturer));
                    }
                    
                    // Try to get product string
                    if let Ok(product) = handle.read_product_string(language, &desc, timeout) {
                        device_info.push_str(&format!(", Product: {}", product));
                    }
                    
                    // Try to get serial number
                    if let Ok(serial) = handle.read_serial_number_string(language, &desc, timeout) {
                        device_info.push_str(&format!(", S/N: {}", serial));
                    }
                }
            }
        }
        
        result.push(device_info);
    }
    
    Ok(result)
}

/// Get the EdgeTPU library version
pub fn version() -> String {
    // In a real implementation, this would call the EdgeTPU library
    // For now, return a mock version
    "Mock EdgeTPU Version 1.0".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        // Reset mock device state
        CoralDevice::set_mock_device_available(true);
        
        let device = CoralDevice::new();
        match device {
            Ok(d) => {
                println!("Device created successfully");
                assert!(d.is_valid());
                assert_eq!(d.vendor_id(), CORAL_USB_VENDOR_ID);
                assert_eq!(d.product_id(), CORAL_USB_PRODUCT_ID);
            },
            Err(e) => {
                println!("Failed to create device: {}", e);
                panic!("Device creation should succeed");
            }
        }
        
        // Test device failure
        CoralDevice::set_mock_device_available(false);
        let device = CoralDevice::new();
        assert!(device.is_err());
    }
    
    #[test]
    fn test_device_with_name() {
        // Reset mock device state
        CoralDevice::set_mock_device_available(true);
        
        let device = CoralDevice::with_device_name("TestDevice");
        match device {
            Ok(d) => {
                println!("Device created successfully with name: {:?}", d.name());
                assert!(d.is_valid());
                assert_eq!(d.name(), Some("TestDevice"));
                assert_eq!(d.vendor_id(), CORAL_USB_VENDOR_ID);
                assert_eq!(d.product_id(), CORAL_USB_PRODUCT_ID);
            },
            Err(e) => {
                println!("Failed to create device: {}", e);
                panic!("Device creation should succeed");
            }
        }
        
        // Test invalid name
        let device = CoralDevice::with_device_name("");
        assert!(device.is_err());
    }
    
    #[test]
    fn test_device_connected() {
        // Reset mock device state
        CoralDevice::set_mock_device_available(true);
        assert!(is_device_connected());
        
        CoralDevice::set_mock_device_available(false);
        assert!(!is_device_connected());
    }
    
    #[test]
    fn test_list_devices() {
        // Reset mock device state
        CoralDevice::set_mock_device_available(true);
        
        let devices = list_devices();
        match devices {
            Ok(list) => {
                println!("Available devices: {:?}", list);
                assert!(!list.is_empty());
                assert!(list[0].contains(&format!("VID: {:04x}", CORAL_USB_VENDOR_ID)));
                assert!(list[0].contains(&format!("PID: {:04x}", CORAL_USB_PRODUCT_ID)));
            },
            Err(e) => {
                println!("Failed to list devices: {}", e);
                panic!("Device listing should succeed");
            }
        }
        
        // Test device listing failure
        CoralDevice::set_mock_device_available(false);
        let devices = list_devices();
        assert!(devices.is_err());
    }
    
    #[test]
    fn test_version() {
        let ver = version();
        println!("EdgeTPU library version: {}", ver);
        assert_eq!(ver, "Mock EdgeTPU Version 1.0");
    }
}