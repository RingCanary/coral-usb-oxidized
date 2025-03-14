use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use rusb::{Context, Device, DeviceDescriptor, UsbContext};
use std::ptr;
use std::ffi::CString;
use std::os::raw::c_char;
#[cfg(feature = "mock")]
use std::os::raw::c_void;
#[cfg(feature = "mock")]
use libc;

// Coral USB Accelerator device information
// Initial device ID when first connected
pub const CORAL_USB_VENDOR_ID: u16 = 0x1a6e;  // Global Unichip Corp.
pub const CORAL_USB_PRODUCT_ID: u16 = 0x089a; // Coral USB Accelerator

// Device ID after initialization/first inference
pub const CORAL_USB_VENDOR_ID_INITIALIZED: u16 = 0x18d1;  // Google Inc.
pub const CORAL_USB_PRODUCT_ID_INITIALIZED: u16 = 0x9302; // Coral USB Accelerator (initialized)

// EdgeTPU device type enum
#[repr(C)]
pub enum EdgeTPUDeviceType {
    EdgetpuApexPci = 0,
    EdgetpuApexUsb = 1,
}

// EdgeTPU option struct
#[repr(C)]
pub struct EdgeTPUOption {
    name: *const c_char,
    value: *const c_char,
}

// Raw EdgeTPU delegate type from C API
#[repr(C)]
pub struct EdgeTPUDelegateRaw {
    _private: [u8; 0], // Opaque struct
}

// Define a custom type for the EdgeTPU delegate
pub type EdgeTPUDelegatePtr = *mut EdgeTPUDelegateRaw;

// Flag to enable mock mode for testing without actual hardware
#[cfg(feature = "mock")]
static MOCK_MODE: AtomicBool = AtomicBool::new(true);

#[cfg(not(feature = "mock"))]
static MOCK_MODE: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
pub enum CoralError {
    DeviceCreationFailed,
    DeviceListFailed,
    InvalidDeviceName,
    DeviceNotFound,
    InitializationFailed,
    PermissionDenied,
    UsbError(rusb::Error),
    DelegateCreationFailed,
    LibraryNotFound,
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
            CoralError::DelegateCreationFailed => write!(f, "Failed to create EdgeTPU delegate"),
            CoralError::LibraryNotFound => write!(f, "EdgeTPU library not found or incompatible"),
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

// FFI bindings for libedgetpu
#[cfg(not(feature = "mock"))]
#[link(name = "edgetpu")]
extern "C" {
    #[link_name = "edgetpu_create_delegate"]
    fn edgetpu_create_delegate(
        device_type: EdgeTPUDeviceType,
        name: *const c_char,
        options: *const EdgeTPUOption,
        num_options: usize,
    ) -> EdgeTPUDelegatePtr;
    
    #[link_name = "edgetpu_free_delegate"]
    fn edgetpu_free_delegate(delegate: EdgeTPUDelegatePtr);
    
    #[link_name = "edgetpu_version"]
    fn edgetpu_version() -> *const c_char;
}

// Mock implementations for testing without actual hardware
#[cfg(feature = "mock")]
mod mock {
    use super::*;
    
    // Mock delegate structure for testing
    static mut MOCK_DELEGATE: EdgeTPUDelegatePtr = ptr::null_mut();
    
    // Mock implementation of edgetpu_create_delegate
    pub unsafe fn edgetpu_create_delegate(
        _device_type: EdgeTPUDeviceType,
        _name: *const c_char,
        _options: *const EdgeTPUOption,
        _num_options: usize,
    ) -> EdgeTPUDelegatePtr {
        // Create a dummy delegate for testing
        if MOCK_DELEGATE.is_null() {
            MOCK_DELEGATE = libc::malloc(1) as EdgeTPUDelegatePtr;
        }
        MOCK_DELEGATE
    }
    
    // Mock implementation of edgetpu_free_delegate
    pub unsafe fn edgetpu_free_delegate(delegate: EdgeTPUDelegatePtr) {
        if !delegate.is_null() && delegate == MOCK_DELEGATE {
            libc::free(delegate as *mut c_void);
            MOCK_DELEGATE = ptr::null_mut();
        }
    }
    
    // Mock implementation of edgetpu_version
    pub unsafe fn edgetpu_version() -> *const c_char {
        b"1.0.0\0".as_ptr() as *const c_char
    }
}

// Function pointers for dynamic loading of libedgetpu
#[cfg(not(feature = "mock"))]
struct EdgeTPULibrary {
    create_delegate: Option<unsafe extern "C" fn(
        EdgeTPUDeviceType,
        *const c_char,
        *const EdgeTPUOption,
        usize,
    ) -> EdgeTPUDelegatePtr>,
    free_delegate: Option<unsafe extern "C" fn(EdgeTPUDelegatePtr)>,
    version: Option<unsafe extern "C" fn() -> *const c_char>,
}

#[cfg(not(feature = "mock"))]
impl EdgeTPULibrary {
    fn new() -> Result<Self, CoralError> {
        // This is a simplified implementation that assumes the library is already loaded
        // In a real implementation, you would use libloading crate to dynamically load the library
        Ok(EdgeTPULibrary {
            create_delegate: Some(edgetpu_create_delegate),
            free_delegate: Some(edgetpu_free_delegate),
            version: Some(edgetpu_version),
        })
    }
    
    unsafe fn create_delegate(
        &self,
        device_type: EdgeTPUDeviceType,
        name: *const c_char,
        options: *const EdgeTPUOption,
        num_options: usize,
    ) -> Result<EdgeTPUDelegatePtr, CoralError> {
        match self.create_delegate {
            Some(func) => {
                let delegate = func(device_type, name, options, num_options);
                if delegate.is_null() {
                    Err(CoralError::DelegateCreationFailed)
                } else {
                    Ok(delegate)
                }
            },
            None => Err(CoralError::LibraryNotFound),
        }
    }
    
    unsafe fn free_delegate(&self, delegate: EdgeTPUDelegatePtr) -> Result<(), CoralError> {
        match self.free_delegate {
            Some(func) => {
                func(delegate);
                Ok(())
            },
            None => Err(CoralError::LibraryNotFound),
        }
    }
    
    unsafe fn get_version(&self) -> Result<String, CoralError> {
        match self.version {
            Some(func) => {
                let c_str = func();
                if c_str.is_null() {
                    return Err(CoralError::LibraryNotFound);
                }
                let c_str = std::ffi::CStr::from_ptr(c_str);
                match c_str.to_str() {
                    Ok(s) => Ok(s.to_string()),
                    Err(_) => Err(CoralError::LibraryNotFound),
                }
            },
            None => Err(CoralError::LibraryNotFound),
        }
    }
}

/// EdgeTPU Delegate for TensorFlow Lite
///
/// This struct encapsulates the EdgeTPU delegate used to offload
/// TensorFlow Lite model computations to the Coral USB Accelerator hardware.
pub struct EdgeTPUDelegate {
    raw: EdgeTPUDelegatePtr,
    #[cfg(not(feature = "mock"))]
    library: Option<EdgeTPULibrary>,
}

impl EdgeTPUDelegate {
    /// Create a new EdgeTPU delegate for USB device
    ///
    /// This function creates a new EdgeTPU delegate for the USB device type,
    /// which is the type used by the Coral USB Accelerator.
    pub fn new() -> Result<Self, CoralError> {
        // Check if the device is actually connected
        if !is_device_connected() {
            return Err(CoralError::DeviceNotFound);
        }
        
        if MOCK_MODE.load(Ordering::SeqCst) {
            #[cfg(feature = "mock")]
            {
                unsafe {
                    let delegate = mock::edgetpu_create_delegate(
                        EdgeTPUDeviceType::EdgetpuApexUsb,
                        ptr::null(),
                        ptr::null(),
                        0,
                    );
                    if delegate.is_null() {
                        return Err(CoralError::DelegateCreationFailed);
                    }
                    Ok(EdgeTPUDelegate { 
                        raw: delegate,
                        #[cfg(not(feature = "mock"))]
                        library: None,
                    })
                }
            }
            #[cfg(not(feature = "mock"))]
            {
                // This should not be reached in mock mode
                Err(CoralError::DelegateCreationFailed)
            }
        } else {
            #[cfg(not(feature = "mock"))]
            {
                // Load the EdgeTPU library
                let library = EdgeTPULibrary::new()?;
                
                unsafe {
                    let delegate = library.create_delegate(
                        EdgeTPUDeviceType::EdgetpuApexUsb,
                        ptr::null(),
                        ptr::null(),
                        0,
                    )?;
                    Ok(EdgeTPUDelegate { 
                        raw: delegate,
                        library: Some(library),
                    })
                }
            }
            #[cfg(feature = "mock")]
            {
                // This should not be reached in non-mock mode
                Err(CoralError::DelegateCreationFailed)
            }
        }
    }
    
    /// Create a new EdgeTPU delegate with custom options
    ///
    /// This function creates a new EdgeTPU delegate with custom options
    /// provided as a string. The options string format depends on the
    /// libedgetpu implementation.
    pub fn with_options(options_str: &str) -> Result<Self, CoralError> {
        // Check if the device is actually connected
        if !is_device_connected() {
            return Err(CoralError::DeviceNotFound);
        }
        
        // Parse options string as key-value pairs
        let mut options = Vec::new();
        let mut option_cstrings = Vec::new();
        
        if !options_str.is_empty() && options_str != "{}" {
            // Very simple JSON parsing for demonstration
            // In a real implementation, you would use a proper JSON parser
            let trimmed = options_str.trim_start_matches('{').trim_end_matches('}').trim();
            if !trimmed.is_empty() {
                for pair in trimmed.split(',') {
                    let parts: Vec<&str> = pair.split(':').collect();
                    if parts.len() == 2 {
                        let key = parts[0].trim().trim_matches('"');
                        let value = parts[1].trim().trim_matches('"');
                        
                        let key_cstr = match CString::new(key) {
                            Ok(s) => s,
                            Err(_) => return Err(CoralError::DelegateCreationFailed),
                        };
                        let value_cstr = match CString::new(value) {
                            Ok(s) => s,
                            Err(_) => return Err(CoralError::DelegateCreationFailed),
                        };
                        
                        option_cstrings.push((key_cstr, value_cstr));
                    }
                }
            }
        }
        
        // Create EdgeTPUOption structs from the parsed options
        for (key, value) in &option_cstrings {
            options.push(EdgeTPUOption {
                name: key.as_ptr(),
                value: value.as_ptr(),
            });
        }
        
        if MOCK_MODE.load(Ordering::SeqCst) {
            #[cfg(feature = "mock")]
            {
                unsafe {
                    let delegate = mock::edgetpu_create_delegate(
                        EdgeTPUDeviceType::EdgetpuApexUsb,
                        ptr::null(),
                        if options.is_empty() { ptr::null() } else { options.as_ptr() },
                        options.len(),
                    );
                    if delegate.is_null() {
                        return Err(CoralError::DelegateCreationFailed);
                    }
                    Ok(EdgeTPUDelegate { 
                        raw: delegate,
                        #[cfg(not(feature = "mock"))]
                        library: None,
                    })
                }
            }
            #[cfg(not(feature = "mock"))]
            {
                // This should not be reached in mock mode
                Err(CoralError::DelegateCreationFailed)
            }
        } else {
            #[cfg(not(feature = "mock"))]
            {
                // Load the EdgeTPU library
                let library = EdgeTPULibrary::new()?;
                
                unsafe {
                    let delegate = library.create_delegate(
                        EdgeTPUDeviceType::EdgetpuApexUsb,
                        ptr::null(),
                        if options.is_empty() { ptr::null() } else { options.as_ptr() },
                        options.len(),
                    )?;
                    Ok(EdgeTPUDelegate { 
                        raw: delegate,
                        library: Some(library),
                    })
                }
            }
            #[cfg(feature = "mock")]
            {
                // This should not be reached in non-mock mode
                Err(CoralError::DelegateCreationFailed)
            }
        }
    }
    
    /// Get the raw delegate pointer
    ///
    /// This function returns the raw delegate pointer for use with
    /// TensorFlow Lite C API. This is useful when integrating with
    /// TensorFlow Lite FFI bindings.
    pub fn as_ptr(&self) -> EdgeTPUDelegatePtr {
        self.raw
    }
    
    /// Check if the delegate is valid
    ///
    /// This function returns true if the delegate is valid and can be used
    /// for inference, false otherwise.
    pub fn is_valid(&self) -> bool {
        !self.raw.is_null()
    }
    
    /// Enable mock mode for testing
    ///
    /// This function enables mock mode for testing without actual hardware.
    /// In mock mode, the delegate creation functions will return mock delegates
    /// that can be used for testing.
    pub fn enable_mock_mode(enable: bool) {
        MOCK_MODE.store(enable, Ordering::SeqCst);
    }
}

impl Drop for EdgeTPUDelegate {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                if MOCK_MODE.load(Ordering::SeqCst) {
                    #[cfg(feature = "mock")]
                    {
                        mock::edgetpu_free_delegate(self.raw);
                    }
                } else {
                    #[cfg(not(feature = "mock"))]
                    {
                        if let Some(library) = &self.library {
                            let _ = library.free_delegate(self.raw);
                        } else {
                            edgetpu_free_delegate(self.raw);
                        }
                    }
                }
                self.raw = ptr::null_mut();
            }
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
    
    /// Create an EdgeTPU delegate for this device
    ///
    /// This function creates an EdgeTPU delegate that can be used with
    /// TensorFlow Lite to accelerate inference on this device.
    pub fn create_delegate(&self) -> Result<EdgeTPUDelegate, CoralError> {
        if !self.is_valid {
            return Err(CoralError::DeviceCreationFailed);
        }
        
        EdgeTPUDelegate::new()
    }
    
    /// Create an EdgeTPU delegate with custom options for this device
    ///
    /// This function creates an EdgeTPU delegate with custom options that
    /// can be used with TensorFlow Lite to accelerate inference on this device.
    pub fn create_delegate_with_options(&self, options: &str) -> Result<EdgeTPUDelegate, CoralError> {
        if !self.is_valid {
            return Err(CoralError::DeviceCreationFailed);
        }
        
        EdgeTPUDelegate::with_options(options)
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
    if cfg!(test) || MOCK_MODE.load(Ordering::SeqCst) {
        return MOCK_DEVICE_AVAILABLE.load(Ordering::SeqCst);
    }
    
    // In normal mode, actually check for the device
    match find_coral_devices() {
        Ok(devices) => !devices.is_empty(),
        Err(_) => false,
    }
}

/// Find all Coral USB devices connected to the system
fn find_coral_devices() -> Result<Vec<CoralDevice>, CoralError> {
    let context = match Context::new() {
        Ok(ctx) => ctx,
        Err(_) => return Err(CoralError::DeviceNotFound),
    };
    
    let devices = match context.devices() {
        Ok(devs) => devs,
        Err(_) => return Err(CoralError::DeviceNotFound),
    };
    
    let mut coral_devices = Vec::new();
    
    for device in devices.iter() {
        let desc = match device.device_descriptor() {
            Ok(d) => d,
            Err(_) => continue,
        };
        
        // Check for both initial and initialized device IDs
        if (desc.vendor_id() == CORAL_USB_VENDOR_ID && desc.product_id() == CORAL_USB_PRODUCT_ID) ||
           (desc.vendor_id() == CORAL_USB_VENDOR_ID_INITIALIZED && desc.product_id() == CORAL_USB_PRODUCT_ID_INITIALIZED) {
            // Found a Coral USB Accelerator
            let name = get_device_name(&device, &desc);
            coral_devices.push(CoralDevice {
                is_valid: true,
                name,
                vendor_id: desc.vendor_id(),
                product_id: desc.product_id(),
            });
        }
    }
    
    if coral_devices.is_empty() {
        Err(CoralError::DeviceNotFound)
    } else {
        Ok(coral_devices)
    }
}

/// Get the device name from the device descriptor
fn get_device_name(device: &Device<Context>, desc: &DeviceDescriptor) -> Option<String> {
    let timeout = Duration::from_secs(1); // 1 second timeout for USB operations
    
    // Try to get manufacturer string
    if let Ok(handle) = device.open() {
        if let Ok(languages) = handle.read_languages(timeout) {
            if !languages.is_empty() {
                if let Ok(manufacturer) = handle.read_manufacturer_string(languages[0], desc, timeout) {
                    return Some(manufacturer);
                }
            }
        }
    }
    
    // If manufacturer string is not available, return None
    None
}

/// List all available Coral USB devices
pub fn list_devices() -> Result<Vec<String>, CoralError> {
    // In test mode, use the mock implementation
    if cfg!(test) || MOCK_MODE.load(Ordering::SeqCst) {
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
    
    for device in devices {
        let mut device_info = String::new();
        device_info.push_str(&format!("Vendor ID: 0x{:04x}", device.vendor_id()));
        device_info.push_str(&format!(", Product ID: 0x{:04x}", device.product_id()));
        
        if let Some(name) = &device.name {
            device_info.push_str(&format!(", Name: {}", name));
        }
        
        // Add information about device state (initial or initialized)
        if device.vendor_id == CORAL_USB_VENDOR_ID && device.product_id == CORAL_USB_PRODUCT_ID {
            device_info.push_str(" (Initial state)");
        } else if device.vendor_id == CORAL_USB_VENDOR_ID_INITIALIZED && device.product_id == CORAL_USB_PRODUCT_ID_INITIALIZED {
            device_info.push_str(" (Initialized state)");
        }
        
        result.push(device_info);
    }
    
    Ok(result)
}

/// Get information about the Coral USB Accelerator
pub fn get_device_info() -> Result<Vec<String>, CoralError> {
    let devices = find_coral_devices()?;
    let mut info = Vec::new();
    
    for device in devices {
        let mut device_info = String::new();
        device_info.push_str(&format!("Vendor ID: 0x{:04x}", device.vendor_id));
        device_info.push_str(&format!(", Product ID: 0x{:04x}", device.product_id));
        
        if let Some(name) = &device.name {
            device_info.push_str(&format!(", Name: {}", name));
        }
        
        // Add information about device state (initial or initialized)
        if device.vendor_id == CORAL_USB_VENDOR_ID && device.product_id == CORAL_USB_PRODUCT_ID {
            device_info.push_str(" (Initial state)");
        } else if device.vendor_id == CORAL_USB_VENDOR_ID_INITIALIZED && device.product_id == CORAL_USB_PRODUCT_ID_INITIALIZED {
            device_info.push_str(" (Initialized state)");
        }
        
        info.push(device_info);
    }
    
    Ok(info)
}

/// Get the EdgeTPU library version
pub fn version() -> String {
    if MOCK_MODE.load(Ordering::SeqCst) {
        #[cfg(feature = "mock")]
        {
            unsafe {
                let c_str = mock::edgetpu_version();
                if c_str.is_null() {
                    return "Unknown".to_string();
                }
                let c_str = std::ffi::CStr::from_ptr(c_str);
                match c_str.to_str() {
                    Ok(s) => s.to_string(),
                    Err(_) => "Unknown".to_string(),
                }
            }
        }
        #[cfg(not(feature = "mock"))]
        {
            "1.0.0 (mock)".to_string()
        }
    } else {
        #[cfg(not(feature = "mock"))]
        {
            unsafe {
                match EdgeTPULibrary::new() {
                    Ok(library) => {
                        match library.get_version() {
                            Ok(version) => version,
                            Err(_) => "Unknown".to_string(),
                        }
                    },
                    Err(_) => "Unknown".to_string(),
                }
            }
        }
        #[cfg(feature = "mock")]
        {
            "1.0.0 (real)".to_string()
        }
    }
}

/// Enable mock mode for testing without actual hardware
pub fn enable_mock_mode(enable: bool) {
    MOCK_MODE.store(enable, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        // Set the mock device to be available
        CoralDevice::set_mock_device_available(true);
        
        // Test successful device creation
        let device = CoralDevice::new();
        assert!(device.is_ok());
        let device = device.unwrap();
        assert!(device.is_valid());
        assert_eq!(device.vendor_id(), CORAL_USB_VENDOR_ID);
        assert_eq!(device.product_id(), CORAL_USB_PRODUCT_ID);
        assert!(device.name().is_none());
        
        // Set the mock device to be unavailable
        CoralDevice::set_mock_device_available(false);
        
        // Test device creation failure
        let device = CoralDevice::new();
        assert!(device.is_err());
        match device {
            Err(CoralError::DeviceNotFound) => (),
            _ => panic!("Expected DeviceNotFound error"),
        }
    }

    #[test]
    fn test_device_with_name() {
        // Set the mock device to be available
        CoralDevice::set_mock_device_available(true);
        
        // Test successful device creation with name
        let device = CoralDevice::with_device_name("test_device");
        assert!(device.is_ok());
        let device = device.unwrap();
        assert!(device.is_valid());
        assert_eq!(device.vendor_id(), CORAL_USB_VENDOR_ID);
        assert_eq!(device.product_id(), CORAL_USB_PRODUCT_ID);
        assert_eq!(device.name(), Some("test_device"));
        
        // Test device creation with empty name
        let device = CoralDevice::with_device_name("");
        assert!(device.is_err());
        match device {
            Err(CoralError::InvalidDeviceName) => (),
            _ => panic!("Expected InvalidDeviceName error"),
        }
    }

    #[test]
    fn test_device_connected() {
        // Set the mock device to be available
        CoralDevice::set_mock_device_available(true);
        assert!(is_device_connected());
        
        // Set the mock device to be unavailable
        CoralDevice::set_mock_device_available(false);
        assert!(!is_device_connected());
    }

    #[test]
    fn test_list_devices() {
        // Set the mock device to be available
        CoralDevice::set_mock_device_available(true);
        
        // Test successful device listing
        let devices = list_devices();
        assert!(devices.is_ok());
        let devices = devices.unwrap();
        assert_eq!(devices.len(), 1);
        assert!(devices[0].contains(&format!("VID: {:04x}", CORAL_USB_VENDOR_ID)));
        assert!(devices[0].contains(&format!("PID: {:04x}", CORAL_USB_PRODUCT_ID)));
        
        // Set the mock device to be unavailable
        CoralDevice::set_mock_device_available(false);
        
        // Test device listing failure
        let devices = list_devices();
        assert!(devices.is_err());
        match devices {
            Err(CoralError::DeviceListFailed) => (),
            _ => panic!("Expected DeviceListFailed error"),
        }
    }

    #[test]
    fn test_version() {
        let version = version();
        assert!(!version.is_empty());
    }
    
    #[test]
    fn test_delegate_creation() {
        // Enable mock mode for testing
        enable_mock_mode(true);
        
        // Set the mock device to be available
        CoralDevice::set_mock_device_available(true);
        
        // Test delegate creation through device
        let device = CoralDevice::new().unwrap();
        let delegate = device.create_delegate();
        assert!(delegate.is_ok());
        let delegate = delegate.unwrap();
        assert!(delegate.is_valid());
        
        // Test delegate creation with options
        let delegate = device.create_delegate_with_options("{}");
        assert!(delegate.is_ok());
        let delegate = delegate.unwrap();
        assert!(delegate.is_valid());
        
        // Disable mock mode
        enable_mock_mode(false);
    }
}