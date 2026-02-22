use crate::delegate::EdgeTPUDelegate;
use crate::error::CoralError;
use rusb::{Context, Device, DeviceDescriptor, UsbContext};
use std::time::Duration;

pub const CORAL_USB_VENDOR_ID: u16 = 0x1a6e;
pub const CORAL_USB_PRODUCT_ID: u16 = 0x089a;

pub const CORAL_USB_VENDOR_ID_INITIALIZED: u16 = 0x18d1;
pub const CORAL_USB_PRODUCT_ID_INITIALIZED: u16 = 0x9302;

pub struct CoralDevice {
    is_valid: bool,
    name: Option<String>,
    vendor_id: u16,
    product_id: u16,
}

impl CoralDevice {
    pub fn new() -> Result<Self, CoralError> {
        find_coral_devices()?
            .into_iter()
            .next()
            .ok_or(CoralError::DeviceNotFound)
    }

    pub fn with_device_name(device_name: &str) -> Result<Self, CoralError> {
        if device_name.is_empty() {
            return Err(CoralError::InvalidDeviceName);
        }

        let mut device = find_coral_devices()?
            .into_iter()
            .next()
            .ok_or(CoralError::DeviceNotFound)?;
        device.name = Some(device_name.to_string());
        Ok(device)
    }

    pub fn create_delegate(&self) -> Result<EdgeTPUDelegate, CoralError> {
        if !self.is_valid {
            return Err(CoralError::DeviceCreationFailed);
        }

        EdgeTPUDelegate::new()
    }

    pub fn create_delegate_with_options(
        &self,
        options: &str,
    ) -> Result<EdgeTPUDelegate, CoralError> {
        if !self.is_valid {
            return Err(CoralError::DeviceCreationFailed);
        }

        EdgeTPUDelegate::with_options(options)
    }

    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn vendor_id(&self) -> u16 {
        self.vendor_id
    }

    pub fn product_id(&self) -> u16 {
        self.product_id
    }
}

impl Drop for CoralDevice {
    fn drop(&mut self) {
        self.is_valid = false;
    }
}

pub fn is_device_connected() -> bool {
    match find_coral_devices() {
        Ok(devices) => !devices.is_empty(),
        Err(_) => false,
    }
}

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

        if (desc.vendor_id() == CORAL_USB_VENDOR_ID && desc.product_id() == CORAL_USB_PRODUCT_ID)
            || (desc.vendor_id() == CORAL_USB_VENDOR_ID_INITIALIZED
                && desc.product_id() == CORAL_USB_PRODUCT_ID_INITIALIZED)
        {
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

fn get_device_name(device: &Device<Context>, desc: &DeviceDescriptor) -> Option<String> {
    let timeout = Duration::from_secs(1);

    if let Ok(handle) = device.open() {
        if let Ok(languages) = handle.read_languages(timeout) {
            if !languages.is_empty() {
                if let Ok(manufacturer) =
                    handle.read_manufacturer_string(languages[0], desc, timeout)
                {
                    return Some(manufacturer);
                }
            }
        }
    }

    None
}
