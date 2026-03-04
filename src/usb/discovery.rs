use crate::error::CoralError;
use crate::usb_ids::{
    CORAL_USB_PRODUCT_ID, CORAL_USB_PRODUCT_ID_INITIALIZED, CORAL_USB_VENDOR_ID,
    CORAL_USB_VENDOR_ID_INITIALIZED,
};
use rusb::{Context, Device, DeviceDescriptor, UsbContext};

#[derive(Debug, Clone, Copy)]
pub struct CoralUsbDeviceInfo {
    pub bus: u8,
    pub address: u8,
    pub vendor_id: u16,
    pub product_id: u16,
}

impl CoralUsbDeviceInfo {
    pub const fn is_runtime(self) -> bool {
        self.vendor_id == CORAL_USB_VENDOR_ID_INITIALIZED
            && self.product_id == CORAL_USB_PRODUCT_ID_INITIALIZED
    }
}

pub(crate) fn as_device_info(
    device: &Device<Context>,
    desc: &DeviceDescriptor,
) -> CoralUsbDeviceInfo {
    CoralUsbDeviceInfo {
        bus: device.bus_number(),
        address: device.address(),
        vendor_id: desc.vendor_id(),
        product_id: desc.product_id(),
    }
}

pub fn is_coral(desc: &DeviceDescriptor) -> bool {
    (desc.vendor_id() == CORAL_USB_VENDOR_ID && desc.product_id() == CORAL_USB_PRODUCT_ID)
        || (desc.vendor_id() == CORAL_USB_VENDOR_ID_INITIALIZED
            && desc.product_id() == CORAL_USB_PRODUCT_ID_INITIALIZED)
}

pub fn collect_coral_devices(
    context: &Context,
) -> Result<Vec<(Device<Context>, DeviceDescriptor)>, CoralError> {
    let devices = context.devices().map_err(CoralError::from)?;
    let mut out = Vec::new();
    for device in devices.iter() {
        let desc = device.device_descriptor().map_err(CoralError::from)?;
        if is_coral(&desc) {
            out.push((device, desc));
        }
    }
    Ok(out)
}

pub fn select_preferred_device(
    devices: &[(Device<Context>, DeviceDescriptor)],
) -> Option<&(Device<Context>, DeviceDescriptor)> {
    devices
        .iter()
        .find(|d| d.1.vendor_id() == CORAL_USB_VENDOR_ID_INITIALIZED)
        .or_else(|| devices.first())
}
