use crate::control_plane::{split_offset, VendorControlStep};
use crate::device::{
    CORAL_USB_PRODUCT_ID, CORAL_USB_PRODUCT_ID_INITIALIZED, CORAL_USB_VENDOR_ID,
    CORAL_USB_VENDOR_ID_INITIALIZED,
};
use crate::error::CoralError;
use rusb::{Context, Device, DeviceDescriptor, DeviceHandle, UsbContext};
use std::cmp::min;
use std::time::Duration;

pub const EP_BULK_OUT: u8 = 0x01;
pub const EP_BULK_IN: u8 = 0x81;
pub const EP_EVENT_IN: u8 = 0x82;
pub const EP_INTERRUPT_IN: u8 = 0x83;

pub const DEFAULT_DESCRIPTOR_CHUNK_SIZE: usize = 0x100000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DescriptorTag {
    Instructions = 0,
    InputActivations = 1,
    Parameters = 2,
    OutputActivations = 3,
    Interrupt0 = 4,
    Interrupt1 = 5,
    Interrupt2 = 6,
    Interrupt3 = 7,
}

impl DescriptorTag {
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    pub const fn name(self) -> &'static str {
        match self {
            DescriptorTag::Instructions => "Instructions",
            DescriptorTag::InputActivations => "InputActivations",
            DescriptorTag::Parameters => "Parameters",
            DescriptorTag::OutputActivations => "OutputActivations",
            DescriptorTag::Interrupt0 => "Interrupt0",
            DescriptorTag::Interrupt1 => "Interrupt1",
            DescriptorTag::Interrupt2 => "Interrupt2",
            DescriptorTag::Interrupt3 => "Interrupt3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorHeader {
    pub payload_len: u32,
    pub descriptor_tag: u32,
}

impl DescriptorHeader {
    pub const fn new(payload_len: u32, descriptor_tag: DescriptorTag) -> Self {
        Self {
            payload_len,
            descriptor_tag: descriptor_tag.as_u32(),
        }
    }

    pub const fn to_le_bytes(self) -> [u8; 8] {
        let mut out = [0u8; 8];
        let len = self.payload_len.to_le_bytes();
        let tag = self.descriptor_tag.to_le_bytes();
        out[0] = len[0];
        out[1] = len[1];
        out[2] = len[2];
        out[3] = len[3];
        out[4] = tag[0];
        out[5] = tag[1];
        out[6] = tag[2];
        out[7] = tag[3];
        out
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EventPacket {
    pub offset: u64,
    pub length: u32,
    pub tag: u8,
}

impl EventPacket {
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 16 {
            return None;
        }
        let offset = u64::from_le_bytes(buf[0..8].try_into().ok()?);
        let length = u32::from_le_bytes(buf[8..12].try_into().ok()?);
        let tag = buf[12] & 0x0f;
        Some(Self {
            offset,
            length,
            tag,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InterruptPacket {
    pub raw: u32,
    pub fatal: bool,
    pub top_level_mask: u32,
}

impl InterruptPacket {
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 4 {
            return None;
        }
        let raw = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        Some(Self {
            raw,
            fatal: (raw & 1) != 0,
            top_level_mask: raw >> 1,
        })
    }
}

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

pub struct EdgeTpuUsbDriver {
    _context: Context,
    handle: DeviceHandle<Context>,
    info: CoralUsbDeviceInfo,
    timeout: Duration,
    descriptor_chunk_size: usize,
    claimed_interface: Option<u8>,
}

impl EdgeTpuUsbDriver {
    pub fn list_devices() -> Result<Vec<CoralUsbDeviceInfo>, CoralError> {
        let context = Context::new().map_err(CoralError::from)?;
        let devices = collect_coral_devices(&context)?;
        Ok(devices
            .into_iter()
            .map(|(device, desc)| CoralUsbDeviceInfo {
                bus: device.bus_number(),
                address: device.address(),
                vendor_id: desc.vendor_id(),
                product_id: desc.product_id(),
            })
            .collect())
    }

    pub fn open_first_prefer_runtime(timeout: Duration) -> Result<Self, CoralError> {
        let context = Context::new().map_err(CoralError::from)?;
        let devices = collect_coral_devices(&context)?;
        if devices.is_empty() {
            return Err(CoralError::DeviceNotFound);
        }

        let preferred = devices
            .iter()
            .find(|d| d.1.vendor_id() == CORAL_USB_VENDOR_ID_INITIALIZED)
            .or_else(|| devices.first())
            .ok_or(CoralError::DeviceNotFound)?;

        let info = CoralUsbDeviceInfo {
            bus: preferred.0.bus_number(),
            address: preferred.0.address(),
            vendor_id: preferred.1.vendor_id(),
            product_id: preferred.1.product_id(),
        };
        let handle = preferred.0.open().map_err(CoralError::from)?;
        let _ = handle.set_auto_detach_kernel_driver(true);

        Ok(Self {
            _context: context,
            handle,
            info,
            timeout,
            descriptor_chunk_size: DEFAULT_DESCRIPTOR_CHUNK_SIZE,
            claimed_interface: None,
        })
    }

    pub fn device_info(&self) -> CoralUsbDeviceInfo {
        self.info
    }

    pub fn is_runtime_device(&self) -> bool {
        self.info.is_runtime()
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    pub fn set_descriptor_chunk_size(&mut self, chunk_size: usize) -> Result<(), CoralError> {
        if chunk_size == 0 {
            return Err(CoralError::ProtocolError(
                "descriptor chunk size must be non-zero".to_string(),
            ));
        }
        self.descriptor_chunk_size = chunk_size;
        Ok(())
    }

    pub fn set_configuration_1(&self) -> Result<(), CoralError> {
        self.handle
            .set_active_configuration(1)
            .map_err(CoralError::from)
    }

    pub fn claim_interface0(&mut self) -> Result<(), CoralError> {
        if self.claimed_interface.is_none() {
            self.handle.claim_interface(0).map_err(CoralError::from)?;
            self.claimed_interface = Some(0);
        }
        Ok(())
    }

    pub fn release_claimed_interface(&mut self) {
        if let Some(intf) = self.claimed_interface.take() {
            let _ = self.handle.release_interface(intf);
        }
    }

    pub fn vendor_read32(&self, offset: u32) -> Result<u32, CoralError> {
        let mut buf = [0u8; 4];
        let (w_value, w_index) = split_offset(offset);
        let read = self
            .handle
            .read_control(
                rusb::request_type(
                    rusb::Direction::In,
                    rusb::RequestType::Vendor,
                    rusb::Recipient::Device,
                ),
                0x01,
                w_value,
                w_index,
                &mut buf,
                self.timeout,
            )
            .map_err(CoralError::from)?;
        if read != 4 {
            return Err(CoralError::ProtocolError(format!(
                "expected 4 bytes for vendor_read32 at 0x{offset:08x}, got {}",
                read
            )));
        }
        Ok(u32::from_le_bytes(buf))
    }

    pub fn vendor_read64(&self, offset: u32) -> Result<u64, CoralError> {
        let mut buf = [0u8; 8];
        let (w_value, w_index) = split_offset(offset);
        let read = self
            .handle
            .read_control(
                rusb::request_type(
                    rusb::Direction::In,
                    rusb::RequestType::Vendor,
                    rusb::Recipient::Device,
                ),
                0x00,
                w_value,
                w_index,
                &mut buf,
                self.timeout,
            )
            .map_err(CoralError::from)?;
        if read != 8 {
            return Err(CoralError::ProtocolError(format!(
                "expected 8 bytes for vendor_read64 at 0x{offset:08x}, got {}",
                read
            )));
        }
        Ok(u64::from_le_bytes(buf))
    }

    pub fn vendor_write32(&self, offset: u32, value: u32) -> Result<(), CoralError> {
        let bytes = value.to_le_bytes();
        self.vendor_write_raw(0x01, offset, &bytes)
    }

    pub fn vendor_write64(&self, offset: u32, value: u64) -> Result<(), CoralError> {
        let bytes = value.to_le_bytes();
        self.vendor_write_raw(0x00, offset, &bytes)
    }

    pub fn vendor_write_raw(
        &self,
        request: u8,
        offset: u32,
        payload: &[u8],
    ) -> Result<(), CoralError> {
        let (w_value, w_index) = split_offset(offset);
        let written = self
            .handle
            .write_control(
                rusb::request_type(
                    rusb::Direction::Out,
                    rusb::RequestType::Vendor,
                    rusb::Recipient::Device,
                ),
                request,
                w_value,
                w_index,
                payload,
                self.timeout,
            )
            .map_err(CoralError::from)?;
        if written != payload.len() {
            return Err(CoralError::ProtocolError(format!(
                "short vendor write at 0x{offset:08x}: wrote {} of {} bytes",
                written,
                payload.len()
            )));
        }
        Ok(())
    }

    pub fn apply_vendor_steps(
        &self,
        steps: &[VendorControlStep],
        verify_reads: bool,
    ) -> Result<(), CoralError> {
        for (idx, step) in steps.iter().enumerate() {
            let (w_value, w_index) = split_offset(step.offset);
            match step.direction {
                crate::control_plane::VendorDirection::In => {
                    let mut buf = vec![0u8; step.payload.len()];
                    let read = match self.handle.read_control(
                        step.request_type(),
                        step.request(),
                        w_value,
                        w_index,
                        &mut buf,
                        self.timeout,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            return Err(CoralError::ProtocolError(format!(
                                "step {} read failed at 0x{:08x}: {}",
                                idx, step.offset, err
                            )));
                        }
                    };
                    if read != buf.len() {
                        return Err(CoralError::ProtocolError(format!(
                            "step {} short read at 0x{:08x}: read {} of {} bytes",
                            idx,
                            step.offset,
                            read,
                            buf.len()
                        )));
                    }
                    if verify_reads && !step.payload.is_empty() && step.payload != &buf[..] {
                        return Err(CoralError::ProtocolError(format!(
                            "step {} read mismatch at 0x{:08x}: expected {:02x?}, got {:02x?}",
                            idx, step.offset, step.payload, buf
                        )));
                    }
                }
                crate::control_plane::VendorDirection::Out => {
                    let written = match self.handle.write_control(
                        step.request_type(),
                        step.request(),
                        w_value,
                        w_index,
                        step.payload,
                        self.timeout,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            return Err(CoralError::ProtocolError(format!(
                                "step {} write failed at 0x{:08x}: {}",
                                idx, step.offset, err
                            )));
                        }
                    };
                    if written != step.payload.len() {
                        return Err(CoralError::ProtocolError(format!(
                            "step {} short write at 0x{:08x}: wrote {} of {} bytes",
                            idx,
                            step.offset,
                            written,
                            step.payload.len()
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn send_descriptor_payload(
        &self,
        tag: DescriptorTag,
        payload: &[u8],
    ) -> Result<(), CoralError> {
        self.send_descriptor_payload_raw(tag.as_u32(), payload)
            .map_err(|err| {
                CoralError::ProtocolError(format!(
                    "descriptor {} write failed: {}",
                    tag.name(),
                    err
                ))
            })
    }

    pub fn send_descriptor_payload_raw(
        &self,
        descriptor_tag: u32,
        payload: &[u8],
    ) -> Result<(), CoralError> {
        self.send_descriptor_header_raw(descriptor_tag, payload.len())?;

        let mut offset = 0usize;
        while offset < payload.len() {
            let chunk_len = min(self.descriptor_chunk_size, payload.len() - offset);
            if let Err(err) = self.write_bulk_out_chunk(&payload[offset..offset + chunk_len]) {
                return Err(CoralError::ProtocolError(format!(
                    "descriptor tag={} payload write failed at offset {} of {} bytes: {}",
                    descriptor_tag,
                    offset,
                    payload.len(),
                    err
                )));
            }
            offset += chunk_len;
        }

        Ok(())
    }

    pub fn send_descriptor_header_raw(
        &self,
        descriptor_tag: u32,
        payload_len: usize,
    ) -> Result<(), CoralError> {
        if payload_len > u32::MAX as usize {
            return Err(CoralError::ProtocolError(format!(
                "descriptor payload too large: {} bytes",
                payload_len
            )));
        }

        let header = DescriptorHeader {
            payload_len: payload_len as u32,
            descriptor_tag,
        }
        .to_le_bytes();
        self.write_bulk_all(EP_BULK_OUT, &header)
    }

    pub fn write_bulk_out_chunk(&self, payload: &[u8]) -> Result<(), CoralError> {
        self.write_bulk_all(EP_BULK_OUT, payload)
    }

    pub fn read_event_packet_with_timeout(&self, timeout: Duration) -> Result<EventPacket, CoralError> {
        let mut buf = [0u8; 16];
        let read = self
            .handle
            .read_bulk(EP_EVENT_IN, &mut buf, timeout)
            .map_err(CoralError::from)?;
        EventPacket::decode(&buf[..read]).ok_or_else(|| {
            CoralError::ProtocolError(format!("failed to decode event packet from {} bytes", read))
        })
    }

    pub fn read_interrupt_packet_with_timeout(
        &self,
        timeout: Duration,
    ) -> Result<InterruptPacket, CoralError> {
        let mut buf = [0u8; 4];
        let read = self
            .handle
            .read_interrupt(EP_INTERRUPT_IN, &mut buf, timeout)
            .map_err(CoralError::from)?;
        InterruptPacket::decode(&buf[..read]).ok_or_else(|| {
            CoralError::ProtocolError(format!(
                "failed to decode interrupt packet from {} bytes",
                read
            ))
        })
    }

    pub fn read_output_bytes_with_timeout(
        &self,
        size: usize,
        timeout: Duration,
    ) -> Result<Vec<u8>, CoralError> {
        let mut out = vec![0u8; size];
        let read = self
            .handle
            .read_bulk(EP_BULK_IN, &mut out, timeout)
            .map_err(CoralError::from)?;
        out.truncate(read);
        Ok(out)
    }

    pub fn read_event_packet(&self) -> Result<EventPacket, CoralError> {
        self.read_event_packet_with_timeout(self.timeout)
    }

    pub fn read_interrupt_packet(&self) -> Result<InterruptPacket, CoralError> {
        self.read_interrupt_packet_with_timeout(self.timeout)
    }

    pub fn read_output_bytes(&self, size: usize) -> Result<Vec<u8>, CoralError> {
        self.read_output_bytes_with_timeout(size, self.timeout)
    }

    pub fn reset_device(&self) -> Result<(), CoralError> {
        self.handle.reset().map_err(CoralError::from)
    }

    pub fn upload_firmware_single_ep(&self, firmware: &[u8]) -> Result<(), CoralError> {
        let mut chunk_index: u16 = 0;

        for chunk in firmware.chunks(0x100) {
            let written = self
                .handle
                .write_control(0x21, 0x01, chunk_index, 0, chunk, self.timeout)
                .map_err(CoralError::from)?;
            if written != chunk.len() {
                return Err(CoralError::ProtocolError(format!(
                    "firmware short write at chunk {}: wrote {} of {} bytes",
                    chunk_index,
                    written,
                    chunk.len()
                )));
            }

            let mut status = [0u8; 6];
            let _ = self
                .handle
                .read_control(0xa1, 0x03, 0, 0, &mut status, self.timeout)
                .map_err(CoralError::from)?;
            chunk_index = chunk_index.wrapping_add(1);
        }

        let _ = self
            .handle
            .write_control(0x21, 0x01, chunk_index, 0, &[], self.timeout)
            .map_err(CoralError::from)?;
        Ok(())
    }

    fn write_bulk_all(&self, endpoint: u8, mut buf: &[u8]) -> Result<(), CoralError> {
        while !buf.is_empty() {
            let written = self
                .handle
                .write_bulk(endpoint, buf, self.timeout)
                .map_err(|err| {
                    CoralError::ProtocolError(format!(
                        "bulk write failed on endpoint 0x{endpoint:02x}: {}",
                        err
                    ))
                })?;
            if written == 0 {
                return Err(CoralError::ProtocolError(format!(
                    "zero-length bulk write on endpoint 0x{endpoint:02x}"
                )));
            }
            buf = &buf[written..];
        }
        Ok(())
    }
}

impl Drop for EdgeTpuUsbDriver {
    fn drop(&mut self) {
        self.release_claimed_interface();
    }
}

fn is_coral(desc: &DeviceDescriptor) -> bool {
    (desc.vendor_id() == CORAL_USB_VENDOR_ID && desc.product_id() == CORAL_USB_PRODUCT_ID)
        || (desc.vendor_id() == CORAL_USB_VENDOR_ID_INITIALIZED
            && desc.product_id() == CORAL_USB_PRODUCT_ID_INITIALIZED)
}

fn collect_coral_devices(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_header_is_little_endian() {
        let bytes = DescriptorHeader::new(0x11223344, DescriptorTag::Parameters).to_le_bytes();
        assert_eq!(bytes, [0x44, 0x33, 0x22, 0x11, 0x02, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn decode_event_packet() {
        let raw = [
            0x78, 0x56, 0x34, 0x12, 0, 0, 0, 0, // offset
            0x10, 0x00, 0x00, 0x00, // len
            0x02, 0, 0, 0,
        ];
        let evt = EventPacket::decode(&raw).expect("decode");
        assert_eq!(evt.offset, 0x1234_5678);
        assert_eq!(evt.length, 0x10);
        assert_eq!(evt.tag, 0x02);
    }

    #[test]
    fn decode_interrupt_packet() {
        let raw = [0x05, 0x00, 0x00, 0x00];
        let pkt = InterruptPacket::decode(&raw).expect("decode");
        assert_eq!(pkt.raw, 5);
        assert!(pkt.fatal);
        assert_eq!(pkt.top_level_mask, 2);
    }
}
