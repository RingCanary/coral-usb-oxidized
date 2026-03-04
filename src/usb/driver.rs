use crate::control_plane::{VendorControlStep, VendorDirection};
use crate::error::CoralError;
use crate::usb::discovery::{
    as_device_info, collect_coral_devices, select_preferred_device, CoralUsbDeviceInfo,
};
use crate::usb::protocol::{DescriptorHeader, DescriptorTag, EventPacket, InterruptPacket};
use crate::usb::transfer::{
    control_read_at_offset, control_write_at_offset, vendor_request_type, write_bulk_all,
};
use rusb::{Context, DeviceHandle, Direction, UsbContext};
use std::cmp::min;
use std::time::Duration;

pub const EP_BULK_OUT: u8 = 0x01;
pub const EP_BULK_IN: u8 = 0x81;
pub const EP_EVENT_IN: u8 = 0x82;
pub const EP_INTERRUPT_IN: u8 = 0x83;

pub const DEFAULT_DESCRIPTOR_CHUNK_SIZE: usize = 0x100000;

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
            .iter()
            .map(|(device, desc)| as_device_info(device, desc))
            .collect())
    }

    pub fn open_first_prefer_runtime(timeout: Duration) -> Result<Self, CoralError> {
        let context = Context::new().map_err(CoralError::from)?;
        let devices = collect_coral_devices(&context)?;
        if devices.is_empty() {
            return Err(CoralError::DeviceNotFound);
        }

        let preferred = select_preferred_device(&devices).ok_or(CoralError::DeviceNotFound)?;

        let info = as_device_info(&preferred.0, &preferred.1);
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

    pub fn raw_libusb_handle(&self) -> *mut rusb::ffi::libusb_device_handle {
        self.handle.as_raw()
    }

    pub fn handle_events_timeout(&self, timeout: Option<Duration>) -> Result<(), CoralError> {
        self.handle
            .context()
            .handle_events(timeout)
            .map_err(CoralError::from)
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
        let read = self.vendor_read_exact(0x01, offset, &mut buf)?;
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
        let read = self.vendor_read_exact(0x00, offset, &mut buf)?;
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
        let written = self.vendor_write_exact(request, offset, payload)?;
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
            match step.direction {
                VendorDirection::In => {
                    let mut buf = vec![0u8; step.payload.len()];
                    let read = match self.control_read(
                        step.request_type(),
                        step.request(),
                        step.offset,
                        &mut buf,
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
                VendorDirection::Out => {
                    let written = match self.control_write(
                        step.request_type(),
                        step.request(),
                        step.offset,
                        step.payload,
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

    pub fn read_event_packet_with_timeout(
        &self,
        timeout: Duration,
    ) -> Result<EventPacket, CoralError> {
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

    fn vendor_read_exact(
        &self,
        request: u8,
        offset: u32,
        buf: &mut [u8],
    ) -> Result<usize, CoralError> {
        self.control_read(vendor_request_type(Direction::In), request, offset, buf)
            .map_err(CoralError::from)
    }

    fn vendor_write_exact(
        &self,
        request: u8,
        offset: u32,
        payload: &[u8],
    ) -> Result<usize, CoralError> {
        self.control_write(
            vendor_request_type(Direction::Out),
            request,
            offset,
            payload,
        )
        .map_err(CoralError::from)
    }

    fn control_read(
        &self,
        request_type: u8,
        request: u8,
        offset: u32,
        buf: &mut [u8],
    ) -> rusb::Result<usize> {
        control_read_at_offset(
            &self.handle,
            self.timeout,
            request_type,
            request,
            offset,
            buf,
        )
    }

    fn control_write(
        &self,
        request_type: u8,
        request: u8,
        offset: u32,
        payload: &[u8],
    ) -> rusb::Result<usize> {
        control_write_at_offset(
            &self.handle,
            self.timeout,
            request_type,
            request,
            offset,
            payload,
        )
    }

    fn write_bulk_all(&self, endpoint: u8, buf: &[u8]) -> Result<(), CoralError> {
        write_bulk_all(&self.handle, endpoint, buf, self.timeout)
    }
}

impl Drop for EdgeTpuUsbDriver {
    fn drop(&mut self) {
        self.release_claimed_interface();
    }
}
