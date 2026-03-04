use crate::control_plane::split_offset;
use crate::error::CoralError;
use rusb::{Context, DeviceHandle, Direction, Recipient, RequestType};
use std::time::Duration;

pub(crate) fn vendor_request_type(direction: Direction) -> u8 {
    rusb::request_type(direction, RequestType::Vendor, Recipient::Device)
}

pub(crate) fn control_read_at_offset(
    handle: &DeviceHandle<Context>,
    timeout: Duration,
    request_type: u8,
    request: u8,
    offset: u32,
    buf: &mut [u8],
) -> rusb::Result<usize> {
    let (w_value, w_index) = split_offset(offset);
    handle.read_control(request_type, request, w_value, w_index, buf, timeout)
}

pub(crate) fn control_write_at_offset(
    handle: &DeviceHandle<Context>,
    timeout: Duration,
    request_type: u8,
    request: u8,
    offset: u32,
    payload: &[u8],
) -> rusb::Result<usize> {
    let (w_value, w_index) = split_offset(offset);
    handle.write_control(request_type, request, w_value, w_index, payload, timeout)
}

pub(crate) fn write_bulk_all(
    handle: &DeviceHandle<Context>,
    endpoint: u8,
    mut buf: &[u8],
    timeout: Duration,
) -> Result<(), CoralError> {
    while !buf.is_empty() {
        let written = handle.write_bulk(endpoint, buf, timeout).map_err(|err| {
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
