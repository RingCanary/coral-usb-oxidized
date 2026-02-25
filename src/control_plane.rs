use rusb::{Direction, Recipient, RequestType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VendorWidth {
    U32,
    U64,
}

impl VendorWidth {
    pub const fn request(self) -> u8 {
        match self {
            VendorWidth::U64 => 0x00,
            VendorWidth::U32 => 0x01,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VendorDirection {
    In,
    Out,
}

impl VendorDirection {
    pub const fn as_usb_direction(self) -> Direction {
        match self {
            VendorDirection::In => Direction::In,
            VendorDirection::Out => Direction::Out,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VendorControlStep {
    pub direction: VendorDirection,
    pub width: VendorWidth,
    pub offset: u32,
    // For OUT steps this is write payload. For IN steps this is expected payload bytes.
    pub payload: &'static [u8],
}

impl VendorControlStep {
    pub const fn read32(offset: u32, expected: &'static [u8]) -> Self {
        Self {
            direction: VendorDirection::In,
            width: VendorWidth::U32,
            offset,
            payload: expected,
        }
    }

    pub const fn read64(offset: u32, expected: &'static [u8]) -> Self {
        Self {
            direction: VendorDirection::In,
            width: VendorWidth::U64,
            offset,
            payload: expected,
        }
    }

    pub const fn write32(offset: u32, bytes: &'static [u8]) -> Self {
        Self {
            direction: VendorDirection::Out,
            width: VendorWidth::U32,
            offset,
            payload: bytes,
        }
    }

    pub const fn write64(offset: u32, bytes: &'static [u8]) -> Self {
        Self {
            direction: VendorDirection::Out,
            width: VendorWidth::U64,
            offset,
            payload: bytes,
        }
    }

    pub fn request_type(self) -> u8 {
        rusb::request_type(
            self.direction.as_usb_direction(),
            RequestType::Vendor,
            Recipient::Device,
        )
    }

    pub const fn request(self) -> u8 {
        self.width.request()
    }
}

pub fn split_offset(offset: u32) -> (u16, u16) {
    let w_value = (offset & 0xffff) as u16;
    let w_index = (offset >> 16) as u16;
    (w_value, w_index)
}

pub fn known_register_name(offset: u32) -> Option<&'static str> {
    match offset {
        0x0004_4018 => Some("scalarCoreRunControl"),
        0x0004_4158 => Some("avDataPopRunControl"),
        0x0004_4198 => Some("parameterPopRunControl"),
        0x0004_41d8 => Some("infeedRunControl"),
        0x0004_4218 => Some("outfeedRunControl"),
        0x0004_8788 => Some("tileconfig0"),
        0x0004_a000 => Some("opRunControl"),
        0x0004_0058 => Some("omc0_ctrl_0x58"),
        0x0004_00c0 => Some("ringBusConsumer0RunControl"),
        0x0004_0110 => Some("ringBusConsumer1RunControl"),
        0x0004_0150 => Some("ringBusProducerRunControl"),
        0x0004_0190 => Some("meshBus0RunControl"),
        0x0004_01d0 => Some("meshBus1RunControl"),
        0x0004_0210 => Some("meshBus2RunControl"),
        0x0004_0250 => Some("meshBus3RunControl"),
        0x0004_0298 => Some("narrowToNarrowRunControl"),
        0x0004_02e0 => Some("narrowToWideRunControl"),
        0x0004_0328 => Some("wideToNarrowRunControl"),
        0x0004_c058 => Some("usbTopInterruptControl"),
        0x0004_c060 => Some("usbTopInterruptStatus"),
        0x0004_c070 => Some("usbFatalErrIntControl"),
        0x0004_c080 => Some("usbTopLevelIntControl"),
        0x0004_c090 => Some("usbScHostIntControl"),
        0x0004_c0a0 => Some("usbScHostIntStatus"),
        0x0004_c148 => Some("usbDmaPauseSet"),
        0x0004_c160 => Some("usbDmaPauseClear"),
        0x0001_a000 => Some("omc0_dc"),
        0x0001_a0d4 => Some("omc0_d4"),
        0x0001_a0d8 => Some("omc0_d8"),
        0x0001_a30c => Some("scu_ctrl_0"),
        0x0001_a314 => Some("scu_ctrl_2"),
        0x0001_a318 => Some("scu_ctrl_3"),
        0x0001_a33c => Some("scu_ctr_7"),
        0x0001_a500 => Some("slv_abm_en"),
        0x0001_a558 => Some("slv_err_resp_isr_mask"),
        0x0001_a600 => Some("mst_abm_en"),
        0x0001_a658 => Some("mst_err_resp_isr_mask"),
        0x0001_a704 => Some("rambist_ctrl_1"),
        _ => None,
    }
}

pub fn format_register(offset: u32) -> String {
    match known_register_name(offset) {
        Some(name) => format!("0x{offset:08x} ({name})"),
        None => format!("0x{offset:08x}"),
    }
}

// Runtime bring-up sequence recovered from geohot/edgetpuxray connect.py.
// This sequence is intentionally kept in-order.
pub const EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE: &[VendorControlStep] = &[
    VendorControlStep::read32(0x0001_a30c, &[0x01, 0x00, 0x6a, 0xb7]),
    VendorControlStep::write32(0x0001_a30c, &[0x59, 0x00, 0x0f, 0x00]),
    VendorControlStep::read32(0x0001_a314, &[0x59, 0x00, 0x0f, 0x00]),
    VendorControlStep::read32(0x0001_a318, &[0x00, 0x00, 0x00, 0x00]),
    VendorControlStep::read32(0x0001_a318, &[0x01, 0x00, 0x00, 0x00]),
    VendorControlStep::write32(0x0001_a318, &[0x5c, 0x02, 0x85, 0x50]),
    VendorControlStep::read32(0x0001_a318, &[0x5c, 0x02, 0xc5, 0x50]),
    VendorControlStep::read64(
        0x0004_4018,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_a000,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_8788,
        &[0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::read64(
        0x0004_8788,
        &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_0020,
        &[0x02, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::read32(0x0001_a314, &[0x00, 0x00, 0x00, 0x00]),
    VendorControlStep::write32(0x0001_a314, &[0x00, 0x00, 0x15, 0x00]),
    VendorControlStep::read32(0x0001_a000, &[0x01, 0x00, 0x00, 0x00]),
    VendorControlStep::write64(
        0x0004_c148,
        &[0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_c160,
        &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_c058,
        &[0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_4018,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_4158,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_4198,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_41d8,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_4218,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_8788,
        &[0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::read64(
        0x0004_8788,
        &[0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_00c0,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_0150,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_0110,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_0250,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_0298,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_02e0,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_0328,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_0190,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_01d0,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_0210,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_c060,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_c070,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_c080,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_c090,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::write64(
        0x0004_c0a0,
        &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ),
    VendorControlStep::read32(0x0001_a0d4, &[0x00, 0x00, 0x00, 0x00]),
    VendorControlStep::write32(0x0001_a0d4, &[0x01, 0x00, 0x00, 0x80]),
    VendorControlStep::read32(0x0001_a704, &[0x00, 0x00, 0x00, 0x00]),
    VendorControlStep::write32(0x0001_a704, &[0x7f, 0x00, 0x00, 0x00]),
    VendorControlStep::read32(0x0001_a33c, &[0x7f, 0x00, 0x70, 0x00]),
    VendorControlStep::write32(0x0001_a33c, &[0x3f, 0x00, 0x00, 0x00]),
    VendorControlStep::write32(0x0001_a500, &[0x01, 0x00, 0x00, 0x00]),
    VendorControlStep::write32(0x0001_a600, &[0x01, 0x00, 0x00, 0x00]),
    VendorControlStep::write32(0x0001_a558, &[0x03, 0x00, 0x00, 0x00]),
    VendorControlStep::write32(0x0001_a658, &[0x03, 0x00, 0x00, 0x00]),
    VendorControlStep::read32(0x0001_a0d8, &[0x01, 0x00, 0x00, 0x00]),
    VendorControlStep::write32(0x0001_a0d8, &[0x00, 0x00, 0x00, 0x80]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_offset_matches_expected() {
        let (w_value, w_index) = split_offset(0x0004_4018);
        assert_eq!(w_value, 0x4018);
        assert_eq!(w_index, 0x0004);
    }

    #[test]
    fn setup_sequence_has_reads_and_writes() {
        assert!(EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE
            .iter()
            .any(|s| s.direction == VendorDirection::In));
        assert!(EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE
            .iter()
            .any(|s| s.direction == VendorDirection::Out));
    }
}
