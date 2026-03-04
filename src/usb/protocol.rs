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
