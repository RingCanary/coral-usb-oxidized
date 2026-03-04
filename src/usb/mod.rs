pub mod discovery;
pub mod driver;
pub mod protocol;
pub mod transfer;

pub use discovery::CoralUsbDeviceInfo;
pub use driver::{
    EdgeTpuUsbDriver, DEFAULT_DESCRIPTOR_CHUNK_SIZE, EP_BULK_IN, EP_BULK_OUT, EP_EVENT_IN,
    EP_INTERRUPT_IN,
};
pub use protocol::{DescriptorHeader, DescriptorTag, EventPacket, InterruptPacket};
