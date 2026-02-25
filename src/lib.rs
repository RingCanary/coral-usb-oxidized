mod clip;
mod control_plane;
mod delegate;
mod device;
mod error;
mod flatbuffer;
mod function_gemma;
mod gemm;
mod interpreter;
mod usb_driver;

pub use crate::clip::{
    quantize_linear_out_in_to_row_major_qi8, quantize_linear_out_in_to_row_major_qi8_with_config,
    ClipError, ClipSafeTensorFile, ClipTensorInfo, ClipVitB32Dims, ClipVitLayerLinearNames,
    ClipVitLinearStage, ClipVitLinearStageMeta, LinearQuantConfig, QuantizationInfo,
};
pub use crate::control_plane::{
    format_register, known_register_name, split_offset, VendorControlStep, VendorDirection,
    VendorWidth, EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE,
};
pub use crate::delegate::{version, EdgeTPUDelegate};
pub use crate::device::{is_device_connected, CoralDevice};
pub use crate::error::{CoralError, DenseGemmError, TfLiteError};
pub use crate::flatbuffer::{
    executable_type_name, extract_serialized_executables_from_tflite, SerializedExecutableBlob,
};
pub use crate::function_gemma::{
    FunctionGemmaDims, FunctionGemmaError, FunctionGemmaLayerLinearNames, FunctionGemmaLinearStage,
    FunctionGemmaLinearStageMeta, FunctionGemmaSafeTensorFile,
};
pub use crate::gemm::{
    dense_256_param_offset, dense_param_offset, DenseGemmTemplate, PreparedDenseGemm,
    DENSE_GEMM256_DIM, DENSE_GEMM256_WEIGHT_BYTES, DENSE_GEMM256_WEIGHT_COUNT,
    DENSE_GEMM256_ZERO_POINT, TEMPLATE_2048, TEMPLATE_2304, TEMPLATE_2688,
};
pub use crate::interpreter::CoralInterpreter;
pub use crate::usb_driver::{
    CoralUsbDeviceInfo, DescriptorHeader, DescriptorTag, EdgeTpuUsbDriver, EventPacket,
    InterruptPacket, DEFAULT_DESCRIPTOR_CHUNK_SIZE, EP_BULK_IN, EP_BULK_OUT, EP_EVENT_IN,
    EP_INTERRUPT_IN,
};
