mod clip;
mod control_plane;
#[cfg(feature = "legacy-runtime")]
mod delegate;
#[cfg(feature = "legacy-runtime")]
mod device;
mod error;
mod flatbuffer;
mod function_gemma;
#[cfg(feature = "legacy-runtime")]
mod gemm;
#[cfg(feature = "legacy-runtime")]
mod interpreter;
mod toolchain;
mod usb_driver;
mod usb_ids;

pub use crate::clip::{
    quantize_linear_out_in_to_row_major_qi8, quantize_linear_out_in_to_row_major_qi8_with_config,
    ClipError, ClipSafeTensorFile, ClipTensorInfo, ClipVitB32Dims, ClipVitLayerLinearNames,
    ClipVitLinearStage, ClipVitLinearStageMeta, LinearQuantConfig, QuantizationInfo,
};
pub use crate::control_plane::{
    format_register, known_register_name, split_offset, VendorControlStep, VendorDirection,
    VendorWidth, EDGETPUXRAY_RUNTIME_SETUP_SEQUENCE, LIBEDGETPU_KNOWN_GOOD_SETUP_SEQUENCE,
};
#[cfg(feature = "legacy-runtime")]
pub use crate::delegate::{version, EdgeTPUDelegate};
#[cfg(feature = "legacy-runtime")]
pub use crate::device::{is_device_connected, CoralDevice};
pub use crate::error::{CoralError, DenseGemmError, TfLiteError};
pub use crate::flatbuffer::{
    executable_type_name, extract_serialized_executables_from_tflite, SerializedExecutableBlob,
};
pub use crate::function_gemma::{
    FunctionGemmaDims, FunctionGemmaError, FunctionGemmaLayerLinearNames, FunctionGemmaLinearStage,
    FunctionGemmaLinearStageMeta, FunctionGemmaSafeTensorFile,
};
#[cfg(feature = "legacy-runtime")]
pub use crate::gemm::{
    dense_256_param_offset, dense_param_offset, DenseGemmTemplate, PreparedDenseGemm,
    DENSE_GEMM256_DIM, DENSE_GEMM256_WEIGHT_BYTES, DENSE_GEMM256_WEIGHT_COUNT,
    DENSE_GEMM256_ZERO_POINT, TEMPLATE_2048, TEMPLATE_2304, TEMPLATE_2688,
};
#[cfg(feature = "legacy-runtime")]
pub use crate::interpreter::CoralInterpreter;
pub use crate::toolchain::{
    compile_dense_template_with_uv, DenseTemplateCompileArtifacts, DenseTemplateCompileRequest,
    ToolchainError,
};
pub use crate::usb_driver::{
    CoralUsbDeviceInfo, DescriptorHeader, DescriptorTag, EdgeTpuUsbDriver, EventPacket,
    InterruptPacket, DEFAULT_DESCRIPTOR_CHUNK_SIZE, EP_BULK_IN, EP_BULK_OUT, EP_EVENT_IN,
    EP_INTERRUPT_IN,
};
pub use crate::usb_ids::{
    CORAL_USB_PRODUCT_ID, CORAL_USB_PRODUCT_ID_INITIALIZED, CORAL_USB_VENDOR_ID,
    CORAL_USB_VENDOR_ID_INITIALIZED,
};
