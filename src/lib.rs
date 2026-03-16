mod clip;
mod control_plane;
#[cfg(feature = "legacy-runtime")]
mod delegate;
#[cfg(feature = "legacy-runtime")]
mod device;
mod error;
mod family_profile;
mod flatbuffer;
mod function_gemma;
#[cfg(feature = "legacy-runtime")]
mod gemm;
#[cfg(feature = "legacy-runtime")]
mod interpreter;
mod param_pack;
mod safetensor_utils;
mod toolchain;
mod usb;
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
pub use crate::family_profile::{
    DenseFamilyInstructionPatchDimMatch, DenseFamilyInstructionPatchGeneric,
    DenseFamilyInstructionPatchOverlay, DenseFamilyInstructionPatches, DenseFamilyProfile,
    DenseFamilyProfileError, DenseFamilyReplayDefaults, DenseFamilyResolvedInstructionOverlayPaths,
};
pub use crate::flatbuffer::{
    executable_type_name, extract_instruction_chunk_from_serialized_executable,
    extract_serialized_executables_from_tflite, extract_tflite_conv1x1_quant_params,
    extract_tflite_conv_quantized_weights, Conv1x1QuantParams, ConvQuantizedWeights,
    SerializedExecutableBlob,
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
pub use crate::param_pack::{
    conv1x1_effective_scales_from_quant_params, conv1x1_param_stream_len,
    conv1x1_param_stream_offset, conv1x1_param_stream_prefix_len,
    conv1x1_stored_zero_points_from_quant_params, dense_param_stream_len,
    dense_param_stream_offset, pack_conv1x1_row_major_i8_to_stream,
    pack_conv1x1_row_major_i8_to_stream_with_quant_params, pack_conv1x1_row_major_u8_to_stream,
    pack_conv1x1_row_major_u8_to_stream_with_quant_params, pack_dense_row_major_i8_to_stream,
    pack_dense_row_major_u8_to_stream, unpack_conv1x1_stream_to_row_major_i8,
    unpack_conv1x1_stream_to_row_major_u8, unpack_dense_stream_to_row_major_i8,
    unpack_dense_stream_to_row_major_u8, Conv1x1ParamPackError, DenseParamPackError,
};
pub use crate::toolchain::{
    compile_dense_template_with_uv, DenseTemplateCompileArtifacts, DenseTemplateCompileRequest,
    ToolchainError,
};
pub use crate::usb::{
    CoralUsbDeviceInfo, DescriptorHeader, DescriptorTag, EdgeTpuUsbDriver, EventPacket,
    InterruptPacket, DEFAULT_DESCRIPTOR_CHUNK_SIZE, EP_BULK_IN, EP_BULK_OUT, EP_EVENT_IN,
    EP_INTERRUPT_IN,
};
pub use crate::usb_ids::{
    CORAL_USB_PRODUCT_ID, CORAL_USB_PRODUCT_ID_INITIALIZED, CORAL_USB_VENDOR_ID,
    CORAL_USB_VENDOR_ID_INITIALIZED,
};
