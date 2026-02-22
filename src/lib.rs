mod delegate;
mod device;
mod error;
mod flatbuffer;
mod gemm;
mod interpreter;

pub use crate::delegate::{
    version, EdgeTPUDelegate, EdgeTPUDelegatePtr, EdgeTPUDelegateRaw, EdgeTPUDeviceType,
    EdgeTPUOption,
};
pub use crate::device::{
    get_device_info, is_device_connected, list_devices, CoralDevice, CORAL_USB_PRODUCT_ID,
    CORAL_USB_PRODUCT_ID_INITIALIZED, CORAL_USB_VENDOR_ID, CORAL_USB_VENDOR_ID_INITIALIZED,
};
pub use crate::error::{CoralError, DenseGemmError, TfLiteError};
pub use crate::gemm::{
    dense_256_param_offset, dense_param_offset, DenseGemmTemplate, PreparedDenseGemm,
    DENSE_GEMM256_DIM, DENSE_GEMM256_WEIGHT_BYTES, DENSE_GEMM256_WEIGHT_COUNT,
    DENSE_GEMM256_ZERO_POINT, TEMPLATE_2048, TEMPLATE_2304, TEMPLATE_2688,
};
pub use crate::interpreter::{
    CoralInterpreter, TfLiteDelegate, TfLiteInterpreter, TfLiteInterpreterOptions, TfLiteModel,
    TfLiteModelWrapper, TfLiteTensor,
};
