mod delegate;
mod device;
mod error;
mod flatbuffer;
mod gemm;
mod interpreter;

pub use crate::delegate::{version, EdgeTPUDelegate};
pub use crate::device::{is_device_connected, CoralDevice};
pub use crate::error::{CoralError, DenseGemmError, TfLiteError};
pub use crate::gemm::{
    dense_256_param_offset, dense_param_offset, DenseGemmTemplate, PreparedDenseGemm,
    DENSE_GEMM256_DIM, DENSE_GEMM256_WEIGHT_BYTES, DENSE_GEMM256_WEIGHT_COUNT,
    DENSE_GEMM256_ZERO_POINT, TEMPLATE_2048, TEMPLATE_2304, TEMPLATE_2688,
};
pub use crate::interpreter::CoralInterpreter;
