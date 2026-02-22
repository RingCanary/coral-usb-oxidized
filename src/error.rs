use std::fmt;

#[derive(Debug)]
pub enum CoralError {
    DeviceCreationFailed,
    DeviceListFailed,
    InvalidDeviceName,
    DeviceNotFound,
    InitializationFailed,
    PermissionDenied,
    UsbError(rusb::Error),
    DelegateCreationFailed,
    LibraryNotFound,
}

impl fmt::Display for CoralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoralError::DeviceCreationFailed => write!(f, "Failed to create EdgeTPU device"),
            CoralError::DeviceListFailed => write!(f, "Failed to list EdgeTPU devices"),
            CoralError::InvalidDeviceName => write!(f, "Invalid device name"),
            CoralError::DeviceNotFound => write!(f, "No Coral USB Accelerator found"),
            CoralError::InitializationFailed => {
                write!(f, "EdgeTPU initialization failed - possible fake device")
            }
            CoralError::PermissionDenied => {
                write!(f, "Permission denied - check USB access rights")
            }
            CoralError::UsbError(e) => write!(f, "USB error: {}", e),
            CoralError::DelegateCreationFailed => write!(f, "Failed to create EdgeTPU delegate"),
            CoralError::LibraryNotFound => write!(f, "EdgeTPU library not found or incompatible"),
        }
    }
}

impl std::error::Error for CoralError {}

impl From<rusb::Error> for CoralError {
    fn from(error: rusb::Error) -> Self {
        match error {
            rusb::Error::Access => CoralError::PermissionDenied,
            rusb::Error::NoDevice => CoralError::DeviceNotFound,
            rusb::Error::NotFound => CoralError::DeviceNotFound,
            rusb::Error::Io => CoralError::InitializationFailed,
            rusb::Error::Pipe => CoralError::InitializationFailed,
            rusb::Error::InvalidParam => CoralError::InitializationFailed,
            _ => CoralError::UsbError(error),
        }
    }
}

#[derive(Debug)]
pub enum TfLiteError {
    ModelLoadFailed,
    InterpreterCreationFailed,
    TensorAllocationFailed,
    InferenceFailed,
    TensorCopyFailed,
    TensorCountMismatch,
    InvalidTensorDimensions,
    InvalidTensorType,
    DelegateModificationFailed,
    Other(String),
}

impl fmt::Display for TfLiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TfLiteError::ModelLoadFailed => write!(f, "Failed to load TensorFlow Lite model"),
            TfLiteError::InterpreterCreationFailed => {
                write!(f, "Failed to create TensorFlow Lite interpreter")
            }
            TfLiteError::TensorAllocationFailed => {
                write!(f, "Failed to allocate TensorFlow Lite tensors")
            }
            TfLiteError::InferenceFailed => write!(f, "Failed to run TensorFlow Lite inference"),
            TfLiteError::TensorCopyFailed => {
                write!(f, "Failed to copy data to or from TensorFlow Lite tensors")
            }
            TfLiteError::TensorCountMismatch => write!(f, "Input/output tensor count mismatch"),
            TfLiteError::InvalidTensorDimensions => write!(f, "Invalid tensor dimensions"),
            TfLiteError::InvalidTensorType => write!(f, "Invalid tensor type"),
            TfLiteError::DelegateModificationFailed => {
                write!(f, "Failed to modify graph with delegate")
            }
            TfLiteError::Other(msg) => write!(f, "TensorFlow Lite error: {}", msg),
        }
    }
}

impl std::error::Error for TfLiteError {}

#[derive(Debug)]
pub enum DenseGemmError {
    InvalidTemplate(String),
    UnsupportedDimensions {
        input_dim: usize,
        output_dim: usize,
        reason: &'static str,
    },
    ParameterRegionNotFound,
    InvalidParameterRegionSize {
        expected: usize,
        actual: usize,
    },
    WeightSizeMismatch {
        expected: usize,
        actual: usize,
    },
    InputSizeMismatch {
        expected: usize,
        actual: usize,
    },
    OutputSizeMismatch {
        expected: usize,
        actual: usize,
    },
    TfLite(TfLiteError),
    Coral(CoralError),
    Io(std::io::Error),
}

impl fmt::Display for DenseGemmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DenseGemmError::InvalidTemplate(msg) => write!(f, "Invalid GEMM template: {}", msg),
            DenseGemmError::UnsupportedDimensions {
                input_dim,
                output_dim,
                reason,
            } => write!(
                f,
                "Unsupported Dense dimensions {}x{}: {}",
                input_dim, output_dim, reason
            ),
            DenseGemmError::ParameterRegionNotFound => {
                write!(f, "No executable parameter region found in template")
            }
            DenseGemmError::InvalidParameterRegionSize { expected, actual } => write!(
                f,
                "Unexpected parameter payload size: expected {}, got {}",
                expected, actual
            ),
            DenseGemmError::WeightSizeMismatch { expected, actual } => write!(
                f,
                "Weight matrix size mismatch: expected {}, got {}",
                expected, actual
            ),
            DenseGemmError::InputSizeMismatch { expected, actual } => write!(
                f,
                "Input vector size mismatch: expected {}, got {}",
                expected, actual
            ),
            DenseGemmError::OutputSizeMismatch { expected, actual } => write!(
                f,
                "Output vector size mismatch: expected {}, got {}",
                expected, actual
            ),
            DenseGemmError::TfLite(err) => write!(f, "TensorFlow Lite error: {}", err),
            DenseGemmError::Coral(err) => write!(f, "Coral error: {}", err),
            DenseGemmError::Io(err) => write!(f, "I/O error: {}", err),
        }
    }
}

impl std::error::Error for DenseGemmError {}

impl From<TfLiteError> for DenseGemmError {
    fn from(value: TfLiteError) -> Self {
        DenseGemmError::TfLite(value)
    }
}

impl From<CoralError> for DenseGemmError {
    fn from(value: CoralError) -> Self {
        DenseGemmError::Coral(value)
    }
}

impl From<std::io::Error> for DenseGemmError {
    fn from(value: std::io::Error) -> Self {
        DenseGemmError::Io(value)
    }
}
