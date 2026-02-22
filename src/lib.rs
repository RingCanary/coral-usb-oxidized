use rusb::{Context, Device, DeviceDescriptor, UsbContext};
use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

// Coral USB Accelerator device information
// Initial device ID when first connected
pub const CORAL_USB_VENDOR_ID: u16 = 0x1a6e; // Global Unichip Corp.
pub const CORAL_USB_PRODUCT_ID: u16 = 0x089a; // Coral USB Accelerator

// Device ID after initialization/first inference
pub const CORAL_USB_VENDOR_ID_INITIALIZED: u16 = 0x18d1; // Google Inc.
pub const CORAL_USB_PRODUCT_ID_INITIALIZED: u16 = 0x9302; // Coral USB Accelerator (initialized)

// EdgeTPU device type enum
#[repr(C)]
pub enum EdgeTPUDeviceType {
    EdgetpuApexPci = 0,
    EdgetpuApexUsb = 1,
}

// EdgeTPU option struct
#[repr(C)]
pub struct EdgeTPUOption {
    name: *const c_char,
    value: *const c_char,
}

// EdgeTPU device information returned by the C API.
#[repr(C)]
struct EdgeTPUDevice {
    device_type: i32,
    path: *const c_char,
}

// Raw EdgeTPU delegate type from C API
#[repr(C)]
pub struct EdgeTPUDelegateRaw {
    _private: [u8; 0], // Opaque struct
}

// Define a custom type for the EdgeTPU delegate
pub type EdgeTPUDelegatePtr = *mut EdgeTPUDelegateRaw;

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

// FFI bindings for libedgetpu
#[link(name = "edgetpu")]
extern "C" {
    #[link_name = "edgetpu_list_devices"]
    fn edgetpu_list_devices(num_devices: *mut usize) -> *mut EdgeTPUDevice;

    #[link_name = "edgetpu_free_devices"]
    fn edgetpu_free_devices(devices: *mut EdgeTPUDevice);

    #[link_name = "edgetpu_create_delegate"]
    fn edgetpu_create_delegate(
        device_type: EdgeTPUDeviceType,
        name: *const c_char,
        options: *const EdgeTPUOption,
        num_options: usize,
    ) -> EdgeTPUDelegatePtr;

    #[link_name = "edgetpu_free_delegate"]
    fn edgetpu_free_delegate(delegate: EdgeTPUDelegatePtr);

    #[link_name = "edgetpu_version"]
    fn edgetpu_version() -> *const c_char;
}

// Function pointers for dynamic loading of libedgetpu
#[derive(Clone)]
struct EdgeTPULibrary {
    create_delegate: Option<
        unsafe extern "C" fn(
            EdgeTPUDeviceType,
            *const c_char,
            *const EdgeTPUOption,
            usize,
        ) -> EdgeTPUDelegatePtr,
    >,
    free_delegate: Option<unsafe extern "C" fn(EdgeTPUDelegatePtr)>,
    version: Option<unsafe extern "C" fn() -> *const c_char>,
}

impl EdgeTPULibrary {
    fn new() -> Result<Self, CoralError> {
        // This is a simplified implementation that assumes the library is already loaded
        // In a real implementation, you would use libloading crate to dynamically load the library
        Ok(EdgeTPULibrary {
            create_delegate: Some(edgetpu_create_delegate),
            free_delegate: Some(edgetpu_free_delegate),
            version: Some(edgetpu_version),
        })
    }

    unsafe fn create_delegate(
        &self,
        device_type: EdgeTPUDeviceType,
        name: *const c_char,
        options: *const EdgeTPUOption,
        num_options: usize,
    ) -> Result<EdgeTPUDelegatePtr, CoralError> {
        match self.create_delegate {
            Some(func) => {
                let delegate = func(device_type, name, options, num_options);
                if delegate.is_null() {
                    Err(CoralError::DelegateCreationFailed)
                } else {
                    Ok(delegate)
                }
            }
            None => Err(CoralError::LibraryNotFound),
        }
    }

    unsafe fn free_delegate(&self, delegate: EdgeTPUDelegatePtr) -> Result<(), CoralError> {
        match self.free_delegate {
            Some(func) => {
                func(delegate);
                Ok(())
            }
            None => Err(CoralError::LibraryNotFound),
        }
    }

    unsafe fn get_version(&self) -> Result<String, CoralError> {
        match self.version {
            Some(func) => {
                let c_str = func();
                if c_str.is_null() {
                    return Err(CoralError::LibraryNotFound);
                }
                let c_str = std::ffi::CStr::from_ptr(c_str);
                match c_str.to_str() {
                    Ok(s) => Ok(s.to_string()),
                    Err(_) => Err(CoralError::LibraryNotFound),
                }
            }
            None => Err(CoralError::LibraryNotFound),
        }
    }
}

fn first_edgetpu_device_path() -> Option<CString> {
    let mut num_devices: usize = 0;
    let devices_ptr = unsafe { edgetpu_list_devices(&mut num_devices as *mut usize) };
    if devices_ptr.is_null() || num_devices == 0 {
        return None;
    }

    let selected_path = unsafe {
        let devices = std::slice::from_raw_parts(devices_ptr, num_devices);
        let mut path = None;

        // Prefer an explicit USB device path when available.
        for dev in devices {
            if dev.device_type == EdgeTPUDeviceType::EdgetpuApexUsb as i32 && !dev.path.is_null() {
                let raw = std::ffi::CStr::from_ptr(dev.path).to_bytes();
                if let Ok(device_path) = CString::new(raw) {
                    path = Some(device_path);
                    break;
                }
            }
        }

        if path.is_none() {
            for dev in devices {
                if !dev.path.is_null() {
                    let raw = std::ffi::CStr::from_ptr(dev.path).to_bytes();
                    if let Ok(device_path) = CString::new(raw) {
                        path = Some(device_path);
                        break;
                    }
                }
            }
        }

        edgetpu_free_devices(devices_ptr);
        path
    };

    selected_path
}

// TensorFlow Lite C API types
pub enum TfLiteModel {}
pub enum TfLiteInterpreter {}
pub enum TfLiteInterpreterOptions {}
pub enum TfLiteTensor {}
pub enum TfLiteDelegate {}

// TensorFlow Lite C API functions
extern "C" {
    fn TfLiteModelCreate(model_data: *const u8, model_size: usize) -> *mut TfLiteModel;
    fn TfLiteModelDelete(model: *mut TfLiteModel);
    fn TfLiteInterpreterCreate(
        model: *mut TfLiteModel,
        options: *mut TfLiteInterpreterOptions,
    ) -> *mut TfLiteInterpreter;
    fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);
    fn TfLiteInterpreterOptionsAddDelegate(
        options: *mut TfLiteInterpreterOptions,
        delegate: *mut TfLiteDelegate,
    );
    fn TfLiteInterpreterSetNumThreads(interpreter: *mut TfLiteInterpreter, num_threads: i32)
        -> i32;
    fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterInvoke(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);
    fn TfLiteInterpreterGetInputTensorCount(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetOutputTensorCount(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterGetInputTensor(
        interpreter: *mut TfLiteInterpreter,
        input_index: i32,
    ) -> *mut TfLiteTensor;
    fn TfLiteInterpreterGetOutputTensor(
        interpreter: *mut TfLiteInterpreter,
        output_index: i32,
    ) -> *mut TfLiteTensor;
    fn TfLiteTensorType(tensor: *mut TfLiteTensor) -> i32;
    fn TfLiteTensorNumDims(tensor: *mut TfLiteTensor) -> i32;
    fn TfLiteTensorDim(tensor: *mut TfLiteTensor, dim_index: i32) -> i32;
    fn TfLiteTensorByteSize(tensor: *mut TfLiteTensor) -> usize;
    fn TfLiteTensorData(tensor: *mut TfLiteTensor) -> *mut std::ffi::c_void;
    fn TfLiteTensorName(tensor: *mut TfLiteTensor) -> *const c_char;
    fn TfLiteTensorCopyFromBuffer(
        tensor: *mut TfLiteTensor,
        input_data: *const std::ffi::c_void,
        input_data_size: usize,
    ) -> i32;
    fn TfLiteTensorCopyToBuffer(
        tensor: *mut TfLiteTensor,
        output_data: *mut std::ffi::c_void,
        output_data_size: usize,
    ) -> i32;
}

/// Error types specific to TensorFlow Lite operations
#[derive(Debug)]
pub enum TfLiteError {
    /// Failed to load the model
    ModelLoadFailed,
    /// Failed to create the interpreter
    InterpreterCreationFailed,
    /// Failed to allocate tensors
    TensorAllocationFailed,
    /// Failed to run inference
    InferenceFailed,
    /// Failed to copy data to or from tensors
    TensorCopyFailed,
    /// Input/output tensor count mismatch
    TensorCountMismatch,
    /// Invalid tensor dimensions
    InvalidTensorDimensions,
    /// Invalid tensor type
    InvalidTensorType,
    /// Failed to modify graph with delegate
    DelegateModificationFailed,
    /// Other error with message
    Other(String),
}

impl std::fmt::Display for TfLiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// TensorFlow Lite model wrapper
pub struct TfLiteModelWrapper {
    model: *mut TfLiteModel,
    _backing_data: Vec<u8>,
}

impl TfLiteModelWrapper {
    /// Create a new TensorFlow Lite model from a file
    pub fn new_from_file(model_path: &str) -> Result<Self, TfLiteError> {
        let model_data = std::fs::read(model_path).map_err(|_| TfLiteError::ModelLoadFailed)?;

        let model = unsafe {
            let model_ptr = TfLiteModelCreate(model_data.as_ptr(), model_data.len());
            if model_ptr.is_null() {
                return Err(TfLiteError::ModelLoadFailed);
            }
            model_ptr
        };

        Ok(TfLiteModelWrapper {
            model,
            _backing_data: model_data,
        })
    }

    /// Create a new TensorFlow Lite model from memory
    pub fn new_from_memory(model_data: &[u8]) -> Result<Self, TfLiteError> {
        let owned_model_data = model_data.to_vec();
        let model = unsafe {
            let model_ptr = TfLiteModelCreate(owned_model_data.as_ptr(), owned_model_data.len());
            if model_ptr.is_null() {
                return Err(TfLiteError::ModelLoadFailed);
            }
            model_ptr
        };

        Ok(TfLiteModelWrapper {
            model,
            _backing_data: owned_model_data,
        })
    }

    /// Get the raw model pointer
    pub fn as_ptr(&self) -> *mut TfLiteModel {
        self.model
    }
}

impl Drop for TfLiteModelWrapper {
    fn drop(&mut self) {
        if !self.model.is_null() {
            unsafe {
                TfLiteModelDelete(self.model);
            }
        }
    }
}

/// TensorFlow Lite interpreter with EdgeTPU delegate
pub struct CoralInterpreter {
    interpreter: *mut TfLiteInterpreter,
    _model: TfLiteModelWrapper, // Keep the model alive as long as the interpreter is alive
    _delegate: EdgeTPUDelegate, // Keep the delegate alive as long as the interpreter is alive
}

impl CoralInterpreter {
    fn from_model(
        model: TfLiteModelWrapper,
        delegate: &EdgeTPUDelegate,
    ) -> Result<Self, TfLiteError> {
        let interpreter = unsafe {
            let options = TfLiteInterpreterOptionsCreate();
            if options.is_null() {
                return Err(TfLiteError::InterpreterCreationFailed);
            }

            TfLiteInterpreterOptionsAddDelegate(options, delegate.as_ptr() as *mut TfLiteDelegate);

            let interpreter_ptr = TfLiteInterpreterCreate(model.as_ptr(), options);
            TfLiteInterpreterOptionsDelete(options);
            if interpreter_ptr.is_null() {
                return Err(TfLiteError::InterpreterCreationFailed);
            }

            interpreter_ptr
        };

        // Allocate tensors
        let status = unsafe { TfLiteInterpreterAllocateTensors(interpreter) };
        if status != 0 {
            unsafe { TfLiteInterpreterDelete(interpreter) };
            return Err(TfLiteError::TensorAllocationFailed);
        }

        // Clone the delegate to keep it alive
        let delegate_clone = delegate.clone();

        Ok(CoralInterpreter {
            interpreter,
            _model: model,
            _delegate: delegate_clone,
        })
    }

    /// Create a new TensorFlow Lite interpreter with an EdgeTPU delegate
    pub fn new(model_path: &str, delegate: &EdgeTPUDelegate) -> Result<Self, TfLiteError> {
        let model = TfLiteModelWrapper::new_from_file(model_path)?;
        Self::from_model(model, delegate)
    }

    /// Create a new TensorFlow Lite interpreter from in-memory model bytes.
    pub fn new_from_memory(
        model_data: &[u8],
        delegate: &EdgeTPUDelegate,
    ) -> Result<Self, TfLiteError> {
        let model = TfLiteModelWrapper::new_from_memory(model_data)?;
        Self::from_model(model, delegate)
    }

    /// Set the number of threads to use for inference
    pub fn set_num_threads(&self, num_threads: i32) -> Result<(), TfLiteError> {
        let status = unsafe { TfLiteInterpreterSetNumThreads(self.interpreter, num_threads) };
        if status != 0 {
            return Err(TfLiteError::Other(format!(
                "Failed to set number of threads: {}",
                status
            )));
        }
        Ok(())
    }

    /// Get the number of input tensors
    pub fn input_tensor_count(&self) -> i32 {
        unsafe { TfLiteInterpreterGetInputTensorCount(self.interpreter) }
    }

    /// Get the number of output tensors
    pub fn output_tensor_count(&self) -> i32 {
        unsafe { TfLiteInterpreterGetOutputTensorCount(self.interpreter) }
    }

    /// Get the input tensor byte size
    pub fn input_tensor_byte_size(&self, input_index: i32) -> Result<usize, TfLiteError> {
        let tensor = unsafe { TfLiteInterpreterGetInputTensor(self.interpreter, input_index) };
        if tensor.is_null() {
            return Err(TfLiteError::Other(format!(
                "Input tensor at index {} not found",
                input_index
            )));
        }

        Ok(unsafe { TfLiteTensorByteSize(tensor) })
    }

    /// Get the output tensor byte size
    pub fn output_tensor_byte_size(&self, output_index: i32) -> Result<usize, TfLiteError> {
        let tensor = unsafe { TfLiteInterpreterGetOutputTensor(self.interpreter, output_index) };
        if tensor.is_null() {
            return Err(TfLiteError::Other(format!(
                "Output tensor at index {} not found",
                output_index
            )));
        }

        Ok(unsafe { TfLiteTensorByteSize(tensor) })
    }

    /// Copy data to an input tensor
    pub fn copy_to_input_tensor(
        &self,
        input_index: i32,
        input_data: &[u8],
    ) -> Result<(), TfLiteError> {
        let tensor = unsafe { TfLiteInterpreterGetInputTensor(self.interpreter, input_index) };
        if tensor.is_null() {
            return Err(TfLiteError::Other(format!(
                "Input tensor at index {} not found",
                input_index
            )));
        }

        let tensor_size = unsafe { TfLiteTensorByteSize(tensor) };
        if tensor_size != input_data.len() {
            return Err(TfLiteError::TensorCopyFailed);
        }

        let status = unsafe {
            TfLiteTensorCopyFromBuffer(
                tensor,
                input_data.as_ptr() as *const std::ffi::c_void,
                input_data.len(),
            )
        };

        if status != 0 {
            return Err(TfLiteError::TensorCopyFailed);
        }

        Ok(())
    }

    /// Copy data from an output tensor
    pub fn copy_from_output_tensor(
        &self,
        output_index: i32,
        output_data: &mut [u8],
    ) -> Result<(), TfLiteError> {
        let tensor = unsafe { TfLiteInterpreterGetOutputTensor(self.interpreter, output_index) };
        if tensor.is_null() {
            return Err(TfLiteError::Other(format!(
                "Output tensor at index {} not found",
                output_index
            )));
        }

        let tensor_size = unsafe { TfLiteTensorByteSize(tensor) };
        if tensor_size != output_data.len() {
            return Err(TfLiteError::TensorCopyFailed);
        }

        let status = unsafe {
            TfLiteTensorCopyToBuffer(
                tensor,
                output_data.as_mut_ptr() as *mut std::ffi::c_void,
                output_data.len(),
            )
        };

        if status != 0 {
            return Err(TfLiteError::TensorCopyFailed);
        }

        Ok(())
    }

    /// Get input tensor dimensions
    pub fn input_tensor_dims(&self, input_index: i32) -> Result<Vec<i32>, TfLiteError> {
        let tensor = unsafe { TfLiteInterpreterGetInputTensor(self.interpreter, input_index) };
        if tensor.is_null() {
            return Err(TfLiteError::Other(format!(
                "Input tensor at index {} not found",
                input_index
            )));
        }

        let num_dims = unsafe { TfLiteTensorNumDims(tensor) };
        let mut dims = Vec::with_capacity(num_dims as usize);

        for i in 0..num_dims {
            let dim = unsafe { TfLiteTensorDim(tensor, i) };
            dims.push(dim);
        }

        Ok(dims)
    }

    /// Get output tensor dimensions
    pub fn output_tensor_dims(&self, output_index: i32) -> Result<Vec<i32>, TfLiteError> {
        let tensor = unsafe { TfLiteInterpreterGetOutputTensor(self.interpreter, output_index) };
        if tensor.is_null() {
            return Err(TfLiteError::Other(format!(
                "Output tensor at index {} not found",
                output_index
            )));
        }

        let num_dims = unsafe { TfLiteTensorNumDims(tensor) };
        let mut dims = Vec::with_capacity(num_dims as usize);

        for i in 0..num_dims {
            let dim = unsafe { TfLiteTensorDim(tensor, i) };
            dims.push(dim);
        }

        Ok(dims)
    }

    /// Get input tensor name
    pub fn input_tensor_name(&self, input_index: i32) -> Result<String, TfLiteError> {
        let tensor = unsafe { TfLiteInterpreterGetInputTensor(self.interpreter, input_index) };
        if tensor.is_null() {
            return Err(TfLiteError::Other(format!(
                "Input tensor at index {} not found",
                input_index
            )));
        }

        let name_ptr = unsafe { TfLiteTensorName(tensor) };
        if name_ptr.is_null() {
            return Ok(String::new());
        }

        let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
        Ok(name.to_string_lossy().into_owned())
    }

    /// Get output tensor name
    pub fn output_tensor_name(&self, output_index: i32) -> Result<String, TfLiteError> {
        let tensor = unsafe { TfLiteInterpreterGetOutputTensor(self.interpreter, output_index) };
        if tensor.is_null() {
            return Err(TfLiteError::Other(format!(
                "Output tensor at index {} not found",
                output_index
            )));
        }

        let name_ptr = unsafe { TfLiteTensorName(tensor) };
        if name_ptr.is_null() {
            return Ok(String::new());
        }

        let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
        Ok(name.to_string_lossy().into_owned())
    }

    /// Run inference
    pub fn run(&self) -> Result<(), TfLiteError> {
        let status = unsafe { TfLiteInterpreterInvoke(self.interpreter) };
        if status != 0 {
            return Err(TfLiteError::InferenceFailed);
        }
        Ok(())
    }
}

impl Drop for CoralInterpreter {
    fn drop(&mut self) {
        if !self.interpreter.is_null() {
            unsafe {
                TfLiteInterpreterDelete(self.interpreter);
            }
        }
    }
}

const DWN1_IDENTIFIER: &[u8; 4] = b"DWN1";
const EXECUTABLE_TYPE_PARAMETER_CACHING: i16 = 1;

pub const DENSE_GEMM256_DIM: usize = 256;
pub const DENSE_GEMM256_WEIGHT_COUNT: usize = DENSE_GEMM256_DIM * DENSE_GEMM256_DIM;
pub const DENSE_GEMM256_WEIGHT_BYTES: usize = DENSE_GEMM256_WEIGHT_COUNT;
pub const DENSE_GEMM256_ZERO_POINT: i16 = 128;

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

#[derive(Debug, Clone, Copy)]
struct Region {
    start: usize,
    end: usize,
}

impl Region {
    fn size(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

struct FlatTable<'a> {
    data: &'a [u8],
    table_offset: usize,
    vtable_offset: usize,
    vtable_len: usize,
}

impl FlatTable<'_> {
    fn field_offset(&self, field_id: usize) -> Result<Option<usize>, DenseGemmError> {
        let entry = self
            .vtable_offset
            .checked_add(4)
            .and_then(|v| v.checked_add(field_id.saturating_mul(2)))
            .ok_or_else(|| DenseGemmError::InvalidTemplate("vtable entry overflow".to_string()))?;
        if entry + 2 > self.vtable_offset + self.vtable_len {
            return Ok(None);
        }

        let rel = read_u16(self.data, entry)? as usize;
        if rel == 0 {
            return Ok(None);
        }

        let abs = self
            .table_offset
            .checked_add(rel)
            .ok_or_else(|| DenseGemmError::InvalidTemplate("field offset overflow".to_string()))?;
        if abs > self.data.len() {
            return Err(DenseGemmError::InvalidTemplate(format!(
                "field {} offset {} out of range",
                field_id, abs
            )));
        }
        Ok(Some(abs))
    }
}

#[derive(Debug)]
struct ExecutableView {
    type_value: i16,
    parameter_region: Option<Region>,
}

#[derive(Debug)]
struct PackageView {
    executables: Vec<ExecutableView>,
}

fn invalid_template(message: impl Into<String>) -> DenseGemmError {
    DenseGemmError::InvalidTemplate(message.into())
}

fn checked_slice<'a>(
    data: &'a [u8],
    start: usize,
    len: usize,
    what: &str,
) -> Result<&'a [u8], DenseGemmError> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid_template(format!("{} range overflow", what)))?;
    if end > data.len() {
        return Err(invalid_template(format!(
            "{} out of bounds: start={} len={} data_len={}",
            what,
            start,
            len,
            data.len()
        )));
    }
    Ok(&data[start..end])
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, DenseGemmError> {
    let bytes = checked_slice(data, offset, 2, "u16 read")?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_i16(data: &[u8], offset: usize) -> Result<i16, DenseGemmError> {
    let bytes = checked_slice(data, offset, 2, "i16 read")?;
    Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, DenseGemmError> {
    let bytes = checked_slice(data, offset, 4, "u32 read")?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_i32(data: &[u8], offset: usize) -> Result<i32, DenseGemmError> {
    let bytes = checked_slice(data, offset, 4, "i32 read")?;
    Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn parse_root_table<'a>(
    data: &'a [u8],
    root_offset: usize,
    file_identifier: Option<&[u8; 4]>,
) -> Result<FlatTable<'a>, DenseGemmError> {
    checked_slice(data, root_offset, 4, "root table")?;

    if let Some(id) = file_identifier {
        let got = checked_slice(data, root_offset + 4, 4, "file identifier")?;
        if got != id {
            return Err(invalid_template(format!(
                "identifier mismatch: expected {:?}, got {:?}",
                id, got
            )));
        }
    }

    let table_rel = read_u32(data, root_offset)? as usize;
    let table_offset = root_offset
        .checked_add(table_rel)
        .ok_or_else(|| invalid_template("table pointer overflow"))?;
    checked_slice(data, table_offset, 4, "table pointer")?;

    let vtable_rel = read_i32(data, table_offset)?;
    if vtable_rel == 0 {
        return Err(invalid_template("invalid vtable relative offset 0"));
    }

    let vtable_offset = if vtable_rel > 0 {
        table_offset
            .checked_sub(vtable_rel as usize)
            .ok_or_else(|| invalid_template("vtable underflow"))?
    } else {
        table_offset
            .checked_add((-vtable_rel) as usize)
            .ok_or_else(|| invalid_template("vtable overflow"))?
    };

    let vtable_len = read_u16(data, vtable_offset)? as usize;
    let object_len = read_u16(data, vtable_offset + 2)? as usize;
    if vtable_len < 4 || vtable_len % 2 != 0 {
        return Err(invalid_template(format!(
            "invalid vtable length {}",
            vtable_len
        )));
    }
    if object_len < 4 {
        return Err(invalid_template(format!(
            "invalid object length {}",
            object_len
        )));
    }
    checked_slice(data, vtable_offset, vtable_len, "vtable bounds")?;
    checked_slice(data, table_offset, object_len, "table bounds")?;

    Ok(FlatTable {
        data,
        table_offset,
        vtable_offset,
        vtable_len,
    })
}

fn read_offset_object(
    table: &FlatTable<'_>,
    field_id: usize,
) -> Result<Option<usize>, DenseGemmError> {
    let Some(off) = table.field_offset(field_id)? else {
        return Ok(None);
    };

    let rel = read_u32(table.data, off)? as usize;
    if rel == 0 {
        return Ok(None);
    }

    let target = off
        .checked_add(rel)
        .ok_or_else(|| invalid_template("offset-object overflow"))?;
    checked_slice(table.data, target, 4, "offset-object target")?;
    Ok(Some(target))
}

fn read_vector_region(
    table: &FlatTable<'_>,
    field_id: usize,
) -> Result<Option<Region>, DenseGemmError> {
    let Some(target) = read_offset_object(table, field_id)? else {
        return Ok(None);
    };

    let vlen = read_u32(table.data, target)? as usize;
    let start = target
        .checked_add(4)
        .ok_or_else(|| invalid_template("vector start overflow"))?;
    let end = start
        .checked_add(vlen)
        .ok_or_else(|| invalid_template("vector end overflow"))?;
    checked_slice(table.data, start, vlen, "vector data")?;
    Ok(Some(Region { start, end }))
}

fn read_i16_field(
    table: &FlatTable<'_>,
    field_id: usize,
    default: i16,
) -> Result<i16, DenseGemmError> {
    let Some(off) = table.field_offset(field_id)? else {
        return Ok(default);
    };
    read_i16(table.data, off)
}

fn scan_dwn1_candidates(data: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    if data.len() < 8 {
        return out;
    }

    for idx in 0..=(data.len() - 4) {
        if &data[idx..idx + 4] == DWN1_IDENTIFIER && idx >= 4 {
            let root = idx - 4;
            if !out.contains(&root) {
                out.push(root);
            }
        }
    }
    out
}

fn parse_multi_executable_layout(multi_bytes: &[u8]) -> Result<Vec<Region>, DenseGemmError> {
    let table = parse_root_table(multi_bytes, 0, None)?;
    let Some(vec_target) = read_offset_object(&table, 0)? else {
        return Err(invalid_template(
            "MultiExecutable.serialized_executables missing",
        ));
    };

    let length = read_u32(multi_bytes, vec_target)? as usize;
    let vec_start = vec_target
        .checked_add(4)
        .ok_or_else(|| invalid_template("serialized_executables vector start overflow"))?;
    let vec_bytes = length
        .checked_mul(4)
        .ok_or_else(|| invalid_template("serialized_executables vector length overflow"))?;
    checked_slice(
        multi_bytes,
        vec_start,
        vec_bytes,
        "serialized_executables vector",
    )?;

    let mut regions = Vec::with_capacity(length);
    for i in 0..length {
        let slot = vec_start + i * 4;
        let rel = read_u32(multi_bytes, slot)? as usize;
        let str_off = slot
            .checked_add(rel)
            .ok_or_else(|| invalid_template("serialized executable string offset overflow"))?;
        let str_len = read_u32(multi_bytes, str_off)? as usize;
        let str_start = str_off
            .checked_add(4)
            .ok_or_else(|| invalid_template("serialized executable string start overflow"))?;
        checked_slice(
            multi_bytes,
            str_start,
            str_len,
            "serialized executable string data",
        )?;
        regions.push(Region {
            start: str_start,
            end: str_start + str_len,
        });
    }

    Ok(regions)
}

fn inspect_packages(blob: &[u8]) -> Vec<PackageView> {
    let mut packages = Vec::new();

    for root_offset in scan_dwn1_candidates(blob) {
        let pkg = (|| -> Result<PackageView, DenseGemmError> {
            let package_table = parse_root_table(blob, root_offset, Some(DWN1_IDENTIFIER))?;
            let Some(multi_region) = read_vector_region(&package_table, 1)? else {
                return Err(invalid_template("package missing multi_executable"));
            };
            let multi_bytes = &blob[multi_region.start..multi_region.end];
            let executable_regions = parse_multi_executable_layout(multi_bytes)?;

            let mut executables = Vec::with_capacity(executable_regions.len());
            for executable_region in executable_regions {
                let abs_start = multi_region
                    .start
                    .checked_add(executable_region.start)
                    .ok_or_else(|| invalid_template("executable region start overflow"))?;
                let abs_end = multi_region
                    .start
                    .checked_add(executable_region.end)
                    .ok_or_else(|| invalid_template("executable region end overflow"))?;
                if abs_end > blob.len() || abs_start >= abs_end {
                    return Err(invalid_template("executable region out of bounds"));
                }

                let executable_blob = &blob[abs_start..abs_end];
                let executable_table = parse_root_table(executable_blob, 0, None)?;
                let type_value = read_i16_field(&executable_table, 13, 0)?;
                let parameter_region = read_vector_region(&executable_table, 6)?;
                let parameter_region = parameter_region.map(|region| Region {
                    start: abs_start + region.start,
                    end: abs_start + region.end,
                });

                executables.push(ExecutableView {
                    type_value,
                    parameter_region,
                });
            }

            Ok(PackageView { executables })
        })();

        if let Ok(parsed) = pkg {
            packages.push(parsed);
        }
    }

    packages
}

fn select_dense_parameter_region(packages: &[PackageView]) -> Result<Region, DenseGemmError> {
    let mut first_nonempty = None;
    for package in packages {
        for executable in &package.executables {
            let Some(region) = executable.parameter_region else {
                continue;
            };
            if region.size() == 0 {
                continue;
            }
            if first_nonempty.is_none() {
                first_nonempty = Some(region);
            }
            if executable.type_value == EXECUTABLE_TYPE_PARAMETER_CACHING {
                return Ok(region);
            }
        }
    }
    first_nonempty.ok_or(DenseGemmError::ParameterRegionNotFound)
}

pub fn dense_param_offset(
    input_dim: usize,
    output_dim: usize,
    row: usize,
    col: usize,
) -> Result<usize, DenseGemmError> {
    if input_dim == 0 || output_dim == 0 {
        return Err(DenseGemmError::UnsupportedDimensions {
            input_dim,
            output_dim,
            reason: "dimensions must be non-zero",
        });
    }
    if input_dim % 64 != 0 || output_dim % 64 != 0 {
        return Err(DenseGemmError::UnsupportedDimensions {
            input_dim,
            output_dim,
            reason: "expected multiples of 64 for recovered layout mapping",
        });
    }
    if input_dim % 4 != 0 {
        return Err(DenseGemmError::UnsupportedDimensions {
            input_dim,
            output_dim,
            reason: "input dimension must be a multiple of 4",
        });
    }
    if row >= input_dim || col >= output_dim {
        return Err(invalid_template(format!(
            "row/col out of range: row={} col={} (expected row < {}, col < {})",
            row, col, input_dim, output_dim
        )));
    }

    let row_tile_count = input_dim / 64;
    let base = (col / 64) * row_tile_count * 4096 + (row / 64) * 4096;
    let inner = ((row % 64) / 4) * 256 + (col % 64) * 4 + (row % 4);
    Ok(base + inner)
}

pub fn dense_256_param_offset(row: usize, col: usize) -> Result<usize, DenseGemmError> {
    dense_param_offset(DENSE_GEMM256_DIM, DENSE_GEMM256_DIM, row, col)
}

#[derive(Clone)]
pub struct DenseGemmTemplate {
    model_bytes: Vec<u8>,
    parameter_region: Region,
    input_dim: usize,
    output_dim: usize,
}

impl DenseGemmTemplate {
    pub fn from_file_with_dims(
        model_path: &str,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DenseGemmError> {
        let model_bytes = std::fs::read(model_path)?;
        Self::from_bytes_with_dims(&model_bytes, input_dim, output_dim)
    }

    pub fn from_bytes_with_dims(
        model_bytes: &[u8],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DenseGemmError> {
        // Validate dimensions against recovered layout assumptions.
        let _ = dense_param_offset(input_dim, output_dim, 0, 0)?;

        let packages = inspect_packages(model_bytes);
        if packages.is_empty() {
            return Err(invalid_template("no valid DWN1 package found"));
        }

        let parameter_region = select_dense_parameter_region(&packages)?;
        let expected_size = input_dim
            .checked_mul(output_dim)
            .ok_or_else(|| invalid_template("dimension multiplication overflow"))?;
        let actual_size = parameter_region.size();
        if actual_size != expected_size {
            return Err(DenseGemmError::InvalidParameterRegionSize {
                expected: expected_size,
                actual: actual_size,
            });
        }

        Ok(Self {
            model_bytes: model_bytes.to_vec(),
            parameter_region,
            input_dim,
            output_dim,
        })
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }

    pub fn model_bytes(&self) -> &[u8] {
        &self.model_bytes
    }

    pub fn parameter_region(&self) -> (usize, usize) {
        (self.parameter_region.start, self.parameter_region.end)
    }

    pub fn payload_bytes(&self) -> &[u8] {
        &self.model_bytes[self.parameter_region.start..self.parameter_region.end]
    }

    pub fn payload_byte_from_qi8(weight_q: i8) -> u8 {
        ((weight_q as i16) + DENSE_GEMM256_ZERO_POINT) as u8
    }

    pub fn qi8_from_payload_byte(payload: u8) -> i8 {
        (payload as i16 - DENSE_GEMM256_ZERO_POINT) as i8
    }

    fn payload_bytes_mut(&mut self) -> Result<&mut [u8], DenseGemmError> {
        let end = self.parameter_region.end;
        if end > self.model_bytes.len() || self.parameter_region.start >= end {
            return Err(invalid_template("parameter payload region out of bounds"));
        }
        Ok(&mut self.model_bytes[self.parameter_region.start..end])
    }

    pub fn fill_matrix_qi8(&mut self, value_q: i8) -> Result<(), DenseGemmError> {
        let encoded = Self::payload_byte_from_qi8(value_q);
        self.payload_bytes_mut()?.fill(encoded);
        Ok(())
    }

    pub fn set_weight_qi8(
        &mut self,
        row: usize,
        col: usize,
        weight_q: i8,
    ) -> Result<(), DenseGemmError> {
        let offset = dense_param_offset(self.input_dim, self.output_dim, row, col)?;
        let payload = self.payload_bytes_mut()?;
        payload[offset] = Self::payload_byte_from_qi8(weight_q);
        Ok(())
    }

    pub fn set_weights_from_slice(
        &mut self,
        weights_row_major_q: &[i8],
    ) -> Result<(), DenseGemmError> {
        let expected_size = self
            .input_dim
            .checked_mul(self.output_dim)
            .ok_or_else(|| invalid_template("dimension multiplication overflow"))?;
        if weights_row_major_q.len() != expected_size {
            return Err(DenseGemmError::WeightSizeMismatch {
                expected: expected_size,
                actual: weights_row_major_q.len(),
            });
        }

        self.fill_matrix_qi8(0)?;
        let input_dim = self.input_dim;
        let output_dim = self.output_dim;
        let payload = self.payload_bytes_mut()?;
        for row in 0..input_dim {
            for col in 0..output_dim {
                let source_index = row * output_dim + col;
                let target_index = dense_param_offset(input_dim, output_dim, row, col)?;
                payload[target_index] =
                    Self::payload_byte_from_qi8(weights_row_major_q[source_index]);
            }
        }
        Ok(())
    }

    pub fn set_identity(&mut self, active_q: i8) -> Result<(), DenseGemmError> {
        if self.input_dim != self.output_dim {
            return Err(DenseGemmError::UnsupportedDimensions {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                reason: "identity requires square matrix",
            });
        }

        self.fill_matrix_qi8(0)?;
        for i in 0..self.input_dim {
            self.set_weight_qi8(i, i, active_q)?;
        }
        Ok(())
    }

    pub fn set_shift_plus1(&mut self, active_q: i8) -> Result<(), DenseGemmError> {
        if self.input_dim != self.output_dim {
            return Err(DenseGemmError::UnsupportedDimensions {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                reason: "shift modes require square matrix",
            });
        }

        self.fill_matrix_qi8(0)?;
        for col in 0..self.output_dim {
            let row = (col + 1) % self.input_dim;
            self.set_weight_qi8(row, col, active_q)?;
        }
        Ok(())
    }

    pub fn set_shift_minus1(&mut self, active_q: i8) -> Result<(), DenseGemmError> {
        if self.input_dim != self.output_dim {
            return Err(DenseGemmError::UnsupportedDimensions {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                reason: "shift modes require square matrix",
            });
        }

        self.fill_matrix_qi8(0)?;
        for col in 0..self.output_dim {
            let row = (col + self.input_dim - 1) % self.input_dim;
            self.set_weight_qi8(row, col, active_q)?;
        }
        Ok(())
    }

    pub fn prepare(&self, delegate: &EdgeTPUDelegate) -> Result<PreparedDenseGemm, DenseGemmError> {
        PreparedDenseGemm::new(
            self.model_bytes(),
            delegate,
            self.input_dim,
            self.output_dim,
        )
    }

    pub fn prepare_with_new_delegate(&self) -> Result<PreparedDenseGemm, DenseGemmError> {
        let device = CoralDevice::new()?;
        let delegate = device.create_delegate()?;
        self.prepare(&delegate)
    }

    pub fn execute(
        &self,
        delegate: &EdgeTPUDelegate,
        input: &[i8],
    ) -> Result<Vec<i8>, DenseGemmError> {
        let prepared = self.prepare(delegate)?;
        prepared.execute(input)
    }
}

pub struct PreparedDenseGemm {
    interpreter: CoralInterpreter,
    input_dim: usize,
    output_dim: usize,
}

impl PreparedDenseGemm {
    fn new(
        model_bytes: &[u8],
        delegate: &EdgeTPUDelegate,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DenseGemmError> {
        let interpreter = CoralInterpreter::new_from_memory(model_bytes, delegate)?;
        let input_size = interpreter.input_tensor_byte_size(0)?;
        if input_size != input_dim {
            return Err(DenseGemmError::InputSizeMismatch {
                expected: input_dim,
                actual: input_size,
            });
        }
        let output_size = interpreter.output_tensor_byte_size(0)?;
        if output_size != output_dim {
            return Err(DenseGemmError::OutputSizeMismatch {
                expected: output_dim,
                actual: output_size,
            });
        }
        Ok(Self {
            interpreter,
            input_dim,
            output_dim,
        })
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.input_dim, self.output_dim)
    }

    pub fn execute(&self, input: &[i8]) -> Result<Vec<i8>, DenseGemmError> {
        if input.len() != self.input_dim {
            return Err(DenseGemmError::InputSizeMismatch {
                expected: self.input_dim,
                actual: input.len(),
            });
        }

        let input_bytes: Vec<u8> = input.iter().map(|value| *value as u8).collect();
        self.interpreter.copy_to_input_tensor(0, &input_bytes)?;
        self.interpreter.run()?;

        let mut output_bytes = vec![0u8; self.output_dim];
        self.interpreter
            .copy_from_output_tensor(0, &mut output_bytes)?;
        Ok(output_bytes.into_iter().map(|value| value as i8).collect())
    }
}

#[derive(Clone)]
pub struct DenseGemm256Template {
    model_bytes: Vec<u8>,
    parameter_region: Region,
}

impl DenseGemm256Template {
    pub fn from_file(model_path: &str) -> Result<Self, DenseGemmError> {
        let model_bytes = std::fs::read(model_path)?;
        Self::from_bytes(&model_bytes)
    }

    pub fn from_bytes(model_bytes: &[u8]) -> Result<Self, DenseGemmError> {
        let packages = inspect_packages(model_bytes);
        if packages.is_empty() {
            return Err(invalid_template("no valid DWN1 package found"));
        }

        let parameter_region = select_dense_parameter_region(&packages)?;
        let actual_size = parameter_region.size();
        if actual_size != DENSE_GEMM256_WEIGHT_BYTES {
            return Err(DenseGemmError::InvalidParameterRegionSize {
                expected: DENSE_GEMM256_WEIGHT_BYTES,
                actual: actual_size,
            });
        }

        Ok(Self {
            model_bytes: model_bytes.to_vec(),
            parameter_region,
        })
    }

    pub fn model_bytes(&self) -> &[u8] {
        &self.model_bytes
    }

    pub fn parameter_region(&self) -> (usize, usize) {
        (self.parameter_region.start, self.parameter_region.end)
    }

    pub fn payload_bytes(&self) -> &[u8] {
        &self.model_bytes[self.parameter_region.start..self.parameter_region.end]
    }

    pub fn payload_byte_from_qi8(weight_q: i8) -> u8 {
        ((weight_q as i16) + DENSE_GEMM256_ZERO_POINT) as u8
    }

    pub fn qi8_from_payload_byte(payload: u8) -> i8 {
        (payload as i16 - DENSE_GEMM256_ZERO_POINT) as i8
    }

    fn payload_bytes_mut(&mut self) -> Result<&mut [u8], DenseGemmError> {
        let end = self.parameter_region.end;
        if end > self.model_bytes.len() || self.parameter_region.start >= end {
            return Err(invalid_template("parameter payload region out of bounds"));
        }
        Ok(&mut self.model_bytes[self.parameter_region.start..end])
    }

    pub fn fill_matrix_qi8(&mut self, value_q: i8) -> Result<(), DenseGemmError> {
        let encoded = Self::payload_byte_from_qi8(value_q);
        self.payload_bytes_mut()?.fill(encoded);
        Ok(())
    }

    pub fn set_weight_qi8(
        &mut self,
        row: usize,
        col: usize,
        weight_q: i8,
    ) -> Result<(), DenseGemmError> {
        let offset = dense_256_param_offset(row, col)?;
        let payload = self.payload_bytes_mut()?;
        payload[offset] = Self::payload_byte_from_qi8(weight_q);
        Ok(())
    }

    pub fn set_weights_from_slice(
        &mut self,
        weights_row_major_q: &[i8],
    ) -> Result<(), DenseGemmError> {
        if weights_row_major_q.len() != DENSE_GEMM256_WEIGHT_COUNT {
            return Err(DenseGemmError::WeightSizeMismatch {
                expected: DENSE_GEMM256_WEIGHT_COUNT,
                actual: weights_row_major_q.len(),
            });
        }

        self.fill_matrix_qi8(0)?;
        let payload = self.payload_bytes_mut()?;
        for row in 0..DENSE_GEMM256_DIM {
            for col in 0..DENSE_GEMM256_DIM {
                let source_index = row * DENSE_GEMM256_DIM + col;
                let target_index = dense_256_param_offset(row, col)?;
                payload[target_index] =
                    Self::payload_byte_from_qi8(weights_row_major_q[source_index]);
            }
        }
        Ok(())
    }

    pub fn set_weights(
        &mut self,
        weights_row_major_q: &[i8; DENSE_GEMM256_WEIGHT_COUNT],
    ) -> Result<(), DenseGemmError> {
        self.set_weights_from_slice(weights_row_major_q)
    }

    pub fn set_identity(&mut self, active_q: i8) -> Result<(), DenseGemmError> {
        self.fill_matrix_qi8(0)?;
        for i in 0..DENSE_GEMM256_DIM {
            self.set_weight_qi8(i, i, active_q)?;
        }
        Ok(())
    }

    pub fn set_shift_plus1(&mut self, active_q: i8) -> Result<(), DenseGemmError> {
        self.fill_matrix_qi8(0)?;
        for col in 0..DENSE_GEMM256_DIM {
            let row = (col + 1) % DENSE_GEMM256_DIM;
            self.set_weight_qi8(row, col, active_q)?;
        }
        Ok(())
    }

    pub fn set_shift_minus1(&mut self, active_q: i8) -> Result<(), DenseGemmError> {
        self.fill_matrix_qi8(0)?;
        for col in 0..DENSE_GEMM256_DIM {
            let row = (col + DENSE_GEMM256_DIM - 1) % DENSE_GEMM256_DIM;
            self.set_weight_qi8(row, col, active_q)?;
        }
        Ok(())
    }

    pub fn set_diagonal(&mut self, diagonal_q: &[i8]) -> Result<(), DenseGemmError> {
        if diagonal_q.len() != DENSE_GEMM256_DIM {
            return Err(DenseGemmError::WeightSizeMismatch {
                expected: DENSE_GEMM256_DIM,
                actual: diagonal_q.len(),
            });
        }

        self.fill_matrix_qi8(0)?;
        for (idx, value_q) in diagonal_q.iter().enumerate() {
            self.set_weight_qi8(idx, idx, *value_q)?;
        }
        Ok(())
    }
}

pub struct GemmTemplate256 {
    template: DenseGemm256Template,
}

pub struct PreparedGemm256 {
    interpreter: CoralInterpreter,
}

impl PreparedGemm256 {
    pub fn execute(
        &self,
        input: &[i8; DENSE_GEMM256_DIM],
    ) -> Result<[i8; DENSE_GEMM256_DIM], DenseGemmError> {
        let mut input_bytes = [0u8; DENSE_GEMM256_DIM];
        for (dst, src) in input_bytes.iter_mut().zip(input.iter()) {
            *dst = *src as u8;
        }
        self.interpreter.copy_to_input_tensor(0, &input_bytes)?;
        self.interpreter.run()?;

        let mut output_bytes = [0u8; DENSE_GEMM256_DIM];
        self.interpreter
            .copy_from_output_tensor(0, &mut output_bytes)?;
        let mut output = [0i8; DENSE_GEMM256_DIM];
        for (dst, src) in output.iter_mut().zip(output_bytes.iter()) {
            *dst = *src as i8;
        }
        Ok(output)
    }
}

impl GemmTemplate256 {
    pub fn from_compiled_template_file(model_path: &str) -> Result<Self, DenseGemmError> {
        Ok(Self {
            template: DenseGemm256Template::from_file(model_path)?,
        })
    }

    pub fn from_compiled_template_bytes(model_bytes: &[u8]) -> Result<Self, DenseGemmError> {
        Ok(Self {
            template: DenseGemm256Template::from_bytes(model_bytes)?,
        })
    }

    pub fn template(&self) -> &DenseGemm256Template {
        &self.template
    }

    pub fn template_mut(&mut self) -> &mut DenseGemm256Template {
        &mut self.template
    }

    pub fn set_weights(
        &mut self,
        weights_row_major_q: &[i8; DENSE_GEMM256_WEIGHT_COUNT],
    ) -> Result<(), DenseGemmError> {
        self.template.set_weights(weights_row_major_q)
    }

    pub fn set_weights_from_slice(
        &mut self,
        weights_row_major_q: &[i8],
    ) -> Result<(), DenseGemmError> {
        self.template.set_weights_from_slice(weights_row_major_q)
    }

    pub fn execute(
        &self,
        delegate: &EdgeTPUDelegate,
        input: &[i8; DENSE_GEMM256_DIM],
    ) -> Result<[i8; DENSE_GEMM256_DIM], DenseGemmError> {
        let prepared = self.prepare(delegate)?;
        prepared.execute(input)
    }

    pub fn prepare(&self, delegate: &EdgeTPUDelegate) -> Result<PreparedGemm256, DenseGemmError> {
        let interpreter = CoralInterpreter::new_from_memory(self.template.model_bytes(), delegate)?;
        let input_size = interpreter.input_tensor_byte_size(0)?;
        if input_size != DENSE_GEMM256_DIM {
            return Err(DenseGemmError::InputSizeMismatch {
                expected: DENSE_GEMM256_DIM,
                actual: input_size,
            });
        }
        let output_size = interpreter.output_tensor_byte_size(0)?;
        if output_size != DENSE_GEMM256_DIM {
            return Err(DenseGemmError::OutputSizeMismatch {
                expected: DENSE_GEMM256_DIM,
                actual: output_size,
            });
        }
        Ok(PreparedGemm256 { interpreter })
    }

    pub fn prepare_with_new_delegate(&self) -> Result<PreparedGemm256, DenseGemmError> {
        let device = CoralDevice::new()?;
        let delegate = device.create_delegate()?;
        self.prepare(&delegate)
    }

    pub fn execute_with_new_delegate(
        &self,
        input: &[i8; DENSE_GEMM256_DIM],
    ) -> Result<[i8; DENSE_GEMM256_DIM], DenseGemmError> {
        let device = CoralDevice::new()?;
        let delegate = device.create_delegate()?;
        self.execute(&delegate, input)
    }
}

/// EdgeTPU Delegate for TensorFlow Lite
///
/// This struct encapsulates the EdgeTPU delegate used to offload
/// TensorFlow Lite operations to the EdgeTPU hardware.
pub struct EdgeTPUDelegate {
    inner: Arc<EdgeTPUDelegateInner>,
}

struct EdgeTPUDelegateInner {
    raw: EdgeTPUDelegatePtr,
    library: Option<EdgeTPULibrary>,
}

impl Clone for EdgeTPUDelegate {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl EdgeTPUDelegate {
    /// Create a new EdgeTPU delegate for USB device
    ///
    /// This function creates a new EdgeTPU delegate for the USB device type,
    /// which is the type used by the Coral USB Accelerator.
    pub fn new() -> Result<Self, CoralError> {
        // Check if the device is actually connected
        if !is_device_connected() {
            return Err(CoralError::DeviceNotFound);
        }

        // Load the EdgeTPU library
        let library = EdgeTPULibrary::new()?;
        let device_name = first_edgetpu_device_path();

        unsafe {
            let delegate = library.create_delegate(
                EdgeTPUDeviceType::EdgetpuApexUsb,
                device_name
                    .as_ref()
                    .map_or(ptr::null(), |name| name.as_ptr()),
                ptr::null(),
                0,
            )?;
            Ok(EdgeTPUDelegate {
                inner: Arc::new(EdgeTPUDelegateInner {
                    raw: delegate,
                    library: Some(library),
                }),
            })
        }
    }

    /// Create a new EdgeTPU delegate with custom options
    ///
    /// This function creates a new EdgeTPU delegate with custom options
    /// provided as a string. The options string format depends on the
    /// libedgetpu implementation.
    pub fn with_options(options_str: &str) -> Result<Self, CoralError> {
        // Check if the device is actually connected
        if !is_device_connected() {
            return Err(CoralError::DeviceNotFound);
        }

        // Parse options string as key-value pairs
        let mut options = Vec::new();
        let mut option_cstrings = Vec::new();

        if !options_str.is_empty() && options_str != "{}" {
            // Very simple JSON parsing for demonstration
            // In a real implementation, you would use a proper JSON parser
            let trimmed = options_str
                .trim_start_matches('{')
                .trim_end_matches('}')
                .trim();
            if !trimmed.is_empty() {
                for pair in trimmed.split(',') {
                    let parts: Vec<&str> = pair.split(':').collect();
                    if parts.len() == 2 {
                        let key = parts[0].trim().trim_matches('"');
                        let value = parts[1].trim().trim_matches('"');

                        let key_cstr = match CString::new(key) {
                            Ok(s) => s,
                            Err(_) => return Err(CoralError::DelegateCreationFailed),
                        };
                        let value_cstr = match CString::new(value) {
                            Ok(s) => s,
                            Err(_) => return Err(CoralError::DelegateCreationFailed),
                        };

                        option_cstrings.push((key_cstr, value_cstr));
                    }
                }
            }
        }

        // Create EdgeTPUOption structs from the parsed options
        for (key, value) in &option_cstrings {
            options.push(EdgeTPUOption {
                name: key.as_ptr(),
                value: value.as_ptr(),
            });
        }

        // Load the EdgeTPU library
        let library = EdgeTPULibrary::new()?;
        let device_name = first_edgetpu_device_path();

        unsafe {
            let delegate = library.create_delegate(
                EdgeTPUDeviceType::EdgetpuApexUsb,
                device_name
                    .as_ref()
                    .map_or(ptr::null(), |name| name.as_ptr()),
                if options.is_empty() {
                    ptr::null()
                } else {
                    options.as_ptr()
                },
                options.len(),
            )?;
            Ok(EdgeTPUDelegate {
                inner: Arc::new(EdgeTPUDelegateInner {
                    raw: delegate,
                    library: Some(library),
                }),
            })
        }
    }

    /// Get the raw pointer to the EdgeTPU delegate
    pub fn as_ptr(&self) -> EdgeTPUDelegatePtr {
        self.inner.raw
    }

    /// Check if the delegate is valid
    ///
    /// This function returns true if the delegate is valid and can be used
    /// for inference, false otherwise.
    pub fn is_valid(&self) -> bool {
        !self.inner.raw.is_null()
    }
}

impl Drop for EdgeTPUDelegateInner {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                if let Some(library) = &self.library {
                    let _ = library.free_delegate(self.raw);
                } else {
                    edgetpu_free_delegate(self.raw);
                }
                self.raw = ptr::null_mut();
            }
        }
    }
}

pub struct CoralDevice {
    is_valid: bool,
    name: Option<String>,
    vendor_id: u16,
    product_id: u16,
    // We could store a device handle here in a real implementation
}

impl CoralDevice {
    /// Create a new Coral device using the default device
    pub fn new() -> Result<Self, CoralError> {
        find_coral_devices()?
            .into_iter()
            .next()
            .ok_or(CoralError::DeviceNotFound)
    }

    /// Create a new Coral device with a specific device name
    pub fn with_device_name(device_name: &str) -> Result<Self, CoralError> {
        if device_name.is_empty() {
            return Err(CoralError::InvalidDeviceName);
        }

        let mut device = find_coral_devices()?
            .into_iter()
            .next()
            .ok_or(CoralError::DeviceNotFound)?;
        device.name = Some(device_name.to_string());
        Ok(device)
    }

    /// Create an EdgeTPU delegate for this device
    ///
    /// This function creates an EdgeTPU delegate that can be used with
    /// TensorFlow Lite to accelerate inference on this device.
    pub fn create_delegate(&self) -> Result<EdgeTPUDelegate, CoralError> {
        if !self.is_valid {
            return Err(CoralError::DeviceCreationFailed);
        }

        EdgeTPUDelegate::new()
    }

    /// Create an EdgeTPU delegate with custom options for this device
    ///
    /// This function creates an EdgeTPU delegate with custom options that
    /// can be used with TensorFlow Lite to accelerate inference on this device.
    pub fn create_delegate_with_options(
        &self,
        options: &str,
    ) -> Result<EdgeTPUDelegate, CoralError> {
        if !self.is_valid {
            return Err(CoralError::DeviceCreationFailed);
        }

        EdgeTPUDelegate::with_options(options)
    }

    /// Check if the device is valid
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    /// Get the device name if available
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the vendor ID of the device
    pub fn vendor_id(&self) -> u16 {
        self.vendor_id
    }

    /// Get the product ID of the device
    pub fn product_id(&self) -> u16 {
        self.product_id
    }
}

impl Drop for CoralDevice {
    fn drop(&mut self) {
        // In a real implementation, this would free the device resources
        self.is_valid = false;
    }
}

/// Check if a Coral USB Accelerator is connected to the system
pub fn is_device_connected() -> bool {
    match find_coral_devices() {
        Ok(devices) => !devices.is_empty(),
        Err(_) => false,
    }
}

/// Find all Coral USB devices connected to the system
fn find_coral_devices() -> Result<Vec<CoralDevice>, CoralError> {
    let context = match Context::new() {
        Ok(ctx) => ctx,
        Err(_) => return Err(CoralError::DeviceNotFound),
    };

    let devices = match context.devices() {
        Ok(devs) => devs,
        Err(_) => return Err(CoralError::DeviceNotFound),
    };

    let mut coral_devices = Vec::new();

    for device in devices.iter() {
        let desc = match device.device_descriptor() {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Check for both initial and initialized device IDs
        if (desc.vendor_id() == CORAL_USB_VENDOR_ID && desc.product_id() == CORAL_USB_PRODUCT_ID)
            || (desc.vendor_id() == CORAL_USB_VENDOR_ID_INITIALIZED
                && desc.product_id() == CORAL_USB_PRODUCT_ID_INITIALIZED)
        {
            // Found a Coral USB Accelerator
            let name = get_device_name(&device, &desc);
            coral_devices.push(CoralDevice {
                is_valid: true,
                name,
                vendor_id: desc.vendor_id(),
                product_id: desc.product_id(),
            });
        }
    }

    if coral_devices.is_empty() {
        Err(CoralError::DeviceNotFound)
    } else {
        Ok(coral_devices)
    }
}

/// Get the device name from the device descriptor
fn get_device_name(device: &Device<Context>, desc: &DeviceDescriptor) -> Option<String> {
    let timeout = Duration::from_secs(1); // 1 second timeout for USB operations

    // Try to get manufacturer string
    if let Ok(handle) = device.open() {
        if let Ok(languages) = handle.read_languages(timeout) {
            if !languages.is_empty() {
                if let Ok(manufacturer) =
                    handle.read_manufacturer_string(languages[0], desc, timeout)
                {
                    return Some(manufacturer);
                }
            }
        }
    }

    // If manufacturer string is not available, return None
    None
}

/// List all available Coral USB devices
pub fn list_devices() -> Result<Vec<String>, CoralError> {
    let devices = find_coral_devices()?;

    if devices.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();

    for device in devices {
        let mut device_info = String::new();
        device_info.push_str(&format!("Vendor ID: 0x{:04x}", device.vendor_id()));
        device_info.push_str(&format!(", Product ID: 0x{:04x}", device.product_id()));

        if let Some(name) = &device.name {
            device_info.push_str(&format!(", Name: {}", name));
        }

        // Add information about device state (initial or initialized)
        if device.vendor_id == CORAL_USB_VENDOR_ID && device.product_id == CORAL_USB_PRODUCT_ID {
            device_info.push_str(" (Initial state)");
        } else if device.vendor_id == CORAL_USB_VENDOR_ID_INITIALIZED
            && device.product_id == CORAL_USB_PRODUCT_ID_INITIALIZED
        {
            device_info.push_str(" (Initialized state)");
        }

        result.push(device_info);
    }

    Ok(result)
}

/// Get information about the Coral USB Accelerator
pub fn get_device_info() -> Result<Vec<String>, CoralError> {
    let devices = find_coral_devices()?;
    let mut info = Vec::new();

    for device in devices {
        let mut device_info = String::new();
        device_info.push_str(&format!("Vendor ID: 0x{:04x}", device.vendor_id()));
        device_info.push_str(&format!(", Product ID: 0x{:04x}", device.product_id()));

        if let Some(name) = &device.name {
            device_info.push_str(&format!(", Name: {}", name));
        }

        // Add information about device state (initial or initialized)
        if device.vendor_id == CORAL_USB_VENDOR_ID && device.product_id == CORAL_USB_PRODUCT_ID {
            device_info.push_str(" (Initial state)");
        } else if device.vendor_id == CORAL_USB_VENDOR_ID_INITIALIZED
            && device.product_id == CORAL_USB_PRODUCT_ID_INITIALIZED
        {
            device_info.push_str(" (Initialized state)");
        }

        info.push(device_info);
    }

    Ok(info)
}

/// Get the EdgeTPU library version
pub fn version() -> String {
    unsafe {
        match EdgeTPULibrary::new() {
            Ok(library) => match library.get_version() {
                Ok(version) => version,
                Err(_) => "Unknown".to_string(),
            },
            Err(_) => "Unknown".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_offset_formula_matches_known_points() {
        assert_eq!(dense_256_param_offset(0, 0).unwrap(), 0);
        assert_eq!(dense_256_param_offset(1, 0).unwrap(), 1);
        assert_eq!(dense_256_param_offset(0, 1).unwrap(), 4);
        assert_eq!(dense_256_param_offset(4, 0).unwrap(), 256);
        assert_eq!(dense_256_param_offset(64, 0).unwrap(), 4096);
        assert_eq!(dense_256_param_offset(0, 64).unwrap(), 16384);
        assert_eq!(dense_256_param_offset(255, 255).unwrap(), 65535);
    }

    #[test]
    fn dense_offset_formula_generalizes_with_dimension() {
        assert_eq!(dense_param_offset(512, 512, 0, 64).unwrap(), 32768);
        assert_eq!(dense_param_offset(512, 512, 511, 511).unwrap(), 262143);
        assert_eq!(dense_param_offset(1024, 1024, 0, 64).unwrap(), 65536);
        assert_eq!(dense_param_offset(1024, 1024, 1023, 1023).unwrap(), 1048575);
    }

    #[test]
    fn dense_offset_rejects_unsupported_dimensions() {
        assert!(matches!(
            dense_param_offset(250, 256, 0, 0),
            Err(DenseGemmError::UnsupportedDimensions { .. })
        ));
    }

    #[test]
    fn dense_payload_encoding_round_trips() {
        let values = [-128i8, -127, -1, 0, 1, 63, 127];
        for value in values {
            let encoded = DenseGemm256Template::payload_byte_from_qi8(value);
            let decoded = DenseGemm256Template::qi8_from_payload_byte(encoded);
            assert_eq!(decoded, value);
        }
    }
}
