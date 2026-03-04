use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

pub enum TfLiteModel {}
pub enum TfLiteInterpreter {}
pub enum TfLiteInterpreterOptions {}
pub enum TfLiteTensor {}
pub enum TfLiteDelegate {}

#[link(name = "tensorflowlite_c")]
unsafe extern "C" {
    fn TfLiteModelCreateFromFile(model_path: *const c_char) -> *mut TfLiteModel;
    fn TfLiteModelDelete(model: *mut TfLiteModel);

    fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);
    fn TfLiteInterpreterOptionsSetNumThreads(
        options: *mut TfLiteInterpreterOptions,
        num_threads: i32,
    );
    fn TfLiteInterpreterOptionsAddDelegate(
        options: *mut TfLiteInterpreterOptions,
        delegate: *mut TfLiteDelegate,
    );

    fn TfLiteInterpreterCreate(
        model: *mut TfLiteModel,
        options: *mut TfLiteInterpreterOptions,
    ) -> *mut TfLiteInterpreter;
    fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);
    fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> i32;
    fn TfLiteInterpreterInvoke(interpreter: *mut TfLiteInterpreter) -> i32;

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

    fn TfLiteTensorName(tensor: *mut TfLiteTensor) -> *const c_char;
    fn TfLiteTensorType(tensor: *mut TfLiteTensor) -> i32;
    fn TfLiteTensorNumDims(tensor: *mut TfLiteTensor) -> i32;
    fn TfLiteTensorDim(tensor: *mut TfLiteTensor, dim_index: i32) -> i32;
    fn TfLiteTensorByteSize(tensor: *mut TfLiteTensor) -> usize;

    fn TfLiteTensorCopyFromBuffer(
        tensor: *mut TfLiteTensor,
        input_data: *const c_void,
        input_data_size: usize,
    ) -> i32;
    fn TfLiteTensorCopyToBuffer(
        tensor: *mut TfLiteTensor,
        output_data: *mut c_void,
        output_data_size: usize,
    ) -> i32;
}

pub struct Model {
    ptr: *mut TfLiteModel,
}

impl Model {
    pub fn from_file(model_path: &str) -> Result<Self, String> {
        let c_model_path = CString::new(model_path)
            .map_err(|_| format!("model path contains embedded NUL byte: {}", model_path))?;
        let ptr = unsafe { TfLiteModelCreateFromFile(c_model_path.as_ptr()) };
        if ptr.is_null() {
            return Err("TfLiteModelCreateFromFile failed".to_string());
        }
        Ok(Self { ptr })
    }

    fn as_mut_ptr(&self) -> *mut TfLiteModel {
        self.ptr
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { TfLiteModelDelete(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

pub struct InterpreterOptions {
    ptr: *mut TfLiteInterpreterOptions,
}

impl InterpreterOptions {
    pub fn new() -> Result<Self, String> {
        let ptr = unsafe { TfLiteInterpreterOptionsCreate() };
        if ptr.is_null() {
            return Err("TfLiteInterpreterOptionsCreate failed".to_string());
        }
        Ok(Self { ptr })
    }

    pub fn set_num_threads(&self, num_threads: i32) {
        unsafe { TfLiteInterpreterOptionsSetNumThreads(self.ptr, num_threads) };
    }

    pub fn add_delegate(&self, delegate: *mut TfLiteDelegate) {
        if !delegate.is_null() {
            unsafe { TfLiteInterpreterOptionsAddDelegate(self.ptr, delegate) };
        }
    }

    fn as_mut_ptr(&self) -> *mut TfLiteInterpreterOptions {
        self.ptr
    }
}

impl Drop for InterpreterOptions {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { TfLiteInterpreterOptionsDelete(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

pub struct Interpreter {
    ptr: *mut TfLiteInterpreter,
}

impl Interpreter {
    pub fn new(model: &Model, options: &InterpreterOptions) -> Result<Self, String> {
        let ptr = unsafe { TfLiteInterpreterCreate(model.as_mut_ptr(), options.as_mut_ptr()) };
        if ptr.is_null() {
            return Err("TfLiteInterpreterCreate failed".to_string());
        }
        Ok(Self { ptr })
    }

    pub fn allocate_tensors(&self) -> Result<(), i32> {
        let status = unsafe { TfLiteInterpreterAllocateTensors(self.ptr) };
        if status != 0 {
            return Err(status);
        }
        Ok(())
    }

    pub fn invoke(&self) -> Result<(), i32> {
        let status = unsafe { TfLiteInterpreterInvoke(self.ptr) };
        if status != 0 {
            return Err(status);
        }
        Ok(())
    }

    pub fn input_tensor_count(&self) -> i32 {
        unsafe { TfLiteInterpreterGetInputTensorCount(self.ptr) }
    }

    pub fn output_tensor_count(&self) -> i32 {
        unsafe { TfLiteInterpreterGetOutputTensorCount(self.ptr) }
    }

    pub fn input_tensor(&self, input_index: i32) -> Option<Tensor> {
        let ptr = unsafe { TfLiteInterpreterGetInputTensor(self.ptr, input_index) };
        Tensor::from_ptr(ptr)
    }

    pub fn output_tensor(&self, output_index: i32) -> Option<Tensor> {
        let ptr = unsafe { TfLiteInterpreterGetOutputTensor(self.ptr, output_index) };
        Tensor::from_ptr(ptr)
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { TfLiteInterpreterDelete(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

#[derive(Clone, Copy)]
pub struct Tensor {
    ptr: *mut TfLiteTensor,
}

impl Tensor {
    fn from_ptr(ptr: *mut TfLiteTensor) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn name(&self) -> Option<String> {
        let name_ptr = unsafe { TfLiteTensorName(self.ptr) };
        if name_ptr.is_null() {
            return None;
        }
        let name = unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned();
        Some(name)
    }

    pub fn tensor_type(&self) -> i32 {
        unsafe { TfLiteTensorType(self.ptr) }
    }

    pub fn num_dims(&self) -> i32 {
        unsafe { TfLiteTensorNumDims(self.ptr) }
    }

    pub fn dim(&self, dim_index: i32) -> i32 {
        unsafe { TfLiteTensorDim(self.ptr, dim_index) }
    }

    pub fn byte_size(&self) -> usize {
        unsafe { TfLiteTensorByteSize(self.ptr) }
    }

    pub fn copy_from(&self, input_data: &[u8]) -> Result<(), i32> {
        let status = unsafe {
            TfLiteTensorCopyFromBuffer(
                self.ptr,
                input_data.as_ptr() as *const c_void,
                input_data.len(),
            )
        };
        if status != 0 {
            return Err(status);
        }
        Ok(())
    }

    pub fn copy_to(&self, output_data: &mut [u8]) -> Result<(), i32> {
        let status = unsafe {
            TfLiteTensorCopyToBuffer(
                self.ptr,
                output_data.as_mut_ptr() as *mut c_void,
                output_data.len(),
            )
        };
        if status != 0 {
            return Err(status);
        }
        Ok(())
    }
}
