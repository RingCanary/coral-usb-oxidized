use crate::delegate::EdgeTPUDelegate;
use crate::error::TfLiteError;
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;

pub enum TfLiteModel {}
pub enum TfLiteInterpreter {}
pub enum TfLiteInterpreterOptions {}
pub enum TfLiteTensor {}
pub enum TfLiteDelegate {}

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
    fn TfLiteTensorData(tensor: *mut TfLiteTensor) -> *mut c_void;
    fn TfLiteTensorName(tensor: *mut TfLiteTensor) -> *const c_char;
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

pub struct TfLiteModelWrapper {
    model: *mut TfLiteModel,
    _backing_data: Vec<u8>,
}

impl TfLiteModelWrapper {
    pub fn new_from_file(model_path: &str) -> Result<Self, TfLiteError> {
        let model_data = std::fs::read(model_path).map_err(|_| TfLiteError::ModelLoadFailed)?;

        let model = unsafe {
            let model_ptr = TfLiteModelCreate(model_data.as_ptr(), model_data.len());
            if model_ptr.is_null() {
                return Err(TfLiteError::ModelLoadFailed);
            }
            model_ptr
        };

        Ok(Self {
            model,
            _backing_data: model_data,
        })
    }

    pub fn new_from_memory(model_data: &[u8]) -> Result<Self, TfLiteError> {
        let owned_model_data = model_data.to_vec();
        let model = unsafe {
            let model_ptr = TfLiteModelCreate(owned_model_data.as_ptr(), owned_model_data.len());
            if model_ptr.is_null() {
                return Err(TfLiteError::ModelLoadFailed);
            }
            model_ptr
        };

        Ok(Self {
            model,
            _backing_data: owned_model_data,
        })
    }

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

pub struct CoralInterpreter {
    interpreter: *mut TfLiteInterpreter,
    _model: TfLiteModelWrapper,
    _delegate: EdgeTPUDelegate,
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

        let status = unsafe { TfLiteInterpreterAllocateTensors(interpreter) };
        if status != 0 {
            unsafe { TfLiteInterpreterDelete(interpreter) };
            return Err(TfLiteError::TensorAllocationFailed);
        }

        Ok(Self {
            interpreter,
            _model: model,
            _delegate: delegate.clone(),
        })
    }

    pub fn new(model_path: &str, delegate: &EdgeTPUDelegate) -> Result<Self, TfLiteError> {
        let model = TfLiteModelWrapper::new_from_file(model_path)?;
        Self::from_model(model, delegate)
    }

    pub fn new_from_memory(
        model_data: &[u8],
        delegate: &EdgeTPUDelegate,
    ) -> Result<Self, TfLiteError> {
        let model = TfLiteModelWrapper::new_from_memory(model_data)?;
        Self::from_model(model, delegate)
    }

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

    pub fn input_tensor_count(&self) -> i32 {
        unsafe { TfLiteInterpreterGetInputTensorCount(self.interpreter) }
    }

    pub fn output_tensor_count(&self) -> i32 {
        unsafe { TfLiteInterpreterGetOutputTensorCount(self.interpreter) }
    }

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
                input_data.as_ptr() as *const c_void,
                input_data.len(),
            )
        };

        if status != 0 {
            return Err(TfLiteError::TensorCopyFailed);
        }

        Ok(())
    }

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
                output_data.as_mut_ptr() as *mut c_void,
                output_data.len(),
            )
        };

        if status != 0 {
            return Err(TfLiteError::TensorCopyFailed);
        }

        Ok(())
    }

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
            dims.push(unsafe { TfLiteTensorDim(tensor, i) });
        }

        Ok(dims)
    }

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
            dims.push(unsafe { TfLiteTensorDim(tensor, i) });
        }

        Ok(dims)
    }

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

        Ok(unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned())
    }

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

        Ok(unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned())
    }

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
