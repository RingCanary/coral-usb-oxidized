use crate::device::is_device_connected;
use crate::error::CoralError;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

#[repr(C)]
pub enum EdgeTPUDeviceType {
    EdgetpuApexPci = 0,
    EdgetpuApexUsb = 1,
}

#[repr(C)]
pub struct EdgeTPUOption {
    name: *const c_char,
    value: *const c_char,
}

#[repr(C)]
struct EdgeTPUDevice {
    device_type: i32,
    path: *const c_char,
}

#[repr(C)]
pub struct EdgeTPUDelegateRaw {
    _private: [u8; 0],
}

pub type EdgeTPUDelegatePtr = *mut EdgeTPUDelegateRaw;

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
        Ok(Self {
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
                let version_ptr = func();
                if version_ptr.is_null() {
                    return Err(CoralError::LibraryNotFound);
                }
                CStr::from_ptr(version_ptr)
                    .to_str()
                    .map(|s| s.to_string())
                    .map_err(|_| CoralError::LibraryNotFound)
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

    unsafe {
        let devices = std::slice::from_raw_parts(devices_ptr, num_devices);
        let mut path = None;

        for dev in devices {
            if dev.device_type == EdgeTPUDeviceType::EdgetpuApexUsb as i32 && !dev.path.is_null() {
                let raw = CStr::from_ptr(dev.path).to_bytes();
                if let Ok(device_path) = CString::new(raw) {
                    path = Some(device_path);
                    break;
                }
            }
        }

        if path.is_none() {
            for dev in devices {
                if !dev.path.is_null() {
                    let raw = CStr::from_ptr(dev.path).to_bytes();
                    if let Ok(device_path) = CString::new(raw) {
                        path = Some(device_path);
                        break;
                    }
                }
            }
        }

        edgetpu_free_devices(devices_ptr);
        path
    }
}

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
    pub fn new() -> Result<Self, CoralError> {
        if !is_device_connected() {
            return Err(CoralError::DeviceNotFound);
        }

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

            Ok(Self {
                inner: Arc::new(EdgeTPUDelegateInner {
                    raw: delegate,
                    library: Some(library),
                }),
            })
        }
    }

    pub fn with_options(options_str: &str) -> Result<Self, CoralError> {
        if !is_device_connected() {
            return Err(CoralError::DeviceNotFound);
        }

        let mut options = Vec::new();
        let mut option_cstrings = Vec::new();

        if !options_str.is_empty() && options_str != "{}" {
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

                        let key_cstr =
                            CString::new(key).map_err(|_| CoralError::DelegateCreationFailed)?;
                        let value_cstr =
                            CString::new(value).map_err(|_| CoralError::DelegateCreationFailed)?;

                        option_cstrings.push((key_cstr, value_cstr));
                    }
                }
            }
        }

        for (key, value) in &option_cstrings {
            options.push(EdgeTPUOption {
                name: key.as_ptr(),
                value: value.as_ptr(),
            });
        }

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

            Ok(Self {
                inner: Arc::new(EdgeTPUDelegateInner {
                    raw: delegate,
                    library: Some(library),
                }),
            })
        }
    }

    pub fn as_ptr(&self) -> EdgeTPUDelegatePtr {
        self.inner.raw
    }

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

pub fn version() -> String {
    unsafe {
        match EdgeTPULibrary::new() {
            Ok(library) => match library.get_version() {
                Ok(ver) => ver,
                Err(_) => "Unknown".to_string(),
            },
            Err(_) => "Unknown".to_string(),
        }
    }
}
