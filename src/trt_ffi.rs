use anyhow::{anyhow, bail, Context, Result};
use cudarc::curand::CudaRng;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtrMut};
use log::{debug, error, info, warn};
use ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::ptr::null_mut;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use cudarc::driver::sys::CUdeviceptr;
use thiserror::Error;
use crate::server::{InferenceEngine, InferenceEngineFactory};

type TRTDimType = i64;

// ---------------- FFI Bindings Module (V3) ----------------
pub mod trt_ffi_clib {
    use libc::{c_int, c_void, size_t};
    use std::ffi::c_char;
    use crate::trt_ffi::TRTDimType;

    // Opaque pointers
    #[repr(C)]
    pub struct IRuntime(c_void);
    #[repr(C)]
    pub struct ICudaEngine(c_void);
    #[repr(C)]
    pub struct IExecutionContext(c_void);

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DataType {
        Float = 0,
        Half = 1,
        Int8 = 2,
        Int32 = 3,
        UINT8 = 5,
        FP8 = 6,
        BF16 = 7,
        INT64 = 8,
        INT4 = 9,
        FP4 = 10,
        E8M0 = 11,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TensorIOMode {
        None = 0,
        Input = 1,
        Output = 2,
    }


    #[link(name = "tensorrt_wrapper_v3")]
    unsafe extern "C" {
        // Lifecycle
        pub fn create_runtime() -> *mut IRuntime;
        pub fn destroy_runtime(runtime: *mut IRuntime);
        pub fn deserialize_engine(
            runtime: *mut IRuntime,
            data: *const c_char,
            size: size_t,
        ) -> *mut ICudaEngine;
        pub fn destroy_engine(engine: *mut ICudaEngine);
        pub fn create_execution_context(engine: *mut ICudaEngine) -> *mut IExecutionContext;
        pub fn destroy_context(context: *mut IExecutionContext);

        // V3 Engine APIs
        pub fn get_num_io_tensors(engine: *mut ICudaEngine) -> i32;
        pub fn get_io_tensor_name(engine: *mut ICudaEngine, index: i32) -> *const c_char;
        pub fn get_tensor_shape(
            engine: *mut ICudaEngine,
            name: *const c_char,
            shape: *mut TRTDimType,
            num_dims: *mut i32,
        ) -> bool;
        pub fn get_tensor_dtype(engine: *mut ICudaEngine, name: *const c_char) -> DataType;
        pub fn get_tensor_mode(engine: *mut ICudaEngine, name: *const c_char) -> TensorIOMode;
        pub fn get_tensor_profile_shape(
            engine: *mut ICudaEngine,
            name: *const c_char,
            profile_idx: c_int,
            min: *mut TRTDimType,
            opt: *mut TRTDimType,
            max: *mut TRTDimType,
            num_dims: *mut i32,
        ) -> bool;

        // V3 Context APIs
        pub fn set_input_shape(
            context: *mut IExecutionContext,
            name: *const c_char,
            shape: *const TRTDimType,
            num_dims: c_int,
        ) -> bool;
        pub fn set_tensor_address(
            context: *mut IExecutionContext,
            name: *const c_char,
            data: *mut c_void,
        ) -> bool;
        pub fn execute_async_v3(context: *mut IExecutionContext, stream: *mut c_void) -> bool;
    }
}

// Device buffer wrapper for cudarc
pub struct DeviceBuffer {
    stream: Arc<CudaStream>,
    ptr: CudaSlice<u8>,
    size: usize,
}

impl DeviceBuffer {
    pub fn alloc(stream: Arc<CudaStream>, size: usize) -> Result<Self> {
        let ptr = unsafe { stream.alloc::<u8>(size)? };
        Ok(Self { stream, ptr, size })
    }

    pub fn from_host_slice<T>(stream: Arc<CudaStream>, data: &[T]) -> Result<Self> {
        let size = data.len() * size_of::<T>();
        let mut ptr = unsafe { stream.alloc::<u8>(size)? };
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                size
            )
        };
        stream.memcpy_htod(byte_slice, &mut ptr)?;
        Ok(Self { stream, ptr, size })
    }

    pub fn clone(&self) -> Result<Self>  {
        Ok(Self {
            stream: self.stream.clone(),
            ptr: self.ptr.clone(),
            size: self.size,
        })
    }

    pub fn copy_to_host<T>(&self, host_data: &mut [T]) -> Result<()> {
        let expected_size = host_data.len() * size_of::<T>();
        if expected_size != self.size {
            bail!("Size mismatch: buffer size {} != expected size {}", self.size, expected_size);
        }
        let byte_slice = unsafe {
            std::slice::from_raw_parts_mut(
                host_data.as_mut_ptr() as *mut u8,
                expected_size
            )
        };
        self.stream.memcpy_dtoh(&self.ptr, byte_slice)?;
        Ok(())
    }

    pub fn device_ptr(&mut self) -> *mut c_void {
        let (ptr, _) = self.ptr.device_ptr_mut(&self.stream);
        ptr as *mut c_void
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_bytes(&self) -> usize {
        self.size
    }
}

// ---------------- Safe Wrapper for Engine (V3) ----------------
#[derive(Debug)]
pub struct TrtEngineV3 {
    runtime: *mut trt_ffi_clib::IRuntime,
    engine: *mut trt_ffi_clib::ICudaEngine,
    pub max_batch_size: usize,
    pub tensor_info: Vec<TensorInfo>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    // Map name to index in `tensor_info` for quick lookup
    name_to_idx: HashMap<String, usize>,
}

impl Drop for TrtEngineV3 {
    fn drop(&mut self) {
        unsafe {
            trt_ffi_clib::destroy_engine(self.engine);
            trt_ffi_clib::destroy_runtime(self.runtime);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<TRTDimType>, // Shape WITHOUT batch dimension
    pub dtype: trt_ffi_clib::DataType,
    pub mode: trt_ffi_clib::TensorIOMode,
}

fn get_element_size(dtype: &trt_ffi_clib::DataType) -> usize {
    match dtype {
        trt_ffi_clib::DataType::Float => 4,
        trt_ffi_clib::DataType::Half => 2,
        trt_ffi_clib::DataType::Int8 => 1,
        trt_ffi_clib::DataType::Int32 => 4,
        trt_ffi_clib::DataType::UINT8 => 1,
        trt_ffi_clib::DataType::FP8 => 1,
        trt_ffi_clib::DataType::BF16 => 2,
        trt_ffi_clib::DataType::INT64 => 8,
        trt_ffi_clib::DataType::INT4 => 1, // Assuming packed
        trt_ffi_clib::DataType::FP4 => 1, // Assuming packed
        trt_ffi_clib::DataType::E8M0 => 1, // Assuming packed
    }
}

impl TensorInfo {
    pub fn get_element_size(&self) -> usize {
        get_element_size(&self.dtype)
    }
}

impl TrtEngineV3 {
    pub fn new(engine_path: impl AsRef<Path>, max_batch_size: usize) -> Result<Self> {
        let runtime = unsafe { trt_ffi_clib::create_runtime() };
        if runtime.is_null() {
            bail!("Failed to create TensorRT Runtime");
        }

        let engine_data = std::fs::read(engine_path.as_ref())?;
        let engine = unsafe {
            trt_ffi_clib::deserialize_engine(runtime, engine_data.as_ptr() as *const _, engine_data.len())
        };
        if engine.is_null() {
            unsafe { trt_ffi_clib::destroy_runtime(runtime) };
            bail!("Failed to deserialize TensorRT Engine");
        }

        // --- Determine max batch size from profile ---
        let mut max_bs_from_profile = -1;
        let num_tensors = unsafe { trt_ffi_clib::get_num_io_tensors(engine) };
        for i in 0..num_tensors {
            let tensor_name_c = unsafe { trt_ffi_clib::get_io_tensor_name(engine, i) };
            let _tensor_name_str = unsafe { CStr::from_ptr(tensor_name_c).to_str()? };

            if unsafe { trt_ffi_clib::get_tensor_mode(engine, tensor_name_c) } == trt_ffi_clib::TensorIOMode::Input {
                let mut min_shape = vec![0;8];
                let mut opt_shape = vec![0;8];
                let mut max_shape = vec![0;8];
                let mut num_dims = 0;
                if unsafe {
                    trt_ffi_clib::get_tensor_profile_shape(
                        engine,
                        tensor_name_c,
                        0,
                        min_shape.as_mut_ptr(),
                        opt_shape.as_mut_ptr(),
                        max_shape.as_mut_ptr(),
                        &mut num_dims,
                    )
                } {
                    if num_dims > 0 && max_shape[0] > max_bs_from_profile {
                        max_bs_from_profile = max_shape[0];
                    }
                }
            }
        }

        let determined_max_bs = if max_bs_from_profile < 0 {
            warn!(
                "Could not determine max batch size from profile, using user value: {}",
                max_batch_size
            );
            max_batch_size
        } else if max_batch_size > max_bs_from_profile as usize {
            warn!(
                "User max_batch_size {} > profile max_batch_size {}. Using profile's value.",
                max_batch_size, max_bs_from_profile
            );
            max_bs_from_profile as usize
        } else {
            max_batch_size
        };

        // --- Collect all tensor information ---
        let mut tensor_info = Vec::new();
        let mut input_names = Vec::new();
        let mut output_names = Vec::new();
        let mut name_to_idx = HashMap::new();

        for i in 0..num_tensors {
            let name_c = unsafe { trt_ffi_clib::get_io_tensor_name(engine, i) };
            let name = unsafe { CStr::from_ptr(name_c).to_string_lossy().into_owned() };

            let mut shape_dims = vec![0;8];
            let mut num_dims = 0;
            if !unsafe {
                trt_ffi_clib::get_tensor_shape(engine, name_c, shape_dims.as_mut_ptr(), &mut num_dims)
            } {
                bail!("Failed to get shape for tensor '{}'", name);
            }
            let mut shape: Vec<TRTDimType> = shape_dims.iter().take(num_dims as usize).map(
                |e| {e.clone()}).collect();

            let dtype = unsafe { trt_ffi_clib::get_tensor_dtype(engine, name_c) };
            let mode = unsafe { trt_ffi_clib::get_tensor_mode(engine, name_c) };

            // Remove dynamic batch dim (-1) for our internal representation
            if !shape.is_empty() && shape[0] == -1 {
                shape.remove(0);
            }

            let info = TensorInfo {
                name: name.clone(),
                shape,
                dtype,
                mode,
            };

            if mode == trt_ffi_clib::TensorIOMode::Input {
                input_names.push(name.clone());
            } else if mode == trt_ffi_clib::TensorIOMode::Output {
                output_names.push(name.clone());
            }

            name_to_idx.insert(name.clone(), tensor_info.len());
            tensor_info.push(info);
        }

        Ok(Self {
            runtime,
            engine,
            max_batch_size: determined_max_bs,
            tensor_info,
            input_names,
            output_names,
            name_to_idx,
        })
    }
}

// ---------------- Safe Wrapper for Context (V3) ----------------
pub struct TrtContextV3 {
    engine: Arc<TrtEngineV3>,
    context: *mut trt_ffi_clib::IExecutionContext,
    // Pre-allocated device memory for all tensors
    device_buffers: HashMap<String, DeviceBuffer>,
    stream: Arc<CudaStream>,
    cuda_ctx: Arc<CudaContext>,
}

impl Drop for TrtContextV3 {
    fn drop(&mut self) {
        unsafe { trt_ffi_clib::destroy_context(self.context) };
    }
}

impl TrtContextV3 {
    pub fn new(engine: Arc<TrtEngineV3>, cuda_ctx: Arc<CudaContext>) -> Result<Self> {
        let context = unsafe { trt_ffi_clib::create_execution_context(engine.engine) };
        if context.is_null() {
            bail!("Failed to create execution context");
        }

        // Create stream within the context
        let stream = cuda_ctx.new_stream()?;
        let mut device_buffers = HashMap::new();

        // Pre-allocate memory for max_batch_size
        for info in &engine.tensor_info {
            let element_size = info.get_element_size();
            let single_item_bytes: usize =
                info.shape.iter().map(|&d| d as usize).product::<usize>() * element_size;
            let total_size = engine.max_batch_size * single_item_bytes;
            let buffer = DeviceBuffer::alloc(stream.clone(), total_size)?;
            device_buffers.insert(info.name.clone(), buffer);
        }

        // Synchronize to ensure all allocations are complete
        cuda_ctx.synchronize()?;

        Ok(Self {
            engine,
            context,
            device_buffers,
            stream,
            cuda_ctx,
        })
    }

    // GPU inference function
    pub fn inference_gpu(
        &mut self,
        inputs: &mut HashMap<String, &mut DeviceBuffer>,
    ) -> Result<HashMap<String, DeviceBuffer>> {
        if inputs.len() != self.engine.input_names.len() {
            bail!(
                "Input count mismatch: expected {}, got {}",
                self.engine.input_names.len(),
                inputs.len()
            );
        }

        // Determine batch size from the first input
        let first_input_name = &self.engine.input_names[0];
        let first_input_info = &self.engine.tensor_info[self.engine.name_to_idx[first_input_name]];
        let element_size = first_input_info.get_element_size();
        let single_item_bytes: usize = first_input_info
            .shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            * element_size;
        let batch_size = inputs[first_input_name].size() / single_item_bytes;

        // Allocate fresh output buffers for this specific run
        let mut outputs: HashMap<String, DeviceBuffer> = HashMap::new();
        for name in &self.engine.output_names {
            let info = &self.engine.tensor_info[self.engine.name_to_idx[name]];
            let single_item_bytes: usize =
                info.shape.iter().map(|&d| d as usize).product::<usize>() * info.get_element_size();
            let output_buffer = DeviceBuffer::alloc(self.stream.clone(), batch_size * single_item_bytes)?;
            outputs.insert(name.clone(), output_buffer);
        }

        // Chunked inference loop
        for start_idx in (0..batch_size).step_by(self.engine.max_batch_size) {
            let current_batch_size = (batch_size - start_idx).min(self.engine.max_batch_size);

            // Set dynamic input shapes for this chunk
            for name in &self.engine.input_names {
                let info = &self.engine.tensor_info[self.engine.name_to_idx[name]];
                let mut shape_with_batch = info.shape.clone();
                shape_with_batch.insert(0, current_batch_size as TRTDimType);
                let name_c = CString::new(name.as_str())?;
                if !unsafe {
                    trt_ffi_clib::set_input_shape(
                        self.context,
                        name_c.as_ptr(),
                        shape_with_batch.as_ptr(),
                        shape_with_batch.len() as i32,
                    )
                } {
                    bail!("Failed to set input shape for tensor '{}'", name);
                }
            }

            // Set tensor addresses for all inputs and outputs for this chunk
            for name in &self.engine.input_names {
                let info = &self.engine.tensor_info[self.engine.name_to_idx[name]];
                let single_item_bytes =
                    info.shape.iter().map(|&d| d as usize).product::<usize>() * info.get_element_size();
                let offset = start_idx * single_item_bytes;
                let reference = inputs.get_mut(name).unwrap();

                let device_ptr = reference.device_ptr();
                let ptr = (device_ptr as usize + offset) as *mut c_void;

                let name_c = CString::new(name.as_str())?;
                if !unsafe { trt_ffi_clib::set_tensor_address(self.context, name_c.as_ptr(), ptr) } {
                    bail!("Failed to set address for input tensor '{}'", name);
                }
            }

            for name in &self.engine.output_names {
                let info = &self.engine.tensor_info[self.engine.name_to_idx[name]];
                let single_item_bytes =
                    info.shape.iter().map(|&d| d as usize).product::<usize>() * info.get_element_size();
                let offset = start_idx * single_item_bytes;

                let device_ptr = outputs.get_mut(name).unwrap().device_ptr();
                let ptr = (device_ptr as usize + offset) as *mut c_void;

                let name_c = CString::new(name.as_str())?;
                if !unsafe { trt_ffi_clib::set_tensor_address(self.context, name_c.as_ptr(), ptr) } {
                    bail!("Failed to set address for output tensor '{}'", name);
                }
            }

            // Execute V3
            let stream_handle = self.stream.cu_stream();
            if !unsafe { trt_ffi_clib::execute_async_v3(self.context, stream_handle as *mut c_void) } {
                bail!("TensorRT V3 execution failed");
            }
        }

        self.cuda_ctx.synchronize()?;
        Ok(outputs)
    }
}

// ---------------- High-level Inferencer (V3) ----------------
pub struct TrtInferencerV3 {
    context: TrtContextV3,
    default_stream: Arc<CudaStream>,
}

impl TrtInferencerV3 {
    pub fn new(engine_path: &String, max_batch_size: usize) -> Result<Self> {
        let cuda_ctx = CudaContext::new(0)?; // GPU 0
        let default_stream = cuda_ctx.new_stream()?;
        let engine = Arc::new(TrtEngineV3::new(engine_path, max_batch_size)?);
        let mut context = TrtContextV3::new(engine, cuda_ctx)?;

        info!("Ramping up engine '{}'...", engine_path);
        Self::ramp_up(&mut context).unwrap_or_else(|e| error!("Ramp-up failed: {:?}", e));

        Ok(Self { context, default_stream })
    }

    fn ramp_up(context: &mut TrtContextV3) -> Result<()> {
        let mut dummy_inputs = Vec::new();
        for name in context.engine.input_names.iter() {
            dummy_inputs.push((name.clone(), context.device_buffers.get(name).unwrap().clone()?));
        }
        let mut dummy_inputs: HashMap<String, &mut DeviceBuffer> = dummy_inputs
            .iter_mut()
            .map(|(name, buf)| (name.clone(), buf))
            .collect();
        (&mut *context).inference_gpu(&mut dummy_inputs)?;
        Ok(())
    }

    pub fn schema(&self) -> &Vec<TensorInfo> {
        &self.context.engine.tensor_info
    }
}

impl InferenceEngine for TrtInferencerV3 {
    fn infer(
        &mut self,
        inputs: &HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        // Validate inputs
        for name in self.context.engine.input_names.iter() {
            if !inputs.contains_key(name) {
                bail!("Missing input tensor '{}'", name);
            } else {
                let info = &self.context.engine.tensor_info[self.context.engine.name_to_idx[name]];
                let expected_shape: Vec<usize> = {
                    let mut s = vec![inputs[name].shape()[0]]; // Batch size
                    s.extend(info.shape.iter().map(|&d| d as usize));
                    s
                };
                if inputs[name].shape() != expected_shape.as_slice() {
                    bail!(
                        "Input tensor '{}' shape mismatch: expected {:?}, got {:?}",
                        name,
                        expected_shape,
                        inputs[name].shape()
                    );
                }
            }
        }
        // 1. Copy CPU to GPU
        let mut gpu_inputs = HashMap::new();
        for (name, array) in inputs {
            let data_slice = array
                .as_slice()
                .ok_or_else(|| anyhow!("Input array '{}' is not contiguous", name))?;

            let buffer = DeviceBuffer::from_host_slice(
                self.default_stream.clone(), data_slice)?;
            gpu_inputs.insert(name.clone(), buffer);
        }

        let mut gpu_inputs_ref: HashMap<String, &mut DeviceBuffer> = gpu_inputs
            .iter_mut()
            .map(|(k, v)| (k.clone(), v))
            .collect();

        // 2. Run GPU inference
        let gpu_outputs = self.context.inference_gpu(&mut gpu_inputs_ref)?;

        // 3. Copy GPU to CPU
        let mut cpu_outputs = HashMap::new();
        let batch_size = inputs.values().next().unwrap().shape()[0];
        for (name, gpu_buffer) in gpu_outputs {
            let info = &self.context.engine.tensor_info[self.context.engine.name_to_idx[&name]];

            let mut host_vec = vec![0.0f32; gpu_buffer.size() / size_of::<f32>()];
            gpu_buffer.copy_to_host(&mut host_vec)?;

            let mut final_shape = vec![batch_size];
            final_shape.extend(info.shape.iter().map(|&d| d as usize));

            let array = Array::from_shape_vec(IxDyn(&final_shape), host_vec)
                .with_context(|| format!("Failed to create ndarray for output '{}'", name))?;
            cpu_outputs.insert(name, array);
        }

        Ok(cpu_outputs)
    }
}

pub struct TRTInferencerFactory {
    engine_file_name: String,
    max_batch_size: usize,
}

impl TRTInferencerFactory {
    pub fn new(engine_file_name: String, max_batch_size: usize) -> Self {
        Self { engine_file_name, max_batch_size }
    }
}

impl InferenceEngineFactory<TrtInferencerV3> for TRTInferencerFactory {
    fn create_engine(&self) -> Result<TrtInferencerV3> {
        TrtInferencerV3::new(&self.engine_file_name, self.max_batch_size)
    }
}