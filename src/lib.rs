mod server;
mod trt_ffi;
mod tensor;
mod protocol;

use std::borrow::Cow;
use crate::server::InferenceServer;
use crate::trt_ffi::TRTInferencerFactory;
use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use numpy::{dtype, npyffi, PyArray, PyArray1, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::HashMap;
use std::time::Duration;
use ndarray::Data;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use crate::protocol::InferenceProtocol;
use crate::tensor::{data_type_from_u8, DataType, Tensor};


#[pyclass]
#[gen_stub_pyclass]
pub struct TRTInferenceServer {
    server_handle: Option<tokio::task::JoinHandle<()>>,
    socket_path: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl Tensor {
    #[new]
    fn create(data: Vec<u8>, shape: Vec<usize>, dtype: DataType) -> PyResult<Self> {
        match Tensor::new(data, shape, dtype) {
            Ok(tensor) => Ok(tensor),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
        }
    }

    fn get_bytes(&'_ self) -> Cow<'_, [u8]> {
        Cow::from(&*self.data)
    }

    fn get_shape(&'_ self) -> Vec<usize> {
        self.shape.clone()
    }

    fn get_dtype(&'_ self) -> DataType {
        self.dtype.clone()
    }
}


#[pymethods]
#[gen_stub_pymethods]
impl TRTInferenceServer {
    #[new]
    fn new() -> Self {
        Self {
            server_handle: None,
            socket_path: String::new(),
        }
    }

    /// 启动TensorRT推理服务器
    fn start_tensorrt_server(
        &mut self,
        engine_path: String,
        max_batch_size: usize,
        socket_path: String,
        batch_timeout_ms: u64
    ) -> PyResult<()> {
        if self.server_handle.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Server is already running",
            ));
        }
        let factory = Box::new(TRTInferencerFactory::new(engine_path, max_batch_size));
        let socket_path_clone = socket_path.clone();
        self.socket_path = socket_path;
        let batch_config = server::BatchConfig {
            max_batch_size,
            batch_timeout: Duration::from_millis(batch_timeout_ms),
        };

        match InferenceServer::new_with_config(factory, batch_config) {
            Ok(server) => {
                let (sender, receiver) = tokio::sync::oneshot::channel::<Result<()>>();
                let handle = pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
                    let fut = server.serve(&socket_path_clone, sender);
                    fut.await
                });
                receiver.blocking_recv().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to start server: {}",
                        e
                    ))
                })?.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
                self.server_handle = Some(handle);
            }
            Err(e) => {
                eprintln!("Failed to create InferenceServer: {}", e);
            }
        }
        Ok(())
    }

    /// 停止服务器
    fn stop_server(&mut self) -> PyResult<()> {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
            let _ = std::fs::remove_file(&self.socket_path);
        }
        Ok(())
    }

    /// 检查服务器是否在运行
    fn is_running(&self) -> bool {
        self.server_handle
            .as_ref()
            .map_or(false, |h| !h.is_finished())
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct AsyncInferenceClient {
    socket_path: String,
}

fn convert<'py>(py: Python<'py>, inputs: Bound<'py, PyDict>) -> PyResult<HashMap<String, Tensor>> {
    let mut input_data = HashMap::new();
    for (key, value) in inputs.iter() {
        let key_str: String = key.extract()?;
        let value = value.downcast::<PyTuple>()?;
        let item_0 = value.get_item(0)?;
        let item_1 = value.get_item(1)?;
        let data = item_0.downcast::<PyArray1<u8>>()?;
        let tensor = item_1.downcast::<Tensor>()?.get();
        // println!("data length: {}", data.len());
        unsafe {
            input_data.insert(key_str, Tensor::create(
                data.as_slice()?.to_vec(), tensor.shape.clone(), tensor.dtype
            )?);
        }
    }
    Ok(input_data)
}

#[gen_stub_pymethods]
#[pymethods]
impl AsyncInferenceClient {
    #[new]
    fn new(socket_path: String) -> Self {
        Self { socket_path }
    }



    /// 异步推理调用
    fn infer<'py>(&self, py: Python<'py>, inputs: Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let socket_path = self.socket_path.clone();
        let input_data = convert(py, inputs)?;

        future_into_py(
            py,
            async move {
                let result = Self::_infer_async(socket_path, &input_data).await;
                // println!("inference success");
                match result {
                    Ok(result) => Ok(result),
                    Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Async inference failed: {}",
                        e
                    ))),
                }
            }
        )
    }

    /// 测试连接
    fn test_connection<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let socket_path = self.socket_path.clone();

        future_into_py(py, async move {
            match UnixStream::connect(socket_path).await {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        })
    }
}

impl AsyncInferenceClient {
    async fn _infer_async(
        socket_path: String,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // 连接到服务器
        let mut stream = UnixStream::connect(socket_path).await?;

        // 发送请求
        InferenceProtocol::write_arrays(&mut stream, inputs).await?;

        // 读取响应
        // 读取magic header
        let mut magic_buf = [0u8; 4];
        stream.read_exact(&mut magic_buf).await?;
        let magic = LittleEndian::read_u32(&magic_buf);
        if magic != InferenceProtocol::MAGIC_HEADER {
            return Err(anyhow!("Invalid magic header"));
        }
        InferenceProtocol::read_arrays(&mut stream).await
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct InferenceClient {
    socket_path: String,
}
#[gen_stub_pymethods]
#[pymethods]
impl InferenceClient {
    #[new]
    fn new(socket_path: String) -> Self {
        Self { socket_path }
    }

    fn infer<'py>(&self, py: Python<'py>, inputs: Bound<'py, PyDict>) -> PyResult<HashMap<String, Tensor>> {
        let rt = tokio::runtime::Runtime::new()?;
        let async_client = AsyncInferenceClient::new(self.socket_path.clone());
        let input_data = convert(py, inputs)?;
        match rt.block_on(AsyncInferenceClient::_infer_async(
            self.socket_path.clone(),
            &input_data,
        )) {
            Ok(results) => Ok(results),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Inference failed: {}",
                e
            ))),
        }
    }

    fn test_connection(&self) -> PyResult<bool> {
        let rt = tokio::runtime::Runtime::new()?;
        match rt.block_on(UnixStream::connect(&self.socket_path)) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

/// Python模块
#[pymodule(name = "_native")]
fn inference_client(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_class::<InferenceClient>()?;
    m.add_class::<AsyncInferenceClient>()?;
    m.add_class::<TRTInferenceServer>()?;
    m.add_class::<DataType>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
