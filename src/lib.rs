mod server;
mod trt_ffi;

use crate::server::InferenceServer;
use crate::trt_ffi::TRTInferencerFactory;
use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::HashMap;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

// 协议常量
const MAGIC_HEADER: u32 = 0x12345678;
const ERROR_MAGIC: u32 = 0x87654321;
const DTYPE_F32: u8 = 1;


#[pyclass]
pub struct InferenceServerPy {
    server_handle: Option<tokio::task::JoinHandle<()>>,
    socket_path: String,
}

#[pymethods]
impl InferenceServerPy {
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

#[pyclass]
pub struct AsyncInferenceClient {
    socket_path: String,
}

#[pymethods]
impl AsyncInferenceClient {
    #[new]
    fn new(socket_path: String) -> Self {
        Self { socket_path }
    }

    /// 异步推理调用
    fn infer<'py>(&self, py: Python<'py>, inputs: &Bound<PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let socket_path = self.socket_path.clone();
        let inputs_dict = self.extract_inputs(inputs)?;

        future_into_py(
            py,
            async move {
                let result = Self::_infer_async(socket_path, inputs_dict).await;

                match result {
                    Ok(result) => {
                        let mut result_dict = HashMap::new();
                        for (name, (data, shape)) in result {
                            Python::with_gil(|py| {
                                // let py_array = PyArray::from_vec(py, data).reshape(shape).unwrap().unbind();
                                // let py_bytes = PyArray::from_vec(py, data).reshape(shape).unwrap().unbind();
                                // result_dict.insert(name, py_array);
                                let py_bytes = PyBytes::new(py, &data).unbind();
                                result_dict.insert(name, (py_bytes, shape));
                            })
                        }
                        Ok(result_dict)
                    },
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
    fn extract_inputs(
        &self,
        inputs: &Bound<PyDict>,
    ) -> PyResult<HashMap<String, (Vec<f32>, Vec<usize>)>> {
        let mut result = HashMap::new();

        for (key, value) in inputs.iter() {
            let name: &str = key.extract()?;
            let array: &Bound<PyArrayDyn<f32>> = value.downcast::<PyArrayDyn<f32>>()?;

            let shape = array.shape().to_vec();
            let data = unsafe { array.as_slice()? }.to_vec();

            result.insert(name.to_string(), (data, shape));
        }

        Ok(result)
    }

    async fn _infer_async(
        socket_path: String,
        inputs: HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<HashMap<String, (Vec<u8>, Vec<usize>)>> {
        // 连接到服务器
        let mut stream = UnixStream::connect(socket_path).await?;

        // 发送请求
        Self::write_request(&mut stream, inputs).await?;

        // 读取响应
        Self::read_response(&mut stream).await
    }

    async fn write_request(
        stream: &mut UnixStream,
        inputs: HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<()> {
        // 写入magic header
        let magic_buf = MAGIC_HEADER.to_le_bytes();
        stream.write_all(&magic_buf).await?;

        // 写入数组数量
        let count_buf = (inputs.len() as u32).to_le_bytes();
        stream.write_all(&count_buf).await?;

        // 写入每个数组
        for (name, (data, shape)) in inputs {
            // 写入数组名长度和名称
            let name_bytes = name.as_bytes();
            let name_len_buf = (name_bytes.len() as u32).to_le_bytes();
            stream.write_all(&name_len_buf).await?;
            stream.write_all(name_bytes).await?;

            // 写入数据类型
            stream.write_all(&[DTYPE_F32]).await?;

            // 写入维度信息
            let ndim_buf = (shape.len() as u32).to_le_bytes();
            stream.write_all(&ndim_buf).await?;

            for &dim in &shape {
                let dim_buf = (dim as u64).to_le_bytes();
                stream.write_all(&dim_buf).await?;
            }

            // 写入数据
            let raw_data = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * size_of::<f32>(),
                )
            };

            let data_len_buf = (raw_data.len() as u64).to_le_bytes();
            stream.write_all(&data_len_buf).await?;
            stream.write_all(raw_data).await?;
        }

        Ok(())
    }

    async fn read_response(
        stream: &mut UnixStream,
    ) -> Result<HashMap<String, (Vec<u8>, Vec<usize>)>> {
        // 读取magic header
        let mut magic_buf = [0u8; 4];
        stream.read_exact(&mut magic_buf).await?;
        let magic = LittleEndian::read_u32(&magic_buf);

        if magic == ERROR_MAGIC {
            // 错误响应
            let mut error_len_buf = [0u8; 4];
            stream.read_exact(&mut error_len_buf).await?;
            let error_len = LittleEndian::read_u32(&error_len_buf) as usize;

            let mut error_buf = vec![0u8; error_len];
            stream.read_exact(&mut error_buf).await?;
            let error_msg = String::from_utf8(error_buf)?;

            return Err(anyhow!("Server error: {}", error_msg));
        } else if magic != MAGIC_HEADER {
            return Err(anyhow!("Invalid response magic header"));
        }

        // 读取数组数量
        let mut count_buf = [0u8; 4];
        stream.read_exact(&mut count_buf).await?;
        let array_count = LittleEndian::read_u32(&count_buf) as usize;

        let mut results = HashMap::new();

        for _ in 0..array_count {
            // 读取数组名长度
            let mut name_len_buf = [0u8; 4];
            stream.read_exact(&mut name_len_buf).await?;
            let name_len = LittleEndian::read_u32(&name_len_buf) as usize;

            // 读取数组名
            let mut name_buf = vec![0u8; name_len];
            stream.read_exact(&mut name_buf).await?;
            let name = String::from_utf8(name_buf)?;

            // 读取数据类型
            let mut dtype_buf = [0u8; 1];
            stream.read_exact(&mut dtype_buf).await?;
            if dtype_buf[0] != DTYPE_F32 {
                return Err(anyhow!("Unsupported dtype in response"));
            }

            // 读取维度数量
            let mut ndim_buf = [0u8; 4];
            stream.read_exact(&mut ndim_buf).await?;
            let ndim = LittleEndian::read_u32(&ndim_buf) as usize;

            // 读取各维度大小
            let mut shape = vec![0usize; ndim];
            for i in 0..ndim {
                let mut dim_buf = [0u8; 8];
                stream.read_exact(&mut dim_buf).await?;
                shape[i] = LittleEndian::read_u64(&dim_buf) as usize;
            }

            // 读取数据长度
            let mut data_len_buf = [0u8; 8];
            stream.read_exact(&mut data_len_buf).await?;
            let data_len = LittleEndian::read_u64(&data_len_buf) as usize;

            // 读取原始数据
            let mut data_buf = vec![0u8; data_len];
            stream.read_exact(&mut data_buf).await?;

            // 转换为f32数组
            // let float_data = unsafe {
            //     std::slice::from_raw_parts(
            //         data_buf.as_ptr() as *const f32,
            //         data_len / std::mem::size_of::<f32>(),
            //     )
            // }
            //     .to_vec();

            results.insert(name, (data_buf, shape));
        }

        Ok(results)
    }
}

// 为了兼容性保留同步客户端
#[pyclass]
pub struct InferenceClient {
    socket_path: String,
}

#[pymethods]
impl InferenceClient {
    #[new]
    fn new(socket_path: String) -> Self {
        Self { socket_path }
    }

    fn infer<'py>(&self, py: Python<'py>, inputs: &Bound<PyDict>) -> PyResult<PyObject> {
        let rt = tokio::runtime::Runtime::new()?;
        let async_client = AsyncInferenceClient::new(self.socket_path.clone());
        let inputs_dict = async_client.extract_inputs(inputs)?;

        match rt.block_on(AsyncInferenceClient::_infer_async(
            self.socket_path.clone(),
            inputs_dict,
        )) {
            Ok(results) => {
                let result_dict = PyDict::new(py);
                for (name, (data, shape)) in results {
                    let py_array = PyArray::from_vec(py, data).reshape(shape)?;
                    result_dict.set_item(name, py_array)?;
                }
                Ok(result_dict.into())
            }
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
#[pymodule(name = "rust_model_inferencer_pyo3")]
fn inference_client(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InferenceClient>()?;
    m.add_class::<AsyncInferenceClient>()?;
    m.add_class::<InferenceServerPy>()?;
    Ok(())
}

// test
#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;
    use pyo3::types::PyDict;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_inference_client() {
        let engine_path = "resnet50.engine".to_string();
        let socket_path = "/tmp/test_infer.sock".to_string();
        let max_batch_size = 16;
        let batch_timeout_ms = 1;

        // 启动服务器
        let mut server = InferenceServerPy::new();
        server
            .start_tensorrt_server(
                engine_path,
                max_batch_size,
                socket_path.clone(),
                batch_timeout_ms,
            )
            .unwrap();

        // 创建客户端
        let client = InferenceClient::new(socket_path.clone());

        // 创建输入数据
        Python::with_gil(|py| {
            let inputs = PyDict::new(py);
            let input_data = vec![0.0f32; 224 * 224 * 3];
            let input_array = PyArray1::from_vec(py, input_data).reshape([1, 224, 224, 3]).unwrap();
            inputs.set_item("pixel_values", input_array).unwrap();

            // 进行推理
            let result = client.infer(py, &inputs).unwrap();
            let result_dict = result.downcast_bound::<PyDict>(py).unwrap();

            // 检查输出
            assert!(result_dict.contains("last_hidden_state").unwrap());
            let value = result_dict.get_item("last_hidden_state")
                .unwrap()
                .unwrap();
            let shape = value.downcast::<PyArrayDyn<f32>>()
                .unwrap().shape();
            println!("last_hidden_state shape: {:?}", shape);
        });

        // 停止服务器
        server.stop_server().unwrap();
    }
}