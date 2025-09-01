use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use crossbeam_channel::{select, unbounded, Receiver, Sender};
use ndarray::{ArrayD, Slice};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{oneshot, mpsc};

// 协议常量
const MAGIC_HEADER: u32 = 0x12345678;
const DTYPE_F32: u8 = 1;

// 批处理配置
#[derive(Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub batch_timeout: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(10),
        }
    }
}

// 请求结构
struct InferenceRequest {
    id: u64,
    input: HashMap<String, ArrayD<f32>>,
    batch_size: usize,
    response_tx: mpsc::UnboundedSender<Result<HashMap<String, ArrayD<f32>>>>,
}

fn calc_batch_size(input: &HashMap<String, ArrayD<f32>>) -> usize {
    input.values().next().map_or(1, |array| array.shape().get(0).cloned().unwrap_or(1))
}

// 分割批处理输出的辅助函数
fn split_array(array: &ArrayD<f32>, start_offset: usize, end_offset: usize) -> Result<ArrayD<f32>> {
    Ok(array.slice_axis(ndarray::Axis(0), Slice::new(start_offset as isize, Some(end_offset as isize), 1)).to_owned())
}

fn split_vector_map(outputs: &HashMap<String, ArrayD<f32>>, start_offset: usize, end_offset: usize) -> Result<HashMap<String, ArrayD<f32>>> {
    let mut split_map = HashMap::new();
    for (key, array) in outputs {
        let split_array = split_array(array, start_offset, end_offset)?;
        split_map.insert(key.clone(), split_array);
    }
    Ok(split_map)
}

// 批处理中合并输入的辅助函数
fn merge_inputs(inputs: &[HashMap<String, ArrayD<f32>>]) -> Result<HashMap<String, ArrayD<f32>>> {
    if inputs.is_empty() {
        return Ok(HashMap::new());
    }

    let mut merged = HashMap::new();
    let first_input = &inputs[0];

    for (key, first_array) in first_input {
        // 收集所有相同key的数组
        let arrays: Vec<&ArrayD<f32>> = inputs
            .iter()
            .map(|input| input.get(key))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow!("Inconsistent input keys across batch"))?;

        // 沿第0维合并数组（假设第0维是batch维度）
        let mut batch_shape = first_array.shape().to_vec();
        batch_shape[0] = arrays.iter().map(|e| -> usize {e.shape()[0]}).sum(); // 设置batch大小

        // 简单的合并实现 - 实际应用中可能需要更复杂的逻辑
        let mut batch_data = Vec::new();
        for array in &arrays {
            batch_data.extend(array.iter().cloned());
        }

        let batch_array = ArrayD::from_shape_vec(batch_shape, batch_data)?;
        merged.insert(key.clone(), batch_array);
    }

    Ok(merged)
}
// 修改后的推理框架接口，支持批处理
pub trait InferenceEngine {
    fn infer(&mut self, input: &HashMap<String, ArrayD<f32>>) -> Result<HashMap<String, ArrayD<f32>>>;

    fn infer_batch(&mut self, inputs: Vec<HashMap<String, ArrayD<f32>>>) -> Result<Vec<HashMap<String, ArrayD<f32>>>> {
        // 合并输入为批处理格式
        if inputs.len() == 1 {
            return Ok(vec![self.infer(&inputs.into_iter().next().unwrap())?]);
        }
        let merged_input = merge_inputs(&inputs)?;

        // 执行批处理推理
        let result = self.infer(&merged_input)?;

        // 分割输出为单个请求格式
        let mut results = vec![];
        let mut start_offset = 0;
        for input in inputs {
            let batch_size = calc_batch_size(&input);
            results.push(split_vector_map(&result, start_offset, start_offset + batch_size)?);
            start_offset += batch_size;
        }

        Ok(results)
    }
}

impl InferenceRequest {
    fn new(input: HashMap<String, ArrayD<f32>>, response_tx: mpsc::UnboundedSender<Result<HashMap<String, ArrayD<f32>>>>) -> Self {
        // 拿到input的第一个value,计算batch_size
        let batch_size = input.values().next().map_or(1, |array| array.shape().get(0).cloned().unwrap_or(1));
        Self {
            id: 0, // 将在处理线程中设置
            input,
            batch_size,
            response_tx,
        }
    }

    fn split(&self, offset: usize) -> Vec<InferenceRequest> {
        let mut requests = vec![];
        requests.push(InferenceRequest{
            id: 0, // 将在处理线程中设置
            input: HashMap::from_iter(self.input.iter().map(|(k, v)| {
                let slice = v.slice_axis(ndarray::Axis(0), Slice::new(0, Some(offset as isize), 1)).to_owned();
                (k.clone(), slice)
            })),
            batch_size: offset,
            response_tx: self.response_tx.clone(),
        });
        requests.push(InferenceRequest{
            id: 0, // 将在处理线程中设置
            input: HashMap::from_iter(self.input.iter().map(|(k, v)| {
                let slice = v.slice_axis(ndarray::Axis(0), Slice::new(offset as isize, None,1)).to_owned();
                (k.clone(), slice)
            })),
            batch_size: offset,
            response_tx: self.response_tx.clone(),
        });
        requests
    }
}

pub trait InferenceEngineFactory<T: InferenceEngine>: Sync + Send {
    fn create_engine(&self) -> Result<T>;
}

// 支持Send + Sync的代理层
pub struct InferenceProxy {
    request_tx: Sender<InferenceRequest>,
}

struct BatchProcessor<T: InferenceEngine> {
    engine: T,
    pending_requests: Vec<InferenceRequest>,
    buffered_requests: Vec<InferenceRequest>,
    request_current_batch_size: usize,
    batch_start_time: Option<Instant>,
    request_counter: u64,
    batch_config: BatchConfig,
}

impl<T: InferenceEngine> BatchProcessor<T> {
    fn new(factory: &dyn InferenceEngineFactory<T>, batch_config: BatchConfig) -> Result<Self> {
        Ok(Self {
            engine: factory.create_engine()?,
            pending_requests: Vec::new(),
            buffered_requests: Vec::new(),
            request_current_batch_size: 0,
            batch_start_time: None,
            request_counter: 0,
            batch_config
        })
    }

    fn execute_requests(
        engine: &mut T,
        requests: Vec<InferenceRequest>,
    ) {
        if requests.is_empty() {
            return;
        }

        println!("Executing batch of {} requests", requests.len());

        // 收集所有输入
        let inputs: Vec<HashMap<String, ArrayD<f32>>> = requests
            .iter()
            .map(|req| req.input.clone())
            .collect();

        // 执行批处理推理
        let batch_result = engine.infer_batch(inputs);

        // 分发结果
        match batch_result {
            Ok(results) => {
                for (request, result) in requests.into_iter().zip(results.into_iter()) {
                    let _ = request.response_tx.send(Ok(result));
                }
            }
            Err(e) => {
                // 批处理失败，所有请求都返回错误
                let error_msg = e.to_string();
                for request in requests {
                    let _ = request.response_tx.send(Err(anyhow!(error_msg.clone())));
                }
            }
        }
    }

    fn execute_batch(&mut self) {
        let batch_requests = std::mem::take(&mut self.pending_requests);
        self.batch_start_time = Some(Instant::now());
        Self::execute_requests(&mut self.engine, batch_requests);

        if !self.buffered_requests.is_empty() {
            self.request_current_batch_size = Self::calc_batch_size(&self.buffered_requests);
            self.pending_requests = std::mem::take(&mut self.buffered_requests);
        } else {
            self.batch_start_time = None;
            self.request_current_batch_size = 0;
        }
    }

    fn should_execute_batch(&self) -> bool {
        if self.pending_requests.is_empty() {
            false
        } else {
            self.request_current_batch_size >= self.batch_config.max_batch_size ||
                self.batch_start_time.map_or(false, |start_time| {
                    start_time.elapsed() >= self.batch_config.batch_timeout
                })
        }
    }

    fn calc_batch_size(requests: &[InferenceRequest]) -> usize {
        requests.iter().map(|r| r.batch_size).sum()
    }

    fn add_request(&mut self, mut request: InferenceRequest) {
        request.id = self.request_counter;

        if self.pending_requests.is_empty() {
            self.batch_start_time = Some(Instant::now());
        } else if (self.request_current_batch_size + request.batch_size) > self.batch_config.max_batch_size {
            let split_requests = request.split(self.batch_config.max_batch_size - self.request_current_batch_size);
            for mut split_request in split_requests {
                split_request.id = self.request_counter;
                self.request_counter += 1;
                if self.request_current_batch_size + split_request.batch_size <= self.batch_config.max_batch_size {
                    self.pending_requests.push(split_request);
                } else {
                    self.buffered_requests.push(split_request);
                }
            }
            self.request_current_batch_size = Self::calc_batch_size(&self.pending_requests);
            return;
        }

        self.request_counter += 1;
        self.request_current_batch_size += request.batch_size;
        self.pending_requests.push(request);
    }
}
impl InferenceProxy {
    pub fn new<T: InferenceEngine + 'static>(factory: Box<dyn InferenceEngineFactory<T>>, batch_config: BatchConfig) -> Result<Self> {
        let (request_tx, request_rx) = unbounded::<InferenceRequest>();
        let (started_tx, started_rx) = oneshot::channel();
        // 启动推理处理线程
        thread::spawn(move || -> Result<()> {
            Ok(Self::inference_worker::<T>(factory.as_ref(), request_rx, batch_config, started_tx)?)
        });

        started_rx.blocking_recv()??;
        Ok(Self { request_tx })
    }

    fn inference_worker<T: InferenceEngine>(
        factory: &dyn InferenceEngineFactory<T>,
        request_rx: Receiver<InferenceRequest>,
        batch_config: BatchConfig,
        started_tx: oneshot::Sender<Result<()>>
    ) -> Result<()> {
        let batch_timeout = batch_config.batch_timeout;
        let mut processor = BatchProcessor::<T>::new(factory, batch_config)?;
        started_tx.send(Ok(())).ok();

        loop {
            if processor.should_execute_batch() {
                processor.execute_batch();
                continue;
            }

            let timeout = if processor.pending_requests.is_empty() {
                None
            } else {
                processor.batch_start_time.map(|start_time| {
                    batch_timeout.saturating_sub(start_time.elapsed())
                })
            };

            let recv_result = if let Some(timeout_duration) = timeout {
                if timeout_duration.is_zero() {
                    processor.execute_batch();
                    continue;
                } else {
                    select! {
                        recv(request_rx) -> result => result,
                        default(timeout_duration) => {
                            processor.execute_batch();
                            continue;
                        }
                    }
                }
            } else {
                request_rx.recv()
            };

            match recv_result {
                Ok(request) => {
                    processor.add_request(request);
                }
                Err(_) => {
                    if !processor.pending_requests.is_empty() {
                        processor.execute_batch();
                    }
                    println!("Inference worker thread exiting");
                    break Err(anyhow!("Inference worker thread exited"));
                }
            }
        }
    }

    pub async fn infer(&self, input: HashMap<String, ArrayD<f32>>) -> Result<HashMap<String, ArrayD<f32>>> {
        let (response_tx, mut response_rx) = mpsc::unbounded_channel();

        let request = InferenceRequest::new(input, response_tx);
        let input_batch_size = request.batch_size;
        // 发送请求到处理线程
        self.request_tx.send(request)
            .map_err(|_| anyhow!("Inference thread has stopped"))?;

        // 等待结果、合并可能的分割请求结果
        let mut results = vec!{};
        let mut batch_count = 0;

        loop {
            let result = response_rx.recv().await;
            match result {
                None => {break Err(anyhow!("Inference thread has stopped"));},
                Some(r) => {
                    let result = r?;
                    batch_count += calc_batch_size(&result);
                    results.push(result);
                    if batch_count == input_batch_size {
                        if results.len() == 1 {
                            return results.into_iter().next().ok_or_else(|| anyhow!("Inference thread has stopped"));
                        }
                        return merge_inputs(&results);
                    }
                }
            }
        }
    }
}

// 实现Send + Sync
unsafe impl Send for InferenceProxy {}
unsafe impl Sync for InferenceProxy {}

pub struct InferenceServer {
    proxy: Arc<InferenceProxy>,
}

impl InferenceServer {
    pub fn new<T: InferenceEngine + 'static>(factory: Box<dyn InferenceEngineFactory<T>>) -> Result<Self> {
        Self::new_with_config::<T>(factory, BatchConfig::default())
    }

    pub fn new_with_config<T: InferenceEngine + 'static>(factory: Box<dyn InferenceEngineFactory<T>>, batch_config: BatchConfig) -> Result<Self> {
        let proxy = Arc::new(InferenceProxy::new::<T>(factory, batch_config)?);
        Ok(Self { proxy })
    }

    pub async fn serve(&self, socket_path: &str, started: oneshot::Sender<Result<()>>) {
        let _ = std::fs::remove_file(socket_path);
        let listener = UnixListener::bind(socket_path);
        let listener = match listener {
            Ok(t) => t,
            Err(e) => {
                started.send(Err(anyhow!("Failed to bind socket: {}", e))).ok();
                return
            }
        };
        println!("Inference server listening on {}", socket_path);
        started.send(Ok(())).ok();
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let proxy = Arc::clone(&self.proxy);
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(stream, &proxy).await {
                            eprintln!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    eprintln!("Failed to accept connection: {}", e);
                }
            }
        }
    }

    async fn handle_connection(mut stream: UnixStream, proxy: &Arc<InferenceProxy>) -> Result<()> {
        loop {
            // 读取magic header
            let mut magic_buf = [0u8; 4];
            match stream.read_exact(&mut magic_buf).await {
                Ok(_) => {},
                Err(_) => break, // 连接关闭
            }

            let magic = LittleEndian::read_u32(&magic_buf);
            if magic != MAGIC_HEADER {
                return Err(anyhow!("Invalid magic header"));
            }

            // 读取请求
            let input_data = Self::read_arrays(&mut stream).await?;

            // 处理推理 - 现在通过代理异步处理，支持批处理
            let response = match proxy.infer(input_data).await {
                Ok(result) => result,
                Err(e) => {
                    // 发送错误响应
                    Self::write_error(&mut stream, &e.to_string()).await?;
                    continue;
                }
            };

            // 发送响应
            Self::write_arrays(&mut stream, response).await?;
        }

        Ok(())
    }

    async fn read_arrays(stream: &mut UnixStream) -> Result<HashMap<String, ArrayD<f32>>> {
        // 读取数组数量
        let mut count_buf = [0u8; 4];
        stream.read_exact(&mut count_buf).await?;
        let array_count = LittleEndian::read_u32(&count_buf) as usize;

        let mut arrays = HashMap::new();

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
                return Err(anyhow!("Unsupported dtype, only f32 is supported"));
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

            // 零拷贝转换：直接从字节构造f32数组
            let float_data = unsafe {
                std::slice::from_raw_parts(
                    data_buf.as_ptr() as *const f32,
                    data_len / size_of::<f32>()
                )
            };

            // 创建ArrayD
            let array = ArrayD::from_shape_vec(shape, float_data.to_vec())?;
            arrays.insert(name, array);
        }

        Ok(arrays)
    }

    async fn write_arrays(stream: &mut UnixStream, arrays: HashMap<String, ArrayD<f32>>) -> Result<()> {
        // 写入magic header (成功标记)
        let magic_buf = MAGIC_HEADER.to_le_bytes();
        stream.write_all(&magic_buf).await?;

        // 写入数组数量
        let count_buf = (arrays.len() as u32).to_le_bytes();
        stream.write_all(&count_buf).await?;

        for (name, array) in arrays {
            // 写入数组名长度和名称
            let name_bytes = name.as_bytes();
            let name_len_buf = (name_bytes.len() as u32).to_le_bytes();
            stream.write_all(&name_len_buf).await?;
            stream.write_all(name_bytes).await?;

            // 写入数据类型
            stream.write_all(&[DTYPE_F32]).await?;

            // 写入维度信息
            let shape = array.shape();
            let ndim_buf = (shape.len() as u32).to_le_bytes();
            stream.write_all(&ndim_buf).await?;

            for &dim in shape {
                let dim_buf = (dim as u64).to_le_bytes();
                stream.write_all(&dim_buf).await?;
            }

            // 获取原始数据并写入
            let raw_data = unsafe {
                std::slice::from_raw_parts(
                    array.as_ptr() as *const u8,
                    array.len() * size_of::<f32>()
                )
            };

            let data_len_buf = (raw_data.len() as u64).to_le_bytes();
            stream.write_all(&data_len_buf).await?;
            stream.write_all(raw_data).await?;
        }

        Ok(())
    }

    async fn write_error(stream: &mut UnixStream, error: &str) -> Result<()> {
        // 写入错误magic header
        let error_magic = 0x87654321u32.to_le_bytes();
        stream.write_all(&error_magic).await?;

        // 写入错误消息
        let error_bytes = error.as_bytes();
        let error_len_buf = (error_bytes.len() as u32).to_le_bytes();
        stream.write_all(&error_len_buf).await?;
        stream.write_all(error_bytes).await?;

        Ok(())
    }
}

// 示例推理引擎 - 支持批处理
pub struct CudaEngine {
    // 假设这里有CUDA上下文和缓冲区
}

impl InferenceEngine for CudaEngine {
    fn infer(&mut self, input: &HashMap<String, ArrayD<f32>>) -> Result<HashMap<String, ArrayD<f32>>> {
        let mut batch_output = HashMap::new();
        for (key, array) in input {
            println!("Processing batched array '{}' with shape {:?}", key, array.shape());
            // 示例：元素乘以2（保持批处理格式）
            let output = array * 2.0;
            batch_output.insert(format!("{}_output", key), output);
        }
        Ok(batch_output)
    }
}

pub struct CudaEngineFactory;

impl InferenceEngineFactory<CudaEngine> for CudaEngineFactory {
    fn create_engine(&self) -> Result<CudaEngine> {
        Ok(CudaEngine {
            // 初始化CUDA上下文和缓冲区
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // 配置批处理参数
    let batch_config = BatchConfig {
        max_batch_size: 4,
        batch_timeout: Duration::from_millis(5),
    };
    let factory = Box::<CudaEngineFactory>::new(CudaEngineFactory{});
    let server = InferenceServer::new_with_config(factory, batch_config)?;
    let (sender, receiver) = oneshot::channel();
    let fut = server.serve("/tmp/inference.sock", sender);
    receiver.await??;
    fut.await;
    Ok(())
}