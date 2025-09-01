# TensorRT Inference Server (Rust + Python)

A high-performance, production-ready inference server built in Rust with Python bindings. Currently supports TensorRT models with plans to expand to additional inference frameworks.

## 🚀 Features

- **High Performance**: Built in Rust for maximum performance and memory safety
- **Dynamic Batching**: Intelligent request batching to maximize GPU utilization
- **Async Support**: Full async/await support in Python clients
- **TensorRT Integration**: Native TensorRT support with CUDA acceleration
- **Unix Socket Communication**: Low-latency IPC using custom binary protocol
- **Python Bindings**: Easy-to-use Python API via PyO3
- **Memory Efficient**: Zero-copy data transfers where possible
- **Extensible Architecture**: Designed to support multiple inference frameworks

## 📋 Requirements

- **System Requirements**:
    - Linux (Ubuntu 18.04+ recommended)
    - NVIDIA GPU with CUDA support
    - Python 3.8+

- **Dependencies**:
    - CUDA Toolkit (11.0+)
    - TensorRT (8.0+)
    - Rust (1.70+)

## 🔧 Installation

### From Source

1. **Clone the repository**:
```bash
git clone git@github.com:CherryLemon/rust_model_inferencer_pyo3.git
cd rust_model_inferencer_pyo3
```

2. **Install Rust dependencies**:
```bash
cargo build --release
```

3. **Build Python module**:
```bash
pip install maturin
maturin develop --release
```

4. **Install Python dependencies**:
```bash
pip install "numpy<2"
```

## 🚀 Quick Start

### 1. Start the Server

```python
import trt_inferencer_pyo3 as trt

# Create and start server
server = trt.InferenceServerPy()
server.start_tensorrt_server(
    engine_path="model.engine",
    max_batch_size=16,
    socket_path="/tmp/inference.sock",
    batch_timeout_ms=10
)

print("Server started!")
```

### 2. Run Inference (Async Client)

```python
import asyncio
import numpy as np
import trt_inferencer_pyo3 as trt

async def run_inference():
    # Create async client
    client = trt.AsyncInferenceClient("/tmp/inference.sock")
    
    # Check connection
    if await client.test_connection():
        print("Connected to server!")
    
    # Prepare input data
    input_data = {
        "input_tensor": np.random.randn(1, 224, 224, 3).astype(np.float32)
    }
    
    # Run inference
    result = await client.infer(input_data)
    
    # Process results
    for name, (data_bytes, shape) in result.items():
        # Convert bytes back to numpy array
        data = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)
        print(f"Output '{name}': shape={shape}, mean={data.mean():.4f}")

# Run async inference
asyncio.run(run_inference())
```

### 3. Run Inference (Sync Client)

```python
import numpy as np
import trt_inferencer_pyo3 as trt

# Create sync client
client = trt.InferenceClient("/tmp/inference.sock")

# Test connection
if client.test_connection():
    print("Connected to server!")

# Prepare input
input_data = {
    "input_tensor": np.random.randn(2, 224, 224, 3).astype(np.float32)
}

# Run inference
result = client.infer(input_data)

# Process results
for name, array in result.items():
    print(f"Output '{name}': shape={array.shape}, dtype={array.dtype}")
```

## 📚 API Reference

### Server API

#### `InferenceServerPy`

- **`start_tensorrt_server(engine_path, max_batch_size, socket_path, batch_timeout_ms)`**
    - Starts TensorRT inference server
    - `engine_path`: Path to TensorRT engine file
    - `max_batch_size`: Maximum batch size for batching
    - `socket_path`: Unix socket path for communication
    - `batch_timeout_ms`: Timeout for batch formation (milliseconds)

- **`stop_server()`**: Stops the running server
- **`is_running()`**: Returns server status

### Client APIs

#### `AsyncInferenceClient`

- **`infer(inputs)`**: Async inference call
    - `inputs`: Dictionary of input tensors (name -> numpy array)
    - Returns: Dictionary of output tensors

- **`test_connection()`**: Async connection test

#### `InferenceClient` (Sync)

- **`infer(inputs)`**: Synchronous inference call
- **`test_connection()`**: Synchronous connection test

## ⚙️ Configuration

### Batch Configuration

```python
# Batch settings affect performance vs latency trade-off
server.start_tensorrt_server(
    engine_path="model.engine",
    max_batch_size=32,        # Larger = better GPU utilization
    socket_path="/tmp/inference.sock",
    batch_timeout_ms=5        # Smaller = lower latency
)
```

### Performance Tuning

- **Batch Size**: Increase for higher throughput, decrease for lower latency
- **Batch Timeout**: Balance between latency and batching efficiency
- **Multiple Clients**: Use connection pooling for high-concurrency scenarios

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Python Client  │───▶│   Unix Socket    │───▶│  Rust Server    │
│   (Async/Sync)  │    │   (IPC Protocol) │    │  (Batch Queue)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │ TensorRT Engine │
                                               │  (GPU Compute)  │
                                               └─────────────────┘
```

### Key Components

1. **Protocol Layer**: Custom binary protocol over Unix sockets
2. **Batch Processor**: Intelligent request batching and scheduling
3. **Engine Interface**: Pluggable inference engine architecture
4. **Memory Management**: Efficient GPU memory allocation and reuse

## 🔮 Roadmap

- [ ] **ONNX Runtime Support**: Add ONNX model support
- [ ] **PyTorch Support**: Direct PyTorch model inference
- [ ] **Model Quantization**: INT8/FP16 optimization support
- [ ] **Distributed Inference**: Multi-GPU and multi-node support
- [ ] **Monitoring**: Built-in metrics and health endpoints
- [ ] **Model Management**: Dynamic model loading/unloading
- [ ] **HTTP API**: REST API alongside Unix socket interface

## 🧪 Testing

```bash
# Run Rust tests
cargo test

# Run Python integration tests
python -m pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Install development dependencies:
```bash
pip install maturin pytest numpy
```

2. Build in development mode:
```bash
maturin develop
```

3. Run tests:
```bash
cargo test && python -m pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyO3](https://github.com/PyO3/pyo3) for Python bindings
- Uses [cudarc](https://github.com/coreylowman/cudarc) for CUDA integration
- Inspired by high-performance serving frameworks like Triton Inference Server

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/CherryLemon/rust_model_inferencer_pyo3/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CherryLemon/rust_model_inferencer_pyo3/discussions)
- **Documentation**: [Wiki](https://github.com/CherryLemon/rust_model_inferencer_pyo3/wiki)

---

**Note**: This project is actively developed and APIs may change. Please pin to specific versions for production use.