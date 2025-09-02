import typing as T
import tempfile
import numpy as np
from ._native import Tensor, InferenceClient, AsyncInferenceClient, TRTInferenceServer, DataType

dtype_np_to_tensor_map = {
    np.float32: DataType.Float,
    np.float16: DataType.Half,
    np.int8: DataType.Int8,
    np.int32: DataType.Int32,
    np.uint8: DataType.UINT8,
    np.int64: DataType.INT64,
}

dtype_tensor_to_np_map = {int(v): k for k, v in dtype_np_to_tensor_map.items()}


def create_tensor_from_numpy(array: np.ndarray) -> (np.ndarray, Tensor):
    if array.dtype.type not in dtype_np_to_tensor_map:
        raise ValueError(f"Unsupported numpy dtype: {array.dtype}")

    data_type = dtype_np_to_tensor_map[array.dtype.type]
    shape = list(array.shape)
    tensor = Tensor(b'', shape, data_type)
    return np.frombuffer(array.data, dtype=np.uint8), tensor


def create_numpy_from_tensor(tensor: Tensor) -> np.ndarray:
    byte_data = tensor.get_bytes()
    dtype = int(tensor.get_dtype())
    shape = tensor.get_shape()
    array = np.frombuffer(byte_data, dtype=dtype_tensor_to_np_map[dtype]).reshape(shape)
    return array


class TRTInferencer:
    def __init__(
            self,
            engine_path: str,
            max_batch_size: int,
            batch_timeout_ms: int
    ):
        self.server = TRTInferenceServer()
        self._socket_file = tempfile.NamedTemporaryFile(prefix="rust_trt_", suffix=".sock", mode='r+b')
        self.server.start_tensorrt_server(
            engine_path=engine_path,
            max_batch_size=max_batch_size,
            socket_path=self._socket_file.name,
            batch_timeout_ms=batch_timeout_ms,
        )
        self._async_client = AsyncInferenceClient(self._socket_file.name)
        self._client = InferenceClient(self._socket_file.name)

    def __del__(self):
        if self.server.is_running():
            self.server.stop_server()

    @property
    def is_running(self) -> bool:
        return self.server.is_running()

    async def predict_async(self, inputs: T.Dict[str, np.ndarray]) -> T.Dict[str, np.ndarray]:
        if not await self._async_client.test_connection():
            raise ConnectionError("无法连接到服务器")

        raw_outputs = await self._async_client.infer({
            k: create_tensor_from_numpy(v)
            for k, v in inputs.items()
        })
        outputs = {name: create_numpy_from_tensor(tensor) for name, tensor in raw_outputs.items()}
        return outputs

    def predict(self, inputs: T.Dict[str, np.ndarray]) -> T.Dict[str, np.ndarray]:
        if not self._client.test_connection():
            raise ConnectionError("无法连接到服务器")

        raw_outputs = self._client.infer({
            k: create_tensor_from_numpy(v)
            for k, v in inputs.items()
        })
        outputs = {name: create_numpy_from_tensor(tensor) for name, tensor in raw_outputs.items()}
        return outputs
