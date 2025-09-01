import numpy as np
import asyncio
from tqdm import trange
from trt_inferencer_pyo3.trt_inferencer_pyo3 import InferenceServerPy as InferenceServer, AsyncInferenceClient
from redeuler.utils.prometheus import timeit
async def test_async():
    # 创建异步客户端
    client = AsyncInferenceClient("/tmp/inference.sock")
    # 测试连接
    if not await client.test_connection():
        print("无法连接到服务器")
        return

    # 准备多个推理请求
    for _ in trange(10000):
        tasks = []
        for i in range(5):
            inputs = {
                "pixel_values": np.random.randn(1, 3, 224, 224).astype(np.float32),
            }
            task = client.infer(inputs)
            tasks.append(task)

        try:
            # 并发执行推理
            results = await timeit()(asyncio.gather)(*tasks)

            # print(f"完成了 {len(results)} 个并发推理请求")
            # print(results[0])
            # for i, outputs in enumerate(results):
            #     print(f"请求 {i+1} 的输出:")
            #     for name, array in outputs.items():
            #         print(f"  {name}: shape={array.shape}")

        except Exception as e:
            print(f"异步推理失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_async())
