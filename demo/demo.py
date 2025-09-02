import numpy as np
import asyncio
from tqdm import trange
from rust_model_inferencer_pyo3.wrapper import TRTInferencer

async def test_async():
    # 启动服务器
    server = TRTInferencer(
        engine_path="/home/lemon/Downloads/resnet50.engine",
        max_batch_size=32,
        batch_timeout_ms=1,
    )

    test_arr = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # 准备多个推理请求
    for _ in trange(10000):
        tasks = []
        for i in range(5):
            inputs = {
                "pixel_values": test_arr,
            }
            task = server.predict_async(inputs)
            tasks.append(task)

        try:
            # 并发执行推理
            results = await asyncio.gather(*tasks)

            print(f"完成了 {len(results)} 个并发推理请求")
            # print(results[0])
            # for i, outputs in enumerate(results):
            #     print(f"请求 {i+1} 的输出:")
            #     for name, (buf, shape) in outputs.items():
            #         array = np.frombuffer(buf, dtype=np.float32).reshape(shape)
            #         print(f"  {name}: shape={array.shape}")

        except Exception as e:
            print(f"异步推理失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_async())
