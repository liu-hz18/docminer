import os
import torch
from vllm import LLM, SamplingParams
from .engine.utils import get_device_count


# 1. 基础配置（适配 ARM64/无GPU场景，优先CPU测试）
model_path = "/models/qwen3-32b"  # 模型路径
gpu_available = torch.cuda.is_available()
npu_available = hasattr(torch, "npu") and torch.npu.is_available()

# 2. 采样参数（简单配置）
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    top_p=0.95
)

# 3. 加载模型（关键：适配不同设备）
print(f"开始加载模型：{model_path}")
print(f"CUDA可用：{gpu_available} | NPU可用：{npu_available}")

try:
    # 核心：VLLM 加载模型（CPU 模式需指定 tensor_parallel_size=1 + cpu_offload=True）
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,  # 单卡/CPU
        cpu_offload=False,  # 无GPU/NPU则CPU offload
        disable_log_stats=True,  # 关闭统计日志（简化输出）
        trust_remote_code=True,  # 自定义模型需开启（如Qwen）
        dtype="auto"  # 自动适配精度（ARM64 建议 float16）
    )
    print("✅ 模型加载成功！")

    # 4. 测试文本生成（验证模型可用）
    prompts = ["你好，请介绍一下自己"]
    outputs = llm.generate(prompts, sampling_params)

    # 打印生成结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n📝 输入：{prompt}")
        print(f"🔍 输出：{generated_text}")

except Exception as e:
    print(f"❌ 模型加载/生成失败：{str(e)}")
    # 打印详细报错（便于排查）
    import traceback
    traceback.print_exc()
