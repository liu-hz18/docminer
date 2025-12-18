# 后台启动服务
docker-compose up -d

# 查看启动日志（验证服务是否正常）
docker-compose logs -f llm-api-service

# 停止服务（如需）
docker-compose down

# 停止并删除持久化卷（如需清理数据）
docker-compose down -v

# 构建镜像（--build 强制重新构建）
docker-compose build --no-cache

# 检查容器状态
docker ps | grep llm-api-container

# 验证服务可访问
curl http://localhost:8000/

# 检查 NPU 设备是否挂载成功
docker exec -it llm-api-container ls /dev/davinci*


# 本地 CUDA 环境
```
python 3.13
cuda 13.0
linux x86

sudo apt install -y pkg-config libssl-dev build-essential

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

pip install uv
pip install vllm
uv pip install "sglang" --prerelease=allow
pip install mineru==2.6.6
pip install accelerate doclayout_yolo ultralytics ftfy shapely pyclipper omegaconf
pip install json5 jieba rank_bm25
```

# 硬件环境
Ascend-hdk-910b-npu-driver_25.3.rc1_linux-aarch64.run
Ascend-hdk-910b-npu-firmware_7.8.0.2.212.run

# huawei ascend npu firmware and driver community version download
https://www.hiascend.com/hardware/firmware-drivers/community


尝试了但失败的命令：
./Ascend-hdk-xxx-npu-driver_25.3.rc1_linux-aarch64.run --full --install-for-all --install-path=/usr/local/Ascend-25.3.rc1

# windows 管理员模式下创建软连接
mklink /d "./models" "D:/models"

# 下载 glm 模型
export MODELSCOPE_DOMAIN=www.modelscope.ai
modelscope download --model zai-org/GLM-4.6 --local_dir ./models/glm-4.6
modelscope download --model zai-org/GLM-4-32B-0414 --local_dir ./models/glm-4-32b-0414
modelscope download --model zai-org/GLM-4.5-Air --local_dir ./models/glm-4.5-air
