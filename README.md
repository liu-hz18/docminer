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