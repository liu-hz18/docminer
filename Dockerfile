FROM quay.io/ascend/vllm-ascend:main
# FROM crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:mineru-a2

# 基础配置
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
   PIP_ROOT_USER_ACTION=ignore \
   PATH="$HOME/.cargo/bin:$PATH"

# 非交互式安装 Rust（最小化配置）
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --profile minimal -y && \
    . "$HOME/.cargo/env" && \
    rustc --version

# 统一安装 Python 依赖（修复语法错误+优化源）
RUN python3 -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple && \
    # 安装 uv 包管理器
    python3 -m pip install uv -i https://mirrors.aliyun.com/pypi/simple && \
    # 安装基础依赖
    python3 -m pip install \
        accelerate doclayout_yolo ultralytics ftfy shapely pyclipper omegaconf \
        json5 jieba rank_bm25 vllm vllm-ascend triton-ascend torch==2.8.0 torch-npu==2.8.0 uvicorn \
        -i https://mirrors.aliyun.com/pypi/simple && \
    # 安装 mineru 及指定版本依赖
    python3 -m pip install \
        'mineru[core]>=2.6.5' \
        numpy==1.26.4 \
        opencv-python==4.11.0.86 \
        -i https://mirrors.aliyun.com/pypi/simple && \
    # 清理缓存
    python3 -m pip cache purge && \
    uv cache clean

# 暴露端口
EXPOSE 8000

# 开发环境 Entrypoint（生产环境需移除 --reload/--reload-dir）
# ENTRYPOINT ["uvicorn", "source.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "source"]
