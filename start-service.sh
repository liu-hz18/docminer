#!/bin/bash
set -e  # 任意命令失败则脚本退出

# 1. 权限初始化（保留原有逻辑）
chmod -R 777 /app/output /app/config /app/source

# 2. 启动8000端口主服务（后台运行，但通过wait监控）
echo "启动主服务（8000端口）..."
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 uvicorn source.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir source --reload-dir config &
MAIN_PID=$!  # 记录主服务进程ID

# 3. 启动8001端口回调服务（后台运行，监控进程ID）
echo "启动回调服务（8001端口）..."
uvicorn source.callback_server:callback_app --host 0.0.0.0 --port 8001 &
CALLBACK_PID=$!  # 记录回调服务进程ID

# 4. 等待所有进程结束（保证容器不退出）
echo "两个服务已启动，PID：主服务=$MAIN_PID | 回调服务=$CALLBACK_PID"
wait $MAIN_PID $CALLBACK_PID
