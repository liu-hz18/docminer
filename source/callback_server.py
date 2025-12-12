# callback_server.py
from fastapi import FastAPI, Request
import logging
import uvicorn

# 初始化回调服务
callback_app = FastAPI(title="本地回调接收服务")

# 日志配置（便于查看回调结果）
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
callback_logger = logging.getLogger("callback_server")


# 回调接收接口（原API的callback_api会调用这个接口）
@callback_app.post("/receive-callback")
async def receive_callback(request: Request):
    """接收原API的回调请求，打印并返回确认"""
    # 获取回调数据（JSON格式）
    callback_data = await request.json()
    # 打印回调结果（核心：验证是否接收到数据）
    callback_logger.info(
        f"接收到回调结果：\n{callback_data['status']} {callback_data['uuid']} {callback_data['filename']}"
    )
    # 返回响应给原API（表示回调成功）
    return {"code": 200, "message": "回调结果已接收", "received_data": callback_data}


if __name__ == "__main__":
    # 启动回调服务：端口8001（避免和原API的8000端口冲突）
    uvicorn.run(callback_app, host="0.0.0.0", port=8001)
