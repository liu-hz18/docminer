# test_api.py
import requests
import uuid

def generate_random_uuid(with_hyphen: bool = True) -> str:
    """
    生成随机 UUID 字符串
    :param with_hyphen: 是否保留横线（默认保留，符合标准 UUID 格式）
    :return: 随机 UUID 字符串
    """
    uuid_obj = uuid.uuid4()
    if with_hyphen:
        return str(uuid_obj)
    else:
        return uuid_obj.hex

# API地址
api_url = "http://localhost:8000/v1/process-document"

# 构造请求参数（Form + File）
files = {
    "file": open("./documents/test2.pdf", "rb")  # 替换为你的测试文件路径
}
data = {
    "task": "guangfu",
    "uuid": generate_random_uuid(False)
}

# 发送POST请求
response = requests.post(api_url, files=files, data=data)

# 打印API响应（立即返回的结果）
print("📌 API立即返回结果：")
print(response.json())
