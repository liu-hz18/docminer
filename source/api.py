import os
import time
import asyncio
import requests
import threading
from loguru import logger
from copy import deepcopy
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from .engine.pipeline import (
    process_document_query,
    load_sys_configs,
    setup_environment,
    setup_logger,
)
from .engine.llmhub import load_llms
from .engine.utils import (
    sanitize_filename,
    get_string_hash,
    clean_up,
    print_lib_versions,
    print_all_env_vars,
)


# 配置常量
backend = os.environ.get("DOCMINER_BACKEND", "cuda")
AVAILABLE_TASK = ["guangfu", "fengdian"]
if backend == "npu":
    CONFIG_PATH = r"/app/config/config-ascend.json"
else:
    CONFIG_PATH = r"/mnt/c/Users/Administrator/Desktop/docminer/config/config.json"
# 回调接口配置（请替换为实际回调地址）
CALLBACK_URL_PROCESS_DOC = (
    "http://localhost:8001/receive-callback"  # 第一次回调（文件）
)


# 初始化FastAPI应用
app = FastAPI(title="文档处理API", version="1.0")

# 加载系统配置（全局初始化）
CONFIG = load_sys_configs(CONFIG_PATH)
setup_environment(CONFIG)

# -------------------------- 任务队列核心配置 --------------------------
# 任务排队列表（按顺序存储UUID，线程安全锁保护）
TASK_QUEUE_LIST = []
TASK_QUEUE_LOCK = threading.Lock()

# 运行中任务集合（记录当前执行的UUID）
RUNNING_TASKS = set()
RUNNING_TASKS_LOCK = threading.Lock()

# 全局异步互斥锁：控制高资源任务串行执行
TASK_LOCK = asyncio.Lock()

# 任务计数器（辅助监控，可选）
TASK_COUNTER = {"pending": 0, "running": 0}
# ---------------------------------------------------------------------


def callback_api(url: str, data: dict):
    """通用回调函数"""
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        logger.info(
            f"回调成功 | URL: {url} | UUID: {data.get('uuid')} | 响应状态: {response.status_code}"
        )
        return True
    except Exception as e:
        logger.error(f"回调失败 | URL: {url} | 错误: {str(e)}", exc_info=True)
        return False


def process_document(task: str, document_path: str, config: dict, uuid: str):
    """封装文档处理逻辑，返回处理结果"""
    if task not in AVAILABLE_TASK:
        raise ValueError(f"不支持的任务类型 | 支持: {AVAILABLE_TASK}")

    config_copy = deepcopy(config)
    config_copy["task"] = task
    config_copy["uuid"] = uuid
    setup_logger(config_copy)

    api_results, _ = process_document_query(
        test_document=document_path,
        config=config_copy,
    )
    return api_results


async def process_document_background(
    task: str,
    uuid: str,
    file_path: str,
    filename: str,
    callback_url: str,
    config: dict,
):
    """带串行控制的后台任务核心逻辑"""
    callback_data: dict = {}
    task_start_time = time.time()

    try:
        # 1. 任务排队：更新计数器，记录排队日志
        TASK_COUNTER["pending"] += 1
        logger.info(
            f"任务进入排队 | UUID: {uuid} | 当前排队数: {TASK_COUNTER['pending']} | 运行中任务数: {TASK_COUNTER['running']}"
        )

        # 2. 获取全局锁（串行执行核心：未获取到则等待）
        async with TASK_LOCK:
            # 更新计数器：任务开始执行
            TASK_COUNTER["pending"] -= 1
            TASK_COUNTER["running"] += 1

            # 将UUID标记为运行中
            with RUNNING_TASKS_LOCK:
                RUNNING_TASKS.add(uuid)

            logger.info(
                f"任务开始执行 | UUID: {uuid} | 排队耗时: {round(time.time()-task_start_time, 2)}s | "
                f"剩余排队数: {TASK_COUNTER['pending']} | 运行中任务数: {TASK_COUNTER['running']}"
            )

            # 3. 执行高资源任务（同步函数用to_thread避免阻塞事件循环）
            try:
                process_result: dict = await asyncio.wait_for(
                    asyncio.to_thread(
                        process_document,
                        task=task,
                        document_path=file_path,
                        config=config,
                        uuid=uuid,
                    ),
                    timeout=1800.0,  # 30分钟超时
                )
            except asyncio.TimeoutError:
                raise Exception("任务执行超时（30min）")

            # 4. 构造成功回调数据
            callback_data = process_result
            callback_data["status"] = "success"
            callback_data["filename"] = filename
            logger.info(
                f"任务执行完成 | UUID: {uuid} | 总耗时: {round(time.time()-task_start_time, 2)}s"
            )

    except Exception as e:
        import traceback

        # 捕获任务执行异常，构造失败回调数据
        callback_data = {
            "uuid": uuid,
            "task": task,
            "status": "failed",
            "error": str(e),
            "filename": filename,
            "queue_time": round(time.time() - task_start_time, 2),
        }
        logger.error(
            f"任务执行异常 | UUID: {uuid} | 异常信息: {str(e)}\n{traceback.format_exc()}",
            exc_info=True,
        )

    finally:
        # 无论成功/失败：更新计数器 + 清理运行中标记 + 移出排队队列
        if TASK_COUNTER["running"] > 0:
            TASK_COUNTER["running"] -= 1

        # 移除运行中标记
        with RUNNING_TASKS_LOCK:
            if uuid in RUNNING_TASKS:
                RUNNING_TASKS.remove(uuid)
                logger.info(f"任务移出运行中 | UUID: {uuid}")

        # 从排队队列移除（防止队列残留）
        with TASK_QUEUE_LOCK:
            if uuid in TASK_QUEUE_LIST:
                TASK_QUEUE_LIST.remove(uuid)
                logger.info(f"UUID出队 | {uuid} | 剩余队列长度: {len(TASK_QUEUE_LIST)}")

    # 5. 执行回调（单独捕获异常，不影响核心流程）
    try:
        if callback_data:
            callback_api(callback_url, callback_data)
    except Exception as e:
        logger.error(
            f"回调执行失败 | UUID: {uuid} | 回调URL: {callback_url} | 异常: {str(e)}"
        )


@app.post("/v1/process-document")
async def process_document_api(
    background_tasks: BackgroundTasks,
    task: str = Form(...),
    uuid: str = Form(...),
    file: UploadFile = File(...),
):
    """
    文档处理API接口
    - Form参数: task (任务类型), uuid (唯一标识)
    - File参数: file (上传的文档文件)
    """
    # 1. 基础日志记录
    logger.info(
        f"接收API请求 | UUID: {uuid} | 任务类型: {task} | 文件名: {file.filename}"
    )

    # 2. 保存上传文件到临时目录（请求阶段完成，避免后台依赖请求上下文）
    folder_path = ""
    file_path = ""
    try:
        original_filename = file.filename
        sanitized_filename = sanitize_filename(os.path.splitext(original_filename)[0])
        original_suffix = os.path.splitext(original_filename)[1]
        file_content = await file.read()
        file_hash = get_string_hash(file_content)

        # 构建文件存储路径
        folder_name = f"{sanitized_filename}-{file_hash}"
        folder_path = os.path.join(CONFIG["output"]["path"], folder_name)
        file_path = os.path.join(folder_path, f"{sanitized_filename}{original_suffix}")

        # 创建文件夹并写入文件
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"create dir: {folder_path} {os.path.exists(folder_path)}")
        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(
            f"文件保存完成 | UUID: {uuid} | 原文件名: {original_filename} | 保存路径: {file_path}"
        )

    except Exception as e:
        import traceback
        logger.error(f"临时文件保存失败 | UUID: {uuid} | 异常: {str(e)}\n{traceback.format_exc()}")
        # 清理已创建的文件夹（如果有）
        if folder_path and os.path.exists(folder_path):
            try:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
                os.rmdir(folder_path)
            except:
                pass
        return {
            "code": 500,
            "status": "failed",
            "message": "临时文件保存失败",
            "data": {"uuid": uuid},
        }

    # 3. 将UUID加入任务队列（线程安全）
    with TASK_QUEUE_LOCK:
        TASK_QUEUE_LIST.append(uuid)
        logger.info(f"UUID入队 | {uuid} | 当前队列长度: {len(TASK_QUEUE_LIST)}")

    # 4. 添加串行化后台任务
    background_tasks.add_task(
        process_document_background,
        task=task,
        uuid=uuid,
        file_path=file_path,
        filename=file.filename,
        callback_url=CALLBACK_URL_PROCESS_DOC,
        config=CONFIG,
    )

    # 5. 立即返回响应
    with TASK_QUEUE_LOCK:
        current_queue_length = len(TASK_QUEUE_LIST)
    return {
        "code": 200,
        "status": "accepted",
        "message": "task enqueued",
        "data": {
            "uuid": uuid,
            "task": task,
            "filename": file.filename,
            "queue_count": current_queue_length + TASK_COUNTER["running"],
            "position_in_queue": current_queue_length,  # 当前任务在队列中的位置（从0开始）
        },
    }


@app.get("/v1/task-status")
async def get_task_status(uuid: str):
    """
    查看指定UUID的任务状态
    - 返回该任务前面还有多少个排队任务
    - 任务状态：pending(排队中)/running(执行中)/completed/failed(已完成/失败)
    """
    # 加锁读取队列和运行中任务（避免并发修改）
    with TASK_QUEUE_LOCK:
        queue_copy = TASK_QUEUE_LIST.copy()
    with RUNNING_TASKS_LOCK:
        running_copy = RUNNING_TASKS.copy()

    # 核心计算逻辑
    task_ahead = -1
    task_status = "unknown"

    if uuid in queue_copy:
        # 任务在排队中：前面的任务数 = 该UUID在队列中的索引
        task_ahead = queue_copy.index(uuid) + 1
        task_status = "pending"
    elif uuid in running_copy:
        # 任务正在执行：前面无排队任务
        task_ahead = 0
        task_status = "running"
    else:
        # 任务已完成/失败（不在队列也不在运行中）
        task_status = "finished"

    return {
        "code": 200,
        "data": {
            "uuid": uuid,
            "position_in_queue": task_ahead,  # 核心：前面的任务数
            "status": task_status,  # 任务当前状态
            "pending_tasks": len(queue_copy),
            "running_tasks": len(running_copy),
            "total_tasks": len(queue_copy) + len(running_copy),
        },
    }


@app.get("/v1/health")
async def health_check():
    return {
        "code": 200,
        "data": "ok",
    }


@app.on_event("startup")
async def startup_event():
    print_all_env_vars()
    print_lib_versions()
    print(f"origin env: {os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')=}")
    await load_llms(CONFIG["llm"])


@app.on_event("shutdown")  # "startup" 对应启动时，"shutdown" 对应退出时
async def shutdown_event():
    print(f"cleanning up service.")
    await clean_up()


if __name__ == "__main__":
    import uvicorn

    print(f"{backend=} {CONFIG_PATH=}")
    # 启动API服务
    uvicorn.run(app="api:app", host="0.0.0.0", port=8000)
