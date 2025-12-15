import os
import sys
import argparse
import datetime
from loguru import logger
from copy import deepcopy
from .engine.pipeline import (
    process_document_query,
    load_sys_configs,
    setup_environment,
    setup_logger,
)
from .engine.llmhub import load_llms
from .engine.utils import clean_up


# 可用任务列表
AVAILABLE_TASK = ["guangfu", "fengdian"]


def main(task: str, document: str, config: dict, uuid: str):
    """
    改造后主函数，增加UUID参数
    """
    if task not in AVAILABLE_TASK:
        logger.error(f"任务类型错误 | 支持的任务: {AVAILABLE_TASK}", exc_info=True)
        sys.exit(-1)

    try:
        # 配置中加入UUID
        config_copy = deepcopy(config)
        config_copy["task"] = task
        config_copy["uuid"] = uuid

        # 执行核心处理流程（传入UUID）
        api_results, output_path = process_document_query(
            test_document=document,
            config=config_copy,
        )

        logger.info(
            f"文档处理完成 | "
            f"UUID: {uuid} | "
            f"结果数量: {len(api_results['results'])} | "
            f"结果路径: {output_path}"
        )
        return api_results, output_path

    except Exception as e:
        import traceback

        logger.error(
            f"文档处理失败 | "
            f"UUID: {uuid} | "
            f"错误信息: {str(e)}\n{traceback.format_exc()}",
            exc_info=True,
        )
        sys.exit(-1)


# 默认测试文档列表（保持与原代码一致）
backend = os.environ.get("DOCMINER_BACKEND", "cuda")
if backend == "npu":
    DEFAULT_TEST_DOCUMENTS = [
        r"/app/documents/test1.pdf",
        r"/app/documents/test2.pdf",
        r"/app/documents/test3.pdf",
    ]
    CONFIG_PATH = r"/app/config/config-ascend.json"
else:
    DEFAULT_TEST_DOCUMENTS = [
        r"/mnt/c/Users/Administrator/Desktop/docminer/documents/test1.pdf",
        r"/mnt/c/Users/Administrator/Desktop/docminer/documents/test2.pdf",
        r"/mnt/c/Users/Administrator/Desktop/docminer/documents/test3.pdf",
    ]
    CONFIG_PATH = r"/mnt/c/Users/Administrator/Desktop/docminer/config/config.json"


# python source/cli.py --pdf /mnt/c/Users/Administrator/Desktop/docminer/documents/test2.pdf --task guangfu
if __name__ == "__main__":
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="文档处理CLI工具")
    parser.add_argument(
        "--config_path",
        type=str,
        default=CONFIG_PATH,
        help="配置文件路径",
    )
    parser.add_argument("--pdf", type=str, required=True, help="解析文件路径")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=AVAILABLE_TASK,
        help=f"任务类型，可选值: {AVAILABLE_TASK}",
    )
    args = parser.parse_args()

    print(f"{backend=} {args.config_path=}")
    # 2. 生成CLI默认UUID（时间戳）
    cli_uuid = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.info(f"生成CLI UUID: {cli_uuid}")

    # 3. 系统初始化
    config = load_sys_configs(args.config_path)
    setup_environment(config)
    load_llms(config["llm"])
    config["uuid"] = cli_uuid
    setup_logger(config)
    logger.info(
        f"系统初始化完成 | 配置路径: {args.config_path} | 任务类型: {args.task}"
    )

    # 4. 执行文档处理流程
    logger.info(f"开始处理文档 | UUID: {cli_uuid} | 文档路径: {args.pdf}")
    main(task=args.task, document=args.pdf, config=config, uuid=cli_uuid)

    logger.info(f"所有文档处理完成 | UUID: {cli_uuid}")

    clean_up()
