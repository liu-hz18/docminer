import os
import json
import torch
from loguru import logger
from .query import optimize_query
from .chat import generate_refined_answer
from .compress import compress_paragraphs
from .utils import get_file_hash
from .recall_keyword import keyword_recall_relevant_paragraphs
from .recall_embedding import (
    get_document_embeddings,
    embedding_retrieve_relevant_paragraphs,
)


def load_sys_configs(config_path: str) -> dict:
    """加载主配置和查询配置文件"""
    # 加载主配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.loads(f.read())

    logger.info("配置文件加载完成")
    logger.info(f"Mineru backend: {config['pipeline']['mineru']['backend']}")
    return config


def load_query_configs(config: dict) -> dict:
    # 加载查询配置（优先使用config内配置的路径）
    actual_query_path = config["query"]["path"]
    with open(actual_query_path, "r", encoding="utf-8") as f:
        query_config = json.loads(f.read())
    return query_config


def setup_logger(config: dict):
    # 初始化日志配置（全局唯一）
    # 确保日志目录存在
    log_dir = os.path.join(config["output"]["path"], "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()

    # 配置按行数滚动的文件日志
    logger.add(
        sink=os.path.join(log_dir, f"{config['uuid']}.log"),
        rotation="08:00",
        retention="3 months",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | pid={process.id: <6} | {module}:{line} - {message}",
        level="DEBUG",
    )

    # 配置控制台输出（可选，方便开发调试）
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | uuid="
        + config["uuid"]
        + " | {module}:{line} - {message}",
        level="INFO",
    )


def setup_environment(config: dict) -> None:
    """设置系统环境变量"""
    os.environ["MINERU_MODEL_SOURCE"] = "local"
    os.environ["MINERU_TOOLS_CONFIG_JSON"] = config["pipeline"]["mineru"]["config"]
    os.environ.setdefault(
        "MINERU_VIRTUAL_VRAM_SIZE", str(config["pipeline"]["mineru"]["vram_size"])
    )
    os.environ["DOCMINER_SEED"] = str(config["pipeline"]["seed"])
    logger.info("环境变量设置完成")


def is_path_prefix(prefix_path: str, target_path: str) -> bool:
    """
    检查 prefix_path 是否是 target_path 的路径前缀（层级包含）
    :param prefix_path: 待检查的前缀路径（文件夹/文件路径）
    :param target_path: 目标文件/文件夹路径
    :return: True=是前缀，False=不是
    """
    # 步骤1：规范化路径（处理./、../、末尾分隔符，统一分隔符）
    # 转为绝对路径，避免相对路径干扰
    norm_prefix = os.path.normpath(os.path.abspath(prefix_path))
    norm_target = os.path.normpath(os.path.abspath(target_path))

    # 步骤2：给前缀路径补充末尾的路径分隔符（避免部分字符串匹配）
    # 例如：避免 "/test" 误判为 "/test123/file.txt" 的前缀
    norm_prefix_with_sep = norm_prefix + os.sep

    # 步骤3：检查目标路径是否以规范化的前缀路径开头
    # 额外处理：若目标路径和前缀路径完全一致，也视为前缀
    return norm_target.startswith(norm_prefix_with_sep) or (norm_target == norm_prefix)


def init_output_paths(config: dict, test_document: str) -> tuple[str, str, str]:
    """初始化输出路径和mineru配置"""
    # 基础路径计算
    output_root_path = config["output"]["path"]
    file_name = os.path.splitext(os.path.basename(test_document))[0]
    if is_path_prefix(output_root_path, test_document):
        full_folder_path = os.path.dirname(test_document)
    else:
        file_hash = get_file_hash(test_document)
        full_folder_path = os.path.join(output_root_path, f"{file_name}-{file_hash}")

    # 配置mineru解析模式
    mineru_config = config["pipeline"]["mineru"]
    if mineru_config["mode"] == "pipeline":
        mineru_config["backend"] = "pipeline"
        mineru_config["parse_method"] = "auto"
    elif mineru_config["mode"] == "vlm":
        mineru_config["backend"] = "vlm-vllm-engine"
        mineru_config["parse_method"] = "vlm"
    else:
        raise ValueError(f"不支持的mineru模式: {mineru_config['mode']}")

    # 目标markdown路径
    target_markdown_path = os.path.join(
        full_folder_path, file_name, mineru_config["parse_method"], f"{file_name}.md"
    )
    target_index_path = os.path.join(
        full_folder_path,
        file_name,
        mineru_config["parse_method"],
        f"{file_name}_content_list.json",
    )

    os.makedirs(full_folder_path, exist_ok=True)
    logger.info(f"输出路径初始化完成: {full_folder_path}")
    return full_folder_path, target_markdown_path, target_index_path, file_name


def parse_doc_to_markdown(
    test_document: str,
    full_folder_path: str,
    target_markdown_path: str,
    mineru_config: dict,
) -> str:
    """使用mineru将PDF解析为Markdown文本"""
    sample_document = ""
    # 未生成则执行解析
    if not os.path.exists(target_markdown_path):
        logger.info(
            f"开始mineru解析: mode={mineru_config['mode']}, 输出={target_markdown_path}"
        )
        from .mineru import parse_doc

        parse_doc(
            input_path_list=[test_document],
            output_dir=full_folder_path,
            config=mineru_config,
        )

    # 读取解析结果
    if os.path.exists(target_markdown_path):
        with open(target_markdown_path, "r", encoding="utf-8") as f:
            sample_document = f.read()

    if not sample_document:
        raise RuntimeError(f"mineru解析失败: {test_document}")

    logger.info("Markdown解析完成")
    return sample_document


def get_or_load_embeddings(
    sample_document: str, full_folder_path: str, embedding_config: dict
) -> tuple[torch.Tensor, list]:
    """获取/加载文档嵌入向量和段落数据"""
    # 缓存路径
    embed_path = os.path.join(full_folder_path, "embedding.pt")
    para_path = os.path.join(full_folder_path, "paragraph.json")
    feed_para_path = os.path.join(full_folder_path, "paragraph.feed.json")

    # 加载缓存
    if os.path.exists(embed_path) and os.path.exists(para_path):
        logger.info(f"加载缓存嵌入: {embed_path}")
        embeddings = torch.load(embed_path)
        with open(para_path, "r", encoding="utf-8") as f:
            paragraphs = json.load(f)
    # 生成新嵌入
    else:
        logger.info("生成文档嵌入向量")
        embeddings, paragraphs, feed_paragraphs = get_document_embeddings(
            sample_document, embedding_config
        )
        torch.save(embeddings, embed_path)
        with open(para_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(paragraphs, indent=4, ensure_ascii=False))
        with open(feed_para_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(feed_paragraphs, indent=4, ensure_ascii=False))

    return embeddings, paragraphs


def optimize_queries(
    query_config: dict, query_optimize_config: dict, full_folder_path: str
) -> list:
    """优化查询语句（带缓存）"""
    opt_query_path = os.path.join(full_folder_path, "optimized_query.json")

    if not os.path.exists(opt_query_path):
        logger.info("优化查询语句")
        queries = optimize_query(
            queries=query_config["query"], config=query_optimize_config
        )
        with open(opt_query_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(queries, indent=4, ensure_ascii=False))
    else:
        logger.info(f"加载缓存查询: {opt_query_path}")
        with open(opt_query_path, "r", encoding="utf-8") as f:
            queries = json.load(f)

    return queries


def process_single_query(
    idx: int,
    query_config: dict,
    sample_document: str,
    paragraphs: list,
    embeddings: torch.Tensor,
    full_folder_path: str,
    config: dict,
    file_name: str,
) -> dict | None:
    """处理单个查询（召回→合并→压缩→生成答案）"""
    logger.info(f"=== 处理查询 {idx}: {query_config['query']} ===")

    # 跳过非必需查询
    if not query_config["required"]:
        return None

    # 初始化查询目录
    query_dir = os.path.join(full_folder_path, f"{idx}")
    os.makedirs(query_dir, exist_ok=True)

    # 1. 关键词召回
    keyword_para = _keyword_recall(
        query_config["keyword_query"], sample_document, config, query_dir, file_name
    )

    # 2. 嵌入召回
    embedding_para = _embedding_recall(
        query_config["embedding_query"],
        paragraphs,
        embeddings,
        config,
        query_dir,
        file_name,
    )

    # 3. 合并召回结果
    merged_para = _merge_recall_results(
        keyword_para, embedding_para, config, query_dir, file_name
    )

    # 4. 文本压缩
    compressed_para = _compress_paragraphs(
        merged_para, config, query_config["llm_query"], query_dir, file_name
    )

    # 5. LLM生成答案
    final_answer = _generate_llm_answer(
        query_config["llm_query"], compressed_para, config, query_dir, file_name
    )

    # 6. 保存单查询结果
    result = {
        "key": query_config["key"],
        "value": final_answer,
        "unit": query_config["unit"],
        "related_contexts": [p["paragraph"] for p in merged_para],
    }
    with open(
        os.path.join(query_dir, f"{file_name}_final.json"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps(result, indent=4, ensure_ascii=False))

    return result


def _keyword_recall(
    query: str, document: str, config: dict, save_dir: str, file_name: str
) -> list:
    """关键词召回（内部辅助函数）"""
    save_path = os.path.join(save_dir, f"{file_name}_keyword_recall.json")
    if not os.path.exists(save_path):
        _, result = keyword_recall_relevant_paragraphs(
            document=document, query=query, config=config["pipeline"]["keyword_recall"]
        )
        result["query"] = query
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=4, ensure_ascii=False))
        return result["paragraphs"]
    else:
        logger.info(f"加载缓存关键词召回结果: {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)["paragraphs"]


def _embedding_recall(
    query: str,
    paragraphs: list,
    embeddings: torch.Tensor,
    config: dict,
    save_dir: str,
    file_name: str,
) -> list:
    """嵌入召回（内部辅助函数）"""
    save_path = os.path.join(save_dir, f"{file_name}_embedding_recall.json")
    if not os.path.exists(save_path):
        _, result = embedding_retrieve_relevant_paragraphs(
            paragraphs=paragraphs,
            query=query,
            embeddings=embeddings,
            config=config["pipeline"]["embedding_recall"],
        )
        result["query"] = query
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=4, ensure_ascii=False))
        return result["paragraphs"]
    else:
        logger.info(f"加载缓存嵌入召回结果: {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)["paragraphs"]


def _merge_recall_results(
    keyword_para: list,
    embedding_para: list,
    config: dict,
    save_dir: str,
    file_name: str,
) -> list:
    """合并召回结果（去重）"""
    merge_topk = config["pipeline"]["merge_recall"]["topk"]
    merged = []
    para_hashes = set()

    # 交替合并关键词/嵌入结果，去重
    max_len = min(len(keyword_para), len(embedding_para))
    for i in range(max_len):
        # 处理关键词结果
        k_para = keyword_para[i]
        k_hash = hash(k_para["paragraph"])
        if k_hash not in para_hashes:
            para_hashes.add(k_hash)
            merged.append(
                {
                    "paragraph": k_para["paragraph"],
                    "score": k_para["score"],
                    "length": k_para.get("length", 0),
                    "source": "关键词召回",
                }
            )
            if len(merged) >= merge_topk:
                break

        # 处理嵌入结果
        e_para = embedding_para[i]
        e_hash = hash(e_para["paragraph"])
        if e_hash not in para_hashes:
            para_hashes.add(e_hash)
            merged.append(
                {
                    "paragraph": e_para["paragraph"],
                    "score": e_para["score"],
                    "length": e_para.get("length", 0),
                    "source": "向量召回",
                }
            )
            if len(merged) >= merge_topk:
                break

    # 保存合并结果
    save_path = os.path.join(save_dir, f"{file_name}_merge.json")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(merged, indent=4, ensure_ascii=False))
    return merged


def _compress_paragraphs(
    paragraphs: list, config: dict, query: str, save_dir: str, file_name: str
) -> list:
    """压缩段落文本（带缓存）"""
    save_path = os.path.join(save_dir, f"{file_name}_compress.json")
    if not os.path.exists(save_path):
        result = compress_paragraphs(
            config=config["pipeline"]["compress"],
            relevant_paragraphs=paragraphs,
            query=query,
        )
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=4, ensure_ascii=False))
        return result
    else:
        logger.info(f"加载缓存压缩结果: {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)


def _generate_llm_answer(
    query: str, paragraphs: list, config: dict, save_dir: str, file_name: str
) -> str:
    """生成LLM答案（带缓存）"""
    save_path = os.path.join(save_dir, f"{file_name}_llm.json")
    if not os.path.exists(save_path):
        final_answer, llm_result = generate_refined_answer(
            config=config["pipeline"]["chat"],
            query=query,
            relevant_paragraphs=paragraphs,
        )
        llm_result["query"] = query
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(llm_result, indent=4, ensure_ascii=False))
        return final_answer
    else:
        logger.info(f"加载缓存LLM结果: {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)["final-answer"]


def save_all_results(all_results: list, full_folder_path: str) -> None:
    """保存所有查询结果"""
    save_path = os.path.join(full_folder_path, f"{all_results['uuid']}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(all_results, indent=4, ensure_ascii=False))
    logger.info(f"所有结果已保存: {save_path}")
    return save_path


def process_document_query(test_document: str, config: dict) -> tuple[list, str]:
    """
    处理单份文档+查询的核心函数（适配请求式调用）
    每次执行前重置大模型缓存，确保环境干净、结果可复现

    Args:
        test_document: 待处理的文档路径（如PDF）
        query_path: 查询配置文件路径
        config_path: 主配置文件路径

    Returns:
        tuple: (所有查询结果列表, 结果输出根路径)

    Raises:
        Exception: 流程执行异常（含详细日志）
    """
    try:
        # 1. 加载配置
        config["query"]["path"] = config["query"]["path"].replace(
            "{task}", config["task"]
        )
        query_config = load_query_configs(config)

        # 3. 初始化输出路径
        full_folder_path, target_markdown_path, target_index_path, file_name = (
            init_output_paths(config, test_document)
        )

        # 4. PDF转Markdown
        sample_document = parse_doc_to_markdown(
            test_document,
            full_folder_path,
            target_markdown_path,
            config["pipeline"]["mineru"],
        )

        # get text to page idx map
        with open(target_index_path, "r", encoding="utf-8") as f:
            content_list = json.loads(f.read())
        content_list = [
            content
            for content in content_list
            if content["type"] in ["text", "table", "image"]
        ]

        # 5. 生成/加载嵌入向量
        embeddings, paragraphs = get_or_load_embeddings(
            sample_document, full_folder_path, config["pipeline"]["embedding_recall"]
        )

        # 6. 优化查询语句
        config["pipeline"]["query_optimize"]["keyword_query_prompt"] = config[
            "pipeline"
        ]["query_optimize"]["keyword_query_prompt"].replace("{task}", config["task"])
        config["pipeline"]["query_optimize"]["embedding_query_prompt"] = config[
            "pipeline"
        ]["query_optimize"]["embedding_query_prompt"].replace("{task}", config["task"])
        config["pipeline"]["query_optimize"]["llm_query_prompt"] = config["pipeline"][
            "query_optimize"
        ]["llm_query_prompt"].replace("{task}", config["task"])
        optimized_queries = optimize_queries(
            query_config, config["pipeline"]["query_optimize"], full_folder_path
        )

        # 7. 批量处理查询
        config["pipeline"]["embedding_recall"]["prompt"] = config["pipeline"][
            "embedding_recall"
        ]["prompt"].replace("{task}", config["task"])
        config["pipeline"]["llm_recall"]["prompt"] = config["pipeline"]["llm_recall"][
            "prompt"
        ].replace("{task}", config["task"])
        config["pipeline"]["compress"]["prompt"] = config["pipeline"]["compress"][
            "prompt"
        ].replace("{task}", config["task"])
        config["pipeline"]["chat"]["prompt"] = config["pipeline"]["chat"][
            "prompt"
        ].replace("{task}", config["task"])
        all_results = []
        for idx, single_query in enumerate(optimized_queries, 1):
            result = process_single_query(
                idx=idx,
                query_config=single_query,
                sample_document=sample_document,
                paragraphs=paragraphs,
                embeddings=embeddings,
                full_folder_path=full_folder_path,
                config=config,
                file_name=file_name,
            )
            if result:
                result["value"] = result["value"].replace(result["unit"], "").strip()
                all_results.append(result)

        # process all result
        final_result = []
        for result in all_results:
            text_ids = []
            for text in result["related_contexts"]:
                for content in content_list:
                    if content["type"] == "text":
                        if text == content["text"]:
                            text_ids.append(
                                {
                                    "type": "text",
                                    "text": content["text"],
                                    "pageid": content["page_idx"],
                                }
                            )
                            break
                    elif content["type"] == "table":
                        if text == content.get("table_body", None):
                            caption = content.get("table_caption", [])
                            caption = caption[0] if len(caption) > 0 else ""
                            text_ids.append(
                                {
                                    "type": "table",
                                    "caption": caption,
                                    "text": content.get("table_body", ""),
                                    "pageid": content["page_idx"],
                                }
                            )
                            break
                    elif content["type"] == "image":
                        caption = content.get("image_caption", [])
                        caption = caption[0] if len(caption) > 0 else ""
                        if text == caption:
                            text_ids.append(
                                {
                                    "type": "image",
                                    "caption": caption,
                                    "pageid": content["page_idx"],
                                }
                            )
                            break
                    else:
                        continue

            final_result.append(
                {
                    "key": result["key"],
                    "value": result["value"],
                    "unit": result["unit"],
                    "related_contexts": text_ids,
                }
            )

        api_result = {
            "uuid": config["uuid"],
            "task": config["task"],
            "results": final_result,
            "doc": sample_document,
        }
        # 8. 保存最终结果
        save_path = save_all_results(api_result, full_folder_path)

        logger.success(
            f"文档处理完成 | 文档路径: {test_document} | 输出路径: {save_path} | 有效查询数: {len(all_results)}"
        )
        return api_result, save_path

    except Exception as e:
        import traceback

        logger.error(
            f"文档处理失败 | 文档路径: {test_document} | 错误信息: {str(e)}\n{traceback.format_exc()}",
            exc_info=True,
        )
        raise
