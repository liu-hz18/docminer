import re
import torch
from typing import List, Tuple, Dict, Any
from loguru import logger
from .llmhub import get_llm


def replace_html_tags(text):
    """
    将连续的<任意内容>标签替换为|，并清理多余分隔符
    :param text: 原始包含html标签的文本
    :return: 替换后的文本
    """
    # 正则匹配所有<开头、>结尾的标签（非贪婪匹配，避免跨标签匹配）
    # 正则解释：<.*?> 匹配 < 开头，> 结尾的任意字符（非贪婪）
    pattern = r"<.*?>"
    # 第一步：将所有标签替换为|
    temp = re.sub(pattern, "|", text)
    # 第二步：清理多余的|（连续|、开头/结尾的|、换行符）
    # 1. 替换连续的|为单个|
    temp = re.sub(r"\|+", "|", temp)
    # 2. 替换换行符为空（去除换行）
    temp = re.sub(r"\n", "", temp)
    # 3. 去除开头和结尾的|
    result = temp.strip("|")
    return result


def get_document_embeddings(
    document: str, config: dict
) -> Tuple[torch.Tensor, List[str]]:
    """
    函数1：计算文档分段的归一化embedding，带缓存机制
    Args:
        document: 原始文档字符串
    Returns:
        归一化的embedding张量 (num_paragraphs, embedding_dim)
        分段后的段落列表
    """
    max_length = config["max_length"]
    # 按\n\n分段并过滤空段落
    paragraphs = re.split(r"\n\s*\n", document)
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    if not paragraphs:
        raise ValueError("no valid paragraphs in document")

    # cut too long paragraphs
    feed_paragraphs = [replace_html_tags(para)[:max_length] for para in paragraphs]
    paragraphs_is_empty = [para == "" or len([para]) <= 0 for para in feed_paragraphs]
    nonempty_paragraphs = [
        para for para in feed_paragraphs if para != "" and len(para) > 0
    ]
    logger.info(
        f"total number of paragraphs: {len(feed_paragraphs)}. non-empty paragraphs: {len(nonempty_paragraphs)}"
    )

    # 初始化vllm embedding模型
    logger.info(f"init embedding model: {config['model']}")
    llm = get_llm(config["model"])

    # 计算段落embedding（文档段落不需要加指令）
    logger.info(f"compute number of {len(nonempty_paragraphs)} embedding...")
    outputs = llm.embed(nonempty_paragraphs)

    # 转换为张量并归一化
    embedding_template = outputs[0].outputs.embedding
    embeddings = []
    index = 0
    for idx, is_empty in enumerate(paragraphs_is_empty):
        if is_empty:
            embeddings.append(torch.zeros_like(embedding_template))
        else:
            embeddings.append(outputs[index].outputs.embedding)
            index += 1
    embeddings = torch.tensor(embeddings)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings, paragraphs, feed_paragraphs


def embedding_retrieve_relevant_paragraphs(
    query: str,
    paragraphs: List[str],
    embeddings: torch.Tensor,
    config: dict,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # 计算查询的归一化embedding
    logger.info("compute query embedding...")

    llm = get_llm(config["model"])

    prompt_file = config["prompt"]
    logger.info(f"load prompt file from {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    query_text = prompt_template.replace("{query}", query)
    query_output = llm.embed([query_text])
    query_embedding = torch.tensor([o.outputs.embedding for o in query_output])
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

    # 计算内积相似度（归一化后内积等价于余弦相似度）
    embeddings = embeddings.to(query_embedding.device)
    scores = torch.matmul(query_embedding, embeddings.T).squeeze(0)

    sorted_paragraphs = list(enumerate(scores))
    sorted_paragraphs.sort(key=lambda x: x[1], reverse=True)
    top_results = [
        {"paragraph": paragraphs[idx], "score": float(score)}
        for idx, score in sorted_paragraphs[: config["topk"]]
    ]

    all_paras = [
        {
            "paragraph": paragraphs[idx],
            "score": float(score),
            "length": len(paragraphs[idx]),
        }
        for idx, score in sorted_paragraphs
    ]
    all_results = {
        "num_paragraphs": len(paragraphs),
        "paragraphs": all_paras,
    }

    logger.success(
        f"Retrieved top-{config['topk']} relevant paragraphs (highest score: {sorted_paragraphs[0][1]:.4f})"
    )
    return top_results, all_results
