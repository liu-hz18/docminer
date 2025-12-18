import re
import torch
from loguru import logger
from typing import Iterable, List, Dict, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from .llmhub import get_llm


def chunked(iterable: Iterable, chunk_size: int) -> Iterable:
    """将可迭代对象按指定chunk_size分块（注意：chunk_size需为偶数，适配[是/否]对）"""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def process_inputs_with_yes_no(
    pairs: List[str], tokenizer, max_length: int
) -> Tuple[List[str], Dict[int, int]]:
    """构造含“是/否”的双版本prompt，并记录原始索引映射

    Returns:
        full_prompts: 拼接后的prompt列表（顺序为：prompt1+是, prompt1+否, prompt2+是, prompt2+否...）
        idx_map: 原始prompt索引到双版本prompt索引的映射（如{0: [0,1], 1: [2,3]}）
    """
    full_prompts = []
    idx_map = {}
    for ori_idx, pair in enumerate(pairs):
        # 基础prompt（prefix+pair+suffix）
        base_prompt = f"{pair}"

        # 构造两个版本：+“是” / +“否”
        prompt_yes = base_prompt + "是"
        prompt_no = base_prompt + "否"

        # 截断超长prompt（保留最后token为“是”/“否”）
        for p in [prompt_yes, prompt_no]:
            tokens = tokenizer.encode(p, add_special_tokens=False)
            if len(tokens) > max_length:
                # 保留最后1个token（“是”/“否”），前面内容截断
                keep_last_token = tokens[-1:]
                available_length = max_length - 1
                prefix_suffix_tokens = tokenizer.encode(
                    f"{pair}", add_special_tokens=False
                )[:available_length]
                truncated_tokens = prefix_suffix_tokens + keep_last_token
                p = tokenizer.decode(truncated_tokens)
            full_prompts.append(p)

        # 记录索引映射
        idx_map[ori_idx] = [ori_idx * 2, ori_idx * 2 + 1]

    return full_prompts, idx_map


def compute_yes_no_logprobs(
    llm: LLM, prompts: List[str], tokenizer, config: dict
) -> List[Tuple[float, float]]:
    """获取每个原始prompt对应的“是”和“否”的logprob

    Returns:
        yes_no_logprobs: [(yes_logprob, no_logprob), ...]（与原始prompt顺序一致）
    """
    # 配置采样参数：不生成新token，返回每个prompt token的top1 logprob
    sampling_params = SamplingParams(**config["generate_kwargs"])

    # vLLM推理（获取prompt的logprob）
    outputs = llm.generate(
        prompts=prompts, sampling_params=sampling_params, use_tqdm=False
    )

    yes_token_id = tokenizer.encode("是", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("否", add_special_tokens=False)[0]

    yes_no_logprobs = []
    # 按原始prompt的双版本分组处理（每两个prompt为一组：是/否）
    for i in range(0, len(outputs), 2):
        # 提取“是”版本的最后token logprob
        output_yes = outputs[i]
        yes_token_logprobs = output_yes.prompt_logprobs[
            -1
        ]  # 最后一个token（“是”）的logprob
        yes_logprob = (
            yes_token_logprobs[yes_token_id].logprob
            if yes_token_logprobs
            else -float("inf")
        )

        # 提取“否”版本的最后token logprob
        output_no = outputs[i + 1]
        no_token_logprobs = output_no.prompt_logprobs[
            -1
        ]  # 最后一个token（“否”）的logprob
        no_logprob = (
            no_token_logprobs[no_token_id].logprob
            if no_token_logprobs
            else -float("inf")
        )

        yes_no_logprobs.append((float(yes_logprob), float(no_logprob)))

    return yes_no_logprobs


# -------------------------- 主召回函数 --------------------------
def llm_recall_relevant_paragraphs(
    document: str,
    query: str,
    config: dict,
) -> Tuple[List[Dict[str, str | float]], Dict]:
    """Recall top-k relevant paragraphs by comparing “是/否” logprobs"""
    # 1. 段落分割
    logger.info("Splitting document with regex: \\n\\s*\\n")
    paragraphs = re.split(r"\n\s*\n", document)
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    if not paragraphs:
        logger.warning("No valid paragraphs found in document")
        return [], {"num_paragraphs": 0, "paragraphs": []}
    logger.info(f"Split document into {len(paragraphs)} valid paragraphs")

    # 2. 加载模型和tokenizer
    try:
        llm = get_llm(config["model"])
        tokenizer = llm.get_tokenizer()
        logger.success("Model loaded successfully")
    except Exception as e:
        import traceback

        logger.error(f"Failed to load model: {str(e)}\n{traceback.format_exc()}")
        raise

    # 3. 构造基础prompt模板
    # 结尾不加内容，后续拼接“是/否”
    prompt_file = config["prompt"]
    logger.info(f"load prompt file from {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    base_pairs = [
        prompt_template.replace("{document}", para).replace("{query}", query)
        for para in paragraphs
    ]

    # 4. 构造含“是/否”的双版本prompt
    full_prompts, idx_map = process_inputs_with_yes_no(
        base_pairs, tokenizer, config["max_length"]
    )

    # 5. 分批计算“是/否”的logprob
    yes_no_scores = []
    # 注意：batch_size需设为偶数（适配双版本prompt）
    batch_size = (
        config["batch_size"]
        if config["batch_size"] % 2 == 0
        else config["batch_size"] - 1
    )
    batch_size = max(batch_size, 2)
    for batch_prompts in tqdm(
        chunked(full_prompts, batch_size),
        total=(len(full_prompts) + batch_size - 1) // batch_size,
        desc="[RECALL] computing yes/no logprobs...",
    ):
        batch_logprobs = compute_yes_no_logprobs(llm, batch_prompts, tokenizer, config)
        yes_no_scores.extend(batch_logprobs)

    # 6. 计算最终分数（yes_logprob - no_logprob，或softmax概率）
    final_scores = []
    for yes_logprob, no_logprob in yes_no_scores:
        # 方式1：直接用logprob差值作为分数
        # score = yes_logprob - no_logprob
        # 方式2：softmax计算“是”的概率
        logits = torch.tensor([no_logprob, yes_logprob], dtype=torch.float32)
        score = torch.nn.functional.softmax(logits, dim=0)[1].item()
        final_scores.append(score)

    # 7. 排序并返回结果
    scored_paragraphs = list(enumerate(final_scores))
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)

    top_results = [
        {"paragraph": paragraphs[idx], "score": float(score)}
        for idx, score in scored_paragraphs[: config["topk"]]
    ]

    all_paras = [
        {
            "paragraph": paragraphs[idx],
            "score": float(score),
            "length": len(paragraphs[idx]),
            "yes_logprob": yes_no_scores[idx][0],
            "no_logprob": yes_no_scores[idx][1],
        }
        for idx, score in scored_paragraphs
    ]
    all_results = {"num_paragraphs": len(paragraphs), "paragraphs": all_paras}

    logger.success(
        f"Retrieved top-{config['topk']} relevant paragraphs (highest score: {scored_paragraphs[0][1]:.4f})"
    )
    return top_results, all_results
