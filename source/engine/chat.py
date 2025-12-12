from loguru import logger
from vllm import SamplingParams, LLM
from typing import List, Dict, Tuple
from .llmhub import get_llm


def generate_refined_answer(
    config: dict,
    query: str,
    relevant_paragraphs: List[Dict[str, str | float]],
) -> Tuple[str, Dict]:
    """Load LLM model with vllm engine for two-stage answer generation"""
    logger.info(f"Loading LLM with vllm from: {config['model']}")
    result = {}

    try:
        # 初始化vllm引擎（自动处理tokenizer和模型加载）
        llm = get_llm(config["model"])
        logger.success("LLM vllm engine loaded successfully")
    except Exception as e:
        import traceback

        logger.error(
            f"Failed to load LLM with vllm: {str(e)}\n{traceback.format_exc()}"
        )
        raise

    # 构造参考上下文
    logger.info(f"[LLM] received {len(relevant_paragraphs)} most relevant paragraphs")
    context = ""
    for i, p in enumerate(relevant_paragraphs, 1):
        context += f"{i}. 段落（召回源: {p['source']}，召回路相关性分数：{p['score']:.4f}）：{p['paragraph']}\n\n"

    # -------------------------- 提取最终答案 --------------------------
    START_TOKEN, END_TOKEN = "<|im_start|>", "<|im_end|>"
    START_THINK, END_THINK = "<think>", "</think>"
    prompt_file = config["prompt"]
    logger.info(f"load prompt file from {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = prompt_template.replace("{context}", context).replace("{query}", query)

    prompt = f"{START_TOKEN}system\n你是一个有用的助手\n{END_TOKEN}\n{START_TOKEN}user\n{prompt}\n{END_TOKEN}\n{START_TOKEN}assistant\n{START_THINK}\n"
    logger.debug(f"LLM chat prompt (length={len(prompt)}): {prompt[:4096]}...")
    result["prompt"] = prompt

    try:
        sampling_params = SamplingParams(**config["generate_kwargs"])

        # vllm生成最终答案
        outputs = llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # 解析最终答案
        final_answer = outputs[0].outputs[0].text.strip()
        result["chat-answer"] = final_answer
        logger.debug(f"LLM final answer: {final_answer}")

        # find first </think>
        final_answer = final_answer.split(END_TOKEN)[0]
        pos = final_answer.find(END_THINK)
        final_answer = (
            final_answer[pos + len(END_THINK) :] if pos != -1 else final_answer
        )
        final_answer = final_answer.strip("\n").strip()

        final_answer = final_answer.split("\n")[0]
        final_answer = (
            final_answer.replace("最终答案：", "")
            .replace("最终答案为", "")
            .replace("最终答案", "")
            .strip()
        )
        result["final-answer"] = final_answer
        logger.success(f"LLM stripped final answer: {final_answer}")

        return final_answer, result
    except Exception as e:
        import traceback

        logger.error(
            f"Failed to generate refined answer: {str(e)}\n{traceback.format_exc()}"
        )
        raise
