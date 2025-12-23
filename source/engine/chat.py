from loguru import logger
from vllm import SamplingParams
from typing import List, Dict, Tuple
from .llmhub import get_llm
from .prompt import get_prompt_and_response_template


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
    prompt_file = config["prompt"]
    logger.info(f"load prompt file from {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = prompt_template.replace("{context}", context).replace("{query}", query)

    prompt_func, response_func = get_prompt_and_response_template(config["prompt_template"])
    prompt = prompt_func(prompt, thinking=True)
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
        chat_answer = outputs[0].outputs[0].text.strip()
        result["chat-answer"] = chat_answer
        logger.debug(f"LLM chat answer: {chat_answer}")

        think_response, final_answer = response_func(chat_answer, thinking=True)
        final_answer = (
            final_answer.replace("最终答案：", "")
            .replace("最终答案为", "")
            .replace("最终答案", "")
            .strip()
        )
        result["thinking"] = think_response
        result["final-answer"] = final_answer
        logger.success(f"LLM final answer: {final_answer}")

        return final_answer, result

    except Exception as e:
        import traceback

        logger.error(
            f"Failed to generate refined answer: {str(e)}\n{traceback.format_exc()}"
        )
        raise
