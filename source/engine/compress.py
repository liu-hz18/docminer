import re
from loguru import logger
from vllm import SamplingParams
from typing import List, Dict
from .utils import extract_table_text
from .llmhub import get_llm
from .prompt import get_prompt_and_response_template


def compress_paragraphs(
    config: dict,
    relevant_paragraphs: List[Dict],
    query: str,
) -> List[Dict]:
    """
    使用vllm优化的小模型压缩长文本/表格段落，提升执行效率和吞吐量

    Args:
        slm_path: 小模型路径
        relevant_paragraphs: 需要处理的段落列表，每个元素包含"paragraph"键

    Returns:
        处理后的段落列表，包含压缩信息
    """
    logger.info(f"Loading LLM with vllm from: {config['model']}")
    result = []

    # 1. 分离需要压缩和不需要压缩的段落
    compress_indices = []  # 需要压缩的段落索引
    compress_prompts = []  # 需要压缩的prompt列表
    non_compress_paragraphs = []  # 不需要压缩的段落

    prompt_file = config["prompt"]
    logger.info(f"load prompt file from {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt_func, response_func = get_prompt_and_response_template(config["prompt_template"])

    for idx, p in enumerate(relevant_paragraphs):
        if (
            "<table>" in p["paragraph"]
            or len(p["paragraph"]) > config["need_compress_length"]
        ):
            if "<table>" in p["paragraph"]:
                try:
                    prompt = prompt_template.replace(
                        "{document}", extract_table_text(p["paragraph"])
                    )
                except Exception as e:
                    logger.info(
                        f"transform HTML table into string failed. Exception: {str(e)}"
                    )
                    prompt = prompt_template.replace("{document}", p["paragraph"])
            else:
                prompt = prompt_template.replace("{document}", p["paragraph"])
            prompt = prompt.replace("{query}", query)
            
            prompt = prompt_func(prompt, thinking=True)
            logger.debug(
                f"Compress-stage prompt (length={len(prompt)}) idx={idx}: {prompt[:4096]}..."
            )
            compress_indices.append(idx)
            compress_prompts.append(prompt)
            # 先初始化结果结构
            result.append(
                {
                    "score": p["score"],
                    "compressed": True,
                    "length": 0,
                    "paragraph": "",
                    "origin_length": len(p["paragraph"]),
                    "origin_paragraph": p["paragraph"],
                    "source": p["source"],
                    "think": "",
                    "prompt": prompt,
                }
            )
        else:
            # 不需要压缩的段落直接处理
            processed_p = {
                "score": p["score"],
                "compressed": False,
                "length": len(p["paragraph"]),
                "paragraph": p["paragraph"],
                "origin_length": len(p["paragraph"]),
                "origin_paragraph": p["paragraph"],
                "source": p["source"],
                "think": "",
                "prompt": "",
            }
            non_compress_paragraphs.append((idx, processed_p))
            result.append(processed_p)

    # 2. 使用vllm批量生成（核心优化点：批量处理提升吞吐量）
    if compress_prompts:
        try:
            # 初始化vllm引擎（自动处理tokenizer，支持bf16、批量推理）
            llm = get_llm(config["model"])

            # 配置vllm采样参数（替代原GenerationConfig）
            sampling_params = SamplingParams(**config["generate_kwargs"])

            logger.success("vllm engine and sampling config loaded successfully")
        except Exception as e:
            import traceback

            logger.error(
                f"Failed to load vllm engine: {str(e)}\n{traceback.format_exc()}"
            )
            raise

        logger.info(f"Batch generating for {len(compress_prompts)} prompts with vllm")

        try:
            # vllm批量推理（自动优化批处理）
            outputs = llm.generate(
                prompts=compress_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,  # 可选：关闭进度条
            )

            # 解析生成结果并回填
            for idx, output in zip(compress_indices, outputs):
                llm_response = output.outputs[0].text.strip()
                think_response, compress_context = response_func(llm_response, thinking=True)

                result[idx]["think"] = think_response

                compress_context = re.sub(r" +", " ", compress_context).strip()
                logger.debug(
                    f"Compressed context (idx={idx}, length={len(compress_context)}): {compress_context[:200]}..."
                )

                # 更新结果
                result[idx]["paragraph"] = compress_context
                result[idx]["length"] = len(compress_context)

        except Exception as e:
            logger.error(f"Failed to batch generate with vllm: {str(e)}")
            raise

    return result
