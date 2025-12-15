from vllm import SamplingParams
from typing import List, Dict
from loguru import logger
from .llmhub import get_llm


def optimize_query(queries: List, config: Dict) -> List:
    logger.info(f"Loading LLM with vllm from: {config['model']}")

    try:
        llm = get_llm(config["model"])
        sampling_params = SamplingParams(**config["generate_kwargs"])
    except Exception as e:
        import traceback

        logger.error(f"Failed to load vllm engine: {str(e)}\n{traceback.format_exc()}")
        raise

    START_TOKEN, END_TOKEN = "<|im_start|>", "<|im_end|>"
    START_THINK, END_THINK = "<think>", "</think>"

    # optimize keyword prompt
    prompts = []
    with open(config["keyword_query_prompt"], "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt_template = f"{START_TOKEN}system\n你是一个有用的助手\n{END_TOKEN}\n{START_TOKEN}user\n{prompt_template}\n{END_TOKEN}\n{START_TOKEN}assistant\n{START_THINK}\n"
    for query_config in queries:
        query = query_config["query"]
        prompts.append(prompt_template.replace("{query}", query))
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=False,  # 可选：关闭进度条
    )
    for idx, output in enumerate(outputs):
        output = output.outputs[0].text.strip().split(END_TOKEN)[0]
        pos = output.find(END_THINK)
        output = output[pos + len(END_THINK) :] if pos != -1 else output
        output = output.strip("\n").strip().split("\n\n")[0]
        output = output.replace("优化后的query：", "").strip("\n").strip()
        queries[idx]["keyword_query"] = output

    # optimize embedding prompt
    prompts = []
    with open(config["embedding_query_prompt"], "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt_template = f"{START_TOKEN}system\n你是一个有用的助手\n{END_TOKEN}\n{START_TOKEN}user\n{prompt_template}\n{END_TOKEN}\n{START_TOKEN}assistant\n{START_THINK}\n"
    for query_config in queries:
        query = query_config["query"]
        prompts.append(prompt_template.replace("{query}", query))
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=False,  # 可选：关闭进度条
    )
    for idx, output in enumerate(outputs):
        output = output.outputs[0].text.strip().split(END_TOKEN)[0]
        pos = output.find(END_THINK)
        output = output[pos + len(END_THINK) :] if pos != -1 else output
        output = output.strip("\n").strip().split("\n\n")[0]
        output = output.replace("优化后的query：", "").strip("\n").strip()
        queries[idx]["embedding_query"] = output

    # optimize LLM prompt
    prompts = []
    with open(config["llm_query_prompt"], "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt_template = f"{START_TOKEN}system\n你是一个有用的助手\n{END_TOKEN}\n{START_TOKEN}user\n{prompt_template}\n{END_TOKEN}\n{START_TOKEN}assistant\n{START_THINK}\n"
    for query_config in queries:
        query = query_config["query"]
        prompts.append(prompt_template.replace("{query}", query))
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=False,  # 可选：关闭进度条
    )
    for idx, output in enumerate(outputs):
        output = output.outputs[0].text.strip().split(END_TOKEN)[0]
        pos = output.find(END_THINK)
        output = output[pos + len(END_THINK) :] if pos != -1 else output
        output = output.strip("\n").strip().split("\n\n")[0]
        output = output.replace("优化后的query：", "").strip("\n").strip()
        queries[idx]["llm_query"] = output

    return queries
