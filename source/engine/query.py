from vllm import SamplingParams
from typing import List, Dict
from loguru import logger
from .llmhub import get_llm
from .prompt import get_prompt_and_response_template


def optimize_query(queries: List, config: Dict) -> List:
    logger.info(f"Loading LLM with vllm from: {config['model']}")

    try:
        llm = get_llm(config["model"])
        sampling_params = SamplingParams(**config["generate_kwargs"])
    except Exception as e:
        import traceback

        logger.error(f"Failed to load vllm engine: {str(e)}\n{traceback.format_exc()}")
        raise

    prompt_func, response_func = get_prompt_and_response_template(config["prompt_template"])

    # optimize keyword prompt
    logger.info("Optimizing keyword queries...")
    prompts = []
    with open(config["keyword_query_prompt"], "r", encoding="utf-8") as f:
        prompt_template = f.read()
    for query_config in queries:
        query = query_config["query"]
        user_prompt = prompt_template.replace("{query}", query)
        prompt = prompt_func(user_prompt, thinking=True)
        prompts.append(prompt)

    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=False,  # 可选：关闭进度条
    )
    for idx, output in enumerate(outputs):
        llm_response = output.outputs[0].text.strip()
        think_response, chat_response = response_func(llm_response, thinking=True)
        output = chat_response.replace("优化后的query：", "").replace("优化后的query: ", "").strip("\n").strip()
        queries[idx]["keyword_query_think"] = think_response
        queries[idx]["keyword_query"] = output

    # optimize embedding prompt
    logger.info("Optimizing embedding queries...")
    prompts = []
    with open(config["embedding_query_prompt"], "r", encoding="utf-8") as f:
        prompt_template = f.read()
    for query_config in queries:
        query = query_config["query"]
        user_prompt = prompt_template.replace("{query}", query)
        prompt = prompt_func(user_prompt, thinking=True)
        prompts.append(prompt)

    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=False,  # 可选：关闭进度条
    )
    for idx, output in enumerate(outputs):
        llm_response = output.outputs[0].text.strip()
        think_response, chat_response = response_func(llm_response, thinking=True)
        output = chat_response.replace("优化后的query：", "").replace("优化后的query: ", "").strip("\n").strip()
        queries[idx]["embedding_query_think"] = think_response
        queries[idx]["embedding_query"] = output

    # optimize LLM prompt
    logger.info("Optimizing chat queries...")
    prompts = []
    with open(config["llm_query_prompt"], "r", encoding="utf-8") as f:
        prompt_template = f.read()
    for query_config in queries:
        query = query_config["query"]
        user_prompt = prompt_template.replace("{query}", query)
        prompt = prompt_func(user_prompt, thinking=True)
        prompts.append(prompt)

    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=False,  # 可选：关闭进度条
    )
    for idx, output in enumerate(outputs):
        llm_response = output.outputs[0].text.strip()
        think_response, chat_response = response_func(llm_response, thinking=True)
        output = chat_response.replace("优化后的query：", "").replace("优化后的query: ", "").strip("\n").strip()
        queries[idx]["llm_query_think"] = think_response
        queries[idx]["llm_query"] = output

    return queries
