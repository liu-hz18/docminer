from typing import Tuple


def qwen3_prompt(user: str, thinking: bool=True):
    system_prmopt = "你是一个名为 Qwen3 的人工智能助手。你是基于通义千问训练的语言模型 Qwen3 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。你要保持用户原始描述的意图，不要虚构内容或现实不存在的人事物。"
    prompt = f"<|im_start|>system\n{system_prmopt}\n<|im_end|>\n<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n"
    if thinking:
        prompt += "<think>\n"
    return prompt

def qwen3_response(response: str, thinking: bool=True) -> Tuple[str, str]:
    llm_response = response.split("<|im_end|>")[0].strip('\n')
    if thinking:
        think_response = llm_response.split("</think>")[0].strip('\n').split('<think>')[-1].strip('\n').strip()
        chat_response = llm_response.split("</think>")[-1].strip('\n').strip()
    else:
        think_response = ""
        chat_response = llm_response.strip()
    return think_response, chat_response


def glm_prompt(user: str, thinking: bool=True):
    system_prmopt = "你是一个名为 GLM-4.5 的人工智能助手。你是基于 Zhipu AI 训练的语言模型 GLM-4.5 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。你要保持用户原始描述的意图，不要虚构内容或现实不存在的人事物。"
    prompt = f"[gMASK]<sop>\n<|system|>\n{system_prmopt}\n<|user|>\n{user}\n<|assistant|>\n"
    if thinking:
        prompt += "<think>\n"
    return prompt

def glm_response(response: str, thinking: bool=True):
    llm_response = response.split("<|endoftext|>")[0].strip('\n').split("<eop>")[0].strip('\n')
    if thinking:
        think_response = llm_response.split("</think>")[0].strip('\n').split('<think>')[-1].strip('\n').strip()
        chat_response = llm_response.split("</think>")[-1].strip('\n').strip()
    else:
        think_response = ""
        chat_response = llm_response.strip()
    return think_response, chat_response


def get_prompt_and_response_template(key: str):
    if "qwen3" in key.lower():
        return qwen3_prompt, qwen3_response
    elif "glm" in key.lower():
        return glm_prompt, glm_response
    else:
        raise ValueError(f"Unsupported model for prompt template: {key}")
