import os

__llmhub = {}
__llmhub_device = {}

def load_llms(config: dict) -> None:
    global __llmhub
    __llmhub = {}  # clear LLM cache
    for k, model_conf in config.items():
        device = model_conf.pop('device')
        __llmhub_device[k] = device
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device
        print(F"ASCEND_RT_VISIBLE_DEVICES={os.environ['ASCEND_RT_VISIBLE_DEVICES']}")
        from vllm import LLM
        __llmhub[k] = LLM(
            **model_conf
        )

def get_llm(key: str):
    global __llmhub
    if key in __llmhub.keys():
        device = __llmhub_device[key]
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device
        return __llmhub[key]
    raise ValueError(f"llm '{key}' not in llmhub. available keys are {list(__llmhub.keys())}")
