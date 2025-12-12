import hashlib
from bs4 import BeautifulSoup
import re
import os
import gc
import torch
from typing import Tuple

try:
    import torch_npu
    # NOTE: 多进程 (vllm 有一个 engine 会负责管理和调用模型分片) 的时候，npu 不能 init。vllm 会在子进程自动 init
    # if not torch_npu.npu.is_initialized():
    #     torch_npu.npu.init()
except ImportError:
    pass


HASH_ALGO = "md5"


def get_file_hash(file_path, hash_algorithm=HASH_ALGO):
    """计算文件的哈希值"""
    hash_obj = hashlib.new(hash_algorithm)
    with open(file_path, "rb") as f:
        # 分块读取大文件，避免内存溢出
        while chunk := f.read(4096):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()[:16]


def get_string_hash(content: str, hash_algorithm=HASH_ALGO):
    """计算文件的哈希值"""
    hash_obj = hashlib.new(hash_algorithm)
    hash_obj.update(content)
    return hash_obj.hexdigest()[:16]


def sanitize_filename(filename: str) -> str:
    """清理文件名中的非法字符，适配Windows/Linux"""
    # 移除/替换系统非法字符
    illegal_chars = r'[\/:*?"<>|]'
    sanitized = re.sub(illegal_chars, "_", filename)
    # 移除首尾空格/下划线，避免前缀/后缀冗余
    return sanitized.strip().strip("_")


def get_device() -> str:
    device_mode = os.getenv('DEVICE_TYPE', None)
    if device_mode is not None:
        return str(device_mode)
    else:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            try:
                if torch_npu.npu.is_available():
                    return "npu"
            except Exception as e:
                pass
        return "cpu"


def init_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    try:
        if torch_npu.npu.is_available():
            torch_npu.npu.manual_seed(seed)
            torch_npu.npu.manual_seed_all(seed)
    except Exception as _:
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_memory(device: str=None) -> None:
    if not device:
        device = get_device()
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif str(device).startswith("npu"):
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()
    gc.collect()


def get_device_count() -> int:
    device = get_device()
    if device.startswith("npu"):
        try:
            import torch_npu
            return torch_npu.npu.device_count()
        except Exception as _:
            pass
    elif device.startswith("cuda"):
        return torch.cuda.device_count()
    return 0


def get_vram(device) -> Tuple[int, int]:
    if isinstance(device, int):
        device_type = get_device()
        device = f"{device_type}:{device}"
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        used_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        cached_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)
        return total_memory, used_memory, cached_memory
    elif str(device).startswith("npu"):
        total_memory = torch_npu.npu.get_device_properties(device).total_memory / (1024 ** 3)
        used_memory = torch_npu.npu.memory_allocated(device) / (1024 ** 3)
        cached_memory = torch_npu.npu.memory_reserved(device) / (1024 ** 3)
        return total_memory, used_memory, cached_memory
    else:
        return None, None, None


def extract_table_text(html_str):
    """
    从表格HTML字符串中提取纯文本，单元格用|分隔，行用换行符分隔
    
    参数:
        html_str (str): 表格的HTML表示字符串（支持含colspan/rowspan等属性）
    
    返回:
        str: 格式化后的纯文本结果
    """
    # 初始化BeautifulSoup解析器（处理HTML标签）
    soup = BeautifulSoup(html_str, 'html.parser')
    
    # 定位表格标签，无表格则返回空字符串
    table = soup.find('table')
    if not table:
        return ""
    
    # 存储每行的格式化文本
    row_results = []
    
    # 遍历表格的每一行
    for tr in table.find_all('tr'):
        # 提取当前行的所有单元格（td=内容，th=表头）
        cells = []
        for cell in tr.find_all(['td', 'th']):
            # 提取单元格文本并清理首尾空白
            cell_text = cell.get_text(strip=True)
            # 替换内部多余空白（换行/制表符/多个空格→单个空格）
            cell_text = re.sub(r'\s+', ' ', cell_text)
            cells.append(cell_text)
        
        # 单元格间用|连接，添加到行结果列表
        row_results.append('|'.join(cells))
    
    # 行之间用换行符连接，返回最终结果
    return '\n'.join(row_results)


DEFAULT_VLLM_KWARGS = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "dtype": "bfloat16",
    "max_num_batched_tokens": 14336,
    "max_num_seqs": 64,
    "disable_log_stats": True,
    # "trust_remote_code": True,
    "max_model_len": 14336,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
    "enforce_eager": False,
}


if __name__ == "__main__":
    temp = "<table><tr><td colspan=\"1\" rowspan=\"1\">方案类型</td><td colspan=\"1\" rowspan=\"1\">变化幅度</td><td colspan=\"1\" rowspan=\"1\">投资回收期所得税后（年）</td><td colspan=\"1\" rowspan=\"1\">项目投资财务内部收益率（所得税前）%）</td><td colspan=\"1\" rowspan=\"1\">项目投资财务内部收益率（所得税后）%）</td><td colspan=\"1\" rowspan=\"1\">资本金财务内部收益率（%）</td><td colspan=\"1\" rowspan=\"1\">项目投资财务净现值（所得税后）（万元）</td><td colspan=\"1\" rowspan=\"1\">资本金财务净现值（万元）</td><td colspan=\"1\" rowspan=\"1\">总投资收益率(RO1）(%)</td><td colspan=\"1\" rowspan=\"1\">投资利税率）</td><td colspan=\"1\" rowspan=\"1\">项目资本金净利润率（ROE）(%)</td><td colspan=\"1\" rowspan=\"1\">GR(%）</td></tr><tr><td colspan=\"1\" rowspan=\"9\">建设投资变化分析（%）（其中包含建设期土地租金1年和运营期土地租金9年）</td></td></tr></table>"
    print(extract_table_text(temp))

