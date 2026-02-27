"""
可复现性：统一随机种子
用于基准测试、实验、验证，确保结果可复现。
"""

import os
import random
from typing import Optional

# 默认基准种子（论文/报告推荐使用固定值）
DEFAULT_BENCHMARK_SEED = 42


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    设置全局随机种子（numpy, random, torch）。
    返回实际使用的 seed。
    """
    s = seed if seed is not None else DEFAULT_BENCHMARK_SEED
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(s)
    return s
