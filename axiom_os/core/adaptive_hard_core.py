"""
Adaptive Hard Core - 极端情况下自动放松物理约束
PROPRIETARY - Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.
Unauthorized copying, modification, or distribution of this algorithm is prohibited.

当系统逼近极端情况时，Hard Core 权重降低，保持 Soft Shell 灵活性。
"""

from typing import Callable, Optional
import torch


def gate_extremeness(
    magnitude: torch.Tensor,
    threshold: float = 5.0,
    steepness: float = 2.0,
) -> torch.Tensor:
    """
    极端程度门控: 正常时 gate≈1, 极端时 gate→0.
    gate = 1 / (1 + (magnitude/threshold)^steepness)
    """
    ratio = (magnitude / (threshold + 1e-8)).clamp(min=0)
    return 1.0 / (1.0 + ratio ** steepness)


def wrap_adaptive_hard_core(
    base_hard_core: Callable,
    threshold: float = 5.0,
    steepness: float = 2.0,
    extremeness_fn: Optional[Callable] = None,
) -> Callable:
    """
    将任意 Hard Core 包装为自适应版本。
    极端时 (|y_hard| 大) 降低 Hard Core 贡献，让 Soft Shell 主导。

    Args:
        base_hard_core: 原始物理预测函数 x -> y_hard
        threshold: 极端阈值，|y_hard| 超过此值则 gate 下降
        steepness: 门控陡度，越大过渡越陡
        extremeness_fn: 可选，自定义极端程度 (y_hard) -> scalar。默认用 L2 范数

    Returns:
        新的 hard_core(x) -> gate * y_hard
    """
    if extremeness_fn is None:
        def _default_extremeness(y: torch.Tensor) -> torch.Tensor:
            return (y ** 2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)

        extremeness_fn = _default_extremeness

    def adaptive_hard_core(x):
        y_hard = base_hard_core(x)
        if isinstance(y_hard, torch.Tensor):
            y = y_hard.float()
        else:
            y = torch.as_tensor(y_hard, dtype=torch.float32)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        mag = extremeness_fn(y)
        device = y.device
        gate = gate_extremeness(mag, threshold=threshold, steepness=steepness)
        return gate * y

    return adaptive_hard_core
