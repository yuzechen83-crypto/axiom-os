"""
分区扰动推断 - 根据输入推断所属分区，用于海马体联想/直觉检索

模拟人类：当前情境 → 联想相关经验 → 作为决策参考
"""

from typing import Optional, List, Tuple
import numpy as np

from .partition import (
    RAR_PARTITIONS,
    TURBULENCE_PARTITIONS,
    PARTITION_REGISTRY,
    Partition,
)


def infer_partition_id(x: np.ndarray, domain: str) -> Optional[str]:
    """
    根据输入 x 推断最相关的 partition_id（取样本数最多的分区）。
    用于海马体扰动检索：当前情境 → 联想分区 → 检索该分区的直觉定律。

    Args:
        x: 输入 (n, d) 或 (d,)
        domain: "mechanics"|"rar"|"fluids"|"battery"

    Returns:
        partition_id 或 None
    """
    weighted = infer_partition_weights(x, domain)
    if not weighted:
        return None
    return max(weighted, key=lambda t: t[1])[0]


def infer_partition_weights(x: np.ndarray, domain: str) -> List[Tuple[str, float]]:
    """
    返回 (partition_id, weight) 列表，weight 表示 x 属于该分区的程度。
    用于软加权扰动：y_pert = Σ w_i * pred_i(partition_law)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = x.shape[0]
    parts = PARTITION_REGISTRY.get(domain, RAR_PARTITIONS)
    out = []
    if domain in ("mechanics", "rar"):
        g_bar = x[:, 0] if x.shape[1] >= 1 else x.ravel()
        lg = np.log10(np.maximum(g_bar, 1e-14))
        for p in parts:
            m = p.mask(x)
            w = float(m.sum()) / n if n > 0 else 0.0
            if w > 0:
                out.append((p.id, w))
    elif domain == "fluids":
        z = x[:, 3] if x.shape[1] >= 4 else x[:, 0]
        for p in parts:
            m = p.mask(x)
            w = float(m.sum()) / n if n > 0 else 0.0
            if w > 0:
                out.append((p.id, w))
    else:
        for p in parts:
            m = p.mask(x)
            w = float(m.sum()) / n if n > 0 else 0.0
            if w > 0:
                out.append((p.id, w))
    return out if out else [(parts[0].id, 1.0)] if parts else []
