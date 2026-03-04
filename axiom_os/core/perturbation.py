"""
分区扰动推断 - 根据输入推断所属分区，用于海马体联想/直觉检索

模拟人类：当前情境 → 联想相关经验 → 作为决策参考
"""

from typing import Optional, List, Tuple
import numpy as np

from .partition import (
    RAR_PARTITIONS,
    TURBULENCE_PARTITIONS,
    HIGH_FREQUENCY_PARTITIONS,
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
    elif domain == "high_freq":
        # 序列 (B, seq_len, 3) 或 (B, 3): 用最后一帧 z 推断分区
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


def infer_partition_per_sample(x: np.ndarray, domain: str) -> Tuple[np.ndarray, List[str]]:
    """
    逐样本推断分区。返回 (indices, partition_ids)：
    - indices: (n,) 每个样本所属分区在 partition_ids 中的索引
    - partition_ids: 分区 id 列表（与 PARTITION_REGISTRY 顺序一致）
    用于 scheme_a/c 的 per-sample 分支选择。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = x.shape[0]
    parts = PARTITION_REGISTRY.get(domain, RAR_PARTITIONS)
    partition_ids = [p.id for p in parts]
    indices = np.full(n, 0, dtype=np.int64)  # 默认第一个分区
    for idx, p in enumerate(parts):
        m = p.mask(x)
        indices[m] = idx
    return indices, partition_ids
