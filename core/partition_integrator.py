"""
Axiom 智能分区模块 - 整合

将各分区学到的公式整合为统一预测。支持硬切换、软门控。
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np

from .partition import Partition, RAR_PARTITIONS


def integrate_hard_switch(
    x: np.ndarray,
    partition_results: Dict[str, Dict[str, Any]],
    partitions: Optional[List[Partition]] = None,
) -> np.ndarray:
    """
    硬切换整合：根据 x 所属分区，选用该分区的公式预测。

    Args:
        x: 输入 (n,) 或 (n,1)，如 g_bar
        partition_results: {partition_id: {formula, g0, ...}}
        partitions: 分区列表，None 用 RAR_PARTITIONS

    Returns:
        预测 y (n,)
    """
    partitions = partitions or RAR_PARTITIONS
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.zeros_like(x, dtype=np.float64)
    assigned = np.zeros(len(x), dtype=bool)

    for p in partitions:
        if p.id not in partition_results:
            continue
        res = partition_results[p.id]
        mask = p.mask(x.reshape(-1, 1) if x.ndim == 1 else x)
        if not np.any(mask):
            continue
        pred = _eval_partition_formula(x[mask], res)
        y[mask] = pred
        assigned[mask] = True

    # 未分配区域：用最近分区或线性插值
    if not np.all(assigned):
        for i in np.where(~assigned)[0]:
            # 用全局 fallback：McGaugh with g0 from mid if available
            g0 = 3700.0
            for pid, res in partition_results.items():
                if res.get("g0") is not None:
                    g0 = res["g0"]
                    break
            xi = x[i]
            denom = 1.0 - np.exp(-np.sqrt(xi / (g0 + 1e-14)))
            y[i] = xi / np.maximum(denom, 0.01)

    return y


def integrate_soft_gate(
    x: np.ndarray,
    partition_results: Dict[str, Dict[str, Any]],
    partitions: Optional[List[Partition]] = None,
    temperature: float = 0.5,
) -> np.ndarray:
    """
    软门控整合：y = Σ w_i(x) * pred_i(x)，权重 w 由边界平滑过渡。

    使用 log10(g_bar) 边界的 softmax 作为权重。
    """
    partitions = partitions or RAR_PARTITIONS
    x = np.asarray(x, dtype=np.float64).ravel()
    lg = np.log10(x + 1e-14)

    preds = []
    for p in partitions:
        if p.id not in partition_results:
            preds.append(np.copy(x))  # fallback to x
            continue
        pred = _eval_partition_formula(x, partition_results[p.id])
        preds.append(pred)

    # 线性插值权重：按 log10(g_bar) 落在低/中/高
    y = np.zeros_like(x)
    for i in range(len(x)):
        lgi = lg[i]
        if lgi < 2.5:
            w_low, w_mid, w_high = 1.0, 0.0, 0.0
        elif lgi < 3.5:
            t = (lgi - 2.5) / 1.0
            w_low, w_mid, w_high = 1 - t, t, 0.0
        else:
            t = min(1.0, (lgi - 3.5) / 1.5)
            w_low, w_mid, w_high = 0.0, 1 - t, t

        p0 = preds[0][i] if len(preds) > 0 else x[i]
        p1 = preds[1][i] if len(preds) > 1 else x[i]
        p2 = preds[2][i] if len(preds) > 2 else x[i]
        y[i] = w_low * p0 + w_mid * p1 + w_high * p2

    return y


def _eval_partition_formula(x: np.ndarray, res: Dict[str, Any]) -> np.ndarray:
    """根据 partition result 计算预测。"""
    formula = res.get("formula", "x")
    g0 = res.get("g0")
    x = np.asarray(x, dtype=np.float64).ravel()

    if "sqrt" in formula and g0 is not None:
        return np.sqrt(g0 * np.maximum(x, 1e-14))
    if formula.strip() == "x":
        return x
    if "exp" in formula and g0 is not None:
        denom = 1.0 - np.exp(-np.sqrt(np.maximum(x / (g0 + 1e-14), 1e-14)))
        return x / np.maximum(denom, 0.01)
    # fallback
    return x
