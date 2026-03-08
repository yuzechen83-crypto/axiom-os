"""
SPNN-Evo 严肃教练
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

对 (x, y_pred) 打分，用于方案 A：L_total = L_data + λ_coach * (1 - coach_score)

检查项：
1. 有限性 (finite)
2. 物理范围：风速 |V| < V_max, u,v 在合理区间
3. 量纲语义：y 为速度 [L/T]
4. 可选：spnn_evo UPI 因果性（若可导入）
"""

import numpy as np
import torch
from typing import Union, Optional

# 湍流风速合理范围 (m/s)
V_MAX = 50.0
U_V_RANGE = (-30.0, 30.0)


def coach_score(
    x: Union[np.ndarray, "np.ndarray"],
    y_pred: Union[np.ndarray, "np.ndarray"],
    domain: str = "fluids",
) -> float:
    """
    单样本或批量打分，返回 [0, 1]。
    x: (4,) 或 (N, 4) = (t, x, y, z)
    y_pred: (2,) 或 (N, 2) = (u, v) m/s
    """
    y = np.asarray(y_pred, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    scores = _score_batch(y, domain)
    return float(np.mean(scores))


def coach_score_batch(
    x: Union[np.ndarray, "np.ndarray"],
    y_pred: Union[np.ndarray, "np.ndarray"],
    domain: str = "fluids",
) -> np.ndarray:
    """
    批量打分，返回 (N,) 每个样本的 score。
    """
    y = np.asarray(y_pred, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    return _score_batch(y, domain)


def _score_batch(y: np.ndarray, domain: str) -> np.ndarray:
    """内部：对 y (N, 2) 逐样本打分"""
    n = y.shape[0]
    scores = np.ones(n, dtype=np.float64)

    # 1. 有限性
    finite = np.all(np.isfinite(y), axis=1)
    scores[~finite] = 0.0

    # 2. 物理范围：u, v 在 [-30, 30] m/s
    in_range = np.all((y >= U_V_RANGE[0]) & (y <= U_V_RANGE[1]), axis=1)
    scores[~in_range] *= 0.5

    # 3. 风速 |V| < V_max
    v_mag = np.sqrt(np.sum(y ** 2, axis=1))
    wind_ok = v_mag < V_MAX
    scores[~wind_ok] *= 0.3

    # 4. 软约束：|V| 过大但未超限时轻微扣分
    # 例如 |V| > 20 时 score *= 0.9
    soft_penalty = np.where(v_mag > 20, 0.95, 1.0)
    scores *= soft_penalty

    return np.clip(scores, 0.0, 1.0)


def coach_loss_torch(y_pred: torch.Tensor, domain: str = "fluids") -> torch.Tensor:
    """
    可微分的 coach 惩罚项，用于 L_total = L_data + λ_coach * coach_loss_torch(y_pred)。
    返回标量，值越小越好（0 表示全部符合物理约束）。
    """
    u, v = y_pred[:, 0], y_pred[:, 1]
    v_mag = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
    # 软惩罚：超出范围部分
    p_u = torch.relu(torch.abs(u) - U_V_RANGE[1]).mean() / U_V_RANGE[1]
    p_v = torch.relu(torch.abs(v) - U_V_RANGE[1]).mean() / U_V_RANGE[1]
    p_wind = torch.relu(v_mag - V_MAX).mean() / V_MAX
    penalty = (p_u + p_v + p_wind) / 3
    return torch.clamp(penalty, 0.0, 1.0)
