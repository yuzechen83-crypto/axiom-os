"""
组合 A：自由能原理 + 系统 1/2 的动态扰动强度调度

- 自由能原理 (Friston): soft_activity ≈ 预测误差，误差大时需先验/联想修正
- 系统 1/2 (Kahneman): 主模型=系统2(推理)，扰动=系统1(直觉)；不确定时更依赖系统1

α(activity) = α_base * f(activity): activity 高 → α 大 → 更信任海马体联想
"""

import math
from typing import Optional


def compute_alpha_free_energy(
    activity: float,
    alpha_base: float,
    theta: float = 0.1,
    k: float = 10.0,
    alpha_max: Optional[float] = None,
) -> float:
    """
    自由能驱动：预测误差(activity)大 → α 增大，更依赖先验(扰动)。

    α = α_base * sigmoid(k * (activity - θ))
    - activity < θ: α 小，主模型可靠
    - activity > θ: α 增大，信任联想修正

    Args:
        activity: soft_activity (预测误差代理)
        alpha_base: 基准扰动强度
        theta: 活动阈值，超过则开始增强 α
        k: sigmoid 陡峭度
        alpha_max: 最大 α，None 则无上限
    """
    sig = 1.0 / (1.0 + math.exp(-k * (activity - theta)))
    alpha = alpha_base * sig
    if alpha_max is not None:
        alpha = min(alpha, alpha_max)
    return float(alpha)


def compute_alpha_system12(
    activity: float,
    alpha_base: float,
    theta_low: float = 0.02,
    theta_high: float = 0.2,
    alpha_max: float = 0.3,
) -> float:
    """
    系统 1/2 平衡：不确定性区间内线性插值。

    - activity < θ_low: 系统2 可靠，α 小
    - θ_low ≤ activity ≤ θ_high: 线性增加，系统1 逐渐介入
    - activity > θ_high: 系统2 不可靠，α = alpha_max

    Args:
        activity: soft_activity
        alpha_base: 基准 (用于 θ_low 处)
        theta_low: 低阈值
        theta_high: 高阈值
        alpha_max: 高不确定性时的最大 α
    """
    if activity <= theta_low:
        return alpha_base * 0.2  # 低 activity 时弱扰动
    if activity >= theta_high:
        return alpha_max
    t = (activity - theta_low) / (theta_high - theta_low)
    return alpha_base * 0.2 + t * (alpha_max - alpha_base * 0.2)


def compute_alpha_hybrid(
    activity: float,
    alpha_base: float,
    theta: float = 0.12,
    k: float = 8.0,
    alpha_max: float = 0.15,
) -> float:
    """
    混合策略：sigmoid 平滑 + 上限。
    组合 A 的默认实现。保守参数：高 activity 时适度增强，避免扰动过强。
    """
    return compute_alpha_free_energy(
        activity, alpha_base, theta=theta, k=k, alpha_max=alpha_max
    )
