"""
Axiom 智能分区模块 - 分区学习（Curriculum）

按 complexity 升序学习各分区，单一场景先学，再整合。
"""

from typing import Callable, Dict, Any, List, Optional, Tuple
import numpy as np

from .partition import Partition, get_partitions_curriculum_order, RAR_PARTITIONS


def learn_partition(
    partition: Partition,
    X: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray, Partition], Dict[str, Any]],
    min_samples: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    在单一分区内学习。

    Args:
        partition: 分区定义
        X: 输入 (n, d)，RAR 下 X 为 g_bar 或 (log_g_bar, g_bar)
        y: 输出 (n,)
        fit_fn: (X_part, y_part, partition) -> {formula, g0, r2, ...}
        min_samples: 最少样本数，不足则跳过

    Returns:
        该分区的学习结果，或 None（样本不足）
    """
    idx = partition.indices(X, y)
    if len(idx) < min_samples:
        return None
    X_part = np.asarray(X)[idx] if X.ndim > 1 else np.asarray(X)[idx]
    if X_part.ndim == 1:
        X_part = X_part.reshape(-1, 1)
    y_part = np.asarray(y)[idx]
    try:
        result = fit_fn(X_part, y_part, partition)
        result["partition_id"] = partition.id
        result["n_samples"] = len(idx)
        return result
    except Exception as e:
        return {"partition_id": partition.id, "error": str(e), "n_samples": len(idx)}


def _schedule_partitions(
    partitions: List[Partition],
    X: np.ndarray,
    y: np.ndarray,
    schedule: str,
) -> List[Partition]:
    """
    动态调度分区顺序。
    schedule: "complexity" | "r2_asc" | "samples_desc"
    - complexity: 按 complexity 升序（原有）
    - r2_asc: 先学 R² 差的分区（需预跑一轮得 R²）
    - samples_desc: 先学样本多的分区
    """
    if schedule == "complexity":
        return sorted(partitions, key=lambda p: p.complexity)

    if schedule == "samples_desc":
        def key(p):
            n = len(p.indices(X, y))
            return -n  # 降序
        return sorted(partitions, key=key)

    if schedule == "r2_asc":
        # 需预跑得 R²，此处简化：先按 complexity，后续可扩展为两阶段
        return sorted(partitions, key=lambda p: p.complexity)

    return sorted(partitions, key=lambda p: p.complexity)


def learn_curriculum(
    X: np.ndarray,
    y: np.ndarray,
    partitions: Optional[List[Partition]] = None,
    fit_fn: Optional[Callable] = None,
    min_samples: int = 10,
    domain: str = "mechanics",
    schedule: str = "complexity",
) -> Dict[str, Any]:
    """
    Curriculum 学习：按 schedule 调度分区顺序，逐分区学习。

    Args:
        X: 输入
        y: 输出
        partitions: 分区列表，None 则按 domain 获取
        fit_fn: 拟合函数，None 则用默认（RAR/battery 按 domain 选择）
        min_samples: 每分区最少样本
        domain: mechanics | battery | fluids
        schedule: "complexity" | "samples_desc" | "r2_asc"

    Returns:
        {
            "partition_results": {partition_id: result},
            "order": [partition_id, ...],
            "n_total": int,
        }
    """
    from .partition import get_partitions_curriculum_order
    partitions = partitions or get_partitions_curriculum_order(domain)
    partitions = _schedule_partitions(partitions, X, y, schedule)

    if fit_fn is None:
        fit_fn = _get_fit_fn_for_domain(domain)

    partition_results = {}
    order = []
    for p in partitions:
        res = learn_partition(p, X, y, fit_fn, min_samples)
        if res is not None and "error" not in res:
            partition_results[p.id] = res
            order.append(p.id)

    return {
        "partition_results": partition_results,
        "order": order,
        "n_total": len(X),
    }


def _get_fit_fn_for_domain(domain: str) -> Callable:
    """按 domain 选择默认 fit。"""
    if domain in ("battery",):
        return _default_battery_partition_fit
    if domain in ("fluids",):
        return _default_turbulence_partition_fit
    return _default_rar_partition_fit


def _default_rar_partition_fit(
    X_part: np.ndarray,
    y_part: np.ndarray,
    partition: Partition,
) -> Dict[str, Any]:
    """
    默认 RAR 分区拟合：低区用 sqrt(g0*g_bar)，高区用 g_bar，中区用 McGaugh。
    """
    g_bar = X_part[:, 0] if X_part.ndim > 1 else X_part.ravel()
    g_bar = np.maximum(g_bar, 1e-14)
    g_obs = np.asarray(y_part, dtype=np.float64)
    pid = partition.id

    if pid == "rar_low":
        # 深 MOND: g_obs ≈ sqrt(g0 * g_bar), 固定 g0=3700
        g0 = 3700.0
        pred = np.sqrt(g0 * g_bar)
        r2 = 1 - np.sum((pred - g_obs) ** 2) / (np.sum((g_obs - g_obs.mean()) ** 2) + 1e-12)
        return {"formula": f"sqrt({g0}*x)", "g0": g0, "r2": float(r2), "regime": "deep_MOND"}

    if pid == "rar_high":
        # 牛顿: g_obs = g_bar
        pred = g_bar
        r2 = 1 - np.sum((pred - g_obs) ** 2) / (np.sum((g_obs - g_obs.mean()) ** 2) + 1e-12)
        return {"formula": "x", "g0": None, "r2": float(r2), "regime": "Newtonian"}

    if pid == "rar_mid":
        # 过渡区: McGaugh 自由拟合 g0
        try:
            from scipy.optimize import minimize
            def loss(g0):
                g0 = np.clip(g0, 100, 50000)
                denom = 1.0 - np.exp(-np.sqrt(g_bar / (g0 + 1e-14)))
                pred = g_bar / np.maximum(denom, 0.01)
                return np.mean((pred - g_obs) ** 2)
            res = minimize(loss, [3700.0], method="L-BFGS-B", bounds=[(100, 50000)])
            g0 = float(np.clip(res.x[0], 100, 50000))
            denom = 1.0 - np.exp(-np.sqrt(g_bar / (g0 + 1e-14)))
            pred = g_bar / np.maximum(denom, 0.01)
            r2 = 1 - np.sum((pred - g_obs) ** 2) / (np.sum((g_obs - g_obs.mean()) ** 2) + 1e-12)
            return {"formula": f"x/(1-exp(-sqrt(x/{g0:.0f})))", "g0": g0, "r2": float(r2), "regime": "transition"}
        except Exception as e:
            return {"formula": "x", "g0": None, "r2": 0.0, "regime": "transition", "error": str(e)}

    return {"formula": "x", "g0": None, "r2": 0.0}


def _default_battery_partition_fit(
    X_part: np.ndarray,
    y_part: np.ndarray,
    partition: Partition,
) -> Dict[str, Any]:
    """Battery 分区拟合：线性衰减。"""
    t = X_part[:, 0] if X_part.ndim > 1 else X_part.ravel()
    y = np.asarray(y_part, dtype=np.float64)
    pid = partition.id
    try:
        slope, intercept = np.polyfit(t, y, 1)
        pred = slope * t + intercept
        r2 = 1 - np.sum((pred - y) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        return {"formula": f"{slope:.4g}*t+{intercept:.4g}", "r2": float(r2), "regime": pid}
    except Exception:
        return {"formula": "1", "r2": 0.0, "regime": pid}


def _default_turbulence_partition_fit(
    X_part: np.ndarray,
    y_part: np.ndarray,
    partition: Partition,
) -> Dict[str, Any]:
    """Turbulence 分区拟合：均值预测。"""
    y = np.asarray(y_part, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1) if y.shape[1] > 1 else y[:, 0]
    pred = np.full_like(y, y.mean())
    r2 = 1 - np.sum((pred - y) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
    return {"formula": f"mean={y.mean():.4g}", "r2": float(r2), "regime": partition.id}
