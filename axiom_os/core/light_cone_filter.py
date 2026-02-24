"""
Light Cone Filter - 超级智能化 L1 物理层
时空因果约束：(c·Δt)² ≥ |Δr|²
用于过滤违反光锥的因果边。与 UPI assert_causality 一致。
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    from .upi import UPIState
    HAS_UPI = True
except ImportError:
    HAS_UPI = False


def check_light_cone(
    t1: float, r1: np.ndarray,
    t2: float, r2: np.ndarray,
    c: float = 1.0,
) -> bool:
    """
    检查 (t2, r2) 是否在 (t1, r1) 的因果未来光锥内。
    条件：(c·Δt)² ≥ |Δr|²，其中 Δt = t2 - t1, Δr = r2 - r1。
    """
    dt = t2 - t1
    if dt < 0:
        return False
    r1 = np.asarray(r1).ravel()
    r2 = np.asarray(r2).ravel()
    dr = r2[: min(len(r2), 3)] - r1[: min(len(r1), 3)]
    spatial_sq = np.dot(dr, dr)
    time_sep = c * dt
    return time_sep * time_sep >= spatial_sq - 1e-12


def filter_causal_edges_by_light_cone(
    spacetime_list: List[Tuple[float, np.ndarray]],
    candidate_edges: List[Tuple[int, int]],
    c: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    用光锥过滤候选因果边。
    spacetime_list[i] = (t_i, r_i) 为变量 i 的时空坐标。
    candidate_edges = [(i, j), ...] 表示 i -> j 的候选边。
    若 (t_i, r_i) 在 (t_j, r_j) 的过去光锥外，则拒绝边 i->j。
    注意：因果意味着原因在结果的过去，即 t_i < t_j 且 j 在 i 的光锥内。
    """
    valid = []
    for i, j in candidate_edges:
        if i >= len(spacetime_list) or j >= len(spacetime_list):
            valid.append((i, j))
            continue
        t_i, r_i = spacetime_list[i]
        t_j, r_j = spacetime_list[j]
        if t_i >= t_j:
            continue  # 原因必须在过去
        if check_light_cone(t_i, r_i, t_j, r_j, c):
            valid.append((i, j))
    return valid


def spacetime_from_timeseries(
    times: np.ndarray,
    n_vars: int,
    spatial: Optional[np.ndarray] = None,
) -> List[Tuple[float, np.ndarray]]:
    """
    从时间序列构造 spacetime 列表。
    times: (n_samples,) 时间戳
    n_vars: 变量数
    spatial: (n_samples, n_vars, 3) 可选空间坐标；若 None 则 r=0（仅时间因果）
    返回：每个变量的 (t_avg, r_avg) 或按样本索引的列表。
    简化：返回 [(t_k, r_k) for k in range(n_vars)]，其中 t_k 为第一个样本时间，r_k=0。
    """
    t0 = float(times[0]) if len(times) > 0 else 0.0
    result = []
    for k in range(n_vars):
        r = np.zeros(3)
        if spatial is not None and spatial.ndim >= 2:
            r = np.asarray(spatial[0, min(k, spatial.shape[1] - 1), :3]).ravel()
        result.append((t0, r))
    return result


if HAS_UPI:
    def assert_causality_upi(here: UPIState, other: UPIState, c: float = 1.0) -> None:
        """调用 UPI 的 assert_causality"""
        here.assert_causality(other, c=c)
