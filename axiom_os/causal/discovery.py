"""
约束因果发现：基于条件独立 + 可选物理约束边集过滤。
不依赖 dowhy/cdt，仅用 numpy/scipy。
"""

from typing import List, Tuple, Optional, Set
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _pearson_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    if not HAS_SCIPY or x.shape[0] < 3:
        return 1.0
    r, p = stats.pearsonr(x, y)
    return float(p)


def _partial_corr_indep(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, alpha: float = 0.05
) -> bool:
    """True if x _||_ y | z (conditional independence at level alpha)."""
    if not HAS_SCIPY or x.shape[0] < 4:
        return False
    # Partial correlation: r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2)) + 1e-12
    r_xy_z = (r_xy - r_xz * r_yz) / denom
    n = x.shape[0]
    t = r_xy_z * np.sqrt(n - 3) / np.sqrt(1 - r_xy_z**2 + 1e-12)
    p = 2 * (1 - stats.t.cdf(abs(t), n - 3))
    return p > alpha


def discover_causal_graph(
    X: np.ndarray,
    var_names: Optional[List[str]] = None,
    allowed_edges: Optional[List[Tuple[int, int]]] = None,
    alpha: float = 0.05,
    use_cond_indep: bool = True,
    max_cond_size: int = 1,
) -> np.ndarray:
    """
    从观测数据发现因果图（邻接矩阵）。
    - X: (n_samples, n_vars)
    - allowed_edges: 可选，(j, i) 表示 j->i 允许；若 None 则允许全连接。
    - alpha: 条件独立检验显著性水平。
    - use_cond_indep: 若 True，用单变量条件独立剔除边（PC 风格）。
    - max_cond_size: 条件集最大大小（1 即只做 X_i _||_ X_j | X_k）。
    返回: adj (n, n)，adj[j,i]=1 表示存在 j->i 边。
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n = X.shape
    if var_names is None:
        var_names = [f"x{i}" for i in range(n)]

    if allowed_edges is not None:
        allowed_set = set(allowed_edges)
    else:
        allowed_set = set((j, i) for j in range(n) for i in range(n))

    # 初始候选：所有 allowed 边
    adj = np.zeros((n, n))
    for (j, i) in allowed_set:
        if 0 <= j < n and 0 <= i < n:
            adj[j, i] = 1

    # 边际独立：若 X_i 与 X_j 独立则删边
    for i in range(n):
        for j in range(n):
            if adj[j, i] == 0:
                continue
            p = _pearson_pvalue(X[:, i], X[:, j])
            if p > alpha:
                adj[j, i] = 0

    if use_cond_indep and max_cond_size >= 1 and n >= 3 and HAS_SCIPY:
        for i in range(n):
            for j in range(n):
                if adj[j, i] == 0:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if _partial_corr_indep(X[:, i], X[:, j], X[:, k], alpha=alpha):
                        adj[j, i] = 0
                        break

    return adj


def adjacency_to_edges(adj: np.ndarray) -> List[Tuple[int, int]]:
    """邻接矩阵 -> (source, target) 边列表。"""
    n = adj.shape[0]
    return [(j, i) for j in range(n) for i in range(n) if adj[j, i] != 0]
