"""
Symplectic Causal Constraints - 超级智能化 L1 物理层
Hamiltonian 辛结构因果：dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
用于 Discovery 候选公式的因果方向约束。
"""

from typing import List, Tuple, Optional, Set
import numpy as np


def get_symplectic_causal_edges(state_dim: int) -> List[Tuple[int, int]]:
    """
    Hamiltonian 因果边：(source, target) 表示 source 可因果影响 target。
    dq_i/dt = ∂H/∂p_i  =>  p_j 可影响 dq_i (通过 H)
    dp_i/dt = -∂H/∂q_i =>  q_j 可影响 dp_i (通过 H)

    约定：x = [q1, q2, ..., p1, p2, ...]，前 n/2 为 q，后 n/2 为 p。
    输出边 (j, i) 表示 x_j 可出现在 dx_i/dt 的公式中。
    """
    n = state_dim
    if n % 2 != 0:
        return []
    nq = n // 2
    edges = []
    # dq_i/dt 依赖 p (索引 nq..n-1)
    for i in range(nq):
        for j in range(nq, n):
            edges.append((j, i))
    # dp_i/dt 依赖 q, p (索引 0..n-1)
    for i in range(nq, n):
        for j in range(n):
            edges.append((j, nq + (i - nq)))
    return edges


def get_symplectic_allowed_inputs(
    output_index: int,
    state_dim: int,
) -> Set[int]:
    """
    给定输出索引（对应某个 dx_i/dt），返回允许的输入变量索引集合。
    output_index 0..nq-1: dq/dt，允许 p 的索引
    output_index nq..n-1: dp/dt，允许 q,p 的索引
    """
    n = state_dim
    if n % 2 != 0:
        return set(range(n))
    nq = n // 2
    if output_index < nq:
        return set(range(nq, n))  # dq/dt 只依赖 p
    return set(range(n))  # dp/dt 依赖 q, p


def filter_formula_by_symplectic(
    formula: str,
    output_index: int,
    state_dim: int,
    var_names: Optional[List[str]] = None,
) -> bool:
    """
    检查公式是否满足辛因果约束。
    若公式中出现了不允许的变量，返回 False。
    var_names: 如 ["x0","x1","x2","x3"] 对应 q1,q2,p1,p2；若 None 则用 x0,x1,...
    """
    allowed = get_symplectic_allowed_inputs(output_index, state_dim)
    names = var_names or [f"x{i}" for i in range(state_dim)]
    for i in range(state_dim):
        if i not in allowed:
            # 检查公式是否包含 names[i] 或 x[i] 或 x[:,i]
            forbidden = [names[i], f"x{i}", f"x_{i}", f"x[:,{i}]", f"x[:, {i}]"]
            for pat in forbidden:
                if pat in formula:
                    return False
    return True


def build_symplectic_causal_mask(state_dim: int) -> np.ndarray:
    """
    构建因果邻接矩阵 mask: (n, n)，mask[i,j]=1 表示 x_j 可影响 dx_i/dt。
    """
    n = state_dim
    mask = np.zeros((n, n), dtype=np.float64)
    edges = get_symplectic_causal_edges(n)
    for j, i in edges:
        if 0 <= i < n and 0 <= j < n:
            mask[i, j] = 1.0
    return mask
