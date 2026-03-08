"""
结构因果模型 (SCM)：线性 SEM + do 干预与简单反事实。
与 Discovery 输出对接，可做「若 X=j 则 Y 会如何」的推理。
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def topological_order(adj: np.ndarray) -> List[int]:
    """邻接矩阵 adj[j,i]=1 表示 j->i，返回拓扑序（先父后子）。"""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0)
    order = []
    while len(order) < n:
        found = False
        for i in range(n):
            if i in order:
                continue
            if in_deg[i] == 0:
                order.append(i)
                for j in range(n):
                    if adj[i, j]:
                        in_deg[j] -= 1
                found = True
                break
        if not found:
            break
    return order if len(order) == n else list(range(n))


class LinearSCM:
    """
    线性 SCM: X_i = sum_j B[j,i]*X_j + U_i，U 为外生。
    B 为 (n,n)，B[j,i] 表示 j->i 的系数。
    """

    def __init__(self, B: np.ndarray, var_names: Optional[List[str]] = None):
        B = np.asarray(B, dtype=np.float64)
        assert B.shape[0] == B.shape[1]
        self.B = B
        self.n = B.shape[0]
        self.var_names = var_names or [f"x{i}" for i in range(self.n)]

    def sample(self, n_samples: int, u_std: Optional[np.ndarray] = None, seed: Optional[int] = None) -> np.ndarray:
        """从外生 U 采样生成观测。U_i ~ N(0, u_std[i]^2)。"""
        if seed is not None:
            np.random.seed(seed)
        u_std = np.ones(self.n) if u_std is None else np.asarray(u_std).ravel()[: self.n]
        U = np.random.randn(n_samples, self.n) * u_std
        # X = B^T X + U => (I - B^T) X = U => X = (I - B^T)^{-1} U
        IBT = np.eye(self.n) - self.B.T
        X = np.linalg.solve(IBT, U.T).T
        return X

    def do(self, intervention: Dict[int, float], X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        干预 do(X_j = v)：断掉 j 的入边、设 X_j = v，按拓扑序重算其余变量。
        X: (n_samples, n) 当前观测；若 None 则用零向量单样本。
        """
        if X is None:
            X = np.zeros((1, self.n))
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        U = X - (X @ self.B)
        order = topological_order((self.B != 0).astype(np.float64))
        out = X.copy()
        for j, v in intervention.items():
            if 0 <= j < self.n:
                out[:, j] = v
        for i in order:
            if i in intervention:
                continue
            out[:, i] = out @ self.B[:, i] + U[:, i]
        return out

    def counterfactual(
        self,
        X_factual: np.ndarray,
        intervention: Dict[int, float],
    ) -> np.ndarray:
        """
        反事实：在观测 X_factual 下，若做了 do(intervention)，结果会如何。
        步骤：1) 由 X_factual 反推 U；2) do 干预；3) 用同一 U 前向计算。
        """
        X_factual = np.asarray(X_factual).ravel()[: self.n]
        U = X_factual - self.B.T @ X_factual
        out = X_factual.copy()
        for j, v in intervention.items():
            if 0 <= j < self.n:
                out[j] = v
        order = topological_order((self.B != 0).astype(np.float64))
        for i in order:
            if i in intervention:
                continue
            out[i] = self.B[:, i] @ out + U[i]
        return out


def fit_linear_scm_from_data(
    X: np.ndarray,
    adj: np.ndarray,
    var_names: Optional[List[str]] = None,
) -> LinearSCM:
    """
    从观测数据与已知 DAG（邻接矩阵）拟合线性 SCM 的 B。
    对每个节点 i，用其父节点 Pa(i) 回归 X_i，得到 B[j,i]。
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    B = np.zeros((n, n))
    for i in range(n):
        parents = [j for j in range(n) if adj[j, i] != 0]
        if not parents:
            B[:, i] = 0
            continue
        y = X[:, i]
        W = X[:, parents]
        W = np.column_stack([W, np.ones(X.shape[0])])
        coef, _, _, _ = np.linalg.lstsq(W, y, rcond=None)
        for k, j in enumerate(parents):
            B[j, i] = coef[k]
    return LinearSCM(B, var_names=var_names)
