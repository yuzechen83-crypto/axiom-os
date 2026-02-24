"""
元轴丛场 - 纤维
Fiber: 在基空间某点处的公式丛（主公式 + 扰动修正 + 可忽略残差）
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any

import numpy as np

from .axes import BasePoint, Regime, ResidualRole


@dataclass
class FormulaEntry:
    """单条公式条目"""
    formula_id: str
    formula: str
    callable: Callable
    output_dim: int
    residual_role: ResidualRole
    regime: Regime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fiber:
    """
    纤维：在基空间某点处的公式丛
    包含：主公式 + 扰动修正列表 + 可忽略残差标记
    """
    base_point: BasePoint
    principal: Optional[FormulaEntry] = None
    perturbations: List[FormulaEntry] = field(default_factory=list)
    negligible_residuals: List[str] = field(default_factory=list)
    composite_rule: Optional[str] = None  # "add" | "weighted" | "cascade"

    def eval(self, x: np.ndarray, alpha_pert: float = 0.1) -> np.ndarray:
        """求值：主 + Σ alpha * 扰动"""
        if self.principal is None:
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(1, -1)
            return np.zeros((x_arr.shape[0], 1), dtype=np.float64)

        if hasattr(x, "detach"):
            x_np = x.detach().cpu().numpy()
        elif hasattr(x, "values"):
            v = x.values
            x_np = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        else:
            x_np = np.asarray(x, dtype=np.float64)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        try:
            y = np.asarray(self.principal.callable(x), dtype=np.float64)
        except Exception:
            y = np.zeros((x_np.shape[0], self.principal.output_dim), dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for p in self.perturbations:
            try:
                pert = np.asarray(p.callable(x), dtype=np.float64)
                if pert.ndim == 1:
                    pert = pert.reshape(-1, 1)
                if pert.shape == y.shape:
                    y = y + alpha_pert * pert
            except Exception:
                continue

        return y

    def eval_perturbation_only(self, x: np.ndarray, alpha_pert: float = 0.1) -> np.ndarray:
        """仅求扰动项之和（用于 eval_perturbation 接口）"""
        if not self.perturbations:
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(1, -1)
            return np.zeros((x_arr.shape[0], 1), dtype=np.float64)

        if hasattr(x, "detach"):
            x_np = x.detach().cpu().numpy()
        elif hasattr(x, "values"):
            v = x.values
            x_np = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        else:
            x_np = np.asarray(x, dtype=np.float64)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        out_dim = self.perturbations[0].output_dim
        y = np.zeros((x_np.shape[0], out_dim), dtype=np.float64)
        for p in self.perturbations:
            try:
                pert = np.asarray(p.callable(x), dtype=np.float64)
                if pert.ndim == 1:
                    pert = pert.reshape(-1, 1)
                if pert.shape == y.shape:
                    y = y + alpha_pert * pert
            except Exception:
                continue
        return y
