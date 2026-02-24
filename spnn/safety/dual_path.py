"""
Dual-Path Validation 双路径验证
安全路径 + 探索路径 → 主脑仲裁
"""

import numpy as np
import torch
from typing import Optional, Callable, Dict, Any, Tuple
from enum import Enum

from ..orchestrator.axiom_os import AxiomOS, RouteDecision


def SafeValidate(x: torch.Tensor, bounds: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    安全路径：严格约束验证
    """
    bounds = bounds or (-1e6, 1e6)
    x = torch.clamp(x, bounds[0], bounds[1])
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


def ExploreValidate(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    探索路径：允许探索性输出
    """
    if not torch.isfinite(x).all():
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return x * scale


class DualPathValidator:
    """
    双路径验证完整流程
    1. 检测异常 A → (D, N, C)
    2. 主脑路由 Route(A)
    3. 并行: SafeValidate, ExploreValidate
    4. 仲裁 Final = Φ_brain(r_s, r_e, 历史经验)
    """

    def __init__(
        self,
        orchestrator: Optional[AxiomOS] = None,
        safe_bounds: Tuple[float, float] = (-1e6, 1e6),
        explore_scale: float = 1.0,
    ):
        self.orchestrator = orchestrator or AxiomOS()
        self.safe_bounds = safe_bounds
        self.explore_scale = explore_scale

    def detect_anomaly(self, x: torch.Tensor) -> Dict[str, float]:
        """
        异常检测: A → (D, N, C)
        D: detection_score, N: novelty, C: confidence
        """
        D = 1.0 - float(torch.isnan(x).any() or torch.isinf(x).any())
        std_val = torch.std(x).item() if x.numel() > 1 else 0.0
        N = float(std_val / (torch.mean(torch.abs(x)).item() + 1e-8))
        N = min(1.0, N)
        C = 1.0 - float(torch.isnan(x).any())
        return {"detection_score": D, "novelty": N, "confidence": C}

    def validate(
        self,
        x: torch.Tensor,
        context: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        双路径验证 → 仲裁输出
        """
        context = context or {}
        anomaly = self.detect_anomaly(x)
        context.update(anomaly)

        route = self.orchestrator.route_anomaly(x, context)

        r_safe = SafeValidate(x, self.safe_bounds)
        r_explore = ExploreValidate(x, self.explore_scale)

        if route == RouteDecision.SAFE:
            return r_safe, {"route": "safe", **context}
        if route == RouteDecision.EXPLORE:
            return r_explore, {"route": "explore", **context}

        # HYBRID: 仲裁
        r_s_np = r_safe.detach().cpu().numpy()
        r_e_np = r_explore.detach().cpu().numpy()
        final_np = self.orchestrator.arbitrate(r_s_np, r_e_np, context)
        return torch.as_tensor(final_np, dtype=x.dtype, device=x.device), {"route": "hybrid", **context}
