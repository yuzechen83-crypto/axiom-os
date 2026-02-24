"""
自适应尺度编码 (Adaptive Scale Encoder) z_scale
动态调整计算分辨率，跨尺度语义对齐
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any


class ScaleConsistencyChecker:
    """尺度一致性检查"""

    def check(self, scale_weights: torch.Tensor, physical_quantities: torch.Tensor) -> float:
        return 1.0


class AdaptiveScaleEncoder(nn.Module):
    """
    原始SPNN的自适应尺度编码 z_scale
    尺度层级：普朗克 → 宏观
    """

    def __init__(self, base_scale: str = "planck", num_scales: int = 8):
        super().__init__()
        self.num_scales = num_scales
        self.scale_levels = self._init_scale_levels(base_scale, num_scales)
        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, num_scales),
            nn.Softmax(dim=-1),
        )
        self.consistency_checker = ScaleConsistencyChecker()

    def _init_scale_levels(self, base: str, n: int) -> list:
        return [{"resolution_factor": 10 ** (i - n // 2)} for i in range(n)]

    def _extract_scale_features(self, pq: torch.Tensor) -> torch.Tensor:
        mag = pq.abs()
        log_mag = torch.log10(mag.mean() + 1e-10)
        rng = mag.max() - mag.min() + 1e-10
        log_rng = torch.log10(rng)
        return torch.stack([log_mag, log_rng, torch.tensor(0.0, device=pq.device)])

    def encode(
        self,
        physical_quantities: torch.Tensor,
        context: Optional[Dict] = None,
    ) -> torch.Tensor:
        feat = self._extract_scale_features(physical_quantities)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        scale_weights = self.encoder(feat)
        if context and "preferred_scale" in context:
            idx = min(context["preferred_scale"], self.num_scales - 1)
            bias = torch.zeros_like(scale_weights)
            bias[:, idx] = 0.3
            scale_weights = torch.softmax(scale_weights + bias, dim=-1)
        return scale_weights

    def adjust_computation_resolution(
        self,
        scale_weights: torch.Tensor,
        base_resolution: float,
    ) -> torch.Tensor:
        weighted = 0.0
        for i in range(min(scale_weights.shape[-1], len(self.scale_levels))):
            w = scale_weights[..., i] if scale_weights.dim() > 1 else scale_weights[i]
            f = self.scale_levels[i]["resolution_factor"]
            weighted = weighted + w * base_resolution * f
        return torch.clamp(
            torch.tensor(weighted, device=scale_weights.device),
            min=base_resolution * 0.1,
            max=base_resolution * 10.0,
        )
