"""
SPNN Multi-Objective Loss
L_SPNN = α·L_phys + β·L_pred + γ·L_reg + δ·L_memory + ε·L_scale
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from ..core.constants import EPSILON


class SPNNLoss(nn.Module):
    """
    多目标损失函数
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.4,
        gamma: float = 0.1,
        delta: float = 0.1,
        epsilon: float = 0.1,
        scale_system=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eps = epsilon
        self.scale_system = scale_system

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        Q_scale: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        memory_items: Optional[list] = None,
        scale_drift: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # L_pred: 预测损失
        l_pred = nn.functional.mse_loss(pred, target)
        losses["pred"] = l_pred

        # L_phys: 物理损失 (归一化误差)
        Q_scale = Q_scale if Q_scale is not None else torch.abs(target) + EPSILON
        l_phys = torch.mean(((pred - target) / Q_scale) ** 2)
        losses["phys"] = l_phys

        # L_reg: 正则化
        l_reg = torch.tensor(0.0, device=pred.device)
        if hidden is not None:
            l_reg = torch.mean(hidden ** 2)
        losses["reg"] = l_reg

        # L_memory: 记忆损失 (简化)
        l_memory = torch.tensor(0.0, device=pred.device)
        if memory_items is not None:
            for m in memory_items[:5]:  # 限制计算量
                if hasattr(m, 'value'):
                    v = torch.as_tensor(m.value, device=pred.device)
                    if hidden is not None and v.shape == hidden.shape[-1:]:
                        l_memory = l_memory - 0.1 * torch.log(torch.tensor(m.confidence + EPSILON, device=pred.device))
        losses["memory"] = l_memory

        # L_scale: 标尺损失
        l_scale = torch.tensor(0.0, device=pred.device)
        if scale_drift is not None:
            l_scale = torch.tensor(scale_drift, device=pred.device)
        losses["scale"] = l_scale

        total = (
            self.alpha * losses["phys"] +
            self.beta * losses["pred"] +
            self.gamma * losses["reg"] +
            self.delta * losses["memory"] +
            self.eps * losses["scale"]
        )
        losses["total"] = total
        return losses


class AntiForgettingLoss(nn.Module):
    """
    灾难性遗忘防护
    L_anti-forgetting = Σ ||f_current(x) - y_old||^2 + λ·KL(p_old || p_current)
    """

    def __init__(self, lambda_kl: float = 0.1):
        super().__init__()
        self.lambda_kl = lambda_kl

    def forward(
        self,
        current_logits: torch.Tensor,
        old_targets: torch.Tensor,
        old_probs: Optional[torch.Tensor] = None,
        current_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l_mse = nn.functional.mse_loss(current_logits, old_targets)
        if old_probs is not None and current_probs is not None:
            l_kl = nn.functional.kl_div(
                (current_probs + 1e-8).log(),
                old_probs + 1e-8,
                reduction="batchmean",
            )
            return l_mse + self.lambda_kl * l_kl
        return l_mse
