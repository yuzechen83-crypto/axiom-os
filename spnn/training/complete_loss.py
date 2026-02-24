"""
SPNN-Opt-Rev5 完整损失函数体系
L_total = L_pred + λ1·L_DAS + λ2·L_BC + λ3·L_entropy + λ4·L_conservation + λ5·L_reg
自适应权重: λ_i(t) = λ_i0·exp(-t/T_i) + λ_i_min
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from ..core.calibration import apply_das
from ..core.constants import EPSILON


@dataclass
class LossWeights:
    lambda_pred: float = 1.0
    lambda_das: float = 0.1
    lambda_bc: float = 0.1
    lambda_entropy: float = 0.05
    lambda_conservation: float = 0.1
    lambda_reg: float = 0.01
    T_decay: float = 1000.0
    lambda_min: float = 0.01


def adaptive_weight(lambda_0: float, t: float, T: float, lambda_min: float) -> float:
    """λ(t) = λ_0·exp(-t/T) + λ_min"""
    return lambda_0 * torch.exp(torch.tensor(-t / T)).item() + lambda_min


class SPNNCompleteLoss(nn.Module):
    """
    完整损失: 预测 + DAS势能 + 边界 + 熵 + 守恒 + 正则化
    """

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        boundary_flux_fn: Optional[Callable] = None,
        entropy_enforcer: Optional[Any] = None,
        conservation_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.weights = weights or LossWeights()
        self.boundary_flux_fn = boundary_flux_fn
        self.entropy_enforcer = entropy_enforcer
        self.conservation_fn = conservation_fn

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        soft_out: Optional[torch.Tensor] = None,
        hard_out: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        step: float = 0.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        w = self.weights

        L_pred = nn.functional.mse_loss(pred, target)
        losses["pred"] = L_pred

        L_das = torch.tensor(0.0, device=pred.device)
        if soft_out is not None and hard_out is not None:
            _, potential = apply_das(soft_out, hard_out)
            L_das = potential.mean()
        losses["das"] = L_das

        L_bc = torch.tensor(0.0, device=pred.device)
        if self.boundary_flux_fn is not None and "bc_data" in kwargs:
            L_bc = self.boundary_flux_fn(**kwargs["bc_data"])
        losses["bc"] = L_bc

        L_entropy = torch.tensor(0.0, device=pred.device)
        if self.entropy_enforcer is not None and state is not None and "dynamics" in kwargs:
            lam = adaptive_weight(
                w.lambda_entropy, step, w.T_decay, w.lambda_min
            )
            L_entropy = self.entropy_enforcer.entropy_loss(
                kwargs["dynamics"], state, lam
            )
        losses["entropy"] = L_entropy

        L_conservation = torch.tensor(0.0, device=pred.device)
        if self.conservation_fn is not None and "conservation_data" in kwargs:
            L_conservation = self.conservation_fn(**kwargs["conservation_data"])
        losses["conservation"] = L_conservation

        L_reg = pred.pow(2).mean()
        losses["reg"] = L_reg

        lam_pred = adaptive_weight(w.lambda_pred, step, w.T_decay, 0.5)
        lam_das = adaptive_weight(w.lambda_das, step, w.T_decay, w.lambda_min)
        lam_bc = adaptive_weight(w.lambda_bc, step, w.T_decay, w.lambda_min)
        lam_ent = adaptive_weight(w.lambda_entropy, step, w.T_decay, w.lambda_min)
        lam_cons = adaptive_weight(w.lambda_conservation, step, w.T_decay, w.lambda_min)
        lam_r = adaptive_weight(w.lambda_reg, step, w.T_decay, w.lambda_min)

        L_total = (
            lam_pred * L_pred +
            lam_das * L_das +
            lam_bc * L_bc +
            lam_ent * L_entropy +
            lam_cons * L_conservation +
            lam_r * L_reg
        )
        losses["total"] = L_total
        return losses
