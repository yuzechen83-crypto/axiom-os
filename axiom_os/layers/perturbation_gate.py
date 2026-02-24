"""
可学习扰动门控 — 直觉由训练得到

记忆作为扰动项：y_final = y_main + α · y_memory
α 由可学习门控网络输出，而非固定规则。
类比人类：基础能力 + 经验修正，何时信任经验由学习习得。
"""

import torch
import torch.nn as nn
from typing import Optional


class LearnablePerturbationGate(nn.Module):
    """
    可学习门控：输入 (x, activity, y_main, y_pert) -> α ∈ [0, alpha_max]

    直觉由训练得到：何时信任记忆扰动、信任多少，由数据驱动学习。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 32,
        alpha_max: float = 0.3,
        use_y_main: bool = True,
        use_y_pert: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha_max = alpha_max
        self.use_y_main = use_y_main
        self.use_y_pert = use_y_pert

        # 输入: x (input_dim) + activity (1) + [y_main (output_dim)] + [y_pert (output_dim)]
        gate_in = input_dim + 1
        if use_y_main:
            gate_in += output_dim
        if use_y_pert:
            gate_in += output_dim

        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        activity: torch.Tensor,
        y_main: Optional[torch.Tensor] = None,
        y_pert: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns alpha per sample, shape (B,) or (B,1).
        """
        parts = [x, activity]
        if self.use_y_main and y_main is not None:
            parts.append(y_main)
        elif self.use_y_main:
            parts.append(torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype))
        if self.use_y_pert and y_pert is not None:
            parts.append(y_pert)
        elif self.use_y_pert:
            parts.append(torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype))

        gate_in = torch.cat(parts, dim=-1)
        alpha = self.gate(gate_in) * self.alpha_max
        return alpha.squeeze(-1) if alpha.dim() > 1 else alpha
