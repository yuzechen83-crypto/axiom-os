"""
Physical MoE - 物理混合专家架构
E = {E_1^流体, E_2^电磁, ..., E_K^跨尺度耦合}
基于物理语义的路由（主脑决策）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Callable

from ..core.constants import c


class PhysicalMoE(nn.Module):
    """
    Route(x) = Top-k(Softmax(W_r · [PhysicalSignature(x); TaskContext; ResourceState]))
    """

    def __init__(
        self,
        experts: List[nn.Module],
        in_dim: int,
        expert_dim: Optional[int] = None,
        top_k: int = 2,
        signature_dim: int = 32,
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.top_k = min(top_k, len(experts))
        self.signature_dim = signature_dim
        self.expert_dim = expert_dim or (experts[0](torch.zeros(1, in_dim)).shape[-1] if experts else in_dim)
        self.router = nn.Sequential(
            nn.Linear(in_dim + signature_dim + 3, len(experts) * 2),  # +3 for task/resource
            nn.GELU(),
            nn.Linear(len(experts) * 2, len(experts)),
        )

    def forward(
        self,
        x: torch.Tensor,
        physical_signature: Optional[torch.Tensor] = None,
        task_context: Optional[torch.Tensor] = None,
        resource_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        sig = physical_signature if physical_signature is not None else torch.zeros(B, self.signature_dim, device=x.device)
        tc = task_context if task_context is not None else torch.ones(B, 1, device=x.device)
        rs = resource_state if resource_state is not None else torch.ones(B, 2, device=x.device)
        combined = torch.cat([x, sig, tc, rs], dim=-1)
        logits = self.router(combined)
        weights = torch.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(weights, self.top_k, dim=-1)

        out_dim = self.expert_dim
        out = torch.zeros(B, out_dim, device=x.device, dtype=x.dtype)

        for i in range(self.top_k):
            idx = top_idx[:, i]
            w = top_w[:, i:i+1]
            for j, expert in enumerate(self.experts):
                mask = (idx == j)
                if mask.any():
                    ex_out = expert(x[mask])
                    out[mask] = out[mask] + w[mask] * ex_out
        return out
