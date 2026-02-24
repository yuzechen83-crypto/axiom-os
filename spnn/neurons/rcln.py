"""
RCLN - Residual Coupling Link Neurons
残差耦合链接神经元
h_RCLN = F_soft(h_i, h_j; θ_new) + λ_res(t) · P[F_hard(P^{-1}(h_i, h_j); θ_old)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable

from ..core.constants import EPSILON


class FHard(nn.Module):
    """
    硬约束内核 F_hard
    F_hard = (1/3)[D + M + A + ε]
    D: 偏差监控, M: 语义映射, A: 语义锚点
    """

    def __init__(
        self,
        dim: int,
        xi: float = 1.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.dim = dim
        self.xi = xi
        self.epsilon = epsilon
        self.theta_old = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.C_hat = nn.Parameter(torch.randn(dim) * 0.01)
        self.S_hat = nn.Parameter(torch.randn(dim) * 0.01)
        self.W_T = nn.Parameter(torch.randn(dim, dim) * 0.01)  # 语义关联记忆

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        l_i: Optional[torch.Tensor] = None,
        l_j: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        D: exp(-(||C_old(h_i)-Ĉ|| + ||S_old(h_j)-Ŝ||) / ε)
        A: T(l_i)^T W_T T(l_j)  (语义锚点，连接海马体)
        """
        C_old = h_i @ self.theta_old
        S_old = h_j @ self.theta_old
        D = torch.exp(-(
            torch.norm(C_old - self.C_hat, dim=-1) +
            torch.norm(S_old - self.S_hat, dim=-1)
        ) / (self.epsilon + 1e-8))

        if l_i is not None and l_j is not None:
            T_i = l_i if l_i.dim() == 2 else l_i.unsqueeze(0)
            T_j = l_j if l_j.dim() == 2 else l_j.unsqueeze(0)
            A = (T_i @ self.W_T @ T_j.T).diag()
        else:
            A = (h_i @ self.W_T @ h_j.T).diag()
            if A.dim() == 0:
                A = A.unsqueeze(0)

        M = torch.sum(h_i * h_j, dim=-1)
        out = (D + M + A + self.epsilon) / 3.0
        return out.unsqueeze(-1) if out.dim() == 1 else out


class FSoft(nn.Module):
    """
    软链接外壳 F_soft
    F_soft = Attn(h_i, h_j; θ) · R(h_i, h_j, S; θ)
    R: 动态路由协议（主脑决策）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.w_gamma = nn.Parameter(torch.randn(dim + 1) * 0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        l_i: Optional[torch.Tensor] = None,
        retrieve_fn: Optional[Callable] = None,
        z_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Attn · R, 其中 R = γ·h_i + (1-γ)·Retrieve(l_i, M_H)
        γ = σ(w_γ^T [z_scale; Score(s_i, s_j)])
        """
        B = h_i.shape[0]
        q = self.q_proj(h_i).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_j).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_j).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        context = (attn @ v).transpose(1, 2).contiguous().view(B, -1)

        # Dynamic routing
        score = torch.sum(h_i * h_j, dim=-1, keepdim=True)
        z_scale = z_scale if z_scale is not None else torch.ones(B, 1, device=h_i.device)
        gamma_input = torch.cat([z_scale, score], dim=-1)
        gamma = torch.sigmoid(gamma_input @ self.w_gamma[:gamma_input.shape[-1]])

        if retrieve_fn is not None and l_i is not None:
            retrieved = retrieve_fn(l_i)
            retrieved = retrieved if retrieved.shape == h_i.shape else retrieved.expand_as(h_i)
            R = gamma * h_i + (1 - gamma) * retrieved
        else:
            R = h_i

        out = context * R
        return self.out_proj(out)


class PhysicalMapping(nn.Module):
    """
    物理映射算子 P(h) = M_phys · h
    M_phys = diag(∏_q (ℓ_q/ℓ_{q,0})^{d_{qk}})
    """

    def __init__(self, dim: int, scale_dim: int = 5):
        super().__init__()
        self.dim = dim
        self.scale_dim = scale_dim
        self.M_phys = nn.Parameter(torch.ones(dim))

    def forward(self, h: torch.Tensor, scale_factors: Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale_factors is not None:
            M = self.M_phys * scale_factors.expand_as(self.M_phys)
        else:
            M = self.M_phys
        return h * M


class RCLN(nn.Module):
    """
    Residual Coupling Link Neuron
    h_RCLN = F_soft(h_i, h_j; θ_new) + λ_res(t) · P[F_hard(P^{-1}(h_i, h_j); θ_old)]
    """

    def __init__(
        self,
        dim: int,
        lambda_res_init: float = 0.5,
        lambda_res_decay: float = 0.99,
        num_heads: int = 4,
        xi: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.lambda_res_init = lambda_res_init
        self.lambda_res_decay = lambda_res_decay
        self._lambda_res = lambda_res_init

        self.f_hard = FHard(dim, xi=xi)
        self.f_soft = FSoft(dim, num_heads=num_heads)
        self.physical_mapping = PhysicalMapping(dim)

        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def set_lambda_res(self, t: float) -> None:
        """主脑调度：λ_res(t) 随训练阶段衰减"""
        self._lambda_res = self.lambda_res_init * (self.lambda_res_decay ** t)

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        l_i: Optional[torch.Tensor] = None,
        l_j: Optional[torch.Tensor] = None,
        retrieve_fn: Optional[Callable] = None,
        z_scale: Optional[torch.Tensor] = None,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        RCLN 前向传播
        """
        # F_soft
        f_soft_out = self.f_soft(h_i, h_j, l_i, retrieve_fn, z_scale)

        # P^{-1} 近似 (identity when scale=1)
        h_i_inv = h_i / (self.physical_mapping.M_phys + EPSILON)
        h_j_inv = h_j / (self.physical_mapping.M_phys + EPSILON)

        # F_hard
        f_hard_out = self.f_hard(h_i_inv, h_j_inv, l_i, l_j)

        # Expand f_hard for residual add (dim matching)
        if f_hard_out.dim() == 2 and f_hard_out.shape[-1] == 1:
            f_hard_out = f_hard_out.expand(-1, self.dim)
        elif f_hard_out.dim() == 1:
            f_hard_out = f_hard_out.unsqueeze(-1).expand(-1, self.dim)

        # P[F_hard]
        f_hard_mapped = self.physical_mapping(f_hard_out, scale_factors)

        # Residual coupling
        h_coupled = f_soft_out + self._lambda_res * f_hard_mapped
        return self.norm(self.proj(h_coupled))
