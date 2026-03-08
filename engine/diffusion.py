"""
Physics-Guided Diffusion Model - Score Network
Predicts noise ε_θ(x_t, t) or score ∇_x log p_t(x) for generative physics.
"""

from typing import Optional
import math
import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional embeddings for diffusion time step t ∈ [0, 1].
    t: (B,) or (B, 1)
    Returns: (B, dim)
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
    t_flat = t.float().view(-1, 1)
    emb = t_flat * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ScoreNet(nn.Module):
    """
    Score / Noise prediction network for diffusion.
    Input: x (state), t (diffusion time), condition (previous state x_prev).
    Output: ε_θ(x_t, t, condition) or score ∇_x log p_t(x).
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: Optional[int] = None,
        hidden_dim: int = 128,
        time_emb_dim: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        cond_dim = condition_dim or state_dim
        self.condition_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = []
        in_dim = state_dim + cond_dim + hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, state_dim))
        self.mlp = nn.Sequential(*layers)

        self.time_emb_dim = time_emb_dim

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, state_dim) - noisy state
        t: (B,) or (B, 1) - diffusion time in [0, 1]
        condition: (B, condition_dim) - previous state (e.g. x_prev)

        Returns: (B, state_dim) - predicted noise ε
        """
        B = x.shape[0]
        if t.dim() == 1:
            t = t.view(-1, 1)
        t_emb = sinusoidal_embedding(t.squeeze(-1), self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        if condition is None:
            condition = torch.zeros(B, self.condition_dim, device=x.device, dtype=x.dtype)
        elif condition.dim() == 1:
            condition = condition.unsqueeze(0).expand(B, -1)

        h = torch.cat([x, condition, t_emb], dim=-1)
        return self.mlp(h)
