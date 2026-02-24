"""
Robust PINN Architecture
- PhysicsResNet: Residual MLP backbone (no symplectic, no leapfrog)
- SoftMeltLayer: g(D) = sigmoid(κ·(D-ε)) from Document Section 2.2
All tensors strictly torch.float64.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..config import DTYPE, HIDDEN_DIM, NUM_LAYERS, SOFT_MELT_KAPPA, SOFT_MELT_EPSILON


class ResBlock(nn.Module):
    """
    ResBlock: x + Linear(SiLU(Linear(x)))
    Residuals enable deep gradient propagation for high-frequency physics.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(self.act(self.linear1(x)))


class PhysicsResNet(nn.Module):
    """
    Backbone: Linear -> SiLU -> [ResBlock * N] -> Linear
    Input: (t, x) concatenated
    Output: u (prediction)
    No SymplecticLinear, no matrix exponentials, no leapfrog.
    Maps (t, x) -> u directly.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dtype: torch.dtype = DTYPE,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        self.entry = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_layers)])
        self.exit_ = nn.Linear(hidden_dim, out_dim)

        self.to(dtype)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (N, in_dim) where in_dim >= 2 for (t, x, ...)
        """
        x = coords.to(self.dtype)
        h = self.entry(x)
        for block in self.blocks:
            h = block(h)
        return self.exit_(h)


class SoftMeltLayer(nn.Module):
    """
    Soft-Melt from Document Section 2.2
    g(D) = sigmoid(κ · (D - ε))
    h_final = g · h_safe + (1 - g) · h_NN

    D = distance between h_NN and safe region (e.g. ||h_NN - h_safe||)
    When D < ε: g -> 0, use h_NN (explore)
    When D > ε: g -> 1, use h_safe (retreat to safe state)
    """

    def __init__(
        self,
        kappa: float = SOFT_MELT_KAPPA,
        epsilon: float = SOFT_MELT_EPSILON,
        dtype: torch.dtype = DTYPE,
    ):
        super().__init__()
        self.kappa = kappa
        self.epsilon = epsilon
        self.dtype = dtype

    def forward(
        self,
        h_NN: torch.Tensor,
        h_safe: torch.Tensor,
        distance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h_NN: neural network output
        h_safe: safe/fallback state
        distance: optional precomputed distance; if None, use ||h_NN - h_safe||
        """
        h_NN = h_NN.to(self.dtype)
        h_safe = h_safe.to(self.dtype)

        if distance is None:
            distance = torch.norm(h_NN - h_safe, dim=-1, keepdim=True)

        # g = sigmoid(κ · (D - ε))
        # When D small: g small -> use h_NN
        # When D large: g large -> use h_safe
        g = torch.sigmoid(self.kappa * (distance - self.epsilon))

        return g * h_safe + (1.0 - g) * h_NN
