"""
Clifford Neural Network Layers - O(3) Equivariant
Operate on Multivectors (Scalars, Vectors, Bivectors, Trivectors).
"""

from typing import Optional
import math
import torch
import torch.nn as nn

from ..core.clifford_ops import geometric_product, multivector_magnitude, N_BLADES


class CliffordLinear(nn.Module):
    """
    Linear layer in Clifford domain. Y = W ⊗ X + B (geometric product).
    Efficient implementation using precomputed Cayley table.
    Y[c_out] = sum_{c_in} W[c_out, c_in] * X[c_in] (geometric product)
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        from ..core.clifford_ops import CAYLEY
        self.register_buffer("cayley", CAYLEY)
        scale = 1.0 / math.sqrt(in_channels * N_BLADES)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, N_BLADES) * scale)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, N_BLADES))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_c, 8) or (B, in_c*8) flattened
        if x.shape[-1] == self.in_channels * N_BLADES and x.dim() == 2:
            x = x.reshape(-1, self.in_channels, N_BLADES)
        # x: (B, in_c, 8), W: (out_c, in_c, 8)
        # geo_prod(W[o,i], x[b,i]) = sum_{j,k} cayley[j,k,:] * W[o,i,j] * x[b,i,k]
        # result[b,o,:] = sum_i geo_prod(W[o,i], x[b,i])
        # = sum_i sum_{j,k} cayley[j,k,:] * W[o,i,j] * x[b,i,k]
        # = sum_{j,k} cayley[j,k,:] * (sum_i W[o,i,j] * x[b,i,k])
        # = sum_{j,k} cayley[j,k,:] * (W[o,:,j] * x[b,:,k]).sum(dim=1)
        # Let A[b,o,j,k] = sum_i W[o,i,j] * x[b,i,k] = einsum("oij,bik->bojk", W, x)
        # result[b,o,:] = einsum("jk...,bojk->bo...", cayley, A)
        A = torch.einsum("oij,bik->bojk", self.weight, x)
        out = torch.einsum("jkl,bojk->bol", self.cayley, A)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out


class EquivariantCliffordLinear(nn.Module):
    """
    Clifford linear layer with scalar-only weights (rotation-invariant).
    Guarantees O(3) equivariance: f(R(x)) = R(f(x)).
    Scalar * multivector = scaling, which commutes with rotation.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        from ..core.clifford_ops import CAYLEY
        self.register_buffer("cayley", CAYLEY)
        scale = 1.0 / math.sqrt(in_channels)
        self._w_scalar = nn.Parameter(torch.randn(out_channels, in_channels) * scale)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, N_BLADES))
            self.bias.data[:, 1:7] = 0.0
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.in_channels * N_BLADES and x.dim() == 2:
            x = x.reshape(-1, self.in_channels, N_BLADES)
        # Scalar weight: W*X = scalar * X (1*X = X), so output = sum over channels of w_s * x
        weight = torch.zeros(self.out_channels, self.in_channels, N_BLADES, device=x.device, dtype=x.dtype)
        weight[:, :, 0] = self._w_scalar
        A = torch.einsum("oij,bik->bojk", weight, x)
        out = torch.einsum("jkl,bojk->bol", self.cayley, A)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out


class CliffordActivation(nn.Module):
    """
    Gated nonlinearity: preserves geometry, scales by magnitude.
    x' = x · σ(w·ρ + b) where ρ = ||x||
    """

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho = multivector_magnitude(x)
        gate = torch.sigmoid(self.w * rho + self.b)
        return x * gate
