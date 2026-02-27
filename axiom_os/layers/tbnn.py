"""
Tensor Basis Neural Network (TBNN) - Ling et al. 2016

Galilean invariant and rotationally equivariant Reynolds stress closure.
tau_ij = sum_n g_n(lambda) * T_n

Input: invariants lambda (5 scalars from S, Omega)
Output: coefficients g_n (10)
Final: tau = sum_n g_n * T_n
"""

from typing import Optional
import torch
import torch.nn as nn

N_INVARIANTS = 5
N_TENSORS = 10


class TBNN(nn.Module):
    """
    Tensor Basis Neural Network for tau_ij prediction.

    Input: invariants (B, 5) from Pope decomposition
    Output: tau_ij (B, 3, 3) = sum_n g_n(lambda) * T_n

    T_n are passed in at forward (computed from S, Omega in numpy or torch).
    """

    def __init__(
        self,
        n_invariants: int = N_INVARIANTS,
        n_tensors: int = N_TENSORS,
        hidden: int = 32,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_invariants = n_invariants
        self.n_tensors = n_tensors
        layers = []
        in_dim = n_invariants
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_tensors))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        invariants: torch.Tensor,
        tensor_basis: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        invariants: (B, 5) - normalized
        tensor_basis: (B, 10, 3, 3) - normalized
        sigma: (B,) - characteristic scale
        Returns: tau (B, 3, 3) = sigma^2 * sum_n g_n * T_n
        """
        g = self.mlp(invariants)  # (B, 10)
        tau_raw = torch.einsum("bn,bnij->bij", g, tensor_basis)
        scale = (sigma**2).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        return scale * tau_raw


def stack_tensor_basis(tensor_basis_list: list) -> torch.Tensor:
    """
    Stack list of 10 tensors (..., 3, 3) -> (..., 10, 3, 3).
    """
    import numpy as np
    stacked = np.stack(tensor_basis_list, axis=-3)  # (..., 10, 3, 3)
    return torch.from_numpy(stacked).float()
