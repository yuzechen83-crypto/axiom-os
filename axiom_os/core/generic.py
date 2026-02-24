"""
GENERIC Formalism - Thermodynamic Unification
Unifies Reversible (Mechanics) and Irreversible (Thermodynamics) dynamics.

Master Equation:
  dz/dt = L(z) @ ∇E(z)  +  M(z) @ ∇S(z)
          ^^^^^^^^^^^^     ^^^^^^^^^^^^
          Reversible       Irreversible
          (Symplectic)     (Dissipative)

- E(z): Total Energy
- S(z): Total Entropy
- L: Poisson Matrix (Anti-symmetric, L^T = -L)
- M: Friction Matrix (Positive Semi-Definite)
- Degeneracy: L·∇S = 0, M·∇E = 0
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


def _skew(A: torch.Tensor) -> torch.Tensor:
    """L = A - A^T: ensures anti-symmetric matrix."""
    return A - A.transpose(-2, -1)


def _psd_from_R(R: torch.Tensor) -> torch.Tensor:
    """M = R^T @ R: ensures positive semi-definite matrix."""
    return R.transpose(-2, -1) @ R


class GENERICSystem(nn.Module):
    """
    GENERIC dynamics as nn.Module.
    Learns E(z), S(z), L(z), M(z) from data.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        entropy_dim: int = 0,
    ):
        """
        Args:
            state_dim: Dimension of state z
            hidden_dim: Hidden units for E, S, L, M networks
            entropy_dim: If > 0, S is learned; else S = 0 (purely mechanical)
        """
        super().__init__()
        self.state_dim = state_dim
        self.entropy_dim = entropy_dim

        # Potentials: E(z) -> scalar, S(z) -> scalar
        self.energy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        if entropy_dim > 0:
            self.entropy_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.entropy_net = None

        # Structure matrices: L = A - A^T, M = R^T R
        # A and R are (state_dim, state_dim)
        self.A_raw = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.R_raw = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.energy_net[0].weight, gain=0.5)
        nn.init.zeros_(self.energy_net[-1].bias)
        if self.entropy_net is not None:
            nn.init.xavier_uniform_(self.entropy_net[0].weight, gain=0.5)
            nn.init.zeros_(self.entropy_net[-1].bias)

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """E(z) -> scalar (or batch of scalars)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.energy_net(z).squeeze(-1)

    def entropy(self, z: torch.Tensor) -> torch.Tensor:
        """S(z) -> scalar. Returns 0 if entropy_dim=0."""
        if self.entropy_net is None:
            return torch.zeros(z.shape[0] if z.dim() > 1 else 1, device=z.device, dtype=z.dtype)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.entropy_net(z).squeeze(-1)

    def get_L(self, z: torch.Tensor) -> torch.Tensor:
        """Anti-symmetric L = A - A^T."""
        return _skew(self.A_raw)

    def get_M(self, z: torch.Tensor) -> torch.Tensor:
        """PSD M = R^T R."""
        return _psd_from_R(self.R_raw)

    def grad_E(self, z: torch.Tensor) -> torch.Tensor:
        """∇E via autograd."""
        z = z.detach().requires_grad_(True)
        E = self.energy(z)
        if E.dim() == 0:
            E = E.unsqueeze(0)
        grad = torch.autograd.grad(E.sum(), z, create_graph=True, retain_graph=True)[0]
        return grad

    def grad_S(self, z: torch.Tensor) -> torch.Tensor:
        """∇S via autograd."""
        if self.entropy_net is None:
            return torch.zeros_like(z)
        z = z.detach().requires_grad_(True)
        S = self.entropy(z)
        if S.dim() == 0:
            S = S.unsqueeze(0)
        grad = torch.autograd.grad(S.sum(), z, create_graph=True, retain_graph=True)[0]
        return grad

    def dynamics(
        self,
        z: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        dz/dt = L @ ∇E + M @ ∇S
        Returns z_dot or (z_dot, aux_dict) with L, M, grad_E, grad_S.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = z.clone().detach().requires_grad_(True)

        E = self.energy(z)
        S = self.entropy(z)
        grad_E = torch.autograd.grad(E.sum(), z, create_graph=True, retain_graph=True)[0]
        grad_S = torch.autograd.grad(S.sum(), z, create_graph=True, retain_graph=True)[0]

        L = self.get_L(z)
        M = self.get_M(z)

        # z_dot = L @ grad_E + M @ grad_S
        # L @ grad_E: (dim, dim) @ (B, dim)^T -> (B, dim)
        z_dot = torch.einsum("ij,bj->bi", L, grad_E) + torch.einsum("ij,bj->bi", M, grad_S)

        if return_aux:
            return z_dot, {
                "L": L, "M": M, "grad_E": grad_E, "grad_S": grad_S,
                "E": E, "S": S,
            }
        return z_dot

    def forward(
        self,
        z: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Alias for dynamics."""
        return self.dynamics(z, return_aux=return_aux)
