"""
Physics Kernel - PDE Residual via torch.autograd
Hamiltonian consistency: q̇ = ∂H/∂p, ṗ = -∂H/∂q
No Leapfrog. No matrix exponentials.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable

from ..config import DTYPE


class HamiltonianSystem:
    """
    Define H(q, p) for testing.
    Examples: simple harmonic oscillator, fluid potential.
    """

    def __init__(self, system_type: str = "harmonic"):
        self.system_type = system_type

    def H(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Hamiltonian H(q, p)
        harmonic: H = (1/2)p² + (1/2)ω²q²
        """
        if self.system_type == "harmonic":
            omega = 1.0
            return 0.5 * (p ** 2) + 0.5 * (omega ** 2) * (q ** 2)
        # fluid: placeholder for potential
        return 0.5 * (p ** 2) + 0.5 * (q ** 2)


def compute_pde_residual(
    model: nn.Module,
    coords: torch.Tensor,
    hamiltonian: Optional[HamiltonianSystem] = None,
    q_idx: int = 1,
    p_idx: int = 2,
) -> torch.Tensor:
    """
    Compute PDE residual using torch.autograd.

    Input:
        model: NN that maps (t, x) -> u
        coords: (N, D) where col 0 = t, col 1 = q, col 2 = p (or similar)
    Action:
        - Enable grad on coords
        - Predict u = model(coords)
        - Compute ∂u/∂t, ∂u/∂x via autograd
        - Hamiltonian check: q̇ = ∂H/∂p, ṗ = -∂H/∂q
    Output:
        mse(time_derivs - hamiltonian_derivs)
    """

    coords = coords.to(DTYPE).detach().requires_grad_(True)
    hamiltonian = hamiltonian or HamiltonianSystem()

    u = model(coords)

    # u must be scalar per point for gradient
    if u.dim() > 1 and u.shape[-1] > 1:
        u = u.sum(dim=-1)

    grad_outputs = torch.ones_like(u, dtype=DTYPE)
    grads = torch.autograd.grad(
        outputs=u,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # grads: (N, D) -> ∂u/∂t, ∂u/∂q, ∂u/∂p, ...
    t = coords[:, 0:1]
    q = coords[:, q_idx:q_idx + 1]
    p = coords[:, p_idx:p_idx + 1]

    # For Hamiltonian: we need dq/dt and dp/dt from model
    # Simplified: enforce ∂u/∂t + Hamiltonian_consistency
    # Residual: ∂u/∂t + (∂H/∂q)(∂u/∂q) + (∂H/∂p)(∂u/∂p) ~ 0
    # Or: match q̇ = ∂H/∂p, ṗ = -∂H/∂q via model derivatives

    H = hamiltonian.H(q, p)
    dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True, retain_graph=True)[0]
    dH_dp = torch.autograd.grad(H.sum(), p, create_graph=True, retain_graph=True)[0]

    # Canonical: q̇ = ∂H/∂p, ṗ = -∂H/∂q
    q_dot_true = dH_dp
    p_dot_true = -dH_dq

    # Model derivatives w.r.t. t (proxy for dynamics)
    du_dt = grads[:, 0:1]
    du_dq = grads[:, q_idx:q_idx + 1]
    du_dp = grads[:, p_idx:p_idx + 1]

    # Physics residual: time evolution of u should match Hamiltonian flow
    # L = || du/dt - (∂H/∂q)(∂u/∂q) - (∂H/∂p)(∂u/∂p) ||^2
    # Or simpler: residual on (q̇, ṗ) consistency
    # Using: R = || du/dt + (∂H/∂q)(∂u/∂q) + (∂H/∂p)(∂u/∂p) ||^2
    # For Hamiltonian: du/dt = dH/dt = 0 along trajectories if H conserved
    # Residual: u should satisfy PDE implied by Hamiltonian

    residual = du_dt + dH_dq * du_dq + dH_dp * du_dp
    return torch.mean(residual ** 2)


def compute_wave_residual(
    model: nn.Module,
    coords: torch.Tensor,
    omega: float = 1.0,
) -> torch.Tensor:
    """
    Residual for 1D wave: u_tt + ω²u = 0
    coords: (N, 2) -> (t, x)
    """
    coords = coords.to(DTYPE).detach().requires_grad_(True)
    u = model(coords)
    if u.dim() > 1:
        u = u.squeeze(-1)

    (grad_u,) = torch.autograd.grad(u.sum(), coords, create_graph=True)
    u_t = grad_u[:, 0]

    (grad_ut,) = torch.autograd.grad(u_t.sum(), coords, create_graph=True)
    u_tt = grad_ut[:, 0]

    residual = u_tt + (omega ** 2) * u
    return torch.mean(residual ** 2)
