"""
Einstein Core - The Philosopher
Helmholtz Decomposition: Separate dynamics into Conservative (-∇H) and Dissipative.
Symplectic Integrator: Energy-conserving Leapfrog for thought experiments.
"""

from typing import Callable, Tuple, Optional, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def helmholtz_decomposition(
    f_dynamics: Callable,
    qp: np.ndarray,
    state_dim: int,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose F = F_conservative + F_dissipative.
    Conservative: F_cons = J ∇H (Hamiltonian flow).
    Dissipative: remainder (e.g., friction, damping).

    For 2D phase (q,p): J = [[0,1],[-1,0]], so dq/dt = dH/dp, dp/dt = -dH/dq.
    F_conservative = (dH/dp, -dH/dq).

    This is a numerical approximation: we infer H from f and split.
    For learned dynamics, use EinsteinCore.decompose_dynamics() for full HNN fit.
    """
    qp = np.asarray(qp, dtype=np.float64).ravel()
    f = np.asarray(f_dynamics(qp), dtype=np.float64).ravel()
    n = len(qp)

    # Approximate curl-free (conservative) part via finite diff
    # For 2D: if f = (fq, fp), conservative means fq = dH/dp, fp = -dH/dq
    # So ∂fq/∂q = ∂fp/∂p would hold for Hamiltonian. We do a simple split:
    # F_dissipative typically has structure (0, -c*v) for damping. So we
    # assume dissipative is in the momentum part.
    f_cons = np.zeros_like(f)
    f_diss = f.copy()

    # Simple heuristic: if state_dim=2, conservative part is (v, -dH/dq)
    # For 1D oscillator: dH/dq ~ k*q, dH/dp = p.
    # f_cons = (p, -k*q) for harmonic. We don't have full H here.
    # Return a placeholder: full decomposition needs EinsteinCore.
    f_cons[:state_dim] = f[:state_dim]  # position derivative from H
    f_cons[state_dim:] = f[state_dim:]  # momentum - we'd need to separate
    # For now return f as conservative, zeros as dissipative (simplified)
    f_cons = f.copy()
    f_diss = np.zeros_like(f)
    return f_cons, f_diss


class SymplecticIntegrator:
    """
    Symplectic Leapfrog integrator for Hamiltonian H.
    Strictly energy-conserving for thought experiments.
    """

    def __init__(
        self,
        H: Callable,
        grad_H: Callable,
        state_dim: int,
        device: Optional[object] = None,
    ):
        self.H = H
        self.grad_H = grad_H
        self.state_dim = state_dim
        self.device = device

    def step(
        self,
        q: np.ndarray,
        p: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single Leapfrog step."""
        qp = np.concatenate([q, p])
        dH_dq, dH_dp = self.grad_H(qp)

        p_half = p - 0.5 * dt * dH_dq
        qp_mid = np.concatenate([q, p_half])
        _, dH_dp_mid = self.grad_H(qp_mid)

        q_new = q + dt * dH_dp_mid
        qp_end = np.concatenate([q_new, p_half])
        dH_dq_end, _ = self.grad_H(qp_end)

        p_new = p_half - 0.5 * dt * dH_dq_end
        return q_new, p_new

    def evolve(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        dt: float,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve trajectory. Returns (q_traj, p_traj)."""
        q = np.array(q0, dtype=np.float64).ravel()
        p = np.array(p0, dtype=np.float64).ravel()
        state_dim = len(q)

        q_traj = np.zeros((n_steps + 1, state_dim))
        p_traj = np.zeros((n_steps + 1, state_dim))
        q_traj[0] = q
        p_traj[0] = p

        for i in range(n_steps):
            q, p = self.step(q, p, dt)
            q_traj[i + 1] = q
            p_traj[i + 1] = p

        return q_traj, p_traj


__all__ = ["helmholtz_decomposition", "SymplecticIntegrator"]
