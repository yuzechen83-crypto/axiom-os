"""
Knowledge - Physics Structures
SymplecticLaw: Container for discovered conservative dynamics.
Stores H(q,p), params, and can evolve itself via built-in Leapfrog.
"""

from typing import Callable, Dict, Optional, Tuple, Any
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SymplecticLaw:
    """
    Container for discovered physics: Hamiltonian dynamics that can run itself.
    The Law stores the essence, not the data.
    """

    def __init__(
        self,
        name: str,
        hamiltonian_func: Callable,
        params: Optional[Dict[str, float]] = None,
        phase_space_dim: int = 2,
        J_matrix: Optional[np.ndarray] = None,
        device: Optional[Any] = None,
    ):
        """
        name: e.g., "HarmonicOscillator_ID_001"
        hamiltonian_func: H(q, p) -> scalar. Accepts (q,p) as numpy or torch.
        params: Discovered constants, e.g., {'k': 10.0, 'm': 1.0}
        phase_space_dim: 2n (e.g., 2 for 1D oscillator: q,p)
        J_matrix: Symplectic structure [[0,I],[-I,0]]. Default: standard 2x2.
        """
        self.name = name
        self.hamiltonian_func = hamiltonian_func
        self.params = params or {}
        self.phase_space_dim = phase_space_dim
        self.n = phase_space_dim // 2  # n position vars, n momentum vars (2n = phase_space_dim)
        self.device = device

        if J_matrix is not None:
            self.J_matrix = np.asarray(J_matrix)
        else:
            # Standard J = [[0, I], [-I, 0]] for 2n-dim
            n = max(1, self.n)
            I = np.eye(n)
            Z = np.zeros((n, n))
            self.J_matrix = np.block([[Z, I], [-I, Z]])

    def H(self, qp: np.ndarray) -> float:
        """Evaluate Hamiltonian at phase-space point qp = [q, p]."""
        qp = np.atleast_1d(np.asarray(qp, dtype=np.float64))
        if HAS_TORCH and callable(self.hamiltonian_func):
            t = torch.from_numpy(qp.astype(np.float32)).unsqueeze(0)
            if self.device is not None:
                t = t.to(self.device)
            with torch.no_grad():
                out = self.hamiltonian_func(t)
            return float(out.item())
        return float(np.asarray(self.hamiltonian_func(qp)).ravel()[0])

    def grad_H(self, qp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dH/dq and dH/dp at qp."""
        qp = np.asarray(qp, dtype=np.float64).ravel()
        n = self.n

        if HAS_TORCH and self.device is not None:
            t = torch.from_numpy(qp.astype(np.float32)).unsqueeze(0).to(self.device).requires_grad_(True)
            H_val = self.hamiltonian_func(t).sum()
            g = torch.autograd.grad(H_val, t)[0]
            g = g.cpu().numpy().ravel()
            return g[:n], g[n:]
        # Finite differences
        eps = 1e-6
        g = np.zeros_like(qp)
        for i in range(len(qp)):
            qp_plus = qp.copy()
            qp_plus[i] += eps
            qp_minus = qp.copy()
            qp_minus[i] -= eps
            g[i] = (self.H(qp_plus) - self.H(qp_minus)) / (2 * eps)
        return g[:n], g[n:]

    def evolve(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        t: float = 50.0,
        dt: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run built-in Symplectic Integrator (Leapfrog) on this law.
        Returns: t_arr, q_traj, p_traj
        Allows the law to "run" itself in the imagination.
        """
        q = np.array(q0, dtype=np.float64).ravel()
        p = np.array(p0, dtype=np.float64).ravel()
        n = len(q)

        n_steps = int(t / dt)
        t_arr = np.arange(n_steps + 1) * dt
        q_traj = np.zeros((n_steps + 1, n))
        p_traj = np.zeros((n_steps + 1, n))
        q_traj[0] = q
        p_traj[0] = p

        for i in range(n_steps):
            qp = np.concatenate([q, p])
            dH_dq, dH_dp = self.grad_H(qp)

            p_half = p - 0.5 * dt * dH_dq
            qp_mid = np.concatenate([q, p_half])
            _, dH_dp_mid = self.grad_H(qp_mid)

            q = q + dt * dH_dp_mid
            qp_end = np.concatenate([q, p_half])
            dH_dq_end, _ = self.grad_H(qp_end)

            p = p_half - 0.5 * dt * dH_dq_end

            q_traj[i + 1] = q
            p_traj[i + 1] = p

        return t_arr, q_traj, p_traj

    def __repr__(self) -> str:
        return f"SymplecticLaw(name={self.name!r}, params={self.params}, dim={self.phase_space_dim})"
