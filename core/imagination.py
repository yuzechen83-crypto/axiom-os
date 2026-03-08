"""
Imagination - Theory Coupling (from SPNN-Evo)
Combine validated Hamiltonians to explore emergent dynamics.
H_new = H_s + H_p + ε·H_int
"""

from typing import Callable, Tuple, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def couple_theories(
    H1: Callable,
    H2: Callable,
    epsilon: float = 0.1,
    H_int: Optional[Callable] = None,
    dim_per_system: int = 2,
) -> Callable:
    """
    Create hyper-Hamiltonian from two validated Hamiltonians.
    H_new = H_s + H_p + ε·H_int

    H1, H2: Callables (qp) -> scalar.
    For coupled 2-oscillator systems, qp = [q1, p1, q2, p2] (4D).
    H1 should use qp[:2], H2 should use qp[2:4].

    H_int: Optional interaction. If None, use H_int = q1*q2 (position coupling).

    dim_per_system: Phase-space dim per system (default 2 for 1D oscillator).
    """
    if H_int is None:
        def default_H_int(qp):
            if hasattr(qp, "numpy"):
                qp = qp.numpy() if qp.requires_grad else qp.detach().numpy()
            qp = np.atleast_1d(np.asarray(qp))
            if len(qp) >= 4:
                q1, p1, q2, p2 = qp[0], qp[1], qp[2], qp[3]
                return q1 * q2  # position coupling
            return 0.0

        H_int = default_H_int

    d = dim_per_system

    def H_new(qp):
        if HAS_TORCH and isinstance(qp, torch.Tensor):
            qp1 = qp[..., :d]
            qp2 = qp[..., d : 2 * d]
            h1 = H1(qp1)
            h2 = H2(qp2)
            hint = H_int(qp)
            if isinstance(hint, (int, float)):
                hint = torch.tensor(hint, device=qp.device, dtype=qp.dtype)
            return h1 + h2 + epsilon * hint
        qp = np.atleast_1d(np.asarray(qp))
        qp1 = qp[:d]
        qp2 = qp[d : 2 * d]
        h1 = float(np.asarray(H1(qp1)).ravel()[0]) if callable(H1) else 0.0
        h2 = float(np.asarray(H2(qp2)).ravel()[0]) if callable(H2) else 0.0
        hint = float(np.asarray(H_int(qp)).ravel()[0])
        return h1 + h2 + epsilon * hint

    return H_new


def leapfrog_coupled(
    H: Callable,
    q0: np.ndarray,
    p0: np.ndarray,
    dt: float = 0.01,
    n_steps: int = 10000,
    device: Optional["torch.device"] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run symplectic Leapfrog on coupled Hamiltonian.
    Returns: t, q_traj, p_traj
    """
    q = np.array(q0, dtype=np.float64).ravel()
    p = np.array(p0, dtype=np.float64).ravel()
    state_dim = len(q)

    if HAS_TORCH and device is not None:
        def grad_H(qp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            t = torch.from_numpy(qp.astype(np.float32)).unsqueeze(0).to(device).requires_grad_(True)
            H_val = H(t).sum()
            g = torch.autograd.grad(H_val, t)[0]
            g = g.cpu().numpy().ravel()
            return g[:state_dim], g[state_dim:]
    else:
        def grad_H(qp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            eps = 1e-6
            qp = np.asarray(qp, dtype=np.float64)
            g = np.zeros_like(qp)
            for i in range(len(qp)):
                qp_plus = qp.copy()
                qp_plus[i] += eps
                qp_minus = qp.copy()
                qp_minus[i] -= eps
                g[i] = (H(qp_plus) - H(qp_minus)) / (2 * eps)
            return g[:state_dim], g[state_dim:]

    t_arr = np.arange(n_steps + 1) * dt
    q_traj = np.zeros((n_steps + 1, state_dim))
    p_traj = np.zeros((n_steps + 1, state_dim))
    q_traj[0] = q
    p_traj[0] = p

    for i in range(n_steps):
        qp = np.concatenate([q, p])
        dH_dq, dH_dp = grad_H(qp)

        p_half = p - 0.5 * dt * dH_dq
        qp_mid = np.concatenate([q, p_half])
        _, dH_dp_mid = grad_H(qp_mid)

        q = q + dt * dH_dp_mid
        qp_end = np.concatenate([q, p_half])
        dH_dq_end, _ = grad_H(qp_end)

        p = p_half - 0.5 * dt * dH_dq_end

        q_traj[i + 1] = q
        p_traj[i + 1] = p

    return t_arr, q_traj, p_traj


def simulate_coupled(
    H_s: Callable,
    H_p: Callable,
    q0: np.ndarray,
    p0: np.ndarray,
    epsilon: float = 0.1,
    dt: float = 0.01,
    n_steps: int = 10000,
    device: Optional["torch.device"] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate coupled system H_new = H_s + H_p + ε·H_int.
    Returns: t, q_traj, p_traj
    """
    H_new = couple_theories(H_s, H_p, epsilon=epsilon)
    return leapfrog_coupled(H_new, q0, p0, dt=dt, n_steps=n_steps, device=device)
