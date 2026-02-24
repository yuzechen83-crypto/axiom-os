"""
Imagination - The Creative Coupling
Combine validated Hamiltonians to explore emergent dynamics.
H_new = H_s + H_p + ε·H_int
Simulate for stable chaos or synchronization.
"""

from typing import Callable, Tuple, Optional, Union
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
    Detect stable chaos or synchronization.
    """
    H_new = couple_theories(H_s, H_p, epsilon=epsilon)
    return leapfrog_coupled(H_new, q0, p0, dt=dt, n_steps=n_steps, device=device)


# -----------------------------------------------------------------------------
# Double Pendulum: Generative Physics (Creation from Single Pendulum Laws)
# -----------------------------------------------------------------------------

def double_pendulum_H(
    g_over_L: float = 1.0,
    L1: float = 1.0,
    L2: float = 1.0,
) -> Callable:
    """
    Hamiltonian for a rigid double pendulum (Lagrangian-derived).
    H_double = (p1² + 2p2² - 2p1p2 cos(q1-q2)) / (2(1+sin²(q1-q2))) + V(q1,q2)
    where V(q1,q2) = g/L * (L1(1-cos q1) + L2(1-cos q2))

    q1, q2: angles from vertical. qp = [q1, q2, p1, p2].
    """
    def V(q1: float, q2: float) -> float:
        return g_over_L * (L1 * (1 - np.cos(q1)) + L2 * (1 - np.cos(q2)))

    def H_double(qp) -> float:
        qp = np.atleast_1d(np.asarray(qp, dtype=np.float64))
        if len(qp) < 4:
            return 0.0
        q1, q2, p1, p2 = float(qp[0]), float(qp[1]), float(qp[2]), float(qp[3])
        dq = q1 - q2
        denom = 2.0 * (1.0 + np.sin(dq) ** 2)
        if abs(denom) < 1e-12:
            denom = 1e-12
        T = (p1**2 + 2*p2**2 - 2*p1*p2*np.cos(dq)) / denom
        return T + V(q1, q2)

    return H_double


def coupled_simulation(
    H1: Optional[Callable] = None,
    H2: Optional[Callable] = None,
    coupling_type: str = "rigid",
    g_over_L: float = 1.0,
    L1: float = 1.0,
    L2: float = 1.0,
) -> Callable:
    """
    Construct a Double Pendulum Hamiltonian from single-pendulum laws.

    H1, H2: Optional. Single pendulum H = ½p² + (1-cos q). Small-angle limit: k = g/L.
            If provided, g_over_L can be inferred; else use g_over_L parameter.

    coupling_type: 'rigid' -> standard double pendulum (two rods connected).
    H_double = (p1² + 2p2² - 2p1p2 cos(q1-q2)) / (2(1+sin²(q1-q2))) + V(q1,q2)

    Returns: H_double(qp) where qp = [q1, q2, p1, p2]
    """
    if coupling_type == "rigid":
        return double_pendulum_H(g_over_L=g_over_L, L1=L1, L2=L2)
    raise ValueError(f"Unknown coupling_type: {coupling_type}")


def angles_to_cartesian(
    q1: np.ndarray,
    q2: np.ndarray,
    L1: float = 1.0,
    L2: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert generalized coordinates (q1, q2) to Cartesian (x1,y1, x2,y2).
    q1, q2: angles from vertical (downward positive).
    """
    x1 = L1 * np.sin(q1)
    y1 = -L1 * np.cos(q1)
    x2 = L1 * np.sin(q1) + L2 * np.sin(q2)
    y2 = -L1 * np.cos(q1) - L2 * np.cos(q2)
    return x1, y1, x2, y2


# -----------------------------------------------------------------------------
# Controlled Symplectic Integrator (Imagination-Augmented Control)
# -----------------------------------------------------------------------------

def step_controlled(
    state: Tuple[np.ndarray, np.ndarray],
    dt: float,
    action: float,
    H: Callable,
    grad_H: Callable,
    torque_idx: int = 0,
    friction: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single controlled symplectic step with optional friction.
    Controlled Physics: p_dot = -dH/dq - c*q_dot + tau
    where q_dot = dH/dp (velocity). Bridges sim-to-real gap.

    state: (q, p)
    action: torque tau applied to first joint
    friction: damping coefficient c (discovered or c~0.1)
    Returns: (q_new, p_new)
    """
    q, p = state
    q = np.asarray(q, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    n = len(q)

    qp = np.concatenate([q, p])
    dH_dq, dH_dp = grad_H(qp)

    p_half = p - 0.5 * dt * dH_dq - 0.5 * dt * friction * dH_dp
    qp_mid = np.concatenate([q, p_half])
    _, dH_dp_mid = grad_H(qp_mid)

    q_new = q + dt * dH_dp_mid
    qp_end = np.concatenate([q_new, p_half])
    dH_dq_end, dH_dp_end = grad_H(qp_end)

    p_new = p_half - 0.5 * dt * dH_dq_end - 0.5 * dt * friction * dH_dp_end

    p_new[torque_idx] += action * dt  # External torque on first joint

    return q_new, p_new


def rollout_controlled(
    H: Callable,
    q0: np.ndarray,
    p0: np.ndarray,
    action_sequence: np.ndarray,
    dt: float = 0.01,
    device: Optional["torch.device"] = None,
    friction: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rollout double pendulum with control sequence.
    action_sequence: (T,) torque values for each step.
    friction: damping c for sim-to-real (p_dot += -c*q_dot).
    Returns: t_arr, q_traj, p_traj
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

    T = len(action_sequence)
    t_arr = np.arange(T + 1) * dt
    q_traj = np.zeros((T + 1, state_dim))
    p_traj = np.zeros((T + 1, state_dim))
    q_traj[0] = q
    p_traj[0] = p

    for i in range(T):
        tau = float(action_sequence[i])
        q, p = step_controlled((q, p), dt, tau, H, grad_H, torque_idx=0, friction=friction)
        q_traj[i + 1] = q
        p_traj[i + 1] = p

    return t_arr, q_traj, p_traj
