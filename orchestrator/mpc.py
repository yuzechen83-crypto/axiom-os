"""
MPC - Model Predictive Path Integral (The Acrobat)
Imagination-Augmented Controller: Simulate N futures in Einstein Core, pick the best.
"""

from typing import Callable, Optional, Tuple
import numpy as np

from ..core.einstein import EinsteinCore, _grad_H_numpy

PI = np.pi


def angle_normalize(theta: np.ndarray) -> np.ndarray:
    """
    Wrap angle to [-π, π]. Handles 0 ≡ 2π topology.
    Prevents phantom errors from 2π wrap-around.
    """
    return ((np.asarray(theta, dtype=np.float64) + PI) % (2 * PI)) - PI


def double_pendulum_H(
    g_over_L: float = 10.0,
    L1: float = 1.0,
    L2: float = 1.0,
) -> Callable:
    """
    Hamiltonian for rigid double pendulum (acrobot).
    H = T + V, qp = [q1, q2, p1, p2].
    """
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
        V = g_over_L * (L1 * (1 - np.cos(q1)) + L2 * (1 - np.cos(q2)))
        return T + V
    return H_double


def _step_controlled(
    q: np.ndarray,
    p: np.ndarray,
    action: float,
    H_func: Callable,
    state_dim: int,
    dt: float,
    friction: float,
    torque_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single controlled symplectic step: leapfrog + friction + torque.
    p_dot += -c*q_dot + tau  (friction and external torque).
    """
    def grad_H(qp):
        return _grad_H_numpy(H_func, qp, state_dim)

    qp = np.concatenate([q, p])
    dH_dq, dH_dp = grad_H(qp)

    # Leapfrog with friction: p_half -= 0.5*dt*friction*dH_dp
    p_half = p - 0.5 * dt * dH_dq - 0.5 * dt * friction * dH_dp
    qp_mid = np.concatenate([q, p_half])
    _, dH_dp_mid = grad_H(qp_mid)

    q_new = q + dt * dH_dp_mid
    qp_end = np.concatenate([q_new, p_half])
    dH_dq_end, dH_dp_end = grad_H(qp_end)

    p_new = p_half - 0.5 * dt * dH_dq_end - 0.5 * dt * friction * dH_dp_end
    p_new[torque_idx] += action * dt
    return q_new, p_new


def step_env(
    q: np.ndarray,
    p: np.ndarray,
    action: float,
    H_func: Callable,
    dt: float = 0.02,
    friction: float = 0.1,
    state_dim: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single environment step: apply action, return (q_next, p_next)."""
    return _step_controlled(q, p, action, H_func, state_dim, dt, friction)


def _rollout_controlled(
    H_func: Callable,
    q0: np.ndarray,
    p0: np.ndarray,
    action_sequence: np.ndarray,
    dt: float,
    friction: float,
    state_dim: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rollout trajectories with action (torque) inputs. Returns (q_traj, p_traj)."""
    q = np.asarray(q0, dtype=np.float64).ravel()
    p = np.asarray(p0, dtype=np.float64).ravel()
    T = len(action_sequence)
    q_traj = np.zeros((T + 1, state_dim))
    p_traj = np.zeros((T + 1, state_dim))
    q_traj[0] = q
    p_traj[0] = p

    for i in range(T):
        q, p = _step_controlled(q, p, float(action_sequence[i]), H_func, state_dim, dt, friction)
        q_traj[i + 1] = q
        p_traj[i + 1] = p
    return q_traj, p_traj


class ImaginationMPC:
    """
    MPPI (Model Predictive Path Integral) controller.
    The Mind's Eye: Rollout N trajectories, pick best action.
    """

    def __init__(
        self,
        H: Optional[Callable] = None,
        horizon_steps: int = 80,
        n_samples: int = 2000,
        dt: float = 0.02,
        friction: float = 0.1,
        action_std: float = 2.0,
        action_bounds: Tuple[float, float] = (-25.0, 25.0),
        state_dim: int = 2,
        target_state: Optional[np.ndarray] = None,
        target_energy: Optional[float] = None,
        distance_threshold: float = 0.5,
    ):
        self.H = H or double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
        self.horizon_steps = horizon_steps
        self.n_samples = n_samples
        self.dt = dt
        self.friction = friction
        self.action_std = action_std
        self.action_bounds = action_bounds
        self.state_dim = state_dim
        self.target_state = target_state if target_state is not None else np.array([PI, PI])
        self.target_energy = target_energy
        self.distance_threshold = distance_threshold
        self._last_tau = 0.0

    @staticmethod
    def energy_shaping_cost(state: np.ndarray, target_energy: float, H_func: Callable) -> float:
        """
        Cost for energy shaping (swing-up phase).
        Penalizes deviation from target energy.
        state: (q, p) concatenated, shape (2*state_dim,) or (n, 2*state_dim).
        """
        state = np.asarray(state)
        if state.ndim == 1:
            E = float(np.asarray(H_func(state)).ravel()[0])
            return (E - target_energy) ** 2
        # Trajectory: mean cost over time
        return float(np.mean([(float(np.asarray(H_func(state[t])).ravel()[0]) - target_energy) ** 2
                             for t in range(state.shape[0])]))

    @staticmethod
    def stabilization_cost(
        state: Tuple[np.ndarray, np.ndarray],
        target_state: np.ndarray,
        w_pos: float = 100.0,
        w_vel: float = 10.0,
    ) -> float:
        """
        LQR-style cost for stabilization (lock-down phase).
        state: (q_traj, p_traj), target_state: (q_target,) for angles.
        """
        q_traj, p_traj = state
        q_traj = np.asarray(q_traj)
        p_traj = np.asarray(p_traj)
        target = np.asarray(target_state).ravel()
        err = angle_normalize(q_traj - target)
        return float(np.mean(w_pos * np.sum(err**2, axis=-1) + w_vel * np.sum(p_traj**2, axis=-1)))

    def _trajectory_cost(
        self,
        q_traj: np.ndarray,
        p_traj: np.ndarray,
        action_sequence: np.ndarray,
        w_u: float = 0.0001,
    ) -> float:
        """
        Sustainable 5:1 Cost - Both angles matter, theta1 prioritized.
        """
        err = angle_normalize(q_traj - self.target_state)
        err1, err2 = err[:, 0], err[:, 1]
        vel1, vel2 = p_traj[:, 0], p_traj[:, 1]
        tau = np.asarray(action_sequence)
        T = len(tau)

        # 5:1 rebalance: both critical, theta1 main target
        w_pos_1, w_pos_2 = 5000.0, 1000.0
        w_vel_1, w_vel_2 = 500.0, 100.0

        # 1. Running Cost
        pos_cost = w_pos_1 * (err1[:T] ** 2) + w_pos_2 * (err2[:T] ** 2)
        vel_cost = w_vel_1 * (vel1[:T] ** 2) + w_vel_2 * (vel2[:T] ** 2)
        running_cost = pos_cost + vel_cost + w_u * (tau**2)
        total_cost = float(np.sum(running_cost))

        # 2. TERMINAL COST (The Anchor) - 100x penalty on last state
        terminal_error = (
            100000.0 * (err1[-1] ** 2 + err2[-1] ** 2)
            + 1000.0 * (vel1[-1] ** 2 + vel2[-1] ** 2)
        )
        total_cost += terminal_error

        return total_cost

    def plan(self, q: np.ndarray, p: np.ndarray) -> float:
        """
        MPPI: Rollout N trajectories, return first action of best.
        Input: (q, p) current state.
        Output: Optimal torque tau*.
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        p = np.asarray(p, dtype=np.float64).ravel()
        if len(q) < self.state_dim:
            q = np.resize(q, self.state_dim)
        if len(p) < self.state_dim:
            p = np.resize(p, self.state_dim)

        best_tau = 0.0
        best_cost = np.inf

        for _ in range(self.n_samples):
            action_seq = np.random.randn(self.horizon_steps) * self.action_std
            action_seq = np.clip(action_seq, self.action_bounds[0], self.action_bounds[1])

            try:
                q_traj, p_traj = _rollout_controlled(
                    self.H,
                    q,
                    p,
                    action_seq,
                    self.dt,
                    self.friction,
                    self.state_dim,
                )
                cost = self._trajectory_cost(q_traj, p_traj, action_seq)
                if cost < best_cost:
                    best_cost = cost
                    best_tau = float(action_seq[0])
            except (ValueError, RuntimeError, FloatingPointError):
                continue

        tau_smooth = 0.8 * best_tau + 0.2 * self._last_tau
        self._last_tau = tau_smooth
        return tau_smooth
