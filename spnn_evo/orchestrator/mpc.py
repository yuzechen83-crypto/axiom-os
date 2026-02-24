"""
MPC - Model Predictive Path Integral (The Brain)
Parallel Imagination Controller: Simulate 100 futures, pick the best.
"Don't learn by crashing (RL). Learn by thinking (Einstein)."
"""

import numpy as np
from typing import Callable, Tuple, Optional

from ..core.imagination import (
    double_pendulum_H,
    rollout_controlled,
)


# Target: upright equilibrium theta1=pi, theta2=pi
PI = np.pi


def angle_normalize(x):
    """Wrap angle to [-pi, pi]. Prevents phantom errors from 2pi wrapping."""
    return ((np.asarray(x) + PI) % (2 * PI)) - PI


def acrobot_cost(
    q_traj: np.ndarray,
    p_traj: np.ndarray,
    action_sequence: np.ndarray,
    theta_target: float = PI,
    w_pos: float = 100.0,
    w_vel: float = 10.0,
    w_u: float = 0.1,
    cost_mode: str = "stabilization",
    near_threshold: float = 0.5,
    use_terminal_cost: bool = True,
) -> float:
    """
    Topology-aware cost. cost_mode:
    - "stabilization": Robust Damper (LQR-style) + Terminal Cost (No Surrender).
    - "hybrid": Switch Energy Shaping (far) <-> LQR Braking (near).

    When use_terminal_cost=True: Heavily penalize the last state if not upright.
    Forces the controller to find a path that ends at the target.
    """
    err1 = angle_normalize(q_traj[:, 0] - theta_target)
    err2 = angle_normalize(q_traj[:, 1] - theta_target)
    vel1 = p_traj[:, 0]
    vel2 = p_traj[:, 1]
    tau = np.asarray(action_sequence)
    T = len(tau)

    if cost_mode == "hybrid":
        is_near = (np.abs(err1) < near_threshold) & (np.abs(err2) < near_threshold)
        energy_cost = 100.0 * ((1.0 + np.cos(q_traj[:, 0])) + (1.0 + np.cos(q_traj[:, 1])))
        stabilize_cost = 500.0 * (err1**2 + err2**2) + 100.0 * (vel1**2 + vel2**2)
        cost = np.where(is_near, stabilize_cost, energy_cost) + w_u * (tau**2)
        return float(np.mean(cost))

    # Sustainable 5:1 Cost - Both angles matter, theta1 prioritized
    # w_pos_1=5000, w_pos_2=1000: both critical, theta1 main target
    # w_vel_1=500, w_vel_2=100: heavy damping both, moderate on elbow
    w_pos_1, w_pos_2 = 5000.0, 1000.0
    w_vel_1, w_vel_2 = 500.0, 100.0
    w_u = 0.0001

    # 1. Running Cost
    pos_cost = w_pos_1 * (err1[:T] ** 2) + w_pos_2 * (err2[:T] ** 2)
    vel_cost = w_vel_1 * (vel1[:T] ** 2) + w_vel_2 * (vel2[:T] ** 2)
    running_cost = pos_cost + vel_cost + w_u * (tau**2)
    total_cost = float(np.sum(running_cost))

    # 2. TERMINAL COST (The Anchor) - 100x penalty on last state
    if use_terminal_cost:
        terminal_error = (
            100000.0 * (err1[-1] ** 2 + err2[-1] ** 2)
            + 1000.0 * (vel1[-1] ** 2 + vel2[-1] ** 2)
        )
        total_cost += terminal_error

    return total_cost


class ParallelImaginationController:
    """
    MPPI-style controller using Einstein's imagination.
    Generate K random action sequences, simulate in parallel, pick best.
    """

    def __init__(
        self,
        H: Optional[Callable] = None,
        g_over_L: float = 10.0,
        L1: float = 1.0,
        L2: float = 1.0,
        horizon_steps: int = 80,
        dt: float = 0.02,
        n_samples: int = 2000,
        action_std: float = 2.0,
        action_bounds: Tuple[float, float] = (-25.0, 25.0),
        friction: float = 0.1,
        temperature: float = 0.01,
        device: Optional[object] = None,
        cost_mode: str = "stabilization",
    ):
        self.H = H or double_pendulum_H(g_over_L=g_over_L, L1=L1, L2=L2)
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.n_samples = n_samples
        self.action_std = action_std
        self.action_bounds = action_bounds
        self.friction = friction
        self.temperature = temperature
        self.device = device
        self.cost_mode = cost_mode
        self._last_tau = 0.0

    def plan(
        self,
        q: np.ndarray,
        p: np.ndarray,
    ) -> float:
        """
        Input: Current real-world state (q, p).
        Output: Optimal torque tau* (first action of best trajectory).
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        p = np.asarray(p, dtype=np.float64).ravel()
        assert len(q) >= 2 and len(p) >= 2

        best_tau = 0.0
        best_cost = np.inf

        for _ in range(self.n_samples):
            action_seq = np.random.randn(self.horizon_steps) * self.action_std
            action_seq = np.clip(action_seq, self.action_bounds[0], self.action_bounds[1])

            try:
                _, q_traj, p_traj = rollout_controlled(
                    self.H,
                    q0=q,
                    p0=p,
                    action_sequence=action_seq,
                    dt=self.dt,
                    device=self.device,
                    friction=self.friction,
                )
                cost = acrobot_cost(q_traj, p_traj, action_seq, cost_mode=self.cost_mode)
                if cost < best_cost:
                    best_cost = cost
                    best_tau = float(action_seq[0])
            except (ValueError, RuntimeError, FloatingPointError):
                continue

        # Action smoothing: blend new action with previous (reduces chatter)
        tau_smooth = 0.8 * best_tau + 0.2 * self._last_tau
        self._last_tau = tau_smooth
        return tau_smooth
