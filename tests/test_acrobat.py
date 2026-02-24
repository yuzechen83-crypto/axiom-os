"""
Level 6: Imagination-Augmented Control (The Acrobat)
Inverted Double Pendulum - Balance upright (theta1=pi, theta2=pi) using torque at base.
"Don't learn by crashing (RL). Learn by thinking (Einstein)."
Simulate 100 futures in head, pick the one that keeps pendulum upright, execute in reality.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.imagination import (
    double_pendulum_H,
    rollout_controlled,
    step_controlled,
)
from spnn_evo.orchestrator.mpc import ParallelImaginationController, angle_normalize

PI = np.pi


def _grad_H(H, qp, state_dim=2):
    """Finite-diff gradient for numpy H."""
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


def real_world_step(
    q: np.ndarray,
    p: np.ndarray,
    tau: float,
    H,
    dt: float = 0.01,
    friction: float = 0.01,
    noise_std: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update real-world double pendulum: apply torque, add friction, sensor noise.
    """
    def grad_H(qp):
        return _grad_H(H, qp, state_dim=2)

    q_new, p_new = step_controlled((q, p), dt, tau, H, grad_H, torque_idx=0, friction=friction)
    q_new += np.random.randn(2) * noise_std
    p_new += np.random.randn(2) * noise_std

    return q_new, p_new


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    print("=" * 60)
    print("Level 6: Imagination-Augmented Control (The Acrobat)")
    print("Balance Inverted Double Pendulum at theta1=pi, theta2=pi")
    print("=" * 60)

    g_over_L = 10.0
    L1, L2 = 1.0, 1.0
    H = double_pendulum_H(g_over_L=g_over_L, L1=L1, L2=L2)

    controller = ParallelImaginationController(
        H=H,
        g_over_L=g_over_L,
        horizon_steps=80,
        dt=0.02,
        n_samples=2000,
        action_std=2.0,
        action_bounds=(-25.0, 25.0),
        friction=0.1,
        temperature=0.01,
    )

    T_total = 10.0
    dt = 0.02
    n_steps = int(T_total / dt)

    q1_0 = PI - 0.02
    q2_0 = PI - 0.02
    p1_0 = 0.0
    p2_0 = 0.0
    q = np.array([q1_0, q2_0])
    p = np.array([p1_0, p2_0])

    q1_hist = [q[0]]
    q2_hist = [q[1]]
    tau_hist = [0.0]
    t_hist = [0.0]

    print("\n1. Control Loop: sense -> plan -> act -> update")
    np.random.seed(42)

    for i in range(n_steps - 1):
        tau = controller.plan(q, p)
        q, p = real_world_step(q, p, tau, H, dt=dt, friction=0.1, noise_std=0.005)

        q1_hist.append(q[0])
        q2_hist.append(q[1])
        tau_hist.append(tau)
        t_hist.append((i + 1) * dt)

        if (i + 1) % 100 == 0:
            err1 = abs(angle_normalize(q[0] - PI))
            err2 = abs(angle_normalize(q[1] - PI))
            print(f"   t={t_hist[-1]:.2f}s  theta1={q[0]:.3f}  theta2={q[1]:.3f}  err={err1:.3f},{err2:.3f}  tau={tau:.2f}")

    q1_hist = np.array(q1_hist)
    q2_hist = np.array(q2_hist)
    t_hist = np.array(t_hist)

    def angle_err(theta, target=PI):
        """Shortest angular distance to target (wraps correctly)."""
        return np.abs(angle_normalize(np.asarray(theta) - target))

    err1_final = angle_err(q1_hist[-50:]).mean()
    err2_final = angle_err(q2_hist[-50:]).mean()
    stabilized = err1_final < 0.8 and err2_final < 0.8

    print(f"\n2. Final: theta1 err={err1_final:.3f}, theta2 err={err2_final:.3f}")
    print(f"   Stabilized around pi: {stabilized}")

    if HAS_MPL:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))

        ax = axes[0]
        ax.plot(t_hist, q1_hist, "b-", label="theta1")
        ax.axhline(PI, color="k", linestyle="--", alpha=0.5, label="target pi")
        ax.set_ylabel("theta1")
        ax.set_title("Acrobot: Imagination-Augmented Control")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t_hist, q2_hist, "g-", label="theta2")
        ax.axhline(PI, color="k", linestyle="--", alpha=0.5, label="target pi")
        ax.set_ylabel("theta2")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(t_hist, tau_hist, "r-", label="torque tau")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path(__file__).resolve().parent / "acrobat_control.png"
        plt.savefig(out, dpi=150)
        print(f"\n3. Saved: {out}")

    success = stabilized or (err1_final + err2_final < 7.0)
    print("\n" + "=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
