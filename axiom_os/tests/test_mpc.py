"""
Unit tests for Imagination MPC (The Acrobat).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.orchestrator.mpc import (
    ImaginationMPC,
    angle_normalize,
    double_pendulum_H,
)

PI = np.pi


def test_angle_normalize():
    """Test 0 ≡ 2π wrap-around."""
    assert abs(angle_normalize(0) - 0) < 1e-10
    assert abs(angle_normalize(2 * PI) - 0) < 1e-10
    assert abs(angle_normalize(-2 * PI) - 0) < 1e-10
    assert abs(abs(angle_normalize(PI)) - PI) < 1e-10  # PI or -PI both valid
    assert abs(angle_normalize(3 * PI) - (-PI)) < 1e-10


def test_energy_shaping_cost():
    """Test energy shaping cost."""
    H = double_pendulum_H(g_over_L=10.0)
    target_E = 20.0
    state = np.array([PI, PI, 0.0, 0.0])  # Upright
    E = H(state)
    cost = ImaginationMPC.energy_shaping_cost(state, target_E, H)
    assert cost == (E - target_E) ** 2


def test_stabilization_cost():
    """Test stabilization cost."""
    q_traj = np.array([[PI, PI], [PI - 0.1, PI - 0.1]])
    p_traj = np.array([[0.0, 0.0], [0.1, 0.1]])
    target = np.array([PI, PI])
    cost = ImaginationMPC.stabilization_cost((q_traj, p_traj), target)
    assert cost >= 0


def test_plan_returns_torque():
    """Test plan returns a scalar torque."""
    mpc = ImaginationMPC(
        n_samples=20,  # Reduced for speed
        horizon_steps=10,
    )
    q = np.array([PI - 0.1, PI - 0.1])
    p = np.array([0.05, 0.05])
    tau = mpc.plan(q, p)
    assert isinstance(tau, (float, np.floating))
    assert -25 <= tau <= 25


if __name__ == "__main__":
    test_angle_normalize()
    test_energy_shaping_cost()
    test_stabilization_cost()
    test_plan_returns_torque()
    print("All MPC tests passed.")
