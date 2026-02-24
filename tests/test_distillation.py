"""
Quick smoke test for Policy Distillation (Muscle Memory).
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.imagination import double_pendulum_H
from spnn_evo.orchestrator.mpc import ParallelImaginationController
from spnn_evo.orchestrator.distillation import (
    PolicyTrainer,
    StudentPolicy,
    DoublePendulumEnv,
    ReplayBuffer,
)


def test_distillation_smoke():
    """Smoke test: collect data, train, infer."""
    H = double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
    mpc = ParallelImaginationController(
        H=H,
        horizon_steps=20,
        n_samples=50,  # Minimal for speed
        cost_mode="stabilization",
    )
    env = DoublePendulumEnv(H=H, dt=0.02, friction=0.1, noise_std=0.0, max_steps=50)
    trainer = PolicyTrainer(mpc=mpc, env=env)

    # Collect
    n = trainer.collect_mpc_data(n_episodes=3, verbose=False)
    assert n > 0 and len(trainer.buffer) > 0

    # Train
    losses = trainer.train_student(epochs=10, batch_size=64, verbose=False)
    assert len(losses) == 10 and losses[-1] < losses[0]

    # Infer
    obs = np.array([np.pi - 0.1, np.pi - 0.1, 0.05, 0.05], dtype=np.float32)
    action = trainer.student.act(obs)
    assert isinstance(action, (float, np.floating))
    assert -25 <= action <= 25

    # Latency
    latency_ms = trainer.infer_latency_ms(n_trials=100)
    assert latency_ms < 10.0  # Should be well under 10ms

    print("test_distillation_smoke: OK")
    return True


if __name__ == "__main__":
    success = test_distillation_smoke()
    sys.exit(0 if success else 1)
