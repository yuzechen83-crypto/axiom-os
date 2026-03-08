"""
Unit tests for MASHD components: HolographicRCLN, BreathingOptimizer.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.layers.holographic import HolographicRCLN, HolographicProjectionLayer
from axiom_os.core.optimizer import BreathingOptimizer, BreathingScheduler


def test_holographic_rcln_forward():
    """HolographicRCLN: y = y_hard + λ·∫K(z)·Ψ(x,z)dz"""
    def hard_core(x):
        v = torch.as_tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x.float()
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return v[:, :1] * 0.5  # simple: 0.5 * first col

    holo = HolographicRCLN(
        input_dim=4,
        hidden_dim=32,
        output_dim=2,
        hard_core_func=hard_core,
        lambda_res=1.0,
        n_z_slices=5,
        L=1.0,
    )
    x = torch.randn(10, 4)
    y = holo(x)
    assert y.shape == (10, 2)
    assert holo._last_y_soft is not None
    assert holo._last_y_soft.shape == (10, 2)


def test_holographic_rcln_no_hard_core():
    """HolographicRCLN without hard core."""
    holo = HolographicRCLN(input_dim=3, hidden_dim=16, output_dim=1, lambda_res=1.0)
    x = torch.randn(5, 3)
    y = holo(x)
    assert y.shape == (5, 1)


def test_breathing_optimizer_step():
    """BreathingOptimizer: entropy force + LR scheduling."""
    model = torch.nn.Linear(4, 2)
    opt = BreathingOptimizer(
        torch.optim.Adam(model.parameters()),
        base_lr=1e-3,
        noise_scale=1e-5,
        use_entropy=True,
    )
    x = torch.randn(20, 4)
    y = torch.randn(20, 2)
    for _ in range(10):
        opt.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step(loss=loss, training_progress=0.5)
    assert opt.get_Z() != 0
    assert opt.get_noise_scale() >= 0


def test_breathing_scheduler_lr():
    """BreathingScheduler updates LR via Duffing Z."""
    model = torch.nn.Linear(2, 1)
    base = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = BreathingScheduler(base, base_lr=1e-3)
    lr0 = base.param_groups[0]["lr"]
    sched.step(training_progress=0.8)
    lr1 = base.param_groups[0]["lr"]
    assert lr1 != lr0


if __name__ == "__main__":
    test_holographic_rcln_forward()
    test_holographic_rcln_no_hard_core()
    test_breathing_optimizer_step()
    test_breathing_scheduler_lr()
    print("All MASHD tests passed.")
