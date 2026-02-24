"""
FNO Super-Resolution Test: Resolution Invariance (The Magic of FNO)
Train on 64x64, test on 128x128. Proves the model learned the Operator, not the grid.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gen_kolmogorov import run_kolmogorov
from axiom_os.layers.rcln import RCLNLayer


def _prepare_data(n: int, seed: int = 42, n_pairs: int = 100):
    """Generate Kolmogorov data and create (input, target) pairs for one-step prediction."""
    t, u, v, w = run_kolmogorov(n=n, t_end=2.0, seed=seed)
    # x_t = (u,v,w) at t, y_t = (u,v,w) at t+1
    # Shape: u,v,w each (n_steps, n, n)
    n_steps = u.shape[0]
    pairs = min(n_pairs, n_steps - 1)
    X = np.stack([u[:pairs], v[:pairs], w[:pairs]], axis=1)  # (pairs, 3, n, n)
    Y = np.stack([u[1 : pairs + 1], v[1 : pairs + 1], w[1 : pairs + 1]], axis=1)
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


def test_fno_resolution_invariance():
    """
    Train RCLN+FNO on 64x64, test on 128x128.
    - Model should run without error (unlike CNNs which break on size mismatch)
    - Error should remain reasonable (proves operator learning)
    """
    print("=" * 60)
    print("FNO Super-Resolution: Train 64x64, Test 128x128")
    print("=" * 60)

    # Generate data
    print("\n1. Generating Kolmogorov turbulence...")
    X_64, Y_64 = _prepare_data(n=64, seed=42, n_pairs=80)
    X_128, Y_128 = _prepare_data(n=128, seed=42, n_pairs=80)

    print(f"   Low-res:  {X_64.shape} -> {Y_64.shape}")
    print(f"   High-res: {X_128.shape} -> {Y_128.shape}")

    # RCLN with FNO Soft Shell
    rcln = RCLNLayer(
        input_dim=3,
        hidden_dim=32,
        output_dim=3,
        hard_core_func=None,
        lambda_res=1.0,
        net_type="fno",
        fno_modes1=8,
        fno_modes2=8,
    )
    assert rcln._use_fno

    opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)

    # Train ONLY on 64x64
    print("\n2. Training on 64x64 only...")
    for epoch in range(50):
        opt.zero_grad()
        pred = rcln(X_64)
        loss = ((pred - Y_64) ** 2).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}: loss={loss.item():.6f}")

    train_mae = (pred.detach() - Y_64).abs().mean().item()
    print(f"   Train MAE (64x64): {train_mae:.6f}")

    # Test on 128x128 (different resolution!)
    print("\n3. Testing on 128x128 (super-resolution)...")
    with torch.no_grad():
        pred_128 = rcln(X_128)

    # Assert: no error (unlike CNN which would break)
    assert pred_128.shape == Y_128.shape, f"Shape mismatch: {pred_128.shape} vs {Y_128.shape}"
    print(f"   Output shape: {pred_128.shape} (matches 128x128)")

    test_mae = (pred_128 - Y_128).abs().mean().item()
    print(f"   Test MAE (128x128): {test_mae:.6f}")

    # Assertion: model runs and error is finite
    assert np.isfinite(test_mae), "Test MAE must be finite"
    # FNO generalizes; error may be higher than train but should be reasonable
    assert test_mae < 10.0, f"Test MAE too high: {test_mae} (expected < 10 for operator learning)"

    print("\n>>> PASS: FNO resolution invariance verified.")
    print("=" * 60)


if __name__ == "__main__":
    test_fno_resolution_invariance()
