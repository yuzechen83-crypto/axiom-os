"""
GENERIC Damped Oscillator Test
Ground Truth: dx/dt = v, dv/dt = -k*x - gamma*v
Goal: Identify L (symplectic) and M (friction) separately from data.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.core.generic import GENERICSystem
from axiom_os.engine.generic_loss import generic_loss_from_aux
from axiom_os.engine.discovery_generic import discover_generic


def generate_damped_oscillator(
    n_samples: int = 500,
    k: float = 1.0,
    gamma: float = 0.2,
    dt: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate damped oscillator trajectories.
    z = [x, v], dz/dt = [v, -k*x - gamma*v]
    Returns (z, dz_dt) each (n_samples, 2)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    z_list = []
    dz_list = []
    x, v = 1.0, 0.0
    for _ in range(n_samples):
        z_list.append([x, v])
        dz_list.append([v, -k * x - gamma * v])
        dx = v * dt
        dv = (-k * x - gamma * v) * dt
        x = x + dx
        v = v + dv

    return np.array(z_list, dtype=np.float32), np.array(dz_list, dtype=np.float32)


def test_generic_damped_oscillator():
    """Train GENERIC on damped oscillator, verify L and M structure."""
    print("=" * 60)
    print("GENERIC Damped Oscillator Test")
    print("=" * 60)

    # Generate data
    z_data, dz_data = generate_damped_oscillator(n_samples=400, gamma=0.2)
    z_t = torch.from_numpy(z_data).float()
    dz_t = torch.from_numpy(dz_data).float()

    print(f"\nData: z {z_data.shape}, dz {dz_data.shape}")

    # GENERIC system: state_dim=2, entropy_dim=1 (learn S for dissipation)
    model = GENERICSystem(state_dim=2, hidden_dim=32, entropy_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train with generic loss
    print("\nTraining with GENERIC loss (data + degeneracy)...")
    for epoch in range(300):
        opt.zero_grad()
        z_dot, aux = model(z_t, return_aux=True)
        loss = generic_loss_from_aux(z_dot, dz_t, aux, lambda_phys=0.5)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            mse = ((z_dot - dz_t) ** 2).mean().item()
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}, MSE={mse:.6f}")

    # Verify structure
    with torch.no_grad():
        L = model.get_L(z_t[:1]).squeeze(0).numpy()
        M = model.get_M(z_t[:1]).squeeze(0).numpy()

    # L must be anti-symmetric: L + L^T = 0
    L_skew = np.max(np.abs(L + L.T))
    assert L_skew < 0.1, f"L should be anti-symmetric, ||L+L^T||={L_skew}"

    # M must be PSD: eigenvalues >= 0
    eig_M = np.linalg.eigvalsh(M)
    assert np.all(eig_M >= -1e-5), f"M should be PSD, eigenvalues={eig_M}"

    print("\n>>> Structure verified:")
    print(f"  L (anti-symmetric):\n{L}")
    print(f"  M (PSD):\n{M}")
    print(f"  ||L + L^T|| = {L_skew:.6f}")
    print(f"  M eigenvalues = {eig_M}")

    # Discovery: extract E, S, L, M
    result = discover_generic(model, z_data, var_names=["x", "v"])
    print("\n>>> Discovery:")
    fe = result['formula_E'] or 'None'
    fs = result['formula_S'] or 'None'
    print(f"  formula_E: {fe[:80]}{'...' if len(fe) > 80 else ''}")
    print(f"  formula_S: {fs[:80]}{'...' if len(fs) > 80 else ''}")
    print(f"  has_friction (M non-zero): {result['has_friction']}")

    # Final MSE (model.dynamics uses autograd, so we need grad enabled)
    z_dot_pred = model(z_t)
    mse_final = ((z_dot_pred.detach() - dz_t) ** 2).mean().item()
    assert mse_final < 0.1, f"MSE too high: {mse_final}"
    print(f"\n>>> PASS: MSE={mse_final:.6f}, L anti-symmetric, M PSD")
    print("=" * 60)


if __name__ == "__main__":
    test_generic_damped_oscillator()
