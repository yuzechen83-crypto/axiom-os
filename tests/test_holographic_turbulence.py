"""
Holographic vs Standard RCLN on 2D Kolmogorov Turbulence.
Hypothesis: Holographic RCLN sees turbulence as energy at z≠0.
z=0 learns mean flow (laminar), z→L learns small eddies (turbulent fluctuations).
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gen_kolmogorov import run_kolmogorov  # pyright: ignore[reportMissingImports]
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.layers.holographic import HolographicFNO
from axiom_os.core.optimizer import BreathingScheduler, BreathingOptimizer


def prepare_data(n: int = 32, n_pairs: int = 40, seed: int = 42):
    """Generate Kolmogorov (u,v,w) pairs for one-step prediction."""
    t, u, v, w = run_kolmogorov(n=n, t_end=1.0, dt=0.01, seed=seed)
    n_steps = u.shape[0]
    pairs = min(n_pairs, n_steps - 1)
    X = np.stack([u[:pairs], v[:pairs], w[:pairs]], axis=1)
    Y = np.stack([u[1 : pairs + 1], v[1 : pairs + 1], w[1 : pairs + 1]], axis=1)
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


def train_standard(X: torch.Tensor, Y: torch.Tensor, epochs: int = 50):
    """Standard RCLN with FNO."""
    rcln = RCLNLayer(
        input_dim=3,
        hidden_dim=32,
        output_dim=3,
        lambda_res=1.0,
        net_type="fno",
        fno_modes1=8,
        fno_modes2=8,
    )
    opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        pred = rcln(X)
        loss = ((pred - Y) ** 2).mean()
        loss.backward()
        opt.step()
    return rcln


def train_holographic(X: torch.Tensor, Y: torch.Tensor, epochs: int = 50, use_breathing: bool = False):
    """Holographic FNO with optional BreathingOptimizer (entropy force T(t)·Noise)."""
    holo = HolographicFNO(
        in_channels=3,
        out_channels=3,
        width=32,
        modes1=8,
        modes2=8,
        n_layers=4,
        n_z_slices=7,
        L=1.0,
    )
    if use_breathing:
        opt = BreathingOptimizer(
            torch.optim.Adam(holo.parameters()),
            base_lr=1e-3,
            noise_scale=1e-4,
            use_entropy=True,
        )
    else:
        opt = torch.optim.Adam(holo.parameters(), lr=1e-3)

    for e in range(epochs):
        opt.zero_grad()
        pred = holo(X)
        loss = ((pred - Y) ** 2).mean()
        loss.backward()
        if use_breathing:
            opt.step(loss=loss, training_progress=(e + 1) / epochs)
        else:
            opt.step()
    return holo


def main():
    print("=" * 60)
    print("Holographic vs Standard RCLN - Kolmogorov Turbulence")
    print("=" * 60)

    X, Y = prepare_data(n=32, n_pairs=40)
    print(f"\nData: X {X.shape}, Y {Y.shape}")

    print("\n1. Training Standard RCLN...")
    rcln = train_standard(X, Y, 30)
    with torch.no_grad():
        pred_std = rcln(X)
    mae_std = (pred_std - Y).abs().mean().item()
    print(f"   Train MAE: {mae_std:.6f}")

    print("\n2. Training Holographic RCLN...")
    holo = train_holographic(X, Y, 30)
    with torch.no_grad():
        pred_holo, per_z = holo(X, return_per_slice=True)
    mae_holo = (pred_holo - Y).abs().mean().item()
    print(f"   Train MAE: {mae_holo:.6f}")

    print("\n3. Extrapolation (different seed, future-like)...")
    X_ext, Y_ext = prepare_data(n=32, n_pairs=40, seed=100)
    with torch.no_grad():
        pred_std_ext = rcln(X_ext)
        pred_holo_ext = holo(X_ext)
    mae_std_ext = (pred_std_ext - Y_ext).abs().mean().item()
    mae_holo_ext = (pred_holo_ext - Y_ext).abs().mean().item()
    print(f"   Standard RCLN extrapolation MAE: {mae_std_ext:.6f}")
    print(f"   Holographic RCLN extrapolation MAE: {mae_holo_ext:.6f}")

    print("\n4. Z-axis: per-slice outputs (z=0 vs z→L)...")
    z_grid = holo.get_z_grid().numpy()
    for i, z_val in enumerate(z_grid[::2]):  # every other
        with torch.no_grad():
            out_z, _ = holo(X[:1], z=torch.tensor(z_val), return_per_slice=True)
        mag = out_z.abs().mean().item()
        print(f"   z={z_val:+.2f}: output magnitude {mag:.6f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, (ax, z_val) in enumerate(zip(axes.flat[:7], z_grid)):
            with torch.no_grad():
                out_z, _ = holo(X[:1], z=torch.tensor(z_val), return_per_slice=True)
            out_z = out_z[0, 0].numpy()
            ax.imshow(out_z, cmap="RdBu_r")
            ax.set_title(f"z={z_val:.2f}")
            ax.axis("off")
        axes.flat[-1].axis("off")
        plt.suptitle("Holographic RCLN: Output at different z slices")
        plt.tight_layout()
        out_path = Path(__file__).resolve().parents[1] / "holographic_z_visualization.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\n   Saved: {out_path}")
    except Exception as e:
        print(f"\n   (matplotlib skip: {e})")

    print("\n" + "=" * 60)
    print("MASHD Holographic Upgrade - Evolution Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
