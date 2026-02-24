"""
PDE Turbulence Discovery Test
SPNN-Evo 2.0: Discover viscosity term from Burgers equation.

Physics: Burgers ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
  Diffusion term: D = ν·u_xx (Newtonian viscosity)

System Knowledge: Unknown viscosity.
Task: Soft Shell learns D(u_xx) from data.
Discovery: D_soft vs u_xx should be a straight line; slope = ν.
PDE Library: Θ = [u, u², ux, uxx, u·ux, ux²]
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.features import (
    compute_derivatives_2d,
    build_pde_library,
    sparse_regression_pde,
)


# -----------------------------------------------------------------------------
# 1. Burgers Turbulence Data: u(t,x), ux, uxx, D = ν·u_xx
# -----------------------------------------------------------------------------

def generate_burgers_viscosity_data(
    n_t: int = 50,
    n_x: int = 64,
    L: float = 1.0,
    nu: float = 0.02,
    n_modes: int = 8,
    seed: Optional[int] = None,
) -> tuple:
    """
    Synthetic 1D Burgers turbulence.
    Target: D = ν·u_xx (viscous diffusion term)
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, 1, n_t)
    x = np.linspace(0, L, n_x)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    u = np.zeros_like(tt)
    for k in range(1, n_modes + 1):
        A = 1.0 / k * np.exp(-0.1 * k)
        decay = np.exp(-nu * (k * np.pi / L) ** 2 * tt)
        u += A * np.sin(k * np.pi * xx / L) * decay
    for k in range(2, min(n_modes + 2, 12)):
        phase = np.random.rand() * 2 * np.pi
        amp = 0.08 * (k ** (-5 / 3))
        u += amp * np.sin(k * np.pi * xx / L + 2 * np.pi * tt / 0.5 + phase)
    u += 0.02 * np.random.randn(*u.shape) * np.exp(-tt)

    ux, uxx = compute_derivatives_2d(u, t, x)
    D_target = nu * uxx
    return tt, xx, u, ux, uxx, D_target


# -----------------------------------------------------------------------------
# 2. Model: Soft learns D(u, ux, uxx)
# -----------------------------------------------------------------------------

class FSoftPDE(nn.Module):
    """Soft: Learns dissipation D from PDE features (u, ux, uxx)"""

    def __init__(self, in_dim: int = 3, hidden_dim: int = 64, init_scale: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        for p in self.net.parameters():
            p.data.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RCLNTurbulence(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_soft = FSoftPDE(in_dim=3, hidden_dim=64)

    def forward(self, x: torch.Tensor, return_components: bool = False):
        y_soft = self.f_soft(x)
        if return_components:
            return y_soft, torch.zeros_like(y_soft), y_soft
        return y_soft


# -----------------------------------------------------------------------------
# 3. Discovery: PDE Library + D_soft vs u_xx plot
# -----------------------------------------------------------------------------

def extract_soft_vs_uxx(
    model: RCLNTurbulence,
    u: np.ndarray,
    ux: np.ndarray,
    uxx: np.ndarray,
    device=None,
) -> tuple:
    """Get D_soft for all (u, ux, uxx) points."""
    device = device or torch.device("cpu")
    model.eval()
    state = np.stack([u.ravel(), ux.ravel(), uxx.ravel()], axis=-1).astype(np.float32)
    x_t = torch.from_numpy(state).to(device)
    with torch.no_grad():
        _, _, d_soft = model(x_t, return_components=True)
    return uxx.ravel(), d_soft.cpu().numpy().ravel(), state


# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------

def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    print("=" * 60)
    print("SPNN-Evo 2.0: PDE Turbulence Discovery")
    print("Target: D = nu * u_xx (Newtonian viscosity)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nu = 0.02
    n_t, n_x = 50, 64

    # 1. Data
    print("\n1. Burgers Data Generation")
    tt, xx, u, ux, uxx, D_target = generate_burgers_viscosity_data(
        n_t=n_t, n_x=n_x, nu=nu, n_modes=8, seed=42
    )
    state = np.stack([u.ravel(), ux.ravel(), uxx.ravel()], axis=-1).astype(np.float32)
    target = D_target.ravel().reshape(-1, 1).astype(np.float32)
    state_t = torch.from_numpy(state).to(device)
    target_t = torch.from_numpy(target).to(device)
    print(f"   {state.shape[0]} samples, nu={nu}")

    # 2. PDE Library
    print("\n2. PDE Library Construction")
    Theta, feat_names = build_pde_library(u, ux, uxx)
    print(f"   Features: {feat_names}")

    # 3. Model & Training
    print("\n3. Training")
    model = RCLNTurbulence().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(2000):
        optimizer.zero_grad()
        pred, _, _ = model(state_t, return_components=True)
        loss = nn.functional.mse_loss(pred, target_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(f"   Epoch {step+1}/2000 loss={loss.item():.6f}")

    # 4. Discovery: D_soft vs u_xx
    print("\n4. Discovery: D_soft vs u_xx (Curvature/Diffusion)")
    uxx_flat, d_soft_vals, _ = extract_soft_vs_uxx(model, u, ux, uxx, device)

    # Regression on PDE library
    xi, formula = sparse_regression_pde(Theta, d_soft_vals, feat_names, threshold=0.005)
    print(f"   Discovered: {formula}")

    # Find u_xx coefficient
    uxx_idx = next((i for i, n in enumerate(feat_names) if "u_xx" in n or "xx" in n), 3)
    nu_discovered = xi[uxx_idx]
    print(f"   u_xx coefficient (viscosity nu): {nu_discovered:.4f} (expected {nu})")

    # Linear fit D_soft vs u_xx (Newtonian: straight line)
    uxx_col = uxx_flat.reshape(-1, 1)
    try:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(uxx_col, d_soft_vals)
        slope = reg.coef_[0]
        d_fit = reg.predict(uxx_col)
    except ImportError:
        slope = np.linalg.lstsq(uxx_col, d_soft_vals, rcond=None)[0][0]
        d_fit = uxx_col.ravel() * slope

    ss_res = np.sum((d_soft_vals - d_fit) ** 2)
    ss_tot = np.sum((d_soft_vals - np.mean(d_soft_vals)) ** 2) + 1e-10
    r2 = 1 - ss_res / ss_tot
    success = r2 > 0.85 and abs(slope - nu) < 0.05
    print(f"   D_soft vs u_xx slope: {slope:.4f}, R2: {r2:.4f}")
    print(f"   Success: {success}")

    # 5. Visualization
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # D_soft vs u_xx (main discovery plot)
        ax = axes[0]
        ax.scatter(uxx_flat, d_soft_vals, s=6, alpha=0.6, label="D_soft")
        uxx_line = np.linspace(uxx_flat.min(), uxx_flat.max(), 100)
        d_line = slope * uxx_line
        ax.plot(uxx_line, d_line, "r-", lw=2, label=f"Fit: D = {slope:.3f}*u_xx")
        ax.axhline(0, color="k", ls=":", alpha=0.5)
        ax.axvline(0, color="k", ls=":", alpha=0.5)
        ax.set_xlabel("u_xx (Curvature / Diffusion)")
        ax.set_ylabel("D_soft (Learned Dissipation)")
        ax.set_title("PDE Discovery: D_soft vs u_xx (Newtonian: straight line)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # u(t,x) field
        ax = axes[1]
        im = ax.imshow(u.T, aspect="auto", origin="lower", extent=[0, 1, 0, 1])
        ax.set_xlabel("Time t")
        ax.set_ylabel("Space x")
        ax.set_title("u(t,x) - Burgers Field")
        plt.colorbar(im, ax=ax)

        # u_xx field
        ax = axes[2]
        im2 = ax.imshow(uxx.T, aspect="auto", origin="lower", extent=[0, 1, 0, 1])
        ax.set_xlabel("Time t")
        ax.set_ylabel("Space x")
        ax.set_title("u_xx - Curvature")
        plt.colorbar(im2, ax=ax)

        plt.tight_layout()
        out = ROOT / "tests" / "turbulence_discovery.png"
        plt.savefig(out, dpi=150)
        print(f"\n   Saved: {out}")

    print("\n" + "=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
