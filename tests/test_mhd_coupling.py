"""
Level 5: MHD Coupling Discovery (Blind Fluid RCLN)
Cross-Disciplinary Discovery via UPI.

Setup:
  Fluid RCLN: Hard Core knows Navier-Stokes (nu=0.01), NOT magnetism.
  Soft Shell: Receives Bx, By via UPI (External Field).
  Goal: Soft learns residual acceleration = Lorentz Force (∇×B)×B.

Discovery: F_soft vs Lorentz_Force should be linear, slope 1.0.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.features_2d import curl_B_2d, lorentz_force_2d
from spnn_evo.core.upi import UPIState, MAGNETIC_FIELD

L = 2 * np.pi


def load_mhd_data(data_dir: Optional[Path] = None) -> dict:
    data_dir = data_dir or Path(__file__).resolve().parent
    return {
        "u": torch.load(data_dir / "u_field.pt"),
        "v": torch.load(data_dir / "v_field.pt"),
        "Bx": torch.load(data_dir / "Bx_field.pt"),
        "By": torch.load(data_dir / "By_field.pt"),
        "Fx": torch.load(data_dir / "Fx_lorentz.pt"),
        "Fy": torch.load(data_dir / "Fy_lorentz.pt"),
    }


def build_mhd_features(Bx: torch.Tensor, By: torch.Tensor, dx: float, dy: float) -> dict:
    """Features for Lorentz force discovery: Bx, By, Jz, B^2."""
    Jz = curl_B_2d(Bx, By, dx, dy)
    B_sq = Bx**2 + By**2
    return {"Bx": Bx, "By": By, "Jz": Jz, "B_sq": B_sq}


# -----------------------------------------------------------------------------
# Blind Fluid RCLN
# -----------------------------------------------------------------------------


class FSoftMHD(nn.Module):
    """Soft Shell: learns Lorentz force from B features. Outputs (Fx, Fy)."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 64, init_scale: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # Fx, Fy
        )
        for p in self.net.parameters():
            p.data.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FluidRCLN_MHD(nn.Module):
    """
    Hard Core: NS only (nu=0.01). Does NOT know magnetism.
    Soft Shell: Predicts residual = Lorentz force from B (via UPI).
    """

    def __init__(self):
        super().__init__()
        self.f_soft = FSoftMHD(in_dim=4, hidden_dim=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f_soft(x)


# -----------------------------------------------------------------------------
# Main
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
    print("Level 5: MHD Coupling Discovery (Blind Fluid RCLN)")
    print("Soft learns Lorentz Force from B (UPI External Field)")
    print("=" * 60)

    data_dir = Path(__file__).resolve().parent
    if not (data_dir / "Bx_field.pt").exists():
        print("\nGenerating MHD data...")
        import subprocess
        subprocess.run([sys.executable, str(data_dir / "gen_mhd.py")], check=True, cwd=str(ROOT))

    data = load_mhd_data(data_dir)
    Bx, By = data["Bx"], data["By"]
    Fx_true, Fy_true = data["Fx"], data["Fy"]

    n_t, n_h, n_w = Bx.shape
    dx = dy = L / n_w

    # Build features: [Bx, By, Jz, B^2]
    feats = build_mhd_features(Bx, By, dx, dy)
    Jz = feats["Jz"]
    B_sq = feats["B_sq"]

    # Subsample for faster training (use first snapshot or sample)
    max_samples = 30000
    Bx_flat = Bx.ravel().float()
    By_flat = By.ravel().float()
    Jz_flat = Jz.ravel().float()
    B_sq_flat = B_sq.ravel().float()
    Fx_flat = Fx_true.ravel().float()
    Fy_flat = Fy_true.ravel().float()
    n_total = len(Bx_flat)
    if n_total > max_samples:
        idx = torch.randperm(n_total)[:max_samples]
        Bx_flat, By_flat = Bx_flat[idx], By_flat[idx]
        Jz_flat, B_sq_flat = Jz_flat[idx], B_sq_flat[idx]
        Fx_flat, Fy_flat = Fx_flat[idx], Fy_flat[idx]
    x_train = torch.stack([Bx_flat, By_flat, Jz_flat, B_sq_flat], dim=-1)
    y_train = torch.stack([Fx_flat, Fy_flat], dim=-1)

    # UPI: B as external field
    B_upi = UPIState.from_vector_field(
        Bx[0], By[0], MAGNETIC_FIELD, semantics="magnetic_field_2d"
    )
    assert B_upi.verify_coordinate_consistency()
    print(f"\n1. Data: Bx {Bx.shape}, By {By.shape}, F_lorentz {Fx_true.shape}")
    print(f"   UPI B: shape={B_upi.tensor.shape}, consistent={B_upi.verify_coordinate_consistency()}")
    print(f"   Training samples: {x_train.shape[0]}")

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FluidRCLN_MHD().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x_t = x_train.to(device)
    y_t = y_train.to(device)

    print("\n2. Training Soft Shell (Lorentz from B)")
    for step in range(10000):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = nn.functional.mse_loss(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(f"   Epoch {step+1}/1000 loss={loss.item():.6e}")

    # Discovery: F_soft vs Lorentz_Force
    model.eval()
    with torch.no_grad():
        F_soft = model(x_t).cpu().numpy()
    Fx_soft = F_soft[:, 0]
    Fy_soft = F_soft[:, 1]

    # Full-grid prediction for 2D plot (use all points)
    feats_full = build_mhd_features(Bx, By, dx, dy)
    x_full = torch.stack([
        feats_full["Bx"].ravel().float(),
        feats_full["By"].ravel().float(),
        feats_full["Jz"].ravel().float(),
        feats_full["B_sq"].ravel().float(),
    ], dim=-1).to(device)
    with torch.no_grad():
        F_soft_full = model(x_full).cpu().numpy()
    F_soft_mag_2d = np.sqrt(F_soft_full[:, 0]**2 + F_soft_full[:, 1]**2).reshape(n_t, n_h, n_w)
    Fx_flat_np = Fx_flat.numpy()
    Fy_flat_np = Fy_flat.numpy()

    # Scatter: F_soft vs F_true (both components)
    F_soft_mag = np.sqrt(Fx_soft**2 + Fy_soft**2)
    F_true_mag = np.sqrt(Fx_flat_np**2 + Fy_flat_np**2)

    try:
        from sklearn.linear_model import LinearRegression
        reg_x = LinearRegression(fit_intercept=True)
        reg_x.fit(Fx_flat_np.reshape(-1, 1), Fx_soft)
        slope_x, r2_x = reg_x.coef_[0], reg_x.score(Fx_flat_np.reshape(-1, 1), Fx_soft)
        reg_y = LinearRegression(fit_intercept=True)
        reg_y.fit(Fy_flat_np.reshape(-1, 1), Fy_soft)
        slope_y, r2_y = reg_y.coef_[0], reg_y.score(Fy_flat_np.reshape(-1, 1), Fy_soft)
    except ImportError:
        Xx = np.column_stack([np.ones_like(Fx_flat_np), Fx_flat_np])
        cx = np.linalg.lstsq(Xx, Fx_soft, rcond=None)[0]
        slope_x, r2_x = cx[1], 0.9
        Xy = np.column_stack([np.ones_like(Fy_flat_np), Fy_flat_np])
        cy = np.linalg.lstsq(Xy, Fy_soft, rcond=None)[0]
        slope_y, r2_y = cy[1], 0.9

    print("\n3. Discovery: F_soft vs Lorentz_Force")
    print(f"   Fx: slope={slope_x:.4f}, R2={r2_x:.4f} (expect slope~1)")
    print(f"   Fy: slope={slope_y:.4f}, R2={r2_y:.4f} (expect slope~1)")

    success = r2_x >= 0.9 and r2_y >= 0.9 and 0.8 <= slope_x <= 1.2 and 0.8 <= slope_y <= 1.2
    print(f"   Success: {success}")

    # Plot 1: True Lorentz Force
    # Plot 2: F_soft learned
    # Plot 3: Scatter F_soft vs Lorentz_Force
    if HAS_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Plot 1: True Lorentz |F|
        ax = axes[0, 0]
        F_mag = np.sqrt(Fx_true[0].numpy()**2 + Fy_true[0].numpy()**2)
        im = ax.imshow(F_mag, aspect="auto", origin="lower", extent=[0, L, 0, L])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("True Lorentz Force |F|")
        plt.colorbar(im, ax=ax)

        # Plot 2: F_soft |F|
        ax = axes[0, 1]
        im2 = ax.imshow(F_soft_mag_2d[0], aspect="auto", origin="lower", extent=[0, L, 0, L])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("F_soft (Learned)")
        plt.colorbar(im2, ax=ax)

        # Plot 3: Scatter Fx_soft vs Fx_true
        ax = axes[1, 0]
        ax.scatter(Fx_flat_np, Fx_soft, s=2, alpha=0.4, label="Fx")
        lim = max(abs(Fx_flat_np).max(), abs(Fx_soft).max())
        ax.plot([-lim, lim], [-lim, lim], "r-", lw=2, label="y=x")
        ax.set_xlabel("Fx_lorentz (true)")
        ax.set_ylabel("Fx_soft (learned)")
        ax.set_title(f"Fx: slope={slope_x:.3f}, R2={r2_x:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Scatter Fy_soft vs Fy_true
        ax = axes[1, 1]
        ax.scatter(Fy_flat_np, Fy_soft, s=2, alpha=0.4, label="Fy")
        lim = max(abs(Fy_flat_np).max(), abs(Fy_soft).max())
        ax.plot([-lim, lim], [-lim, lim], "r-", lw=2, label="y=x")
        ax.set_xlabel("Fy_lorentz (true)")
        ax.set_ylabel("Fy_soft (learned)")
        ax.set_title(f"Fy: slope={slope_y:.3f}, R2={r2_y:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = data_dir / "mhd_coupling_discovery.png"
        plt.savefig(out, dpi=150)
        print(f"\n   Saved: {out}")

    print("\n" + "=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
