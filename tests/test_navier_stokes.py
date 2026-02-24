"""
2D Navier-Stokes Vorticity Discovery
SPNN-Evo 2.0: Discover viscosity from vorticity equation.

Physics: ∂ω/∂t + (u·∇)ω = ν∇²ω
  ω = ∇×u⃗ = v_x - u_y (vorticity)
  Diffusion target: D = ν∇²ω

System Knowledge: Unknown viscosity.
Task: Soft Shell learns D(∇²ω) from vorticity data.
Plot: D_soft vs ∇²ω (expect linear, slope = ν).

Data: Kolmogorov Flow (periodic) or Lid-driven Cavity (bounded).
"""

import sys
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.features import (
    VectorFeatureExtractor,
    build_vorticity_library,
    sparse_regression_pde,
)
from spnn_evo.core.upi import UPIState, VELOCITY


# -----------------------------------------------------------------------------
# 1. 2D Flow Data: Kolmogorov Flow & Lid-driven Cavity
# -----------------------------------------------------------------------------

def generate_kolmogorov_flow(
    n_t: int = 20,
    n_h: int = 32,
    n_w: int = 32,
    L: float = 2 * np.pi,
    nu: float = 0.01,
    n_modes: int = 5,
    seed: Optional[int] = None,
) -> tuple:
    """
    Synthetic 2D Kolmogorov-style flow u=(u,v).
    Periodic domain. Target: D = ν∇²ω (viscous diffusion in vorticity eq).
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, 1, n_t)
    y = np.linspace(0, L, n_h)
    x = np.linspace(0, L, n_w)
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    u = np.zeros_like(tt)
    v = np.zeros_like(tt)
    for kx in range(1, n_modes + 1):
        for ky in range(1, n_modes + 1):
            A = 0.1 / (kx * ky)
            phase = np.random.rand() * 2 * np.pi
            decay = np.exp(-nu * (kx**2 + ky**2) * tt)
            u += A * np.sin(kx * xx) * np.cos(ky * yy) * np.cos(2 * np.pi * tt + phase) * decay
            v += -A * np.cos(kx * xx) * np.sin(ky * yy) * np.cos(2 * np.pi * tt + phase) * decay
    u += 0.01 * np.random.randn(*u.shape)
    v += 0.01 * np.random.randn(*v.shape)

    dx = x[1] - x[0] if len(x) > 1 else L / n_w
    dy = y[1] - y[0] if len(y) > 1 else L / n_h
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    extractor = VectorFeatureExtractor(dx=dx, dy=dy, dt=dt)
    features = extractor.extract(u, v)

    D_target = nu * features["laplacian_omega"]
    return u, v, features, D_target, (t, y, x)


def generate_lid_driven_cavity(
    n_t: int = 15,
    n_h: int = 32,
    n_w: int = 32,
    L: float = 1.0,
    nu: float = 0.01,
    U_lid: float = 1.0,
    seed: Optional[int] = None,
) -> tuple:
    """
    Synthetic 2D Lid-driven Cavity flow u=(u,v).
    Square domain [0,L]², top lid moves with U_lid. No-slip walls.
    Uses polynomial/trig basis to approximate cavity recirculation.
    Target: D = ν∇²ω (viscous diffusion in vorticity eq).
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, 0.5, n_t)
    y = np.linspace(0, L, n_h)
    x = np.linspace(0, L, n_w)
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")

    # Streamfunction-like basis: ψ ∝ sin(πy/L)*x*(L-x) for cavity
    # u = ∂ψ/∂y, v = -∂ψ/∂x. Add lid velocity at top.
    eta_y = yy / L
    eta_x = xx / L
    # Lid profile: parabolic near walls, U_lid in center
    lid_profile = 4 * eta_x * (1 - eta_x)  # sin²-like
    u_lid = U_lid * lid_profile * (eta_y > 0.95).astype(float)

    # Interior: decaying modes
    u = np.zeros_like(tt)
    v = np.zeros_like(tt)
    for k in range(1, 5):
        decay = np.exp(-nu * (k * np.pi / L) ** 2 * tt)
        u += 0.1 / k * np.sin(k * np.pi * eta_x) * np.cos(k * np.pi * eta_y) * decay
        v += -0.1 / k * np.cos(k * np.pi * eta_x) * np.sin(k * np.pi * eta_y) * decay
    u += u_lid
    u += 0.005 * np.random.randn(*u.shape)
    v += 0.005 * np.random.randn(*v.shape)

    dx = x[1] - x[0] if len(x) > 1 else L / n_w
    dy = y[1] - y[0] if len(y) > 1 else L / n_h
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    extractor = VectorFeatureExtractor(dx=dx, dy=dy, dt=dt)
    features = extractor.extract(u, v)

    D_target = nu * features["laplacian_omega"]
    return u, v, features, D_target, (t, y, x)


def load_or_generate_2d_flow(
    dataset: Literal["kolmogorov", "cavity"] = "kolmogorov",
    n_t: int = 20,
    n_h: int = 32,
    n_w: int = 32,
    nu: float = 0.01,
    seed: Optional[int] = 42,
) -> tuple:
    """Load or generate 2D flow. Returns (u, v, features, D_target, (t,y,x))."""
    if dataset == "cavity":
        return generate_lid_driven_cavity(
            n_t=n_t, n_h=n_h, n_w=n_w, nu=nu, seed=seed
        )
    return generate_kolmogorov_flow(
        n_t=n_t, n_h=n_h, n_w=n_w, nu=nu, n_modes=5, seed=seed
    )


# -----------------------------------------------------------------------------
# 2. Model: Soft learns D(omega, lap_omega, advection)
# -----------------------------------------------------------------------------

class FSoftVorticity(nn.Module):
    """Soft: Learns diffusion D from vorticity features"""

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


class RCLNVorticity(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_soft = FSoftVorticity(in_dim=3, hidden_dim=64)

    def forward(self, x: torch.Tensor, return_components: bool = False):
        y_soft = self.f_soft(x)
        if return_components:
            return y_soft, torch.zeros_like(y_soft), y_soft
        return y_soft


# -----------------------------------------------------------------------------
# 3. Main
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
    print("SPNN-Evo 2.0: 2D Navier-Stokes Vorticity Discovery")
    print("Target: D = nu * laplacian(omega) = nu * lap(omega)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nu = 0.01
    n_t, n_h, n_w = 20, 32, 32
    dataset = "kolmogorov"  # "kolmogorov" or "cavity" (Lid-driven Cavity)

    # 1. Data (Kolmogorov Flow or Lid-driven Cavity)
    print(f"\n1. 2D Flow: {dataset}")
    u, v, features, D_target, (t, y, x) = load_or_generate_2d_flow(
        dataset=dataset, n_t=n_t, n_h=n_h, n_w=n_w, nu=nu, seed=42
    )
    omega = features["curl_u"]
    lap_omega = features["laplacian_omega"]
    advection = features["advection"]
    div_u = features["div_u"]
    print(f"   u,v shape: {u.shape}, |div_u| mean: {np.abs(div_u).mean():.2e} (incompressible ~0)")

    # Vector UPI verification (field + rank-1 single vector)
    u_t = torch.from_numpy(u[0]).double()
    v_t = torch.from_numpy(v[0]).double()
    upi_vel = UPIState.from_vector_field(u_t, v_t, VELOCITY, semantics="velocity_2d")
    assert upi_vel.verify_coordinate_consistency(), "Vector UPI coordinate consistency failed"
    print(f"   Vector UPI (field): shape={upi_vel.tensor.shape}, consistent={upi_vel.verify_coordinate_consistency()}")

    # Rank-1 single vector (2,) or (3,)
    upi_single = UPIState.from_single_vector(
        torch.tensor([1.0, 0.0]), VELOCITY, semantics="unit_x"
    )
    assert upi_single.verify_coordinate_consistency(), "Rank-1 vector UPI failed"
    print(f"   Vector UPI (single): shape={upi_single.tensor.shape}, consistent={upi_single.verify_coordinate_consistency()}")

    state = np.stack([omega.ravel(), lap_omega.ravel(), advection.ravel()], axis=-1).astype(np.float32)
    target = D_target.ravel().reshape(-1, 1).astype(np.float32)
    state_t = torch.from_numpy(state).to(device)
    target_t = torch.from_numpy(target).to(device)
    print(f"   {state.shape[0]} samples, nu={nu}")

    # 2. Vorticity Library
    print("\n2. Vorticity Library")
    Theta, feat_names = build_vorticity_library(omega, lap_omega, advection)
    print(f"   Features: {feat_names}")

    # 3. Training
    print("\n3. Training")
    model = RCLNVorticity().to(device)
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

    # 4. Discovery: D_soft vs ∇²ω
    print("\n4. Discovery: D_soft vs laplacian(omega)")
    model.eval()
    with torch.no_grad():
        _, _, d_soft = model(state_t, return_components=True)
    d_soft_vals = d_soft.cpu().numpy().ravel()
    lap_flat = lap_omega.ravel()

    xi, formula = sparse_regression_pde(Theta, d_soft_vals, feat_names, threshold=0.001)
    print(f"   Discovered: {formula}")

    lap_idx = next((i for i, n in enumerate(feat_names) if "lap" in n or "laplacian" in n), 2)
    nu_discovered = xi[lap_idx]
    print(f"   laplacian_omega coefficient (nu): {nu_discovered:.4f} (expected {nu})")

    # Linear fit D_soft vs lap_omega
    lap_col = lap_flat.reshape(-1, 1)
    try:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(lap_col, d_soft_vals)
        slope = reg.coef_[0]
        d_fit = reg.predict(lap_col)
    except ImportError:
        slope = np.linalg.lstsq(lap_col, d_soft_vals, rcond=None)[0][0]
        d_fit = lap_col.ravel() * slope

    ss_res = np.sum((d_soft_vals - d_fit) ** 2)
    ss_tot = np.sum((d_soft_vals - np.mean(d_soft_vals)) ** 2) + 1e-10
    r2 = 1 - ss_res / ss_tot
    success = r2 > 0.80 and abs(slope - nu) < 0.02
    print(f"   D_soft vs lap(omega) slope: {slope:.4f}, R2: {r2:.4f}")
    print(f"   Success: {success}")

    # 5. Visualization
    if HAS_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        ax = axes[0, 0]
        ax.scatter(lap_flat, d_soft_vals, s=4, alpha=0.5, label="D_soft")
        lap_line = np.linspace(lap_flat.min(), lap_flat.max(), 100)
        ax.plot(lap_line, slope * lap_line, "r-", lw=2, label=f"D = {slope:.3f}*lap(omega)")
        ax.axhline(0, color="k", ls=":", alpha=0.5)
        ax.axvline(0, color="k", ls=":", alpha=0.5)
        ax.set_xlabel("laplacian(omega) = ∇²ω")
        ax.set_ylabel("D_soft (Learned Diffusion)")
        ax.set_title("NS Discovery: D_soft vs ∇²ω")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        im = ax.imshow(omega[0], aspect="auto", origin="lower", extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Vorticity ω (t=0)")
        plt.colorbar(im, ax=ax)

        ax = axes[1, 0]
        im2 = ax.imshow(lap_omega[0], aspect="auto", origin="lower", extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("∇²ω (t=0)")
        plt.colorbar(im2, ax=ax)

        ax = axes[1, 1]
        speed = np.sqrt(u[0]**2 + v[0]**2)
        im3 = ax.imshow(speed, aspect="auto", origin="lower", extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("|u| (t=0)")
        plt.colorbar(im3, ax=ax)

        plt.tight_layout()
        out = ROOT / "tests" / "navier_stokes_discovery.png"
        plt.savefig(out, dpi=150)
        print(f"\n   Saved: {out}")

    print("\n" + "=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
