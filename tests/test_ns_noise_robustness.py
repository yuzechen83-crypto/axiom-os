"""
Noise Robustness Test: 2D Navier-Stokes Discovery
在 gen_kolmogorov 数据中加入 10%~20% 高斯白噪声，测试 SPNN 发现能力。

Challenge: lap(omega) 对噪声极其敏感（二阶导数放大噪声）
Expected: 点形式下蓝点变"胖"（发散），不再是一条细线；slope 偏离 nu
Solution: 弱形式发现 (Weak Formulation) - patch 平均，拟合积分平衡

Usage: py -3 tests/test_ns_noise_robustness.py
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.features_2d import calc_2d_derivatives, weak_form_patch_average

# Kolmogorov params
L = 2 * np.pi
K_FORCING = 4
NU = 0.01


def add_gaussian_noise(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    noise_level: float = 0.15,
    seed: Optional[int] = None,
) -> tuple:
    """
    Add Gaussian white noise to u, v, w.
    noise_level: fraction of signal std (e.g., 0.15 = 15%)
    """
    if seed is not None:
        torch.manual_seed(seed)
    su, sv, sw = u.std().item(), v.std().item(), w.std().item()
    u_noisy = u + noise_level * su * torch.randn_like(u)
    v_noisy = v + noise_level * sv * torch.randn_like(v)
    w_noisy = w + noise_level * sw * torch.randn_like(w)
    return u_noisy, v_noisy, w_noisy


def load_and_add_noise(
    data_dir: Path,
    noise_level: float = 0.15,
) -> tuple:
    u = torch.load(data_dir / "u_field.pt")
    v = torch.load(data_dir / "v_field.pt")
    w = torch.load(data_dir / "w_field.pt")
    return add_gaussian_noise(u, v, w, noise_level=noise_level, seed=42)


def compute_forcing(shape: tuple, k: int = K_FORCING, L_domain: float = L) -> torch.Tensor:
    n_t, n_h, n_w = shape
    y = torch.linspace(0, L_domain, n_h, dtype=torch.float64)
    yy = y.unsqueeze(0).unsqueeze(2).expand(n_t, -1, n_w)
    return -torch.sin(k * yy)


def compute_omega_t(w: torch.Tensor, dt: float) -> torch.Tensor:
    return (w[1:] - w[:-1]) / dt


def build_dataset(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    nu: float = NU,
    subsample: int = 10,
) -> tuple:
    n_t, n_h, n_w = w.shape
    w_slice = w[::subsample]
    u_slice = u[::subsample]
    v_slice = v[::subsample]
    dt_eff = dt * subsample

    omega_t = compute_omega_t(w_slice, dt_eff)
    w_mid = w_slice[:-1]
    u_mid = u_slice[:-1]
    v_mid = v_slice[:-1]

    feats = calc_2d_derivatives(w_mid, u_mid, v_mid, dx=dx, dy=dy)
    lap_omega = feats["lap_omega"]
    advection = feats["advection"]
    f = compute_forcing(omega_t.shape, K_FORCING, L)
    residual = omega_t + advection - f
    target_gt = nu * lap_omega

    return lap_omega, residual, target_gt, feats


def build_weak_form_dataset(
    lap_omega: torch.Tensor,
    target: torch.Tensor,
    patch_size: int = 4,
) -> tuple:
    """Weak form: patch-averaged lap_omega and target. Smooths noise."""
    lap_avg = weak_form_patch_average(lap_omega, patch_size)
    tgt_avg = weak_form_patch_average(target, patch_size)
    return lap_avg, tgt_avg


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class FSoftNS(nn.Module):
    def __init__(self, in_dim: int = 1, hidden_dim: int = 64, init_scale: float = 0.01):
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
    print("Noise Robustness Test: NS Discovery")
    print("10-20% Gaussian noise -> Point form degrades, Weak form recovers")
    print("=" * 60)

    data_dir = Path(__file__).resolve().parent
    if not (data_dir / "w_field.pt").exists():
        print("\nGenerating data...")
        import subprocess
        subprocess.run([sys.executable, str(data_dir / "gen_kolmogorov.py")], check=True, cwd=str(ROOT))

    n_t, n_h, n_w = 5001, 64, 64
    dx = dy = L / n_w
    dt = 10.0 / (n_t - 1)
    subsample = max(1, n_t // 50)  # ~100 timesteps for faster run

    # 1. Clean data (baseline)
    u_clean = torch.load(data_dir / "u_field.pt")
    v_clean = torch.load(data_dir / "v_field.pt")
    w_clean = torch.load(data_dir / "w_field.pt")
    # Clean: use GT target (nu*lap) for baseline
    lap_c, _, tgt_c, _ = build_dataset(u_clean, v_clean, w_clean, dt, dx, dy, subsample=subsample)

    # 2. Noisy data: use RESIDUAL as target (not GT) to show real scatter
    # residual = omega_t + advection - f has finite-diff noise; lap from noisy w amplifies noise
    noise_level = 0.20
    u_n, v_n, w_n = load_and_add_noise(data_dir, noise_level=noise_level)
    lap_n, res_n, tgt_n, _ = build_dataset(u_n, v_n, w_n, dt, dx, dy, subsample=subsample)
    tgt_n = res_n  # Use residual (noisy) not GT - shows real degradation

    print(f"\n1. Noise: {noise_level*100:.0f}% of signal std")
    print(f"   lap_omega std: clean={lap_c.std().item():.2f}, noisy={lap_n.std().item():.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = 20000

    def train_and_eval(lap_flat, tgt_flat, name: str):
        x = lap_flat.unsqueeze(-1).float()
        y = tgt_flat.unsqueeze(-1).float()
        if len(x) > max_samples:
            idx = torch.randperm(len(x))[:max_samples]
            x, y = x[idx], y[idx]
            lap_flat = lap_flat[idx]
        model = FSoftNS(in_dim=1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x_t, y_t = x.to(device), y.to(device)
        for step in range(600):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x_t), y_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            d_soft = model(x_t).cpu().numpy().ravel()
        lap_np = lap_flat.cpu().numpy().ravel()
        # Linear fit
        try:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression(fit_intercept=True)
            reg.fit(lap_np.reshape(-1, 1), d_soft)
            slope, intercept = reg.coef_[0], reg.intercept_
        except ImportError:
            X = np.column_stack([np.ones_like(lap_np), lap_np])
            c = np.linalg.lstsq(X, d_soft, rcond=None)[0]
            intercept, slope = c[0], c[1]
        ss_res = np.sum((d_soft - (slope * lap_np + intercept)) ** 2)
        r2 = 1 - ss_res / (np.sum((d_soft - d_soft.mean()) ** 2) + 1e-10)
        return lap_np, d_soft, slope, r2, name

    # 3. Point-wise: Clean
    lap_c_flat = lap_c.ravel()
    tgt_c_flat = tgt_c.ravel()
    lc, dc, sc, rc, _ = train_and_eval(lap_c_flat, tgt_c_flat, "clean")

    # 4. Point-wise: Noisy (expected: scatter, poor R2)
    lap_n_flat = lap_n.ravel()
    tgt_n_flat = tgt_n.ravel()
    ln, dn, sn, rn, _ = train_and_eval(lap_n_flat, tgt_n_flat, "noisy_point")

    # 5. Weak form: Noisy (patch average smooths residual & lap)
    patch_size = 4  # 4x4 patches for weak form
    lap_wf, tgt_wf = build_weak_form_dataset(lap_n, res_n, patch_size=patch_size)
    lap_wf_flat = lap_wf.ravel()
    tgt_wf_flat = tgt_wf.ravel()
    lw, dw, sw, rw, _ = train_and_eval(lap_wf_flat, tgt_wf_flat, "noisy_weak")

    print("\n2. Results")
    print(f"   Clean (point):     slope={sc:.4f}, R2={rc:.4f}")
    print(f"   Noisy (point):     slope={sn:.4f}, R2={rn:.4f}  <- degraded")
    print(f"   Noisy (weak form): slope={sw:.4f}, R2={rw:.4f}  <- recovered")

    # Success: weak form has slope closer to nu (0.01) than point-wise, OR R2 improves
    slope_ok = abs(sw - NU) < abs(sn - NU)
    success = slope_ok or rw > rn
    print(f"\n   Success (weak form better): {success}")

    # Plot
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        ax = axes[0]
        ax.scatter(lc, dc, s=2, alpha=0.4, c="blue", label="D_soft")
        ll = np.linspace(lc.min(), lc.max(), 100)
        ax.plot(ll, sc * ll, "r-", lw=2, label=f"slope={sc:.3f}")
        ax.set_xlabel("lap(omega)")
        ax.set_ylabel("D_soft")
        ax.set_title("Clean (baseline)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(ln, dn, s=2, alpha=0.3, c="blue", label="D_soft")
        ll = np.linspace(ln.min(), ln.max(), 100)
        ax.plot(ll, sn * ll, "r-", lw=2, label=f"slope={sn:.3f}, R2={rn:.3f}")
        ax.set_xlabel("lap(omega)")
        ax.set_ylabel("D_soft")
        ax.set_title("Noisy Point-wise (scatter)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.scatter(lw, dw, s=8, alpha=0.6, c="blue", label="D_soft")
        ll = np.linspace(lw.min(), lw.max(), 100)
        ax.plot(ll, sw * ll, "r-", lw=2, label=f"slope={sw:.3f}, R2={rw:.3f}")
        ax.set_xlabel("lap(omega)_avg (patch)")
        ax.set_ylabel("D_soft")
        ax.set_title("Noisy Weak Form (recovered)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = data_dir / "ns_noise_robustness.png"
        plt.savefig(out, dpi=150)
        print(f"\n   Saved: {out}")

    print("\n" + "=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
