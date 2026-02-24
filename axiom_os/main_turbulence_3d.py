"""
3D Atmospheric Turbulence Test - Real Data (Optimized)
- Physical Hard Core: mean wind + log-height profile (boundary layer prior)
- Huber loss: robust to extreme values
- Training: 1000 epochs, LR scheduler
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.engine.discovery import DiscoveryEngine


def main():
    print("=" * 60)
    print("3D Atmospheric Turbulence - Real Data Test")
    print("=" * 60)

    coords, targets, meta = load_atmospheric_turbulence_3d(
        lat_center=39.9,
        lon_center=116.4,
        n_lat=3,
        n_lon=3,
        delta_deg=0.15,
        forecast_days=3,
        use_synthetic_if_fail=False,
    )

    n = len(coords)
    data_src = meta.get("data_source", "unknown")
    print(f"\nData: {n} points, Source: {data_src.upper()}")
    print(f"Coords: (t, x, y, z) normalized [0,1]")
    print(f"Targets: (u, v) wind m/s, range u=[{targets[:, 0].min():.2f}, {targets[:, 0].max():.2f}], v=[{targets[:, 1].min():.2f}, {targets[:, 1].max():.2f}]")

    # Train/Test split
    split = int(0.8 * n)
    X_train = torch.from_numpy(coords[:split]).float()
    Y_train = torch.from_numpy(targets[:split]).float()
    X_test = torch.from_numpy(coords[split:]).float()
    Y_test = torch.from_numpy(targets[split:]).float()

    # Adaptive Hard Core: looser in extreme wind
    u_mean = float(Y_train[:, 0].mean())
    v_mean = float(Y_train[:, 1].mean())
    wind_mag = np.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2)
    thresh = max(float(np.percentile(wind_mag, 85)), 5.0)
    hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=thresh, use_enhanced=False)
    print(f"\nHard Core: adaptive (threshold={thresh:.1f} m/s), u_mean={u_mean:.3f}, v_mean={v_mean:.3f}")

    # RCLN with larger hidden, Huber loss, LR scheduler
    rcln = RCLNLayer(
        input_dim=4,
        hidden_dim=128,
        output_dim=2,
        hard_core_func=hard_core,
        lambda_res=1.0,
    )
    opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=80)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    for epoch in range(1000):
        opt.zero_grad()
        pred = rcln(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rcln.parameters(), 1.0)
        opt.step()
        scheduler.step(loss.item())
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}, lr={opt.param_groups[0]['lr']:.2e}")

    with torch.no_grad():
        pred_test = rcln(X_test).numpy()
    mae_u = np.mean(np.abs(pred_test[:, 0] - Y_test.numpy()[:, 0]))
    mae_v = np.mean(np.abs(pred_test[:, 1] - Y_test.numpy()[:, 1]))
    print(f"\n>>> RCLN Test MAE: u={mae_u:.4f} m/s, v={mae_v:.4f} m/s")

    # Discovery: multivariate (t,x,y,z) -> u and v
    with torch.no_grad():
        _ = rcln(X_train)
    y_soft = rcln._last_y_soft.numpy()
    engine = DiscoveryEngine(use_pysr=False)
    X_train_np = coords[:split]
    var_names = ["t", "x", "y", "z"]

    formula_u, pred_u, coefs_u = engine.discover_multivariate(
        X_train_np, y_soft[:split, 0], var_names=var_names, selector="bic",
    )
    formula_v, pred_v, coefs_v = engine.discover_multivariate(
        X_train_np, y_soft[:split, 1], var_names=var_names, selector="bic",
    )

    print("\n>>> Discovery (multivariate t,x,y,z):")
    if formula_u:
        s = formula_u[:100] + ("..." if len(formula_u) > 100 else "")
        print(f"  u_soft: {s}")
    else:
        fb = engine.discover(X_train_np, y_soft[:split, 0])
        print(f"  u_soft: {fb[:90] + '...' if fb and len(fb) > 90 else (fb or '(none)')}")
    if formula_v:
        s = formula_v[:100] + ("..." if len(formula_v) > 100 else "")
        print(f"  v_soft: {s}")
    else:
        fb = engine.discover(X_train_np, y_soft[:split, 1])
        print(f"  v_soft: {fb[:90] + '...' if fb and len(fb) > 90 else (fb or '(none)')}")

    # 3D visualization
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    sc = ax1.scatter(
        coords[::max(1, n // 500), 1],
        coords[::max(1, n // 500), 2],
        coords[::max(1, n // 500), 3],
        c=targets[::max(1, n // 500), 0],
        cmap="RdBu_r",
        s=5,
        alpha=0.7,
    )
    ax1.set_xlabel("x (lat norm)")
    ax1.set_ylabel("y (lon norm)")
    ax1.set_zlabel("z (height norm)")
    ax1.set_title("True u wind (3D)")
    plt.colorbar(sc, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(132, projection="3d")
    with torch.no_grad():
        pred_all = rcln(torch.from_numpy(coords).float()).numpy()
    sc2 = ax2.scatter(
        coords[::max(1, n // 500), 1],
        coords[::max(1, n // 500), 2],
        coords[::max(1, n // 500), 3],
        c=pred_all[::max(1, n // 500), 0],
        cmap="RdBu_r",
        s=5,
        alpha=0.7,
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("RCLN Pred u (3D)")
    plt.colorbar(sc2, ax=ax2, shrink=0.5)

    ax3 = fig.add_subplot(133)
    ax3.scatter(targets[split:, 0], pred_test[:, 0], alpha=0.5, s=10, label="u")
    ax3.scatter(targets[split:, 1], pred_test[:, 1], alpha=0.5, s=10, label="v")
    ax3.plot([-5, 5], [-5, 5], "k--", alpha=0.5)
    ax3.set_xlabel("True (m/s)")
    ax3.set_ylabel("Pred (m/s)")
    ax3.set_title("Scatter: True vs Pred")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "turbulence_3d_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
