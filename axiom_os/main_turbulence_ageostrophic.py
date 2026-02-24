"""
Ageostrophic Wind Discovery (Refined)
1. Residual: R = V_true - V_pred
2. Feature standardization: Z-scores (mean=0, std=1) for comparable coefficients
3. Physics: gradP, gradT, inv_f, gradP_dt, isallobaric = -(1/f^2)*gradP_dt
4. Discovery: R_u, R_v, and vector magnitude |R|
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

from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d_with_physics
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.engine.discovery import DiscoveryEngine


def main():
    print("=" * 70)
    print("Ageostrophic Wind Discovery - Missing Meteorological Factors")
    print("=" * 70)

    coords, targets, physics, meta = load_atmospheric_turbulence_3d_with_physics(
        lat_center=39.9,
        lon_center=116.4,
        n_lat=3,
        n_lon=3,
        delta_deg=0.15,
        forecast_days=3,
        use_synthetic_if_fail=True,
    )

    n = len(coords)
    data_src = meta.get("data_source", "unknown")
    print(f"\nData: {n} points, Source: {data_src.upper()}")
    print(f"Physics: gradP_x/y, gradT_x/y, inv_f, gradP_dt, isallobaric")
    print(f"  gradP: [{physics[:, 0].min():.2f}, {physics[:, 0].max():.2f}] hPa/deg")
    print(f"  gradP_dt: [{physics[:, 5].min():.2f}, {physics[:, 5].max():.2f}] hPa/h")
    print(f"  isallobaric: [{physics[:, 6].min():.2e}, {physics[:, 6].max():.2e}]")

    split = int(0.8 * n)
    X_train = torch.from_numpy(coords[:split]).float()
    Y_train = torch.from_numpy(targets[:split]).float()
    X_test = torch.from_numpy(coords[split:]).float()
    Y_test = torch.from_numpy(targets[split:]).float()
    physics_train = physics[:split]
    physics_test = physics[split:]

    u_mean = float(Y_train[:, 0].mean())
    v_mean = float(Y_train[:, 1].mean())
    wind_mag = np.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2)
    thresh = max(float(np.percentile(wind_mag, 85)), 5.0)
    hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=thresh, use_enhanced=False)

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
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

    with torch.no_grad():
        pred_train = rcln(X_train).numpy()
        pred_test = rcln(X_test).numpy()

    mae_u = np.mean(np.abs(pred_test[:, 0] - Y_test.numpy()[:, 0]))
    mae_v = np.mean(np.abs(pred_test[:, 1] - Y_test.numpy()[:, 1]))
    print(f"\n>>> RCLN Test MAE: u={mae_u:.4f} m/s, v={mae_v:.4f} m/s")

    # Residual: R_v = V_true - V_pred (what soft shell tries to correct)
    R_u = (Y_train.numpy()[:, 0] - pred_train[:, 0]).astype(np.float64)
    R_v = (Y_train.numpy()[:, 1] - pred_train[:, 1]).astype(np.float64)
    print(f"\n>>> Residual R = V_true - V_pred:")
    print(f"  R_u: mean={R_u.mean():.4f}, std={R_u.std():.4f} m/s")
    print(f"  R_v: mean={R_v.mean():.4f}, std={R_v.std():.4f} m/s")

    # Physics features: base + derived (inv_f*gradP, isallobaric)
    inv_f = physics_train[:, 4:5]
    gradP_x = physics_train[:, 0:1]
    gradP_y = physics_train[:, 1:2]
    gradT_x = physics_train[:, 2:3]
    gradT_y = physics_train[:, 3:4]
    gradP_dt = physics_train[:, 5:6]
    isallobaric = physics_train[:, 6:7]
    X_phys = np.hstack([
        physics_train,
        (inv_f * gradP_x),
        (inv_f * gradP_y),
        (inv_f * gradT_x),
        (inv_f * gradT_y),
    ])
    var_names = ["gradP_x", "gradP_y", "gradT_x", "gradT_y", "inv_f", "gradP_dt", "isallobaric",
                 "inv_f*gradP_x", "inv_f*gradP_y", "inv_f*gradT_x", "inv_f*gradT_y"]

    engine = DiscoveryEngine(use_pysr=False)

    print("\n>>> Discovery (Z-score standardized, comparable coefficients):")
    print("  R_u vs physics:")
    formula_Ru, pred_Ru, _ = engine.discover_multivariate(
        X_phys, R_u, var_names=var_names, selector="bic", standardize=True,
    )
    if formula_Ru:
        print(f"  R_u = {formula_Ru[:120]}{'...' if len(formula_Ru) > 120 else ''}")
        if "inv_f*gradP" in formula_Ru or "inv_f" in formula_Ru and "gradP" in formula_Ru:
            print("  [OK] Ageostrophic-like term (inv_f * gradP) detected!")

    print("\n  R_v vs physics:")
    formula_Rv, pred_Rv, _ = engine.discover_multivariate(
        X_phys, R_v, var_names=var_names, selector="bic", standardize=True,
    )
    if formula_Rv:
        print(f"  R_v = {formula_Rv[:120]}{'...' if len(formula_Rv) > 120 else ''}")
        if "inv_f*gradP" in formula_Rv or ("inv_f" in formula_Rv and "gradP" in formula_Rv):
            print("  [OK] Ageostrophic-like term (inv_f * gradP) detected!")
        if "isallobaric" in formula_Rv:
            print("  [OK] Isallobaric term (pressure tendency) detected!")

    # Vector magnitude: |R| = sqrt(R_u^2 + R_v^2)
    R_mag = np.sqrt(R_u ** 2 + R_v ** 2).astype(np.float64)
    print("\n  |R| (vector magnitude) vs physics:")
    formula_Rmag, pred_Rmag, _ = engine.discover_multivariate(
        X_phys, R_mag, var_names=var_names, selector="bic", standardize=True,
    )
    if formula_Rmag:
        print(f"  |R| = {formula_Rmag[:100]}{'...' if len(formula_Rmag) > 100 else ''}")

    # R2 of discovered formula on residual
    if formula_Ru and pred_Ru is not None:
        ss_res = np.sum((R_u - pred_Ru) ** 2)
        ss_tot = np.sum((R_u - R_u.mean()) ** 2) + 1e-12
        r2_u = 1 - ss_res / ss_tot
        print(f"\n  R2(R_u) = {r2_u:.4f}")
    if formula_Rv and pred_Rv is not None:
        ss_res = np.sum((R_v - pred_Rv) ** 2)
        ss_tot = np.sum((R_v - R_v.mean()) ** 2) + 1e-12
        r2_v = 1 - ss_res / ss_tot
        print(f"  R2(R_v) = {r2_v:.4f}")
    if formula_Rmag and pred_Rmag is not None:
        ss_res = np.sum((R_mag - pred_Rmag) ** 2)
        ss_tot = np.sum((R_mag - R_mag.mean()) ** 2) + 1e-12
        r2_mag = 1 - ss_res / ss_tot
        print(f"  R2(|R|) = {r2_mag:.4f}")

    # Plot: residual vs physics-based prediction
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.scatter(R_u, pred_Ru if pred_Ru is not None else np.zeros_like(R_u), alpha=0.5, s=10)
    rng = max(abs(R_u.min()), abs(R_u.max()), 0.1)
    ax1.plot([-rng, rng], [-rng, rng], "k--", alpha=0.5)
    ax1.set_xlabel("R_u (true)")
    ax1.set_ylabel("R_u (discovered)")
    ax1.set_title("R_u: True vs Pred (Z-score fit)")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(R_v, pred_Rv if pred_Rv is not None else np.zeros_like(R_v), alpha=0.5, s=10)
    rng = max(abs(R_v.min()), abs(R_v.max()), 0.1)
    ax2.plot([-rng, rng], [-rng, rng], "k--", alpha=0.5)
    ax2.set_xlabel("R_v (true)")
    ax2.set_ylabel("R_v (discovered)")
    ax2.set_title("R_v: True vs Pred (Z-score fit)")
    ax2.grid(True, alpha=0.3)

    ax3.scatter(R_mag, pred_Rmag if pred_Rmag is not None else np.zeros_like(R_mag), alpha=0.5, s=10)
    rng = max(R_mag.max(), 0.1)
    ax3.plot([0, rng], [0, rng], "k--", alpha=0.5)
    ax3.set_xlabel("|R| (true)")
    ax3.set_ylabel("|R| (discovered)")
    ax3.set_title("|R|: True vs Pred")
    ax3.grid(True, alpha=0.3)

    ax4.scatter(physics_train[:, 6], R_mag, alpha=0.3, s=5)  # isallobaric
    ax4.set_xlabel("isallobaric = -(1/f^2)*gradP_dt")
    ax4.set_ylabel("|R| (m/s)")
    ax4.set_title("|R| vs Isallobaric term")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "turbulence_ageostrophic_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
