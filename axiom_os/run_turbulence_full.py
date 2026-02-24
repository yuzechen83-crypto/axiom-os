"""
3D 湍流完整真实数据测试 — 高复合场，epochs=2000

优化：SPNN 物理标尺 + 突触式 λ_res 衰减 + 时间调制 Hard Core
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

from axiom_os.core.turbulence_scale import TurbulencePhysicalScale
from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.engine.discovery import DiscoveryEngine
from axiom_os.coach import coach_score, coach_loss_torch


def main():
    print("=" * 60)
    print("3D 湍流完整真实数据测试 — 高复合场")
    print("=" * 60)

    # 高复合场：5x5 空间网格，5 天预报，更密 delta
    coords, targets, meta = load_atmospheric_turbulence_3d(
        lat_center=39.9,
        lon_center=116.4,
        n_lat=5,
        n_lon=5,
        delta_deg=0.12,
        forecast_days=5,
        use_synthetic_if_fail=True,
    )

    n = len(coords)
    data_src = meta.get("data_source", "unknown")
    print(f"\nData: {n} points, Source: {data_src.upper()}")
    print(f"Coords: (t, x, y, z) normalized [0,1]")
    print(f"Targets: (u, v) wind m/s, range u=[{targets[:, 0].min():.2f}, {targets[:, 0].max():.2f}], v=[{targets[:, 1].min():.2f}, {targets[:, 1].max():.2f}]")

    split = int(0.8 * n)
    X_train = torch.from_numpy(coords[:split]).float()
    Y_train = torch.from_numpy(targets[:split]).float()
    X_test = torch.from_numpy(coords[split:]).float()
    Y_test = torch.from_numpy(targets[split:]).float()

    u_mean = float(Y_train[:, 0].mean())
    v_mean = float(Y_train[:, 1].mean())
    wind_mag = np.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2)
    thresh = max(float(np.percentile(wind_mag, 85)), 5.0)
    hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=thresh, use_enhanced=True)
    print(f"\nHard Core: adaptive+enhanced (t_mod, z_scale), threshold={thresh:.1f} m/s")

    # SPNN 物理标尺
    scale_sys = TurbulencePhysicalScale()
    scale_sys.auto_detect(coords, targets)
    print(f"PhysicalScale: v_c={scale_sys.scales.v_c:.2f} m/s")

    rcln = RCLNLayer(
        input_dim=4,
        hidden_dim=128,
        output_dim=2,
        hard_core_func=hard_core,
        lambda_res=1.0,
    )
    opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=150)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    # 分量加权：u 误差更大，提高 u 的 loss 权重以平衡 MAE
    w_u, w_v = 1.5, 1.0
    # 方案 A：Coach 辅助损失 L_total = L_data + λ_coach * (1 - coach_score)
    lambda_coach = 0.15

    epochs = 2000
    print(f"\nTraining: {epochs} epochs (u={w_u}, v={w_v}, λ_coach={lambda_coach}, λ_decay)...")
    for epoch in range(epochs):
        # 突触式 λ_res 衰减：后期 Soft 权重相对增大
        rcln.set_lambda_decay(epoch, epochs, decay_min=0.6)
        opt.zero_grad()
        pred = rcln(X_train)
        loss_u = loss_fn(pred[:, 0:1], Y_train[:, 0:1])
        loss_v = loss_fn(pred[:, 1:2], Y_train[:, 1:2])
        loss_data = w_u * loss_u + w_v * loss_v
        # Coach 辅助损失：物理约束惩罚
        l_coach = coach_loss_torch(pred, domain="fluids")
        loss = loss_data + lambda_coach * l_coach
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rcln.parameters(), 1.0)
        opt.step()
        scheduler.step(loss.item())
        if (epoch + 1) % 400 == 0:
            with torch.no_grad():
                score = coach_score(coords[:split], pred.numpy(), domain="fluids")
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f} (data={loss_data.item():.6f}, coach_pen={l_coach.item():.4f}), coach_score={score:.4f}, λ={rcln.lambda_res:.3f}, lr={opt.param_groups[0]['lr']:.2e}")

    with torch.no_grad():
        pred_test = rcln(X_test).numpy()
    mae_u = np.mean(np.abs(pred_test[:, 0] - Y_test.numpy()[:, 0]))
    mae_v = np.mean(np.abs(pred_test[:, 1] - Y_test.numpy()[:, 1]))
    mae_rcln = (mae_u + mae_v) / 2
    coach_test = coach_score(coords[split:], pred_test, domain="fluids")
    print(f"\n>>> RCLN Test MAE: u={mae_u:.4f} m/s, v={mae_v:.4f} m/s")
    print(f">>> Coach 打分 (Test): {coach_test:.4f} [0=差, 1=优]")

    # Baseline 对比：MLP、Persistence、Linear
    def _mae(pred, true):
        return np.mean(np.abs(pred - true))

    # 1. MLP baseline（无 Hard Core）
    mlp = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 2),
    )
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt_mlp.zero_grad()
        p = mlp(X_train)
        loss = torch.nn.functional.huber_loss(p, Y_train, delta=1.0)
        loss.backward()
        opt_mlp.step()
    with torch.no_grad():
        pred_mlp = mlp(X_test).numpy()
    mae_mlp_u = _mae(pred_mlp[:, 0], Y_test.numpy()[:, 0])
    mae_mlp_v = _mae(pred_mlp[:, 1], Y_test.numpy()[:, 1])
    mae_mlp = (mae_mlp_u + mae_mlp_v) / 2

    # 2. Persistence（均值预测）
    pred_pers = np.tile([u_mean, v_mean], (len(Y_test), 1))
    mae_pers_u = _mae(pred_pers[:, 0], Y_test.numpy()[:, 0])
    mae_pers_v = _mae(pred_pers[:, 1], Y_test.numpy()[:, 1])
    mae_pers = (mae_pers_u + mae_pers_v) / 2

    # 3. Linear regression (numpy, 无 sklearn 依赖)
    X_tr, X_te = coords[:split], coords[split:]
    y_u_tr, y_v_tr = targets[:split, 0], targets[:split, 1]
    ones = np.ones((len(X_tr), 1))
    X_aug = np.hstack([ones, X_tr])
    beta_u = np.linalg.lstsq(X_aug, y_u_tr, rcond=None)[0]
    beta_v = np.linalg.lstsq(X_aug, y_v_tr, rcond=None)[0]
    X_te_aug = np.hstack([np.ones((len(X_te), 1)), X_te])
    pred_lin = np.column_stack([
        X_te_aug @ beta_u,
        X_te_aug @ beta_v,
    ])
    mae_lin_u = _mae(pred_lin[:, 0], targets[split:, 0])
    mae_lin_v = _mae(pred_lin[:, 1], targets[split:, 1])
    mae_lin = (mae_lin_u + mae_lin_v) / 2

    coach_pers = coach_score(coords[split:], pred_pers, domain="fluids")
    coach_lin = coach_score(coords[split:], pred_lin, domain="fluids")
    coach_mlp = coach_score(coords[split:], pred_mlp, domain="fluids")
    print("\n>>> Baseline 对比:")
    print(f"    Persistence:  MAE avg={mae_pers:.4f} m/s  Coach={coach_pers:.4f}")
    print(f"    Linear:       MAE avg={mae_lin:.4f} m/s  Coach={coach_lin:.4f}")
    print(f"    MLP:          MAE avg={mae_mlp:.4f} m/s  Coach={coach_mlp:.4f}")
    print(f"    RCLN:         MAE avg={mae_rcln:.4f} m/s  Coach={coach_test:.4f}")
    best_mae = min(mae_pers, mae_lin, mae_mlp, mae_rcln)
    best_name = ["Persistence", "Linear", "MLP", "RCLN"][
        [mae_pers, mae_lin, mae_mlp, mae_rcln].index(best_mae)
    ]
    if mae_rcln <= best_mae:
        print(f"    -> RCLN 最优 (avg MAE={mae_rcln:.4f})")
    else:
        print(f"    -> 最优: {best_name} (avg MAE={best_mae:.4f}), RCLN={mae_rcln:.4f}")

    # Discovery
    with torch.no_grad():
        _ = rcln(X_train)
    y_soft = rcln._last_y_soft.numpy()
    engine = DiscoveryEngine(use_pysr=False)
    X_train_np = coords[:split]
    var_names = ["t", "x", "y", "z"]
    formula_u, _, _ = engine.discover_multivariate(X_train_np, y_soft[:split, 0], var_names=var_names, selector="bic")
    formula_v, _, _ = engine.discover_multivariate(X_train_np, y_soft[:split, 1], var_names=var_names, selector="bic")
    print("\n>>> Discovery (multivariate t,x,y,z):")
    print(f"  u_soft: {(formula_u or '(none)')[:100]}...")
    print(f"  v_soft: {(formula_v or '(none)')[:100]}...")

    # 3D 可视化
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    step = max(1, n // 500)
    sc = ax1.scatter(coords[::step, 1], coords[::step, 2], coords[::step, 3], c=targets[::step, 0], cmap="RdBu_r", s=5, alpha=0.7)
    ax1.set_xlabel("x (lat norm)")
    ax1.set_ylabel("y (lon norm)")
    ax1.set_zlabel("z (height norm)")
    ax1.set_title("True u wind (3D)")
    plt.colorbar(sc, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(132, projection="3d")
    with torch.no_grad():
        pred_all = rcln(torch.from_numpy(coords).float()).numpy()
    sc2 = ax2.scatter(coords[::step, 1], coords[::step, 2], coords[::step, 3], c=pred_all[::step, 0], cmap="RdBu_r", s=5, alpha=0.7)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("RCLN Pred u (3D)")
    plt.colorbar(sc2, ax=ax2, shrink=0.5)

    ax3 = fig.add_subplot(133)
    ax3.scatter(targets[split:, 0], pred_test[:, 0], alpha=0.5, s=10, label="u")
    ax3.scatter(targets[split:, 1], pred_test[:, 1], alpha=0.5, s=10, label="v")
    ax3.plot([-6, 6], [-6, 6], "k--", alpha=0.5)
    ax3.set_xlabel("True (m/s)")
    ax3.set_ylabel("Pred (m/s)")
    ax3.set_title(f"Scatter: u MAE={mae_u:.3f}, v MAE={mae_v:.3f}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = ROOT / "axiom_os" / "turbulence_3d_full_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")

    # 残差诊断：按 t, z, 风速区间检查系统性偏差
    res_u = pred_test[:, 0] - Y_test.numpy()[:, 0]
    res_v = pred_test[:, 1] - Y_test.numpy()[:, 1]
    t_test = coords[split:, 0]
    z_test = coords[split:, 3]
    wind_mag_test = np.sqrt(targets[split:, 0] ** 2 + targets[split:, 1] ** 2)

    fig2, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].scatter(t_test, res_u, alpha=0.4, s=8, c="C0", label="u")
    axes[0, 0].axhline(0, color="k", ls="--", lw=1)
    axes[0, 0].set_xlabel("t (norm)")
    axes[0, 0].set_ylabel("Residual u (m/s)")
    axes[0, 0].set_title("Residual u vs t")
    axes[0, 0].legend()

    axes[0, 1].scatter(z_test, res_u, alpha=0.4, s=8, c="C0", label="u")
    axes[0, 1].axhline(0, color="k", ls="--", lw=1)
    axes[0, 1].set_xlabel("z (height norm)")
    axes[0, 1].set_ylabel("Residual u (m/s)")
    axes[0, 1].set_title("Residual u vs z")

    axes[0, 2].scatter(wind_mag_test, res_u, alpha=0.4, s=8, c="C0", label="u")
    axes[0, 2].axhline(0, color="k", ls="--", lw=1)
    axes[0, 2].set_xlabel("|V| (m/s)")
    axes[0, 2].set_ylabel("Residual u (m/s)")
    axes[0, 2].set_title("Residual u vs wind mag")

    axes[1, 0].scatter(t_test, res_v, alpha=0.4, s=8, c="C1", label="v")
    axes[1, 0].axhline(0, color="k", ls="--", lw=1)
    axes[1, 0].set_xlabel("t (norm)")
    axes[1, 0].set_ylabel("Residual v (m/s)")
    axes[1, 0].set_title("Residual v vs t")

    axes[1, 1].scatter(z_test, res_v, alpha=0.4, s=8, c="C1", label="v")
    axes[1, 1].axhline(0, color="k", ls="--", lw=1)
    axes[1, 1].set_xlabel("z (height norm)")
    axes[1, 1].set_ylabel("Residual v (m/s)")
    axes[1, 1].set_title("Residual v vs z")

    axes[1, 2].scatter(wind_mag_test, res_v, alpha=0.4, s=8, c="C1", label="v")
    axes[1, 2].axhline(0, color="k", ls="--", lw=1)
    axes[1, 2].set_xlabel("|V| (m/s)")
    axes[1, 2].set_ylabel("Residual v (m/s)")
    axes[1, 2].set_title("Residual v vs wind mag")

    plt.suptitle(f"Residual Diagnostics: u MAE={mae_u:.3f}, v MAE={mae_v:.3f}")
    plt.tight_layout()
    diag_path = ROOT / "axiom_os" / "turbulence_3d_full_residuals.png"
    plt.savefig(diag_path, dpi=150)
    plt.close()
    print(f"Saved: {diag_path}")

    # Baseline 对比图
    fig3, ax = plt.subplots(figsize=(6, 4))
    methods = ["Persistence", "Linear", "MLP", "RCLN"]
    maes = [mae_pers, mae_lin, mae_mlp, mae_rcln]
    colors = ["#888", "#6ab", "#9a6", "#c44"]
    bars = ax.bar(methods, maes, color=colors)
    ax.set_ylabel("MAE (m/s)")
    ax.set_title("Turbulence Baseline Comparison")
    ax.axhline(min(maes), color="green", ls="--", alpha=0.5, label="Best")
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    base_path = ROOT / "axiom_os" / "turbulence_baseline_comparison.png"
    plt.savefig(base_path, dpi=150)
    plt.close()
    print(f"Saved: {base_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
