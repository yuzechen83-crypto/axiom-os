"""
完整 Axiom-OS 3D 湍流真实数据测试
Complete Axiom pipeline: Hippocampus + RCLN + Discovery + 物理先验
- 真实数据: Open-Meteo API (use_synthetic_if_fail=False)
- 物理 Hard Core: 平均风 + log(z) 边界层廓线
- Discovery: u_soft, v_soft 符号回归
- Hippocampus: 知识存储
- 完整评估: MAE, RMSE, R2, 3D 可视化

MASHD 升级: USE_MASHD=True 时启用 HolographicRCLN + BreathingOptimizer (熵力注入)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.core import UPIState, Hippocampus, wrap_adaptive_hard_core, BreathingOptimizer
from axiom_os.datasets.atmospheric_turbulence import (
    load_atmospheric_turbulence_3d,
    load_atmospheric_turbulence_3d_with_physics,
)
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.layers.holographic import HolographicRCLN
from axiom_os.engine.discovery import DiscoveryEngine

# MASHD 全息 + 呼吸优化: True=启用, False=标准 RCLN
USE_MASHD = True


def make_wind_hard_core(u_mean: float, v_mean: float):
    """物理 Hard Core: 平均风 + log(z) 边界层廓线"""

    def hard_core(x):
        if isinstance(x, torch.Tensor):
            v = x.float()
        else:
            v = torch.as_tensor(x, dtype=torch.float32)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        z = v[:, 3:4].clamp(1e-6, 1.0)
        profile = 1.0 + 0.4 * torch.log1p(z)
        u_hard = u_mean * profile
        v_hard = v_mean * profile
        return torch.cat([u_hard, v_hard], dim=1)

    return hard_core


def make_wind_hard_core_adaptive(
    u_mean: float,
    v_mean: float,
    threshold: float = 5.0,
    steepness: float = 2.0,
):
    """
    自适应 Hard Core: 极端时自动放松，保持 Soft Shell 灵活性。
    gate = 1/(1+(|y_hard|/threshold)^steepness)
    当 |y_hard| 大 (极端风) 时 gate→0，Hard Core 贡献降低。
    """
    base = make_wind_hard_core(u_mean, v_mean)
    return wrap_adaptive_hard_core(base, threshold=threshold, steepness=steepness)


def main():
    print("=" * 70)
    print("完整 Axiom-OS 3D 湍流真实数据测试")
    print("Complete Axiom: Hippocampus + RCLN + Discovery on Real Data")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. 加载真实数据 (强制真实, 不降级到合成)
    # -------------------------------------------------------------------------
    use_physics = True
    try:
        coords, targets, physics, meta = load_atmospheric_turbulence_3d_with_physics(
            lat_center=39.9,
            lon_center=116.4,
            n_lat=3,
            n_lon=3,
            delta_deg=0.15,
            forecast_days=3,
            use_synthetic_if_fail=False,
        )
        print("\n[1] 数据: load_atmospheric_turbulence_3d_with_physics (含物理场)")
    except RuntimeError:
        use_physics = False
        coords, targets, meta = load_atmospheric_turbulence_3d(
            lat_center=39.9,
            lon_center=116.4,
            n_lat=3,
            n_lon=3,
            delta_deg=0.15,
            forecast_days=3,
            use_synthetic_if_fail=False,
        )
        physics = None
        print("\n[1] 数据: load_atmospheric_turbulence_3d (基础风场)")

    n = len(coords)
    data_src = meta.get("data_source", "unknown")
    print(f"    样本数: {n}, 来源: {data_src.upper()}")
    print(f"    坐标: (t, x, y, z) 归一化 [0,1]")
    print(f"    目标: u=[{targets[:, 0].min():.2f}, {targets[:, 0].max():.2f}], v=[{targets[:, 1].min():.2f}, {targets[:, 1].max():.2f}] m/s")

    # -------------------------------------------------------------------------
    # 2. Hippocampus + RCLN 初始化
    # -------------------------------------------------------------------------
    hippocampus = Hippocampus(dim=32, capacity=5000)
    split = int(0.8 * n)
    X_train = torch.from_numpy(coords[:split]).float()
    Y_train = torch.from_numpy(targets[:split]).float()
    X_test = torch.from_numpy(coords[split:]).float()
    Y_test = torch.from_numpy(targets[split:]).float()

    u_mean = float(Y_train[:, 0].mean())
    v_mean = float(Y_train[:, 1].mean())
    # 自适应 Hard Core: 极端风时 gate→0，保持 Soft Shell 灵活性
    wind_mag = np.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2)
    wind_threshold = float(np.percentile(wind_mag, 85))
    wind_threshold = max(wind_threshold, 5.0)
    hard_core = make_wind_hard_core_adaptive(
        u_mean, v_mean, threshold=wind_threshold, steepness=2.0
    )

    if USE_MASHD:
        rcln = HolographicRCLN(
            input_dim=4,
            hidden_dim=128,
            output_dim=2,
            hard_core_func=hard_core,
            lambda_res=1.0,
            n_z_slices=7,
            L=1.0,
        )
        opt = BreathingOptimizer(
            torch.optim.Adam(rcln.parameters()),
            base_lr=1e-3,
            noise_scale=1e-4,
            use_entropy=True,
        )
        print("\n[2] Hippocampus + HolographicRCLN (MASHD) 初始化")
    else:
        rcln = RCLNLayer(
            input_dim=4,
            hidden_dim=128,
            output_dim=2,
            hard_core_func=hard_core,
            lambda_res=1.0,
        )
        opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
        print("\n[2] Hippocampus + RCLN 初始化")
    scheduler_plateau = None
    if not USE_MASHD:
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=80)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    print(f"    Hard Core: 自适应 (极端时放松), u_mean={u_mean:.3f}, v_mean={v_mean:.3f}")
    print(f"    极端阈值: |y_hard|>{wind_threshold:.1f} m/s 时 gate 下降")

    # -------------------------------------------------------------------------
    # 3. 训练 RCLN
    # -------------------------------------------------------------------------
    print("\n[3] 训练 RCLN (1000 epochs)" + (" [MASHD 熵力]" if USE_MASHD else ""))
    for epoch in range(1000):
        opt.zero_grad()
        pred = rcln(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rcln.parameters(), 1.0)
        if USE_MASHD:
            opt.step(loss=loss, training_progress=(epoch + 1) / 1000)
        else:
            opt.step()
            scheduler_plateau.step(loss.item())
        lr = opt.optimizer.param_groups[0]["lr"] if USE_MASHD else opt.param_groups[0]["lr"]
        if (epoch + 1) % 200 == 0:
            z_info = f", Z={opt.get_Z():.3f}" if USE_MASHD else ""
            print(f"    Epoch {epoch+1}: loss={loss.item():.6f}, lr={lr:.2e}{z_info}")

    # -------------------------------------------------------------------------
    # 4. 完整评估指标
    # -------------------------------------------------------------------------
    with torch.no_grad():
        pred_train = rcln(X_train).numpy()
        pred_test = rcln(X_test).numpy()

    Y_train_np = Y_train.numpy()
    Y_test_np = Y_test.numpy()

    def metrics(y_true, y_pred, name=""):
        mae_u = np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))
        mae_v = np.mean(np.abs(y_pred[:, 1] - y_true[:, 1]))
        rmse_u = np.sqrt(np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2))
        rmse_v = np.sqrt(np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2))
        ss_res_u = np.sum((y_true[:, 0] - y_pred[:, 0]) ** 2)
        ss_tot_u = np.sum((y_true[:, 0] - y_true[:, 0].mean()) ** 2) + 1e-12
        ss_res_v = np.sum((y_true[:, 1] - y_pred[:, 1]) ** 2)
        ss_tot_v = np.sum((y_true[:, 1] - y_true[:, 1].mean()) ** 2) + 1e-12
        r2_u = 1 - ss_res_u / ss_tot_u
        r2_v = 1 - ss_res_v / ss_tot_v
        return {
            "mae_u": mae_u, "mae_v": mae_v,
            "rmse_u": rmse_u, "rmse_v": rmse_v,
            "r2_u": r2_u, "r2_v": r2_v,
        }

    train_metrics = metrics(Y_train_np, pred_train, "Train")
    test_metrics = metrics(Y_test_np, pred_test, "Test")

    print("\n[4] 完整评估指标")
    print("    Train: MAE u={:.4f} v={:.4f} | RMSE u={:.4f} v={:.4f} | R2 u={:.4f} v={:.4f}".format(
        train_metrics["mae_u"], train_metrics["mae_v"],
        train_metrics["rmse_u"], train_metrics["rmse_v"],
        train_metrics["r2_u"], train_metrics["r2_v"],
    ))
    print("    Test:  MAE u={:.4f} v={:.4f} | RMSE u={:.4f} v={:.4f} | R2 u={:.4f} v={:.4f}".format(
        test_metrics["mae_u"], test_metrics["mae_v"],
        test_metrics["rmse_u"], test_metrics["rmse_v"],
        test_metrics["r2_u"], test_metrics["r2_v"],
    ))

    # -------------------------------------------------------------------------
    # 5. Discovery Engine
    # -------------------------------------------------------------------------
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

    print("\n[5] Discovery (u_soft, v_soft 符号回归)")
    if formula_u:
        s = formula_u[:120] + ("..." if len(formula_u) > 120 else "")
        print(f"    u_soft: {s}")
    else:
        print("    u_soft: (none)")
    if formula_v:
        s = formula_v[:120] + ("..." if len(formula_v) > 120 else "")
        print(f"    v_soft: {s}")
    else:
        print("    v_soft: (none)")

    # Discovery R2
    if formula_u and pred_u is not None:
        ss_res = np.sum((y_soft[:split, 0] - pred_u) ** 2)
        ss_tot = np.sum((y_soft[:split, 0] - y_soft[:split, 0].mean()) ** 2) + 1e-12
        print(f"    Discovery R2(u_soft) = {1 - ss_res/ss_tot:.4f}")
    if formula_v and pred_v is not None:
        ss_res = np.sum((y_soft[:split, 1] - pred_v) ** 2)
        ss_tot = np.sum((y_soft[:split, 1] - y_soft[:split, 1].mean()) ** 2) + 1e-12
        print(f"    Discovery R2(v_soft) = {1 - ss_res/ss_tot:.4f}")

    # -------------------------------------------------------------------------
    # 6. Hippocampus 知识存储 (记录发现的公式)
    # -------------------------------------------------------------------------
    knowledge = {}
    if formula_u:
        knowledge["u_soft"] = formula_u
    if formula_v:
        knowledge["v_soft"] = formula_v
    for kid, form in knowledge.items():
        hippocampus.knowledge_base[f"turbulence_{kid}"] = {
            "formula": form,
            "callable": None,
            "target": kid,
        }

    print("\n[6] Hippocampus 知识库")
    print(f"    已存储: {list(hippocampus.knowledge_base.keys())}")

    # -------------------------------------------------------------------------
    # 7. 物理残差 Discovery (若含 physics 特征)
    # -------------------------------------------------------------------------
    if use_physics and physics is not None:
        print("\n[7] 物理残差 Discovery (gradP, gradT, isallobaric)")
        R_u = (Y_train_np[:, 0] - pred_train[:, 0]).astype(np.float64)
        R_v = (Y_train_np[:, 1] - pred_train[:, 1]).astype(np.float64)
        physics_train = physics[:split]
        X_phys = np.hstack([
            physics_train,
            (physics_train[:, 4:5] * physics_train[:, 0:1]),
            (physics_train[:, 4:5] * physics_train[:, 1:2]),
        ])
        var_names_phys = ["gradP_x", "gradP_y", "gradT_x", "gradT_y", "inv_f", "gradP_dt", "isallobaric",
                          "inv_f*gradP_x", "inv_f*gradP_y"]
        formula_Ru, _, _ = engine.discover_multivariate(
            X_phys, R_u, var_names=var_names_phys, selector="bic", standardize=True,
        )
        formula_Rv, _, _ = engine.discover_multivariate(
            X_phys, R_v, var_names=var_names_phys, selector="bic", standardize=True,
        )
        if formula_Ru:
            print(f"    R_u = {formula_Ru[:80]}...")
        if formula_Rv:
            print(f"    R_v = {formula_Rv[:80]}...")

    # -------------------------------------------------------------------------
    # 8. 3D 可视化
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    step = max(1, n // 500)
    sc = ax1.scatter(
        coords[::step, 1], coords[::step, 2], coords[::step, 3],
        c=targets[::step, 0], cmap="RdBu_r", s=5, alpha=0.7,
    )
    ax1.set_xlabel("x (lat)")
    ax1.set_ylabel("y (lon)")
    ax1.set_zlabel("z (height)")
    ax1.set_title("True u wind (3D)")
    plt.colorbar(sc, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(132, projection="3d")
    with torch.no_grad():
        pred_all = rcln(torch.from_numpy(coords).float()).numpy()
    sc2 = ax2.scatter(
        coords[::step, 1], coords[::step, 2], coords[::step, 3],
        c=pred_all[::step, 0], cmap="RdBu_r", s=5, alpha=0.7,
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("RCLN Pred u (3D)")
    plt.colorbar(sc2, ax=ax2, shrink=0.5)

    ax3 = fig.add_subplot(133)
    ax3.scatter(targets[split:, 0], pred_test[:, 0], alpha=0.5, s=10, label="u")
    ax3.scatter(targets[split:, 1], pred_test[:, 1], alpha=0.5, s=10, label="v")
    ax3.plot([-15, 15], [-15, 15], "k--", alpha=0.5)
    ax3.set_xlabel("True (m/s)")
    ax3.set_ylabel("Pred (m/s)")
    ax3.set_title("Scatter: True vs Pred")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = ROOT / "axiom_os" / "turbulence_3d_full_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\n[8] 可视化已保存: {out_path}")
    print("=" * 70)
    print("完整 Axiom 3D 湍流测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
