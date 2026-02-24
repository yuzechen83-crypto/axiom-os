"""
3D Turbulence 智能分区发现 - 按高度 z 分区学习

Curriculum: turb_z_low (低空) → turb_z_mid (中层) → turb_z_high (高空)
整合: z 软门控
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
from axiom_os.core.partition import TURBULENCE_PARTITIONS, get_partitions_curriculum_order
from axiom_os.core import wrap_adaptive_hard_core
from axiom_os.core.hippocampus import Hippocampus
from axiom_os.layers.rcln import RCLNLayer


def _make_wind_hard_core(u_mean: float, v_mean: float, threshold: float = 5.0):
    """Adaptive Hard Core for wind."""
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
    base = hard_core
    return wrap_adaptive_hard_core(base, threshold=threshold, steepness=2.0)


def _train_partition_rcln(
    X_part: np.ndarray,
    Y_part: np.ndarray,
    u_mean: float,
    v_mean: float,
    epochs: int = 300,
) -> tuple:
    """在分区内训练小型 RCLN，返回 predictor 和 MAE。"""
    X_t = torch.from_numpy(X_part).float()
    Y_t = torch.from_numpy(Y_part).float()
    hard_core = _make_wind_hard_core(u_mean, v_mean, threshold=5.0)
    rcln = RCLNLayer(input_dim=4, hidden_dim=32, output_dim=2, hard_core_func=hard_core, lambda_res=1.0)
    opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        pred = rcln(X_t)
        loss = torch.nn.functional.mse_loss(pred, Y_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_np = rcln(X_t).numpy()
    mae_u = np.mean(np.abs(pred_np[:, 0] - Y_part[:, 0]))
    mae_v = np.mean(np.abs(pred_np[:, 1] - Y_part[:, 1]))
    return rcln, mae_u, mae_v


def integrate_turbulence_soft_gate(
    coords: np.ndarray,
    partition_models: dict,
    partitions: list,
) -> np.ndarray:
    """
    按 z 软门控整合：w 由 z 在 [0, 0.33], [0.33, 0.67], [0.67, 1] 的归属决定。
    """
    z = coords[:, 3] if coords.shape[1] >= 4 else coords[:, 0]
    z = np.clip(z, 1e-6, 1.0)
    n = len(z)
    pred_all = np.zeros((n, 2), dtype=np.float64)

    order = [p.id for p in partitions if p.id in partition_models]
    if not order:
        return pred_all

    preds = []
    for pid in order:
        model = partition_models[pid]
        with torch.no_grad():
            pred = model(torch.from_numpy(coords).float()).numpy()
        preds.append(pred)

    # 软权重：z < 0.33 -> low; 0.33-0.67 -> mid; > 0.67 -> high
    for i in range(n):
        zi = z[i]
        if zi < 0.33:
            t = zi / 0.33
            w_low = 1.0 - t
            w_mid = t
            w_high = 0.0
        elif zi < 0.67:
            t = (zi - 0.33) / 0.34
            w_low = 0.0
            w_mid = 1.0 - t
            w_high = t
        else:
            t = min(1.0, (zi - 0.67) / 0.33)
            w_low = 0.0
            w_mid = 1.0 - t
            w_high = t

        p0 = preds[0][i] if len(preds) > 0 else np.zeros(2)
        p1 = preds[1][i] if len(preds) > 1 else np.zeros(2)
        p2 = preds[2][i] if len(preds) > 2 else np.zeros(2)
        pred_all[i] = w_low * p0 + w_mid * p1 + w_high * p2

    return pred_all


def run_turbulence_partitioned_discovery(
    min_samples: int = 50,
    partition_epochs: int = 300,
) -> dict:
    """
    分区湍流发现：按 z 分区，每分区训练 RCLN，软门控整合。
    """
    try:
        coords, targets, meta = load_atmospheric_turbulence_3d(
            lat_center=39.9,
            lon_center=116.4,
            n_lat=3,
            n_lon=3,
            delta_deg=0.15,
            forecast_days=3,
            use_synthetic_if_fail=True,
        )
    except Exception as e:
        return {"error": str(e)}
    if coords is None or len(coords) < 50:
        return {"error": "Insufficient data"}

    n = len(coords)
    split = int(0.8 * n)
    X_train = coords[:split]
    Y_train = targets[:split]
    X_test = coords[split:]
    Y_test = targets[split:]

    u_mean = float(np.mean(Y_train[:, 0]))
    v_mean = float(np.mean(Y_train[:, 1]))

    partitions = get_partitions_curriculum_order(domain="fluids")
    partition_models = {}
    partition_results = {}

    for p in partitions:
        mask = p.mask(X_train, Y_train)
        n_part = int(mask.sum())
        if n_part < min_samples:
            print(f"  Skip {p.id}: n={n_part} < {min_samples}")
            continue
        X_part = X_train[mask]
        Y_part = Y_train[mask]
        rcln, mae_u, mae_v = _train_partition_rcln(X_part, Y_part, u_mean, v_mean, epochs=partition_epochs)
        partition_models[p.id] = rcln
        partition_results[p.id] = {"n_samples": n_part, "mae_u": mae_u, "mae_v": mae_v}
        print(f"  {p.id}: n={n_part} MAE u={mae_u:.4f} v={mae_v:.4f}")

    if not partition_models:
        return {"error": "No partition had enough samples"}

    # 注册扰动项到海马体：分区学习成分作为联想/直觉，决策参考
    hippocampus = Hippocampus()
    for pid, model in partition_models.items():
        def _make_pred_fn(m):
            def fn(x, _m=m):
                t = torch.as_tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x.float()
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                with torch.no_grad():
                    out = _m(t)
                return out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)
            return fn
        hippocampus.register_perturbation(
            _make_pred_fn(model),
            partition_id=pid,
            domain="fluids",
            output_dim=2,
            n_samples=partition_results[pid]["n_samples"],
        )
    print("  [Hippocampus] 已注册扰动项 (fluids):", list(partition_models.keys()))

    # 整合预测
    pred_test = integrate_turbulence_soft_gate(X_test, partition_models, partitions)
    mae_u = np.mean(np.abs(pred_test[:, 0] - Y_test[:, 0]))
    mae_v = np.mean(np.abs(pred_test[:, 1] - Y_test[:, 1]))

    # 基线：全局 RCLN
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    hard_core = _make_wind_hard_core(u_mean, v_mean, threshold=5.0)
    rcln_global = RCLNLayer(input_dim=4, hidden_dim=128, output_dim=2, hard_core_func=hard_core, lambda_res=1.0)
    opt = torch.optim.Adam(rcln_global.parameters(), lr=1e-3)
    for _ in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(rcln_global(X_train_t), Y_train_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_baseline = rcln_global(torch.from_numpy(X_test).float()).numpy()
    mae_u_baseline = np.mean(np.abs(pred_baseline[:, 0] - Y_test[:, 0]))
    mae_v_baseline = np.mean(np.abs(pred_baseline[:, 1] - Y_test[:, 1]))

    order = [p.id for p in partitions if p.id in partition_models]

    # 绘图
    out_path = Path(__file__).resolve().parents[1] / "turbulence_partitioned_plot.png"
    try:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].scatter(Y_test[:, 0], pred_test[:, 0], alpha=0.5, s=10, label="u (partitioned)")
        axes[0].scatter(Y_test[:, 1], pred_test[:, 1], alpha=0.5, s=10, label="v (partitioned)")
        axes[0].plot([-6, 6], [-6, 6], "k--", alpha=0.5)
        axes[0].set_xlabel("True (m/s)")
        axes[0].set_ylabel("Pred (m/s)")
        axes[0].set_title(f"Partitioned: u MAE={mae_u:.3f}, v MAE={mae_v:.3f}")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(Y_test[:, 0], pred_baseline[:, 0], alpha=0.5, s=10, label="u (global)")
        axes[1].scatter(Y_test[:, 1], pred_baseline[:, 1], alpha=0.5, s=10, label="v (global)")
        axes[1].plot([-6, 6], [-6, 6], "k--", alpha=0.5)
        axes[1].set_xlabel("True (m/s)")
        axes[1].set_ylabel("Pred (m/s)")
        axes[1].set_title(f"Baseline: u MAE={mae_u_baseline:.3f}, v MAE={mae_v_baseline:.3f}")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].bar(["u_part", "v_part", "u_glob", "v_glob"], [mae_u, mae_v, mae_u_baseline, mae_v_baseline], color=["C0", "C0", "C1", "C1"])
        axes[2].set_ylabel("MAE (m/s)")
        axes[2].set_title("MAE Comparison")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved: {out_path}")
    except Exception as e:
        print(f"Plot skip: {e}")

    return {
        "partition_results": partition_results,
        "order": order,
        "hippocampus": hippocampus,
        "partition_models": partition_models,
        "mae_u": float(mae_u),
        "mae_v": float(mae_v),
        "mae_u_baseline": float(mae_u_baseline),
        "mae_v_baseline": float(mae_v_baseline),
        "n_samples": len(X_test),
        "n_train": len(X_train),
        "u_mean": u_mean,
        "v_mean": v_mean,
    }


def main():
    print("=" * 60)
    print("3D Turbulence 智能分区发现 - 按高度 z 分区")
    print("=" * 60)

    res = run_turbulence_partitioned_discovery(min_samples=50, partition_epochs=300)

    if "error" in res:
        print(f"Error: {res['error']}")
        return

    print("\n[1] 分区学习顺序:", res["order"])
    print("\n[2] 各分区结果:")
    for pid, pr in res["partition_results"].items():
        print(f"  {pid}: n={pr['n_samples']} MAE u={pr['mae_u']:.4f} v={pr['mae_v']:.4f}")

    print(f"\n[3] 整合 MAE: u={res['mae_u']:.4f} m/s, v={res['mae_v']:.4f} m/s")
    print(f"    基线 MAE: u={res['mae_u_baseline']:.4f} m/s, v={res['mae_v_baseline']:.4f} m/s")
    print(f"    测试样本: {res['n_samples']}")
    print("=" * 60)


def run_turbulence_with_perturbation(
    hippocampus: Hippocampus,
    u_mean: float,
    v_mean: float,
    alpha_pert: float = 0.1,
    epochs: int = 500,
    use_dynamic_alpha: bool = False,
    use_learned_perturbation_gate: bool = False,
) -> dict:
    """
    使用海马体扰动项训练湍流 RCLN：y = y_hard + λ*y_soft + α*perturbation
    扰动项来自分区学习，作为联想/直觉参考。
    """
    try:
        coords, targets, meta = load_atmospheric_turbulence_3d(
            lat_center=39.9, lon_center=116.4, n_lat=3, n_lon=3,
            delta_deg=0.15, forecast_days=3, use_synthetic_if_fail=True,
        )
    except Exception as e:
        return {"error": str(e)}
    if coords is None or len(coords) < 50:
        return {"error": "Insufficient data"}

    n = len(coords)
    split = int(0.8 * n)
    X_train = torch.from_numpy(coords[:split]).float()
    Y_train = torch.from_numpy(targets[:split]).float()
    X_test = torch.from_numpy(coords[split:]).float()
    Y_test = targets[split:]

    hard_core = _make_wind_hard_core(u_mean, v_mean, threshold=5.0)
    rcln = RCLNLayer(
        input_dim=4, hidden_dim=128, output_dim=2,
        hard_core_func=hard_core, lambda_res=1.0,
        hippocampus=hippocampus,
        alpha_pert=alpha_pert,
        perturb_domain="fluids",
        use_dynamic_alpha=use_dynamic_alpha,
        use_learned_perturbation_gate=use_learned_perturbation_gate,
        alpha_pert_theta=0.12,
    )
    opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        pred = rcln(X_train)
        loss = torch.nn.functional.mse_loss(pred, Y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_test = rcln(X_test).numpy()
    mae_u = np.mean(np.abs(pred_test[:, 0] - Y_test[:, 0]))
    mae_v = np.mean(np.abs(pred_test[:, 1] - Y_test[:, 1]))

    return {
        "mae_u": float(mae_u),
        "mae_v": float(mae_v),
        "n_samples": len(Y_test),
    }


if __name__ == "__main__":
    main()
