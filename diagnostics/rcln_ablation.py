"""
RCLN 消融实验：lambda_res (Hard/Soft 权重) 对 MAE 的影响。
运行: python -m axiom_os.diagnostics.rcln_ablation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from axiom_os.benchmarks.seed_utils import set_global_seed


def run_rcln_turbulence(lambda_res: float, epochs: int = 100, seed: int = 42) -> float:
    """湍流 RCLN 训练，返回 test MAE"""
    from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
    from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
    from axiom_os.layers.rcln import RCLNLayer

    set_global_seed(seed)
    coords, targets, _ = load_atmospheric_turbulence_3d(
        n_lat=3, n_lon=3, delta_deg=0.15, forecast_days=3, use_synthetic_if_fail=True
    )
    n = len(coords)
    split = int(0.8 * n)
    X_train = torch.from_numpy(coords[:split]).float()
    Y_train = torch.from_numpy(targets[:split]).float()
    X_test = torch.from_numpy(coords[split:]).float()
    Y_test = torch.from_numpy(targets[split:]).float()

    u_mean = float(Y_train[:, 0].mean())
    v_mean = float(Y_train[:, 1].mean())
    hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=5.0, use_enhanced=False)
    rcln = RCLNLayer(
        input_dim=4, hidden_dim=64, output_dim=2,
        hard_core_func=hard_core, lambda_res=lambda_res
    )
    opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        pred = rcln(X_train)
        loss = torch.nn.functional.mse_loss(pred, Y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_test = rcln(X_test)
        mae = float(torch.mean(torch.abs(pred_test - Y_test)).item())
    return mae


def main(epochs: int = 100, seed: int = 42):
    set_global_seed(seed)
    print("=" * 60)
    print("RCLN lambda_res 消融 (湍流)")
    print("=" * 60)

    configs = [(0.0, "纯 Soft"), (0.5, "λ=0.5"), (1.0, "λ=1.0 (默认)"), (2.0, "λ=2.0")]
    print("-" * 50)
    for lam, label in configs:
        mae = run_rcln_turbulence(lambda_res=lam, epochs=epochs, seed=seed)
        print(f"  {label:15s}: MAE = {mae:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(epochs=args.epochs, seed=args.seed)
