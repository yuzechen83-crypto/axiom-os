"""
湍流训练 - SPNN Turbulence Training
Burgers 方程湍流数据 + 物理残差训练
"""

import torch
import numpy as np
from spnn import SPNN
from spnn.training.turbulence import (
    run_turbulence_training,
    TurbulenceConfig,
    generate_burgers_turbulence,
)


def main():
    print("=" * 60)
    print("SPNN 湍流训练")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1D Burgers: (t, x) -> u
    config = TurbulenceConfig(
        n_t=25,
        n_x=32,
        domain_t=(0.0, 1.0),
        domain_x=(0.0, 1.0),
        nu=0.02,
        n_modes=6,
    )

    model = SPNN(
        in_dim=2,   # (t, x)
        hidden_dim=64,
        out_dim=1,  # u
        num_rcln_layers=2,
        memory_capacity=2000,
        tau_active=0.1,
        device=device,
    )

    print("\n训练中...")
    run_turbulence_training(
        model,
        config=config,
        epochs=5000,
        batch_size=128,
        lr=1e-3,
        lambda_phys=0.0,  # SPNN 归一化断开图，先纯数据训练；物理残差可用 robust_pinn
        use_2d=False,
    )

    # 验证
    model.eval()
    coords, u_true = generate_burgers_turbulence(
        n_t=20, n_x=32, nu=config.nu, n_modes=config.n_modes, seed=42
    )
    X = torch.as_tensor(coords, device=device)
    with torch.no_grad():
        pred, _ = model(X, l_e="turbulence")
    pred_np = pred.cpu().numpy()
    mse = np.mean((pred_np - u_true) ** 2)
    print(f"\n验证 MSE: {mse:.6f}")

    # 采样预测
    print("\n采样预测 (前5点):")
    print("  真实 u:", u_true[:5].flatten())
    print("  预测 u:", pred_np[:5].flatten())

    print("\n" + "=" * 60)
    print("湍流训练完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
