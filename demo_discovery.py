"""
Demo: 发现摩擦力实验
验证 RCLN 和发现引擎能否从数据中"悟"出公式 F = -0.5 * v
"""

import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend (avoids Tcl/Tk on headless)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.core.upi import UPIState, Units
from axiom_os.layers.rcln import RCLNLayer
from axiom_os.engine.discovery import DiscoveryEngine


def main():
    print("=" * 60)
    print("Demo: 发现摩擦力 (F = -0.5 * v)")
    print("=" * 60)

    # 1. 造数据 (真实物理: F = -0.5 * v)
    v = torch.linspace(-5, 5, 100).reshape(-1, 1).double()
    F_true = -0.5 * v
    # 包装成 UPI (velocity: L/T = [0,1,-1,0,0])
    inputs = UPIState(v, units=Units.VELOCITY, semantics="velocity")

    # 2. 初始化 RCLN (硬核不懂摩擦，认为 F=0)
    def hard_core_zero(x):
        if isinstance(x, torch.Tensor):
            vals = x
        elif hasattr(x, "values") and not isinstance(x, torch.Tensor):
            vals = x.values
        else:
            vals = x
        t = torch.as_tensor(vals, dtype=torch.float32)
        return torch.zeros_like(t)

    rcln = RCLNLayer(
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        hard_core_func=hard_core_zero,
        lambda_res=1.0,
    )
    optimizer = torch.optim.Adam(rcln.parameters(), lr=0.01)

    # 3. 训练循环 (RCLN Soft Shell 学习)
    print("\n>>> Training RCLN Soft Shell...")
    for i in range(1000):
        optimizer.zero_grad()
        F_pred = rcln(inputs)
        loss = torch.mean((F_pred.float() - F_true.float()) ** 2)
        loss.backward()
        optimizer.step()
        if (i + 1) % 200 == 0:
            print(f"    Epoch {i+1}, loss={loss.item():.6f}")

    # 4. 发现循环 (提取公式)
    print("\n>>> Running Discovery Engine...")
    engine = DiscoveryEngine(use_pysr=False)
    x_np = inputs.values.detach().cpu().numpy()
    # Get y_soft from RCLN (soft shell learned F ≈ -0.5*v)
    with torch.no_grad():
        _ = rcln(inputs)
    y_soft_np = rcln._last_y_soft.cpu().numpy()
    # Sanity: y_soft should correlate negatively with v (F = -0.5*v)
    corr = np.corrcoef(x_np.ravel(), y_soft_np.ravel())[0, 1]
    print(f"    Correlation v vs y_soft: {corr:.4f} (expected -1 for F=-0.5*v)")
    # Pass (x, y_soft) pairs - use discover for direct x->y fit
    formula = engine.discover(x_np, y_soft_np)

    print(f"\n[OK] Discovery Result: {formula}")
    print("\n预期: 系数接近 -0.5 (F = -0.5 * v)")
    if formula and "-0.4" in formula or "-0.5" in formula or "-0.6" in formula:
        print("[OK] Success: coefficient close to -0.5")
    else:
        print("[Note] Coefficient may vary slightly (Lasso linear fit)")

    # 5. 可视化
    with torch.no_grad():
        F_pred = rcln(inputs).numpy()
    v_np = v.numpy()
    F_true_np = F_true.numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(v_np, F_true_np, "b-", label="True: F = -0.5*v", linewidth=2)
    plt.plot(v_np, F_pred, "r--", label="RCLN Pred", linewidth=1.5)
    plt.xlabel("Velocity v")
    plt.ylabel("Force F")
    plt.title("Discovery Demo: Friction F = -0.5 * v")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = Path(__file__).resolve().parent / "demo_discovery.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")
    plt.close()

    print("=" * 60)


if __name__ == "__main__":
    main()
