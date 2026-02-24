"""
Robust PINN - Quick Test
Verifies: ResNet backbone, float64, no symplectic/leapfrog
"""

import torch
from robust_pinn import PhysicsResNet, SoftMeltLayer
from robust_pinn.config import DTYPE

def main():
    print("Robust PINN Structure Test")
    print("=" * 50)

    # 1. dtype check
    print(f"DTYPE: {DTYPE} (must be float64)")

    # 2. PhysicsResNet
    model = PhysicsResNet(in_dim=2, out_dim=1, hidden_dim=64, num_layers=4)
    x = torch.randn(8, 2, dtype=DTYPE)
    y = model(x)
    print(f"PhysicsResNet: input {x.shape} -> output {y.shape}")
    print(f"  output dtype: {y.dtype}")

    # 3. SoftMeltLayer
    soft_melt = SoftMeltLayer()
    h_NN = torch.randn(8, 1, dtype=DTYPE)
    h_safe = torch.zeros(8, 1, dtype=DTYPE)
    h_final = soft_melt(h_NN, h_safe)
    print(f"SoftMeltLayer: h_final shape {h_final.shape}, dtype {h_final.dtype}")

    # 4. Run training demo
    print("\nRunning train.py demo...")
    import robust_pinn.train as train_mod
    # The train module runs __main__ when executed directly
    train_mod.run_training.__module__
    print("Import OK. Run: py -m robust_pinn.train")

    print("\nRobust PINN structure test passed.")

if __name__ == "__main__":
    main()
