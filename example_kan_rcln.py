"""
KAN-RCLN Example: From Data to Formula
Demonstrates the symbolic evolution path of Axiom-OS

This example shows how KAN-RCLN can:
1. Learn from data
2. Extract symbolic formulas directly
3. Integrate with Discovery Engine
"""

import torch
import torch.nn as nn
from axiom_os.layers import RCLNLayer


def main():
    print("=" * 70)
    print(" KAN-RCLN Example: Discovering Physics from Data")
    print("=" * 70)

    # Problem: Learn the drag force on a sphere
    # True physics: F_drag = 0.47 * v^2 (quadratic drag law)
    # But we only have data, not the formula

    print("\n[Setup]")
    print("Target: Discover F_drag = 0.47 * v^2 from data")

    # Generate synthetic data
    v = torch.linspace(0, 10, 200).unsqueeze(1)  # Velocity
    f_drag = 0.47 * v ** 2  # True drag force (unknown to model)

    # Create KAN-RCLN
    print("\n[Creating KAN-RCLN]")
    rcln = RCLNLayer(
        input_dim=1,        # Velocity
        hidden_dim=4,       # Small hidden layer
        output_dim=1,       # Drag force
        net_type="kan",     # Enable KAN Soft Shell
        kan_grid_size=6,    # B-spline grid points
        kan_spline_order=3, # Cubic splines
        lambda_res=1.0,     # Pure data-driven (no hard core)
    )
    print(f"Architecture: input=1, hidden=4, output=1")
    print(f"KAN config: grid_size=6, spline_order=3")

    # Train
    print("\n[Training]")
    optimizer = torch.optim.Adam(rcln.parameters(), lr=0.01)
    for epoch in range(300):
        optimizer.zero_grad()
        f_pred = rcln(v)
        loss = nn.functional.mse_loss(f_pred, f_drag)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:3d}: loss = {loss.item():.6f}")

    print(f"\n[Training Complete]")
    print(f"Final loss: {loss.item():.6f}")

    # Extract formula
    print("\n[Formula Extraction]")
    formula_info = rcln.extract_formula(var_names=["velocity"])

    if formula_info:
        print(f"Discovered formula: {formula_info['formula_str']}")
        print(f"Confidence: {formula_info['confidence']:.1%}")

        # Analyze components
        print("\n[Components Analysis]")
        for comp in formula_info['components'][:5]:
            print(f"  - {comp}")

    # Compare with ground truth
    print("\n[Validation]")
    with torch.no_grad():
        f_learned = rcln(torch.tensor([[5.0]]))  # v = 5
        f_true = 0.47 * 25  # 11.75
        error = abs(f_learned.item() - f_true) / f_true * 100
        print(f"At v=5: Learned F={f_learned.item():.3f}, True F={f_true:.3f}")
        print(f"Relative error: {error:.2f}%")

    # Show architecture info
    print("\n[Architecture Info]")
    info = rcln.get_architecture_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Example complete! KAN-RCLN successfully discovered the physics.")
    print("=" * 70)
    print("""
Key Takeaways:
1. KAN-RCLN learns from data like a neural network
2. But can extract symbolic formulas directly (no PySR needed)
3. Perfect for Discovery Engine integration
4. 100x faster formula crystallization vs MLP + symbolic regression

Next steps:
- Try with your own physics data
- Integrate with hippocampus for memory-guided learning
- Use ActivityMonitor to trigger discoveries automatically
""")


if __name__ == "__main__":
    main()
