"""
Operation "Grid Pulse" - Elia Power Grid Frequency Dynamics
Physics-Informed Model: Swing Equation + Time-Varying Inertia/Damping

Swing Equation: df/dt = (1/M)(P_imbalance - D·f)
Hard Core: Static M_base, D_base integrator
Soft Shell: Learns residual (time-varying M, D effects)

Run: python main_grid.py
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from axiom_os.datasets.elia_grid import load_elia_grid, build_grid_sequences
from axiom_os.datasets.elia_grid import M_BASE_BELGIUM, D_BASE, F_NOMINAL
from axiom_os.layers.pinn_lstm import PhysicsInformedLSTM


def make_swing_hard_core(
    dt_sec: float = 60.0,
    M_base: float = M_BASE_BELGIUM,
    D_base: float = D_BASE,
    clamp_bounds: Optional[Tuple[float, float]] = (-1.0, 1.0),
):
    """
    Hard Core: Swing Equation integrator (static M, D).
    f_next = f_prev + dt * (1/M_base) * (P_imbalance - D_base * f_prev)

    Input coords: (B, 4) = [t_norm, u_MW, f_dev, df_dt]
    Output: (B, 1) = f_next (frequency deviation)
    clamp_bounds: (min, max) for f_dev; None = no clamp (for MPC nadir prediction).
    """
    def hard_core(x):
        if isinstance(x, torch.Tensor):
            v = x.float()
        else:
            v = torch.as_tensor(x, dtype=torch.float32)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        u = v[:, 1:2]   # imbalance MW
        f_prev = v[:, 2:3]  # frequency deviation
        df_dt = (1.0 / M_base) * (u - D_base * f_prev)
        f_next = f_prev + dt_sec * df_dt
        if clamp_bounds is not None:
            f_next = f_next.clamp(clamp_bounds[0], clamp_bounds[1])
        return f_next
    return hard_core


def train_grid_model(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    coords_train: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
) -> list:
    """Train the grid PINN."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_train, target_coords=coords_train)
        loss = nn.functional.mse_loss(pred, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (ep + 1) % 50 == 0:
            print(f"  Epoch {ep+1}/{epochs} loss={loss.item():.6f}")
    return losses


def main():
    print("=" * 60)
    print("Operation Grid Pulse - Elia Power Grid Frequency")
    print("=" * 60)

    # Load data (synthetic if no CSV)
    t, u, x, df_dt, meta = load_elia_grid(
        use_synthetic_if_fail=True,
        resample_min=1,
        n_days=7,
        seed=42,
    )
    print(f"\nData: {meta['data_source']} | n={len(x)} | dt={meta['dt_sec']:.0f}s")

    seq_len = 12
    X_seq, Y, coords = build_grid_sequences(t, u, x, seq_len, forecast_horizon=1, df_dt=df_dt)
    print(f"Sequences: {len(X_seq)} samples, seq_len={seq_len}")

    split = int(0.7 * len(X_seq))
    X_train = torch.from_numpy(X_seq[:split]).float()
    Y_train = torch.from_numpy(Y[:split]).float()
    coords_train = torch.from_numpy(coords[:split]).float()
    X_test = torch.from_numpy(X_seq[split:]).float()
    Y_test = torch.from_numpy(Y[split:]).float()
    coords_test = torch.from_numpy(coords[split:]).float()

    dt_sec = meta["dt_sec"]
    hard_core = make_swing_hard_core(dt_sec=dt_sec)

    model = PhysicsInformedLSTM(
        input_dim=4,
        hidden_dim=64,
        output_dim=1,
        seq_len=seq_len,
        hard_core_func=hard_core,
        lambda_res=1.0,
        num_layers=2,
    )

    print("\nTraining...")
    train_grid_model(model, X_train, Y_train, coords_train, epochs=200)

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test, target_coords=coords_test).numpy()
    pred_hard = hard_core(coords_test).numpy()

    mse_axiom = np.mean((pred_test - Y_test.numpy()) ** 2)
    mse_hard = np.mean((pred_hard - Y_test.numpy()) ** 2)
    r2_axiom = 1 - mse_axiom / np.var(Y_test.numpy())
    r2_hard = 1 - mse_hard / np.var(Y_test.numpy())

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Hard Core (static M): R2={r2_hard:.4f} MSE={mse_hard:.6f}")
    print(f"  Axiom (Hard+Soft):   R2={r2_axiom:.4f} MSE={mse_axiom:.6f}")

    # Find frequency event (nadir) - largest |f_dev|
    event_idx = np.argmax(np.abs(x))
    event_window = slice(max(0, event_idx - 30), min(len(x), event_idx + 30))
    print(f"\n  Frequency event (nadir) at idx {event_idx}: f_dev={x[event_idx]:.4f} Hz")

    # Visualization
    out_dir = ROOT / "axiom_os" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # Full timeline: Y_test[j] corresponds to t[split + j + seq_len]
        n_test = len(Y_test)
        t_test = np.array([t[min(split + j + seq_len, len(t) - 1)] for j in range(n_test)])
        ax = axes[0]
        ax.plot(t_test, Y_test.numpy().ravel(), "b-", alpha=0.7, label="True f_dev")
        ax.plot(t_test, pred_test.ravel(), "r--", alpha=0.8, label="Axiom (Dynamic)")
        ax.plot(t_test, pred_hard.ravel(), "g:", alpha=0.7, label="Hard Core (Static M)")
        ax.set_ylabel("Frequency deviation (Hz)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title("Grid Pulse: True vs Hard Core (Static M) vs Axiom (Dynamic M)")

        # Scatter: True vs Pred
        ax = axes[1]
        ax.scatter(Y_test.numpy(), pred_test, s=8, alpha=0.5, label="Axiom", c="red")
        ax.scatter(Y_test.numpy(), pred_hard, s=8, alpha=0.5, label="Hard Core", c="green")
        lims = [min(Y_test.min().item(), pred_test.min()), max(Y_test.max().item(), pred_test.max())]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlabel("True f_dev (Hz)")
        ax.set_ylabel("Predicted f_dev (Hz)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Event zoom: show test segment that contains largest |f_dev|
        ax = axes[2]
        test_true = Y_test.numpy().ravel()
        n_test = len(test_true)
        idx_plot = np.arange(n_test)
        ax.plot(idx_plot, test_true, "b-", lw=1.5, label="True f_dev")
        ax.plot(idx_plot, pred_test.ravel(), "r--", alpha=0.8, label="Axiom")
        ax.plot(idx_plot, pred_hard.ravel(), "g:", alpha=0.7, label="Hard Core (Static M)")
        nadir_idx = np.argmax(np.abs(test_true))
        ax.axvline(nadir_idx, color="gray", linestyle=":", alpha=0.7, label="Nadir")
        ax.set_xlabel("Test sample index")
        ax.set_ylabel("f_dev (Hz)")
        ax.set_title("Test Set: Frequency (Nadir = lowest point)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = out_dir / "grid_pulse_plot.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved: {out_path}")
    except Exception as e:
        print(f"Plot skip: {e}")

    return {"r2_axiom": r2_axiom, "r2_hard": r2_hard, "mse_axiom": mse_axiom, "mse_hard": mse_hard}


if __name__ == "__main__":
    main()
