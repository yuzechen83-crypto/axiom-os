"""
Einstein Module - The Thought Experiment
Level: Symplectic Validation & Imagination

Scenario: Damped Harmonic Oscillator F = -kx - cv
Task: Einstein extracts H = ½kx² + ½p² and identifies -cv as pure dissipation.
Einstein runs simulation using ONLY H (no dissipation).
Expectation: "Imagined" trajectory = perfect, non-decaying sine wave (Eternal Vibration).

Visual: Blue (Real Data, Decaying) | Red Dashed (RCLN Prediction, Decaying) | Gold (Einstein's Imagination, Eternal)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.hippocampus import EinsteinCore

# Reuse RCLN from verify_discovery
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_discovery import (
    simulate_damped_oscillator,
    RCLNOscillator,
    train_model,
    integrate_path,
)


def build_dynamics_model(rcln: nn.Module) -> nn.Module:
    """
    Wrap RCLN so f(x,v) = (dx/dt, dv/dt) = (v, a).
    RCLN takes [x,v] and outputs acceleration a.
    """

    class DynamicsWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, xv: torch.Tensor) -> torch.Tensor:
            a = self.inner(xv)
            v = xv[..., 1:2]
            return torch.cat([v, a], dim=-1)

    return DynamicsWrapper(rcln)


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    print("=" * 60)
    print("Einstein Module: The Thought Experiment")
    print("Damped Harmonic Oscillator → Eternal Vibration")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, k, c = 1.0, 10.0, 0.5
    n_steps = 1000
    dt = 0.02
    x0, v0 = 1.0, 0.0

    # 1. Generate real data (decaying)
    print("\n1. Data: Damped Oscillator (Real World)")
    t_real, x_real, v_real, a_real = simulate_damped_oscillator(
        m=m, k=k, c=c, x0=x0, v0=v0, dt=dt, n_steps=n_steps
    )
    state_t = torch.from_numpy(np.stack([x_real, v_real], axis=-1)).float().to(device)
    a_t = torch.from_numpy(a_real).float().unsqueeze(-1).to(device)
    print(f"   {n_steps} steps, decay visible in amplitude")

    # 2. Train RCLN (Hard: -kx, Soft: learns -cv)
    print("\n2. RCLN: Train on decaying data")
    model = RCLNOscillator(k=k, m=m, soft_init_scale=0.01).to(device)
    train_model(model, state_t, a_t, epochs=500, lr=1e-2, device=device)

    # 3. Einstein: Decompose dynamics
    print("\n3. Einstein Core: Decompose f(x,v) → H + R")
    dynamics = build_dynamics_model(model)
    dynamics.eval()
    einstein = EinsteinCore(hnn_hidden=32, r_hidden=16, lr=1e-2, epochs=3000)

    H_func = einstein.decompose_dynamics(
        dynamics,
        data=state_t,
        state_dim=2,
        device=device,
    )
    print("   Extracted Hamiltonian H(q,p)")

    # 4. Einstein: Dream in symplectic (eternal oscillation)
    print("\n4. Einstein: Dream (Leapfrog, T=10^5 steps for validation)")
    n_dream = 100_000
    t_dream, q_dream, p_dream, drift_ok = einstein.dream_in_symplectic(
        H_func,
        q0=np.array([x0]),
        p0=np.array([v0]),
        dt=dt,
        n_steps=n_dream,
        device=device,
    )
    x_einstein = q_dream[:, 0]
    v_einstein = p_dream[:, 0]
    print(f"   Energy drift OK: {drift_ok}")

    # For plot: use same time window as real data (0 to n_steps*dt)
    n_plot = n_steps
    t_plot = t_dream[: n_plot + 1]
    x_einstein_plot = x_einstein[: n_plot + 1]
    v_einstein_plot = v_einstein[: n_plot + 1]

    # 5. RCLN prediction (decaying, matches data)
    x_rcln, v_rcln = integrate_path(model, x0, v0, t_real, use_hard_only=False, device=device)

    # 6. Visualization
    print("\n5. Visualization")
    if not HAS_MPL:
        print("   (matplotlib not installed, skipping plot)")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Position vs Time
        ax = axes[0]
        ax.plot(t_real, x_real, "b-", label="Real Data (Decaying)", linewidth=2)
        ax.plot(t_real, x_rcln, "r--", label="RCLN Prediction (Decaying)", linewidth=1.5)
        ax.plot(t_plot, x_einstein_plot, color="gold", linestyle="-", linewidth=1.5,
                label="Einstein's Imagination (Eternal)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position x")
        ax.set_title("Einstein Thought Experiment: The Underlying Truth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase portrait
        ax = axes[1]
        ax.plot(x_real, v_real, "b-", label="Real (Decaying spiral)", linewidth=2)
        ax.plot(x_rcln, v_rcln, "r--", label="RCLN", linewidth=1.5)
        ax.plot(x_einstein_plot, v_einstein_plot, color="gold", linestyle="-", linewidth=1,
                label="Einstein (Eternal ellipse)")
        ax.set_xlabel("Position x")
        ax.set_ylabel("Velocity v")
        ax.set_title("Phase Portrait")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        out = Path(__file__).resolve().parent / "einstein_thought_experiment.png"
        plt.savefig(out, dpi=150)
        print(f"   Saved: {out}")

    # 7. Validation
    amp_real_end = np.abs(x_real[-100:]).mean()
    amp_einstein_end = np.abs(x_einstein[-1000:]).mean()
    amp_einstein_start = np.abs(x_einstein[:1000]).mean()
    real_decayed = amp_real_end < 0.5 * np.abs(x_real[0])
    einstein_eternal = amp_einstein_end > 0.5 * amp_einstein_start

    success = real_decayed and einstein_eternal
    print(f"\n6. Validation")
    print(f"   Real data decayed: {real_decayed}")
    print(f"   Einstein eternal (no decay): {einstein_eternal}")
    print(f"   Energy conserved: {drift_ok}")
    print(f"   Success: {success}")

    print("\n" + "=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
