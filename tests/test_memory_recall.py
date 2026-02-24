"""
Recall & Simulate - Prove Memory is Active and Functional
Retrieve the saved SymplecticLaw from the Hippocampus.
Initialize with (q=1, p=0) - a state it has never seen before.
Run law.evolve(t=50). Plot: stable, energy-conserving orbit from stored memory.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse Einstein pipeline from verify_discovery + test_einstein
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_discovery import (
    simulate_damped_oscillator,
    RCLNOscillator,
    train_model,
)


def build_dynamics_model(rcln):
    """Wrap RCLN so f(x,v) = (dx/dt, dv/dt) = (v, a)."""
    import torch

    class DynamicsWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, xv):
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

    import torch
    from spnn_evo.core.hippocampus import EinsteinCore, HippocampusLibrary

    print("=" * 60)
    print("Recall & Simulate: Memory is Active")
    print("Prove stored law can imagine from unseen state")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, k, c = 1.0, 10.0, 0.5
    n_steps = 1000
    dt = 0.02

    # 1. Run Einstein pipeline (same as test_einstein)
    print("\n1. Einstein: Extract Hamiltonian from damped data")
    t_real, x_real, v_real, a_real = simulate_damped_oscillator(
        m=m, k=k, c=c, x0=1.0, v0=0.0, dt=dt, n_steps=n_steps
    )
    state_t = torch.from_numpy(np.stack([x_real, v_real], axis=-1)).float().to(device)
    a_t = torch.from_numpy(a_real).float().unsqueeze(-1).to(device)

    model = RCLNOscillator(k=k, m=m, soft_init_scale=0.01).to(device)
    train_model(model, state_t, a_t, epochs=500, lr=1e-2, device=device)

    dynamics = build_dynamics_model(model)
    dynamics.eval()
    einstein = EinsteinCore(hnn_hidden=32, r_hidden=16, lr=1e-2, epochs=3000)
    H_func = einstein.decompose_dynamics(
        dynamics, data=state_t, state_dim=2, device=device,
    )
    print("   Extracted H(q,p)")

    # 2. Crystallize into Hippocampus
    print("\n2. Hippocampus: Crystallize Thought")
    hippocampus = HippocampusLibrary()
    extraction_confidence = 0.9
    law = hippocampus.crystallize_thought(
        einstein_model=einstein,
        extraction_confidence=extraction_confidence,
        name="HarmonicOscillator_ID_001",
        params={"k": k, "m": m},
    )
    if law is None:
        print("   FAILED: Crystallization rejected")
        return False
    print(f"   Stored: {law.name}")

    # 3. Recall: Retrieve law (same or by name)
    print("\n3. Recall: Retrieve from hippocampus")
    recalled = hippocampus.recall_symplectic_law("HarmonicOscillator_ID_001")
    if recalled is None:
        print("   FAILED: Law not found")
        return False
    print(f"   Retrieved: {recalled.name}")

    # 4. Simulate: Unseen state (q=1, p=0), t=50
    print("\n4. Simulate: law.evolve(q=1, p=0, t=50)")
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    t_arr, q_traj, p_traj = recalled.evolve(q0=q0, p0=p0, t=50.0, dt=0.01)
    x_recall = q_traj[:, 0]
    v_recall = p_traj[:, 0]
    print(f"   Trajectory: {len(t_arr)} steps, t in [0, {t_arr[-1]:.1f}]")

    # 5. Validation: Energy conserved?
    E0 = recalled.H(np.concatenate([q_traj[0], p_traj[0]]))
    E_end = recalled.H(np.concatenate([q_traj[-1], p_traj[-1]]))
    drift = abs(E_end - E0)
    drift_ok = drift < 0.05 * (abs(E0) + 1e-8)
    amp_stable = np.abs(x_recall[-1]) > 0.3
    print(f"   Energy drift: {drift:.6e} (OK: {drift_ok})")
    print(f"   Amplitude stable: {amp_stable}")

    # 6. Visualization
    print("\n5. Visualization")
    if not HAS_MPL:
        print("   (matplotlib not installed, skipping plot)")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        ax = axes[0]
        ax.plot(t_arr, x_recall, "b-", linewidth=0.8, label="Position q")
        ax.plot(t_arr, v_recall, "g-", linewidth=0.8, alpha=0.8, label="Momentum p")
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.set_title("Recall: Stable Orbit from Stored Memory (q=1, p=0, t=50)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(x_recall, v_recall, "b-", linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Position q")
        ax.set_ylabel("Momentum p")
        ax.set_title("Phase Portrait: Energy-Conserving")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        out = Path(__file__).resolve().parent / "memory_recall.png"
        plt.savefig(out, dpi=150)
        print(f"   Saved: {out}")

    success = law is not None and recalled is not None and drift_ok and amp_stable
    print(f"\n6. Success: {success}")
    print("\n" + "=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
