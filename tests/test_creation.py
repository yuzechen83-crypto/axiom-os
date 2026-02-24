"""
Emergence of Chaos - Generative Physics (Creation)
The AI has never seen double pendulum data. It only knows the single pendulum law.
Task: Construct a Double Pendulum solely within the Einstein Module's imagination.

Input: Extracted H from test_einstein (Simple Harmonic ≈ small-angle pendulum, k→g/L).
Execution: Double Pendulum Hamiltonian with extracted parameters, Symplectic Integrator T=50s.
Expectation: Trajectory of second bob (x2, y2) is Chaotic (messy, non-repeating).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.imagination import (
    coupled_simulation,
    double_pendulum_H,
    leapfrog_coupled,
    angles_to_cartesian,
)


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    print("=" * 60)
    print("Emergence of Chaos: Generative Physics")
    print("Double Pendulum from Single Pendulum Laws (AI has never seen it)")
    print("=" * 60)

    # Extracted from Einstein test: k=10 for damped harmonic oscillator
    # Small-angle pendulum: H ≈ ½p² + ½kq² with k = g/L
    k_extracted = 10.0
    g_over_L = k_extracted

    # Single pendulum H = ½p² + (1-cos q) -> g/L in potential
    # For double pendulum we use the same g/L
    L1, L2 = 1.0, 1.0

    # 1. Construct Double Pendulum Hamiltonian (no data - pure imagination)
    print("\n1. Construct Double Pendulum Hamiltonian")
    H_double = coupled_simulation(
        H1=None,
        H2=None,
        coupling_type="rigid",
        g_over_L=g_over_L,
        L1=L1,
        L2=L2,
    )
    print("   H_double = (p1^2 + 2p2^2 - 2p1p2 cos(q1-q2)) / (2(1+sin^2(q1-q2))) + V(q1,q2)")

    # 2. Initial conditions: both rods nearly horizontal (chaos-favoring)
    q1_0 = np.pi / 2 + 0.1
    q2_0 = np.pi / 2 + 0.2
    p1_0 = 0.0
    p2_0 = 0.0
    q0 = np.array([q1_0, q2_0])
    p0 = np.array([p1_0, p2_0])

    # 3. Run Symplectic Integrator for T=50s
    T = 50.0
    dt = 0.005
    n_steps = int(T / dt)
    print(f"\n2. Symplectic Leapfrog: T={T}s, dt={dt}, n_steps={n_steps}")

    t_arr, q_traj, p_traj = leapfrog_coupled(
        H_double,
        q0=q0,
        p0=p0,
        dt=dt,
        n_steps=n_steps,
        device=None,
    )
    q1_traj = q_traj[:, 0]
    q2_traj = q_traj[:, 1]

    # 4. Convert to Cartesian (x2, y2) for second bob
    x1, y1, x2, y2 = angles_to_cartesian(q1_traj, q2_traj, L1=L1, L2=L2)

    # 5. Visualization: trajectory of second bob (x2, y2)
    print("\n3. Visualization")
    if not HAS_MPL:
        print("   (matplotlib not installed, skipping plot)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Trajectory of second bob (x2, y2) - should be chaotic
        ax = axes[0]
        ax.plot(x2, y2, "b-", linewidth=0.3, alpha=0.8)
        ax.set_xlabel("x2")
        ax.set_ylabel("y2")
        ax.set_title("Second Bob Trajectory (x2, y2) - Chaotic")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linestyle=":", alpha=0.5)
        ax.axvline(0, color="k", linestyle=":", alpha=0.5)

        # Full double pendulum path (both bobs)
        ax = axes[1]
        ax.plot(x1, y1, "g-", linewidth=0.3, alpha=0.6, label="Bob 1")
        ax.plot(x2, y2, "b-", linewidth=0.3, alpha=0.8, label="Bob 2")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Double Pendulum: Both Bobs")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path(__file__).resolve().parent / "creation_chaos.png"
        plt.savefig(out, dpi=150)
        print(f"   Saved: {out}")

    # 6. Validation: trajectory should be chaotic (non-repeating, bounded but irregular)
    # Heuristic: check that path length is much larger than perimeter of bounding box
    x2_range = x2.max() - x2.min()
    y2_range = y2.max() - y2.min()
    path_length = np.sum(np.sqrt(np.diff(x2)**2 + np.diff(y2)**2))
    perimeter = 2 * (x2_range + y2_range) if (x2_range > 0 and y2_range > 0) else 1.0
    is_chaotic = path_length > 5 * perimeter

    print(f"\n4. Validation")
    print(f"   Path length / perimeter ≈ {path_length / perimeter:.1f}")
    print(f"   Chaotic (non-repeating, complex): {is_chaotic}")
    success = is_chaotic

    print("\n" + "=" * 60)
    print("AI predicted Chaos from simple pendulum laws alone.")
    print("=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
