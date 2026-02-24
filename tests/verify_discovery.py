"""
Discovery Verification Test
SPNN-Evo 2.0: Verify that the architecture can discover missing physics from data.

Scenario: Damped Harmonic Oscillator
Ground Truth: mẍ + cẋ + kx = 0 (Mass-Spring-Damper)
System Knowledge: Only F = -kx (Spring force). Ignorant of friction (c=0).
Task: Soft Shell should learn the missing friction term cẋ.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.upi import UPIState, Units
from spnn_evo.core.hippocampus import HippocampusLibrary

# Acceleration = LT^-2
ACCELERATION = Units(0, 1, -2, 0, 0)


# -----------------------------------------------------------------------------
# 1. Data Generation
# -----------------------------------------------------------------------------

def simulate_damped_oscillator(
    m: float = 1.0,
    k: float = 10.0,
    c: float = 0.5,
    x0: float = 1.0,
    v0: float = 0.0,
    dt: float = 0.02,
    n_steps: int = 1000,
) -> tuple:
    """
    Simulate mẍ + cẋ + kx = 0.
    Returns: t, x, v, a (time, position, velocity, acceleration)
    """
    def deriv(y, t):
        x, v = y
        a = -(c / m) * v - (k / m) * x
        return [v, a]

    t = np.arange(n_steps) * dt
    sol = odeint(deriv, [x0, v0], t)
    x = sol[:, 0]
    v = sol[:, 1]
    a = -(c / m) * v - (k / m) * x
    return t, x, v, a


def wrap_in_upi_states(
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    device: torch.device,
) -> tuple:
    """Wrap data in UPI States with correct units."""
    # State: [x, v] -> LENGTH, VELOCITY
    state = np.stack([x, v], axis=-1).astype(np.float64)
    state_t = torch.from_numpy(state)

    # Target: a (acceleration) -> ACCELERATION
    a_t = torch.from_numpy(a.astype(np.float64)).unsqueeze(-1)

    # UPI States (state [x,v]: LENGTH, VELOCITY; target a: ACCELERATION)
    states = [
        UPIState(
            state_t[i : i + 1],
            (0, 1, -1, 0, 0),  # phase-space units
            spacetime=torch.tensor([t[i], 0, 0, 0], dtype=torch.float64),
            semantics="oscillator_state",
        )
        for i in range(len(t))
    ]
    targets = [
        UPIState(
            a_t[i : i + 1],
            (0, 1, -2, 0, 0),  # acceleration LT^-2
            spacetime=torch.tensor([t[i], 0, 0, 0], dtype=torch.float64),
            semantics="acceleration",
        )
        for i in range(len(t))
    ]
    return state_t, a_t, states, targets


# -----------------------------------------------------------------------------
# 2. Model: RCLN with F=-kx Hard, 2-Layer Soft MLP
# -----------------------------------------------------------------------------

class FHardSpring(nn.Module):
    """Hard Logic: F = -kx - c*v (spring + optional crystallized damping)"""

    def __init__(self, k: float = 10.0, m: float = 1.0, c_damp: float = 0.0):
        super().__init__()
        self.k = k
        self.m = m
        self.c_damp = c_damp

    def set_damping(self, c: float) -> None:
        """Crystallization: move discovered damping from Soft to Hard."""
        self.c_damp = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[:, 0:1]
        vel = x[:, 1:2]
        a_spring = -(self.k / self.m) * pos
        a_damp = -(self.c_damp / self.m) * vel if self.c_damp != 0 else torch.zeros_like(a_spring)
        return a_spring + a_damp


class FSoftShell(nn.Module):
    """Soft Shell: 2-layer MLP, small weights. Learns missing physics."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 16, out_dim: int = 1, init_scale: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        for p in self.net.parameters():
            p.data.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RCLNOscillator(nn.Module):
    """
    RCLN for oscillator: y = F_hard(x) + F_soft(x)
    Hard: a = -kx/m (spring only)
    Soft: learns residual (friction -cv/m)
    """

    def __init__(self, k: float = 10.0, m: float = 1.0, soft_init_scale: float = 0.01):
        super().__init__()
        self.f_hard = FHardSpring(k=k, m=m)
        self.f_soft = FSoftShell(in_dim=2, hidden_dim=16, out_dim=1, init_scale=soft_init_scale)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> tuple:
        y_hard = self.f_hard(x)
        y_soft = self.f_soft(x)
        y = y_hard + y_soft
        if return_components:
            return y, y_hard, y_soft
        return y


# -----------------------------------------------------------------------------
# 3. Evolution Scheduler (Phase 1 -> Phase 2)
# -----------------------------------------------------------------------------

class EvolutionScheduler:
    """Phase 1: Training. Phase 2: Crystallization / Discovery."""

    def __init__(self, phase1_steps: int = 800, total_steps: int = 1000):
        self.phase1_steps = phase1_steps
        self.total_steps = total_steps

    def phase(self, step: int) -> int:
        return 1 if step < self.phase1_steps else 2

    def is_phase1(self, step: int) -> bool:
        return self.phase(step) == 1

    def is_phase2(self, step: int) -> bool:
        return self.phase(step) == 2

    def lambda_soft(self, step: int) -> float:
        """Soft contribution weight (1.0 in phase 1, can reduce in phase 2)"""
        return 1.0


# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

def train_model(
    model: RCLNOscillator,
    state: torch.Tensor,
    target: torch.Tensor,
    epochs: int = 500,
    lr: float = 1e-2,
    device: torch.device = None,
) -> list:
    """Train model. Goal: Soft learns to minimize error from missing friction."""
    device = device or torch.device("cpu")
    model = model.to(device)
    state = state.to(device)
    target = target.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = EvolutionScheduler(phase1_steps=int(epochs * 0.8), total_steps=epochs)
    losses = []

    model.train()
    for step in range(epochs):
        optimizer.zero_grad()
        pred, _, _ = model(state, return_components=True)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"  Epoch {step+1}/{epochs} loss={loss.item():.6f} phase={scheduler.phase(step)}")
    return losses


# -----------------------------------------------------------------------------
# 5. Discovery Loop: Extract F_soft vs v
# -----------------------------------------------------------------------------

def extract_soft_vs_velocity(
    model: RCLNOscillator,
    v_range: tuple = (-5.0, 5.0),
    n_samples: int = 101,
    x_fixed: float = 0.0,
    device: torch.device = None,
) -> tuple:
    """
    Isolate Soft Shell. Feed v ∈ [v_min, v_max] with x=x_fixed.
    F_soft should approximate -c*v = -0.5*v (friction).
    """
    device = device or torch.device("cpu")
    model.eval()
    v_vals = np.linspace(v_range[0], v_range[1], n_samples)
    x_vals = np.full_like(v_vals, x_fixed)
    inputs = np.stack([x_vals, v_vals], axis=-1).astype(np.float32)
    x_t = torch.from_numpy(inputs).to(device)

    with torch.no_grad():
        _, _, f_soft = model(x_t, return_components=True)
    f_soft_np = f_soft.cpu().numpy().ravel()
    return v_vals, f_soft_np


def discover_formula_linear_regression(v: np.ndarray, f_soft: np.ndarray) -> tuple:
    """
    Linear regression: F_soft ≈ c0 + c1*v
    Success: c1 ≈ -0.5 (friction coefficient)
    """
    X = np.column_stack([np.ones_like(v), v])
    coeffs, residuals, rank, s = np.linalg.lstsq(X, f_soft, rcond=None)
    c0, c1 = coeffs[0], coeffs[1]
    formula = f"F_soft = {c0:.4f} + {c1:.4f} * v"
    return formula, c0, c1


# -----------------------------------------------------------------------------
# 6. Path Integration: True vs Hard vs SPNN
# -----------------------------------------------------------------------------

def integrate_path(
    model: RCLNOscillator,
    x0: float,
    v0: float,
    t: np.ndarray,
    use_hard_only: bool = False,
    device: torch.device = None,
) -> tuple:
    """Integrate trajectory using model (or hard-only)."""
    device = device or torch.device("cpu")
    model.eval()
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.02
    x_path = [x0]
    v_path = [v0]

    x_cur, v_cur = x0, v0
    with torch.no_grad():
        for _ in range(len(t) - 1):
            state = torch.tensor([[x_cur, v_cur]], dtype=torch.float32, device=device)
            if use_hard_only:
                a = model.f_hard(state)
            else:
                a, _, _ = model(state, return_components=True)
            a_val = a.cpu().item()
            v_cur = v_cur + a_val * dt
            x_cur = x_cur + v_cur * dt
            x_path.append(x_cur)
            v_path.append(v_cur)
    return np.array(x_path), np.array(v_path)


# -----------------------------------------------------------------------------
# 7. Main + Visualization
# -----------------------------------------------------------------------------

def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False

    print("=" * 60)
    print("SPNN-Evo 2.0: Discovery Verification")
    print("Damped Harmonic Oscillator")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, k, c = 1.0, 10.0, 0.5
    n_steps = 1000
    dt = 0.02

    # 1. Data Generation
    print("\n1. Data Generation")
    t, x_true, v_true, a_true = simulate_damped_oscillator(
        m=m, k=k, c=c, x0=1.0, v0=0.0, dt=dt, n_steps=n_steps
    )
    state_t, a_t, states, targets = wrap_in_upi_states(t, x_true, v_true, a_true, device)
    state_t = state_t.float()
    a_t = a_t.float()
    print(f"   Generated {n_steps} steps. State shape: {state_t.shape}, Target shape: {a_t.shape}")

    # 2. Model Setup
    print("\n2. Model Setup")
    model = RCLNOscillator(k=k, m=m, soft_init_scale=0.01)
    print("   Hard: F = -kx (spring only, c=0)")
    print("   Soft: 2-layer MLP, small init")

    # 3. Training
    print("\n3. Training (EvolutionScheduler Phase 1 -> Phase 2)")
    train_model(model, state_t, a_t, epochs=500, lr=1e-2, device=device)

    # 4. Discovery Extraction
    print("\n4. Discovery Loop: Extract F_soft vs v")
    v_vals, f_soft_vals = extract_soft_vs_velocity(
        model, v_range=(-5, 5), n_samples=101, x_fixed=0.0, device=device
    )
    formula, c0, c1 = discover_formula_linear_regression(v_vals, f_soft_vals)
    print(f"   Discovered: {formula}")
    print(f"   Coefficient of v: {c1:.4f} (expected ~ -0.5 for friction c)")
    success = abs(c1 - (-0.5)) < 0.3
    print(f"   Success: {success}")

    # Crystallization: move discovered term from Soft to Hard
    if success:
        library = HippocampusLibrary()
        library.update_physics("damping", abs(c1))
        model.f_hard.set_damping(abs(c1))
        for p in model.f_soft.parameters():
            p.data.mul_(0.01)
        print("   Crystallized: damping moved to Hard Core, Soft reset")

    # 5. Path Integration
    print("\n5. Path Comparison")
    x_hard, v_hard = integrate_path(model, 1.0, 0.0, t, use_hard_only=True, device=device)
    x_spnn, v_spnn = integrate_path(model, 1.0, 0.0, t, use_hard_only=False, device=device)

    # 6. Visualization
    print("\n6. Visualization")
    if not HAS_MATPLOTLIB:
        print("   (matplotlib not installed, skipping plots)")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax = axes[0, 0]
        ax.plot(t, x_true, "b-", label="True (mẍ+cẋ+kx=0)", linewidth=2)
        ax.plot(t, x_hard, "r--", label="Hard-only (F=-kx)", linewidth=1.5)
        ax.plot(t, x_spnn, "g-.", label="SPNN (Hard+Soft)", linewidth=1.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Position x")
        ax.set_title("Position: True vs Hard vs SPNN")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(t, v_true, "b-", label="True", linewidth=2)
        ax.plot(t, v_hard, "r--", label="Hard-only", linewidth=1.5)
        ax.plot(t, v_spnn, "g-.", label="SPNN", linewidth=1.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity v")
        ax.set_title("Velocity: True vs Hard vs SPNN")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.scatter(v_vals, f_soft_vals, s=5, alpha=0.7, label="F_soft(v)")
        v_line = np.linspace(-5, 5, 50)
        f_fit = c0 + c1 * v_line
        ax.plot(v_line, f_fit, "r-", linewidth=2, label=f"Fit: {formula}")
        ax.axhline(0, color="k", linestyle=":", alpha=0.5)
        ax.axvline(0, color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("Velocity v")
        ax.set_ylabel("F_soft (learned)")
        ax.set_title("Discovery: F_soft vs v (x=0)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(x_true, v_true, "b-", label="True", linewidth=2)
        ax.plot(x_hard, v_hard, "r--", label="Hard-only", linewidth=1.5)
        ax.plot(x_spnn, v_spnn, "g-.", label="SPNN", linewidth=1.5)
        ax.set_xlabel("Position x")
        ax.set_ylabel("Velocity v")
        ax.set_title("Phase Portrait")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        out_path = ROOT / "tests" / "discovery_verification.png"
        plt.savefig(out_path, dpi=150)
        print(f"   Saved: {out_path}")

    print("\n" + "=" * 60)
    print("Discovery Verification Complete")
    print("=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
