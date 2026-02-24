"""
Nonlinear Drag Discovery Test
SPNN-Evo 2.0: Verify discovery of quadratic drag (Air Resistance).

Physics: Ground Truth F = -kx - c·v·|v| (Quadratic Drag)
System Knowledge: Only F = -kx
Expectation:
  - F_soft vs v should show a parabola
  - Symbolic regression should identify v² term
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spnn_evo.core.upi import UPIState, Units


# -----------------------------------------------------------------------------
# 1. Data Generation: Quadratic Drag
# -----------------------------------------------------------------------------

def simulate_quadratic_drag(
    m: float = 1.0,
    k: float = 10.0,
    c: float = 0.1,
    x0: float = 1.0,
    v0: float = 0.0,
    dt: float = 0.02,
    n_steps: int = 1000,
) -> tuple:
    """
    Simulate mẍ + c·v·|v| + kx = 0  (Quadratic drag: F_drag = -c·v·|v|)
    a = -(k/m)x - (c/m)·v·|v|
    """
    def deriv(y, t):
        x, v = y
        a = -(k / m) * x - (c / m) * v * np.abs(v)
        return [v, a]

    t = np.arange(n_steps) * dt
    sol = odeint(deriv, [x0, v0], t)
    x = sol[:, 0]
    v = sol[:, 1]
    a = -(k / m) * x - (c / m) * v * np.abs(v)
    return t, x, v, a


def wrap_in_upi_states(t, x, v, a, device):
    state = np.stack([x, v], axis=-1).astype(np.float64)
    state_t = torch.from_numpy(state)
    a_t = torch.from_numpy(a.astype(np.float64)).unsqueeze(-1)
    return state_t, a_t


# -----------------------------------------------------------------------------
# 2. Model: RCLN with F=-kx Hard, Soft learns -c·v·|v|
# -----------------------------------------------------------------------------

class FHardSpring(nn.Module):
    """Hard: F = -kx only (no drag)"""

    def __init__(self, k: float = 10.0, m: float = 1.0):
        super().__init__()
        self.k = k
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[:, 0:1]
        return -(self.k / self.m) * pos


class FSoftShell(nn.Module):
    """
    Soft: learns quadratic drag -c·v·|v|/m
    Uses SiLU (Swish): smooth, non-monotonic, unbounded - superior for physics polynomials.
    """
    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, out_dim: int = 1, init_scale: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        for p in self.net.parameters():
            p.data.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RCLNOscillator(nn.Module):
    def __init__(self, k: float = 10.0, m: float = 1.0, soft_init_scale: float = 0.01):
        super().__init__()
        self.f_hard = FHardSpring(k=k, m=m)
        self.f_soft = FSoftShell(in_dim=2, hidden_dim=64, out_dim=1, init_scale=soft_init_scale)

    def forward(self, x: torch.Tensor, return_components: bool = False):
        y_hard = self.f_hard(x)
        y_soft = self.f_soft(x)
        y = y_hard + y_soft
        if return_components:
            return y, y_hard, y_soft
        return y


# -----------------------------------------------------------------------------
# 3. Discovery: Polynomial fit for v²
# -----------------------------------------------------------------------------

def extract_soft_vs_velocity(model, v_range=(-5, 5), n_samples=101, x_fixed=0.0, device=None):
    device = device or torch.device("cpu")
    model.eval()
    v_vals = np.linspace(v_range[0], v_range[1], n_samples)
    x_vals = np.full_like(v_vals, x_fixed)
    inputs = np.stack([x_vals, v_vals], axis=-1).astype(np.float32)
    x_t = torch.from_numpy(inputs).to(device)
    with torch.no_grad():
        _, _, f_soft = model(x_t, return_components=True)
    return v_vals, f_soft.cpu().numpy().ravel()


def discover_formula_polynomial(v: np.ndarray, f_soft: np.ndarray, degree: int = 2, debug: bool = False) -> tuple:
    """
    Regression with Signed Quadratic: F_soft ≈ c1*v + c2*(v·|v|)
    Physics: F=0 at v=0, so fit_intercept=False.
    Target F ∝ -v·|v| is ODD (antisymmetric).
    """
    v_data = np.asarray(v, dtype=np.float64).flatten()
    f_soft_pred = np.asarray(f_soft, dtype=np.float64).flatten()
    assert len(v_data) == len(f_soft_pred), "v and F_soft must have same length"

    if debug:
        print("   [DEBUG] V shape:", v_data.shape)
        print("   [DEBUG] F_soft shape:", f_soft_pred.shape)
        print("   [DEBUG] Feature v*|v| mean:", np.mean(v_data * np.abs(v_data)))
        print("   [DEBUG] Target F_soft mean:", np.mean(f_soft_pred))

    # Explicit feature construction: [v, v*|v|]
    v_col = v_data.reshape(-1, 1)
    feat_quad = v_col * np.abs(v_col)
    X_features = np.hstack([v_col, feat_quad])

    try:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=False)  # Physics: F=0 at v=0
        reg.fit(X_features, f_soft_pred)
        c1 = reg.coef_[0]
        c2 = reg.coef_[1]
        f_fit = reg.predict(X_features)
    except ImportError:
        coeffs, _, _, _ = np.linalg.lstsq(X_features, f_soft_pred, rcond=None)
        c1, c2 = coeffs[0], coeffs[1]
        f_fit = X_features @ coeffs

    # Validation: quadratic term MUST be non-zero
    assert abs(c2) > 0.005, f"Quadratic term coef[1]={c2:.6f} too small (expected > 0.005)"
    if debug:
        print("   [DEBUG] coef[0] (v):", c1, "coef[1] (v*|v|):", c2)

    terms = []
    if abs(c1) > 1e-6:
        terms.append(f"{c1:+.4f}*v")
    if abs(c2) > 1e-6:
        terms.append(f"{c2:+.4f}*v*|v|")
    formula = "F_soft = " + ("".join(terms) if terms else "0")
    return formula, 0.0, c1, c2, f_fit


# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------

def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    print("=" * 60)
    print("SPNN-Evo 2.0: Nonlinear Drag Discovery")
    print("Ground Truth: F = -kx - c·v·|v| (Quadratic Drag)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, k, c = 1.0, 10.0, 0.1
    n_steps = 1000
    dt = 0.02

    # 1. Data
    print("\n1. Data Generation")
    x0, v0 = 2.0, 5.0  # High-velocity regime: forces network to explore where drag is strongest
    t, x_true, v_true, a_true = simulate_quadratic_drag(
        m=m, k=k, c=c, x0=x0, v0=v0, dt=dt, n_steps=n_steps
    )
    state_t, a_t = wrap_in_upi_states(t, x_true, v_true, a_true, device)
    state_t = state_t.float()
    a_t = a_t.float()
    print(f"   {n_steps} steps, c={c}, x0={x0}, v0={v0} (high-velocity regime)")

    # 2. Model
    print("\n2. Model Setup")
    model = RCLNOscillator(k=k, m=m, soft_init_scale=0.01)
    print("   Hard: F = -kx only")
    print("   Soft: 2-layer MLP with SiLU (unbounded, for v^2)")

    # 3. Training
    print("\n3. Training")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model = model.to(device)
    for step in range(800):
        optimizer.zero_grad()
        pred, _, _ = model(state_t, return_components=True)
        loss = nn.functional.mse_loss(pred, a_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 200 == 0:
            print(f"   Epoch {step+1}/800 loss={loss.item():.6f}")

    # 4. Discovery
    print("\n4. Discovery: F_soft vs v (expect parabola)")
    v_vals, f_soft_vals = extract_soft_vs_velocity(
        model, v_range=(-5, 5), n_samples=101, x_fixed=0.0, device=device
    )
    formula, c0, c1, c2, f_fit = discover_formula_polynomial(v_vals, f_soft_vals, degree=2, debug=False)
    print(f"   Discovered: {formula}")
    print(f"   v*|v| coefficient: {c2:.4f} (expected ~ -{c} for quadratic drag)")

    # Success: v·|v| term detected (signed quadratic), good R2 (parity fix: red fit matches blue)
    has_signed_sq = abs(c2) > 1e-6
    correct_sign = c2 < 0
    ss_res = np.sum((f_soft_vals - f_fit) ** 2)
    ss_tot = np.sum((f_soft_vals - np.mean(f_soft_vals)) ** 2) + 1e-10
    r2 = 1 - ss_res / ss_tot
    no_saturation = r2 > 0.90
    success = no_saturation and has_signed_sq
    print(f"   Has v*|v| term: {has_signed_sq}, Correct sign: {correct_sign}")
    print(f"   R2 (F_soft vs quadratic fit): {r2:.4f} (target > 0.90)")
    print(f"   Success: {success}")

    # 5. Visualization
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        ax.scatter(v_vals, f_soft_vals, s=8, alpha=0.7, label="F_soft(v)")
        # Red line: same v array as scatter, apply discovered coefficients
        v_plot = np.sort(v_vals)  # Sort for smooth curve
        f_fit_plot = c1 * v_plot + c2 * (v_plot * np.abs(v_plot))
        ax.plot(v_plot, f_fit_plot, "r-", linewidth=2, label=f"Fit: {formula[:50]}...")
        ax.axhline(0, color="k", linestyle=":", alpha=0.5)
        ax.axvline(0, color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("Velocity v")
        ax.set_ylabel("F_soft (learned)")
        ax.set_title("Discovery: F_soft vs v (antisymmetric v*|v|)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t, x_true, "b-", label="True (quadratic drag)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position x")
        ax.set_title("Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = ROOT / "tests" / "nonlinear_drag_discovery.png"
        plt.savefig(out, dpi=150)
        print(f"\n   Saved: {out}")

    print("\n" + "=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
