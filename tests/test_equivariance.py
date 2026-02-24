"""
O(3) Equivariance Test for Clifford RCLN
Proves that the Clifford network is rotationally equivariant:
  Model(R(x)) ≈ R(Model(x))
where R is a 90° rotation around the Z-axis.
A standard MLP fails this test with massive error.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.core.clifford_ops import N_BLADES
from axiom_os.layers.clifford_nn import CliffordLinear, EquivariantCliffordLinear, CliffordActivation
from axiom_os.layers.rcln import RCLNLayer, CustomCliffordSoftShell, MLPSoftShell, _to_multivector, N_BLADES_3D


# --- Rotation utilities ---

def rotation_z_90() -> torch.Tensor:
    """90° counterclockwise around Z: (x,y,z) -> (-y, x, z)."""
    return torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=torch.float32)


def rotate_vector(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply rotation matrix R to 3D vector(s). v: (..., 3)."""
    return torch.einsum("ij,...j->...i", R, v)


def rotate_multivector_vector_part(mv: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate only the vector part (indices 1,2,3) of a multivector.
    mv: (..., 8), R: (3,3)
    """
    out = mv.clone()
    v = mv[..., 1:4]
    out[..., 1:4] = rotate_vector(v, R)
    return out


# --- Equivariant Clifford model (no final projection, vector output only) ---

class EquivariantCliffordModel(nn.Module):
    """
    Pure Clifford model with rotation-invariant weights.
    Outputs the vector part of the multivector. Guarantees O(3) equivariance.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.cl1 = EquivariantCliffordLinear(1, hidden_dim, bias=False)
        self.act1 = CliffordActivation()
        self.cl2 = EquivariantCliffordLinear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3) -> (B, 1, 8)
        mv = _to_multivector(x, N_BLADES_3D)
        mv = self.cl1(mv)
        mv = self.act1(mv)
        mv = self.cl2(mv)
        # Extract vector part (indices 1,2,3)
        return mv[..., 0, 1:4]


# --- Synthetic dataset: velocity -> force (simple linear for training) ---

def make_velocity_force_data(n: int = 500, seed: int = 42) -> tuple:
    """Generate (velocity, force) pairs. Force = 0.5 * velocity (equivariant)."""
    torch.manual_seed(seed)
    v = torch.randn(n, 3) * 2.0
    f = 0.5 * v
    return v, f


# --- Tests ---

def test_clifford_equivariance_minimal():
    """Minimal: single EquivariantCliffordLinear, no activation."""
    R = rotation_z_90()
    layer = EquivariantCliffordLinear(1, 1, bias=False)
    layer.eval()

    with torch.no_grad():
        x = torch.tensor([[1.0, 2.0, 3.0]])
        mv_x = _to_multivector(x, N_BLADES_3D)
        out = layer(mv_x)
        v_out = out[0, 0, 1:4]

        Rx = rotate_vector(x, R)
        mv_Rx = _to_multivector(Rx, N_BLADES_3D)
        out_R = layer(mv_Rx)
        v_out_R = out_R[0, 0, 1:4]

        Rv_out = rotate_vector(v_out.unsqueeze(0), R).squeeze(0)
        err = (v_out_R - Rv_out).abs().max().item()
        print(f"Minimal test: v_out={v_out}, v_out_R={v_out_R}, R(v_out)={Rv_out}, err={err}")
        assert err < 1e-5, f"Minimal equivariance failed: {err}"


def test_clifford_equivariance():
    """Equivariant Clifford model: Model(R(x)) ≈ R(Model(x)) within 1e-5."""
    R = rotation_z_90()
    model = EquivariantCliffordModel(hidden_dim=32)
    model.eval()

    # Equivariance holds for ANY weights (trained or not)
    with torch.no_grad():
        x = torch.tensor([[1.0, 2.0, 3.0]])
        y = model(x)
        Rx = rotate_vector(x, R)
        y_rotated_input = model(Rx)
        Ry = rotate_vector(y, R)

        err = (y_rotated_input - Ry).abs().max().item()
        assert err < 1e-5, f"Clifford equivariance failed: max error = {err}"
    print(f"[PASS] Clifford equivariance (untrained): max error = {err:.2e}")

    # Train on equivariant task and verify equivariance still holds
    v, f = make_velocity_force_data(200)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(150):
        pred = model(v)
        loss = ((pred - f) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        x = torch.tensor([[1.0, 2.0, 3.0]])
        y = model(x)
        Rx = rotate_vector(x, R)
        y_rotated_input = model(Rx)
        Ry = rotate_vector(y, R)
        err2 = (y_rotated_input - Ry).abs().max().item()
        assert err2 < 1e-5, f"Clifford equivariance failed after training: max error = {err2}"
    print(f"[PASS] Clifford equivariance (trained): max error = {err2:.2e}")


def test_mlp_fails_equivariance():
    """Standard MLP: Model(R(x)) != R(Model(x)) - large error expected."""
    R = rotation_z_90()
    mlp = nn.Sequential(
        nn.Linear(3, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )
    torch.nn.init.xavier_uniform_(mlp[0].weight)
    torch.nn.init.xavier_uniform_(mlp[2].weight)
    torch.nn.init.xavier_uniform_(mlp[4].weight)

    # Train on same task
    v, f = make_velocity_force_data(200)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    for _ in range(100):
        pred = mlp(v)
        loss = ((pred - f) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        x = torch.tensor([[1.0, 2.0, 3.0]])
        y = mlp(x)
        Rx = rotate_vector(x, R)
        y_rotated_input = mlp(Rx)
        Ry = rotate_vector(y, R)

        err = (y_rotated_input - Ry).abs().max().item()
        # MLP is not equivariant by design; error should be >> Clifford's (~0)
        assert err > 1e-4, f"MLP should fail equivariance (err={err})"
    print(f"[PASS] MLP fails equivariance as expected: max error = {err:.2e}")


def test_rcln_clifford_soft_shell():
    """RCLN with CustomCliffordSoftShell runs and produces reasonable output."""
    layer = RCLNLayer(
        input_dim=3,
        hidden_dim=16,
        output_dim=3,
        hard_core_func=None,
        lambda_res=1.0,
        use_clifford=True,
        use_clifford_custom=True,
    )
    x = torch.randn(4, 3)
    y = layer(x)
    assert y.shape == (4, 3)
    print("[PASS] RCLN Clifford soft shell forward")


def main():
    print("=" * 60)
    print("O(3) Equivariance Test - Clifford Geometric Algebra")
    print("=" * 60)
    test_clifford_equivariance_minimal()
    test_clifford_equivariance()
    test_mlp_fails_equivariance()
    test_rcln_clifford_soft_shell()
    print("=" * 60)
    print("All equivariance tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
