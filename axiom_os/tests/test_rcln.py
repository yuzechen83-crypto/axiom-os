"""
Unit tests for RCLN Layer (Residual Coupler Linking Neuron).
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.core.upi import UPIState, Units
from axiom_os.layers.rcln import RCLNLayer, HAS_CLIFFORD


def test_rcln_no_hard_core():
    """Test RCLN with no hard core (pure soft shell)."""
    rcln = RCLNLayer(input_dim=4, hidden_dim=64, output_dim=2, lambda_res=1.0)
    x = UPIState(values=torch.tensor([1.0, 2.0, 0.5, 0.1]), units=Units.UNITLESS, semantics="")
    y = rcln(x)
    assert y.shape == (1, 2)
    assert y.dtype == torch.float32


def test_rcln_with_hard_core():
    """Test RCLN with hard core physics function."""
    def hard_core(x):
        # Simple identity: return first two values as output
        v = x.values if hasattr(x, "values") else x
        v = torch.as_tensor(v, dtype=torch.float32)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return v[:, :2]  # (batch, 2)

    rcln = RCLNLayer(
        input_dim=4, hidden_dim=64, output_dim=2,
        hard_core_func=hard_core,
        lambda_res=0.5,
    )
    x = UPIState(values=torch.tensor([1.0, 2.0, 0.5, 0.1]), units=Units.UNITLESS, semantics="")
    y = rcln(x)
    assert y.shape == (1, 2)
    # y = y_hard + 0.5 * y_soft; y_hard = [1, 2]
    assert y[0, 0].item() != 0.0


def test_get_soft_activity():
    """Test get_soft_activity returns mean magnitude of y_soft."""
    rcln = RCLNLayer(input_dim=4, hidden_dim=64, output_dim=2, lambda_res=1.0)
    assert rcln.get_soft_activity() == 0.0  # Before forward

    x = UPIState(values=torch.tensor([1.0, 2.0, 0.5, 0.1]), units=Units.UNITLESS, semantics="")
    _ = rcln(x)
    activity = rcln.get_soft_activity()
    assert activity >= 0.0
    assert isinstance(activity, float)


def test_rcln_accepts_tensor():
    """Test RCLN accepts raw tensor (not just UPIState)."""
    rcln = RCLNLayer(input_dim=4, hidden_dim=64, output_dim=2, lambda_res=1.0)
    x = torch.tensor([[1.0, 2.0, 0.5, 0.1]])
    y = rcln(x)
    assert y.shape == (1, 2)


def test_rcln_force_mlp_fallback():
    """Test use_spectral=False, use_clifford=False forces MLP."""
    rcln = RCLNLayer(
        input_dim=4, hidden_dim=64, output_dim=2,
        lambda_res=1.0, use_spectral=False, use_clifford=False,
    )
    assert not rcln._use_spectral
    assert not rcln._use_clifford
    x = torch.tensor([[1.0, 2.0, 0.5, 0.1]])
    y = rcln(x)
    assert y.shape == (1, 2)


def test_rcln_clifford_multivector_structure():
    """Test Clifford soft shell accepts [s,v,B] multivector layout (8 blades)."""
    rcln = RCLNLayer(
        input_dim=8, hidden_dim=16, output_dim=4,
        lambda_res=1.0, use_spectral=False, use_clifford=True,
    )
    if not rcln._use_clifford:
        return  # Skip if Clifford not available
    # Input as full 8-blade multivector [s, v1, v2, v3, B12, B13, B23, T]
    x = torch.tensor([[1.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.001]])
    y = rcln(x)
    assert y.shape == (1, 4)


def test_rcln_spectral_soft_shell():
    """Test Spectral (FFT) soft shell: FFT → weights → IFFT."""
    rcln = RCLNLayer(
        input_dim=8, hidden_dim=32, output_dim=4,
        lambda_res=1.0, use_spectral=True,
    )
    assert rcln._use_spectral
    x = torch.tensor([[1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    y = rcln(x)
    assert y.shape == (1, 4)
    # Gradient flow
    loss = y.sum()
    loss.backward()


if __name__ == "__main__":
    test_rcln_no_hard_core()
    test_rcln_with_hard_core()
    test_get_soft_activity()
    test_rcln_accepts_tensor()
    test_rcln_force_mlp_fallback()
    test_rcln_clifford_multivector_structure()
    test_rcln_spectral_soft_shell()
    print("All RCLN tests passed.")
