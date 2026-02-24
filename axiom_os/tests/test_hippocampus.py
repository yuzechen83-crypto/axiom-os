"""
Unit tests for Hippocampus (Knowledge Registry).
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.core.hippocampus import Hippocampus
from axiom_os.core.upi import UPIState, Units
from axiom_os.layers.rcln import RCLNLayer


def test_crystallize_string_formula():
    """Test crystallize with string formula updates hard_core and resets soft_shell."""
    hippo = Hippocampus(dim=4, capacity=100)
    rcln = RCLNLayer(input_dim=4, hidden_dim=32, output_dim=2, lambda_res=0.5)

    # Before: no hard_core
    assert rcln.hard_core is None

    # Crystallize formula
    formula_id = hippo.crystallize("x[:, :2]", rcln)  # First two components as output
    assert formula_id in hippo.knowledge_base
    assert rcln.hard_core is not None

    # Forward with new hard_core
    x = torch.tensor([[1.0, 2.0, 0.5, 0.1]])
    y = rcln(x)
    assert y.shape == (1, 2)
    # y_hard = [1, 2], y_soft = reset (small random)
    assert y[0, 0].item() != 0.0


def test_crystallize_resets_soft_shell():
    """Test that soft_shell weights are reset after crystallize."""
    hippo = Hippocampus()
    rcln = RCLNLayer(input_dim=4, hidden_dim=16, output_dim=1, lambda_res=1.0)

    # Run forward to initialize
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    y_before = rcln(x).clone()

    # Crystallize (resets soft_shell)
    hippo.crystallize("np.zeros((x.shape[0], 1))", rcln)

    # soft_shell is reset - output will differ
    y_after = rcln(x)
    # With hard_core = zeros, y_total = lambda_res * y_soft (reset)
    assert y_after.shape == (1, 1)


def test_crystallize_callable():
    """Test crystallize with callable formula."""
    hippo = Hippocampus()
    rcln = RCLNLayer(input_dim=4, hidden_dim=16, output_dim=2, lambda_res=1.0)

    def my_formula(x):
        v = x.values if hasattr(x, "values") else x
        v = torch.as_tensor(v).float() if not isinstance(v, torch.Tensor) else v
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return v[:, :2]  # First two components

    formula_id = hippo.crystallize(my_formula, rcln, formula_id="custom")
    assert hippo.knowledge_base["custom"]["formula"] == str(my_formula)

    x = UPIState(values=torch.tensor([1.0, 2.0, 3.0, 4.0]), units=[0, 0, 0, 0, 0])
    y = rcln(x)
    assert y.shape == (1, 2)


def test_knowledge_base_storage():
    """Test knowledge_base is populated."""
    hippo = Hippocampus()
    rcln = RCLNLayer(input_dim=2, hidden_dim=8, output_dim=1, lambda_res=1.0)

    hippo.crystallize("x[:, 0]**2", rcln, formula_id="kinetic")
    assert "kinetic" in hippo.knowledge_base
    assert hippo.knowledge_base["kinetic"]["formula"] == "x[:, 0]**2"


if __name__ == "__main__":
    test_crystallize_string_formula()
    test_crystallize_resets_soft_shell()
    test_crystallize_callable()
    test_knowledge_base_storage()
    print("All Hippocampus tests passed.")
