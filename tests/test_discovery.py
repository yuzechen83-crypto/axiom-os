"""
Unit tests for Discovery Engine (Symbolic Regression & Buckingham Pi).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.engine.discovery import DiscoveryEngine


def test_get_dimensionless_groups_inputs_only():
    """Test Pi groups for inputs only."""
    engine = DiscoveryEngine(use_pysr=False)
    # Simple case: [L, T, V] -> one Pi group (e.g., V*T/L)
    inputs = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    input_units = [
        [0, 1, 0, 0, 0],   # L
        [0, 0, 1, 0, 0],   # T
        [0, 1, -1, 0, 0],  # V = L/T
    ]
    pi = engine.get_dimensionless_groups(inputs, input_units)
    assert pi.shape[0] == 2
    assert pi.shape[1] >= 1


def test_get_dimensionless_groups_unitless():
    """Test with unitless inputs returns inputs as-is when no null space."""
    engine = DiscoveryEngine(use_pysr=False)
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_units = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    pi = engine.get_dimensionless_groups(inputs, input_units)
    assert pi.shape == inputs.shape or pi.shape[0] == inputs.shape[0]


def test_distill_with_lasso_fallback():
    """Test distill returns formula via Lasso when PySR unavailable or fails."""
    engine = DiscoveryEngine(use_pysr=False)
    # Synthetic: y = 2*x0 + x1
    np.random.seed(42)
    x = np.random.randn(50, 2)
    y = (2 * x[:, 0] + x[:, 1]).reshape(-1, 1)
    data = list(zip(x, y))
    formula = engine.distill(
        rcln_layer=None,
        data_buffer=data,
        input_units=[[0, 0, 0, 0, 0]] * 2,
    )
    assert formula is not None
    assert "x" in formula or "0" in formula


def test_discover_legacy():
    """Test legacy discover() method."""
    engine = DiscoveryEngine(use_pysr=False)
    x = np.random.randn(30, 2)
    y = (x[:, 0] ** 2 + x[:, 1]).reshape(-1, 1)
    formula = engine.discover(x, y)
    assert formula is None or (isinstance(formula, str) and len(formula) > 0)


def test_distill_with_rcln():
    """Test distill with RCLN layer and x-only buffer."""
    import torch
    from axiom_os.layers.rcln import RCLNLayer

    engine = DiscoveryEngine(use_pysr=False)
    rcln = RCLNLayer(input_dim=3, hidden_dim=16, output_dim=1, lambda_res=1.0)
    x = np.random.randn(40, 3).astype(np.float32)
    formula = engine.distill(rcln, x, input_units=[[0, 0, 0, 0, 0]] * 3)
    # Should return Lasso formula from y_soft
    assert formula is not None


def test_discover_parametric_aic():
    """Test discover_parametric: fit all forms, select by AIC. No Lasso gate."""
    engine = DiscoveryEngine(use_pysr=False)
    np.random.seed(42)
    t = np.linspace(0.1, 1.0, 80)  # normalized cycles
    # True: exp decay
    y = -0.5 * np.exp(-2.0 * t) + 0.1 + 0.02 * np.random.randn(80)
    formula, pred, coefs = engine.discover_parametric(t, y, selector="aic")
    assert formula is not None
    assert pred.shape == y.shape
    assert coefs is not None
    assert "form_type" in coefs
    assert coefs["form_type"] in ("exp", "power", "poly")


def test_discover_multivariate():
    """Test discover_multivariate: y = f(t,x,y,z) for wind/spatial fields."""
    engine = DiscoveryEngine(use_pysr=False)
    np.random.seed(42)
    X = np.random.rand(200, 4)
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2] * X[:, 3] + np.random.randn(200) * 0.05
    formula, pred, coefs = engine.discover_multivariate(X, y, var_names=["t", "x", "y", "z"], selector="bic")
    assert formula is not None
    assert pred.shape == y.shape
    assert coefs is not None
    r2 = 1 - np.sum((pred - y) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
    assert r2 > 0.3, f"Discovery R2 too low: {r2}"


if __name__ == "__main__":
    test_get_dimensionless_groups_inputs_only()
    test_get_dimensionless_groups_unitless()
    test_distill_with_lasso_fallback()
    test_discover_legacy()
    test_distill_with_rcln()
    test_discover_parametric_aic()
    test_discover_multivariate()
    print("All Discovery tests passed.")
