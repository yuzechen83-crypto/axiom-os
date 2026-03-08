"""
Unit tests for Universal Physical Interface (UPI) protocol.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]  # Project root (parent of axiom_os)
sys.path.insert(0, str(ROOT))

from axiom_os.core.upi import UPIState, Units


def test_upi_state_creation():
    """Test UPIState creation with values and units."""
    state = UPIState(
        values=torch.tensor([1.0, 2.0, 3.0]),
        units=[0, 1, 0, 0, 0],  # LENGTH
        semantics="Position",
    )
    assert state.values.dtype == torch.float64
    assert state.values.shape == (3,)
    assert state.units.shape == (5,)
    assert state.semantics == "Position"


def test_upi_state_with_spacetime():
    """Test UPIState with spacetime coordinates."""
    state = UPIState(
        values=torch.tensor([1.0]),
        units=Units.VELOCITY,
        spacetime=[1.0, 0.0, 0.0, 0.0],  # t, x, y, z
        semantics="Velocity",
    )
    assert state.spacetime.shape == (4,)
    assert float(state.spacetime[0]) == 1.0


def test_add_same_units():
    """Test adding quantities with same units."""
    a = UPIState(values=torch.tensor([1.0]), units=Units.MASS, semantics="Mass")
    b = UPIState(values=torch.tensor([2.0]), units=Units.MASS, semantics="Mass")
    c = a + b
    assert float(c.values) == 3.0


def test_add_different_units_raises():
    """Test that adding different units raises ValueError."""
    mass = UPIState(values=torch.tensor([1.0]), units=Units.MASS, semantics="Mass")
    time_val = UPIState(values=torch.tensor([1.0]), units=Units.TIME, semantics="Time")
    try:
        _ = mass + time_val
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unit mismatch" in str(e) or "different" in str(e).lower()


def test_mul_div_units():
    """Test multiplication and division unit propagation."""
    mass = UPIState(values=torch.tensor([2.0]), units=Units.MASS, semantics="Mass")
    length = UPIState(values=torch.tensor([3.0]), units=Units.LENGTH, semantics="Length")
    prod = mass * length
    assert prod.units.tolist() == [1, 1, 0, 0, 0]
    quot = prod / mass
    assert quot.units.tolist() == [0, 1, 0, 0, 0]


def test_causality():
    """Test assert_causality within and outside light cone."""
    here = UPIState(
        values=torch.tensor([0.0]),
        units=Units.UNITLESS,
        spacetime=[0, 0, 0, 0],
    )
    future_near = UPIState(
        values=torch.tensor([0.0]),
        units=Units.UNITLESS,
        spacetime=[1, 0.5, 0, 0],  # Within light cone
    )
    here.assert_causality(future_near)

    future_far = UPIState(
        values=torch.tensor([0.0]),
        units=Units.UNITLESS,
        spacetime=[0.5, 2, 0, 0],  # Outside light cone
    )
    try:
        here.assert_causality(future_far)
        assert False, "Should have raised causality violation"
    except ValueError:
        pass


if __name__ == "__main__":
    test_upi_state_creation()
    test_upi_state_with_spacetime()
    test_add_same_units()
    test_add_different_units_raises()
    test_mul_div_units()
    test_causality()
    print("All UPI tests passed.")
