"""
Universal Physical Interface (UPI) - Protocol
Units-aware data exchange between physical systems and AI.
Prevents physical errors (e.g., adding Mass to Time).

Units: [M, L, T, Q, Θ] = Mass, Length, Time, Charge, Temperature (SI base dimensions).
"""

from typing import Optional, Union
import torch


class DimensionError(ValueError):
    """Raised when physical dimensions are incompatible (e.g., Mass + Time)."""
    pass


# Convenience: 5D unit vectors [M, L, T, Q, Θ]
class Units:
    """Common unit vectors for [M, L, T, Q, Θ]."""
    MASS = [1, 0, 0, 0, 0]
    LENGTH = [0, 1, 0, 0, 0]
    TIME = [0, 0, 1, 0, 0]
    CHARGE = [0, 0, 0, 1, 0]
    TEMPERATURE = [0, 0, 0, 0, 1]
    VELOCITY = [0, 1, -1, 0, 0]   # L/T
    FORCE = [1, 1, -2, 0, 0]      # ML/T²
    UNITLESS = [0, 0, 0, 0, 0]


class UPIState:
    """
    State vector with physical units metadata.
    Units: [M, L, T, Q, Θ] = Mass, Length, Time, Charge, Temperature (SI base dimensions).
    """

    def __init__(
        self,
        values: Union[torch.Tensor, list, tuple],
        units: Union[torch.Tensor, list, tuple],
        spacetime: Optional[Union[torch.Tensor, list, tuple]] = None,
        semantics: str = "",
    ):
        self.values = torch.as_tensor(values, dtype=torch.float64)
        self.units = torch.as_tensor(units, dtype=torch.long)
        if self.units.dim() == 0:
            self.units = self.units.unsqueeze(0)
        if self.units.numel() != 5:
            raise ValueError(f"units must be length 5 [M,L,T,Q,Θ], got {self.units.numel()}")
        self.units = self.units.reshape(5)

        self.spacetime = torch.as_tensor(
            spacetime if spacetime is not None else [0.0, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        ).reshape(4)
        self.semantics = str(semantics)

    def _check_units_match(self, other: "UPIState") -> None:
        if not torch.equal(self.units, other.units):
            raise DimensionError(
                f"Unit mismatch: {self.units.tolist()} vs {other.units.tolist()}. "
                "Cannot add/subtract quantities with different dimensions."
            )

    def __add__(self, other: "UPIState") -> "UPIState":
        self._check_units_match(other)
        return UPIState(
            values=self.values + other.values,
            units=self.units.clone(),
            spacetime=self.spacetime.clone(),
            semantics=self.semantics or other.semantics,
        )

    def __sub__(self, other: "UPIState") -> "UPIState":
        self._check_units_match(other)
        return UPIState(
            values=self.values - other.values,
            units=self.units.clone(),
            spacetime=self.spacetime.clone(),
            semantics=self.semantics or other.semantics,
        )

    def __mul__(self, other: "UPIState") -> "UPIState":
        # Result units = self.units + other.units (dimensional analysis)
        new_units = self.units + other.units
        v = self.values * other.values
        return UPIState(
            values=v,
            units=new_units,
            spacetime=self.spacetime.clone(),
            semantics=f"{self.semantics}*{other.semantics}".strip("*"),
        )

    def __truediv__(self, other: "UPIState") -> "UPIState":
        # Result units = self.units - other.units
        new_units = self.units - other.units
        v = self.values / other.values
        return UPIState(
            values=v,
            units=new_units,
            spacetime=self.spacetime.clone(),
            semantics=f"{self.semantics}/{other.semantics}".strip("/"),
        )

    def assert_causality(self, other: "UPIState", c: float = 1.0) -> None:
        """
        Check if the interaction is within the light cone (c=1 by default).
        Event other is causally reachable from self if:
        (Δt)² >= |Δr|² / c²  =>  c·Δt >= |Δr|
        """
        dt = float(other.spacetime[0] - self.spacetime[0])
        dx = float(other.spacetime[1] - self.spacetime[1])
        dy = float(other.spacetime[2] - self.spacetime[2])
        dz = float(other.spacetime[3] - self.spacetime[3])
        spatial_dist_sq = dx * dx + dy * dy + dz * dz
        time_sep = c * dt
        if time_sep < 0:
            raise ValueError(
                f"Causality violated: other is in the past (Δt={dt}, c·Δt={time_sep})"
            )
        if time_sep * time_sep < spatial_dist_sq:
            raise ValueError(
                f"Causality violated: events outside light cone "
                f"(c·Δt={time_sep:.6f}, |Δr|={spatial_dist_sq**0.5:.6f})"
            )

    def __repr__(self) -> str:
        return f"UPIState(values={self.values}, units={self.units.tolist()}, semantics={self.semantics!r})"


if __name__ == "__main__":
    # Test: adding different units raises ValueError
    print("Testing UPIState unit mismatch (add)...")
    mass = UPIState(
        values=torch.tensor([1.0]),
        units=[1, 0, 0, 0, 0],  # [M, L, T, Q, Θ]
        semantics="Mass",
    )
    time_val = UPIState(
        values=torch.tensor([1.0]),
        units=[0, 0, 1, 0, 0],  # Time
        semantics="Time",
    )
    try:
        _ = mass + time_val
        print("FAIL: Should have raised ValueError")
        exit(1)
    except ValueError as e:
        print(f"OK: Caught expected error: {e}")

    # Test: same units can be added
    mass2 = UPIState(values=torch.tensor([2.0]), units=[1, 0, 0, 0, 0], semantics="Mass")
    result = mass + mass2
    assert float(result.values) == 3.0
    print("OK: Same units add correctly")

    # Test: mul/div units
    length = UPIState(values=torch.tensor([3.0]), units=[0, 1, 0, 0, 0], semantics="Length")
    prod = mass * length
    assert prod.units.tolist() == [1, 1, 0, 0, 0]  # M*L
    print("OK: Multiplication adds units")

    quot = prod / mass
    assert quot.units.tolist() == [0, 1, 0, 0, 0]
    print("OK: Division subtracts units")

    # Test: assert_causality
    here = UPIState(values=torch.tensor([0.0]), units=[0, 0, 0, 0, 0], spacetime=[0, 0, 0, 0])
    future_near = UPIState(values=torch.tensor([0.0]), units=[0, 0, 0, 0, 0], spacetime=[1, 0.5, 0, 0])
    here.assert_causality(future_near)
    print("OK: Causality holds (within light cone)")

    try:
        future_far = UPIState(values=torch.tensor([0.0]), units=[0, 0, 0, 0, 0], spacetime=[0.5, 2, 0, 0])
        here.assert_causality(future_far)
        print("FAIL: Should have raised causality violation")
        exit(1)
    except ValueError as e:
        print(f"OK: Causality violation caught: {e}")

    print("\nAll UPIState tests passed.")
