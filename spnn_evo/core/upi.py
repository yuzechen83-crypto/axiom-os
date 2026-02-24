"""
UPI - Universal Physical Interface (The Language)
科学巴别塔通票：使流体、电磁、量子力学能够相互对话
Standard data packet for cross-disciplinary interaction.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math

# Speed of light [m/s] - Causal anchor
C_LIGHT = 299792458.0


@dataclass
class Units:
    """
    Dimensional powers [M, L, T, Q, Θ]
    Mass, Length, Time, Charge, Temperature
    e.g., Velocity = [0, 1, -1, 0, 0]
    """
    M: int = 0
    L: int = 0
    T: int = 0
    Q: int = 0
    Theta: int = 0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.M, self.L, self.T, self.Q, self.Theta], dtype=torch.int64)

    def __eq__(self, other: "Units") -> bool:
        if not isinstance(other, Units):
            return False
        return self.M == other.M and self.L == other.L and self.T == other.T and self.Q == other.Q and self.Theta == other.Theta

    def __add__(self, other: "Units") -> "Units":
        """Addition requires same units"""
        if not self.__eq__(other):
            raise ValueError(f"Unit mismatch: {self} vs {other}")
        return Units(self.M, self.L, self.T, self.Q, self.Theta)

    def __mul__(self, other: "Units") -> "Units":
        """Multiplication: powers add"""
        return Units(
            self.M + other.M,
            self.L + other.L,
            self.T + other.T,
            self.Q + other.Q,
            self.Theta + other.Theta,
        )

    def __str__(self) -> str:
        return f"[M^{self.M} L^{self.L} T^{self.T} Q^{self.Q} Θ^{self.Theta}]"


# Common units
UNITLESS = Units(0, 0, 0, 0, 0)
VELOCITY = Units(0, 1, -1, 0, 0)
LENGTH = Units(0, 1, 0, 0, 0)
TIME = Units(0, 0, 1, 0, 0)
MASS = Units(1, 0, 0, 0, 0)
FORCE = Units(1, 1, -2, 0, 0)
PRESSURE = Units(1, -1, -2, 0, 0)
DENSITY = Units(1, -3, 0, 0, 0)
CHARGE = Units(0, 0, 0, 1, 0)
VOLTAGE = Units(1, 2, -2, -1, 0)
MAGNETIC_FIELD = Units(1, 0, -1, -1, 0)


class UPIState:
    """
    Universal Physical Interface State
    The standard data packet for cross-disciplinary interaction.
    Supports scalars and vectors (rank-1 tensors in physics sense).

    Vector: shape [..., n_components] with n_components in {2, 3}.
      - (N, 2) or (H, W, 2): 2D velocity u=(u,v)
      - (2,) or (3,): single 2D/3D vector (rank-1 tensor)
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        units: Union[Units, torch.Tensor, Tuple[int, int, int, int, int]],
        spacetime: Optional[torch.Tensor] = None,
        semantics: str = "",
        is_vector: bool = False,
    ):
        self.tensor = tensor if tensor.dtype == torch.float64 else tensor.double()
        if isinstance(units, Units):
            self.units = units.to_tensor()
        elif isinstance(units, (tuple, list)):
            self.units = torch.tensor(units, dtype=torch.int64)
        else:
            self.units = torch.as_tensor(units, dtype=torch.int64)
        self.spacetime = spacetime if spacetime is not None else torch.zeros(4, dtype=torch.float64)
        self.semantics = semantics
        self.is_vector = is_vector

    def verify_coordinate_consistency(self) -> bool:
        """
        Verify vector fields have consistent coordinate layout.
        For u=(u,v): last dim must be 2. For 3D u=(u,v,w): last dim 3.
        Accepts rank-1 tensors: (2,) or (3,) for single vectors.
        """
        if not self.is_vector:
            return True
        n_comp = self.tensor.shape[-1]
        # Allow dim>=1: (2,), (3,), (N,2), (T,H,W,2), etc.
        return n_comp in (2, 3) and self.tensor.dim() >= 1

    @classmethod
    def from_vector_field(
        cls,
        u: torch.Tensor,
        v: torch.Tensor,
        units: Union[Units, torch.Tensor, Tuple[int, int, int, int, int]],
        spacetime: Optional[torch.Tensor] = None,
        semantics: str = "velocity_2d",
    ) -> "UPIState":
        """Create UPIState from 2D velocity components u, v. Stack as (..., 2)."""
        uv = torch.stack([u, v], dim=-1)
        return cls(uv, units, spacetime, semantics, is_vector=True)

    @classmethod
    def from_single_vector(
        cls,
        components: torch.Tensor,
        units: Union[Units, torch.Tensor, Tuple[int, int, int, int, int]],
        spacetime: Optional[torch.Tensor] = None,
        semantics: str = "vector",
    ) -> "UPIState":
        """Create UPIState from rank-1 vector (2,) or (3,). Ensures coordinate consistency."""
        vec = components.double() if components.dtype != torch.float64 else components
        if vec.dim() == 0:
            vec = vec.unsqueeze(0)
        state = cls(vec, units, spacetime, semantics, is_vector=True)
        if not state.verify_coordinate_consistency():
            raise ValueError(f"Vector must have 2 or 3 components, got shape {vec.shape}")
        return state

    def _check_units(self, other: "UPIState") -> bool:
        return torch.equal(self.units, other.units)

    def __add__(self, other: "UPIState") -> "UPIState":
        """Only allow if units match. Preserve is_vector for vector + vector."""
        if not self._check_units(other):
            raise ValueError(f"Unit mismatch: {self.units} vs {other.units}")
        t = self.tensor + other.tensor
        is_vec = self.is_vector and getattr(other, "is_vector", False)
        return UPIState(
            t, Units(*self.units.tolist()), self.spacetime, self.semantics, is_vector=is_vec
        )

    def __sub__(self, other: "UPIState") -> "UPIState":
        """Only allow if units match (dimensional consistency)."""
        if not self._check_units(other):
            raise ValueError(f"Unit mismatch: {self.units} vs {other.units}")
        t = self.tensor - other.tensor
        is_vec = self.is_vector and getattr(other, "is_vector", False)
        return UPIState(
            t, Units(*self.units.tolist()), self.spacetime, self.semantics, is_vector=is_vec
        )

    def __mul__(self, other: Union["UPIState", float, int]) -> "UPIState":
        """Update units accordingly (e.g., L * T^-1 = L T^-1). Preserve is_vector for scalar*vector."""
        if isinstance(other, (int, float)):
            return UPIState(
                self.tensor * other,
                Units(*self.units.tolist()),
                self.spacetime,
                self.semantics,
                is_vector=self.is_vector,
            )
        u_self = Units(*self.units.tolist())
        u_other = Units(*other.units.tolist())
        u_new = u_self * u_other
        is_vec = self.is_vector and getattr(other, "is_vector", False)
        return UPIState(
            self.tensor * other.tensor,
            u_new,
            self.spacetime,
            f"{self.semantics}*{other.semantics}",
            is_vector=is_vec,
        )

    def __radd__(self, other: "UPIState") -> "UPIState":
        return self.__add__(other)

    def __rsub__(self, other: "UPIState") -> "UPIState":
        return (other - self)

    def assert_causality(self, other: Optional["UPIState"] = None) -> bool:
        """
        Ensure interaction is within the light cone.
        |Δx| ≤ c |Δt| for causal connectability.
        If other is None, checks self.spacetime is temporally ordered (t >= 0).
        """
        if other is None:
            t = float(self.spacetime[0])
            if t < 0:
                raise ValueError(f"Causality violation: t={t} < 0")
            return True
        dt = float(other.spacetime[0] - self.spacetime[0])
        dx = float(torch.norm(other.spacetime[1:] - self.spacetime[1:]))
        if abs(dx) > C_LIGHT * abs(dt) + 1e-6:
            raise ValueError(f"Light-cone violation: |Δx|={dx:.2e} > c|Δt|={C_LIGHT * abs(dt):.2e}")
        return True

    def to_device(self, device: torch.device) -> "UPIState":
        return UPIState(
            self.tensor.to(device),
            self.units.to(device),
            self.spacetime.to(device),
            self.semantics,
        )

    def __repr__(self) -> str:
        return f"UPIState({self.semantics}, units={self.units.tolist()}, shape={self.tensor.shape})"
