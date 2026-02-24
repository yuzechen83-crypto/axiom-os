"""
Clifford Geometric Algebra Engine - Cl(3,0)
Lightweight, differentiable Geometric Product in PyTorch.

Basis (8 components): 1, e1, e2, e3, e12, e23, e31, e123
Index:                0  1   2   3   4    5    6    7
  - Scalar: 0
  - Vector: 1,2,3
  - Bivector: 4,5,6 (e12, e23, e31)
  - Trivector: 7 (e123)

Geometric Product: ab = a·b + a∧b (inner + outer)
"""

from typing import Tuple
import torch

# Cl(3,0) Cayley table: C[i,j,k] = coefficient of basis k in product of basis i * basis j
# Result: (i * j) = sum_k C[i,j,k] * e_k
# Stored as sparse: for each (i,j), we have (coef, k). Dense: (8,8,8) tensor
N_BLADES = 8
BLADE_NAMES = ["1", "e1", "e2", "e3", "e12", "e23", "e31", "e123"]


def _build_cayley_table() -> torch.Tensor:
    """
    Build Cayley table for Cl(3,0).
    cayley[i, j, k] = coefficient of blade k in (blade_i * blade_j)
    """
    # e_i^2 = 1, e_i e_j = -e_j e_i
    # e12 = e1e2, e23 = e2e3, e31 = e3e1
    # e123 = e1e2e3
    # e12^2 = e23^2 = e31^2 = -1
    # e123^2 = -1

    cayley = torch.zeros(8, 8, 8)

    def set_prod(i: int, j: int, k: int, coef: float) -> None:
        cayley[i, j, k] = coef

    # 1 * x = x
    for j in range(8):
        set_prod(0, j, j, 1.0)
    for i in range(8):
        set_prod(i, 0, i, 1.0)

    # e1, e2, e3 (indices 1,2,3)
    set_prod(1, 1, 0, 1.0)   # e1*e1 = 1
    set_prod(2, 2, 0, 1.0)   # e2*e2 = 1
    set_prod(3, 3, 0, 1.0)   # e3*e3 = 1

    set_prod(1, 2, 4, 1.0)   # e1*e2 = e12
    set_prod(2, 1, 4, -1.0)  # e2*e1 = -e12
    set_prod(1, 3, 6, -1.0)  # e1*e3 = -e31 (e13 = -e31)
    set_prod(3, 1, 6, 1.0)   # e3*e1 = e31
    set_prod(2, 3, 5, 1.0)   # e2*e3 = e23
    set_prod(3, 2, 5, -1.0)  # e3*e2 = -e23

    # e_i * e_jk
    set_prod(1, 4, 2, 1.0)   # e1*e12 = e2
    set_prod(4, 1, 2, -1.0)  # e12*e1 = -e2
    set_prod(1, 5, 7, 1.0)   # e1*e23 = e123
    set_prod(5, 1, 7, -1.0)  # e23*e1 = -e123
    set_prod(1, 6, 3, -1.0)  # e1*e31 = -e3
    set_prod(6, 1, 3, 1.0)   # e31*e1 = e3

    set_prod(2, 4, 1, -1.0)  # e2*e12 = -e1
    set_prod(4, 2, 1, 1.0)   # e12*e2 = e1
    set_prod(2, 5, 3, 1.0)   # e2*e23 = e3
    set_prod(5, 2, 3, -1.0)  # e23*e2 = -e3
    set_prod(2, 6, 7, -1.0)  # e2*e31 = -e123
    set_prod(6, 2, 7, 1.0)   # e31*e2 = e123

    set_prod(3, 4, 7, 1.0)   # e3*e12 = e123
    set_prod(4, 3, 7, -1.0)  # e12*e3 = -e123
    set_prod(3, 5, 2, -1.0)  # e3*e23 = -e2
    set_prod(5, 3, 2, 1.0)   # e23*e3 = e2
    set_prod(3, 6, 1, 1.0)   # e3*e31 = e1
    set_prod(6, 3, 1, -1.0)  # e31*e3 = -e1

    # e_ij * e_kl
    set_prod(4, 4, 0, -1.0)  # e12*e12 = -1
    set_prod(5, 5, 0, -1.0)  # e23*e23 = -1
    set_prod(6, 6, 0, -1.0)  # e31*e31 = -1

    set_prod(4, 5, 6, -1.0)  # e12*e23 = -e31
    set_prod(5, 4, 6, 1.0)   # e23*e12 = e31
    set_prod(4, 6, 5, 1.0)   # e12*e31 = e23
    set_prod(6, 4, 5, -1.0)  # e31*e12 = -e23
    set_prod(5, 6, 4, -1.0)  # e23*e31 = -e12
    set_prod(6, 5, 4, 1.0)   # e31*e23 = e12

    # e_ij * e123 and e123 * e_ij
    set_prod(4, 7, 3, 1.0)   # e12*e123 = e3
    set_prod(7, 4, 3, 1.0)   # e123*e12 = e3
    set_prod(5, 7, 1, 1.0)   # e23*e123 = e1
    set_prod(7, 5, 1, 1.0)   # e123*e23 = e1
    set_prod(6, 7, 2, 1.0)   # e31*e123 = e2
    set_prod(7, 6, 2, 1.0)   # e123*e31 = e2

    set_prod(7, 7, 0, -1.0)  # e123*e123 = -1

    return cayley


CAYLEY = _build_cayley_table()


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Geometric product of two multivectors.
    a, b: shape (..., 8)
    Returns: shape (..., 8)
    Logic: c_k = sum_ij cayley[i,j,k] * a_i * b_j
    """
    # c[...,k] = sum_i sum_j cayley[i,j,k] * a[...,i] * b[...,j]
    return torch.einsum("ijk,...i,...j->...k", CAYLEY.to(a.device), a, b)


def multivector_magnitude(x: torch.Tensor) -> torch.Tensor:
    """||x|| = sqrt(sum of squared blade components)."""
    return (x ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()


def rotate_multivector(mv: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate a multivector by 3x3 rotation matrix R.
    - Scalar (0): invariant
    - Vector (1,2,3): R @ v
    - Bivector (4,5,6): e12, e23, e31 transform as (b12, b23, b31) -> (b12, b31, -b23) for R_z(90°)
      General: adjoint representation of SO(3) on bivectors
    - Trivector (7): invariant (det R = 1)
    """
    out = mv.clone()
    # Vector part
    out[..., 1:4] = torch.einsum("ij,...j->...i", R, mv[..., 1:4])
    # Bivector part: (e12, e23, e31) -> compute transformation matrix
    # e_i e_j -> R(e_i) R(e_j), so we need the 3x3 matrix for (b12,b23,b31)
    b = mv[..., 4:7]
    # Adjoint: for R in SO(3), bivector transforms as B' = R B R^T (matrix form)
    # Our basis order: e12(4), e23(5), e31(6) -> antisymmetric matrix [[0,b12,-b31],[-b12,0,b23],[b31,-b23,0]]
    # Simpler: use the fact that in 3D bivectors ~ vectors via Hodge: e12~e3, e23~e1, e31~e2
    # So (b12,b23,b31) corresponds to vector (b23,b31,b12). Rotate: R @ (b23,b31,b12)
    # Then (b12',b23',b31') = (rotated[2], rotated[0], rotated[1])
    v_dual = torch.stack([b[..., 1], b[..., 2], b[..., 0]], dim=-1)
    v_dual_rot = torch.einsum("ij,...j->...i", R, v_dual)
    out[..., 4:7] = torch.stack([v_dual_rot[..., 2], v_dual_rot[..., 0], v_dual_rot[..., 1]], dim=-1)
    return out


def vector_from_velocity(u: float, v: float, w: float) -> torch.Tensor:
    """Map velocity (u,v,w) to multivector: scalar=0, vector=(u,v,w), rest=0."""
    return torch.tensor([0.0, u, v, w, 0.0, 0.0, 0.0, 0.0])


class CliffordTensor:
    """
    Wrapper for multivector tensor (..., 8).
    Parts: scalar (0), vector (1-3), bivector (4-6), trivector (7).
    """

    def __init__(self, data: torch.Tensor):
        if data.shape[-1] != 8:
            raise ValueError(f"Last dim must be 8, got {data.shape[-1]}")
        self.data = data

    @property
    def scalar(self) -> torch.Tensor:
        return self.data[..., 0:1]

    @property
    def vector(self) -> torch.Tensor:
        return self.data[..., 1:4]

    @property
    def bivector(self) -> torch.Tensor:
        return self.data[..., 4:7]

    @property
    def trivector(self) -> torch.Tensor:
        return self.data[..., 7:8]

    def __mul__(self, other: "CliffordTensor") -> "CliffordTensor":
        return CliffordTensor(geometric_product(self.data, other.data))
