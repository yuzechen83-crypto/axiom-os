"""
边界流束算子 (Boundary Flux Operator)
U_BC = ∮_∂Ω γ(s)·||Flux_AI(s) - Flux_Axiom(s)||² ds
Dirichlet / Neumann / Robin 分类处理
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

FLUX_TOLERANCE = 1e-5


class BoundaryType(Enum):
    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"
    ROBIN = "Robin"


@dataclass
class BoundarySpec:
    name: str
    type: BoundaryType = BoundaryType.DIRICHLET
    alpha: float = 1.0
    value: Optional[float] = None
    flux: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[Callable] = None
    allowed_range: Optional[tuple] = None


@dataclass
class DirichletBC(BoundarySpec):
    type: BoundaryType = BoundaryType.DIRICHLET
    value: float = 0.0


@dataclass
class NeumannBC(BoundarySpec):
    type: BoundaryType = BoundaryType.NEUMANN
    flux: float = 0.0


@dataclass
class RobinBC(BoundarySpec):
    type: BoundaryType = BoundaryType.ROBIN
    a: float = 1.0
    b: float = 0.5
    c: float = 0.0


class BoundaryFluxOperator:
    """
    U_BC = U_Dirichlet + U_Neumann + U_Robin
    """

    def __init__(self, gamma_fn: Optional[Callable] = None):
        self.gamma_fn = gamma_fn or (lambda s: 1.0)

    def u_dirichlet(
        self,
        u_ai: torch.Tensor,
        u_prescribed: torch.Tensor,
        alpha: float = 1.0,
        gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """U_Dirichlet = Σ α_D · ||u_AI - u_prescribed||²"""
        diff = u_ai - u_prescribed
        w = gamma if gamma is not None else torch.ones_like(diff)
        return alpha * torch.mean(w * (diff ** 2))

    def u_neumann(
        self,
        grad_u_n: torch.Tensor,
        q_prescribed: torch.Tensor,
        alpha: float = 1.0,
        gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """U_Neumann = Σ α_N · ||∇u·n - q_prescribed||²"""
        diff = grad_u_n - q_prescribed
        w = gamma if gamma is not None else torch.ones_like(diff)
        return alpha * torch.mean(w * (diff ** 2))

    def u_robin(
        self,
        u_ai: torch.Tensor,
        grad_u_n: torch.Tensor,
        a: float,
        b: float,
        c: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """U_Robin = Σ α_R · ||a·u + b·∇u·n - c||²"""
        lhs = a * u_ai + b * grad_u_n
        diff = lhs - c
        return alpha * torch.mean(diff ** 2)

    def boundary_gate(
        self,
        u_bc: torch.Tensor,
        kappa_bc: float = 10.0,
        theta_bc: float = 0.1,
    ) -> torch.Tensor:
        """g_boundary = σ(κ_BC·(||U_BC|| - Θ_BC))"""
        norm = u_bc.abs() if u_bc.numel() == 1 else u_bc.norm()
        return torch.sigmoid(kappa_bc * (norm - theta_bc))


class BoundaryConsistencyProtocol:
    """边界一致性检查协议"""

    def __init__(self, tolerance: float = FLUX_TOLERANCE):
        self.tolerance = tolerance
        self.violations_history: List[Dict] = []

    def extract_boundary(self, solution: torch.Tensor, boundary: BoundarySpec, indices: torch.Tensor) -> torch.Tensor:
        return solution[indices]

    def compute_deviation(self, ai_values: torch.Tensor, prescribed: torch.Tensor) -> torch.Tensor:
        return (ai_values - prescribed).abs()

    def check_boundary(
        self,
        ai_solution: torch.Tensor,
        boundaries: List[BoundarySpec],
        boundary_indices: Dict[str, torch.Tensor],
        grad_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        violations = []
        for spec in boundaries:
            idx = boundary_indices.get(spec.name)
            if idx is None:
                continue
            ai_vals = self.extract_boundary(ai_solution, spec, idx)
            if spec.type == BoundaryType.DIRICHLET:
                prescribed = torch.full_like(ai_vals, spec.value or 0.0)
                dev = self.compute_deviation(ai_vals, prescribed)
                if spec.allowed_range and (ai_vals < spec.allowed_range[0]).any() or (ai_vals > spec.allowed_range[1]).any():
                    violations.append({
                        "boundary": spec.name,
                        "type": "range_violation",
                        "magnitude": dev.max().item(),
                    })
            elif spec.type == BoundaryType.NEUMANN and grad_fn is not None:
                grad_n = grad_fn(ai_solution, idx)
                total_flux = grad_n.sum().item()
                expected = spec.flux or 0.0
                if abs(total_flux - expected) > self.tolerance:
                    violations.append({
                        "boundary": spec.name,
                        "type": "flux_violation",
                        "magnitude": abs(total_flux - expected),
                    })
        self.violations_history.extend(violations)
        return violations
