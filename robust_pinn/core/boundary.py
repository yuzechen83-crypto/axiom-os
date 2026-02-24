"""
Boundary Ghost Points - Document Section 3
Generate points slightly outside domain Ω.
Enforce Neumann: ∇u·n = 0 (flux = 0).
"""

import torch
from typing import Tuple, Optional
from ..config import DTYPE, GHOST_MARGIN


class GhostPointHandler:
    """
    Ghost points: virtual points slightly outside domain.
    Loss: Neumann condition ∇u·n = 0
    """

    def __init__(
        self,
        domain_bounds: Tuple[Tuple[float, float], ...],
        margin: float = GHOST_MARGIN,
        dtype: torch.dtype = DTYPE,
    ):
        """
        domain_bounds: e.g. ((t_min, t_max), (x_min, x_max))
        """
        self.bounds = domain_bounds
        self.margin = margin
        self.dtype = dtype

    def generate_ghost_points(
        self,
        n_per_face: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate ghost points outside each face.
        Returns: (coords, normal_vectors)
        coords: (N, dim) points slightly outside Ω
        normals: (N, dim) outward normals for Neumann
        """
        dim = len(self.bounds)
        coords_list = []
        normals_list = []

        for d in range(dim):
            lo, hi = self.bounds[d]
            # Face at lo: point at lo - margin, normal = +1 in dim d
            pts_lo = torch.rand(n_per_face, dim, device=device, dtype=self.dtype)
            for j in range(dim):
                if j == d:
                    pts_lo[:, j] = lo - self.margin
                else:
                    rng = self.bounds[j][1] - self.bounds[j][0]
                    pts_lo[:, j] = self.bounds[j][0] + torch.rand(n_per_face, device=device, dtype=self.dtype) * rng
            norm_lo = torch.zeros(n_per_face, dim, device=device, dtype=self.dtype)
            norm_lo[:, d] = 1.0
            coords_list.append(pts_lo)
            normals_list.append(norm_lo)

            # Face at hi
            pts_hi = torch.rand(n_per_face, dim, device=device, dtype=self.dtype)
            for j in range(dim):
                if j == d:
                    pts_hi[:, j] = hi + self.margin
                else:
                    rng = self.bounds[j][1] - self.bounds[j][0]
                    pts_hi[:, j] = self.bounds[j][0] + torch.rand(n_per_face, device=device, dtype=self.dtype) * rng
            norm_hi = torch.zeros(n_per_face, dim, device=device, dtype=self.dtype)
            norm_hi[:, d] = -1.0
            coords_list.append(pts_hi)
            normals_list.append(norm_hi)

        coords = torch.cat(coords_list, dim=0)
        normals = torch.cat(normals_list, dim=0)
        return coords, normals

    def boundary_loss(
        self,
        model: torch.nn.Module,
        n_per_face: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        L_bc = ||∇u·n||² on ghost points
        Neumann: ∇u·n = 0
        """
        coords, normals = self.generate_ghost_points(n_per_face, device)
        coords = coords.detach().requires_grad_(True)

        u = model(coords)
        if u.dim() > 1 and u.shape[-1] > 1:
            u = u.sum(dim=-1)

        (grad_u,) = torch.autograd.grad(u.sum(), coords, create_graph=True)
        # ∇u·n
        flux = (grad_u * normals).sum(dim=-1)
        return torch.mean(flux ** 2)
