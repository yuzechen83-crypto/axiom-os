"""
SPNN-Opt-Rev5 Robust PINN - Training Orchestrator
L_total = w_data·L_data + w_phys·L_residual + w_bc·L_boundary
- GradNorm for dynamic loss balancing
- Soft-Start: w_phys = 0 initially, ramp up over warmup
- clip_grad_norm_ (max_norm=1.0)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from .config import (
    DTYPE,
    MAX_GRAD_NORM,
    WARMUP_EPOCHS,
    W_DATA,
    W_PHYS,
    W_BC,
    GRADNORM_ALPHA,
)
from .core.arch import PhysicsResNet, SoftMeltLayer
from .core.physics import HamiltonianSystem, compute_wave_residual
from .core.boundary import GhostPointHandler


class GradNormBalancer:
    """
    Dynamic Loss Balancing via GradNorm.
    Adjust weights so that loss magnitudes and gradient magnitudes are balanced.
    """

    def __init__(self, num_losses: int, alpha: float = GRADNORM_ALPHA, lr: float = 0.025):
        self.weights = nn.Parameter(torch.ones(num_losses, dtype=DTYPE))
        self.alpha = alpha
        self.lr = lr
        self.initial_losses: Optional[torch.Tensor] = None

    def get_weights(self) -> torch.Tensor:
        return torch.softmax(self.weights, dim=0)

    def update(
        self,
        losses: torch.Tensor,
        shared_params: nn.Parameter,
    ) -> None:
        """
        losses: (L_data, L_phys, L_bc)
        Update weights based on gradient norms.
        """
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
        w = self.get_weights()
        weighted = (w * losses).sum()
        grad_shared = torch.autograd.grad(weighted, shared_params, retain_graph=True)[0]
        G = grad_shared.norm()
        L_ratios = losses.detach() / (self.initial_losses + 1e-8)
        inv_train_rates = L_ratios ** self.alpha
        target = G * inv_train_rates / (inv_train_rates.sum() + 1e-8)
        # Simplified: step weights toward balance
        with torch.no_grad():
            self.weights.sub_(self.lr * (losses.detach() - target))


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    coords_data: torch.Tensor,
    u_data: torch.Tensor,
    coords_colloc: torch.Tensor,
    boundary_handler: Optional[GhostPointHandler],
    epoch: int,
    w_phys_schedule: float,
    dtype: torch.dtype = DTYPE,
) -> Dict[str, float]:
    """
    Single training epoch.
    Soft-Start: w_phys ramps from 0 to W_PHYS over WARMUP_EPOCHS.
    """
    model.train()
    optimizer.zero_grad()

    coords_data = coords_data.to(dtype).to(model.parameters().__iter__().__next__().device)
    u_data = u_data.to(dtype).to(coords_data.device)
    coords_colloc = coords_colloc.to(dtype).to(coords_data.device)
    device = coords_data.device

    # L_data
    u_pred = model(coords_data)
    if u_pred.dim() > 1 and u_pred.shape[-1] > 1:
        u_pred = u_pred.squeeze(-1)
    if u_data.dim() > 1 and u_data.shape[-1] > 1:
        u_data = u_data.squeeze(-1)
    L_data = torch.nn.functional.mse_loss(u_pred, u_data)

    # L_residual (physics)
    L_phys = compute_wave_residual(model, coords_colloc, omega=1.0)

    # L_boundary
    L_bc = torch.tensor(0.0, device=device, dtype=dtype)
    if boundary_handler is not None:
        L_bc = boundary_handler.boundary_loss(model, n_per_face=16, device=device)

    # Soft-Start: ramp w_phys
    w_phys = w_phys_schedule

    L_total = W_DATA * L_data + w_phys * L_phys + W_BC * L_bc
    L_total.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return {
        "L_data": L_data.item(),
        "L_phys": L_phys.item(),
        "L_bc": L_bc.item() if isinstance(L_bc, torch.Tensor) else L_bc,
        "L_total": L_total.item(),
    }


def run_training(
    model: nn.Module,
    coords_data: torch.Tensor,
    u_data: torch.Tensor,
    domain_bounds: Tuple[Tuple[float, float], ...],
    epochs: int = 500,
    lr: float = 1e-3,
) -> nn.Module:
    """
    Full training loop with Soft-Start.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Collocation points (interior)
    dim = coords_data.shape[1]
    n_colloc = 256
    coords_colloc = torch.rand(n_colloc, dim, device=device, dtype=DTYPE)
    for d in range(dim):
        lo, hi = domain_bounds[d]
        coords_colloc[:, d] = lo + (hi - lo) * coords_colloc[:, d]

    boundary_handler = GhostPointHandler(domain_bounds=domain_bounds)

    for epoch in range(epochs):
        w_phys = W_PHYS * min(1.0, (epoch + 1) / WARMUP_EPOCHS)
        metrics = train_epoch(
            model, optimizer,
            coords_data, u_data,
            coords_colloc,
            boundary_handler,
            epoch,
            w_phys,
        )
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: L_total={metrics['L_total']:.6f} L_data={metrics['L_data']:.6f} L_phys={metrics['L_phys']:.6f}")

    return model


if __name__ == "__main__":
    # Demo: 1D wave u_tt + u = 0, domain (t,x) in [0,1]x[0,1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsResNet(in_dim=2, out_dim=1, hidden_dim=128, num_layers=4).to(device).to(DTYPE)

    # Synthetic data: u = sin(t) * sin(x)
    t = torch.linspace(0, 1, 50)
    x = torch.linspace(0, 1, 50)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    coords = torch.stack([tt.ravel(), xx.ravel()], dim=1)
    u_true = torch.sin(tt * 3.14159) * torch.sin(xx * 3.14159)
    u_data = u_true.ravel().unsqueeze(1)

    coords = coords.to(device)
    u_data = u_data.to(device)

    domain_bounds = ((0.0, 1.0), (0.0, 1.0))
    run_training(model, coords, u_data, domain_bounds, epochs=30)
