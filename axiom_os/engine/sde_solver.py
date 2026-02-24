"""
Diffusion SDE - Forward & Reverse Process
DDPM / Score SDE with variance schedule.
Physics injection: model predicts residual relative to Hard Core.
"""

from typing import Optional, Callable
import math
import torch
import torch.nn as nn


class DiffusionSDE:
    """
    Diffusion process with linear variance schedule.
    Forward: q_sample(x_0, t) -> noisy x_t
    Loss: ||ε_θ(x_t, t, condition) - ε||²
    Supports physics residual: predict noise on (x_t - hard_core_pred).
    """

    def __init__(
        self,
        state_dim: int,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        n_steps: int = 100,
        schedule: str = "linear",
    ):
        self.state_dim = state_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_steps = n_steps

        betas = torch.linspace(beta_min, beta_max, n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        x_0: (B, state_dim)
        t: (B,) integer indices in [0, n_steps-1]
        Returns: (B, state_dim) noisy x_t
        """
        device = x_0.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1)

        if noise is None:
            noise = torch.randn_like(x_0, device=device)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def loss_fn(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        hard_core_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDPM loss: E_t,ε [ ||ε_θ(x_t, t, condition) - ε||² ]
        If hard_core_pred given: predict noise on residual (x_0 - hard_core_pred).
        """
        B = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.n_steps, (B,), device=device, dtype=torch.long)

        if hard_core_pred is not None:
            residual = x_0 - hard_core_pred
            noise = torch.randn_like(residual, device=device)
            x_t = self.q_sample(residual, t, noise)
            target = noise
        else:
            noise = torch.randn_like(x_0, device=device)
            x_t = self.q_sample(x_0, t, noise)
            target = noise

        t_norm = t.float() / max(self.n_steps - 1, 1)
        pred = model(x_t, t_norm, condition)
        return ((pred - target) ** 2).mean()

    def get_alpha(self, t: torch.Tensor, device: torch.device) -> tuple:
        """Get α_t, √α̅_t, √(1-α̅_t) for reverse sampling."""
        a = self.alphas.to(device)[t].view(-1, 1)
        sqrt_a = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1)
        return a, sqrt_a, sqrt_1ma


def create_diffusion_sde(
    state_dim: int,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
    n_steps: int = 100,
) -> DiffusionSDE:
    """Factory for DiffusionSDE."""
    return DiffusionSDE(
        state_dim=state_dim,
        beta_min=beta_min,
        beta_max=beta_max,
        n_steps=n_steps,
    )
