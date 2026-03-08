"""
Physics-Guided Diffusion Sampler
Denoise starting from Hard Core prediction; optional energy guidance.
"""

from typing import Optional, Callable
import torch
import torch.nn as nn

from .diffusion import ScoreNet
from .sde_solver import DiffusionSDE


def sample_with_physics(
    model: nn.Module,
    sde: DiffusionSDE,
    hard_core_pred: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    n_steps: Optional[int] = None,
    sigma_init: float = 1.0,
    energy_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    lambda_energy: float = 0.0,
) -> torch.Tensor:
    """
    Reverse diffusion with physics guidance.
    Start from x_T ~ N(hard_core_pred, σ² I), denoise to x_0.

    Args:
        model: ScoreNet predicting ε_θ(x_t, t, condition)
        sde: DiffusionSDE with schedule
        hard_core_pred: (B, state_dim) physics prediction
        condition: (B, cond_dim) previous state
        n_steps: number of reverse steps (default: sde.n_steps)
        sigma_init: std of initial noise around hard_core_pred
        energy_func: optional E(x) for guidance; score_hat = score_net - λ∇E
        lambda_energy: strength of energy guidance

    Returns:
        (B, state_dim) sampled x_0
    """
    n_steps = n_steps or sde.n_steps
    device = hard_core_pred.device
    B, state_dim = hard_core_pred.shape

    # Start near physics guess: x_T = hard_core_pred + σ * ε
    x = hard_core_pred + sigma_init * torch.randn_like(hard_core_pred, device=device)

    dt = 1.0 / n_steps
    model.eval()

    with torch.no_grad():
        for i in range(n_steps - 1, -1, -1):
            t_idx = torch.full((B,), i, device=device, dtype=torch.long)
            t_norm = torch.full((B,), i / max(n_steps - 1, 1), device=device, dtype=torch.float32)

            # Predict noise (we predict noise on residual if trained that way; here we denoise x directly)
            eps_pred = model(x, t_norm, condition)

            # DDPM reverse: x_{t-1} from x_t using predicted noise
            alpha, sqrt_alpha, sqrt_1m_alpha = sde.get_alpha(t_idx, device)

            # x_{t-1} = (1/sqrt(α_t)) * (x_t - (1-α_t)/sqrt(1-α̅_t) * ε_pred) + σ_t * z
            coef = (1 - alpha) / (sqrt_1m_alpha + 1e-8)
            x_mean = (x - coef * eps_pred) / (sqrt_alpha + 1e-8)

            if i > 0:
                # Add noise (DDPM style)
                beta_t = sde.betas.to(device)[t_idx].view(-1, 1)
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x, device=device)
                x = x_mean + sigma_t * z
            else:
                x = x_mean

            # Optional: energy guidance (finite-diff ∇E outside no_grad)
            if energy_func is not None and lambda_energy > 0:
                eps_fd = 1e-5
                grad_E = torch.zeros_like(x, device=device)
                for d in range(x.shape[1]):
                    x_plus = x.clone()
                    x_plus[:, d] += eps_fd
                    x_minus = x.clone()
                    x_minus[:, d] -= eps_fd
                    E_plus = energy_func(x_plus)
                    E_minus = energy_func(x_minus)
                    grad_E[:, d] = (E_plus - E_minus).view(-1) / (2 * eps_fd)
                x = x - lambda_energy * dt * grad_E

    return x


def sample_with_physics_residual(
    model: nn.Module,
    sde: DiffusionSDE,
    hard_core_pred: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    n_steps: Optional[int] = None,
    sigma_init: float = 1.0,
) -> torch.Tensor:
    """
    When model was trained on residual (x_0 - hard_core_pred), we denoise
    the residual and add back to hard_core_pred.
    """
    n_steps = n_steps or sde.n_steps
    device = hard_core_pred.device
    B, state_dim = hard_core_pred.shape

    # Residual starts as pure noise
    residual = sigma_init * torch.randn_like(hard_core_pred, device=device)

    dt = 1.0 / n_steps
    model.eval()

    with torch.no_grad():
        for i in range(n_steps - 1, -1, -1):
            t_idx = torch.full((B,), i, device=device, dtype=torch.long)
            t_norm = torch.full((B,), i / max(n_steps - 1, 1), device=device, dtype=torch.float32)

            eps_pred = model(residual, t_norm, condition)
            alpha, sqrt_alpha, sqrt_1m_alpha = sde.get_alpha(t_idx, device)

            coef = (1 - alpha) / (sqrt_1m_alpha + 1e-8)
            x_mean = (residual - coef * eps_pred) / (sqrt_alpha + 1e-8)

            if i > 0:
                beta_t = sde.betas.to(device)[t_idx].view(-1, 1)
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(residual, device=device)
                residual = x_mean + sigma_t * z
            else:
                residual = x_mean

    return hard_core_pred + residual
