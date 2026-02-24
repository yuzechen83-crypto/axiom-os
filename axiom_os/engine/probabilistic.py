"""
Probabilistic Inference - Diffusion-based uncertainty quantification
When uncertainty_mode=True: sample from p(x_{t+1} | x_t) via diffusion.
"""

from typing import Optional, Tuple, Callable
import torch

from .diffusion import ScoreNet
from .sde_solver import DiffusionSDE
from .sampler import sample_with_physics_residual


def probabilistic_predict(
    score_net: ScoreNet,
    sde: DiffusionSDE,
    x: torch.Tensor,
    hard_core_func: Callable,
    n_samples: int = 10,
    n_steps: int = 20,
    sigma_init: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Probabilistic one-step prediction: p(x_{t+1} | x_t).
    Returns mean_prediction and uncertainty_bound (std across samples).

    Args:
        score_net: Trained ScoreNet
        sde: DiffusionSDE
        x: (B, state_dim) current state
        hard_core_func: physics predictor x -> x_next
        n_samples: number of samples for mean/std
        n_steps: reverse diffusion steps
        sigma_init: initial noise scale

    Returns:
        mean_prediction: (B, state_dim)
        uncertainty_bound: (B, state_dim) or scalar std
    """
    with torch.no_grad():
        hard_pred = hard_core_func(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        samples = []
        for _ in range(n_samples):
            s = sample_with_physics_residual(
                score_net,
                sde,
                hard_pred,
                condition=x,
                n_steps=n_steps,
                sigma_init=sigma_init,
            )
            samples.append(s)
        samples = torch.stack(samples, dim=0)
        mean_pred = samples.mean(dim=0)
        std_pred = samples.std(dim=0)
    return mean_pred, std_pred
