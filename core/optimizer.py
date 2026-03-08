"""
Breathing Optimizer - MASHD Meta-Axis Dynamics
Z̈ + kZ + λZ³ = TrainingProgress

High Z (Expansion): High LR, High Noise (Exploration/Entropy).
Low Z (Contraction): Low LR, Zero Noise (Convergence/Gravity).

BreathingOptimizer: Wraps optimizer + entropy force T(t)·Noise injection.
"""

from typing import Optional, Callable
import torch
from torch.optim import Optimizer


class BreathingScheduler:
    """
    Dynamic scheduler based on Duffing equation.
    Couples LR and noise to Meta-Axis position Z(t).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        k: float = 1.0,
        lambda_: float = 0.1,
        gamma: float = 0.9,
        dt: float = 0.1,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.k = k
        self.lambda_ = lambda_
        self.gamma = gamma
        self.dt = dt
        self.Z = 0.5
        self.Z_dot = 0.0
        self._step = 0

    def step(self, loss: Optional[float] = None, training_progress: Optional[float] = None) -> None:
        """
        Duffing: Z̈ + kZ + λZ³ = TrainingProgress.
        Set lr = base_lr * (1 + Z).
        """
        if training_progress is None:
            training_progress = min(1.0, self._step / 1000.0)
        self._step += 1

        # Duffing: Z̈ = -kZ - λZ³ + TrainingProgress
        Z_ddot = -self.k * self.Z - self.lambda_ * (self.Z ** 3) + training_progress
        self.Z_dot = self.gamma * self.Z_dot + self.dt * Z_ddot
        self.Z = self.Z + self.dt * self.Z_dot
        self.Z = max(-0.99, min(2.0, self.Z))

        lr = self.base_lr * (1.0 + self.Z)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def get_noise_scale(self) -> float:
        """High Z → high noise (exploration), Low Z → zero noise."""
        return max(0.0, self.Z)

    def get_Z(self) -> float:
        return self.Z


class BreathingOptimizer:
    """
    MASHD Thermodynamic Optimizer: Wraps optimizer + entropy force T(t)·Noise.

    After backward(), injects gradient noise: grad += T(t) * randn_like(grad).
    T(t) = noise_scale * get_noise_scale() from BreathingScheduler.
    High Z → entropy force dominant (exploration); Low Z → gravity dominant (convergence).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        noise_scale: float = 1e-4,
        use_entropy: bool = True,
        k: float = 1.0,
        lambda_: float = 0.1,
        gamma: float = 0.9,
        dt: float = 0.1,
    ):
        self.optimizer = optimizer
        self.scheduler = BreathingScheduler(
            optimizer, base_lr=base_lr, k=k, lambda_=lambda_, gamma=gamma, dt=dt
        )
        self.noise_scale = noise_scale
        self.use_entropy = use_entropy

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(
        self,
        loss: Optional[torch.Tensor] = None,
        training_progress: Optional[float] = None,
    ) -> None:
        """
        Call after loss.backward(). Injects entropy noise, then optimizer.step(), scheduler.step().
        """
        # Entropy force: T(t)·Noise(z) on gradients
        if self.use_entropy:
            T = self.noise_scale * self.scheduler.get_noise_scale()
            if T > 1e-8:
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad, device=p.device) * T)

        self.optimizer.step()
        loss_val = float(loss.item()) if loss is not None and hasattr(loss, "item") else None
        self.scheduler.step(loss=loss_val, training_progress=training_progress)

    def get_noise_scale(self) -> float:
        return self.scheduler.get_noise_scale()

    def get_Z(self) -> float:
        return self.scheduler.get_Z()
