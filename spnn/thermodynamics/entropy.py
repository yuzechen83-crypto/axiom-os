"""
热力学第二定律硬约束 (Entropy Consistency)
Φ_dissipation = F_AI · ∇_x S(x) ≥ 0
克劳修斯-杜安不等式形式化
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EntropyCheckResult:
    entropy_production: float
    violation: bool
    severity: float


class EntropyConsistencyEnforcer:
    """
    确保AI不产生"反熵"解
    dS/dt ≥ 0 (孤立系统熵不减)
    """

    def __init__(
        self,
        entropy_function: Callable[[torch.Tensor], torch.Tensor],
        tolerance: float = 1e-6,
    ):
        self.S = entropy_function
        self.epsilon = tolerance
        self._correction_log: list = []

    def check_entropy_production(
        self,
        ai_dynamics: torch.Tensor,
        state: torch.Tensor,
    ) -> EntropyCheckResult:
        """
        Φ_dissipation = F_AI · ∇_x S(x)
        检查是否满足 Φ ≥ -ε
        """
        state = state.detach().requires_grad_(True)
        entropy = self.S(state)
        grad_S = torch.autograd.grad(entropy.sum(), state, create_graph=True)[0]
        entropy_production = (ai_dynamics * grad_S).sum()
        ep = entropy_production.item()
        violation = ep < -self.epsilon
        severity = max(0.0, -ep) if violation else 0.0
        return EntropyCheckResult(ep, violation, severity)

    def project_to_entropy_increasing(
        self,
        dynamics: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        将动力学投影到熵不减子空间
        min ||v - dynamics||², s.t. v·∇S ≥ 0
        """
        state = state.detach().requires_grad_(True)
        S = self.S(state)
        grad_S = torch.autograd.grad(S.sum(), state, create_graph=True)[0]
        current_production = (dynamics * grad_S).sum()
        if current_production >= -self.epsilon:
            return dynamics
        grad_S_norm = grad_S / (grad_S.norm() + 1e-8)
        required = -current_production / (grad_S.norm() + 1e-8)
        corrected = dynamics + required * grad_S_norm
        new_production = (corrected * grad_S).sum()
        assert new_production >= -self.epsilon * 1.1, "Projection failed"
        return corrected

    def enforce_constraint(
        self,
        ai_dynamics: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """若违反熵增则修正"""
        result = self.check_entropy_production(ai_dynamics, state)
        if result.violation:
            corrected = self.project_to_entropy_increasing(ai_dynamics, state)
            self._correction_log.append({
                "original_norm": ai_dynamics.norm().item(),
                "corrected_norm": corrected.norm().item(),
                "severity": result.severity,
            })
            return corrected
        return ai_dynamics

    def entropy_loss(
        self,
        ai_dynamics: torch.Tensor,
        state: torch.Tensor,
        lambda_entropy: float = 1.0,
    ) -> torch.Tensor:
        """L_entropy = λ·max(0, -Φ + ε)²"""
        result = self.check_entropy_production(ai_dynamics, state)
        violation = max(0.0, -result.entropy_production + self.epsilon)
        return torch.tensor(lambda_entropy * (violation ** 2), device=ai_dynamics.device)
