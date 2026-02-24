"""
公理权重自适应 (Axiom-Credibility Mechanism)
β(t) ∈ [0,1], dβ/dt = η·(Consistency - β)
可信度感知的DAS: h = (1-g·β)·h_soft + (g·β)·h_hard
"""

import torch
import time
import numpy as np
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field


@dataclass
class CalibrationRecord:
    observed: Dict
    predicted: Any
    error: float
    consistency: float
    timestamp: float


class AxiomCredibilityManager:
    """
    管理物理公理的可信度与校准
    β_total = β_model · ∏ β_param · β_domain
    """

    def __init__(
        self,
        hard_kernel: Callable,
        calibration_window: int = 1000,
        beta_momentum: float = 0.99,
        calibration_threshold: float = 0.1,
        beta_threshold: float = 0.5,
    ):
        self.hard_kernel = hard_kernel
        self.calibration_data: list = []
        self.beta = 1.0
        self.beta_history: list = []
        self.calibration_window = calibration_window
        self.beta_momentum = beta_momentum
        self.calibration_threshold = calibration_threshold
        self.beta_threshold = beta_threshold
        self.error_variance = 0.01

    def update_credibility(
        self,
        observed_data: Dict[str, Any],
        context: Optional[Dict] = None,
    ) -> float:
        """
        Consistency(D,f) = exp(-(1/N) Σ ||y_i - f(x_i)||²/σ_i²)
        β_new = momentum·β + (1-momentum)·consistency
        """
        with torch.no_grad():
            inp = observed_data.get("input")
            out_true = observed_data.get("output")
            if inp is None or out_true is None:
                return self.beta
            predicted = self.hard_kernel(inp, context) if context is not None else self.hard_kernel(inp, None)
            if isinstance(predicted, tuple):
                predicted = predicted[0]
            error = torch.norm(predicted - out_true).item()
            rel_err = error / (torch.norm(out_true).item() + 1e-8)
            consistency = float(np.exp(-(rel_err ** 2) / (2 * self.error_variance)))
        self.beta = self.beta_momentum * self.beta + (1 - self.beta_momentum) * consistency
        self.calibration_data.append({
            "observed": observed_data,
            "predicted": predicted,
            "error": error,
            "consistency": consistency,
            "timestamp": time.time(),
        })
        if len(self.calibration_data) > self.calibration_window:
            self.calibration_data.pop(0)
        self.beta_history.append(self.beta)
        return self.beta

    def needs_calibration(self) -> bool:
        if len(self.calibration_data) < 100:
            return False
        recent = self.calibration_data[-100:]
        avg_error = np.mean([d["error"] for d in recent])
        return avg_error > self.calibration_threshold or self.beta < self.beta_threshold

    def adaptive_shielding(
        self,
        soft_out: torch.Tensor,
        hard_out: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """h_credibility_aware = (1 - g·β)·h_soft + (g·β)·h_hard"""
        beta = min(1.0, max(0.0, self.beta))
        effective_gate = gate * beta
        if effective_gate.dim() < soft_out.dim():
            effective_gate = effective_gate.expand_as(soft_out)
        return (1 - effective_gate) * soft_out + effective_gate * hard_out

    def effective_kappa(self, kappa_0: float, gamma: float = 1.0) -> float:
        """κ_effective = κ_0 · β^γ"""
        return kappa_0 * (self.beta ** gamma)
