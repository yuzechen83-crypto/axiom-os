"""
SPNN 工业级包装器
ThermodynamicallySafeModule, boundary_conditions 装饰器
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from functools import wraps

from .thermodynamics import EntropyConsistencyEnforcer
from .boundary import BoundaryConsistencyProtocol
from .axiom import ErrorAttributionDiagnostic
from .core.calibration import apply_das, safe_norm, erf_gate


class ThermodynamicallySafeModule(nn.Module):
    """
    热力学安全模块: 基础模块 + 熵一致性检查与修正
    """

    def __init__(
        self,
        base_module: nn.Module,
        entropy_function: Callable[[torch.Tensor], torch.Tensor],
        tolerance: float = 1e-6,
    ):
        super().__init__()
        self.base_module = base_module
        self.entropy_enforcer = EntropyConsistencyEnforcer(entropy_function, tolerance)

    def forward(
        self,
        x: torch.Tensor,
        enforce: bool = True,
    ) -> torch.Tensor:
        raw = self.base_module(x)
        if enforce and self.training:
            # 若输出可视为动力学，则检查熵
            if raw.requires_grad and x.requires_grad:
                safe = self.entropy_enforcer.enforce_constraint(raw, x)
                return safe
        return raw


def boundary_conditions(bc_config: Dict[str, Dict]) -> Callable:
    """边界条件装饰器 (配置解析)"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._boundary_config = bc_config
        return wrapper
    return decorator


class DASWithDiagnostic(nn.Module):
    """
    DAS + 异常溯源: gate > 0.8 时自动触发诊断
    """

    def __init__(self, diagnostic_threshold: float = 0.8):
        super().__init__()
        self.diagnostic = ErrorAttributionDiagnostic()
        self.threshold = diagnostic_threshold

    def forward(
        self,
        soft_out: torch.Tensor,
        hard_out: torch.Tensor,
        context: Optional[Dict] = None,
    ) -> torch.Tensor:
        shielded, potential = apply_das(soft_out, hard_out)
        residual = soft_out - hard_out
        rho_w = safe_norm(residual)
        gate = erf_gate(rho_w.mean(), kappa=10.0, rho_thresh=0.5)
        gate_val = gate.item() if gate.numel() == 1 else gate.mean().item()
        if gate_val > self.threshold:
            gate_t = gate if isinstance(gate, torch.Tensor) else torch.tensor(gate_val, device=soft_out.device)
            report = self.diagnostic.diagnose_violation(
                soft_out, hard_out, gate_t.unsqueeze(0), context
            )
            if report.root_cause.value == "axiom_parametric_drift":
                pass  # 可在此触发校准
        return shielded
