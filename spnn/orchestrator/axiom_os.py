"""
Axiom-OS Orchestrator - Main Brain O
O = (O_core, O_middle, O_app)
协议驱动推演、智能调度、异常仲裁
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..core.physical_scale import PhysicalScaleSystem
from ..core.upi_interface import UPIInterface
from ..memory.hippocampus import Hippocampus


class RouteDecision(Enum):
    SAFE = "safe"
    EXPLORE = "explore"
    HYBRID = "hybrid"


@dataclass
class ProtocolStep:
    """协议步骤"""
    name: str
    verify_fn: Callable
    params: Dict = field(default_factory=dict)


class AxiomOS:
    """
    O = (O_core, O_middle, O_app)
    主脑：全局协调、协议驱动、智能调度、双路径验证
    """

    def __init__(
        self,
        scale_system: Optional[PhysicalScaleSystem] = None,
        upi: Optional[UPIInterface] = None,
        hippocampus: Optional[Hippocampus] = None,
        scheduler_freq: float = 50.0,
        risk_tolerance: float = 0.05,
        timeout: float = 5.0,
    ):
        self.scale_system = scale_system or PhysicalScaleSystem()
        self.upi = upi or UPIInterface(scale_system=self.scale_system)
        self.hippocampus = hippocampus
        self.scheduler_freq = scheduler_freq
        self.risk_tolerance = risk_tolerance
        self.timeout = timeout

        self._protocol_chain: List[ProtocolStep] = []
        self._violations: List[Dict] = []
        self._performance_history: List[float] = []
        self._resource_state: Dict[str, float] = {"cpu": 1.0, "mem": 1.0, "gpu": 1.0}

    def add_protocol(self, name: str, verify_fn: Callable, **params) -> None:
        """添加协议步骤"""
        self._protocol_chain.append(ProtocolStep(name=name, verify_fn=verify_fn, params=params))

    def execute_protocol_chain(
        self,
        x: np.ndarray,
        step_fn: Callable,
        **kwargs,
    ) -> Tuple[np.ndarray, bool]:
        """
        协议驱动推演: x_k = O(x_{k-1}, p_k)
        Verify(x, p) = UPIVerify · CausalCheck · ScaleCheck · MemoryCheck
        """
        current = x
        for step in self._protocol_chain:
            current = step_fn(current, step, **kwargs)
            if not step.verify_fn(current, **step.params):
                self._violations.append({"step": step.name, "x": current})
                return current, False
        return current, True

    def compute_schedule(
        self,
        task_priority: float,
        resource_demand: Dict[str, float],
        physics_violation: float = 0.0,
    ) -> Dict[str, Any]:
        """
        智能调度
        π* = argmin [Cost(π) + λ·Risk(π) + μ·PhysicsViolation(π)]
        """
        cost = (
            0.3 * (1.0 - self._resource_state.get("cpu", 1.0)) +
            0.3 * (1.0 - self._resource_state.get("mem", 1.0)) +
            0.2 * (1.0 - task_priority) +
            0.2 * physics_violation
        )
        return {
            "lambda_res": max(0.1, 0.8 - 0.1 * self._performance_history.__len__() if self._performance_history else 0.5),
            "alpha": 0.5 + 0.2 * task_priority,
            "risk_tolerance": self.risk_tolerance * (1.0 + physics_violation),
            "cost": cost,
        }

    def route_anomaly(self, anomaly: Any, context: Dict) -> RouteDecision:
        """
        主脑决策路由：双路径验证
        Route(A) = f_brain(D, N, C, context)
        """
        D = context.get("detection_score", 0.5)
        N = context.get("novelty", 0.5)
        C = context.get("confidence", 0.5)

        if D > 0.8 and N < 0.3:
            return RouteDecision.SAFE
        if D < 0.3 and N > 0.7:
            return RouteDecision.EXPLORE
        return RouteDecision.HYBRID

    def arbitrate(
        self,
        result_safe: np.ndarray,
        result_explore: np.ndarray,
        context: Dict,
    ) -> np.ndarray:
        """
        双路径仲裁
        FinalDecision = Φ_brain(result_safe, result_explore, context)
        Φ_brain = softmax(W·[r_s; r_e; c] + b)
        """
        r_s = np.asarray(result_safe).flatten()
        r_e = np.asarray(result_explore).flatten()
        max_len = max(len(r_s), len(r_e))
        r_s = np.pad(r_s, (0, max_len - len(r_s)), constant_values=0)
        r_e = np.pad(r_e, (0, max_len - len(r_e)), constant_values=0)
        c = np.array([context.get("task_priority", 0.5), context.get("risk", 0.5)])
        combined = np.concatenate([r_s, r_e, c])
        weights = np.exp(combined - combined.max())
        weights = weights / weights.sum()
        w_s, w_e = weights[:len(r_s)].sum(), weights[len(r_s):len(r_s)+len(r_e)].sum()
        return w_s * result_safe + w_e * result_explore

    def update_performance(self, metric: float) -> None:
        self._performance_history.append(metric)
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-500:]

    def update_resource_state(self, state: Dict[str, float]) -> None:
        self._resource_state.update(state)

    def get_training_params(self, phase: int) -> Dict[str, float]:
        """
        三阶段自适应调度
        阶段1: 物理定型 (α=0.7, λ_res=0.8)
        阶段2: 柔性演化 (α=0.5, λ_res=0.3)
        阶段3: 精细调优 (α=0.5, λ_res=0.1)
        """
        schedules = [
            {"alpha": 0.7, "lambda_res": 0.8, "eta": 1e-3},
            {"alpha": 0.5, "lambda_res": 0.3, "eta": 5e-4},
            {"alpha": 0.5, "lambda_res": 0.1, "eta": 1e-4},
        ]
        return schedules[min(phase, 2)]
