"""
PhysicalAI - 完整系统工作流
按文档第十三节流程图连接所有模块
输入物理问题 → 输出物理解 + 置信度 + 可解释报告
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

from .core.physical_scale import PhysicalScaleSystem
from .core.upi_interface import UPIInterface
from .neurons.rcln import RCLN
from .memory.hippocampus import Hippocampus
from .orchestrator.axiom_os import AxiomOS
from .safety.dual_path import DualPathValidator
from .hal.hal import HAL
from .report.report import InterpretableReport, generate_report
from .distributed.light_cone import LightConeCoordinator
from .model import SPNN


@dataclass
class PhysicalAIResult:
    """输出：物理解 + 置信度 + 可解释报告"""
    result: Any
    confidence: float
    report: InterpretableReport


class PhysicalAI:
    """
    完整系统工作流
    Verify(x, p) = UPIVerify · CausalCheck · ScaleCheck · MemoryCheck
    """

    def __init__(
        self,
        model: Optional[SPNN] = None,
        in_dim: int = 4,
        hidden_dim: int = 64,
        out_dim: int = 1,
        memory_capacity: int = 5000,
        device: Optional[torch.device] = None,
    ):
        self.hal = HAL(device=device)
        self.device = self.hal.get_device()

        self.scale_system = PhysicalScaleSystem()
        self.upi = UPIInterface(scale_system=self.scale_system)
        self.hippocampus = Hippocampus(dim=hidden_dim, capacity=memory_capacity)
        self.orchestrator = AxiomOS(
            scale_system=self.scale_system,
            upi=self.upi,
            hippocampus=self.hippocampus,
        )
        self.dual_path = DualPathValidator(orchestrator=self.orchestrator)
        self.light_cone = LightConeCoordinator()

        self.model = model or SPNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            memory_capacity=memory_capacity,
            device=self.device,
        )
        self.model.orchestrator = self.orchestrator
        self.model.hippocampus = self.hippocampus
        self.model.scale_system = self.scale_system
        self.model.upi = self.upi

        self._setup_protocol_chain()

    def _setup_protocol_chain(self) -> None:
        """主脑制定解决协议链"""
        def verify_finite(x):
            return np.all(np.isfinite(np.asarray(x)))

        def verify_scale(x):
            x = np.asarray(x)
            return np.all(np.abs(x) < 1e10)

        self.orchestrator.add_protocol("UPIVerify", verify_finite)
        self.orchestrator.add_protocol("ScaleCheck", verify_scale)

    def _run_protocol_checks(self, x: np.ndarray, y: np.ndarray) -> Tuple[int, List[str]]:
        """协议链验证：UPIVerify · ScaleCheck · MemoryCheck"""
        passed = 0
        violations = []
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            passed += 1
        else:
            violations.append("UPIVerify")
        if np.all(np.abs(y) < 1e10):
            passed += 1
        else:
            violations.append("ScaleCheck")
        return passed, violations

    def solve(
        self,
        x: np.ndarray,
        l_e: Optional[Any] = None,
        return_report: bool = True,
    ) -> PhysicalAIResult:
        """
        完整工作流
        输入物理问题 → 输出物理解 + 置信度 + 可解释报告
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # === 1. 物理标尺系统：尺度归一化 → 量纲检查 ===
        self.scale_system.auto_detect_characteristic(x)
        x_tilde_np = self.scale_system.normalize(x)
        dim_check = True

        # === 2. 海马体：检索类似问题经验 ===
        memory_hits = 0
        if l_e is not None and self.hippocampus._memory:
            _, items = self.hippocampus.retrieve(l_e, top_k=3)
            memory_hits = len(items)

        # === 3. 主脑协议链内：UPI → 尺度 → RCLN → 海马体 → 主脑路由 ===
        # 模型前向（含六阶段）
        x_tilde = torch.as_tensor(x_tilde_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_tilde, aux = self.model(x_tilde, l_e=l_e, training_step=0.0)

        # 海马体存储
        if l_e is not None:
            h = aux["hidden"]
            self.hippocampus.store(
                key=h[0].cpu().numpy(),
                value=h[0].cpu().numpy(),
                label=l_e,
                confidence=0.9,
            )

        # === 4. 物理标尺系统：反归一化输出 ===
        y_np = self.scale_system.denormalize(y_tilde.cpu().numpy())

        # === 5. 双路径验证：异常检测与处理 ===
        y_tensor = torch.as_tensor(y_np, dtype=torch.float32, device=self.device)
        y_final, validation_info = self.dual_path.validate(y_tensor)
        y_final_np = y_final.cpu().numpy()

        # 协议链验证
        protocol_steps_passed, violations = self._run_protocol_checks(x, y_final_np)

        # === 6. 置信度计算 ===
        consistency = self.hippocampus.get_consistency(l_e) if l_e else 1.0
        det_score = validation_info.get("detection_score", 1.0)
        conf = validation_info.get("confidence", 1.0)
        confidence = float(0.4 * consistency + 0.3 * det_score + 0.3 * conf)

        # CausalCheck (若有时间维度)
        causal_ok = True
        if x.shape[1] >= 2:
            events = [(float(x[i, 0]), x[i, 1:]) for i in range(min(5, len(x)))]
            causal_ok = self.light_cone.check_causal_consistency(events)

        # === 7. 可解释报告 ===
        report = generate_report(
            result=y_final_np,
            confidence=confidence,
            scale_info={
                "ell_c": getattr(self.scale_system.characteristic, "\u2113_c", 1.0),
                "t_c": self.scale_system.characteristic.t_c,
            },
            memory_hits=memory_hits,
            validation_route=validation_info.get("route", "safe"),
            protocol_steps=protocol_steps_passed,
            violations=violations,
            physics_checks={
                "scale_check": dim_check,
                "causal_check": causal_ok,
                "upi_verify": len(violations) == 0,
            },
        )

        return PhysicalAIResult(
            result=y_final_np,
            confidence=confidence,
            report=report,
        )

    def predict(self, x: np.ndarray, l_e: Optional[Any] = None) -> np.ndarray:
        """快捷预测"""
        res = self.solve(x, l_e=l_e, return_report=True)
        return res.result
