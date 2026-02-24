"""
Interpretable Report 可解释报告
输出物理解 + 置信度 + 可解释报告
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class InterpretableReport:
    """可解释报告结构"""

    result: Any = None
    confidence: float = 1.0
    scale_info: Dict[str, float] = field(default_factory=dict)
    memory_hits: int = 0
    validation_route: str = "safe"
    protocol_steps_passed: int = 0
    violations: list = field(default_factory=list)
    physics_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "confidence": self.confidence,
            "scale_info": self.scale_info,
            "memory_hits": self.memory_hits,
            "validation_route": self.validation_route,
            "protocol_steps_passed": self.protocol_steps_passed,
            "violations": self.violations,
            "physics_checks": self.physics_checks,
        }

    def __str__(self) -> str:
        lines = [
            "=== SPNN Interpretable Report ===",
            f"Result: {self.result}",
            f"Confidence: {self.confidence:.4f}",
            f"Validation Route: {self.validation_route}",
            f"Protocol Steps Passed: {self.protocol_steps_passed}",
            f"Memory Retrieval Hits: {self.memory_hits}",
            f"Scale Info: {self.scale_info}",
        ]
        if self.violations:
            lines.append(f"Violations: {self.violations}")
        if self.physics_checks:
            lines.append(f"Physics Checks: {self.physics_checks}")
        return "\n".join(lines)


def generate_report(
    result: Any,
    confidence: float,
    scale_info: Optional[Dict] = None,
    memory_hits: int = 0,
    validation_route: str = "safe",
    protocol_steps: int = 0,
    violations: Optional[list] = None,
    physics_checks: Optional[Dict] = None,
) -> InterpretableReport:
    return InterpretableReport(
        result=result,
        confidence=confidence,
        scale_info=scale_info or {},
        memory_hits=memory_hits,
        validation_route=validation_route,
        protocol_steps_passed=protocol_steps,
        violations=violations or [],
        physics_checks=physics_checks or {},
    )
