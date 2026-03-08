"""
元轴丛场 - 连锁反应
结晶时：更新联络，建立 principal ↔ perturbation 关系
"""

from typing import List, Optional

from .base import BaseSpace
from .fiber import Fiber, FormulaEntry
from .connection import ConnectionType
from .axes import BasePoint, Regime, ResidualRole


class ChainReaction:
    """
    结晶时的连锁反应：
    1. 新公式加入 → 检查同基点 fiber 的 principal
    2. 若已有 principal，建立扰动联络
    3. 返回被触发的 formula_id 列表
    """
    def __init__(self, base_space: BaseSpace):
        self.base_space = base_space

    def on_crystallize(
        self,
        formula_id: str,
        formula: str,
        callable_fn,
        output_dim: int,
        base_point: BasePoint,
        residual_role: ResidualRole = ResidualRole.PRINCIPAL,
    ) -> List[str]:
        """
        结晶时调用：更新 fiber，建立联络，返回被触发的 formula_id 列表
        """
        entry = FormulaEntry(
            formula_id=formula_id,
            formula=formula,
            callable=callable_fn,
            output_dim=output_dim,
            residual_role=residual_role,
            regime=base_point.regime,
            metadata={},
        )

        fiber = self.base_space.get_fiber(base_point)
        if fiber is None:
            fiber = Fiber(base_point=base_point)
            self.base_space.set_fiber(base_point, fiber)

        triggered: List[str] = []
        if residual_role == ResidualRole.PRINCIPAL:
            fiber.principal = entry
        else:
            fiber.perturbations.append(entry)
            if fiber.principal is not None:
                self.base_space.connection.link_perturbation(
                    fiber.principal.formula_id, formula_id
                )
                triggered.append(fiber.principal.formula_id)
            self.base_space.set_fiber(base_point, fiber)

        return triggered

    def mark_negligible(self, base_point: BasePoint, formula_id: str) -> None:
        """标记某残差为可忽略"""
        fiber = self.base_space.get_fiber(base_point)
        if fiber is not None and formula_id not in fiber.negligible_residuals:
            fiber.negligible_residuals.append(formula_id)
