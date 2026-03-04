"""
元轴丛场 (Meta Axis Bundle Field)
海马体核心结构：基空间 + 纤维 + 联络 + 截面 + 连锁反应

- 基空间：Partition × Domain × Regime
- 纤维：主公式 + 扰动修正 + 可忽略集
- 联络：公式间修正链
- 截面：查询 → 公式组合
- 连锁反应：结晶时更新联络
"""

from typing import Optional

from .axes import BasePoint, AxisIndex, Regime, ResidualRole
from .fiber import Fiber, FormulaEntry
from .connection import Connection, ConnectionType, ConnectionEdge
from .base import BaseSpace
from .section import Section
from .chain_reaction import ChainReaction


class MetaAxisBundleField:
    """
    元轴丛场：海马体的核心结构
    """
    def __init__(self):
        self.base_space = BaseSpace()
        self.section = Section(self.base_space)
        self.chain_reaction = ChainReaction(self.base_space)

    def crystallize(
        self,
        formula: str,
        callable_fn,
        output_dim: int,
        domain: str,
        partition_id: str,
        regime: str = "weak",
        formula_id: Optional[str] = None,
        residual_role: str = "principal",
    ) -> str:
        """
        结晶：将公式加入纤维，触发连锁反应。
        返回 formula_id。
        """
        from .axes import Regime, ResidualRole
        try:
            r = Regime(regime)
        except ValueError:
            r = Regime.WEAK
        try:
            rr = ResidualRole(residual_role)
        except ValueError:
            rr = ResidualRole.PRINCIPAL

        if formula_id is None:
            formula_id = f"law_{len(self.base_space._fibers)}_{domain}_{partition_id}"

        bp = BasePoint(domain=domain, partition_id=partition_id, regime=r)
        self.chain_reaction.on_crystallize(
            formula_id=formula_id,
            formula=formula,
            callable_fn=callable_fn,
            output_dim=output_dim,
            base_point=bp,
            residual_role=rr,
        )
        return formula_id

    def select(
        self,
        x,
        domain: str,
        task: str = "predict",
        regime_hint: Optional[str] = None,
    ):
        """截面选择：返回 (fiber, base_point, triggered_ids)"""
        r = None
        if regime_hint:
            try:
                r = Regime(regime_hint)
            except ValueError:
                pass
        return self.section.select(x, domain, task, r)

    def eval_perturbation(
        self,
        x,
        partition_id: Optional[str] = None,
        domain: Optional[str] = None,
        alpha_pert: float = 0.1,
    ):
        """
        扰动求值：兼容 Hippocampus.eval_perturbation 接口。
        仅返回扰动项之和（不包含 principal），用于 RCLN 的联想/直觉叠加。
        返回 (n, output_dim) 或 None。
        """
        dom = domain or "mechanics"
        fiber, bp, _ = self.select(x, dom)
        if fiber is None or not fiber.perturbations:
            return None
        return fiber.eval_perturbation_only(x, alpha_pert=alpha_pert)


__all__ = [
    "MetaAxisBundleField",
    "BasePoint",
    "AxisIndex",
    "Regime",
    "ResidualRole",
    "Fiber",
    "FormulaEntry",
    "Connection",
    "ConnectionType",
    "ConnectionEdge",
    "BaseSpace",
    "Section",
    "ChainReaction",
]
