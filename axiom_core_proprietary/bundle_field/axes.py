"""
元轴丛场 - 轴定义
BasePoint: 基空间中的点 (domain, partition_id, regime)
AxisIndex: 轴索引，用于检索与组织
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Regime(Enum):
    """扰动 regime"""
    WEAK = "weak"       # 弱扰动，一阶修正
    MEDIUM = "medium"   # 中扰动，需切换或高阶
    STRONG = "strong"   # 强扰动，复合场


class ResidualRole(Enum):
    """残差角色"""
    PRINCIPAL = "principal"         # 主项，必须建模
    PERTURBATION = "perturbation"  # 扰动修正
    NEGLIGIBLE = "negligible"      # 可忽略


@dataclass
class BasePoint:
    """基空间中的点：(domain, partition_id, regime)"""
    domain: str
    partition_id: str
    regime: Regime = Regime.WEAK

    def __hash__(self):
        return hash((self.domain, self.partition_id, self.regime.value))

    def __eq__(self, other):
        if not isinstance(other, BasePoint):
            return False
        return (
            self.domain == other.domain
            and self.partition_id == other.partition_id
            and self.regime == other.regime
        )


@dataclass
class AxisIndex:
    """轴索引：用于检索与组织"""
    domain: str
    partition_id: Optional[str] = None
    regime: Optional[Regime] = None
    residual_role: Optional[ResidualRole] = None
