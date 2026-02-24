"""
Axiom 智能分区模块 - 分区定义

先学单一场景，再整合到复杂场景。分区按 domain、condition、complexity 定义。
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
import numpy as np


@dataclass
class Partition:
    """
    分区定义：id, domain, condition, complexity.
    condition(x) -> bool mask 或 indices，判断样本是否属于该分区。
    """

    id: str
    domain: str = "generic"
    condition: Optional[Callable] = None  # (X, y?) -> mask or indices
    complexity: int = 1  # 1=简单, 2=中等, 3=复杂，用于 curriculum
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mask(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """返回布尔 mask，True 表示样本属于该分区。"""
        if self.condition is None:
            return np.ones(len(X), dtype=bool)
        out = self.condition(X, y)
        if isinstance(out, np.ndarray) and out.dtype == bool:
            return out
        if isinstance(out, (list, np.ndarray)) and len(out) > 0:
            mask = np.zeros(len(X), dtype=bool)
            mask[np.asarray(out)] = True
            return mask
        return np.ones(len(X), dtype=bool)

    def indices(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """返回属于该分区的样本索引。"""
        m = self.mask(X, y)
        return np.where(m)[0]


# ---------------------------------------------------------------------------
# RAR 分区：按 log10(g_bar) 划分低/中/高加速度
# ---------------------------------------------------------------------------

def _rar_low(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """g_bar < 1000 (log10 < 3)"""
    g_bar = X[:, 0] if X.ndim > 1 else X
    return g_bar < 1000


def _rar_mid(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """1000 <= g_bar < 10000"""
    g_bar = X[:, 0] if X.ndim > 1 else X
    return (g_bar >= 1000) & (g_bar < 10000)


def _rar_high(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """g_bar >= 10000"""
    g_bar = X[:, 0] if X.ndim > 1 else X
    return g_bar >= 10000


RAR_PARTITIONS: List[Partition] = [
    Partition(
        id="rar_low",
        domain="mechanics",
        condition=_rar_low,
        complexity=1,
        metadata={"g_bar_max": 1000, "regime": "deep_MOND"},
    ),
    Partition(
        id="rar_mid",
        domain="mechanics",
        condition=_rar_mid,
        complexity=2,
        metadata={"g_bar_min": 1000, "g_bar_max": 10000, "regime": "transition"},
    ),
    Partition(
        id="rar_high",
        domain="mechanics",
        condition=_rar_high,
        complexity=1,
        metadata={"g_bar_min": 10000, "regime": "Newtonian"},
    ),
]


# ---------------------------------------------------------------------------
# Battery 分区：按 cycle 归一化 (0-1) 划分早/中/晚
# ---------------------------------------------------------------------------

def _battery_early(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """t < 0.3, 早期老化"""
    t = X[:, 0] if X.ndim > 1 else X
    return t < 0.3


def _battery_mid(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """0.3 <= t < 0.7"""
    t = X[:, 0] if X.ndim > 1 else X
    return (t >= 0.3) & (t < 0.7)


def _battery_late(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """t >= 0.7, 晚期老化"""
    t = X[:, 0] if X.ndim > 1 else X
    return t >= 0.7


BATTERY_PARTITIONS: List[Partition] = [
    Partition(id="battery_early", domain="battery", condition=_battery_early, complexity=1, metadata={"t_max": 0.3}),
    Partition(id="battery_mid", domain="battery", condition=_battery_mid, complexity=2, metadata={"t_min": 0.3, "t_max": 0.7}),
    Partition(id="battery_late", domain="battery", condition=_battery_late, complexity=1, metadata={"t_min": 0.7}),
]


# ---------------------------------------------------------------------------
# Turbulence 分区：按 z (高度) 或 wind magnitude
# ---------------------------------------------------------------------------

def _turbulence_low_z(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """z < 0.33, 低空"""
    z = X[:, 3] if X.shape[1] >= 4 else X[:, 0]
    return z < 0.33


def _turbulence_mid_z(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """0.33 <= z < 0.67"""
    z = X[:, 3] if X.shape[1] >= 4 else X[:, 0]
    return (z >= 0.33) & (z < 0.67)


def _turbulence_high_z(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """z >= 0.67"""
    z = X[:, 3] if X.shape[1] >= 4 else X[:, 0]
    return z >= 0.67


TURBULENCE_PARTITIONS: List[Partition] = [
    Partition(id="turb_z_low", domain="fluids", condition=_turbulence_low_z, complexity=1, metadata={"z_max": 0.33}),
    Partition(id="turb_z_mid", domain="fluids", condition=_turbulence_mid_z, complexity=2, metadata={"z_min": 0.33, "z_max": 0.67}),
    Partition(id="turb_z_high", domain="fluids", condition=_turbulence_high_z, complexity=1, metadata={"z_min": 0.67}),
]


# ---------------------------------------------------------------------------
# 跨域分区注册表
# ---------------------------------------------------------------------------

PARTITION_REGISTRY: Dict[str, List[Partition]] = {
    "mechanics": RAR_PARTITIONS,
    "battery": BATTERY_PARTITIONS,
    "fluids": TURBULENCE_PARTITIONS,
    "rar": RAR_PARTITIONS,
}


def get_partitions_by_domain(domain: str) -> List[Partition]:
    """按 domain 获取分区列表。"""
    if domain in PARTITION_REGISTRY:
        return PARTITION_REGISTRY[domain]
    return [p for p in RAR_PARTITIONS if p.domain == domain]


def get_partitions_curriculum_order(domain: str = "mechanics") -> List[Partition]:
    """按 complexity 升序返回分区，用于 curriculum 学习。"""
    parts = get_partitions_by_domain(domain)
    return sorted(parts, key=lambda p: p.complexity)


def list_domains() -> List[str]:
    """列出所有支持分区的 domain。"""
    return list(PARTITION_REGISTRY.keys())
