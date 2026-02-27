"""
元轴丛场 - 基空间
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

BaseSpace: Partition × Domain × Regime 的离散化，每点对应一个 Fiber
"""

from typing import Dict, List, Optional

from .axes import BasePoint, AxisIndex, Regime
from .fiber import Fiber
from .connection import Connection


class BaseSpace:
    """
    基空间：Partition × Domain × Regime 的离散化
    每个点对应一个 Fiber
    """
    def __init__(self):
        self._fibers: Dict[str, Fiber] = {}
        self._connection = Connection()

    def _key(self, bp: BasePoint) -> str:
        return f"{bp.domain}:{bp.partition_id}:{bp.regime.value}"

    @property
    def connection(self) -> Connection:
        return self._connection

    def get_fiber(self, base_point: BasePoint) -> Optional[Fiber]:
        k = self._key(base_point)
        return self._fibers.get(k)

    def set_fiber(self, base_point: BasePoint, fiber: Fiber) -> None:
        k = self._key(base_point)
        self._fibers[k] = fiber

    def get_fiber_by_domain(self, domain: str) -> Optional[Fiber]:
        """同 domain 下返回第一个可用 fiber（fallback 用）"""
        for k, fib in self._fibers.items():
            if k.startswith(f"{domain}:"):
                return fib
        return None

    def list_base_points(self, axis_index: Optional[AxisIndex] = None) -> List[BasePoint]:
        """按轴索引过滤返回基空间点"""
        points = []
        seen = set()
        for k in self._fibers:
            parts = k.split(":")
            if len(parts) >= 3:
                domain, partition_id, regime_str = parts[0], parts[1], parts[2]
                try:
                    regime = Regime(regime_str)
                except ValueError:
                    regime = Regime.WEAK
                bp = BasePoint(domain=domain, partition_id=partition_id, regime=regime)
                if bp in seen:
                    continue
                seen.add(bp)
                if axis_index:
                    if axis_index.domain and axis_index.domain != domain:
                        continue
                    if axis_index.partition_id and axis_index.partition_id != partition_id:
                        continue
                    if axis_index.regime and axis_index.regime != regime:
                        continue
                points.append(bp)
        return points
