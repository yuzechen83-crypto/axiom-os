"""
元轴丛场 - 截面
Section: 给定查询 (x, task, domain) → 选出公式组合
实现「有益的连锁反应」：检索 → 触发相关 → 组合 → 输出
"""

from typing import Tuple, Optional, List

import numpy as np

from .base import BaseSpace
from .fiber import Fiber
from .axes import BasePoint, Regime


def _infer_partition_id(x: np.ndarray, domain: str) -> Optional[str]:
    """推断分区 id，避免循环导入"""
    try:
        from axiom_os.core.perturbation import infer_partition_id
        return infer_partition_id(x, domain)
    except ImportError:
        return None


class Section:
    """
    截面：给定查询 (x, task, domain) → 选出公式组合
    """
    def __init__(self, base_space: BaseSpace):
        self.base_space = base_space

    def select(
        self,
        x: np.ndarray,
        domain: str,
        task: str = "predict",
        regime_hint: Optional[Regime] = None,
    ) -> Tuple[Optional[Fiber], Optional[BasePoint], List[str]]:
        """
        选择截面：返回 (fiber, base_point, triggered_ids)
        triggered_ids: 联络触发的扰动公式 id 列表
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        partition_id = _infer_partition_id(x_arr, domain)
        if partition_id is None:
            partition_id = "default"
        regime = regime_hint or Regime.WEAK
        bp = BasePoint(domain=domain, partition_id=partition_id, regime=regime)

        fiber = self.base_space.get_fiber(bp)
        if fiber is None:
            fiber, bp = self._fallback(bp)

        triggered: List[str] = []
        if fiber is not None and fiber.principal is not None:
            triggered = self.base_space.connection.get_perturbation_chain(
                fiber.principal.formula_id
            )

        return fiber, bp, triggered

    def _fallback(self, bp: BasePoint) -> Tuple[Optional[Fiber], BasePoint]:
        """最近邻 fallback：同 domain 下任意 partition"""
        fib = self.base_space.get_fiber_by_domain(bp.domain)
        if fib is not None:
            return fib, fib.base_point
        return None, bp
