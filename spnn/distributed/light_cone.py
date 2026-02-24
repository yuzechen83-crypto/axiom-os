"""
Light-Cone Time Coordinator 光锥时间协调器
因果保障: C_i = {(t,x) | -c²(t-t_i)² + ||x-x_i||² ≤ R_i²}
光锥延迟强制保证、因果序严格维护、异步执行同步点
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field

from ..core.constants import c


@dataclass
class LightCone:
    """模块光锥体"""
    t_i: float
    x_i: np.ndarray
    R_i: float
    tau_i: float

    def contains(self, t: float, x: np.ndarray) -> bool:
        """检查 (t, x) 是否在光锥内"""
        dt = t - self.t_i
        dx = np.linalg.norm(np.asarray(x) - np.asarray(self.x_i))
        return -c**2 * dt**2 + dx**2 <= self.R_i**2


class LightConeCoordinator:
    """
    全局时间协调
    Δt_G = min_i Δt_i^stable · BrainFactor(urgency, accuracy)
    schedule_communication: 光锥延迟强制保证
    advance_time: 处理 receive_time <= current_time 的事件
    """

    def __init__(self, c_light: float = c, modules_positions: Optional[Dict[str, np.ndarray]] = None):
        self.c = c_light
        self.cones: List[LightCone] = []
        self.modules = modules_positions or {}
        self.event_queue: List[Dict] = []
        self._deliver_callback: Optional[Callable] = None

    def set_deliver_callback(self, cb: Callable[[Dict], None]) -> None:
        self._deliver_callback = cb

    def schedule_communication(
        self,
        src_module: str,
        dst_module: str,
        data: Any,
        timestamp: float,
    ) -> float:
        """
        计算最小传输时间，调度接收时间
        t_grid_update >= t_switching + d_controller/c
        """
        pos_src = self.modules.get(src_module, np.zeros(3))
        pos_dst = self.modules.get(dst_module, np.zeros(3))
        dx = np.linalg.norm(np.asarray(pos_dst) - np.asarray(pos_src))
        min_transfer_time = dx / (self.c + 1e-12)
        receive_time = timestamp + min_transfer_time
        self.event_queue.append({
            "type": "data_transfer",
            "src": src_module,
            "dst": dst_module,
            "data": data,
            "send_time": timestamp,
            "receive_time": receive_time,
        })
        self.event_queue.sort(key=lambda x: x["receive_time"])
        return receive_time

    def advance_time(self, current_time: float) -> int:
        """处理所有 receive_time <= current_time 的事件"""
        delivered = 0
        while self.event_queue and self.event_queue[0]["receive_time"] <= current_time:
            event = self.event_queue.pop(0)
            if self._deliver_callback:
                self._deliver_callback(event)
            delivered += 1
        return delivered

    def add_cone(self, t_i: float, x_i: np.ndarray, tau_i: float) -> LightCone:
        R_i = self.c * tau_i
        cone = LightCone(t_i=t_i, x_i=np.asarray(x_i), R_i=R_i, tau_i=tau_i)
        self.cones.append(cone)
        return cone

    def compute_sync_dt(
        self,
        stable_dts: List[float],
        urgency: float = 0.5,
        accuracy: float = 0.5,
    ) -> float:
        min_dt = min(stable_dts) if stable_dts else 1.0
        brain_factor = 0.5 + 0.3 * (1 - urgency) + 0.2 * accuracy
        return min_dt * brain_factor

    def check_causal_consistency(self, events: List[Tuple[float, np.ndarray]]) -> bool:
        """因果一致性检查"""
        for i, (t_i, x_i) in enumerate(events):
            for j, (t_j, x_j) in enumerate(events):
                if i >= j:
                    continue
                dt = t_j - t_i
                dx = np.linalg.norm(np.asarray(x_j) - np.asarray(x_i))
                if dt > 0 and dx > self.c * dt:
                    return False
        return True
