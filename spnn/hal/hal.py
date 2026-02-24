"""
Hardware Abstraction Layer ℋ
设备管理、精度控制、资源监控
"""

import torch
from typing import Optional, Dict, Any


class HAL:
    """
    硬件抽象层：统一设备与精度管理
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        prefer_cuda: bool = True,
    ):
        if device is None:
            device = torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")
        self.device = device
        self.dtype = dtype
        self._resource_state: Dict[str, float] = {"cpu": 1.0, "mem": 1.0, "gpu": 1.0}

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype)

    def get_device(self) -> torch.device:
        return self.device

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def update_resource_state(self, state: Dict[str, float]) -> None:
        self._resource_state.update(state)

    def get_resource_state(self) -> Dict[str, float]:
        if self.device.type == "cuda" and torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024**3)
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)
            self._resource_state["gpu_mem_alloc"] = mem_alloc
            self._resource_state["gpu_mem_reserved"] = mem_reserved
        return self._resource_state.copy()
