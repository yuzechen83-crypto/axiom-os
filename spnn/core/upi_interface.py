"""
UPI Interface System - Unified Physics Interface
供应商安全契约 (SSC) 与量纲规范
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any

from .constants import EPSILON
from .physical_scale import PhysicalScaleSystem


class UPIInterface:
    """
    UPI: 统一物理接口
    - 输入范围验证 (SSC)
    - 量纲标准化
    - 契约验证
    """

    def __init__(
        self,
        scale_system: Optional[PhysicalScaleSystem] = None,
        input_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        dim_check_freq: int = 1,
    ):
        self.scale_system = scale_system or PhysicalScaleSystem()
        self.input_bounds = input_bounds  # (lower, upper)无量纲
        self.dim_check_freq = dim_check_freq
        self._step_counter = 0

    def validate_input(
        self,
        x: np.ndarray,
        dim_exponents: Optional[np.ndarray] = None,
        ℓ: Optional[np.ndarray] = None,
    ) -> bool:
        """
        输入范围验证 (SSC)
        V_input(x) = ∏_k I(x_k / ℓ_k^{d_k} ∈ [l̃_k, ũ_k])
        """
        x = np.asarray(x)
        if self.input_bounds is None:
            return True
        lower, upper = self.input_bounds
        lower = np.asarray(lower).flatten()
        upper = np.asarray(upper).flatten()
        ℓ = ℓ or np.ones(x.shape[-1])
        ℓ = np.asarray(ℓ).flatten()
        dims = dim_exponents or np.zeros(len(ℓ))

        x_norm = x / (np.power(ℓ, dims) + EPSILON)
        if x_norm.shape[-1] != len(lower):
            lower = np.broadcast_to(lower, x_norm.shape[-1])
            upper = np.broadcast_to(upper, x_norm.shape[-1])
        return bool(np.all(x_norm >= lower - EPSILON) and np.all(x_norm <= upper + EPSILON))

    def standardize_output(self, h: np.ndarray) -> np.ndarray:
        """UPI 标准化输出为无量纲隐藏状态"""
        h = np.asarray(h)
        norm = np.linalg.norm(h, axis=-1, keepdims=True) + EPSILON
        return h / norm

    def verify(
        self,
        x: np.ndarray,
        h: np.ndarray,
        dim_exponents: Optional[np.ndarray] = None,
    ) -> bool:
        """
        UPIVerify: 综合契约验证
        """
        if not self.validate_input(x, dim_exponents):
            return False
        self._step_counter += 1
        if self._step_counter % self.dim_check_freq == 0:
            if not np.all(np.isfinite(h)):
                return False
        return True

    def get_upi_address(self, component_id: str, params: Dict[str, Any]) -> str:
        """
        UPI 地址：用于安全证书
        """
        return f"upi://{component_id}?v=5"
