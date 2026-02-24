"""
Turbulence Physical Scale System (SPNN-inspired)
物理标尺：从 (t,x,y,z,u,v) 自动检测特征尺度，支持 PhysicalMapping。
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TurbulenceScales:
    """湍流特征尺度"""
    t_c: float = 1.0      # 时间尺度 (h 或 norm)
    x_c: float = 1.0      # 经度尺度 (deg)
    y_c: float = 1.0      # 纬度尺度 (deg)
    z_c: float = 1.0      # 高度尺度 (m)
    v_c: float = 1.0      # 风速尺度 (m/s)


class TurbulencePhysicalScale:
    """
    湍流物理标尺系统（参考 SPNN PhysicalScaleSystem）
    自动检测 (t,x,y,z) 和 (u,v) 的特征尺度，用于物理锚定与 PhysicalMapping。
    """

    def __init__(self):
        self.scales = TurbulenceScales()

    def auto_detect(
        self,
        coords: np.ndarray,
        targets: np.ndarray,
    ) -> TurbulenceScales:
        """
        从 coords (N,4)=(t,x,y,z) 和 targets (N,2)=(u,v) 自动检测特征尺度。
        coords 已归一化 [0,1] 时，t_c=x_c=y_c=z_c=1；v_c 从 targets 估计。
        """
        coords = np.asarray(coords, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)

        # 时间尺度：若 t 已归一化，t_c=1；否则用 t 的 std
        t_std = np.std(coords[:, 0]) if coords.shape[1] >= 1 else 1.0
        t_c = max(t_std, 1e-10) if t_std > 0.01 else 1.0

        # 空间尺度：x,y,z 已归一化则 std 约 0.3
        x_c = max(np.std(coords[:, 1]) if coords.shape[1] >= 2 else 1.0, 1e-10)
        y_c = max(np.std(coords[:, 2]) if coords.shape[1] >= 3 else 1.0, 1e-10)
        z_c = max(np.std(coords[:, 3]) if coords.shape[1] >= 4 else 1.0, 1e-10)
        if x_c < 0.01:
            x_c = 1.0
        if y_c < 0.01:
            y_c = 1.0
        if z_c < 0.01:
            z_c = 1.0

        # 风速尺度：|V| 的 std 或 median
        if targets.shape[1] >= 2:
            v_mag = np.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2)
            v_c = float(np.median(v_mag) + np.std(v_mag) + 1e-6)
            v_c = max(v_c, 1.0)
        else:
            v_c = max(np.std(targets) + 1e-6, 1.0)

        self.scales = TurbulenceScales(t_c=t_c, x_c=x_c, y_c=y_c, z_c=z_c, v_c=v_c)
        return self.scales

    def get_scale_factors(
        self,
        z_norm: np.ndarray,
        t_norm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        根据 z, t 计算 PhysicalMapping 的 scale_factors。
        M_phys = diag(scale_factors)，用于不同高度/时间的尺度映射。
        高层 z→1 时 scale 略增大（边界层风速随高度增加）。
        """
        z = np.asarray(z_norm).flatten()
        if t_norm is not None:
            t = np.asarray(t_norm).flatten()
        else:
            t = np.ones_like(z)
        # 简单模型：z 越高 scale 略大 (1 + 0.2*z)，t 的日变化 (1 + 0.1*sin(2πt))
        scale = 1.0 + 0.2 * z + 0.05 * np.sin(2 * np.pi * t)
        return scale.astype(np.float32)
