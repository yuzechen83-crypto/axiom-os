"""
共享的湍流 Hard Core 实现
PROPRIETARY - Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.
Unauthorized copying, modification, or distribution of this algorithm is prohibited.

供 run_turbulence_full, main_turbulence_3d, main_turbulence_ageostrophic 使用
"""

import numpy as np
import torch

from .adaptive_hard_core import wrap_adaptive_hard_core


def make_wind_hard_core(u_mean: float, v_mean: float):
    """Physical Hard Core: mean wind + log-height profile (boundary layer)."""
    def hard_core(x):
        if isinstance(x, torch.Tensor):
            v = x.float()
        else:
            v = torch.as_tensor(x, dtype=torch.float32)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        z = v[:, 3:4].clamp(1e-6, 1.0)
        profile = 1.0 + 0.4 * torch.log1p(z)
        u_hard = u_mean * profile
        v_hard = v_mean * profile
        return torch.cat([u_hard, v_hard], dim=1)
    return hard_core


def make_wind_hard_core_enhanced(u_mean: float, v_mean: float, use_time_mod: bool = True):
    """
    SPNN-inspired: 时间调制 + PhysicalMapping 尺度。
    时间调制：弱日变化 sin(2πt) 缓解 u 残差的时间系统性偏差。
    """
    base = make_wind_hard_core(u_mean, v_mean)

    def enhanced(x):
        y_base = base(x)
        if not use_time_mod:
            return y_base
        if isinstance(x, torch.Tensor):
            v = x.float()
        else:
            v = torch.as_tensor(x, dtype=torch.float32)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        t = v[:, 0:1].clamp(0, 1)
        z = v[:, 3:4].clamp(1e-6, 1.0)
        t_mod_u = 1.0 + 0.08 * torch.sin(2 * np.pi * t)
        t_mod_v = 1.0 + 0.04 * torch.sin(2 * np.pi * t + 0.5)
        z_scale = 1.0 + 0.15 * z
        scale_u = t_mod_u * z_scale
        scale_v = t_mod_v * z_scale
        return torch.cat([y_base[:, 0:1] * scale_u, y_base[:, 1:2] * scale_v], dim=1)

    return enhanced


def make_wind_hard_core_adaptive(
    u_mean: float,
    v_mean: float,
    threshold: float = 5.0,
    use_enhanced: bool = True,
):
    """Adaptive: looser in extreme wind, keeps Soft Shell flexible."""
    base = make_wind_hard_core_enhanced(u_mean, v_mean) if use_enhanced else make_wind_hard_core(u_mean, v_mean)
    return wrap_adaptive_hard_core(base, threshold=threshold, steepness=2.0)
