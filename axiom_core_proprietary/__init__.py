"""
axiom_core_proprietary - 核心算法私有包
PROPRIETARY - Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.
需授权获取。请联系 yuzechen83-crypto。
"""

from .discovery import DiscoveryEngine
from .hippocampus import Hippocampus
from .coach import coach_score, coach_loss_torch, coach_score_batch

__all__ = [
    "RCLNLayer", "DiscoveryHotspot", "ActivityMonitor",
    "SpectralConv1d", "SpectralSoftShell", "HAS_CLIFFORD",
    "DiscoveryEngine", "Hippocampus",
    "coach_score", "coach_loss_torch", "coach_score_batch",
]


def __getattr__(name):
    """Lazy load rcln to avoid circular import with axiom_os.layers"""
    if name in ("RCLNLayer", "DiscoveryHotspot", "ActivityMonitor", "SpectralConv1d", "SpectralSoftShell", "HAS_CLIFFORD"):
        from .rcln import RCLNLayer, DiscoveryHotspot, ActivityMonitor, SpectralConv1d, SpectralSoftShell, HAS_CLIFFORD
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
