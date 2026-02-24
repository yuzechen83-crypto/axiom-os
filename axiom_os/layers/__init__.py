"""
Axiom-OS Layers: RCLN, Holographic (MASHD)
"""

from .rcln import (
    RCLNLayer,
    HAS_CLIFFORD,
    SpectralConv1d,
    SpectralSoftShell,
    DiscoveryHotspot,
    ActivityMonitor,
)
from .fno import SpectralConv2d, FNO2d
from .clifford_nn import CliffordLinear, EquivariantCliffordLinear, CliffordActivation
from .holographic import HolographicProjectionLayer, HolographicFNO, HolographicRCLN, projection_kernel
from .clifford_transformer import CliffordSelfAttention, CliffordTransformerSoftShell
from .tensor_net import HolographicTensorNet, MERASoftShell
from .meta_kernel import MetaProjectionLayer, MetaProjectionModel

__all__ = [
    "RCLNLayer",
    "DiscoveryHotspot",
    "ActivityMonitor",
    "HAS_CLIFFORD",
    "SpectralConv1d",
    "SpectralSoftShell",
    "SpectralConv2d",
    "FNO2d",
    "CliffordLinear",
    "EquivariantCliffordLinear",
    "CliffordActivation",
    "HolographicProjectionLayer",
    "HolographicFNO",
    "HolographicRCLN",
    "projection_kernel",
    "CliffordSelfAttention",
    "CliffordTransformerSoftShell",
    "HolographicTensorNet",
    "MERASoftShell",
    "MetaProjectionLayer",
    "MetaProjectionModel",
]
