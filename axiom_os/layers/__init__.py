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
from .fno import SpectralConv2d, FNO2d, SpectralConv3d, FNO3d
from .clifford_nn import CliffordLinear, EquivariantCliffordLinear, CliffordActivation
from .holographic import HolographicProjectionLayer, HolographicFNO, HolographicRCLN, projection_kernel
from .clifford_transformer import CliffordSelfAttention, CliffordTransformerSoftShell
from .tensor_net import HolographicTensorNet, MERASoftShell
from .meta_kernel import MetaProjectionLayer, MetaProjectionModel
from .pinn_lstm import PhysicsInformedLSTM, LSTMSoftShell, pde_residual_loss_temporal
from .tbnn import TBNN, stack_tensor_basis

__all__ = [
    "RCLNLayer",
    "DiscoveryHotspot",
    "ActivityMonitor",
    "HAS_CLIFFORD",
    "SpectralConv1d",
    "SpectralSoftShell",
    "SpectralConv2d",
    "FNO2d",
    "SpectralConv3d",
    "FNO3d",
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
    "PhysicsInformedLSTM",
    "LSTMSoftShell",
    "pde_residual_loss_temporal",
    "TBNN",
    "stack_tensor_basis",
]
