"""
SPNN-Opt-Rev5 Robust PINN
Industrial-grade Physics-Informed Neural Network
No Symplectic, No Leapfrog. ResNet + Physics Loss.
"""

from .config import DTYPE
from .core.arch import PhysicsResNet, SoftMeltLayer
from .core.physics import HamiltonianSystem, compute_pde_residual, compute_wave_residual
from .core.boundary import GhostPointHandler
