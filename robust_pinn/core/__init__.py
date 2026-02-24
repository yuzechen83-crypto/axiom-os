"""Robust PINN Core: Architecture, Physics, Boundary"""

from .arch import PhysicsResNet, SoftMeltLayer
from .physics import HamiltonianSystem, compute_pde_residual, compute_wave_residual
from .boundary import GhostPointHandler
