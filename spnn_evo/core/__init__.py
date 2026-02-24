"""SPNN-Evo Core: UPI, Hippocampus, PDE Features, Einstein, Imagination, Knowledge"""

from .upi import UPIState, Units
from .hippocampus import HippocampusLibrary
from .knowledge import SymplecticLaw

try:
    from .hippocampus import EinsteinCore
except ImportError:
    EinsteinCore = None

try:
    from .einstein import helmholtz_decomposition, SymplecticIntegrator
except ImportError:
    helmholtz_decomposition = None
    SymplecticIntegrator = None

from . import imagination
from .features import (
    build_pde_library,
    build_vorticity_library,
    compute_derivatives_2d,
    sparse_regression_pde,
    VectorFeatureExtractor,
)
from .features_2d import (
    calc_2d_derivatives,
    weak_form_patch_average,
    grad_x,
    grad_y,
    laplacian,
    advection,
    vorticity_from_velocity,
    gaussian_smooth,
)
