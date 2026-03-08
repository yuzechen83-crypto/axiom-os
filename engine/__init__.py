"""
Axiom-OS Engine: Symbolic Regression, Diffusion, Flow Matching, Topology
"""

from .discovery import DiscoveryEngine
from .generic_loss import generic_loss, generic_loss_from_aux
from .discovery_generic import discover_potentials, discover_generic
from .forms import (
    FormCandidate,
    ExpForm,
    PowerForm,
    LogForm,
    PiecewiseLinearForm,
    PolyForm,
    BATTERY_AGING_FORMS,
)
from .diffusion import ScoreNet, sinusoidal_embedding
from .sde_solver import DiffusionSDE, create_diffusion_sde
from .sampler import sample_with_physics, sample_with_physics_residual
from .probabilistic import probabilistic_predict
from .gflownet import GFlowNetDiscovery, EquationState
from .flow_matching import (
    VectorFieldNet,
    MetaFlowMatching,
    conditional_flow_matching_loss,
    sample_flow_matching,
)
from .topology import (
    compute_persistence,
    TopologicalEarlyWarning,
)
from .discovery_check import (
    discovery_check_radial_profile,
    discovery_check_kernel_shape,
)
from .fine_tune import fine_tune_constant, fine_tune_rar

__all__ = [
    "DiscoveryEngine",
    "generic_loss",
    "generic_loss_from_aux",
    "discover_potentials",
    "discover_generic",
    "FormCandidate",
    "ExpForm",
    "PowerForm",
    "LogForm",
    "PiecewiseLinearForm",
    "PolyForm",
    "BATTERY_AGING_FORMS",
    "ScoreNet",
    "sinusoidal_embedding",
    "DiffusionSDE",
    "create_diffusion_sde",
    "sample_with_physics",
    "sample_with_physics_residual",
    "probabilistic_predict",
    "GFlowNetDiscovery",
    "EquationState",
    "VectorFieldNet",
    "MetaFlowMatching",
    "conditional_flow_matching_loss",
    "sample_flow_matching",
    "compute_persistence",
    "TopologicalEarlyWarning",
    "discovery_check_radial_profile",
    "discovery_check_kernel_shape",
    "fine_tune_constant",
    "fine_tune_rar",
]
