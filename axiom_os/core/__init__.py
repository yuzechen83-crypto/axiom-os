"""
Axiom-OS Core: Universal Physical Interface, Knowledge, Einstein, Hippocampus
"""

from .upi import UPIState, Units, DimensionError
from .knowledge import SymplecticLawRegistry
from .einstein import (
    SymplecticIntegrator,
    EinsteinCore,
    MetriplecticStructure,
    metriplectic_from_H,
)
from .generic import GENERICSystem
from .hippocampus import Hippocampus
from .optimizer import BreathingScheduler, BreathingOptimizer
from .adaptive_hard_core import wrap_adaptive_hard_core, gate_extremeness
from .wind_hard_core import (
    make_wind_hard_core,
    make_wind_hard_core_enhanced,
    make_wind_hard_core_adaptive,
)
from .clifford_ops import (
    geometric_product,
    multivector_magnitude,
    rotate_multivector,
    CliffordTensor,
    CAYLEY,
    N_BLADES,
)
from .imagination import couple_theories, leapfrog_coupled, simulate_coupled
from .task import Task, Goal, TaskStatus, TaskType, create_goal
from .symplectic_causal import (
    get_symplectic_causal_edges,
    get_symplectic_allowed_inputs,
    filter_formula_by_symplectic,
    build_symplectic_causal_mask,
)
from .light_cone_filter import check_light_cone, filter_causal_edges_by_light_cone
from .causal_constraints import allowed_edges, allowed_inputs_for_output
from .perturbation import infer_partition_id, infer_partition_weights
from .alpha_scheduler import (
    compute_alpha_free_energy,
    compute_alpha_system12,
    compute_alpha_hybrid,
)
from .turbulence_scale import TurbulencePhysicalScale, TurbulenceScales
from .turbulence_invariants import (
    grad_u_from_velocity,
    decompose_grad_u,
    compute_invariants,
    compute_tensor_basis,
    extract_invariants_and_basis,
)
from .bundle_field import MetaAxisBundleField
from .partition import (
    Partition,
    RAR_PARTITIONS,
    BATTERY_PARTITIONS,
    TURBULENCE_PARTITIONS,
    PARTITION_REGISTRY,
    get_partitions_by_domain,
    get_partitions_curriculum_order,
    list_domains,
)
from .partition_learner import learn_curriculum, learn_partition
from .partition_integrator import integrate_hard_switch, integrate_soft_gate
from .features import (
    DerivativeMethod,
    build_pde_library,
    VectorFeatureExtractor,
    build_vorticity_library,
    sparse_regression_pde,
)
from .features_2d import (
    grad_x,
    grad_y,
    laplacian,
    advection,
    vorticity_from_velocity,
    weak_form_patch_average,
)

__all__ = [
    "wrap_adaptive_hard_core",
    "make_wind_hard_core",
    "make_wind_hard_core_enhanced",
    "make_wind_hard_core_adaptive",
    "gate_extremeness",
    "UPIState",
    "Units",
    "DimensionError",
    "SymplecticLawRegistry",
    "SymplecticIntegrator",
    "EinsteinCore",
    "MetriplecticStructure",
    "metriplectic_from_H",
    "GENERICSystem",
    "Hippocampus",
    "MetaAxisBundleField",
    "BreathingScheduler",
    "BreathingOptimizer",
    "geometric_product",
    "multivector_magnitude",
    "rotate_multivector",
    "CliffordTensor",
    "CAYLEY",
    "N_BLADES",
    "couple_theories",
    "leapfrog_coupled",
    "simulate_coupled",
    "DerivativeMethod",
    "build_pde_library",
    "VectorFeatureExtractor",
    "build_vorticity_library",
    "sparse_regression_pde",
    "grad_x",
    "grad_y",
    "laplacian",
    "advection",
    "vorticity_from_velocity",
    "weak_form_patch_average",
    "Task",
    "Goal",
    "TaskStatus",
    "TaskType",
    "create_goal",
    "get_symplectic_causal_edges",
    "get_symplectic_allowed_inputs",
    "filter_formula_by_symplectic",
    "build_symplectic_causal_mask",
    "check_light_cone",
    "filter_causal_edges_by_light_cone",
    "allowed_edges",
    "allowed_inputs_for_output",
    "Partition",
    "RAR_PARTITIONS",
    "BATTERY_PARTITIONS",
    "TURBULENCE_PARTITIONS",
    "PARTITION_REGISTRY",
    "get_partitions_by_domain",
    "get_partitions_curriculum_order",
    "list_domains",
    "learn_curriculum",
    "learn_partition",
    "TurbulencePhysicalScale",
    "TurbulenceScales",
    "grad_u_from_velocity",
    "decompose_grad_u",
    "compute_invariants",
    "compute_tensor_basis",
    "extract_invariants_and_basis",
    "integrate_hard_switch",
    "integrate_soft_gate",
    "infer_partition_id",
    "infer_partition_weights",
    "compute_alpha_free_energy",
    "compute_alpha_system12",
    "compute_alpha_hybrid",
]
