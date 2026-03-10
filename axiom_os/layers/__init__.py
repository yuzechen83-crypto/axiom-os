"""
Axiom-OS Layers: RCLN 2.0 - Evolution Paths

Four Evolution Schemes for Physics-AI:
    1. FNO-RCLN      (Operator)    - Resolution-invariant operators
    2. Clifford-RCLN (Geometric)   - O(3) equivariant algebra
    3. KAN-RCLN      (Symbolic)    - Direct formula extraction
    4. Mamba-RCLN    (Sequential)  - Linear complexity SSM

NO MLP FALLBACK - Choose your path explicitly!
"""

# Core RCLN 2.0
from .rcln_v2 import (
    RCLNLayer,
    create_rcln,
    create_fno_rcln,
    create_clifford_rcln,
    create_kan_rcln,
    create_mamba_rcln,
    DiscoveryHotspot,
    ActivityMonitor,
    SoftShellRegistry,
)

# Evolution Path 1: FNO (Operator)
from .fno import (
    SpectralConv2d,
    FNO2d,
    SpectralConv3d,
    FNO3d,
)
from .spectral import (
    SpectralConv1d,
)

# Evolution Path 2: Clifford (Geometric)
from .clifford_nn import (
    CliffordLinear,
    EquivariantCliffordLinear,
    CliffordActivation,
)
from .clifford_transformer import (
    CliffordSelfAttention,
    CliffordTransformerSoftShell,
)

# Evolution Path 3: KAN (Symbolic)
from .kan_layer import (
    KANSoftShell,
    KANLayer,
    SplineActivation,
    KANFormulaExtractor,
    create_kan_rcln as _create_kan,
)

# Evolution Path 4: Mamba (Sequential)
from .mamba_layer import (
    MambaSoftShell,
    SSMBlock,
    MambaRCLNCell,
    create_mamba_rcln as _create_mamba,
)

# Legacy: Spectral
from .spectral import (
    SpectralConv1d as LegacySpectralConv1d,
    SpectralSoftShell as LegacySpectralSoftShell,
)

# Other layers (Holographic, TBNN, etc.)
from .holographic import (
    HolographicProjectionLayer,
    HolographicFNO,
    HolographicRCLN,
    projection_kernel,
)
from .tensor_net import (
    HolographicTensorNet,
    MERASoftShell,
)
from .meta_kernel import (
    MetaProjectionLayer,
    MetaProjectionModel,
)
from .pinn_lstm import (
    PhysicsInformedLSTM,
    LSTMSoftShell,
    pde_residual_loss_temporal,
)
from .tbnn import (
    TBNN,
    stack_tensor_basis,
)
from .perturbation_gate import LearnablePerturbationGate

__all__ = [
    # Core RCLN 2.0
    "RCLNLayer",
    "create_rcln",
    "create_fno_rcln",
    "create_clifford_rcln",
    "create_kan_rcln",
    "create_mamba_rcln",
    "DiscoveryHotspot",
    "ActivityMonitor",
    "SoftShellRegistry",
    
    # Evolution Path 1: FNO (Operator)
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConv3d",
    "FNO2d",
    "FNO3d",
    
    # Evolution Path 2: Clifford (Geometric)
    "CliffordLinear",
    "EquivariantCliffordLinear",
    "CliffordActivation",
    "CliffordSelfAttention",
    "CliffordTransformerSoftShell",
    
    # Evolution Path 3: KAN (Symbolic)
    "KANSoftShell",
    "KANLayer",
    "SplineActivation",
    "KANFormulaExtractor",
    
    # Evolution Path 4: Mamba (Sequential)
    "MambaSoftShell",
    "SSMBlock",
    "MambaRCLNCell",
    
    # Legacy
    "LegacySpectralConv1d",
    "LegacySpectralSoftShell",
    
    # Holographic
    "HolographicProjectionLayer",
    "HolographicFNO",
    "HolographicRCLN",
    "projection_kernel",
    "HolographicTensorNet",
    "MERASoftShell",
    
    # Meta & Temporal
    "MetaProjectionLayer",
    "MetaProjectionModel",
    "PhysicsInformedLSTM",
    "LSTMSoftShell",
    "pde_residual_loss_temporal",
    
    # Tensor
    "TBNN",
    "stack_tensor_basis",
    "LearnablePerturbationGate",
]


def list_evolution_paths():
    """Print available evolution paths with descriptions."""
    print("=" * 60)
    print("Axiom-OS RCLN 2.0 - Evolution Paths")
    print("=" * 60)
    print()
    paths = SoftShellRegistry.list_paths()
    for i, (key, desc) in enumerate(paths.items(), 1):
        print(f"{i}. {key.upper()}")
        print(f"   {desc}")
        print()
    print("=" * 60)
    print("Usage: RCLNLayer(..., net_type='<path>')")
    print("=" * 60)
