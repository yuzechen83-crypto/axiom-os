"""
RCLN 2.0 - Residual Coupler Linking Neuron
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

The Hybrid Engine: Analytical physics (Hard Core) + Neural approximation (Soft Shell).
y_total = y_hard + λ·y_soft

EVOLUTION PATHS (Scheme 1-4):
    ┌─────────────────────────────────────────────────────────────┐
    │  Scheme 1: FNO-RCLN    (Operator Evolution)                 │
    │  └── net_type="fno"                                         │
    │  └── Resolution-invariant, global operators                 │
    │  └── Best: Fluid dynamics, weather, wave propagation        │
    │                                                             │
    │  Scheme 2: Clifford-RCLN (Geometric Evolution)              │
    │  └── net_type="clifford" / "clifford_transformer"           │
    │  └── O(3) equivariant, multivector algebra                  │
    │  └── Best: Robotics, electromagnetism, molecular dynamics   │
    │                                                             │
    │  Scheme 3: KAN-RCLN    (Symbolic Evolution)                 │
    │  └── net_type="kan"                                         │
    │  └── Direct formula extraction, interpretable               │
    │  └── Best: Discovery Engine, control, equation learning     │
    │                                                             │
    │  Scheme 4: Mamba-RCLN  (Sequential Evolution)               │
    │  └── net_type="mamba"                                       │
    │  └── Linear complexity, continuous-time                     │
    │  └── Best: Long sequences, real-time control, ODEs          │
    └─────────────────────────────────────────────────────────────┘

NO MLP FALLBACK - Choose your evolution path explicitly.
"""

from typing import Optional, Callable, Union, NamedTuple, Any, List, Dict
from collections import deque
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hard Core 3.0 - Differentiable Physics
from axiom_os.core.differentiable_physics import (
    DifferentiablePhysicsEngine,
    DifferentiableHardCore,
    SolverLevel,
)

# Evolution Path 1: FNO (Operator)
from .fno import FNO2d, FNO3d

# Evolution Path 2: Clifford (Geometric)
try:
    from cliffordlayers.nn.modules import CliffordLinear
    HAS_CLIFFORD = True
except ImportError:
    HAS_CLIFFORD = False
from .clifford_nn import CliffordLinear as CustomCliffordLinear, CliffordActivation
from .clifford_transformer import CliffordTransformerSoftShell
from .clifford_neural_operator import CliffordNeuralOperator3D

# Evolution Path 3: KAN (Symbolic)
from .kan_layer import KANSoftShell, KANFormulaExtractor

# Evolution Path 4: Mamba (Sequential)
from .mamba_layer import MambaSoftShell

# Legacy support (for specific use cases)
from .spectral import SpectralConv1d, SpectralSoftShell


# =============================================================================
# Activity Monitor & Discovery Hotspot
# =============================================================================

class DiscoveryHotspot(NamedTuple):
    """Flagged when F_soft consistently exceeds threshold over a sliding window."""
    instance_id: str
    avg_soft_magnitude: float
    sample_count: int
    last_input: Optional[torch.Tensor] = None
    last_soft_output: Optional[torch.Tensor] = None


class ActivityMonitor:
    """Monitors |F_soft|. If consistently > threshold over window, flag as Discovery Hotspot."""

    def __init__(self, threshold: float = 0.5, window_size: int = 32):
        self.threshold = threshold
        self.window_size = window_size
        self._magnitudes: deque = deque(maxlen=window_size)
        self._last_input: Optional[torch.Tensor] = None
        self._last_soft: Optional[torch.Tensor] = None
        self.instance_id: str = ""

    def update(self, f_soft: torch.Tensor, x: Optional[torch.Tensor] = None) -> Optional[DiscoveryHotspot]:
        mag = f_soft.detach().abs().mean().item()
        self._magnitudes.append(mag)
        self._last_input = x.detach() if x is not None else self._last_input
        self._last_soft = f_soft.detach()

        if len(self._magnitudes) >= self.window_size:
            avg = sum(self._magnitudes) / len(self._magnitudes)
            if avg > self.threshold:
                return DiscoveryHotspot(
                    instance_id=self.instance_id,
                    avg_soft_magnitude=avg,
                    sample_count=len(self._magnitudes),
                    last_input=self._last_input,
                    last_soft_output=self._last_soft,
                )
        return None

    def reset(self) -> None:
        self._magnitudes.clear()
        self._last_input = None
        self._last_soft = None


# =============================================================================
# Soft Shell Builders - Evolution Paths
# =============================================================================

class SoftShellRegistry:
    """Registry for soft shell architectures."""

    EVOLUTION_PATHS = {
        "fno": "Fourier Neural Operator - Resolution invariant",
        "fno3d": "FNO 3D - Volumetric data",
        "clifford": "Geometric Algebra - O(3) equivariant",
        "clifford_transformer": "Clifford Transformer - Attention + Geometric",
        "clifford_neural_operator": "Clifford Neural Operator - Spectral + Geometric",
        "kan": "Kolmogorov-Arnold Network - Symbolic",
        "mamba": "State Space Model - Linear complexity",
        "spectral": "Spectral Convolution - Frequency domain",
    }

    @classmethod
    def list_paths(cls) -> Dict[str, str]:
        """List all available evolution paths."""
        return cls.EVOLUTION_PATHS.copy()


def build_soft_shell(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    net_type: str,
    **kwargs
) -> nn.Module:
    """
    Build Soft Shell based on evolution path.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        net_type: Evolution path - "fno", "clifford", "kan", "mamba", "spectral"
        **kwargs: Path-specific parameters
    
    Returns:
        Soft shell module
    
    Raises:
        ValueError: If net_type is not recognized
    """
    net_type = net_type.lower()

    # === Scheme 1: FNO-RCLN (Operator Evolution) ===
    if net_type == "fno":
        return FNO2d(
            in_channels=input_dim,
            out_channels=output_dim,
            width=hidden_dim,
            modes1=kwargs.get("fno_modes1", 12),
            modes2=kwargs.get("fno_modes2", 12),
            n_layers=kwargs.get("fno_layers", 4),
        )

    if net_type == "fno3d":
        return FNO3d(
            in_channels=input_dim,
            out_channels=output_dim,
            width=hidden_dim,
            modes1=kwargs.get("fno_modes1", 8),
            modes2=kwargs.get("fno_modes2", 8),
            modes3=kwargs.get("fno_modes3", 8),
            n_layers=kwargs.get("fno_layers", 4),
        )

    # === Scheme 2: Clifford-RCLN (Geometric Evolution) ===
    if net_type == "clifford":
        if HAS_CLIFFORD and kwargs.get("use_builtin_clifford", False):
            # Use external cliffordlayers library
            return _build_builtin_clifford_shell(input_dim, hidden_dim, output_dim)
        # Use custom built-in Clifford
        return _build_custom_clifford_shell(input_dim, hidden_dim, output_dim)

    if net_type == "clifford_transformer":
        return CliffordTransformerSoftShell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("clifford_layers", 2),
        )
    
    if net_type == "clifford_neural_operator":
        return CliffordNeuralOperator3D(
            in_channels=input_dim,
            out_channels=output_dim,
            width=hidden_dim,
            modes1=kwargs.get("cno_modes1", 8),
            modes2=kwargs.get("cno_modes2", 8),
            modes3=kwargs.get("cno_modes3", 8),
            n_layers=kwargs.get("cno_layers", 4),
        )

    # === Scheme 3: KAN-RCLN (Symbolic Evolution) ===
    if net_type == "kan":
        return KANSoftShell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            grid_size=kwargs.get("kan_grid_size", 5),
            spline_order=kwargs.get("kan_spline_order", 3),
            n_layers=kwargs.get("kan_layers", 2),
        )

    # === Scheme 4: Mamba-RCLN (Sequential Evolution) ===
    if net_type == "mamba":
        return MambaSoftShell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=kwargs.get("mamba_layers", 2),
            state_dim=kwargs.get("mamba_state_dim", 16),
            expand_factor=kwargs.get("mamba_expand", 2),
        )

    # === Legacy: Spectral ===
    if net_type == "spectral":
        return SpectralSoftShell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_modes=kwargs.get("spectral_modes", min(input_dim // 2 + 1, 64)),
        )

    raise ValueError(
        f"Unknown evolution path: '{net_type}'.\n"
        f"Available paths: {list(SoftShellRegistry.EVOLUTION_PATHS.keys())}\n"
        f"Choose one explicitly - no MLP fallback!"
    )


def _build_builtin_clifford_shell(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """Build Clifford shell using cliffordlayers library."""
    # 3D Euclidean: 8 blades [s, v1, v2, v3, b12, b23, b31, t]
    N_BLADES = 8

    class BuiltinCliffordShell(nn.Module):
        def __init__(self):
            super().__init__()
            self.cl1 = CliffordLinear((1, 1, 1), in_channels=1, out_channels=hidden_dim)
            self.act1 = CliffordActivation()
            self.cl2 = CliffordLinear((1, 1, 1), in_channels=hidden_dim, out_channels=1)
            self.act2 = CliffordActivation()
            self.proj = nn.Linear(N_BLADES, output_dim)

        def forward(self, x):
            if x.dim() == 2:
                # (B, D) -> (B, 1, n_blades)
                B, D = x.shape
                mv = torch.zeros(B, 1, N_BLADES, device=x.device, dtype=x.dtype)
                mv[:, 0, :min(D, N_BLADES)] = x[:, :min(D, N_BLADES)]
            else:
                mv = x

            mv = self.cl1(mv)
            mv = self.act1(mv)
            mv = self.cl2(mv)
            mv = self.act2(mv)
            return self.proj(mv.squeeze(1))

    return BuiltinCliffordShell()


def _build_custom_clifford_shell(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """Build Clifford shell using custom implementation."""
    N_BLADES = 8  # 3D Euclidean

    class CustomCliffordShell(nn.Module):
        def __init__(self):
            super().__init__()
            self.cl1 = CustomCliffordLinear(1, hidden_dim)
            self.act1 = CliffordActivation()
            self.cl2 = CustomCliffordLinear(hidden_dim, 1)
            self.act2 = CliffordActivation()
            self.proj = nn.Linear(N_BLADES, output_dim)

        def forward(self, x):
            if x.dim() == 2:
                B, D = x.shape
                mv = torch.zeros(B, 1, N_BLADES, device=x.device, dtype=x.dtype)
                mv[:, 0, :min(D, N_BLADES)] = x[:, :min(D, N_BLADES)]
            else:
                mv = x

            mv = self.cl1(mv)
            mv = self.act1(mv)
            mv = self.cl2(mv)
            mv = self.act2(mv)
            return self.proj(mv.squeeze(1))

    return CustomCliffordShell()


# =============================================================================
# RCLN Layer 2.0 - The Hybrid Engine
# =============================================================================

class RCLNLayer(nn.Module):
    """
    Residual Coupler Linking Neuron 2.0
    
    The Hybrid Engine for Physics-AI:
        y_total = y_hard + λ·y_soft
    
    Hard Core (y_hard): Analytical physics equations (domain knowledge)
    Soft Shell (y_soft): Neural approximation (data-driven)
    
    Evolution Paths:
        - FNO: Resolution-invariant operators for spatial fields
        - Clifford: Geometric equivariant for vector physics
        - KAN: Symbolic interpretable for formula discovery
        - Mamba: Linear-complexity for sequential physics
    
    NO MLP FALLBACK - Explicitly choose your evolution path!
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        # Evolution path (REQUIRED - no default, must be first)
        net_type: str,
        # Physics hard core - supports Callable, DifferentiablePhysicsEngine, or DifferentiableHardCore
        hard_core_func: Optional[Union[Callable, DifferentiablePhysicsEngine, DifferentiableHardCore]] = None,
        # Hybrid weights
        lambda_res: float = 1.0,
        # Path-specific parameters
        **kwargs
    ):
        """
        Initialize RCLN Layer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            hard_core_func: Physics function, DifferentiablePhysicsEngine, or DifferentiableHardCore
            lambda_res: Soft shell weight (0 = pure physics, 1 = pure data)
            net_type: Evolution path - "fno", "clifford", "kan", "mamba", "spectral"
            **kwargs: Path-specific parameters (see build_soft_shell)
        
        Raises:
            ValueError: If net_type not specified or invalid
        """
        super().__init__()

        if not net_type:
            raise ValueError(
                "net_type is REQUIRED. Choose your evolution path:\n"
                "  'fno'      - Fourier Neural Operator (fluids, weather)\n"
                "  'clifford' - Geometric Algebra (robotics, EM)\n"
                "  'kan'      - Kolmogorov-Arnold Network (discovery)\n"
                "  'mamba'    - State Space Model (sequences, control)\n"
                "  'spectral' - Spectral Convolution (legacy)\n"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hard_core = hard_core_func
        self.lambda_res = lambda_res
        self._lambda_res_init = lambda_res
        self.net_type = net_type
        
        # Track if using Hard Core 3.0
        self._is_hardcore_v3 = isinstance(hard_core_func, (DifferentiablePhysicsEngine, DifferentiableHardCore))

        # Build soft shell based on evolution path
        self.soft_shell = build_soft_shell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            net_type=net_type,
            **kwargs
        )

        # Activity Monitor (optional)
        self.use_activity_monitor = kwargs.get("use_activity_monitor", False)
        self._activity_monitor: Optional[ActivityMonitor] = None
        if self.use_activity_monitor:
            self._activity_monitor = ActivityMonitor(
                threshold=kwargs.get("soft_threshold", 0.5),
                window_size=kwargs.get("monitor_window", 32),
            )
            self._activity_monitor.instance_id = str(id(self))

        # Hippocampus integration
        self.hippocampus = kwargs.get("hippocampus", None)
        self.alpha_pert = kwargs.get("alpha_pert", 0.0) if self.hippocampus else 0.0
        self.perturb_domain = kwargs.get("perturb_domain", None)

        # Uncertainty quantification
        self.uncertainty_threshold = kwargs.get("uncertainty_threshold", 0.5)
        self._last_y_soft: Optional[torch.Tensor] = None

    def set_lambda_res(self, value: float) -> None:
        """Synaptic: Orchestrator adjusts λ_res (plasticity decay)."""
        self.lambda_res = float(value)

    def set_lambda_decay(self, epoch: int, total_epochs: int, decay_min: float = 0.5) -> None:
        """
        Synaptic λ_res decay: As training progresses, Soft weight increases.
        λ = λ_init * (1 - (1 - decay_min) * epoch / total_epochs)
        """
        if total_epochs <= 0:
            return
        frac = min(1.0, epoch / total_epochs)
        self.lambda_res = self._lambda_res_init * (1.0 - (1.0 - decay_min) * frac)

    def forward(
        self,
        x: Union[torch.Tensor, Any],
        return_hotspot: bool = False,
        return_uncertainty: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (or UPIState)
            return_hotspot: Return DiscoveryHotspot if activity monitor triggers
            return_uncertainty: Return uncertainty flag
        
        Returns:
            y_total or (y_total, hotspot) or (y_total, uncertain)
        """
        # Extract values from UPIState if needed
        if isinstance(x, torch.Tensor):
            x_vals = x.float()
        elif hasattr(x, "values") and callable(getattr(x, "values", None)):
            # UPIState
            x_vals = x.values.float()
        else:
            x_vals = torch.as_tensor(x, dtype=torch.float32)

        if x_vals.dim() == 1:
            x_vals = x_vals.unsqueeze(0)

        # Hard Core: Physics equation (支持 Hard Core 3.0)
        if self.hard_core is not None:
            if isinstance(self.hard_core, (DifferentiablePhysicsEngine, DifferentiableHardCore)):
                # Hard Core 3.0: 可微物理引擎
                y_hard = self.hard_core(x_vals)
            else:
                # 传统的 callable hard_core
                y_hard = self.hard_core(x)
                if not isinstance(y_hard, torch.Tensor):
                    import numpy as np
                    arr = np.asarray(y_hard, dtype=np.float64)
                    y_hard = torch.tensor(arr.copy(), dtype=torch.float32, device=x_vals.device)
            
            if isinstance(y_hard, torch.Tensor):
                y_hard = y_hard.float()
                if y_hard.dim() == 1:
                    y_hard = y_hard.unsqueeze(0)
        else:
            y_hard = None

        # Soft Shell: Neural approximation
        y_soft = self.soft_shell(x_vals)
        self._last_y_soft = y_soft.detach()

        # Combine: y = y_hard + λ·y_soft
        if y_hard is not None:
            y_total = y_hard + self.lambda_res * y_soft
        else:
            y_total = self.lambda_res * y_soft

        # Hippocampus perturbation (if available)
        if self.hippocampus is not None and self.alpha_pert > 0 and self.perturb_domain:
            y_total = self._apply_hippocampus_perturbation(y_total, x_vals, y_soft)

        # Activity Monitor
        hotspot = None
        if return_hotspot and self._activity_monitor is not None:
            hotspot = self._activity_monitor.update(y_soft, x_vals)

        # Uncertainty quantification
        if return_uncertainty:
            uncertain, _ = self.get_uncertainty_status()
            return y_total, uncertain

        if return_hotspot and hotspot is not None:
            return y_total, hotspot

        return y_total

    def _apply_hippocampus_perturbation(
        self,
        y_total: torch.Tensor,
        x_vals: torch.Tensor,
        y_soft: torch.Tensor,
    ) -> torch.Tensor:
        """Apply memory-guided perturbation from hippocampus."""
        try:
            import numpy as np
            from axiom_os.core.perturbation import infer_partition_id

            x_np = x_vals.detach().cpu().numpy()
            pid = infer_partition_id(x_np, self.perturb_domain)
            pert = self.hippocampus.eval_perturbation(
                x_vals, partition_id=pid, domain=self.perturb_domain
            )

            if pert is not None:
                pert_t = torch.as_tensor(pert, dtype=y_total.dtype, device=y_total.device)
                if pert_t.shape != y_total.shape and pert_t.numel() == y_total.numel():
                    pert_t = pert_t.reshape(y_total.shape)

                if pert_t.shape == y_total.shape:
                    activity = float(y_soft.abs().mean().item())
                    # Simple alpha scheduling
                    alpha = self.alpha_pert * min(1.0, activity / 0.5)
                    if alpha > 1e-6:
                        y_total = y_total + alpha * pert_t

        except Exception as e:
            import warnings
            warnings.warn(f"Hippocampus perturbation failed: {e}", UserWarning, stacklevel=2)

        return y_total

    def get_soft_activity(self) -> float:
        """Return mean magnitude of y_soft from last forward pass."""
        if self._last_y_soft is None:
            return 0.0
        return float(self._last_y_soft.abs().mean().item())

    def get_uncertainty_status(self) -> tuple:
        """Returns (uncertain: bool, activity: float)."""
        activity = self.get_soft_activity()
        return (activity > self.uncertainty_threshold, activity)

    def reset_activity_monitor(self) -> None:
        """Reset ActivityMonitor (e.g., after crystallization)."""
        if self._activity_monitor is not None:
            self._activity_monitor.reset()

    def extract_formula(self, var_names: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Extract symbolic formula (KAN mode only).
        
        Returns:
            Dict with formula_str, components, confidence or None
        """
        if self.net_type != "kan":
            return None

        extractor = KANFormulaExtractor(self.soft_shell)
        return extractor.extract_physics_formula(var_names)

    def get_architecture_info(self) -> Dict[str, Any]:
        """Get architecture information."""
        hardcore_type = "none"
        if self.hard_core is not None:
            if isinstance(self.hard_core, DifferentiableHardCore):
                hardcore_type = "differentiable_hardcore_v3"
            elif isinstance(self.hard_core, DifferentiablePhysicsEngine):
                hardcore_type = "differentiable_physics_engine_v3"
            else:
                hardcore_type = "callable"
        
        return {
            "net_type": self.net_type,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "lambda_res": self.lambda_res,
            "has_hard_core": self.hard_core is not None,
            "hard_core_type": hardcore_type,
            "has_hippocampus": self.hippocampus is not None,
            "use_activity_monitor": self.use_activity_monitor,
            "evolution_path": SoftShellRegistry.EVOLUTION_PATHS.get(self.net_type, "Unknown"),
        }

    @classmethod
    def list_evolution_paths(cls) -> Dict[str, str]:
        """List all available evolution paths."""
        return SoftShellRegistry.list_paths()


# =============================================================================
# Factory Functions
# =============================================================================

def create_rcln(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    net_type: str,
    **kwargs
) -> RCLNLayer:
    """
    Factory function for RCLN Layer.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        net_type: Evolution path (fno, clifford, kan, mamba, spectral)
        **kwargs: Additional parameters
    
    Returns:
        RCLNLayer instance
    """
    return RCLNLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        net_type=net_type,
        **kwargs
    )


def create_fno_rcln(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    modes1: int = 12,
    modes2: int = 12,
    **kwargs
) -> RCLNLayer:
    """Create FNO-RCLN (Scheme 1: Operator Evolution)."""
    return create_rcln(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        net_type="fno",
        fno_modes1=modes1,
        fno_modes2=modes2,
        **kwargs
    )


def create_clifford_rcln(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    use_transformer: bool = False,
    **kwargs
) -> RCLNLayer:
    """Create Clifford-RCLN (Scheme 2: Geometric Evolution)."""
    net_type = "clifford_transformer" if use_transformer else "clifford"
    return create_rcln(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        net_type=net_type,
        **kwargs
    )


def create_kan_rcln(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    grid_size: int = 5,
    spline_order: int = 3,
    **kwargs
) -> RCLNLayer:
    """Create KAN-RCLN (Scheme 3: Symbolic Evolution)."""
    return create_rcln(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        net_type="kan",
        kan_grid_size=grid_size,
        kan_spline_order=spline_order,
        **kwargs
    )


def create_mamba_rcln(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    state_dim: int = 16,
    n_layers: int = 2,
    **kwargs
) -> RCLNLayer:
    """Create Mamba-RCLN (Scheme 4: Sequential Evolution)."""
    return create_rcln(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        net_type="mamba",
        mamba_state_dim=state_dim,
        mamba_layers=n_layers,
        **kwargs
    )


# =============================================================================
# Backward Compatibility
# =============================================================================

# Legacy imports for existing code
def MLPSoftShell(*args, **kwargs):
    """MLP fallback removed - raises error with migration guide."""
    raise NotImplementedError(
        "MLP Soft Shell has been REMOVED in RCLN 2.0.\n\n"
        "Please choose an evolution path:\n"
        "  1. FNO-RCLN:     net_type='fno'      (fluids, weather)\n"
        "  2. Clifford-RCLN: net_type='clifford' (robotics, EM)\n"
        "  3. KAN-RCLN:      net_type='kan'      (discovery)\n"
        "  4. Mamba-RCLN:    net_type='mamba'    (sequences)\n\n"
        "For legacy spectral support: net_type='spectral'"
    )
