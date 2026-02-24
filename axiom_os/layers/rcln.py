"""
RCLN - Residual Coupler Linking Neuron
PROPRIETARY - Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.
Unauthorized copying, modification, or distribution of this algorithm is prohibited.

The Hybrid Engine: Analytical physics (Hard Core) + Neural approximation (Soft Shell).
y_total = y_hard + λ·y_soft

Soft Shell options:
- FNO: Fourier Neural Operator (resolution invariant, for 2D spatial data)
- Spectral: 1D FFT → learnable weights → IFFT
- Clifford: Multivectors [s, v, B], rotational equivariance
- MLP: Standard fallback

Activity Monitor (from SPNN-Evo): Sliding-window |F_soft| → DiscoveryHotspot when consistently active.
"""

from typing import Optional, Callable, Union, NamedTuple, Any
from collections import deque
import math
import torch
import torch.nn as nn

# Clifford layers: optional dependency
try:
    from cliffordlayers.nn.modules import CliffordLinear
    HAS_CLIFFORD = True
except ImportError:
    HAS_CLIFFORD = False

# FNO: always available (no external deps)
from .fno import FNO2d

# Custom Clifford (built-in, no deps)
from .clifford_nn import CliffordLinear as CliffordLinearCustom, CliffordActivation
from .clifford_transformer import CliffordTransformerSoftShell
from .tensor_net import MERASoftShell

# 3D Clifford algebra: n_blades = 2^3 = 8
# Layout: [s, v1, v2, v3, B12, B13, B23, T] = scalar, vector(3), bivector(3), trivector(1)
N_BLADES_3D = 8


# -----------------------------------------------------------------------------
# Activity Monitor & Discovery Hotspot (from SPNN-Evo)
# -----------------------------------------------------------------------------


class DiscoveryHotspot(NamedTuple):
    """Flagged when F_soft consistently exceeds threshold over a sliding window."""

    instance_id: str
    avg_soft_magnitude: float
    sample_count: int
    last_input: Optional[torch.Tensor] = None
    last_soft_output: Optional[torch.Tensor] = None


class ActivityMonitor:
    """
    Monitors |F_soft|. If consistently > threshold over window, flag as Discovery Hotspot.
    """

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
# Euclidean 3D signature for rotational equivariance
CLIFFORD_G_3D = (1.0, 1.0, 1.0)


def _to_multivector(x: torch.Tensor, n_blades: int = N_BLADES_3D) -> torch.Tensor:
    """
    Convert flat tensor (B, D) to multivector format (B, 1, n_blades).
    Layout: [scalar, v1, v2, v3, b12, b23, b31, trivector].
    For 3D vector (u,v,w): maps to indices 1,2,3 (vector part).
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B = x.shape[0]
    D = x.shape[1]
    device = x.device
    dtype = x.dtype
    mv = torch.zeros(B, 1, n_blades, device=device, dtype=dtype)
    if D >= n_blades:
        mv[:, 0, :] = x[:, :n_blades]
    elif D == 3:
        # 3D vector (e.g. velocity u,v,w) at vector indices 1,2,3
        mv[:, 0, 1:4] = x
    else:
        mv[:, 0, :D] = x
    return mv


def _from_multivector(mv: torch.Tensor, output_dim: int) -> torch.Tensor:
    """
    Convert multivector (B, 1, n_blades) to flat tensor (B, output_dim).
    Takes first output_dim blade components.
    """
    B = mv.shape[0]
    n_blades = mv.shape[2]
    if output_dim >= n_blades:
        out = torch.zeros(B, output_dim, device=mv.device, dtype=mv.dtype)
        out[:, :n_blades] = mv.squeeze(1)
        return out
    return mv[:, 0, :output_dim].contiguous()


class CliffordSoftShell(nn.Module):
    """
    Soft Shell using Clifford Linear Layers.
    Accepts multivectors [s, v, B] and preserves rotational equivariance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        g: tuple = CLIFFORD_G_3D,
        n_blades: int = N_BLADES_3D,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_blades = n_blades

        # Clifford linear layers: 1 channel in/out, hidden_channels in between
        # Input: (B, 1, n_blades), Output: (B, 1, n_blades)
        self.clifford1 = CliffordLinear(g, in_channels=1, out_channels=hidden_dim, bias=True)
        self.clifford2 = CliffordLinear(g, in_channels=hidden_dim, out_channels=hidden_dim, bias=True)
        self.clifford3 = CliffordLinear(g, in_channels=hidden_dim, out_channels=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) -> (B, 1, n_blades)
        mv = _to_multivector(x, self.n_blades)
        mv = self.clifford1(mv)
        mv = nn.functional.silu(mv)
        mv = self.clifford2(mv)
        mv = nn.functional.silu(mv)
        mv = self.clifford3(mv)
        # (B, 1, n_blades) -> (B, output_dim)
        return _from_multivector(mv, self.output_dim)


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution (Fourier Layer).
    Transforms input via FFT, multiplies by learnable weights in frequency domain,
    transforms back via IFFT. Preserves global frequency structure.
    """

    def __init__(self, in_dim: int, n_modes: Optional[int] = None):
        super().__init__()
        self.in_dim = in_dim
        self.n_modes = n_modes or (in_dim // 2 + 1)
        # Learnable complex weights per frequency mode (stored as real, imag)
        scale = 1.0 / math.sqrt(self.n_modes)
        self.weight_real = nn.Parameter(torch.randn(self.n_modes) * scale)
        self.weight_imag = nn.Parameter(torch.randn(self.n_modes) * scale)

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.n_modes)
        nn.init.normal_(self.weight_real, 0, scale)
        nn.init.normal_(self.weight_imag, 0, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) real tensor
        Returns: (B, L) real tensor
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        L = x.shape[-1]
        # FFT: (B, L) -> (B, L//2+1) complex
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        n_freq = spectrum.shape[-1]
        # Learnable weights: use first n_modes, pad rest with 1+0j (identity)
        n = min(n_freq, self.n_modes)
        w = torch.complex(self.weight_real[:n], self.weight_imag[:n])
        out_spec = spectrum.clone()
        out_spec[..., :n] = spectrum[..., :n] * w
        # IFFT: (B, n_freq) -> (B, L)
        return torch.fft.irfft(out_spec, n=L, dim=-1).to(x.dtype)


class SpectralSoftShell(nn.Module):
    """
    Soft Shell using Spectral Convolution (Fourier Layer).
    FFT → learnable frequency weights → IFFT → SiLU → Linear projection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_modes: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_modes = n_modes or min(input_dim // 2 + 1, 64)  # Cap modes for stability

        self.spectral = SpectralConv1d(input_dim, n_modes=self.n_modes)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.spectral(x)
        h = nn.functional.silu(h)
        return self.proj(h)


class FNOSoftShell(nn.Module):
    """
    Soft Shell using Fourier Neural Operator (FNO2d).
    Expects 4D input (B, C, H, W) or (B, H, W, C). Resolution invariant.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fno = FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W)
        if x.dim() == 4 and x.shape[-1] == self.in_channels and x.shape[1] != self.in_channels:
            x = x.permute(0, 3, 1, 2)
        return self.fno(x)


class CustomCliffordSoftShell(nn.Module):
    """
    Soft Shell using built-in Clifford layers (core/clifford_ops).
    O(3) equivariant. Maps vector inputs (e.g. velocity u,v,w) to multivector format.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_blades = N_BLADES_3D
        # 1 channel in (padded to 8 blades), hidden channels, 1 channel out
        self.cl1 = CliffordLinearCustom(1, hidden_dim)
        self.act1 = CliffordActivation()
        self.cl2 = CliffordLinearCustom(hidden_dim, 1)
        self.act2 = CliffordActivation()
        self.proj = nn.Linear(N_BLADES_3D, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mv = _to_multivector(x, self.n_blades)  # (B, 1, 8)
        mv = self.cl1(mv)
        mv = self.act1(mv)
        mv = self.cl2(mv)
        mv = self.act2(mv)
        return self.proj(mv.squeeze(1))

    def reset_parameters(self) -> None:
        for m in [self.cl1, self.cl2, self.proj]:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


class MLPSoftShell(nn.Module):
    """Standard MLP fallback when Clifford/Spectral/FNO layers not selected."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _build_soft_shell(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    use_clifford: bool = True,
    use_spectral: bool = False,
    net_type: str = "mlp",
    fno_modes1: int = 12,
    fno_modes2: int = 12,
    use_clifford_custom: bool = True,
) -> nn.Module:
    """Build soft shell: FNO > CliffordTransformer > Spectral > Clifford > MLP."""
    if net_type == "fno":
        return FNOSoftShell(
            in_channels=input_dim,
            out_channels=output_dim,
            width=hidden_dim,
            modes1=fno_modes1,
            modes2=fno_modes2,
        )
    if net_type == "clifford_transformer":
        return CliffordTransformerSoftShell(input_dim, hidden_dim, output_dim)
    if net_type == "mera":
        return MERASoftShell(input_dim, hidden_dim, output_dim, chi=8, n_layers=3)
    if use_spectral:
        return SpectralSoftShell(input_dim, hidden_dim, output_dim)
    if use_clifford:
        if use_clifford_custom:
            return CustomCliffordSoftShell(input_dim, hidden_dim, output_dim)
        if HAS_CLIFFORD:
            return CliffordSoftShell(input_dim, hidden_dim, output_dim)
    return MLPSoftShell(input_dim, hidden_dim, output_dim)


class RCLNLayer(nn.Module):
    """
    Residual Coupler Linking Neuron.
    Combines hard physics (optional) with soft neural approximation.
    Soft Shell: Spectral (FFT→weights→IFFT) by default, or Clifford/MLP.
    Uses SiLU for unbounded outputs (crucial for v², etc.).
    Optional ActivityMonitor: sliding-window |F_soft| → DiscoveryHotspot.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hard_core_func: Optional[Callable] = None,
        lambda_res: float = 1.0,
        hippocampus: Optional[Any] = None,
        alpha_pert: float = 0.0,
        perturb_domain: Optional[str] = None,
        use_dynamic_alpha: bool = False,
        alpha_pert_theta: float = 0.12,
        use_learned_perturbation_gate: bool = False,
        use_clifford: Optional[bool] = None,
        use_spectral: Optional[bool] = None,
        net_type: str = "mlp",
        fno_modes1: int = 12,
        fno_modes2: int = 12,
        use_clifford_custom: bool = True,
        use_activity_monitor: bool = False,
        soft_threshold: float = 0.5,
        monitor_window: int = 32,
        uncertainty_threshold: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hard_core = hard_core_func  # Physics equation; None → zero
        self.lambda_res = lambda_res
        self._lambda_res_init = lambda_res
        self.hippocampus = hippocampus
        self.alpha_pert = alpha_pert if hippocampus is not None else 0.0
        self.perturb_domain = perturb_domain
        self.use_dynamic_alpha = use_dynamic_alpha
        self.alpha_pert_theta = alpha_pert_theta
        self.use_learned_perturbation_gate = use_learned_perturbation_gate
        self._perturbation_gate = None
        if use_learned_perturbation_gate and hippocampus is not None:
            from .perturbation_gate import LearnablePerturbationGate
            self._perturbation_gate = LearnablePerturbationGate(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=32,
                alpha_max=max(alpha_pert * 2, 0.2),
            )
        self.net_type = net_type

        # Soft Shell: FNO > Spectral > Clifford > MLP
        if use_spectral is None:
            use_spectral = False
        if use_clifford is None:
            use_clifford = HAS_CLIFFORD and not use_spectral and net_type != "fno"
        self.soft_shell = _build_soft_shell(
            input_dim, hidden_dim, output_dim,
            use_clifford=use_clifford, use_spectral=use_spectral,
            net_type=net_type, fno_modes1=fno_modes1, fno_modes2=fno_modes2,
            use_clifford_custom=use_clifford_custom,
        )
        self._use_clifford = use_clifford and not use_spectral and net_type not in ("fno", "clifford_transformer")
        self._use_spectral = use_spectral
        self._use_fno = net_type == "fno"
        self._use_clifford_transformer = net_type == "clifford_transformer"

        self._last_y_soft: Optional[torch.Tensor] = None

        # Activity Monitor (from SPNN-Evo)
        self.use_activity_monitor = use_activity_monitor
        self._activity_monitor: Optional[ActivityMonitor] = None
        if use_activity_monitor:
            self._activity_monitor = ActivityMonitor(threshold=soft_threshold, window_size=monitor_window)
            self._activity_monitor.instance_id = str(id(self))

        # Uncertainty gate: "know what we don't know"
        self.uncertainty_threshold = uncertainty_threshold

    def set_lambda_res(self, value: float) -> None:
        """突触式：主脑调度 λ_res（可塑性衰减）"""
        self.lambda_res = float(value)

    def set_lambda_decay(self, epoch: int, total_epochs: int, decay_min: float = 0.5) -> None:
        """
        突触式 λ_res 衰减：随训练进行 Soft 权重逐渐增大（Hard 相对减弱）。
        λ = λ_init * (1 - (1 - decay_min) * epoch / total_epochs)
        """
        if total_epochs <= 0:
            return
        frac = min(1.0, epoch / total_epochs)
        self.lambda_res = self._lambda_res_init * (1.0 - (1.0 - decay_min) * frac)

    def get_uncertainty_status(self) -> tuple:
        """
        Returns (uncertain: bool, activity: float).
        uncertain=True when soft_activity > threshold (prediction unreliable).
        """
        activity = self.get_soft_activity()
        return (activity > self.uncertainty_threshold, activity)

    def forward(
        self,
        x: Union["UPIState", torch.Tensor],
        return_hotspot: bool = False,
        return_uncertainty: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass.
        Input: x (UPIState or tensor of values)
        Output: y_total = y_hard + lambda_res * y_soft
        When return_hotspot=True and use_activity_monitor: (y_total, hotspot)
        When return_uncertainty=True: (y_total, uncertain) where uncertain=True if soft_activity > threshold
        """
        # Extract values (UPIState or raw tensor)
        if isinstance(x, torch.Tensor):
            x_vals = x.float()
        elif hasattr(x, "values") and not isinstance(x, torch.Tensor):
            x_vals = x.values.float()
        else:
            x_vals = torch.as_tensor(x, dtype=torch.float32)
        if x_vals.dim() == 1:
            x_vals = x_vals.unsqueeze(0)
        # FNO expects 4D (B,C,H,W); MLP/Clifford/Spectral/Transformer expect 2D (B,D)
        if self._use_fno and x_vals.dim() != 4:
            raise ValueError("RCLN with net_type='fno' expects 4D input (B,C,H,W) or (B,H,W,C)")

        # Hard core: physics equation
        if self.hard_core is not None:
            y_hard = self.hard_core(x)
            if isinstance(y_hard, torch.Tensor):
                y_hard = y_hard.float()
            else:
                import numpy as np
                arr = np.asarray(y_hard, dtype=np.float64)
                y_hard = torch.tensor(arr.copy(), dtype=torch.float32, device=x_vals.device)
            if y_hard.dim() == 1:
                y_hard = y_hard.unsqueeze(0)
        else:
            y_hard = None

        # Soft shell: Clifford or MLP
        y_soft = self.soft_shell(x_vals)
        self._last_y_soft = y_soft.detach()

        if y_hard is not None:
            y_total = y_hard + self.lambda_res * y_soft
        else:
            y_total = self.lambda_res * y_soft

        # 海马体扰动：记忆作为扰动项，直觉由训练得到
        # use_learned_perturbation_gate: α 由可学习门控输出
        # 否则: 固定 α 或 组合A 规则
        if self.hippocampus is not None and (self.alpha_pert > 0 or self._perturbation_gate is not None) and self.perturb_domain:
            try:
                import numpy as np
                from axiom_os.core.perturbation import infer_partition_id
                x_np = x_vals.detach().cpu().numpy() if hasattr(x_vals, "detach") else np.asarray(x_vals)
                pid = infer_partition_id(x_np, self.perturb_domain)
                pert = self.hippocampus.eval_perturbation(x_vals, partition_id=pid, domain=self.perturb_domain)
                if pert is not None:
                    pert_t = torch.as_tensor(pert, dtype=y_total.dtype, device=y_total.device)
                    if pert_t.shape != y_total.shape and pert_t.numel() == y_total.numel():
                        pert_t = pert_t.reshape(y_total.shape)

                    if pert_t.shape == y_total.shape:
                        if self._perturbation_gate is not None:
                            activity_per_sample = y_soft.abs().mean(dim=-1, keepdim=True)
                            alpha = self._perturbation_gate(x_vals, activity_per_sample, y_total, pert_t)
                            alpha = alpha.unsqueeze(-1) if alpha.dim() == 1 else alpha
                            y_total = y_total + alpha * pert_t
                        else:
                            from axiom_os.core.alpha_scheduler import compute_alpha_hybrid
                            activity = float(y_soft.abs().mean().item())
                            alpha = (
                                compute_alpha_hybrid(activity, self.alpha_pert, theta=self.alpha_pert_theta)
                                if self.use_dynamic_alpha
                                else self.alpha_pert
                            )
                            if alpha > 1e-6:
                                y_total = y_total + alpha * pert_t
            except Exception as e:
                import warnings
                warnings.warn(
                    f"RCLN hippocampus perturbation failed (non-fatal): {e}",
                    UserWarning,
                    stacklevel=1,
                )

        hotspot = None
        if return_hotspot and self._activity_monitor is not None:
            hotspot = self._activity_monitor.update(y_soft, x_vals)

        if return_uncertainty:
            uncertain, _ = self.get_uncertainty_status()
            return y_total, uncertain

        if return_hotspot and self._activity_monitor is not None:
            return y_total, hotspot
        return y_total

    def get_soft_activity(self) -> float:
        """
        Return the mean magnitude of y_soft from the last forward pass.
        Used to trigger the Discovery Engine when soft shell is highly active.
        """
        if self._last_y_soft is None:
            return 0.0
        return float(self._last_y_soft.abs().mean().item())

    def reset_activity_monitor(self) -> None:
        """Reset the ActivityMonitor (e.g. after crystallization)."""
        if self._activity_monitor is not None:
            self._activity_monitor.reset()
