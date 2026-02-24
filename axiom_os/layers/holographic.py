"""
Holographic Projection Layer - MASHD Meta-Axis Integration
SPNN-Holo: Complex phenomena as projections of higher-dimensional dynamics.

Math: Output(x) = Σ_k w_k · Net(x, z_k)
  - z_k: discretized points along Meta-Axis [-L, L]
  - w_k: projection kernel K(z) × Simpson integration weights
  - K(z) = 1 / sqrt(1 - (z/L)² + ε)

HolographicRCLN: y = y_hard + λ · ∫ K(z)·Ψ(x,z) dz
"""

from typing import Optional, List, Callable, Union
import torch
import torch.nn as nn

from .fno import FNO2d


def projection_kernel(z: torch.Tensor, L: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Theoretical projection kernel from MASHD:
    K(z) = 1 / sqrt(1 - (z/L)² + ε)
    Regularized to avoid singularity at z → ±L.
    """
    z = torch.as_tensor(z, dtype=torch.float32)
    denom = 1.0 - (z / L) ** 2 + eps
    return 1.0 / torch.sqrt(torch.clamp(denom, min=eps))


def simpson_weights(n: int, a: float, b: float) -> torch.Tensor:
    """
    Simpson's rule weights for n points (n must be odd) on [a, b].
    Returns tensor of shape (n,) with coefficients.
    """
    if n < 3 or n % 2 == 0:
        n = max(3, n | 1)
    h = (b - a) / (n - 1)
    w = torch.zeros(n)
    w[0] = h / 3.0
    w[-1] = h / 3.0
    for i in range(1, n - 1):
        w[i] = (2.0 * h / 3.0) if i % 2 == 0 else (4.0 * h / 3.0)
    return w


class HolographicProjectionLayer(nn.Module):
    """
    MASHD Holographic Projection: Output(x) = Σ_k w_k · Net(x, z_k)

    Integrates over the Meta-Axis z ∈ [-L, L] using:
    - Projection kernel K(z) = 1/sqrt(1-(z/L)²+ε)
    - Simpson's rule for numerical integration
    - z injected as extra channel (4D) or concat (2D)
    """

    def __init__(
        self,
        base_net: nn.Module,
        n_z_slices: int = 7,
        L: float = 1.0,
        eps: float = 1e-6,
        use_learnable_weights: bool = False,
    ):
        super().__init__()
        self.base_net = base_net
        self.n_z_slices = max(3, n_z_slices | 1)
        self.L = L
        self.eps = eps
        self.use_learnable_weights = use_learnable_weights

        self.register_buffer("z_grid", torch.linspace(-L, L, self.n_z_slices))

        K_vals = projection_kernel(self.z_grid, L, eps)
        simpson_w = simpson_weights(self.n_z_slices, -L, L).to(K_vals.device)
        w_init = K_vals * simpson_w
        w_init = w_init / (w_init.sum() + 1e-8)
        if use_learnable_weights:
            self.projection_weights = nn.Parameter(w_init.clone())
        else:
            self.register_buffer("projection_weights", w_init)

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        return_per_slice: bool = False,
    ) -> torch.Tensor:
        if z is not None:
            z_val = z.item() if z.numel() == 1 else z.mean().item()
            out = self._forward_at_z(x, z_val)
            if return_per_slice:
                return out, [out]
            return out

        outputs: List[torch.Tensor] = []
        for k in range(self.n_z_slices):
            z_k = self.z_grid[k].item()
            out_k = self._forward_at_z(x, z_k)
            outputs.append(out_k)

        stacked = torch.stack(outputs, dim=0)
        w = self.projection_weights.view(-1, *([1] * (stacked.dim() - 1)))
        out = (w * stacked).sum(dim=0)

        if return_per_slice:
            return out, outputs
        return out

    def _forward_at_z(self, x: torch.Tensor, z_val: float) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            z_ch = torch.full((B, 1, H, W), z_val, dtype=x.dtype, device=x.device)
            x_with_z = torch.cat([x, z_ch], dim=1)
            return self.base_net(x_with_z)
        else:
            B = x.shape[0]
            z_b = torch.full((B, 1), z_val, dtype=x.dtype, device=x.device)
            x_with_z = torch.cat([x, z_b], dim=1)
            return self.base_net(x_with_z)

    def get_z_grid(self) -> torch.Tensor:
        return self.z_grid

    def get_projection_weights(self) -> torch.Tensor:
        return self.projection_weights


class HolographicFNO(nn.Module):
    """
    FNO wrapped with Holographic Projection over the Meta-Axis.
    For 2D turbulence: z=0 learns mean flow, z→L learns turbulent fluctuations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
        n_z_slices: int = 7,
        L: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fno = FNO2d(
            in_channels=in_channels + 1,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            n_layers=n_layers,
        )
        self.holo = HolographicProjectionLayer(
            base_net=self.fno,
            n_z_slices=max(3, n_z_slices | 1),
            L=L,
            use_learnable_weights=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        return_per_slice: bool = False,
    ) -> torch.Tensor:
        return self.holo(x, z=z, return_per_slice=return_per_slice)

    def get_z_grid(self) -> torch.Tensor:
        return self.holo.get_z_grid()


class HolographicRCLN(nn.Module):
    """
    SPNN-Holo: Holographic RCLN - y = y_hard + λ · ∫ K(z)·Ψ(x,z) dz

    Combines physics Hard Core with holographic Soft Shell over Meta-Axis.
    Ψ(x,z): MLP in extended (x,z) space; z=0 → macroscopic, z→L → turbulent fluctuations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hard_core_func: Optional[Callable] = None,
        lambda_res: float = 1.0,
        n_z_slices: int = 7,
        L: float = 1.0,
        use_learnable_weights: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hard_core = hard_core_func
        self.lambda_res = lambda_res

        # Ψ(x,z): base net accepts input_dim+1 (extra z channel)
        self._psi = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.holo = HolographicProjectionLayer(
            base_net=self._psi,
            n_z_slices=max(3, n_z_slices | 1),
            L=L,
            use_learnable_weights=use_learnable_weights,
        )
        self._last_y_soft: Optional[torch.Tensor] = None

    def forward(
        self,
        x: Union[torch.Tensor, object],
        z: Optional[torch.Tensor] = None,
        return_per_slice: bool = False,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_vals = x.float()
        elif hasattr(x, "values"):
            x_vals = x.values.float()
        else:
            x_vals = torch.as_tensor(x, dtype=torch.float32)
        if x_vals.dim() == 1:
            x_vals = x_vals.unsqueeze(0)

        # Hard core
        if self.hard_core is not None:
            y_hard = self.hard_core(x)
            if isinstance(y_hard, torch.Tensor):
                y_hard = y_hard.float()
            else:
                y_hard = torch.as_tensor(y_hard, dtype=torch.float32, device=x_vals.device)
            if y_hard.dim() == 1:
                y_hard = y_hard.unsqueeze(0)
        else:
            y_hard = None

        # Holographic soft: ∫ K(z)·Ψ(x,z) dz
        if return_per_slice:
            y_soft, per_slice = self.holo(x_vals, z=z, return_per_slice=True)
        else:
            y_soft = self.holo(x_vals, z=z, return_per_slice=False)
            per_slice = None
        self._last_y_soft = y_soft.detach()

        if y_hard is not None:
            y_total = y_hard + self.lambda_res * y_soft
        else:
            y_total = self.lambda_res * y_soft

        if return_per_slice:
            return y_total, per_slice
        return y_total

    def get_z_grid(self) -> torch.Tensor:
        return self.holo.get_z_grid()

    def get_soft_activity(self) -> float:
        if self._last_y_soft is None:
            return 0.0
        return float(self._last_y_soft.abs().mean().item())
