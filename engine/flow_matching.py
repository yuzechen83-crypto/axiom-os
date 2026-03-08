"""
Meta-Flow Matching (CFM) - Axiom-OS v4.0 Holographic Generator
Replace Diffusion SDE with deterministic Flow Matching along the Meta-Axis.

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
z=L: Simple prior (laminar, Gaussian). z=0: Complex data (turbulence).
Flow from z=1 to z=0 projects complexity from bulk to boundary.
"""

from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint_adjoint as odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    try:
        from torchdiffeq import odeint
        HAS_TORCHDIFFEQ = True
    except ImportError:
        HAS_TORCHDIFFEQ = False


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for meta-coordinate z ∈ [0, 1]."""
    half = dim // 2
    emb = torch.arange(half, device=t.device, dtype=t.dtype) * (3.14159 / half)
    t_flat = t.float().view(-1, 1)
    return torch.cat([torch.sin(t_flat * emb), torch.cos(t_flat * emb)], dim=-1)


class VectorFieldNet(nn.Module):
    """
    Velocity field v_z(x): predicts dx/dz along the Meta-Axis.
    Input: state x, meta-coordinate z (as time t ∈ [0, 1]).
    Output: velocity field dx/dz.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        z_emb_dim: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.z_mlp = nn.Sequential(
            nn.Linear(z_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layers = []
        in_dim = state_dim + hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, state_dim))
        self.mlp = nn.Sequential(*layers)
        self.z_emb_dim = z_emb_dim

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: (B, state_dim) - current state
        z: (B,) or (B, 1) - meta-coordinate in [0, 1], z=1 simple, z=0 complex
        Returns: (B, state_dim) - velocity dx/dz
        """
        if z.dim() == 1:
            z = z.view(-1, 1)
        z_emb = _sinusoidal_embedding(z.squeeze(-1), self.z_emb_dim)
        z_emb = self.z_mlp(z_emb)
        h = torch.cat([x, z_emb], dim=-1)
        return self.mlp(h)


def conditional_flow_matching_loss(
    v_net: VectorFieldNet,
    x_simple: torch.Tensor,
    x_complex: torch.Tensor,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    CFM Loss: minimize ||v_z(x_z) - (x_complex - x_simple)||^2.
    x_z = z*x_simple + (1-z)*x_complex (linear interpolation).
    Target velocity: x_complex - x_simple (flow from simple to complex).
    """
    B = x_simple.shape[0]
    device = x_simple.device
    if z is None:
        z = torch.rand(B, device=device)
    z = z.view(-1, 1)
    x_z = z * x_simple + (1 - z) * x_complex
    v_target = x_complex - x_simple
    v_pred = v_net(x_z, z.squeeze(-1))
    return ((v_pred - v_target) ** 2).mean()


def sample_prior(batch_size: int, state_dim: int, device: torch.device) -> torch.Tensor:
    """Simple prior: Gaussian (laminar)."""
    return torch.randn(batch_size, state_dim, device=device) * 0.5


def _ode_func(v_net: VectorFieldNet, z: float) -> Callable:
    """Build ODE function for integration: dx/dz = v_net(x, z)."""

    def func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: scalar or (B,) from odeint. We use t as z.
        if t.dim() == 0:
            z_val = t.expand(x.shape[0])
        else:
            z_val = t
        return v_net(x, z_val)

    return func


def sample_flow_matching(
    v_net: VectorFieldNet,
    x_init: torch.Tensor,
    z_start: float = 1.0,
    z_end: float = 0.0,
    n_steps: int = 10,
) -> torch.Tensor:
    """
    Holographic projection: integrate ODE from z=1 (simple) to z=0 (complex).
    Uses torchdiffeq if available, else Euler.
    """
    if HAS_TORCHDIFFEQ:
        t_span = torch.linspace(z_start, z_end, n_steps + 1, device=x_init.device, dtype=x_init.dtype)
        def func(t, x):
            z_val = t.expand(x.shape[0]) if t.dim() == 0 else t
            return v_net(x, z_val)
        out = odeint(func, x_init, t_span, method="euler")
        return out[-1]
    # Fallback: Euler
    x = x_init.clone()
    dz = (z_end - z_start) / n_steps
    for i in range(n_steps):
        z = z_start + (i + 0.5) * dz
        z_t = torch.full((x.shape[0],), z, device=x.device, dtype=x.dtype)
        v = v_net(x, z_t)
        x = x + v * dz
    return x


class MetaFlowMatching(nn.Module):
    """
    End-to-end Meta-Flow Matching for generative physics.
    Train: CFM loss. Sample: ODE from z=1 (simple) to z=0 (complex).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        z_emb_dim: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.v_net = VectorFieldNet(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            z_emb_dim=z_emb_dim,
            n_layers=n_layers,
        )
        self.state_dim = state_dim

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.v_net(x, z)

    def loss(
        self,
        x_simple: torch.Tensor,
        x_complex: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return conditional_flow_matching_loss(
            self.v_net, x_simple, x_complex, z
        )

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        x_init: Optional[torch.Tensor] = None,
        n_steps: int = 20,
    ) -> torch.Tensor:
        if x_init is None:
            x_init = sample_prior(batch_size, self.state_dim, device or torch.device("cpu"))
        return sample_flow_matching(
            self.v_net, x_init, z_start=1.0, z_end=0.0, n_steps=n_steps
        )
