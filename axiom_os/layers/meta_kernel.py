"""
Meta-Axis Projection Kernel - Modified Gravity / Dark Matter Alternative

Theory:
  - Geometric kernel: K(z) = k0 / sqrt(1 - (z/L)²),  ΔV² ∝ ∫_{-L}^{L} K(z)·ρ(r,z) dz
  - McGaugh projection: ν(g) = 1/(1 - exp(-√(g/a₀))),  g_obs = g_bar · ν(g_bar)
  - Grand Unification: a₀ = c²/L (L cosmological scale ~80 Gly)
"""

from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn

# Physical constants
C_LIGHT_MS = 2.998e8  # m/s
M_PER_GLY = 9.46e24  # 1 Gly in m (approx)
KM_S_SQ_PER_KPC_TO_MS2 = 1e6 / 3.08567758128e19  # (km/s)²/kpc -> m/s²
A0_SI_DEFAULT = 1.2e-10  # MOND a₀ in m/s²


def compute_meta_length(a0_si: float) -> dict:
    """
    Grand Unification: Meta-Axis length L from acceleration scale a₀.
    Hypothesis: L = c²/a₀ (curvature radius).
    a0_si: acceleration in m/s² (e.g. 1.2e-10 for MOND).
    Returns: L_m (meters), L_Gly (giga-light-years).
    """
    if a0_si <= 0 or not np.isfinite(a0_si):
        return {"L_m": np.nan, "L_Gly": np.nan}
    L_m = (C_LIGHT_MS ** 2) / a0_si
    L_Gly = L_m / M_PER_GLY
    return {"L_m": L_m, "L_Gly": L_Gly}


def g0_from_a0_si(a0_si: float) -> float:
    """Convert a₀ (m/s²) to g₀ [(km/s)²/kpc]."""
    return a0_si / KM_S_SQ_PER_KPC_TO_MS2


def a0_si_from_g0(g0: float) -> float:
    """Convert g₀ [(km/s)²/kpc] to a₀ (m/s²)."""
    return g0 * KM_S_SQ_PER_KPC_TO_MS2


def nu_mcgaugh(g: Union[float, np.ndarray], a0: Union[float, np.ndarray], eps: float = 1e-14) -> Union[float, np.ndarray]:
    """
    McGaugh projection operator: ν(g) = 1/(1 - exp(-√(g/a₀))).
    g_obs = g_bar · ν(g_bar).
    g, a0: same units (e.g. (km/s)²/kpc or m/s²).
    """
    g = np.asarray(g)
    a0 = np.asarray(a0)
    x = np.maximum(g / (a0 + eps), eps)
    denom = 1.0 - np.exp(-np.sqrt(x))
    denom = np.maximum(denom, eps)
    return 1.0 / denom


def nu_mcgaugh_torch(g: torch.Tensor, a0: Union[float, torch.Tensor], eps: float = 1e-14) -> torch.Tensor:
    """Torch version of McGaugh ν(g)."""
    if isinstance(a0, (int, float)):
        a0 = torch.tensor(a0, dtype=g.dtype, device=g.device)
    x = (g / (a0 + eps)).clamp(min=eps)
    denom = (1.0 - torch.exp(-torch.sqrt(x))).clamp(min=eps)
    return 1.0 / denom


def meta_kernel_weights(z: torch.Tensor, L: float, k0: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Theoretical projection kernel: W_z = k0 / sqrt(1 - (z/L)² + ε)
    """
    z = torch.as_tensor(z, dtype=torch.float32)
    denom = 1.0 - (z / L) ** 2 + eps
    return k0 / torch.sqrt(torch.clamp(denom, min=eps))


def simpson_weights_meta(n: int, a: float, b: float) -> torch.Tensor:
    """Simpson rule weights for integration along Meta-Axis."""
    if n < 3 or n % 2 == 0:
        n = max(3, n | 1)
    h = (b - a) / (n - 1)
    w = torch.zeros(n)
    w[0] = h / 3.0
    w[-1] = h / 3.0
    for i in range(1, n - 1):
        w[i] = (2.0 * h / 3.0) if i % 2 == 0 else (4.0 * h / 3.0)
    return w


class MetaProjectionLayer(nn.Module):
    """
    Meta-Axis Projection. Two modes:
      - geometric: ΔV² ∝ ∫_{-L}^{L} K(z)·ρ(r,z) dz, K(z)=k0/sqrt(1-(z/L)²)
      - mcgaugh: g_obs = g_bar · ν(g_bar), ν(g)=1/(1-exp(-√(g/a₀)))
    When a0_init is set, L is derived from L=c²/a₀ (physical prior).
    """

    def __init__(
        self,
        n_z_points: int = 100,
        L_init: float = 10.2,
        k0_init: float = 1.0,
        alpha: float = 1.0,
        eps: float = 1e-6,
        learn_L: bool = True,
        learn_k0: bool = True,
        projection_mode: str = "geometric",
        a0_init: Optional[float] = None,
    ):
        super().__init__()
        self.projection_mode = projection_mode
        self.n_z = max(3, n_z_points | 1)
        self.alpha = alpha
        self.eps = eps

        if projection_mode == "mcgaugh":
            g0_default = float(a0_init) if a0_init is not None else 3700.0
            self.register_parameter("log_g0", nn.Parameter(torch.tensor(np.log10(g0_default + 1e-10))))
            self.register_buffer("L", torch.tensor(0.0))
            self.register_buffer("k0", torch.tensor(1.0))
            self.learn_L = False
            self.learn_k0 = False
        else:
            if a0_init is not None:
                a0_si = a0_init if a0_init < 1e-6 else a0_si_from_g0(a0_init)
                meta = compute_meta_length(a0_si)
                L_init = meta.get("L_Gly", L_init) if np.isfinite(meta.get("L_m", np.nan)) else L_init
            L = nn.Parameter(torch.tensor(float(L_init))) if learn_L else L_init
            k0 = nn.Parameter(torch.tensor(float(k0_init))) if learn_k0 else k0_init
            if learn_L:
                self.register_parameter("L", L)
            else:
                self.register_buffer("L", torch.tensor(L_init))
            if learn_k0:
                self.register_parameter("k0", k0)
            else:
                self.register_buffer("k0", torch.tensor(k0_init))
            self.learn_L = learn_L
            self.learn_k0 = learn_k0

        self._z_grid: Optional[torch.Tensor] = None

    def _get_z_grid(self, device: torch.device) -> torch.Tensor:
        if self._z_grid is None or self._z_grid.device != device:
            L_val = self.L if isinstance(self.L, torch.Tensor) else self.L
            L_val = L_val.item() if hasattr(L_val, "item") else float(L_val)
            self._z_grid = torch.linspace(-L_val * 0.999, L_val * 0.999, self.n_z, device=device)
        return self._z_grid

    def _compute_integral_weights(self) -> torch.Tensor:
        """W_z = k0/sqrt(1-(z/L)²) * (1-|z|/L)^α * simpson_w"""
        L_val = self.L
        if isinstance(L_val, torch.Tensor):
            L_val = torch.clamp(L_val, min=1.0).item()
        else:
            L_val = max(float(L_val), 1.0)
        z = torch.linspace(-L_val * 0.999, L_val * 0.999, self.n_z, device=self.L.device)
        K_z = meta_kernel_weights(z, L_val, 1.0, self.eps)
        rho_z = (1.0 - torch.abs(z) / L_val).clamp(min=0) ** self.alpha
        simpson_w = simpson_weights_meta(self.n_z, -L_val, L_val).to(z.device)
        w = self.k0 * K_z * rho_z * simpson_w
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        geometric: x = ρ(r) (B,) or (B,1). Returns ΔV² ∝ ∫ K(z)·ρ dz.
        mcgaugh: x = g_bar (B,) in (km/s)²/kpc. Returns g_obs = g_bar · ν(g_bar).
        """
        if self.projection_mode == "mcgaugh":
            g_bar = x.squeeze()
            g0 = 10.0 ** self.log_g0.clamp(-2.0, 6.0)
            nu = nu_mcgaugh_torch(g_bar, g0, self.eps)
            return g_bar * nu

        rho_r = x
        if rho_r.dim() == 1:
            rho_r = rho_r.unsqueeze(-1)
        w = self._compute_integral_weights()
        C = w.sum()
        out = C * rho_r.squeeze(-1)
        return out


class MetaProjectionModel(nn.Module):
    """
    Full model: r, V_bary_sq -> MetaProjectionLayer.
    geometric: ρ_proxy = V_bary²/(r^α+ε) -> ∫ K(z)·ρ dz.
    mcgaugh: g_bar = V_bary²/r -> g_obs = g_bar·ν(g_bar).
    """

    def __init__(
        self,
        n_z_points: int = 100,
        L_init: float = 10.2,
        k0_init: float = 1.0,
        alpha: float = 1.0,
        learn_L: bool = False,
        projection_mode: str = "geometric",
        a0_init: Optional[float] = None,
    ):
        super().__init__()
        self.projection_mode = projection_mode
        self.meta = MetaProjectionLayer(
            n_z_points=n_z_points,
            L_init=L_init,
            k0_init=k0_init,
            alpha=alpha,
            learn_L=learn_L,
            learn_k0=True,
            projection_mode=projection_mode,
            a0_init=a0_init,
        )
        self.rho_scale = nn.Parameter(torch.tensor(1.0))
        self.log_r_exp = nn.Parameter(torch.tensor(0.693))

    def forward(self, r: torch.Tensor, V_bary_sq: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        r_safe = r.clamp(min=0.1)
        if self.projection_mode == "mcgaugh":
            g_bar = V_bary_sq / (r_safe + eps)
            return self.meta(g_bar)
        r_exp = torch.exp(self.log_r_exp).clamp(1.0, 4.0)
        rho_proxy = self.rho_scale * V_bary_sq / (r_safe**r_exp + eps)
        return self.meta(rho_proxy)
