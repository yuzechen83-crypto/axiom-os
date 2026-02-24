"""
PDE Discovery Feature Library
Construct candidate library Θ for identifying PDE terms from spatiotemporal fields.
Fluid dynamics: viscosity ν·u_xx, advection u·u_x, etc.
"""

import numpy as np
from typing import Optional, Tuple, List
from enum import Enum


class DerivativeMethod(Enum):
    FINITE_DIFF = "finite_diff"
    SPECTRAL = "spectral"


def compute_ux_finite_diff(u: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """∂u/∂x via central finite differences."""
    dx = np.diff(x)
    if axis == -1:
        ux = np.gradient(u, dx, axis=-1)
    else:
        ux = np.gradient(u, dx[0] if dx.ndim > 0 else dx, axis=axis)
    return np.asarray(ux, dtype=np.float64)


def compute_uxx_finite_diff(u: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """∂²u/∂x² via central finite differences."""
    dx = np.diff(x)
    dx_mean = np.mean(dx) if np.size(dx) > 0 else 1.0
    if u.ndim == 1:
        uxx = np.gradient(np.gradient(u, dx_mean), dx_mean)
    else:
        uxx = np.gradient(np.gradient(u, dx_mean, axis=axis), dx_mean, axis=axis)
    return np.asarray(uxx, dtype=np.float64)


def compute_derivatives_2d(
    u: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    method: DerivativeMethod = DerivativeMethod.FINITE_DIFF,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ux, uxx for u(t, x) on a 2D grid (t, x).
    Returns: ux, uxx (same shape as u)
    """
    if method == DerivativeMethod.FINITE_DIFF:
        # u is (n_t, n_x)
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        ux = np.gradient(u, dx, axis=1)
        uxx = np.gradient(np.gradient(u, dx, axis=1), dx, axis=1)
    else:
        ux = compute_ux_finite_diff(u, x, axis=1)
        uxx = compute_uxx_finite_diff(u, x, axis=1)
    return ux, uxx


def build_pde_library(
    u: np.ndarray,
    ux: np.ndarray,
    uxx: np.ndarray,
    include_terms: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build candidate library Θ for PDE discovery.
    Columns: [u, u², ux, uxx, u·ux, ux²]
    Returns: (Theta, feature_names)
    """
    u_flat = np.asarray(u, dtype=np.float64).ravel()
    ux_flat = np.asarray(ux, dtype=np.float64).ravel()
    uxx_flat = np.asarray(uxx, dtype=np.float64).ravel()
    n = len(u_flat)

    default_terms = ["u", "u2", "ux", "uxx", "u_ux", "ux2"]
    terms = include_terms or default_terms

    cols = []
    names = []
    if "u" in terms:
        cols.append(u_flat.reshape(-1, 1))
        names.append("u")
    if "u2" in terms:
        cols.append((u_flat ** 2).reshape(-1, 1))
        names.append("u^2")
    if "ux" in terms:
        cols.append(ux_flat.reshape(-1, 1))
        names.append("u_x")
    if "uxx" in terms:
        cols.append(uxx_flat.reshape(-1, 1))
        names.append("u_xx")
    if "u_ux" in terms:
        cols.append((u_flat * ux_flat).reshape(-1, 1))
        names.append("u*u_x")
    if "ux2" in terms:
        cols.append((ux_flat ** 2).reshape(-1, 1))
        names.append("u_x^2")

    if not cols:
        cols = [u_flat.reshape(-1, 1), (u_flat**2).reshape(-1, 1),
                ux_flat.reshape(-1, 1), uxx_flat.reshape(-1, 1),
                (u_flat * ux_flat).reshape(-1, 1), (ux_flat**2).reshape(-1, 1)]
        names = ["u", "u^2", "u_x", "u_xx", "u*u_x", "u_x^2"]

    Theta = np.hstack(cols)
    return Theta, names


# -----------------------------------------------------------------------------
# 2D Vector Field (Navier-Stokes)
# -----------------------------------------------------------------------------


class VectorFeatureExtractor:
    """
    Extract vector calculus invariants from 2D velocity field u⃗=(u,v).

    Input: u(T,H,W), v(T,H,W)  [Time, Height/Y, Width/X]
    Convention: axis 0=time, axis 1=Y (height), axis 2=X (width)
    Derivatives: ∂_x along axis 2, ∂_y along axis 1

    Invariants:
      div_u   = ∇·u⃗ = u_x + v_y  (≈0 for incompressible)
      curl_u  = ω = ∇×u⃗ = v_x - u_y  (vorticity, scalar in 2D)
      laplacian_omega = ∇²ω = ω_xx + ω_yy  (diffusion target)
      advection = (u·∇)ω = u·ω_x + v·ω_y

    Vorticity eq: ∂ω/∂t + (u·∇)ω = ν∇²ω
    """

    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dt: float = 1.0,
    ):
        self.dx = dx
        self.dy = dy
        self.dt = dt

    @classmethod
    def from_coordinates(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        t: Optional[np.ndarray] = None,
    ) -> "VectorFeatureExtractor":
        """
        Build extractor from coordinate arrays for variable spacing.
        x, y: 1D arrays (spatial). t: optional 1D (time).
        """
        dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        dt = float(t[1] - t[0]) if t is not None and len(t) > 1 else 1.0
        return cls(dx=dx, dy=dy, dt=dt)

    def extract(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> dict:
        """
        u, v: shape (T, H, W) or (H, W) for single snapshot.
        Returns dict with:
          u_x, u_y, v_x, v_y  (gradients ∂_x, ∂_y)
          div_u (∇·u⃗ ≈ 0 incompressible)
          curl_u (ω = v_x - u_y vorticity)
          laplacian_omega (∇²ω diffusion target)
          advection ((u·∇)ω = uω_x + vω_y)
          omega_x, omega_y
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if u.ndim == 2:
            u = u[np.newaxis, ...]
            v = v[np.newaxis, ...]

        # Gradients: axis 0=t, 1=y, 2=x  (∂_x = axis 2, ∂_y = axis 1)
        u_x = np.gradient(u, self.dx, axis=2)
        u_y = np.gradient(u, self.dy, axis=1)
        v_x = np.gradient(v, self.dx, axis=2)
        v_y = np.gradient(v, self.dy, axis=1)

        # div_u = ∇·u⃗ = u_x + v_y (≈0 for incompressible)
        div_u = u_x + v_y

        # curl_u (vorticity) ω = ∇×u⃗ = v_x - u_y (2D scalar)
        curl_u = v_x - u_y

        # laplacian_omega = ∇²ω = ω_xx + ω_yy (diffusion term)
        omega_xx = np.gradient(np.gradient(curl_u, self.dx, axis=2), self.dx, axis=2)
        omega_yy = np.gradient(np.gradient(curl_u, self.dy, axis=1), self.dy, axis=1)
        laplacian_omega = omega_xx + omega_yy

        # advection (u·∇)ω = u·ω_x + v·ω_y
        omega_x = np.gradient(curl_u, self.dx, axis=2)
        omega_y = np.gradient(curl_u, self.dy, axis=1)
        advection = u * omega_x + v * omega_y

        return {
            "u_x": u_x,
            "u_y": u_y,
            "v_x": v_x,
            "v_y": v_y,
            "div_u": div_u,
            "curl_u": curl_u,
            "laplacian_omega": laplacian_omega,
            "advection": advection,
            "omega_x": omega_x,
            "omega_y": omega_y,
        }


def build_vorticity_library(
    curl_u: np.ndarray,
    laplacian_omega: np.ndarray,
    advection: np.ndarray,
    include_terms: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Library for vorticity equation: ∂ω/∂t + (u·∇)ω = ν∇²ω
    Target: diffusion D = ν∇²ω
    Columns: [omega, omega^2, lap_omega, advection, ...]
    """
    omega_flat = np.asarray(curl_u, dtype=np.float64).ravel()
    lap_flat = np.asarray(laplacian_omega, dtype=np.float64).ravel()
    adv_flat = np.asarray(advection, dtype=np.float64).ravel()

    terms = include_terms or ["omega", "omega2", "lap_omega", "advection"]
    cols, names = [], []
    if "omega" in terms:
        cols.append(omega_flat.reshape(-1, 1))
        names.append("omega")
    if "omega2" in terms:
        cols.append((omega_flat ** 2).reshape(-1, 1))
        names.append("omega^2")
    if "lap_omega" in terms:
        cols.append(lap_flat.reshape(-1, 1))
        names.append("laplacian_omega")
    if "advection" in terms:
        cols.append(adv_flat.reshape(-1, 1))
        names.append("advection")

    if not cols:
        cols = [omega_flat.reshape(-1, 1), lap_flat.reshape(-1, 1), adv_flat.reshape(-1, 1)]
        names = ["omega", "laplacian_omega", "advection"]
    return np.hstack(cols), names


def sparse_regression_pde(
    Theta: np.ndarray,
    target: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.01,
) -> Tuple[np.ndarray, str]:
    """
    Least-squares regression: target = Theta @ xi
    Returns (xi, formula_str)
    """
    target_flat = np.asarray(target, dtype=np.float64).ravel()
    assert Theta.shape[0] == len(target_flat)

    try:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(Theta, target_flat)
        xi = reg.coef_
    except ImportError:
        xi, _, _, _ = np.linalg.lstsq(Theta, target_flat, rcond=None)

    terms = []
    for i, (c, name) in enumerate(zip(xi, feature_names)):
        if abs(c) > threshold:
            terms.append(f"{c:.4g}*{name}")

    formula = "D = " + (" + ".join(terms) if terms else "0")
    return xi, formula
