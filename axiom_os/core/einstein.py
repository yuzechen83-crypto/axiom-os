"""
Einstein - Symplectic & Metriplectic Integrators
Mental Simulation using structure-preserving physics.

Metriplectic (GENERIC) formulation:
  dx/dt = L∇E + M∇S
  - E: energy potential, S: entropy potential
  - L: skew-symmetric (Poisson), M: symmetric PSD (metric)
  - Degeneracy: L∇S = 0 (reversible part preserves entropy)
  - Degeneracy: M∇E = 0 (dissipative part preserves energy)

Hamiltonian case: H only, L = J (symplectic), S = const.
"""

from typing import Callable, Tuple, Optional, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _grad_scalar_numpy(
    f: Callable,
    x: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute gradient of scalar f(x) via finite differences."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        g[i] = (float(np.asarray(f(x_plus)).ravel()[0]) - float(np.asarray(f(x_minus)).ravel()[0])) / (2 * eps)
    return g


def _grad_H_numpy(H_func: Callable, qp: np.ndarray, state_dim: int, eps: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dH/dq and dH/dp via finite differences. Backward compat."""
    g = _grad_scalar_numpy(H_func, qp, eps)
    return g[:state_dim], g[state_dim:]


def _skew(A: np.ndarray) -> np.ndarray:
    """A - A^T: ensures skew-symmetric matrix."""
    return np.asarray(A, dtype=np.float64) - np.asarray(A, dtype=np.float64).T


def _sym_psd(B: np.ndarray) -> np.ndarray:
    """B @ B^T: ensures symmetric positive semi-definite matrix."""
    B = np.asarray(B, dtype=np.float64)
    return B @ B.T


def _project_orthogonal(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """P = I - v v^T / (v^T v + eps). Projects onto orthogonal complement of v."""
    v = np.asarray(v, dtype=np.float64).ravel()
    n = len(v)
    norm_sq = np.dot(v, v) + eps
    if norm_sq < eps:
        return np.eye(n)  # v ≈ 0: no projection
    vv = np.outer(v, v)
    return np.eye(n) - vv / norm_sq


class MetriplecticStructure:
    """
    Two potentials (E, S) and two matrices (L, M) with degeneracy enforcement.
    Dynamics: dx/dt = L∇E + M∇S
    Degeneracy: L∇S = 0, M∇E = 0
    """

    def __init__(
        self,
        E_func: Callable[[np.ndarray], float],
        S_func: Callable[[np.ndarray], float],
        L_raw_func: Callable[[np.ndarray], np.ndarray],
        M_raw_func: Callable[[np.ndarray], np.ndarray],
        dim: int,
        eps: float = 1e-7,
    ):
        """
        Args:
            E_func: Energy potential E(x), returns scalar
            S_func: Entropy potential S(x), returns scalar
            L_raw_func: Raw matrix for Poisson structure, returns (dim,dim). Will be made skew.
            M_raw_func: Raw matrix for metric structure, returns (dim,dim). Will be made sym PSD.
            dim: Phase space dimension (len of x)
            eps: Finite-diff step and numerical stability
        """
        self.E_func = E_func
        self.S_func = S_func
        self.L_raw_func = L_raw_func
        self.M_raw_func = M_raw_func
        self.dim = dim
        self.eps = eps

    def grad_E(self, x: np.ndarray) -> np.ndarray:
        """∇E at x."""
        return _grad_scalar_numpy(self.E_func, x, self.eps)

    def grad_S(self, x: np.ndarray) -> np.ndarray:
        """∇S at x."""
        return _grad_scalar_numpy(self.S_func, x, self.eps)

    def get_L(self, x: np.ndarray) -> np.ndarray:
        """
        Skew-symmetric L with degeneracy L∇S = 0.
        L = P_S @ (L_raw - L_raw^T) @ P_S, where P_S projects orthogonally to ∇S.
        """
        grad_S = self.grad_S(x)
        L_raw = np.asarray(self.L_raw_func(x), dtype=np.float64)
        if L_raw.shape != (self.dim, self.dim):
            L_raw = np.eye(self.dim) * 0.01  # fallback
        L_skew = _skew(L_raw)
        P_S = _project_orthogonal(grad_S, self.eps)
        return P_S @ L_skew @ P_S

    def get_M(self, x: np.ndarray) -> np.ndarray:
        """
        Symmetric PSD M with degeneracy M∇E = 0.
        M = P_E @ M_raw @ P_E, where P_E projects orthogonally to ∇E.
        """
        grad_E = self.grad_E(x)
        M_raw = np.asarray(self.M_raw_func(x), dtype=np.float64)
        if M_raw.shape != (self.dim, self.dim):
            M_raw = np.eye(self.dim) * 0.01
        M_sym = _sym_psd(M_raw)
        P_E = _project_orthogonal(grad_E, self.eps)
        return P_E @ M_sym @ P_E

    def dynamics(self, x: np.ndarray) -> np.ndarray:
        """dx/dt = L∇E + M∇S."""
        L = self.get_L(x)
        M = self.get_M(x)
        gE = self.grad_E(x)
        gS = self.grad_S(x)
        return L @ gE + M @ gS

    def degeneracy_violation(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Returns (||L∇S||^2, ||M∇E||^2). Should be ~0 when degeneracy holds.
        """
        L = self.get_L(x)
        M = self.get_M(x)
        gE = self.grad_E(x)
        gS = self.grad_S(x)
        LgS = np.dot(L @ gS, L @ gS)
        MgE = np.dot(M @ gE, M @ gE)
        return LgS, MgE


def _default_L_raw(dim: int) -> Callable:
    """Default L_raw: canonical symplectic J = [[0,I],[-I,0]] for 2*state_dim."""
    n = dim // 2
    J = np.zeros((dim, dim))
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)

    def L_raw(x):
        return J
    return L_raw


def _default_M_raw(dim: int) -> Callable:
    """Default M_raw: zero (no dissipation)."""

    def M_raw(x):
        return np.zeros((dim, dim))
    return M_raw


def metriplectic_from_H(
    H_func: Callable,
    state_dim: int,
    S_func: Optional[Callable] = None,
    M_raw_func: Optional[Callable] = None,
    eps: float = 1e-7,
) -> MetriplecticStructure:
    """
    Build MetriplecticStructure from Hamiltonian H.
    E = H, S = 0 (or custom), L = J (symplectic), M = 0 (or custom).
    """
    dim = 2 * state_dim

    def E(x):
        return float(np.asarray(H_func(x)).ravel()[0])

    if S_func is None:
        def S(x):
            return 0.0
    else:
        S = S_func

    L_raw = _default_L_raw(dim)
    if M_raw_func is None:
        M_raw = _default_M_raw(dim)
    else:
        M_raw = M_raw_func

    return MetriplecticStructure(E, S, L_raw, M_raw, dim, eps)


class SymplecticIntegrator:
    """
    Leapfrog/Verlet integrator for Hamiltonian systems.
    Preserves symplectic structure (energy conservation).
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt

    def step(
        self,
        q: np.ndarray,
        p: np.ndarray,
        H: Callable,
        grad_H: Callable,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single symplectic step."""
        q = np.asarray(q, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        qp = np.concatenate([q, p])
        dH_dq, dH_dp = grad_H(qp)

        p_half = p - 0.5 * self.dt * dH_dq
        qp_mid = np.concatenate([q, p_half])
        _, dH_dp_mid = grad_H(qp_mid)

        q_new = q + self.dt * dH_dp_mid
        qp_end = np.concatenate([q_new, p_half])
        dH_dq_end, _ = grad_H(qp_end)

        p_new = p_half - 0.5 * self.dt * dH_dq_end
        return q_new, p_new


class EinsteinCore:
    """
    Symplectic & Metriplectic Reasoner.
    - step_leapfrog: Hamiltonian H only (backward compat)
    - step_metriplectic: Full E, S, L, M with degeneracy
    - decompose_dynamics: Placeholder
    """

    def __init__(self, state_dim: int = 2, eps: float = 1e-7):
        self.state_dim = state_dim
        self.eps = eps

    @staticmethod
    def step_leapfrog(
        H_func: Callable,
        q: np.ndarray,
        p: np.ndarray,
        dt: float,
        state_dim: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Symplectic leapfrog step for Hamiltonian H(q,p).
        Backward compatible with H-only formulation.
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        p = np.asarray(p, dtype=np.float64).ravel()
        nq = len(q)
        if state_dim is None:
            state_dim = nq

        def grad_H(qp):
            return _grad_H_numpy(H_func, qp, state_dim)

        qp = np.concatenate([q, p])
        dH_dq, dH_dp = grad_H(qp)
        p_half = p - 0.5 * dt * dH_dq

        qp_mid = np.concatenate([q, p_half])
        _, dH_dp_mid = grad_H(qp_mid)
        q_new = q + dt * dH_dp_mid

        qp_end = np.concatenate([q_new, p_half])
        dH_dq_end, _ = grad_H(qp_end)
        p_new = p_half - 0.5 * dt * dH_dq_end

        return q_new, p_new

    @staticmethod
    def step_metriplectic(
        struct: MetriplecticStructure,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Euler or symplectic-like step for metriplectic dynamics.
        dx/dt = L∇E + M∇S with degeneracy L∇S=0, M∇E=0.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        dx = struct.dynamics(x)
        return x + dt * dx

    def decompose_dynamics(
        self,
        soft_shell_model: Callable,
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        """
        (Placeholder) Decompose learned dynamics into Conservative and Dissipative.
        Returns (H_func_or_None, dissipative_func_or_None).
        """
        return None, None
