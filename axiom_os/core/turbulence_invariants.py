"""
Turbulence Invariants and Tensor Basis (Pope 1975, Ling et al. 2016 TBNN)

Galilean invariant and rotationally equivariant features for Reynolds stress modeling.
Decompose: grad_u = S + Omega (symmetric strain + antisymmetric rotation).

Input Invariants (scalar, Galilean invariant):
  lambda1 = Tr(S^2)
  lambda2 = Tr(Omega^2)
  lambda3 = Tr(S^3)
  lambda4 = Tr(S Omega^2)
  lambda5 = Tr(S^2 Omega^2)

Tensor Basis (symmetric, rotationally equivariant):
  T1 = S
  T2 = S*Omega - Omega*S
  T3 = S^2 - (1/3)Tr(S^2)*I
  T4 = Omega^2 - (1/3)Tr(Omega^2)*I
  T5 = S*Omega^2 - Omega^2*S
  T6 = S^2*Omega - Omega*S^2
  T7 = S^2*Omega^2 + Omega^2*S^2 - (2/3)Tr(S^2*Omega^2)*I
  T8 = S*Omega*S^2 - S^2*Omega*S
  T9 = Omega*S*Omega^2 - Omega^2*S*Omega
  T10 = Omega*S^2*Omega^2 - Omega^2*S^2*Omega

Output: tau_ij = sum_n g_n(lambda) * T_n  (TBNN)
"""

from typing import Tuple, List
import numpy as np


def grad_u_from_velocity(u: np.ndarray) -> np.ndarray:
    """
    Compute velocity gradient tensor from u. u: (..., 3).
    Returns grad_u: (..., 3, 3) where [..., i, j] = du_i/dx_j.
    """
    shape = u.shape[:-1] + (3, 3)
    grad_u = np.zeros(shape, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            grad_u[..., i, j] = np.gradient(u[..., i], axis=j)
    return grad_u


def decompose_grad_u(grad_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose grad_u = S + Omega.
    S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)  (symmetric strain)
    Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)  (antisymmetric rotation)
    """
    S = 0.5 * (grad_u + np.swapaxes(grad_u, -2, -1))
    Omega = 0.5 * (grad_u - np.swapaxes(grad_u, -2, -1))
    return S, Omega


def matmul_3x3(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batch matrix multiply (...,3,3) x (...,3,3) -> (...,3,3)."""
    return np.einsum("...ij,...jk->...ik", A, B)


def trace_3x3(A: np.ndarray) -> np.ndarray:
    """Tr(A) for (...,3,3)."""
    return np.einsum("...ii->...", A)


def identity_3x3(shape: Tuple) -> np.ndarray:
    """Identity (...,3,3) with leading shape."""
    I = np.zeros(shape + (3, 3), dtype=np.float64)
    I[..., 0, 0] = 1
    I[..., 1, 1] = 1
    I[..., 2, 2] = 1
    return I


EPS = 1e-12


def compute_sigma(S: np.ndarray) -> np.ndarray:
    """Characteristic scale: sigma = sqrt(Tr(S^2) + eps)."""
    S2 = matmul_3x3(S, S)
    tr_S2 = trace_3x3(S2)
    return np.sqrt(np.maximum(tr_S2, 0.0) + EPS)


def compute_invariants(S: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """
    Pope invariants (first 5). Returns (..., 5).
    """
    S2 = matmul_3x3(S, S)
    S3 = matmul_3x3(S2, S)
    O2 = matmul_3x3(Omega, Omega)
    SO2 = matmul_3x3(S, O2)
    S2O2 = matmul_3x3(S2, O2)

    lam1 = trace_3x3(S2)
    lam2 = trace_3x3(O2)
    lam3 = trace_3x3(S3)
    lam4 = trace_3x3(SO2)
    lam5 = trace_3x3(S2O2)

    return np.stack([lam1, lam2, lam3, lam4, lam5], axis=-1)


def compute_invariants_normalized(S: np.ndarray, Omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-dimensional invariants: lambda_hat_n = lambda_n / sigma^order.
    Returns (invariants_normalized (..., 5), sigma (...,)).
    """
    sigma = compute_sigma(S)
    inv_raw = compute_invariants(S, Omega)
    lam1, lam2, lam3, lam4, lam5 = inv_raw[..., 0], inv_raw[..., 1], inv_raw[..., 2], inv_raw[..., 3], inv_raw[..., 4]
    sig2 = sigma**2
    sig3 = sigma**3
    sig4 = sigma**4
    inv_norm = np.stack([
        lam1 / (sig2 + EPS),   # lambda1 ~ 1
        lam2 / (sig2 + EPS),
        lam3 / (sig3 + EPS),
        lam4 / (sig3 + EPS),
        lam5 / (sig4 + EPS),
    ], axis=-1)
    return inv_norm, sigma


def compute_tensor_basis(S: np.ndarray, Omega: np.ndarray) -> List[np.ndarray]:
    """
    Tensor basis T1..T10. Each T_n: (..., 3, 3).
    Returns list of 10 tensors.
    """
    S2 = matmul_3x3(S, S)
    O2 = matmul_3x3(Omega, Omega)
    SO = matmul_3x3(S, Omega)
    OS = matmul_3x3(Omega, S)
    S2O = matmul_3x3(S2, Omega)
    OS2 = matmul_3x3(Omega, S2)
    SO2 = matmul_3x3(S, O2)
    O2S = matmul_3x3(O2, S)
    S2O2 = matmul_3x3(S2, O2)
    O2S2 = matmul_3x3(O2, S2)
    SO2S = matmul_3x3(SO2, S)
    S2OS = matmul_3x3(S2, OS)
    OSO2 = matmul_3x3(OS, O2)
    O2SO = matmul_3x3(O2, SO)
    OS2O2 = matmul_3x3(OS2, O2)
    O2S2O = matmul_3x3(O2, S2O)

    lead_shape = S.shape[:-2]
    I = identity_3x3(lead_shape)

    tr_S2 = trace_3x3(S2)
    tr_O2 = trace_3x3(O2)
    tr_S2O2 = trace_3x3(S2O2)

    T1 = S
    T2 = SO - OS
    T3 = S2 - (1.0 / 3.0) * tr_S2[..., None, None] * I
    T4 = O2 - (1.0 / 3.0) * tr_O2[..., None, None] * I
    T5 = SO2 - O2S
    T6 = S2O - OS2
    T7 = S2O2 + O2S2 - (2.0 / 3.0) * tr_S2O2[..., None, None] * I
    T8 = SO2S - S2OS
    T9 = OSO2 - O2SO
    T10 = OS2O2 - O2S2O

    return [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]


def compute_tensor_basis_normalized(S: np.ndarray, Omega: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Normalized tensor basis: T_hat_n = T_n / sigma^order.
    Order: T1,S2,T3,T4=1,2; T5,T6,T8,T9=3; T7,T10=4.
    Returns (list of 10 tensors, sigma).
    """
    sigma = compute_sigma(S)
    basis = compute_tensor_basis(S, Omega)
    sig2 = sigma[..., None, None] ** 2
    sig3 = sigma[..., None, None] ** 3
    sig4 = sigma[..., None, None] ** 4
    T1, T2, T3, T4, T5, T6, T7, T8, T9, T10 = basis
    basis_norm = [
        T1 / (sigma[..., None, None] + EPS),
        T2 / (sig2 + EPS),
        T3 / (sig2 + EPS),
        T4 / (sig2 + EPS),
        T5 / (sig3 + EPS),
        T6 / (sig3 + EPS),
        T7 / (sig4 + EPS),
        T8 / (sig3 + EPS),
        T9 / (sig3 + EPS),
        T10 / (sig4 + EPS),
    ]
    return basis_norm, sigma


def extract_invariants_and_basis(u: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    From velocity field u (..., 3), compute invariants and tensor basis.
    Returns: invariants (..., 5), list of 10 tensors (..., 3, 3).
    """
    grad_u = grad_u_from_velocity(u)
    S, Omega = decompose_grad_u(grad_u)
    inv = compute_invariants(S, Omega)
    basis = compute_tensor_basis(S, Omega)
    return inv, basis


def extract_invariants_and_basis_normalized(u: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Non-dimensional invariants and tensor basis.
    Returns: invariants_normalized (..., 5), basis_normalized (10 tensors), sigma (...,).
    tau_ij = sigma^2 * sum_n g_n(lambda_hat) * T_hat_n
    """
    grad_u = grad_u_from_velocity(u)
    S, Omega = decompose_grad_u(grad_u)
    inv_norm, sigma = compute_invariants_normalized(S, Omega)
    basis_norm, _ = compute_tensor_basis_normalized(S, Omega)
    return inv_norm, basis_norm, sigma
