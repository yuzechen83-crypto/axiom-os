"""
Pseudo-Spectral Solver for 2D Kolmogorov Flow (Level 4)
Periodic boundary conditions. Vorticity formulation.

Physics: ∂ω/∂t + (u·∇)ω = ν∇²ω + f
  Forcing: f = -sin(ky), k=4
  ω = ∇×u (vorticity), u = -∂ψ/∂y, v = ∂ψ/∂x, ∇²ψ = -ω
  Re = 100 => ν = 0.01

Time integration: RK4 (default) or Crank-Nicolson (diffusion) + Adams-Bashforth (advection)
Outputs: u_field.pt, v_field.pt, w_field.pt for t in [0, 10]
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

# Default params (Re=100 => nu=0.01)
N = 64
L = 2 * np.pi
NU = 0.01  # viscosity, Re = U*L/nu ~ 100
K_FORCING = 4
T_END = 10.0
DT = 0.002  # CFL + diffusion limited


def _fft2_r2c(x: np.ndarray) -> np.ndarray:
    """Real 2D FFT. Returns full spectrum (no rfft half-save)."""
    return np.fft.fft2(x)


def _ifft2_c2r(x: np.ndarray) -> np.ndarray:
    """Inverse 2D FFT."""
    return np.fft.ifft2(x).real


def _compute_laplacian_spectral(wh_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """∇²ω in spectral space: -k² ω̂."""
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid div by zero
    return -k2 * wh_hat


def _compute_psi_from_omega(wh_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Solve ∇²ψ = -ω => ψ̂ = ω̂/k²."""
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    psi_hat = wh_hat / k2
    psi_hat[0, 0] = 0.0
    return psi_hat


def _compute_velocity_from_psi(psi: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """u = -∂ψ/∂y, v = ∂ψ/∂x."""
    psi_hat = _fft2_r2c(psi)
    u_hat = -1j * ky * psi_hat
    v_hat = 1j * kx * psi_hat
    u = _ifft2_c2r(u_hat)
    v = _ifft2_c2r(v_hat)
    return u, v


def _get_uv_from_omega(w: np.ndarray, kx_2d: np.ndarray, ky_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get u, v from vorticity ω via stream function."""
    w_hat = _fft2_r2c(w)
    psi_hat = _compute_psi_from_omega(w_hat, kx_2d, ky_2d)
    psi = _ifft2_c2r(psi_hat).real
    return _compute_velocity_from_psi(psi, kx_2d, ky_2d)


def _compute_advection_spectral(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, kx: np.ndarray, ky: np.ndarray
) -> np.ndarray:
    """(u·∇)ω in physical space, then FFT. Dealias with 2/3 rule."""
    wx = np.real(_ifft2_c2r(1j * kx * _fft2_r2c(w)))
    wy = np.real(_ifft2_c2r(1j * ky * _fft2_r2c(w)))
    adv = u * wx + v * wy
    adv_hat = _fft2_r2c(adv)
    # 2/3 dealiasing
    nx, ny = w.shape
    k_max = nx // 3
    kx_2d, ky_2d = np.meshgrid(
        np.fft.fftfreq(nx) * nx,
        np.fft.fftfreq(ny) * ny,
        indexing="ij",
    )
    mask = (np.abs(kx_2d) <= k_max) & (np.abs(ky_2d) <= k_max)
    adv_hat[~mask] = 0.0
    return adv_hat


def _forcing_spectral(shape: Tuple[int, int], k_forcing: int, L: float) -> np.ndarray:
    """f = -sin(ky) in physical space, then FFT."""
    ny, nx = shape
    y = np.linspace(0, L, ny, endpoint=False)
    x = np.linspace(0, L, nx, endpoint=False)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    f = -np.sin(k_forcing * yy)
    return _fft2_r2c(f)


def run_kolmogorov(
    n: int = N,
    L_domain: float = L,
    nu: float = NU,
    k_forcing: int = K_FORCING,
    t_end: float = T_END,
    dt: float = DT,
    seed: Optional[int] = None,
    method: str = "rk4",  # "rk4" or "cn_ab" (Crank-Nicolson + Adams-Bashforth)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pseudo-spectral integration of 2D Kolmogorov flow.
    method: "rk4" (default) or "cn_ab" (Crank-Nicolson diffusion + Adams-Bashforth advection)
    Returns: t, u, v, w (each u,v,w shape (n_steps, n, n))
    """
    if seed is not None:
        np.random.seed(seed)

    # Wavenumbers
    kx = np.fft.fftfreq(n) * (2 * np.pi * n / L_domain)
    ky = np.fft.fftfreq(n) * (2 * np.pi * n / L_domain)
    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx_2d**2 + ky_2d**2
    k2[0, 0] = 1.0

    # Initial condition: small random vorticity
    w = 0.1 * np.random.randn(n, n)
    w_hat = _fft2_r2c(w)
    # Zero mean
    w_hat[0, 0] = 0.0
    # Dealias init
    k_max = n // 3
    mask = (np.abs(kx_2d) <= k_max) & (np.abs(ky_2d) <= k_max)
    w_hat[~mask] = 0.0
    w = _ifft2_c2r(w_hat).real

    # Forcing (constant in time)
    f_hat = _forcing_spectral((n, n), k_forcing, L_domain)
    f_hat[~mask] = 0.0

    adv_hat_prev = None  # For Adams-Bashforth 2
    n_steps = int(t_end / dt) + 1
    u_all = np.zeros((n_steps, n, n), dtype=np.float32)
    v_all = np.zeros((n_steps, n, n), dtype=np.float32)
    w_all = np.zeros((n_steps, n, n), dtype=np.float32)

    for step in range(n_steps):
        w_all[step] = w.copy()
        psi_hat = _compute_psi_from_omega(_fft2_r2c(w), kx_2d, ky_2d)
        u, v = _compute_velocity_from_psi(_ifft2_c2r(psi_hat).real, kx_2d, ky_2d)
        u_all[step] = u
        v_all[step] = v

        if step == n_steps - 1:
            break

        adv_hat = _compute_advection_spectral(u, v, w, kx_2d, ky_2d)
        lap_hat = _compute_laplacian_spectral(w_hat, kx_2d, ky_2d)

        if method == "cn_ab":
            # Crank-Nicolson for diffusion (implicit), Adams-Bashforth 2 for advection
            # (1 + dt*nu/2 * k2) w_new = (1 - dt*nu/2 * k2) w_old + dt*(1.5*(-adv) - 0.5*(-adv_prev)) + dt*f
            if adv_hat_prev is None:
                adv_hat_prev = adv_hat.copy()  # Euler for first step
            # Implicit diffusion: (1 - a*L)w_new = RHS, a = dt*nu/2, L = -k2
            a = 0.5 * dt * nu
            rhs_hat = (1 - a * (-k2)) * w_hat + dt * (1.5 * (-adv_hat) - 0.5 * (-adv_hat_prev)) + dt * f_hat
            rhs_hat[~mask] = 0.0
            denom = 1 + a * k2
            denom[0, 0] = 1.0
            w_hat_new = rhs_hat / denom
            adv_hat_prev = adv_hat.copy()
        else:
            # RK4 (default)
            k1 = -adv_hat + nu * lap_hat + f_hat
            w1 = _ifft2_c2r(w_hat + 0.5 * dt * k1).real
            u1, v1 = _get_uv_from_omega(w1, kx_2d, ky_2d)
            adv1 = _compute_advection_spectral(u1, v1, w1, kx_2d, ky_2d)
            w1_hat = _fft2_r2c(w1)
            lap1 = _compute_laplacian_spectral(w1_hat, kx_2d, ky_2d)
            k2_val = -adv1 + nu * lap1 + f_hat

            w2 = _ifft2_c2r(w_hat + 0.5 * dt * k2_val).real
            u2, v2 = _get_uv_from_omega(w2, kx_2d, ky_2d)
            adv2 = _compute_advection_spectral(u2, v2, w2, kx_2d, ky_2d)
            w2_hat = _fft2_r2c(w2)
            lap2 = _compute_laplacian_spectral(w2_hat, kx_2d, ky_2d)
            k3_val = -adv2 + nu * lap2 + f_hat

            w3 = _ifft2_c2r(w_hat + dt * k3_val).real
            u3, v3 = _get_uv_from_omega(w3, kx_2d, ky_2d)
            adv3 = _compute_advection_spectral(u3, v3, w3, kx_2d, ky_2d)
            w3_hat = _fft2_r2c(w3)
            lap3 = _compute_laplacian_spectral(w3_hat, kx_2d, ky_2d)
            k4_val = -adv3 + nu * lap3 + f_hat

            w_hat_new = w_hat + (dt / 6.0) * (k1 + 2 * k2_val + 2 * k3_val + k4_val)
        w_hat_new[~mask] = 0.0
        w = _ifft2_c2r(w_hat_new).real

    t = np.linspace(0, t_end, n_steps)
    return t, u_all, v_all, w_all


def main():
    out_dir = Path(__file__).resolve().parent
    print("Kolmogorov Flow: Pseudo-Spectral Solver")
    print(f"  Grid: {N}x{N}, L={L}, nu={NU}, k={K_FORCING}, T={T_END}s")

    t, u, v, w = run_kolmogorov(n=N, L_domain=L, nu=NU, k_forcing=K_FORCING, t_end=T_END, seed=42)

    # Save as PyTorch tensors
    torch.save(torch.from_numpy(u), out_dir / "u_field.pt")
    torch.save(torch.from_numpy(v), out_dir / "v_field.pt")
    torch.save(torch.from_numpy(w), out_dir / "w_field.pt")

    print(f"  Saved: u_field.pt {u.shape}, v_field.pt {v.shape}, w_field.pt {w.shape}")
    print(f"  omega range: [{w.min():.4f}, {w.max():.4f}]")
    return u, v, w


if __name__ == "__main__":
    main()
