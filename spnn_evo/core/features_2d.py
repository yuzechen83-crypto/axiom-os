"""
2D Feature Engineering for Navier-Stokes Discovery (Level 4)
Input: w (vorticity) shape [Batch, H, W]

Operators:
  grad_x(w), grad_y(w): ∂xω, ∂yω via torch.gradient
  laplacian(w): ∂xxω + ∂yyω = ∇²ω
  advection(u, v, w): u·∂xω + v·∂yω = (u·∇)ω

Weak Form: patch averaging for noise-robust discovery.
"""

import torch
from typing import Tuple, Optional


def grad_x(w: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """∂ω/∂x along last axis."""
    return torch.gradient(w, spacing=dx, dim=-1)[0]


def grad_y(w: torch.Tensor, dy: float = 1.0) -> torch.Tensor:
    """∂ω/∂y along second-to-last axis."""
    return torch.gradient(w, spacing=dy, dim=-2)[0]


def laplacian(w: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """∇²ω = ∂xxω + ∂yyω"""
    wx = grad_x(w, dx)
    wy = grad_y(w, dy)
    wxx = torch.gradient(wx, spacing=dx, dim=-1)[0]
    wyy = torch.gradient(wy, spacing=dy, dim=-2)[0]
    return wxx + wyy


def advection(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """(u·∇)ω = u·∂xω + v·∂yω"""
    wx = grad_x(w, dx)
    wy = grad_y(w, dy)
    return u * wx + v * wy


def vorticity_from_velocity(u: torch.Tensor, v: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """ω = ∇×u = ∂v/∂x - ∂u/∂y (2D scalar vorticity)"""
    v_x = grad_x(v, dx)
    u_y = grad_y(u, dy)
    return v_x - u_y


def curl_B_2d(Bx: torch.Tensor, By: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """J_z = ∇×B = ∂By/∂x - ∂Bx/∂y (2D out-of-plane current)"""
    By_x = grad_x(By, dx)
    Bx_y = grad_y(Bx, dy)
    return By_x - Bx_y


def lorentz_force_2d(
    Bx: torch.Tensor,
    By: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """S = (∇×B)×B: Fx = J_z*By, Fy = -J_z*Bx"""
    Jz = curl_B_2d(Bx, By, dx, dy)
    Fx = Jz * By
    Fy = -Jz * Bx
    return Fx, Fy


def gaussian_smooth(
    field: torch.Tensor,
    sigma: float = 1.0,
    kernel_size: int = 5,
) -> torch.Tensor:
    """
    Physics Filter: Gaussian smoothing to reduce derivative noise.
    Applied before computing Laplacian to tame ∇² amplification.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    x = torch.arange(kernel_size, dtype=field.dtype, device=field.device) - pad
    g1d = torch.exp(-0.5 * (x / sigma) ** 2)
    g1d = g1d / g1d.sum()
    # 2D separable kernel
    kernel = g1d.unsqueeze(0) * g1d.unsqueeze(1)  # (k, k)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    # field: (..., H, W) -> need (N, 1, H, W) for conv2d
    need_batch = field.dim() == 2
    if need_batch:
        field = field.unsqueeze(0)
    f = field.unsqueeze(1)  # (T, 1, H, W)
    f = torch.nn.functional.pad(f, (pad, pad, pad, pad), mode="replicate")
    f = torch.nn.functional.conv2d(f, kernel, padding=0)
    f = f.squeeze(1)
    if need_batch:
        f = f.squeeze(0)
    return f


def sobolev_filter(
    field: torch.Tensor,
    alpha: float = 0.05,
    dx: float = 1.0,
    dy: float = 1.0,
    n_iter: int = 3,
) -> torch.Tensor:
    """
    Physics Filter: (1 + α∇²)^{-1} - low-pass, damps high-k.
    Approximated via Jacobi: u_new = u - α*lap(u) (damps high frequencies).
    """
    for _ in range(n_iter):
        lap_f = laplacian(field, dx, dy)
        field = field - alpha * lap_f
    return field


def calc_2d_derivatives(
    w: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    dim_x: int = -1,
    dim_y: int = -2,
) -> dict:
    """
    Compute 2D derivatives for vorticity ω and optional advection.

    Args:
        w: Vorticity field, shape (..., H, W) or (T, H, W)
        u, v: Velocity components (optional, for advection)
        dx, dy: Grid spacing
        dim_x: Axis for x (default -1)
        dim_y: Axis for y (default -2)

    Returns:
        dict with:
          omega_x: ∂ω/∂x
          omega_y: ∂ω/∂y
          lap_omega: ∇²ω = ∂xxω + ∂yyω
          advection: (u·∇)ω = u·ω_x + v·ω_y  (if u,v provided)
    """
    out = {}

    # ∂ω/∂x, ∂ω/∂y via torch.gradient
    # gradient returns list of gradients along each dimension
    grads = torch.gradient(w, spacing=(dy, dx), dim=(-2, -1))
    omega_y = grads[0]  # gradient along dim -2 (y)
    omega_x = grads[1]  # gradient along dim -1 (x)
    out["omega_x"] = omega_x
    out["omega_y"] = omega_y

    # Laplacian ∇²ω = ∂xxω + ∂yyω
    omega_xx = torch.gradient(omega_x, spacing=dx, dim=-1)[0]
    omega_yy = torch.gradient(omega_y, spacing=dy, dim=-2)[0]
    lap_omega = omega_xx + omega_yy
    out["lap_omega"] = lap_omega

    # Advection (u·∇)ω = u·ω_x + v·ω_y
    if u is not None and v is not None:
        advection = u * omega_x + v * omega_y
        out["advection"] = advection

    return out


def weak_form_patch_average(
    field: torch.Tensor,
    patch_size: int = 4,
) -> torch.Tensor:
    """
    Weak form: spatial average over non-overlapping patches.
    Reduces noise amplification from ∇² (second derivative).

    Args:
        field: (T, H, W) or (H, W)
        patch_size: patch side length (e.g., 4 -> 4x4 patches)

    Returns:
        Averaged field, shape reduced by patch_size in H, W
    """
    if field.dim() == 2:
        field = field.unsqueeze(0)
    t, h, w = field.shape
    # Trim to divisible
    h_trim = (h // patch_size) * patch_size
    w_trim = (w // patch_size) * patch_size
    f = field[:, :h_trim, :w_trim]
    # Reshape: (T, H/p, p, W/p, p) -> (T, H/p, W/p, p*p) -> mean over last dim
    f = f.reshape(t, h_trim // patch_size, patch_size, w_trim // patch_size, patch_size)
    f = f.permute(0, 1, 3, 2, 4).reshape(t, h_trim // patch_size, w_trim // patch_size, -1)
    return f.mean(dim=-1)


def calc_2d_derivatives_numpy(
    w: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> dict:
    """
    NumPy-based fallback. Converts to numpy, uses np.gradient, returns tensors.
    """
    import numpy as np
    w_np = w.detach().cpu().numpy()
    omega_x_np = np.gradient(w_np, dx, axis=-1)
    omega_y_np = np.gradient(w_np, dy, axis=-2)
    omega_xx_np = np.gradient(omega_x_np, dx, axis=-1)
    omega_yy_np = np.gradient(omega_y_np, dy, axis=-2)
    lap_omega_np = omega_xx_np + omega_yy_np

    out = {
        "omega_x": torch.from_numpy(omega_x_np).to(w.device, w.dtype),
        "omega_y": torch.from_numpy(omega_y_np).to(w.device, w.dtype),
        "lap_omega": torch.from_numpy(lap_omega_np).to(w.device, w.dtype),
    }
    if u is not None and v is not None:
        u_np = u.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        advection_np = u_np * omega_x_np + v_np * omega_y_np
        out["advection"] = torch.from_numpy(advection_np).to(w.device, w.dtype)
    return out
