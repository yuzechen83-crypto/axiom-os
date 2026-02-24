"""
2D Orszag-Tang MHD Data Generator (Level 5)
Standard MHD benchmark. Periodic [0, 2pi]^2.

Physics: Lorentz Force S = (∇×B)×B
  J_z = ∂By/∂x - ∂Bx/∂y,  Fx = J_z*By, Fy = -J_z*Bx

Orszag-Tang IC: u=-sin(y), v=sin(x), Bx=-sin(y), By=sin(2x)
Output: u, v, Bx, By, Fx_lorentz, Fy_lorentz (multiple snapshots via time variation)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple

N = 64
L = 2 * np.pi


def _compute_lorentz_force_numpy(Bx: np.ndarray, By: np.ndarray, L_domain: float) -> Tuple[np.ndarray, np.ndarray]:
    """J_z = ∂By/∂x - ∂Bx/∂y,  Fx = J_z*By, Fy = -J_z*Bx. Uses numpy gradient."""
    n = Bx.shape[-1]
    dx = dy = L_domain / n
    By_x = np.gradient(By, dx, axis=-1)
    Bx_y = np.gradient(Bx, dy, axis=-2)
    Jz = By_x - Bx_y
    Fx = Jz * By
    Fy = -Jz * Bx
    return Fx, Fy


def generate_orszag_tang_snapshots(
    n: int = N,
    L_domain: float = L,
    n_snapshots: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Orszag-Tang snapshots with slight time variation.
    Uses analytic IC + exponential decay for B to simulate diffusion.
    Returns: u, v, Bx, By, Fx_lorentz, Fy_lorentz each (n_snapshots, n, n)
    """
    y = np.linspace(0, L_domain, n, endpoint=False)
    x = np.linspace(0, L_domain, n, endpoint=False)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    t = np.linspace(0, 0.2, n_snapshots)
    decay = np.exp(-0.1 * t)[:, np.newaxis, np.newaxis]

    # Base IC
    u0 = -np.sin(yy)
    v0 = np.sin(xx)
    Bx0 = -np.sin(yy)
    By0 = np.sin(2 * xx)

    u = np.broadcast_to(u0, (n_snapshots, n, n)).copy()
    v = np.broadcast_to(v0, (n_snapshots, n, n)).copy()
    Bx = Bx0 * decay
    By = By0 * decay

    Fx = np.zeros((n_snapshots, n, n), dtype=np.float32)
    Fy = np.zeros((n_snapshots, n, n), dtype=np.float32)
    for i in range(n_snapshots):
        Fx[i], Fy[i] = _compute_lorentz_force_numpy(Bx[i], By[i], L_domain)

    return u.astype(np.float32), v.astype(np.float32), Bx.astype(np.float32), By.astype(np.float32), Fx, Fy


def main():
    out_dir = Path(__file__).resolve().parent
    print("Orszag-Tang MHD: Data Generator")
    print(f"  Grid: {N}x{N}, L={L}")

    u, v, Bx, By, Fx, Fy = generate_orszag_tang_snapshots(n=N, n_snapshots=20)

    torch.save(torch.from_numpy(u), out_dir / "u_field.pt")
    torch.save(torch.from_numpy(v), out_dir / "v_field.pt")
    torch.save(torch.from_numpy(Bx), out_dir / "Bx_field.pt")
    torch.save(torch.from_numpy(By), out_dir / "By_field.pt")
    torch.save(torch.from_numpy(Fx), out_dir / "Fx_lorentz.pt")
    torch.save(torch.from_numpy(Fy), out_dir / "Fy_lorentz.pt")

    Fmag = np.sqrt(Fx**2 + Fy**2)
    print(f"  Saved: u,v,Bx,By,Fx_lorentz,Fy_lorentz {u.shape}")
    print(f"  Lorentz |F| range: [{Fmag.min():.4f}, {Fmag.max():.4f}]")
    return u, v, Bx, By, Fx, Fy


if __name__ == "__main__":
    main()
