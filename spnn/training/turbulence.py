"""
湍流训练 - Turbulence Training
Burgers 方程 / 合成湍流数据 + 物理残差
∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from ..model import SPNN


@dataclass
class TurbulenceConfig:
    """湍流训练配置"""
    n_t: int = 50
    n_x: int = 64
    domain_t: Tuple[float, float] = (0.0, 1.0)
    domain_x: Tuple[float, float] = (0.0, 1.0)
    nu: float = 0.01       # 粘度
    n_modes: int = 8       # 合成湍流模态数
    reynolds: float = 1000 # 雷诺数 (用于尺度)


def generate_burgers_turbulence(
    n_t: int = 50,
    n_x: int = 64,
    domain_t: Tuple[float, float] = (0.0, 1.0),
    domain_x: Tuple[float, float] = (0.0, 1.0),
    nu: float = 0.01,
    n_modes: int = 8,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    合成湍流数据 (1D Burgers 近似)
    u(t,x) = Σ_k A_k sin(kπx/L) exp(-νk²π²t/L²) + 湍流扰动
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(domain_t[0], domain_t[1], n_t)
    x = np.linspace(domain_x[0], domain_x[1], n_x)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    L = domain_x[1] - domain_x[0]

    # 基模 + 湍流扰动
    u = np.zeros_like(tt)
    for k in range(1, n_modes + 1):
        A_k = 1.0 / k * np.exp(-0.1 * k)
        decay = np.exp(-nu * (k * np.pi / L) ** 2 * tt)
        u += A_k * np.sin(k * np.pi * xx / L) * decay

    # 湍流扰动 (Kolmogorov 谱风格)
    for k in range(2, min(n_modes + 2, 15)):
        phase = np.random.rand() * 2 * np.pi
        amp = 0.1 * (k ** (-5/3))  # Kolmogorov -5/3
        u += amp * np.sin(k * np.pi * xx / L + 2 * np.pi * tt / 0.5 + phase)
    u += 0.05 * np.random.randn(*u.shape) * np.exp(-tt)  # 衰减噪声

    coords = np.stack([tt.ravel(), xx.ravel()], axis=1)
    u_flat = u.ravel().reshape(-1, 1)
    return coords.astype(np.float32), u_flat.astype(np.float32)


def generate_2d_turbulence(
    n_t: int = 30,
    n_x: int = 32,
    n_y: int = 32,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    n_modes: int = 5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D 合成湍流 (u, v)
    coords: (t, x, y), output: (u, v)
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, 1, n_t)
    x = np.linspace(domain[0], domain[1], n_x)
    y = np.linspace(domain[2], domain[3], n_y)
    tt, xx, yy = np.meshgrid(t, x, y, indexing="ij")
    tt = tt.ravel()
    xx = xx.ravel()
    yy = yy.ravel()

    u = np.zeros_like(tt)
    v = np.zeros_like(tt)
    for kx in range(1, n_modes + 1):
        for ky in range(1, n_modes + 1):
            amp = 0.1 / (kx * ky)
            phase = np.random.rand() * 2 * np.pi
            u += amp * np.sin(kx * np.pi * xx) * np.cos(ky * np.pi * yy) * np.cos(2 * np.pi * tt + phase)
            v += amp * np.cos(kx * np.pi * xx) * np.sin(ky * np.pi * yy) * np.cos(2 * np.pi * tt + phase)
    u += 0.02 * np.random.randn(len(tt))
    v += 0.02 * np.random.randn(len(tt))

    coords = np.stack([tt, xx, yy], axis=1).astype(np.float32)
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return coords, uv


def burgers_residual_1d(
    model: torch.nn.Module,
    coords: torch.Tensor,
    nu: float = 0.01,
) -> torch.Tensor:
    """
    Burgers 残差: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x²
    coords: (N, 2) = (t, x)
    """
    coords = coords.detach().requires_grad_(True)
    out = model(coords)
    u = out[0] if isinstance(out, tuple) else out
    if u.dim() > 1 and u.shape[-1] > 1:
        u = u[:, 0:1]
    u = u.squeeze(-1)

    (grad_u,) = torch.autograd.grad(u.sum(), coords, create_graph=True)
    u_t = grad_u[:, 0]
    u_x = grad_u[:, 1]

    (grad_ux,) = torch.autograd.grad(u_x.sum(), coords, create_graph=True)
    u_xx = grad_ux[:, 1]

    residual = u_t + u * u_x - nu * u_xx
    return torch.mean(residual ** 2)


class TurbulenceLoss(torch.nn.Module):
    """湍流多目标损失: L_data + λ_phys·L_burgers"""

    def __init__(self, lambda_phys: float = 0.1):
        super().__init__()
        self.lambda_phys = lambda_phys

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        coords_colloc: Optional[torch.Tensor] = None,
        model: Optional[torch.nn.Module] = None,
        nu: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        l_data = torch.nn.functional.mse_loss(pred, target)
        l_phys = torch.tensor(0.0, device=pred.device)
        if coords_colloc is not None and model is not None:
            l_phys = burgers_residual_1d(model, coords_colloc, nu=nu)
        return {
            "data": l_data,
            "phys": l_phys,
            "total": l_data + self.lambda_phys * l_phys,
        }


def run_turbulence_training(
    model: SPNN,
    config: Optional[TurbulenceConfig] = None,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    lambda_phys: float = 0.1,
    use_2d: bool = False,
    use_physics_residual: bool = False,
    physics_model: Optional[torch.nn.Module] = None,
) -> SPNN:
    """
    湍流训练主循环
    """
    config = config or TurbulenceConfig()
    device = next(model.parameters()).device

    if use_2d:
        coords, targets = generate_2d_turbulence(
            n_t=config.n_t,
            n_x=config.n_x,
            n_y=config.n_x,
            n_modes=config.n_modes,
        )
        # 若 model out_dim=1 需改为 2
        if model.out_dim == 1:
            targets = targets[:, 0:1]  # 仅 u 分量
    else:
        coords, targets = generate_burgers_turbulence(
            n_t=config.n_t,
            n_x=config.n_x,
            domain_t=config.domain_t,
            domain_x=config.domain_x,
            nu=config.nu,
            n_modes=config.n_modes,
        )

    X = torch.as_tensor(coords, device=device)
    Y = torch.as_tensor(targets, device=device)

    # 配点 (collocation)
    n_colloc = min(512, len(coords) // 2)
    idx = np.random.choice(len(coords), n_colloc, replace=False)
    coords_colloc = torch.as_tensor(coords[idx], device=device)

    colloc_for_phys = coords_colloc if use_physics_residual else None
    model_for_phys = physics_model if (use_physics_residual and physics_model is not None) else None
    params = list(model.parameters()) + (list(model_for_phys.parameters()) if model_for_phys else [])

    loss_fn = TurbulenceLoss(lambda_phys=lambda_phys)
    optimizer = torch.optim.AdamW(params, lr=lr)

    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        total_loss = 0.0
        for i in range(0, len(perm), batch_size):
            idx_b = perm[i : i + batch_size]
            x_b = X[idx_b]
            y_b = Y[idx_b]
            optimizer.zero_grad()
            pred, aux = model(x_b, l_e="turbulence", training_step=epoch / 1000.0)
            losses = loss_fn(pred, y_b, colloc_for_phys, model_for_phys, config.nu)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += losses["total"].item()
        avg = total_loss / max(1, (len(perm) + batch_size - 1) // batch_size)
        if (epoch + 1) % 20 == 0:
            print(f"Turbulence Epoch {epoch+1}: L_total={avg:.6f}")

    return model
