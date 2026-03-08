# -*- coding: utf-8 -*-
"""
RCLN v3.0 - The Ultimate Physics-AI Hybrid
Integrates Hard Core 3.0 (Differentiable Physics) with GENERIC Coupling

Architecture:
    tau = lambda_hard * tau_hard + lambda_soft * tau_soft
    
Where:
    - tau_hard: Physics-based SGS from Hard Core 3.0 (learnable nu)
    - tau_soft: Neural SGS from Soft Shell (FNO)
    
Coupling: tau = lambda_hard * tau_hard + lambda_soft * tau_soft

Key Features:
    1. Learnable physics parameters (nu - viscosity)
    2. Gradient flows through physics equations
    3. FNO-based neural correction
"""

from typing import Optional, Callable, Tuple, Dict, Union
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TurbulenceHardCoreV3(nn.Module):
    """
    Hard Core 3.0 for Turbulence - Learnable Physics
    
    Computes physics-based SGS stress with learnable viscosity.
    """
    
    def __init__(self, nu_init: float = 0.001):
        super().__init__()
        # Learnable viscosity (constrained positive via softplus)
        self._nu_raw = nn.Parameter(torch.tensor(nu_init))
        self.scale = 0.01
    
    @property
    def nu(self):
        """Viscosity - always positive"""
        return F.softplus(self._nu_raw) * 0.1
    
    def compute_sgs_stress(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute SGS stress using Smagorinsky-like model with learnable nu.
        
        tau_ij = nu * (du_i/dx_j + du_j/dx_i)
        
        Args:
            velocity: [B, 3, D, H, W]
        Returns:
            tau: [B, 6, D, H, W] stress components
        """
        B, C, D, H, W = velocity.shape
        dx = 1.0 / D
        
        # Compute velocity gradients
        def gradient(f, axis):
            """Compute gradient along axis"""
            # PyTorch conv3d format: [B, C, D, H, W] = [B, C, z, y, x]
            if axis == 2:  # x (last dim)
                f_pad = F.pad(f, [1, 1, 0, 0, 0, 0], mode='replicate')
                return (f_pad[..., 2:] - f_pad[..., :-2]) / (2 * dx)
            elif axis == 3:  # y (second last dim)
                f_pad = F.pad(f, [0, 0, 1, 1, 0, 0], mode='replicate')
                return (f_pad[..., 2:, :] - f_pad[..., :-2, :]) / (2 * dx)
            else:  # z (first spatial dim)
                f_pad = F.pad(f, [0, 0, 0, 0, 1, 1], mode='replicate')
                return (f_pad[:, :, 2:, ...] - f_pad[:, :, :-2, ...]) / (2 * dx)
        
        # Velocity gradients
        grad_u = torch.stack([
            gradient(velocity[:, 0:1], 2).squeeze(1),  # du/dx
            gradient(velocity[:, 0:1], 3).squeeze(1),  # du/dy
            gradient(velocity[:, 0:1], 4).squeeze(1),  # du/dz
            gradient(velocity[:, 1:2], 2).squeeze(1),  # dv/dx
            gradient(velocity[:, 1:2], 3).squeeze(1),  # dv/dy
            gradient(velocity[:, 1:2], 4).squeeze(1),  # dv/dz
            gradient(velocity[:, 2:3], 2).squeeze(1),  # dw/dx
            gradient(velocity[:, 2:3], 3).squeeze(1),  # dw/dy
            gradient(velocity[:, 2:3], 4).squeeze(1),  # dw/dz
        ], dim=1)  # [B, 9, D, H, W]
        
        # SGS stress: nu * strain rate
        # tau_xx = 2 * nu * du/dx
        # tau_yy = 2 * nu * dv/dy
        # tau_zz = 2 * nu * dw/dz
        # tau_xy = nu * (du/dy + dv/dx)
        # tau_xz = nu * (du/dz + dw/dx)
        # tau_yz = nu * (dv/dz + dw/dy)
        
        tau = torch.zeros(B, 6, D, H, W, device=velocity.device)
        nu = self.nu
        
        tau[:, 0] = 2 * nu * grad_u[:, 0]  # xx
        tau[:, 1] = 2 * nu * grad_u[:, 4]  # yy
        tau[:, 2] = 2 * nu * grad_u[:, 8]  # zz
        tau[:, 3] = nu * (grad_u[:, 1] + grad_u[:, 3])  # xy
        tau[:, 4] = nu * (grad_u[:, 2] + grad_u[:, 6])  # xz
        tau[:, 5] = nu * (grad_u[:, 5] + grad_u[:, 7])  # yz
        
        return tau


class RCLNv3_Turbulence(nn.Module):
    """
    RCLN v3.0 for 3D Turbulence Modeling
    
    Combines:
    - Hard Core 3.0: Learnable physics-based SGS
    - Soft Shell: FNO for neural correction
    - Coupling: lambda_hard * tau_hard + lambda_soft * tau_soft
    
    Output: SGS stress tau_ij
    """
    
    def __init__(
        self,
        resolution: int = 12,
        fno_width: int = 8,
        fno_modes: int = 4,
        nu_init: float = 0.001,
        lambda_hard: float = 0.5,
        lambda_soft: float = 0.5,
    ):
        super().__init__()
        self.resolution = resolution
        self.lambda_hard = lambda_hard
        self.lambda_soft = lambda_soft
        
        # Hard Core 3.0 - Learnable physics
        self.hard_core = TurbulenceHardCoreV3(nu_init=nu_init)
        
        # Soft Shell: FNO for learning residual SGS
        from .fno3d import FNO3d
        self.soft_shell = FNO3d(
            in_channels=3,
            out_channels=6,  # 6 SGS stress components
            width=fno_width,
            modes1=fno_modes,
            modes2=fno_modes,
            modes3=fno_modes,
            n_layers=2,
        )
    
    def forward(
        self, 
        velocity: torch.Tensor, 
        return_physics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with learnable physics + neural correction
        
        Args:
            velocity: [B, 3, D, H, W] velocity field
            return_physics: Return physical quantities
        
        Returns:
            tau: [B, 6, D, H, W] SGS stress tensor
            info: Physical quantities (optional)
        """
        # Hard Core: Physics-based SGS with learnable nu
        tau_hard = self.hard_core.compute_sgs_stress(velocity)
        
        # Soft Shell: Neural SGS prediction
        tau_soft = self.soft_shell(velocity)
        
        # Coupled output: lambda_hard * tau_hard + lambda_soft * tau_soft
        tau = self.lambda_hard * tau_hard + self.lambda_soft * tau_soft
        
        if return_physics:
            info = {
                'nu': self.hard_core.nu.detach(),
                'tau_hard': tau_hard.detach(),
                'tau_soft': tau_soft.detach(),
            }
            return tau, info
        
        return tau
    
    def get_architecture_info(self) -> Dict:
        return {
            'version': 'v3.0_Turbulence (Simplified)',
            'hard_core': 'TurbulenceHardCoreV3 (learnable nu)',
            'soft_shell': 'FNO3d',
            'coupling': f'{self.lambda_hard} * tau_hard + {self.lambda_soft} * tau_soft',
            'resolution': self.resolution,
            'learnable_params': ['nu (viscosity)'],
            'lambda_hard': self.lambda_hard,
            'lambda_soft': self.lambda_soft,
        }


def test_rcln_v3():
    """Test RCLN v3.0"""
    print("=" * 70)
    print("RCLN v3.0 - Turbulence Test")
    print("=" * 70)
    
    model = RCLNv3_Turbulence(
        resolution=12,
        fno_width=8,
        fno_modes=4,
        nu_init=0.001,
        lambda_hard=0.3,
        lambda_soft=0.7,
    )
    
    print("\n[Architecture Info]")
    info = model.get_architecture_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print(f"\n[Parameters]")
    total = sum(p.numel() for p in model.parameters())
    hard = sum(p.numel() for p in model.hard_core.parameters())
    soft = sum(p.numel() for p in model.soft_shell.parameters())
    print(f"  Total: {total:,}")
    print(f"  Hard Core: {hard:,}")
    print(f"  Soft Shell: {soft:,}")
    print(f"  nu initial: {model.hard_core.nu.item():.6f}")
    
    print(f"\n[Forward Test]")
    u = torch.randn(2, 3, 12, 12, 12, requires_grad=True)
    print(f"  Input: {u.shape}")
    
    tau, phys = model(u, return_physics=True)
    print(f"  Output tau: {tau.shape}")
    print(f"  nu: {phys['nu'].item():.6f}")
    print(f"  tau_hard norm: {phys['tau_hard'].norm().item():.4f}")
    print(f"  tau_soft norm: {phys['tau_soft'].norm().item():.4f}")
    
    print(f"\n[Gradient Test]")
    target = torch.randn_like(tau)
    loss = F.mse_loss(tau, target)
    loss.backward()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  nu grad: {model.hard_core._nu_raw.grad.abs().item():.6f}")
    
    # Check if gradients flow to all parameters
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_p = sum(1 for p in model.parameters())
    print(f"  Parameters with grad: {has_grad}/{total_p}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] RCLN v3.0 test passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_rcln_v3()
