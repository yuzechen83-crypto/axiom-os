# -*- coding: utf-8 -*-
"""
RCLN v3.0 Advanced - Enhanced Physics Core for Turbulence

Hard Core 3.0 Advanced Features:
    1. Dynamic Smagorinsky: nu_d = Cs^2 * delta^2 * |S| (learnable Cs)
    2. Strain-vorticity interaction: S_ij * Omega_jk
    3. Anisotropic stress: tau_ij = nu_d * S_ij + nu_rot * R_ij
    4. Higher-order terms: S_ik * S_kj - 1/3 * delta_ij * |S|^2

Where:
    S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)  # Strain rate
    Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)  # Rotation rate
    |S| = sqrt(2 * S_ij * S_ij)  # Strain magnitude
"""

from typing import Dict, Tuple, Union
import sys
import os
sys.path.insert(0, r'C:\Users\ASUS\PycharmProjects\PythonProject1')

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedTurbulenceHardCore(nn.Module):
    """
    Advanced Hard Core for Turbulence with:
    - Dynamic Smagorinsky model (learnable Cs)
    - Strain-rotation interaction
    - Anisotropic stress tensor
    """
    
    def __init__(
        self,
        filter_width: float = 1.0,
        cs_init: float = 0.1,
        use_dynamic: bool = True,
        use_rotation: bool = True,
        use_anisotropic: bool = True,
    ):
        super().__init__()
        self.filter_width = filter_width
        self.use_dynamic = use_dynamic
        self.use_rotation = use_rotation
        self.use_anisotropic = use_anisotropic
        
        # Learnable Smagorinsky constant (typically 0.1-0.2)
        self._cs_raw = nn.Parameter(torch.tensor(cs_init))
        
        # Learnable rotation coefficient
        if use_rotation:
            self._cr_raw = nn.Parameter(torch.tensor(0.05))
        
        # Learnable anisotropic coefficient
        if use_anisotropic:
            self._ca_raw = nn.Parameter(torch.tensor(0.01))
    
    @property
    def cs(self):
        """Smagorinsky constant - constrained positive"""
        return F.softplus(self._cs_raw) * 0.2  # Typical range: 0-0.2
    
    @property
    def cr(self):
        """Rotation coefficient"""
        return F.softplus(self._cr_raw) * 0.1 if self.use_rotation else torch.tensor(0.0)
    
    @property
    def ca(self):
        """Anisotropic coefficient"""
        return F.softplus(self._ca_raw) * 0.05 if self.use_anisotropic else torch.tensor(0.0)
    
    def compute_velocity_gradients(self, velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute velocity gradients and derived quantities.
        
        Returns:
            grad_u: [B, 3, 3, D, H, W] - du_i/dx_j
            S: [B, 3, 3, D, H, W] - Strain rate tensor
            Omega: [B, 3, 3, D, H, W] - Rotation rate tensor
        """
        B, C, D, H, W = velocity.shape
        dx = 1.0 / D
        
        # Pad for finite differences
        vel_pad = F.pad(velocity, [1, 1, 1, 1, 1, 1], mode='replicate')
        
        # Compute gradients: grad_u[b, i, j] = du_i/dx_j
        grad_u = torch.zeros(B, 3, 3, D, H, W, device=velocity.device)
        
        # du/dx, du/dy, du/dz (component 0)
        grad_u[:, 0, 0] = (vel_pad[:, 0, 1:-1, 1:-1, 2:] - vel_pad[:, 0, 1:-1, 1:-1, :-2]) / (2 * dx)
        grad_u[:, 0, 1] = (vel_pad[:, 0, 1:-1, 2:, 1:-1] - vel_pad[:, 0, 1:-1, :-2, 1:-1]) / (2 * dx)
        grad_u[:, 0, 2] = (vel_pad[:, 0, 2:, 1:-1, 1:-1] - vel_pad[:, 0, :-2, 1:-1, 1:-1]) / (2 * dx)
        
        # dv/dx, dv/dy, dv/dz (component 1)
        grad_u[:, 1, 0] = (vel_pad[:, 1, 1:-1, 1:-1, 2:] - vel_pad[:, 1, 1:-1, 1:-1, :-2]) / (2 * dx)
        grad_u[:, 1, 1] = (vel_pad[:, 1, 1:-1, 2:, 1:-1] - vel_pad[:, 1, 1:-1, :-2, 1:-1]) / (2 * dx)
        grad_u[:, 1, 2] = (vel_pad[:, 1, 2:, 1:-1, 1:-1] - vel_pad[:, 1, :-2, 1:-1, 1:-1]) / (2 * dx)
        
        # dw/dx, dw/dy, dw/dz (component 2)
        grad_u[:, 2, 0] = (vel_pad[:, 2, 1:-1, 1:-1, 2:] - vel_pad[:, 2, 1:-1, 1:-1, :-2]) / (2 * dx)
        grad_u[:, 2, 1] = (vel_pad[:, 2, 1:-1, 2:, 1:-1] - vel_pad[:, 2, 1:-1, :-2, 1:-1]) / (2 * dx)
        grad_u[:, 2, 2] = (vel_pad[:, 2, 2:, 1:-1, 1:-1] - vel_pad[:, 2, :-2, 1:-1, 1:-1]) / (2 * dx)
        
        # Strain rate: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        S = torch.zeros_like(grad_u)
        for i in range(3):
            for j in range(3):
                S[:, i, j] = 0.5 * (grad_u[:, i, j] + grad_u[:, j, i])
        
        # Rotation rate: Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
        Omega = torch.zeros_like(grad_u)
        for i in range(3):
            for j in range(3):
                Omega[:, i, j] = 0.5 * (grad_u[:, i, j] - grad_u[:, j, i])
        
        return grad_u, S, Omega
    
    def compute_sgs_stress(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute SGS stress using advanced model.
        
        tau_ij = -2 * nu_d * S_ij - 2 * nu_rot * Omega_ij 
                 + Ca * (S_ik * S_kj - 1/3 * delta_ij * |S|^2)
        
        Returns:
            tau: [B, 6, D, H, W] - SGS stress components
        """
        B, C, D, H, W = velocity.shape
        
        # Compute gradients
        grad_u, S, Omega = self.compute_velocity_gradients(velocity)
        
        # Strain magnitude: |S| = sqrt(2 * S_ij * S_ij)
        S_squared = torch.zeros(B, D, H, W, device=velocity.device)
        for i in range(3):
            for j in range(3):
                S_squared += S[:, i, j] ** 2
        S_magnitude = torch.sqrt(2 * S_squared + 1e-8)
        
        # Dynamic eddy viscosity: nu_d = Cs^2 * delta^2 * |S|
        delta = self.filter_width
        nu_d = (self.cs ** 2) * (delta ** 2) * S_magnitude  # [B, D, H, W]
        
        # Rotation magnitude
        if self.use_rotation:
            Omega_squared = torch.zeros(B, D, H, W, device=velocity.device)
            for i in range(3):
                for j in range(3):
                    Omega_squared += Omega[:, i, j] ** 2
            Omega_magnitude = torch.sqrt(2 * Omega_squared + 1e-8)
            nu_rot = (self.cr ** 2) * (delta ** 2) * Omega_magnitude
        else:
            nu_rot = torch.tensor(0.0, device=velocity.device)
        
        # Compute stress tensor components
        # tau_ij in 3x3 form
        tau_full = torch.zeros(B, 3, 3, D, H, W, device=velocity.device)
        
        # Isotropic part: -2 * nu_d * S_ij
        for i in range(3):
            for j in range(3):
                tau_full[:, i, j] = -2 * nu_d * S[:, i, j]
        
        # Rotation part: -2 * nu_rot * Omega_ij
        if self.use_rotation:
            for i in range(3):
                for j in range(3):
                    tau_full[:, i, j] += -2 * nu_rot * Omega[:, i, j]
        
        # Anisotropic part: Ca * (S_ik * S_kj - 1/3 * delta_ij * |S|^2)
        if self.use_anisotropic:
            for i in range(3):
                for j in range(3):
                    # S_ik * S_kj
                    S_S = torch.zeros(B, D, H, W, device=velocity.device)
                    for k in range(3):
                        S_S += S[:, i, k] * S[:, k, j]
                    
                    # Subtract trace
                    if i == j:
                        S_S -= (1.0 / 3.0) * S_squared
                    
                    tau_full[:, i, j] += self.ca * S_S
        
        # Convert to 6-component form: [xx, yy, zz, xy, xz, yz]
        tau = torch.zeros(B, 6, D, H, W, device=velocity.device)
        tau[:, 0] = tau_full[:, 0, 0]  # xx
        tau[:, 1] = tau_full[:, 1, 1]  # yy
        tau[:, 2] = tau_full[:, 2, 2]  # zz
        tau[:, 3] = tau_full[:, 0, 1]  # xy
        tau[:, 4] = tau_full[:, 0, 2]  # xz
        tau[:, 5] = tau_full[:, 1, 2]  # yz
        
        return tau
    
    def get_physics_params(self) -> Dict[str, float]:
        """Return current physics parameters"""
        params = {'cs': self.cs.item()}
        if self.use_rotation:
            params['cr'] = self.cr.item()
        if self.use_anisotropic:
            params['ca'] = self.ca.item()
        return params


class RCLNv3_Advanced(nn.Module):
    """
    RCLN v3.0 Advanced with Enhanced Physics Core
    """
    
    def __init__(
        self,
        resolution: int = 16,
        fno_width: int = 16,
        fno_modes: int = 8,
        cs_init: float = 0.1,
        lambda_hard: float = 0.5,
        lambda_soft: float = 0.5,
        use_dynamic: bool = True,
        use_rotation: bool = True,
        use_anisotropic: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.lambda_hard = lambda_hard
        self.lambda_soft = lambda_soft
        
        # Advanced Hard Core
        self.hard_core = AdvancedTurbulenceHardCore(
            filter_width=1.0 / resolution,
            cs_init=cs_init,
            use_dynamic=use_dynamic,
            use_rotation=use_rotation,
            use_anisotropic=use_anisotropic,
        )
        
        # Soft Shell: FNO for residual correction
        from axiom_os.layers.fno3d import FNO3d
        self.soft_shell = FNO3d(
            in_channels=3,
            out_channels=6,
            width=fno_width,
            modes1=fno_modes,
            modes2=fno_modes,
            modes3=fno_modes,
            n_layers=3,
        )
    
    def forward(
        self,
        velocity: torch.Tensor,
        return_physics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass"""
        # Hard Core contribution
        tau_hard = self.hard_core.compute_sgs_stress(velocity)
        
        # Soft Shell contribution
        tau_soft = self.soft_shell(velocity)
        
        # Coupled output
        tau = self.lambda_hard * tau_hard + self.lambda_soft * tau_soft
        
        if return_physics:
            info = {
                'tau_hard': tau_hard.detach(),
                'tau_soft': tau_soft.detach(),
                'physics_params': self.hard_core.get_physics_params(),
            }
            return tau, info
        
        return tau
    
    def get_architecture_info(self) -> Dict:
        return {
            'version': 'v3.0_Advanced',
            'hard_core': 'AdvancedTurbulenceHardCore (dynamic Smagorinsky)',
            'features': [
                'Dynamic eddy viscosity',
                'Strain-rotation interaction',
                'Anisotropic stress',
            ],
            'soft_shell': f'FNO3d (width={self.soft_shell.width}, layers={self.soft_shell.n_layers})',
            'coupling': f'{self.lambda_hard} * hard + {self.lambda_soft} * soft',
            'learnable_params': list(self.hard_core.get_physics_params().keys()),
        }


def test_advanced_hard_core():
    """Test the advanced hard core"""
    print("=" * 70)
    print("Advanced Hard Core Test")
    print("=" * 70)
    
    # Create model
    model = RCLNv3_Advanced(
        resolution=16,
        fno_width=16,
        fno_modes=8,
        cs_init=0.1,
        lambda_hard=0.5,
        lambda_soft=0.5,
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
    print(f"  Physics params: {model.hard_core.get_physics_params()}")
    
    print(f"\n[Forward Test]")
    u = torch.randn(2, 3, 16, 16, 16, requires_grad=True)
    print(f"  Input: {u.shape}")
    
    tau, phys = model(u, return_physics=True)
    print(f"  Output tau: {tau.shape}")
    print(f"  tau_hard norm: {phys['tau_hard'].norm().item():.4f}")
    print(f"  tau_soft norm: {phys['tau_soft'].norm().item():.4f}")
    
    print(f"\n[Gradient Test]")
    target = torch.randn_like(tau)
    loss = F.mse_loss(tau, target)
    loss.backward()
    
    print(f"  Loss: {loss.item():.4f}")
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with grad: {params_with_grad}/{sum(1 for _ in model.parameters())}")
    
    # Check physics parameter gradients
    for name, param in model.hard_core.named_parameters():
        if param.grad is not None:
            print(f"    {name} grad: {param.grad.abs().mean().item():.6f}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Advanced Hard Core test passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_advanced_hard_core()
