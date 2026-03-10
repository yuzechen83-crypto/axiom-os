"""
Fluid Core Module - Axiom-OS B-01 Benchmark
Hard Core: Smagorinsky Model (Physics)
Soft Shell: FNO Residual (AI)
Hybrid: Axiom-OS Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SmagorinskyCore(nn.Module):
    """
    Smagorinsky Subgrid-Scale Model - The Physics Hard Core.
    
    τ_hard = -(Cs * Δ)² |S| S_ij
    
    Where:
        S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)  [Strain rate tensor]
        |S| = sqrt(2 * S_ij * S_ij)            [Strain magnitude]
        Cs = 0.1                                [Smagorinsky constant]
        Δ = grid spacing
    
    This is the industry-standard SGS model used in most CFD codes.
    Assumption: SGS stress is aligned with resolved strain (eddy viscosity hypothesis).
    """
    
    def __init__(self, cs: float = 0.1, delta: Optional[float] = None):
        super().__init__()
        self.cs = cs
        self.delta = delta  # If None, computed from grid size
        
    def compute_strain_rate(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute strain rate tensor S_ij from velocity field.
        
        Args:
            u: Velocity field (batch, 2, H, W) - [u, v] components
        
        Returns:
            S: Strain rate tensor (batch, 2, 2, H, W)
        """
        batch_size, _, H, W = u.shape
        device = u.device
        
        # Grid spacing (assume uniform)
        if self.delta is None:
            delta = 1.0 / H
        else:
            delta = self.delta
        
        # Compute gradients using finite differences
        # ∂u/∂x, ∂u/∂y
        u_x = u[:, 0:1, :, :]  # (batch, 1, H, W)
        u_y = u[:, 1:2, :, :]  # (batch, 1, H, W)
        
        # Central differences for interior points
        # ∂u/∂x
        du_dx = torch.zeros_like(u_x)
        du_dx[:, :, :, 1:-1] = (u_x[:, :, :, 2:] - u_x[:, :, :, :-2]) / (2 * delta)
        du_dx[:, :, :, 0] = (u_x[:, :, :, 1] - u_x[:, :, :, 0]) / delta  # Forward
        du_dx[:, :, :, -1] = (u_x[:, :, :, -1] - u_x[:, :, :, -2]) / delta  # Backward
        
        # ∂u/∂y
        du_dy = torch.zeros_like(u_x)
        du_dy[:, :, 1:-1, :] = (u_x[:, :, 2:, :] - u_x[:, :, :-2, :]) / (2 * delta)
        du_dy[:, :, 0, :] = (u_x[:, :, 1, :] - u_x[:, :, 0, :]) / delta
        du_dy[:, :, -1, :] = (u_x[:, :, -1, :] - u_x[:, :, -2, :]) / delta
        
        # ∂v/∂x
        dv_dx = torch.zeros_like(u_y)
        dv_dx[:, :, :, 1:-1] = (u_y[:, :, :, 2:] - u_y[:, :, :, :-2]) / (2 * delta)
        dv_dx[:, :, :, 0] = (u_y[:, :, :, 1] - u_y[:, :, :, 0]) / delta
        dv_dx[:, :, :, -1] = (u_y[:, :, :, -1] - u_y[:, :, :, -2]) / delta
        
        # ∂v/∂y
        dv_dy = torch.zeros_like(u_y)
        dv_dy[:, :, 1:-1, :] = (u_y[:, :, 2:, :] - u_y[:, :, :-2, :]) / (2 * delta)
        dv_dy[:, :, 0, :] = (u_y[:, :, 1, :] - u_y[:, :, 0, :]) / delta
        dv_dy[:, :, -1, :] = (u_y[:, :, -1, :] - u_y[:, :, -2, :]) / delta
        
        # Strain rate tensor S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        # S_11 = du/dx
        # S_12 = S_21 = 0.5 * (du/dy + dv/dx)
        # S_22 = dv/dy
        
        S = torch.zeros(batch_size, 2, 2, H, W, device=device)
        S[:, 0, 0, :, :] = du_dx[:, 0, :, :]
        S[:, 0, 1, :, :] = 0.5 * (du_dy[:, 0, :, :] + dv_dx[:, 0, :, :])
        S[:, 1, 0, :, :] = S[:, 0, 1, :, :]  # Symmetric
        S[:, 1, 1, :, :] = dv_dy[:, 0, :, :]
        
        return S
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute SGS stress using Smagorinsky model.
        
        Args:
            u: Velocity field (batch, 2, H, W)
        
        Returns:
            tau: SGS stress tensor (batch, 3, H, W) - [τ_xx, τ_xy, τ_yy]
                 Note: τ_xy = τ_yx (symmetric)
        """
        batch_size, _, H, W = u.shape
        
        # Grid spacing
        if self.delta is None:
            delta = 1.0 / H
        else:
            delta = self.delta
        
        # Compute strain rate
        S = self.compute_strain_rate(u)  # (batch, 2, 2, H, W)
        
        # Strain magnitude |S| = sqrt(2 * S_ij * S_ij)
        S_squared = torch.sum(S * S, dim=(1, 2))  # (batch, H, W)
        S_mag = torch.sqrt(2.0 * S_squared + 1e-10)  # Add small epsilon for stability
        
        # Smagorinsky eddy viscosity
        # ν_sgs = (Cs * Δ)² |S|
        nu_sgs = (self.cs * delta) ** 2 * S_mag  # (batch, H, W)
        
        # SGS stress: τ_ij = -2 * ν_sgs * S_ij
        # For output, we return [τ_xx, τ_xy, τ_yy]
        tau_xx = -2.0 * nu_sgs * S[:, 0, 0, :, :]
        tau_xy = -2.0 * nu_sgs * S[:, 0, 1, :, :]
        tau_yy = -2.0 * nu_sgs * S[:, 1, 1, :, :]
        
        tau = torch.stack([tau_xx, tau_xy, tau_yy], dim=1)  # (batch, 3, H, W)
        
        return tau


class SpectralConv2d(nn.Module):
    """Fourier spectral convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Complex weights for Fourier modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space."""
        # input: (batch, in_channel, x, y, 2) - complex represented as 2 real channels
        # weights: (in_channel, out_channel, x, y, 2) - complex weights
        # output: (batch, out_channel, x, y, 2)
        
        # Separate real and imaginary parts
        input_real = input[..., 0]  # (batch, in_channel, x, y)
        input_imag = input[..., 1]  # (batch, in_channel, x, y)
        weights_real = weights[..., 0]  # (in_channel, out_channel, x, y)
        weights_imag = weights[..., 1]  # (in_channel, out_channel, x, y)
        
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # Output real
        out_real = torch.einsum("bixy,ioxy->boxy", input_real, weights_real) - \
                   torch.einsum("bixy,ioxy->boxy", input_imag, weights_imag)
        # Output imag
        out_imag = torch.einsum("bixy,ioxy->boxy", input_real, weights_imag) + \
                   torch.einsum("bixy,ioxy->boxy", input_imag, weights_real)
        
        # Stack back
        return torch.stack([out_real, out_imag], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        # Initialize output in Fourier space
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 2, 
                            device=x.device, dtype=torch.float32)
        
        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # Inverse FFT
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)))
        
        return x


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2D fields.
    Learns operator: u → τ (velocity to SGS stress)
    """
    
    def __init__(self, in_channels: int = 2, out_channels: int = 3, 
                 width: int = 32, modes: Tuple[int, int] = (12, 12), 
                 n_layers: int = 4):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.n_layers = n_layers
        
        # Lift channel dimension
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        
        # FNO layers
        self.spectral_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(width, width, modes[0], modes[1]))
            self.w_layers.append(nn.Conv2d(width, width, kernel_size=1))
        
        # Project to output
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input field (batch, in_channels, H, W)
        
        Returns:
            out: Output field (batch, out_channels, H, W)
        """
        # Lift
        x = self.lift(x)
        
        # FNO layers with residual connections
        for spectral, w in zip(self.spectral_layers, self.w_layers):
            x1 = spectral(x)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)
        
        # Project
        x = self.project(x)
        
        return x


class HybridFluidRCLN(nn.Module):
    """
    Axiom-OS Hybrid Architecture for Fluid SGS Modeling.
    
    Architecture:
        τ = τ_hard + τ_soft
          = Smagorinsky(u) + FNO_residual(u)
    
    Hard Core: Smagorinsky model (physics-based baseline)
    Soft Shell: FNO predicting the residual (AI correction)
    
    This is the key innovation of Axiom-OS:
    - Hard core provides physical correctness (energy dissipation, realizability)
    - Soft shell learns the gap (anisotropy, non-local effects, backscatter)
    """
    
    def __init__(self, cs: float = 0.1, fno_width: int = 32, 
                 fno_modes: Tuple[int, int] = (12, 12), fno_layers: int = 4):
        super().__init__()
        
        # Hard Core: Physics-based Smagorinsky
        self.hard = SmagorinskyCore(cs=cs)
        
        # Soft Shell: FNO for residual learning
        self.soft = FNO2d(
            in_channels=2,  # [u, v] velocity
            out_channels=3,  # [τ_xx, τ_xy, τ_yy] stress
            width=fno_width,
            modes=fno_modes,
            n_layers=fno_layers
        )
        
        # Learnable coupling parameter (optional)
        self.lambda_residual = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute SGS stress using hybrid approach.
        
        Args:
            u: Velocity field (batch, 2, H, W)
        
        Returns:
            tau: SGS stress tensor (batch, 3, H, W)
        """
        # Hard core prediction (physics)
        tau_hard = self.hard(u)
        
        # Soft shell prediction (AI residual)
        tau_soft = self.soft(u)
        
        # Hybrid combination
        tau = tau_hard + torch.sigmoid(self.lambda_residual) * tau_soft
        
        return tau
    
    def forward_decomposed(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with decomposition for analysis.
        
        Returns:
            tau: Total stress
            tau_hard: Hard core component
            tau_soft: Soft shell component
        """
        tau_hard = self.hard(u)
        tau_soft = self.soft(u)
        tau = tau_hard + torch.sigmoid(self.lambda_residual) * tau_soft
        
        return tau, tau_hard, tau_soft


def compute_correlation_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute R² correlation coefficient.
    
    Args:
        pred: Predicted values (batch, channels, H, W)
        target: Ground truth values (batch, channels, H, W)
    
    Returns:
        r2: R² score
    """
    pred_flat = pred.detach().cpu().numpy().flatten()
    target_flat = target.detach().cpu().numpy().flatten()
    
    # Correlation coefficient
    numerator = np.sum((pred_flat - np.mean(pred_flat)) * (target_flat - np.mean(target_flat)))
    denominator = np.sqrt(np.sum((pred_flat - np.mean(pred_flat))**2) * 
                         np.sum((target_flat - np.mean(target_flat))**2))
    
    if denominator < 1e-10:
        return 0.0
    
    r = numerator / denominator
    r2 = r ** 2
    
    return float(r2)


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean squared error."""
    return F.mse_loss(pred, target).item()


if __name__ == "__main__":
    # Quick test
    print("Testing Fluid Core Module...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, H, W = 2, 64, 64
    
    # Create test velocity field
    u = torch.randn(batch_size, 2, H, W).to(device)
    
    # Test Smagorinsky
    print("\n1. Testing SmagorinskyCore...")
    smag = SmagorinskyCore(cs=0.1).to(device)
    tau_smag = smag(u)
    print(f"   Input: {u.shape} -> Output: {tau_smag.shape}")
    print(f"   τ range: [{tau_smag.min():.4f}, {tau_smag.max():.4f}]")
    
    # Test FNO
    print("\n2. Testing FNO2d...")
    fno = FNO2d(in_channels=2, out_channels=3, width=32, modes=(12, 12)).to(device)
    tau_fno = fno(u)
    print(f"   Input: {u.shape} -> Output: {tau_fno.shape}")
    print(f"   τ range: [{tau_fno.min():.4f}, {tau_fno.max():.4f}]")
    
    # Test Hybrid
    print("\n3. Testing HybridFluidRCLN...")
    hybrid = HybridFluidRCLN(cs=0.1, fno_width=32).to(device)
    tau_hybrid, tau_hard, tau_soft = hybrid.forward_decomposed(u)
    print(f"   Input: {u.shape} -> Output: {tau_hybrid.shape}")
    print(f"   Hard component: [{tau_hard.min():.4f}, {tau_hard.max():.4f}]")
    print(f"   Soft component: [{tau_soft.min():.4f}, {tau_soft.max():.4f}]")
    print(f"   Total: [{tau_hybrid.min():.4f}, {tau_hybrid.max():.4f}]")
    
    print("\n✓ Fluid Core Module Ready for B-01 Benchmark!")
