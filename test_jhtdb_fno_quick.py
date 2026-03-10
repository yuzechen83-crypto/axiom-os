"""
Quick test for JHTDB FNO-RCLN experiment
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("JHTDB FNO-RCLN Quick Test")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Import FNO3d
from axiom_os.layers.fno3d import FNO3d

print("\n[1] Testing FNO3d layer...")

# Test parameters
batch_size = 1
resolution = 16  # Small for quick test
fno_width = 8
fno_modes = 4

# Create model
fno = FNO3d(
    in_channels=3,
    out_channels=6,
    width=fno_width,
    modes1=fno_modes,
    modes2=fno_modes,
    modes3=fno_modes
).to(device)

print(f"  FNO3d created: width={fno_width}, modes={fno_modes}")

# Test forward pass
u = torch.randn(batch_size, 3, resolution, resolution, resolution, device=device)
print(f"  Input shape: {u.shape}")

with torch.no_grad():
    output = fno(u)
print(f"  Output shape: {output.shape}")
print("  FNO3d forward: OK")

# Count parameters
n_params = sum(p.numel() for p in fno.parameters())
print(f"  Parameters: {n_params:,}")

print("\n[2] Testing Navier-Stokes Hard Core...")

class NavierStokesHardCore3D(nn.Module):
    """3D Navier-Stokes hard core."""
    def __init__(self, nu=0.000185):
        super().__init__()
        self.nu = nu
    
    def forward(self, u, dt=0.01):
        B, C, D, H, W = u.shape
        dx = 1.0 / D
        
        # Compute derivatives using finite differences
        du_dx = torch.zeros_like(u)
        du_dx[:, :, 1:-1, :, :] = (u[:, :, 2:, :, :] - u[:, :, :-2, :, :]) / (2*dx)
        
        du_dy = torch.zeros_like(u)
        du_dy[:, :, :, 1:-1, :] = (u[:, :, :, 2:, :] - u[:, :, :, :-2, :]) / (2*dx)
        
        du_dz = torch.zeros_like(u)
        du_dz[:, :, :, :, 1:-1] = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / (2*dx)
        
        # Convection: (u·∇)u
        convection = torch.zeros_like(u)
        for i in range(3):
            convection[:, i] = u[:, 0] * du_dx[:, i] + u[:, 1] * du_dy[:, i] + u[:, 2] * du_dz[:, i]
        
        # Laplacian
        laplacian = torch.zeros_like(u)
        laplacian[:, :, 1:-1, 1:-1, 1:-1] = (
            u[:, :, 2:, 1:-1, 1:-1] + u[:, :, :-2, 1:-1, 1:-1] +
            u[:, :, 1:-1, 2:, 1:-1] + u[:, :, 1:-1, :-2, 1:-1] +
            u[:, :, 1:-1, 1:-1, 2:] + u[:, :, 1:-1, 1:-1, :-2] -
            6 * u[:, :, 1:-1, 1:-1, 1:-1]
        ) / (dx**2)
        
        # NS residual
        du_dt = -convection + self.nu * laplacian
        return du_dt

ns_core = NavierStokesHardCore3D(nu=0.000185).to(device)
with torch.no_grad():
    du_dt = ns_core(u)
print(f"  NS Hard Core output shape: {du_dt.shape}")
print("  NS Hard Core forward: OK")

print("\n[3] Testing Full FNO-RCLN...")

class Full3DFNORCLN(nn.Module):
    """Full 3D FNO-RCLN for turbulence modeling."""
    def __init__(self, resolution=16, fno_width=8, fno_modes=4, nu=0.000185, lambda_res=0.5):
        super().__init__()
        self.resolution = resolution
        self.lambda_res = lambda_res
        
        # Hard core: Navier-Stokes
        self.ns_hard_core = NavierStokesHardCore3D(nu=nu)
        
        # Soft shell: 3D FNO
        self.fno_soft_shell = FNO3d(
            in_channels=3,
            out_channels=6,
            width=fno_width,
            modes1=fno_modes,
            modes2=fno_modes,
            modes3=fno_modes
        )
    
    def forward(self, u):
        # Hard core contribution
        with torch.no_grad():
            du_dt_hard = self.ns_hard_core(u)
        
        # Convert NS residual to stress-like quantity
        B, C, D, H, W = du_dt_hard.shape
        tau_hard = torch.zeros(B, 6, D, H, W, device=u.device)
        tau_hard[:, 0] = du_dt_hard[:, 0]
        tau_hard[:, 1] = du_dt_hard[:, 1]
        tau_hard[:, 2] = du_dt_hard[:, 2]
        tau_hard[:, 3] = 0.5 * (du_dt_hard[:, 0] + du_dt_hard[:, 1])
        tau_hard[:, 4] = 0.5 * (du_dt_hard[:, 0] + du_dt_hard[:, 2])
        tau_hard[:, 5] = 0.5 * (du_dt_hard[:, 1] + du_dt_hard[:, 2])
        
        # Soft shell contribution
        tau_soft = self.fno_soft_shell(u)
        
        # Coupling: residual
        tau = tau_hard + self.lambda_res * tau_soft
        
        return tau

model = Full3DFNORCLN(
    resolution=resolution,
    fno_width=fno_width,
    fno_modes=fno_modes,
    nu=0.000185,
    lambda_res=0.5
).to(device)

print(f"  Model created")
n_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {n_params:,}")

# Forward pass
with torch.no_grad():
    tau = model(u)
print(f"  Output shape: {tau.shape}")
print("  Full FNO-RCLN forward: OK")

print("\n[4] Testing backward pass...")

# Make u require grad
u_train = torch.randn(batch_size, 3, resolution, resolution, resolution, device=device, requires_grad=True)
tau_pred = model(u_train)
target = torch.randn_like(tau_pred)

loss = nn.MSELoss()(tau_pred, target)
loss.backward()

print(f"  Loss: {loss.item():.6f}")
print(f"  Input grad shape: {u_train.grad.shape}")
print("  Backward pass: OK")

if device.type == 'cuda':
    mem_used = torch.cuda.memory_allocated() / 1024**3
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU Memory: {mem_used:.2f}/{mem_total:.2f} GB")

print("\n" + "=" * 70)
print("All tests passed! FNO-RCLN is ready for JHTDB experiment.")
print("=" * 70)
print("\nTo run full experiment:")
print("  python -m axiom_os.experiments.jhtdb_1024_fno3d_gpu")
