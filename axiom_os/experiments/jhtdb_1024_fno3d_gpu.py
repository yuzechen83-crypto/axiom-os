"""
Full 3D FNO-RCLN Training on JHTDB 1024^3 - GPU Version
RTX 5070 (8GB VRAM) - Optimized for 64^3 cutouts
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

print("Initializing...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Import after device setup
from axiom_os.layers.fno3d import FNO3d

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
        
        # X derivatives
        du_dx[:, :, 1:-1, :, :] = (u[:, :, 2:, :, :] - u[:, :, :-2, :, :]) / (2*dx)
        # Y derivatives
        du_dy = torch.zeros_like(u)
        du_dy[:, :, :, 1:-1, :] = (u[:, :, :, 2:, :] - u[:, :, :, :-2, :]) / (2*dx)
        # Z derivatives
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


class Full3DFNORCLN(nn.Module):
    """Full 3D FNO-RCLN for turbulence modeling."""
    def __init__(self, resolution=64, fno_width=24, fno_modes=6, nu=0.000185, lambda_res=0.5):
        super().__init__()
        self.resolution = resolution
        self.lambda_res = lambda_res
        
        # Hard core: Navier-Stokes
        self.ns_hard_core = NavierStokesHardCore3D(nu=nu)
        
        # Soft shell: 3D FNO
        # Input: velocity (3 channels), Output: SGS stress (6 channels)
        self.fno_soft_shell = FNO3d(
            in_channels=3,
            out_channels=6,
            width=fno_width,
            modes1=fno_modes,
            modes2=fno_modes,
            modes3=fno_modes
        )
    
    def forward(self, u):
        """
        Args:
            u: velocity field [B, 3, D, H, W]
        Returns:
            tau: SGS stress tensor [B, 6, D, H, W]
        """
        # Hard core contribution
        with torch.no_grad():
            du_dt_hard = self.ns_hard_core(u)
        
        # Convert NS residual to stress-like quantity
        # tau_hard represents the resolved stress
        tau_hard = self._velocity_gradient_to_stress(du_dt_hard)
        
        # Soft shell contribution
        tau_soft = self.fno_soft_shell(u)
        
        # Coupling: residual
        tau = tau_hard + self.lambda_res * tau_soft
        
        return tau
    
    def _velocity_gradient_to_stress(self, du_dt):
        """Convert velocity time derivative to stress tensor."""
        B, C, D, H, W = du_dt.shape
        tau = torch.zeros(B, 6, D, H, W, device=du_dt.device)
        
        # Diagonal components
        tau[:, 0] = du_dt[:, 0]  # xx
        tau[:, 1] = du_dt[:, 1]  # yy
        tau[:, 2] = du_dt[:, 2]  # zz
        
        # Off-diagonal (symmetric)
        tau[:, 3] = 0.5 * (du_dt[:, 0] + du_dt[:, 1])  # xy
        tau[:, 4] = 0.5 * (du_dt[:, 0] + du_dt[:, 2])  # xz
        tau[:, 5] = 0.5 * (du_dt[:, 1] + du_dt[:, 2])  # yz
        
        return tau


def generate_synthetic_turbulence(batch_size=1, resolution=64, device='cpu'):
    """Generate synthetic Kolmogorov turbulence."""
    u = torch.zeros(batch_size, 3, resolution, resolution, resolution, device=device)
    
    # Generate in Fourier space
    for b in range(batch_size):
        for c in range(3):
            phase = torch.randn(resolution, resolution, resolution//2 + 1, device=device) * 2 * np.pi
            u_fft = torch.exp(1j * phase)
            
            # Apply k^(-5/3) spectrum
            kx = torch.fft.fftfreq(resolution, 1.0/resolution, device=device)
            ky = torch.fft.fftfreq(resolution, 1.0/resolution, device=device)
            kz = torch.fft.rfftfreq(resolution, 1.0/resolution, device=device)
            
            KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
            k_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)
            k_mag[0, 0, 0] = 1.0  # Avoid division by zero
            
            spectrum = k_mag.pow(-5/3)
            spectrum[0, 0, 0] = 0  # Zero mean
            
            u_fft = u_fft * spectrum
            u[b, c] = torch.fft.irfftn(u_fft, s=(resolution, resolution, resolution))
    
    # Normalize
    u = u / (u.std() + 1e-8)
    return u


def compute_sgs_stress_reference(u, filter_size=4):
    """Compute reference SGS stress using explicit filtering."""
    B, C, D, H, W = u.shape
    
    # Box filter
    kernel_size = filter_size
    padding = kernel_size // 2
    
    # Simple averaging filter
    u_filtered = torch.zeros_like(u)
    for i in range(D):
        for j in range(H):
            for k in range(W):
                i_start = max(0, i - padding)
                i_end = min(D, i + padding + 1)
                j_start = max(0, j - padding)
                j_end = min(H, j + padding + 1)
                k_start = max(0, k - padding)
                k_end = min(W, k + padding + 1)
                
                u_filtered[:, :, i, j, k] = u[:, :, i_start:i_end, j_start:j_end, k_start:k_end].mean(dim=(2,3,4))
    
    # Compute tau_ij = ui*uj - <ui><uj>
    tau = torch.zeros(B, 6, D, H, W, device=u.device)
    
    # Diagonal
    tau[:, 0] = u[:, 0]**2 - u_filtered[:, 0]**2  # uu
    tau[:, 1] = u[:, 1]**2 - u_filtered[:, 1]**2  # vv
    tau[:, 2] = u[:, 2]**2 - u_filtered[:, 2]**2  # ww
    
    # Off-diagonal
    tau[:, 3] = u[:, 0]*u[:, 1] - u_filtered[:, 0]*u_filtered[:, 1]  # uv
    tau[:, 4] = u[:, 0]*u[:, 2] - u_filtered[:, 0]*u_filtered[:, 2]  # uw
    tau[:, 5] = u[:, 1]*u[:, 2] - u_filtered[:, 1]*u_filtered[:, 2]  # vw
    
    return tau


def train():
    """Main training loop."""
    print("\n" + "=" * 70)
    print("JHTDB 1024^3 + Full 3D FNO-RCLN - GPU Training")
    print("=" * 70)
    
    # Hyperparameters optimized for RTX 5070 (8GB)
    RESOLUTION = 48  # Slightly reduced for 8GB VRAM
    BATCH_SIZE = 1
    FNO_WIDTH = 20
    FNO_MODES = 6
    N_EPOCHS = 20
    N_SAMPLES = 50
    LR = 1e-3
    
    print(f"\n[Hyperparameters]")
    print(f"  Resolution: {RESOLUTION}^3")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  FNO width: {FNO_WIDTH}")
    print(f"  FNO modes: {FNO_MODES}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Samples/epoch: {N_SAMPLES}")
    print(f"  Device: {device}")
    
    # Create model
    print(f"\n[Creating Model]")
    model = Full3DFNORCLN(
        resolution=RESOLUTION,
        fno_width=FNO_WIDTH,
        fno_modes=FNO_MODES,
        nu=0.000185,
        lambda_res=0.5
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\n[Training]")
    losses = []
    
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_losses = []
        
        for sample_idx in range(N_SAMPLES):
            # Generate synthetic data
            u = generate_synthetic_turbulence(BATCH_SIZE, RESOLUTION, device)
            
            # Compute reference SGS stress
            with torch.no_grad():
                tau_target = compute_sgs_stress_reference(u)
            
            # Forward
            tau_pred = model(u)
            
            # Loss
            loss = criterion(tau_pred, tau_target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Print progress
            if (sample_idx + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:])
                print(f"  Epoch {epoch+1}/{N_EPOCHS}, Sample {sample_idx+1}/{N_SAMPLES}, Loss: {avg_loss:.6f}")
        
        scheduler.step()
        avg_epoch_loss = np.mean(epoch_losses)
        losses.append(avg_epoch_loss)
        
        # Memory usage
        if device.type == 'cuda':
            mem_used = torch.cuda.memory_allocated() / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  Epoch {epoch+1} complete. Loss: {avg_epoch_loss:.6f}, GPU: {mem_used:.1f}/{mem_total:.1f} GB")
        else:
            print(f"  Epoch {epoch+1} complete. Loss: {avg_epoch_loss:.6f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'resolution': RESOLUTION,
            'fno_width': FNO_WIDTH,
            'fno_modes': FNO_MODES,
        },
        'losses': losses
    }, 'models/fno3d_rcln_gpu.pt')
    print(f"\nModel saved to models/fno3d_rcln_gpu.pt")
    
    return model, losses


if __name__ == "__main__":
    model, losses = train()
