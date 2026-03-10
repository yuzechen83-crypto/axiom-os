"""
Axiom-OS + PDEBench Real Data Experiment
=========================================
Using PDEBench 2D/3D Navier-Stokes data for turbulence modeling.
This uses REAL simulation data (not synthetic).

Dataset: PDEBench 2D_CFD (compressible Navier-Stokes)
Source: https://doi.org/10.5281/zenodo.6993294

Expected file: 2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5 (~2.3 GB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import time
import h5py
from typing import Tuple, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# Part 1: PDEBench Data Loader
# ==============================================================================

class PDEBenchLoader:
    """Load and preprocess PDEBench CFD data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"PDEBench data not found: {data_path}\n"
                f"Please download from: https://doi.org/10.5281/zenodo.6993294\n"
                f"Expected file: 2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5"
            )
        
        # Inspect file
        print("Inspecting PDEBench data file...")
        with h5py.File(data_path, 'r') as f:
            print(f"  Keys: {list(f.keys())}")
            if 'tensor' in f:
                self.full_shape = f['tensor'].shape
                print(f"  Full shape: {self.full_shape}")
                # Usually (n_samples, time_steps, nx, ny, channels)
            if 'nu' in f:
                print(f"  Viscosity: {f['nu'][()]}")
    
    def load_velocity_data(self, 
                          n_samples: Optional[int] = None,
                          time_idx: int = 0) -> torch.Tensor:
        """
        Load velocity data from PDEBench.
        
        For 2D_CFD, channels are typically:
        - Channel 0: density
        - Channel 1: x-velocity
        - Channel 2: y-velocity
        - Channel 3: pressure
        """
        print(f"\nLoading PDEBench data...")
        
        with h5py.File(self.data_path, 'r') as f:
            data = f['tensor']
            
            # Determine actual shape
            total_samples = data.shape[0]
            if n_samples is None or n_samples > total_samples:
                n_samples = total_samples
            
            # Load subset
            print(f"  Loading {n_samples}/{total_samples} samples...")
            
            # PDEBench format: (samples, time, x, y, channels)
            # We want: (samples, velocity_channels, x, y)
            raw_data = data[:n_samples, time_idx]  # (n_samples, nx, ny, channels)
            
            # Extract velocity (assuming channels 1 and 2 are vx, vy)
            # Note: adjust based on actual data format
            if raw_data.shape[-1] >= 3:
                # Format: [density, vx, vy, pressure, ...]
                velocity = raw_data[..., 1:3]  # (n_samples, nx, ny, 2)
            else:
                # Fallback: use all channels as velocity
                velocity = raw_data[..., :2]
            
            # Convert to (n_samples, 2, nx, ny) for 2D
            velocity = np.transpose(velocity, (0, 3, 1, 2))
            
            # Normalize
            mean = velocity.mean(axis=(0, 2, 3), keepdims=True)
            std = velocity.std(axis=(0, 2, 3), keepdims=True)
            velocity = (velocity - mean) / (std + 1e-8)
            
        print(f"  Loaded velocity: {velocity.shape}")
        print(f"  Range: [{velocity.min():.3f}, {velocity.max():.3f}]")
        
        return torch.from_numpy(velocity).float()
    
    def compute_derivatives_2d(self, velocity: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute spatial derivatives for SGS stress calculation.
        
        Returns du/dx, du/dy, dv/dx, dv/dy
        """
        B, C, Nx, Ny = velocity.shape
        
        # Compute gradients using finite differences
        # du/dx
        du_dx = (torch.roll(velocity[:, 0], -1, dims=1) - 
                torch.roll(velocity[:, 0], 1, dims=1)) / 2.0
        
        # du/dy
        du_dy = (torch.roll(velocity[:, 0], -1, dims=2) - 
                torch.roll(velocity[:, 0], 1, dims=2)) / 2.0
        
        # dv/dx
        dv_dx = (torch.roll(velocity[:, 1], -1, dims=1) - 
                torch.roll(velocity[:, 1], 1, dims=1)) / 2.0
        
        # dv/dy
        dv_dy = (torch.roll(velocity[:, 1], -1, dims=2) - 
                torch.roll(velocity[:, 1], 1, dims=2)) / 2.0
        
        return du_dx, du_dy, dv_dx, dv_dy
    
    def compute_sgs_stress_2d(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute SGS stress for 2D turbulence.
        
        Returns tau with shape (B, 3, Nx, Ny):
        - tau_xx, tau_yy, tau_xy
        """
        print("Computing 2D SGS stress...")
        B, C, Nx, Ny = velocity.shape
        
        # Get derivatives
        du_dx, du_dy, dv_dx, dv_dy = self.compute_derivatives_2d(velocity)
        
        # Smagorinsky model for SGS (simplified)
        # tau_ij = -2 * nu_t * S_ij
        # where S_ij is strain rate
        
        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)
        
        # Eddy viscosity (simplified)
        S_mag = torch.sqrt(2 * (S_xx**2 + S_yy**2 + 2*S_xy**2))
        nu_t = 0.01 * S_mag  # Smagorinsky constant
        
        # SGS stress
        tau = torch.zeros(B, 3, Nx, Ny)
        tau[:, 0] = -2 * nu_t * S_xx  # tau_xx
        tau[:, 1] = -2 * nu_t * S_yy  # tau_yy
        tau[:, 2] = -2 * nu_t * S_xy  # tau_xy
        
        print(f"  SGS shape: {tau.shape}")
        print(f"  Range: [{tau.min():.3f}, {tau.max():.3f}]")
        
        return tau
    
    def prepare_training_data(self,
                             n_samples: int = 100,
                             train_split: float = 0.8,
                             cache_dir: str = './pdebench_cache') -> Tuple[torch.Tensor, ...]:
        """Prepare train/val split"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'data_n{n_samples}.pt')
        
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            data = torch.load(cache_file)
            return data['u_train'], data['tau_train'], data['u_val'], data['tau_val']
        
        # Load velocity
        velocity = self.load_velocity_data(n_samples=n_samples)
        
        # Compute SGS
        tau = self.compute_sgs_stress_2d(velocity)
        
        # Add dummy z-dimension for 3D model compatibility
        velocity = velocity.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
        tau = tau.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
        
        # Split
        n_train = int(len(velocity) * train_split)
        u_train, u_val = velocity[:n_train], velocity[n_train:]
        tau_train, tau_val = tau[:n_train], tau[n_train:]
        
        # Cache
        torch.save({
            'u_train': u_train,
            'tau_train': tau_train,
            'u_val': u_val,
            'tau_val': tau_val
        }, cache_file)
        print(f"Cached to {cache_file}")
        
        return u_train, tau_train, u_val, tau_val


# ==============================================================================
# Part 2: 2D FNO Model (adapted from 3D)
# ==============================================================================

class FNO2DLayer(nn.Module):
    """2D Fourier Neural Operator Layer"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes, 2) * scale
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Nx, Ny = x.shape
        
        # FFT
        x_ft = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        
        # Multiply low frequencies
        out_ft = torch.zeros_like(x_ft)
        m = min(self.modes, x_ft.shape[2], x_ft.shape[3])
        W = torch.view_as_complex(self.weights[:, :, :m, :m])
        
        out_ft[:, :, :m, :m] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m, :m],
            W
        )
        
        # IFFT
        x_fno = torch.fft.irfft2(out_ft, s=(Nx, Ny), dim=(2, 3), norm='ortho')
        
        return x_fno + self.skip(x)


class TurbulenceFNO2D(nn.Module):
    """2D FNO for turbulence SGS modeling"""
    
    def __init__(self, modes: int = 12, width: int = 64, n_layers: int = 4):
        super().__init__()
        
        self.lift = nn.Conv2d(2, width, 1)  # 2 velocity components
        self.fno_layers = nn.ModuleList([
            FNO2DLayer(width, width, modes) for _ in range(n_layers)
        ])
        self.activation = nn.GELU()
        self.project = nn.Sequential(
            nn.Conv2d(width, width // 2, 1),
            nn.GELU(),
            nn.Conv2d(width // 2, 3, 1)  # 3 SGS components
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, Nx, Ny)
        x = self.lift(x)
        for layer in self.fno_layers:
            x = x + layer(x)
            x = self.activation(x)
        x = self.project(x)
        return x


# ==============================================================================
# Part 3: Main Experiment
# ==============================================================================

def main():
    """Run PDEBench experiment with real data"""
    
    print("="*70)
    print("Axiom-OS + PDEBench Real Data Experiment")
    print("="*70)
    print("This experiment uses REAL simulation data from PDEBench!")
    print("="*70)
    
    # Configuration
    DATA_PATH = './pdebench_data/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5'
    N_SAMPLES = 100
    N_EPOCHS = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    
    # Check for data
    if not os.path.exists(DATA_PATH):
        print(f"\n{'='*70}")
        print("ERROR: PDEBench data not found!")
        print(f"{'='*70}")
        print(f"\nExpected file: {DATA_PATH}")
        print("\nTo download:")
        print("  1. Visit: https://doi.org/10.5281/zenodo.6993294")
        print("  2. Download: 2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5 (~2.3 GB)")
        print("  3. Place in: ./pdebench_data/")
        print("\nOr use wget:")
        print("  mkdir -p pdebench_data")
        print("  cd pdebench_data")
        print("  wget https://zenodo.org/record/6993294/files/2D_CFD_Rand_M1_0_Eta1e-08_Zeta1e-08.hdf5")
        print(f"{'='*70}")
        return
    
    # Load data
    print("\n" + "="*70)
    print("Loading PDEBench Real Data")
    print("="*70)
    
    try:
        loader = PDEBenchLoader(DATA_PATH)
        u_train, tau_train, u_val, tau_val = loader.prepare_training_data(
            n_samples=N_SAMPLES,
            train_split=0.8
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"\nData loaded:")
    print(f"  Train: {u_train.shape} -> {tau_train.shape}")
    print(f"  Val:   {u_val.shape} -> {tau_val.shape}")
    
    # Use only 2D (remove dummy z-dimension for 2D model)
    u_train_2d = u_train[:, :, :, :, 0]  # (B, 2, Nx, Ny)
    tau_train_2d = tau_train[:, :, :, :, 0]  # (B, 3, Nx, Ny)
    u_val_2d = u_val[:, :, :, :, 0]
    tau_val_2d = tau_val[:, :, :, :, 0]
    
    print(f"\n2D data:")
    print(f"  Train: {u_train_2d.shape} -> {tau_train_2d.shape}")
    
    # Initialize model
    print("\n" + "="*70)
    print("Training FNO2D-RCLN")
    print("="*70)
    
    model = TurbulenceFNO2D(modes=12, width=64, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Training
    model = model.to(device)
    u_train_2d = u_train_2d.to(device)
    tau_train_2d = tau_train_2d.to(device)
    u_val_2d = u_val_2d.to(device)
    tau_val_2d = tau_val_2d.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_r2 = -float('inf')
    
    print(f"\nTraining for {N_EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(N_EPOCHS):
        # Train
        model.train()
        optimizer.zero_grad()
        
        pred = model(u_train_2d)
        loss = F.mse_loss(pred, tau_train_2d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Val
        model.eval()
        with torch.no_grad():
            pred_val = model(u_val_2d)
            val_loss = F.mse_loss(pred_val, tau_val_2d).item()
            
            # R2
            t_flat = tau_val_2d.flatten()
            p_flat = pred_val.flatten()
            ss_res = ((t_flat - p_flat)**2).sum().item()
            ss_tot = ((t_flat - t_flat.mean())**2).sum().item()
            r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['val_r2'].append(r2)
        
        if r2 > best_r2:
            best_r2 = r2
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} ({elapsed:.1f}s): "
                  f"Train={loss:.5f}, Val={val_loss:.5f}, R2={r2:.4f}")
    
    # Results
    print("\n" + "="*70)
    print("Final Results (REAL PDEBench Data)")
    print("="*70)
    print(f"Best R2: {best_r2:.4f}")
    print(f"Final R2: {history['val_r2'][-1]:.4f}")
    
    # Plot
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.semilogy(history['train_loss'], label='Train')
    ax.semilogy(history['val_loss'], label='Val')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R2
    ax = axes[0, 1]
    ax.plot(history['val_r2'])
    ax.set_title(f'Validation R2 (Best: {best_r2:.4f})')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Info
    ax = axes[0, 2]
    info = f"Data: PDEBench 2D CFD\nSamples: {N_SAMPLES}\nBest R2: {best_r2:.4f}"
    ax.text(0.5, 0.5, info, transform=ax.transAxes, ha='center', va='center',
            fontsize=12, fontfamily='monospace')
    ax.axis('off')
    
    # Visualizations
    model.eval()
    with torch.no_grad():
        pred_val = model(u_val_2d).cpu()
    
    vmin = min(tau_val_2d[0].min().item(), pred_val[0].min().item())
    vmax = max(tau_val_2d[0].max().item(), pred_val[0].max().item())
    
    ax = axes[1, 0]
    im = ax.imshow(pred_val[0, 0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title('Predicted T_xx')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 1]
    im = ax.imshow(tau_val_2d[0, 0].cpu(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title('True T_xx')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 2]
    error = (pred_val[0, 0] - tau_val_2d[0, 0].cpu()).abs()
    im = ax.imshow(error, cmap='Reds')
    ax.set_title('Absolute Error')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Axiom-OS + PDEBench: REAL 2D Turbulence Data', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'axiom_pdebench_real_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    print("\n" + "="*70)
    print("Experiment completed with REAL data!")
    print("="*70)


if __name__ == "__main__":
    main()
