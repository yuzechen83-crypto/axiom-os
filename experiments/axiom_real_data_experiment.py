"""
Axiom-OS Real Data Experiment - Multi-Source
=============================================
Attempts to use real data from multiple sources:
1. PDEBench (local files or download)
2. JHTDB via SciServer (if available)
3. JHTDB via direct HTTP (if network allows)
4. High-quality synthetic (fallback)

Author: Axiom-OS Team
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
import json
import time
from typing import Tuple, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# Part 1: Multi-Source Data Loader
# ==============================================================================

class RealDataLoader:
    """
    Unified loader for real turbulence data from multiple sources.
    Priority: PDEBench > JHTDB SciServer > JHTDB HTTP > Synthetic
    """
    
    def __init__(self, cache_dir: str = './real_data_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.data_source = None
        
    def load_pdebench(self, 
                      data_path: Optional[str] = None,
                      dataset_name: str = '1D_Advection',
                      download: bool = False) -> Optional[Dict]:
        """
        Load PDEBench data.
        
        PDEBench datasets:
        - 1D_Advection, 1D_Burgers, 1D_CFD, 1D_Diffusion
        - 2D_Darcy, 2D_Diffusion, 2D_CFD, 2D_ShallowWater
        - 3D_CFD (if available)
        """
        print("\n[Source 1] Attempting PDEBench...")
        
        try:
            # Try to import PDEBench loader
            from axiom_os.datasets.pdebench import PDEBenchDataset
            
            if data_path and os.path.exists(data_path):
                print(f"  Loading from: {data_path}")
                data = np.load(data_path)
                self.data_source = f"PDEBench-{dataset_name}"
                return {
                    'velocity': torch.from_numpy(data['velocity']).float(),
                    'source': 'PDEBench'
                }
            elif download:
                print("  Attempting download...")
                # Download logic would go here
                return None
            else:
                print("  PDEBench data not found locally")
                return None
                
        except Exception as e:
            print(f"  PDEBench error: {e}")
            return None
    
    def load_jhtdb_sciserver(self, 
                              dataset: str = 'isotropic1024coarse',
                              n_samples: int = 10,
                              cube_size: int = 64) -> Optional[Dict]:
        """
        Load JHTDB data via SciServer (for use in SciServer Jupyter environment).
        
        Requires:
        - Running in SciServer Jupyter notebook
        - pip install giverny
        - Authenticated SciServer account
        """
        print("\n[Source 2] Attempting JHTDB via SciServer...")
        
        try:
            from giverny.turbulence_dataset import TurbulenceDataset
            from giverny.turbulence_tools import getCutout
            
            print(f"  SciServer giverny available!")
            print(f"  Dataset: {dataset}")
            
            # Create dataset object
            dataset_obj = TurbulenceDataset(
                dataset_name=dataset,
                output_path=self.cache_dir,
                x_start=512, y_start=512, z_start=512,
                x_end=512+cube_size, y_end=512+cube_size, z_end=512+cube_size,
                time=0.0,
                field='u'
            )
            
            # Download multiple samples at different times/locations
            velocity_list = []
            times = np.linspace(0, 2.56, n_samples)
            
            for i, t in enumerate(times):
                print(f"  Downloading sample {i+1}/{n_samples} at t={t:.2f}...")
                dataset_obj.time = t
                # Random offset for spatial variety
                np.random.seed(i)
                offset = np.random.randint(0, 1024-cube_size, 3)
                dataset_obj.x_start, dataset_obj.y_start, dataset_obj.z_start = offset
                dataset_obj.x_end = offset[0] + cube_size
                dataset_obj.y_end = offset[1] + cube_size
                dataset_obj.z_end = offset[2] + cube_size
                
                u = dataset_obj.download_data()
                velocity_list.append(torch.from_numpy(u).float())
            
            velocity = torch.stack(velocity_list, dim=0)
            self.data_source = f"JHTDB-SciServer-{dataset}"
            
            return {
                'velocity': velocity,
                'source': 'JHTDB-SciServer'
            }
            
        except ImportError:
            print("  SciServer/giverny not available (not in SciServer environment)")
            return None
        except Exception as e:
            print(f"  SciServer error: {e}")
            return None
    
    def load_jhtdb_http(self,
                       dataset: str = 'isotropic1024coarse',
                       n_samples: int = 5,
                       cube_size: int = 32) -> Optional[Dict]:
        """
        Load JHTDB via direct HTTP API.
        Works without SciServer but may have rate limits.
        """
        print("\n[Source 3] Attempting JHTDB via HTTP...")
        
        try:
            from experiments.jhtdb_direct_access import JHTDBClient
            
            client = JHTDBClient()
            print(f"  JHTDB HTTP client created")
            
            # Try to get a small test cube first
            print(f"  Testing with small cube...")
            test_cube = client.get_cutout(dataset, 0.0, (512, 512, 512), 16, 'u')
            print(f"  Test successful! Shape: {test_cube.shape}")
            
            # Download full dataset
            velocity_list = []
            times = np.linspace(0, 2.56, n_samples)
            
            for i, t in enumerate(times):
                print(f"  Downloading {i+1}/{n_samples} at t={t:.2f}...")
                np.random.seed(i)
                origin = tuple(np.random.randint(0, 1024-cube_size, 3))
                
                u = client.get_cutout(dataset, t, origin, cube_size, 'u')
                # Convert to torch (N, 3, H, W, D)
                u_tensor = torch.from_numpy(u).permute(3, 0, 1, 2).float()
                velocity_list.append(u_tensor)
            
            velocity = torch.stack(velocity_list, dim=0)
            self.data_source = f"JHTDB-HTTP-{dataset}"
            
            return {
                'velocity': velocity,
                'source': 'JHTDB-HTTP'
            }
            
        except Exception as e:
            print(f"  HTTP error: {e}")
            return None
    
    def generate_realistic_turbulence(self,
                                     n_samples: int = 128,
                                     grid_size: int = 32,
                                     re_lambda: float = 433.0) -> Dict:
        """
        Generate high-quality synthetic turbulence as fallback.
        Uses proper Kolmogorov spectrum to match JHTDB characteristics.
        """
        print("\n[Source 4] Generating realistic synthetic turbulence...")
        
        from experiments.axiom_jhtdb_full import TurbulenceGenerator
        
        generator = TurbulenceGenerator(re_lambda=re_lambda)
        velocity = generator.generate_velocity_field(n_samples, grid_size)
        
        self.data_source = f"Synthetic-Re{re_lambda:.0f}"
        
        return {
            'velocity': velocity,
            'source': 'Synthetic-Kolmogorov'
        }
    
    def load_data(self,
                  n_samples: int = 128,
                  grid_size: int = 32,
                  train_split: float = 0.8) -> Tuple[torch.Tensor, ...]:
        """
        Try all data sources and return the first successful one.
        """
        cache_file = os.path.join(
            self.cache_dir, 
            f"real_data_n{n_samples}_g{grid_size}.pt"
        )
        
        # Check cache
        if os.path.exists(cache_file):
            print(f"\nLoading cached data from {cache_file}")
            data = torch.load(cache_file)
            self.data_source = data.get('source', 'Cached')
            return data['u_train'], data['tau_train'], data['u_val'], data['tau_val']
        
        # Try sources in order
        result = None
        
        # 1. Try PDEBench
        result = self.load_pdebench()
        
        # 2. Try JHTDB SciServer
        if result is None:
            result = self.load_jhtdb_sciserver(n_samples=n_samples, cube_size=grid_size)
        
        # 3. Try JHTDB HTTP
        if result is None:
            result = self.load_jhtdb_http(n_samples=min(n_samples, 10), cube_size=grid_size)
        
        # 4. Fallback to synthetic
        if result is None:
            result = self.generate_realistic_turbulence(n_samples, grid_size)
        
        # Process velocity to get SGS stress
        velocity = result['velocity']
        print(f"\nLoaded velocity: {velocity.shape}")
        print(f"Data source: {self.data_source}")
        
        # Compute SGS stress
        print("Computing SGS stress...")
        from experiments.axiom_jhtdb_full import TurbulenceGenerator
        generator = TurbulenceGenerator()
        tau = generator.compute_sgs_stress(velocity)
        
        print(f"SGS stress: {tau.shape}")
        print(f"Velocity range: [{velocity.min():.3f}, {velocity.max():.3f}]")
        print(f"SGS range: [{tau.min():.3f}, {tau.max():.3f}]")
        
        # Split train/val
        n_train = int(len(velocity) * train_split)
        u_train, u_val = velocity[:n_train], velocity[n_train:]
        tau_train, tau_val = tau[:n_train], tau[n_train:]
        
        # Cache
        torch.save({
            'u_train': u_train,
            'tau_train': tau_train,
            'u_val': u_val,
            'tau_val': tau_val,
            'source': self.data_source
        }, cache_file)
        print(f"Cached to {cache_file}")
        
        return u_train, tau_train, u_val, tau_val


# ==============================================================================
# Part 2: Models (same as before)
# ==============================================================================

class FNO3DLayer(nn.Module):
    """3D Fourier Neural Operator Layer"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes, modes, 2) * scale
        )
        self.skip = nn.Conv3d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N1, N2, N3 = x.shape
        
        # FFT
        x_ft = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')
        
        # Multiply low frequencies
        out_ft = torch.zeros_like(x_ft)
        m = min(self.modes, x_ft.shape[2], x_ft.shape[3], x_ft.shape[4])
        W = torch.view_as_complex(self.weights[:, :, :m, :m, :m])
        
        out_ft[:, :, :m, :m, :m] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :m, :m, :m],
            W
        )
        
        # IFFT
        x_fno = torch.fft.irfftn(out_ft, s=(N1, N2, N3), dim=(2, 3, 4), norm='ortho')
        
        return x_fno + self.skip(x)


class TurbulenceFNO3D(nn.Module):
    """Full 3D FNO for turbulence SGS modeling"""
    
    def __init__(self, modes: int = 12, width: int = 64, n_layers: int = 4):
        super().__init__()
        
        self.lift = nn.Conv3d(3, width, 1)
        self.fno_layers = nn.ModuleList([
            FNO3DLayer(width, width, modes) for _ in range(n_layers)
        ])
        self.activation = nn.GELU()
        self.project = nn.Sequential(
            nn.Conv3d(width, width // 2, 1),
            nn.GELU(),
            nn.Conv3d(width // 2, 6, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for layer in self.fno_layers:
            x = x + layer(x)
            x = self.activation(x)
        x = self.project(x)
        return x


# ==============================================================================
# Part 3: Training
# ==============================================================================

def train_model(model, u_train, tau_train, u_val, tau_val, 
                n_epochs: int = 50, batch_size: int = 4, 
                device: str = 'cuda') -> Dict:
    """Train model with progress tracking"""
    
    model = model.to(device)
    u_train = u_train.to(device)
    tau_train = tau_train.to(device)
    u_val = u_val.to(device)
    tau_val = tau_val.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'r2_components': []}
    best_r2 = -float('inf')
    
    print(f"\nTraining for {n_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        n_samples = len(u_train)
        indices = torch.randperm(n_samples)
        total_loss = 0.0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            u_batch = u_train[batch_idx]
            tau_batch = tau_train[batch_idx]
            
            optimizer.zero_grad()
            pred = model(u_batch)
            loss = F.mse_loss(pred, tau_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / (n_samples // batch_size + 1)
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(u_val)
            val_loss = F.mse_loss(pred_val, tau_val).item()
            
            # R2 per component
            r2_list = []
            for c in range(6):
                t = tau_val[:, c].flatten()
                p = pred_val[:, c].flatten()
                ss_res = ((t - p)**2).sum().item()
                ss_tot = ((t - t.mean())**2).sum().item()
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                r2_list.append(r2)
            
            r2_mean = np.mean(r2_list)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(r2_mean)
        history['r2_components'].append(r2_list)
        
        if r2_mean > best_r2:
            best_r2 = r2_mean
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} ({elapsed:.1f}s): "
                  f"Train={train_loss:.5f}, Val={val_loss:.5f}, R2={r2_mean:.4f}")
    
    return history, best_r2


# ==============================================================================
# Part 4: Main Experiment
# ==============================================================================

def main():
    """Run real data experiment"""
    
    print("="*70)
    print("Axiom-OS Real Data Experiment")
    print("="*70)
    print("Data sources (in priority order):")
    print("  1. PDEBench (local files)")
    print("  2. JHTDB via SciServer (if in Jupyter)")
    print("  3. JHTDB via HTTP (if network allows)")
    print("  4. High-quality synthetic Kolmogorov turbulence (fallback)")
    print("="*70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Configuration
    N_SAMPLES = 128
    GRID_SIZE = 32
    N_EPOCHS = 50
    
    # Load data
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    loader = RealDataLoader()
    u_train, tau_train, u_val, tau_val = loader.load_data(
        n_samples=N_SAMPLES,
        grid_size=GRID_SIZE,
        train_split=0.8
    )
    
    print(f"\nData source: {loader.data_source}")
    print(f"Train: {u_train.shape} -> {tau_train.shape}")
    print(f"Val:   {u_val.shape} -> {tau_val.shape}")
    
    # Initialize model
    print("\n" + "="*70)
    print("Training FNO3D-RCLN")
    print("="*70)
    
    model = TurbulenceFNO3D(modes=12, width=64, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Train
    history, best_r2 = train_model(
        model, u_train, tau_train, u_val, tau_val,
        n_epochs=N_EPOCHS, batch_size=4, device=device
    )
    
    # Final evaluation
    print("\n" + "="*70)
    print("Final Results")
    print("="*70)
    print(f"Best R2: {best_r2:.4f}")
    print(f"Final R2 by component:")
    components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
    for name, r2 in zip(components, history['r2_components'][-1]):
        print(f"  T_{name}: {r2:.4f}")
    
    # Plot
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.semilogy(history['train_loss'], label='Train')
    ax.semilogy(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R2 evolution
    ax = axes[0, 1]
    ax.plot(history['val_r2'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R2 Score')
    ax.set_title('Validation R2')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Final R2 by component
    ax = axes[0, 2]
    final_r2 = history['r2_components'][-1]
    colors = ['#1f77b4' if r > 0 else '#d62728' for r in final_r2]
    ax.bar(components, final_r2, color=colors)
    ax.set_ylabel('R2 Score')
    ax.set_title(f'Final R2 (Mean: {history["val_r2"][-1]:.3f})')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Data source info
    ax = axes[1, 0]
    info_text = f"Data Source: {loader.data_source}\n"
    info_text += f"Samples: {N_SAMPLES}\n"
    info_text += f"Grid: {GRID_SIZE}^3\n"
    info_text += f"Best R2: {best_r2:.4f}\n"
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', fontfamily='monospace')
    ax.axis('off')
    
    # Spatial visualization
    model.eval()
    with torch.no_grad():
        pred_val = model(u_val.to(device)).cpu()
    
    slice_idx = GRID_SIZE // 2
    vmax = max(tau_val[0, 0].abs().max().item(), pred_val[0, 0].abs().max().item())
    
    ax = axes[1, 1]
    im = ax.imshow(pred_val[0, 0, :, :, slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title('Predicted T_xx')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 2]
    im = ax.imshow(tau_val[0, 0, :, :, slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title('True T_xx')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'Axiom-OS Real Data: {loader.data_source}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = f'axiom_real_data_{loader.data_source.replace("-", "_")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("Experiment completed!")
    print("="*70)
    
    return {
        'data_source': loader.data_source,
        'best_r2': best_r2,
        'history': history
    }


if __name__ == "__main__":
    results = main()
