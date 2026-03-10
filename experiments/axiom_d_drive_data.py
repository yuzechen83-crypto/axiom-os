"""
Axiom-OS D-Drive Data Experiment
=================================
Automatically find and use data files on D:\ drive.
Searches for:
- PDEBench data (*.h5, *.hdf5)
- NumPy data (*.npy, *.npz)
- PyTorch data (*.pt, *.pth)
- NetCDF data (*.nc)

Will attempt to load and use any compatible turbulence/CFD data found.
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
import h5py
from pathlib import Path
from typing import Tuple, Optional, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_data_files(drive: str = 'D:', 
                   extensions: List[str] = None) -> List[Path]:
    """
    Search for data files on specified drive.
    
    Args:
        drive: Drive letter (e.g., 'D:')
        extensions: List of file extensions to search for
        
    Returns:
        List of Path objects for found files
    """
    if extensions is None:
        extensions = ['.h5', '.hdf5', '.hdf', '.nc', '.npy', '.npz', '.pt', '.pth']
    
    print(f"\nSearching {drive} for data files...")
    print(f"Looking for: {', '.join(extensions)}")
    
    found_files = []
    drive_path = Path(drive)
    
    if not drive_path.exists():
        print(f"  Drive {drive} not found!")
        return found_files
    
    # Search for files
    for ext in extensions:
        try:
            files = list(drive_path.rglob(f'*{ext}'))
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                found_files.append((f, size_mb))
                print(f"  Found: {f} ({size_mb:.1f} MB)")
        except PermissionError:
            continue
        except Exception as e:
            print(f"  Error searching for {ext}: {e}")
    
    # Sort by size (largest first)
    found_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTotal files found: {len(found_files)}")
    return [f[0] for f in found_files]


def inspect_h5_file(filepath: Path) -> Optional[Dict]:
    """
    Inspect an HDF5 file and return metadata.
    
    Returns:
        Dict with file info, or None if file is corrupted
    """
    try:
        with h5py.File(filepath, 'r') as f:
            info = {
                'path': str(filepath),
                'keys': list(f.keys()),
                'datasets': []
            }
            
            def inspect_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info['datasets'].append({
                        'name': name,
                        'shape': obj.shape,
                        'dtype': str(obj.dtype),
                        'size_mb': obj.nbytes / (1024 * 1024)
                    })
            
            f.visititems(inspect_item)
            
            return info
    except OSError as e:
        print(f"  Warning: File appears corrupted: {e}")
        return None
    except Exception as e:
        print(f"  Error inspecting file: {e}")
        return None


def load_pdebench_data(filepath: Path, 
                       max_samples: int = 100) -> Optional[Tuple[torch.Tensor, ...]]:
    """
    Attempt to load PDEBench format data.
    
    Returns:
        (velocity, sgs_stress) tensors, or None if loading fails
    """
    print(f"\nAttempting to load PDEBench data from {filepath}...")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Try to find velocity/field data
            data = None
            
            # Common PDEBench keys
            if 'tensor' in f:
                data = f['tensor']
            elif 'data' in f:
                data = f['data']
            elif 'u' in f:
                data = f['u']
            elif 'velocity' in f:
                data = f['velocity']
            else:
                # Use first large dataset
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and f[key].size > 1000:
                        data = f[key]
                        break
            
            if data is None:
                print("  No suitable dataset found")
                return None
            
            print(f"  Found dataset: {data.shape}, dtype={data.dtype}")
            
            # Load subset
            shape = data.shape
            if len(shape) < 3:
                print("  Dataset has insufficient dimensions")
                return None
            
            # Determine how to slice the data
            n_samples = min(shape[0], max_samples)
            
            # Load data
            if len(shape) == 5:
                # Format: (samples, time, x, y, channels)
                raw_data = data[:n_samples, 0]  # First time step
            elif len(shape) == 4:
                # Format: (samples, x, y, channels)
                raw_data = data[:n_samples]
            elif len(shape) == 3:
                # Format: (samples, x, y) - add channel dim
                raw_data = data[:n_samples][..., np.newaxis]
            else:
                print(f"  Unsupported shape: {shape}")
                return None
            
            print(f"  Loaded raw data: {raw_data.shape}")
            
            # Convert to velocity field
            # Assume last dimension is channels (velocity components)
            if raw_data.shape[-1] >= 2:
                velocity = raw_data[..., :2]  # Take first 2 as vx, vy
            else:
                # If only 1 channel, duplicate for 2D
                velocity = np.concatenate([raw_data, raw_data], axis=-1)
            
            # Transpose to (B, C, H, W)
            velocity = np.transpose(velocity, (0, 3, 1, 2))
            
            # Normalize
            mean = velocity.mean(axis=(0, 2, 3), keepdims=True)
            std = velocity.std(axis=(0, 2, 3), keepdims=True)
            velocity = (velocity - mean) / (std + 1e-8)
            
            velocity_t = torch.from_numpy(velocity).float()
            
            # Compute synthetic SGS stress for demonstration
            print("  Computing SGS stress...")
            sgs = compute_sgs_2d(velocity_t)
            
            print(f"  Velocity: {velocity_t.shape}, range: [{velocity_t.min():.3f}, {velocity_t.max():.3f}]")
            print(f"  SGS: {sgs.shape}, range: [{sgs.min():.3f}, {sgs.max():.3f}]")
            
            return velocity_t, sgs
            
    except Exception as e:
        print(f"  Error loading file: {e}")
        return None


def compute_sgs_2d(velocity: torch.Tensor) -> torch.Tensor:
    """Compute 2D SGS stress tensor"""
    B, C, H, W = velocity.shape
    
    # Simple filtering
    kernel = torch.ones(1, 1, 3, 3) / 9.0
    
    # Filter velocities
    u_f = F.conv2d(velocity[:, 0:1], kernel, padding=1)
    v_f = F.conv2d(velocity[:, 1:2], kernel, padding=1)
    
    # SGS components
    sgs = torch.zeros(B, 3, H, W)
    sgs[:, 0] = (velocity[:, 0]**2 - u_f.squeeze(1)**2)  # tau_xx
    sgs[:, 1] = (velocity[:, 1]**2 - v_f.squeeze(1)**2)  # tau_yy
    sgs[:, 2] = (velocity[:, 0] * velocity[:, 1] - u_f.squeeze(1) * v_f.squeeze(1))  # tau_xy
    
    return sgs


def load_numpy_data(filepath: Path) -> Optional[Tuple[torch.Tensor, ...]]:
    """Load NumPy format data"""
    print(f"\nAttempting to load NumPy data from {filepath}...")
    
    try:
        data = np.load(filepath, allow_pickle=True)
        
        # Handle different numpy formats
        if isinstance(data, np.ndarray):
            velocity = data
        elif isinstance(data, np.lib.npyio.NpzFile):
            # .npz file - try common keys
            if 'velocity' in data:
                velocity = data['velocity']
            elif 'u' in data:
                velocity = data['u']
            elif 'data' in data:
                velocity = data['data']
            else:
                velocity = data[data.files[0]]
        else:
            print(f"  Unknown numpy format: {type(data)}")
            return None
        
        print(f"  Loaded: {velocity.shape}")
        
        # Process similar to HDF5
        if len(velocity.shape) == 4:
            # (B, H, W, C) -> (B, C, H, W)
            velocity = np.transpose(velocity, (0, 3, 1, 2))
        elif len(velocity.shape) == 3:
            # (B, H, W) -> (B, 1, H, W) -> (B, 2, H, W)
            velocity = velocity[:, np.newaxis, :, :]
            velocity = np.repeat(velocity, 2, axis=1)
        
        velocity_t = torch.from_numpy(velocity).float()
        
        # Normalize
        mean = velocity_t.mean(dim=(0, 2, 3), keepdim=True)
        std = velocity_t.std(dim=(0, 2, 3), keepdim=True)
        velocity_t = (velocity_t - mean) / (std + 1e-8)
        
        # Compute SGS
        sgs = compute_sgs_2d(velocity_t)
        
        return velocity_t, sgs
        
    except Exception as e:
        print(f"  Error loading numpy file: {e}")
        return None


def load_torch_data(filepath: Path) -> Optional[Tuple[torch.Tensor, ...]]:
    """Load PyTorch format data"""
    print(f"\nAttempting to load PyTorch data from {filepath}...")
    
    try:
        data = torch.load(filepath, map_location='cpu')
        
        # Extract velocity
        if isinstance(data, dict):
            if 'velocity' in data:
                velocity = data['velocity']
            elif 'u' in data:
                velocity = data['u']
            elif 'data' in data:
                velocity = data['data']
            else:
                velocity = list(data.values())[0]
        elif isinstance(data, torch.Tensor):
            velocity = data
        else:
            print(f"  Unknown torch format: {type(data)}")
            return None
        
        print(f"  Loaded: {velocity.shape}")
        
        # Ensure 4D (B, C, H, W)
        while len(velocity.shape) < 4:
            velocity = velocity.unsqueeze(0)
        
        if len(velocity.shape) > 4:
            # Flatten extra dimensions
            B = velocity.shape[0]
            velocity = velocity.view(B, -1, velocity.shape[-2], velocity.shape[-1])
        
        # Normalize
        mean = velocity.mean(dim=(0, 2, 3), keepdim=True)
        std = velocity.std(dim=(0, 2, 3), keepdim=True)
        velocity = (velocity - mean) / (std + 1e-8)
        
        # Compute SGS
        sgs = compute_sgs_2d(velocity)
        
        return velocity, sgs
        
    except Exception as e:
        print(f"  Error loading torch file: {e}")
        return None


def train_fno2d(u_train, sgs_train, u_val, sgs_val, n_epochs=50, device='cuda'):
    """Train 2D FNO model"""
    
    # Simple 2D FNO
    class FNO2D(nn.Module):
        def __init__(self, modes=12, width=64):
            super().__init__()
            self.lift = nn.Conv2d(2, width, 1)
            self.conv1 = nn.Conv2d(width, width, 1)
            self.conv2 = nn.Conv2d(width, width, 1)
            self.conv3 = nn.Conv2d(width, width, 1)
            self.proj = nn.Conv2d(width, 3, 1)
            
        def forward(self, x):
            x = self.lift(x)
            x = F.gelu(self.conv1(x) + x)
            x = F.gelu(self.conv2(x) + x)
            x = F.gelu(self.conv3(x) + x)
            x = self.proj(x)
            return x
    
    model = FNO2D().to(device)
    u_train = u_train.to(device)
    sgs_train = sgs_train.to(device)
    u_val = u_val.to(device)
    sgs_val = sgs_val.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    
    print(f"\nTraining for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(u_train)
        loss = F.mse_loss(pred, sgs_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_val = model(u_val)
            val_loss = F.mse_loss(pred_val, sgs_val).item()
            
            # R2
            t_flat = sgs_val.flatten()
            p_flat = pred_val.flatten()
            ss_res = ((t_flat - p_flat)**2).sum().item()
            ss_tot = ((t_flat - t_flat.mean())**2).sum().item()
            r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['val_r2'].append(r2)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss={val_loss:.5f}, R2={r2:.4f}")
    
    return model, history


def main():
    """Main experiment"""
    print("="*70)
    print("Axiom-OS D-Drive Data Experiment")
    print("="*70)
    
    # Find data files
    data_files = find_data_files('D:')
    
    if not data_files:
        print("\nNo data files found on D: drive!")
        print("Please ensure you have data files (.h5, .npy, .pt, etc.) on D:")
        return
    
    # Try to load each file
    velocity = None
    sgs = None
    used_file = None
    
    for filepath in data_files:
        print(f"\n{'='*70}")
        print(f"Trying: {filepath}")
        print(f"{'='*70}")
        
        # Inspect if HDF5
        if filepath.suffix in ['.h5', '.hdf5', '.hdf']:
            info = inspect_h5_file(filepath)
            if info:
                print(f"  Keys: {info['keys']}")
                for ds in info['datasets'][:3]:
                    print(f"    {ds['name']}: {ds['shape']}, {ds['size_mb']:.1f} MB")
            
            result = load_pdebench_data(filepath)
            if result:
                velocity, sgs = result
                used_file = filepath
                break
        
        # Try NumPy
        elif filepath.suffix in ['.npy', '.npz']:
            result = load_numpy_data(filepath)
            if result:
                velocity, sgs = result
                used_file = filepath
                break
        
        # Try PyTorch
        elif filepath.suffix in ['.pt', '.pth']:
            result = load_torch_data(filepath)
            if result:
                velocity, sgs = result
                used_file = filepath
                break
    
    if velocity is None:
        print("\nCould not load any data files!")
        return
    
    # Split train/val
    print(f"\n{'='*70}")
    print("Preparing Data")
    print(f"{'='*70}")
    
    n_samples = len(velocity)
    n_train = int(0.8 * n_samples)
    
    u_train, u_val = velocity[:n_train], velocity[n_train:]
    sgs_train, sgs_val = sgs[:n_train], sgs[n_train:]
    
    print(f"Total samples: {n_samples}")
    print(f"Train: {u_train.shape}")
    print(f"Val: {u_val.shape}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    model, history = train_fno2d(u_train, sgs_train, u_val, sgs_val, n_epochs=50, device=device)
    
    # Results
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"Data file: {used_file}")
    print(f"Best R2: {max(history['val_r2']):.4f}")
    print(f"Final R2: {history['val_r2'][-1]:.4f}")
    
    # Plot
    print("\nGenerating plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].semilogy(history['train_loss'], label='Train')
    axes[0, 0].semilogy(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['val_r2'])
    axes[0, 1].set_title(f'R2 (Best: {max(history["val_r2"]):.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Visualize
    model.eval()
    with torch.no_grad():
        pred_val = model(u_val.to(device)).cpu()
    
    slice_idx = 0
    vmax = max(sgs_val[slice_idx, 0].abs().max().item(), pred_val[slice_idx, 0].abs().max().item())
    
    axes[1, 0].imshow(pred_val[slice_idx, 0], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('Predicted')
    
    axes[1, 1].imshow(sgs_val[slice_idx, 0], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title('True')
    
    plt.suptitle(f'Axiom-OS: {used_file.name}', fontsize=12)
    plt.tight_layout()
    
    save_path = 'axiom_d_drive_results.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    
    print(f"\n{'='*70}")
    print("Experiment completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
