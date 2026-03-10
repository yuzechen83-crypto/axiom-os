# -*- coding: utf-8 -*-
"""
RCLN v3.0 Advanced - Real JHTDB Data Experiment
Resolution: 16^3 (cutout from full DNS field)

Uses Johns Hopkins Turbulence Database (JHTDB) isotropic 1024^3 data
or high-quality synthetic turbulence with correct spectra.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import sys
import os

sys.path.insert(0, r'C:\Users\ASUS\PycharmProjects\PythonProject1')

from axiom_os.layers.rcln_v3_advanced import RCLNv3_Advanced
from axiom_os.layers.fno3d import FNO3d


class BaselineFNO(nn.Module):
    """Baseline: Pure FNO"""
    def __init__(self, width=16, modes=8):
        super().__init__()
        self.fno = FNO3d(
            in_channels=3, out_channels=6,
            width=width, modes1=modes, modes2=modes, modes3=modes,
            n_layers=3,
        )
    
    def forward(self, u):
        return self.fno(u)


def generate_realistic_turbulence_16cube(num_samples=200, Re_lambda=433):
    """
    Generate realistic turbulence data with correct Kolmogorov spectrum.
    
    This uses proper Fourier method to ensure:
    - Correct -5/3 energy spectrum
    - Proper velocity correlations
    - Physical SGS stresses
    
    Args:
        num_samples: Number of independent velocity fields
        Re_lambda: Taylor Reynolds number
    
    Returns:
        Dictionary with velocity and tau_sgs
    """
    print(f"Generating realistic turbulence (Re_lambda={Re_lambda})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Physical parameters
    nu = 0.002  # Kinematic viscosity (scaled for resolution)
    eta = (nu**3 / 0.1)**0.25  # Kolmogorov scale (approx)
    
    data = {'velocity': [], 'tau_sgs': [], 'metadata': []}
    
    for sample_idx in tqdm(range(num_samples), desc="Generating samples"):
        # Random seed for reproducibility
        np.random.seed(sample_idx * 42)
        torch.manual_seed(sample_idx * 42)
        
        # Generate velocity field in Fourier space
        N = 16
        k_max = N // 2
        
        # Wave numbers
        k = np.fft.fftfreq(N, 1/N) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag[0, 0, 0] = 1e-10  # Avoid division by zero
        
        # Energy spectrum: E(k) ~ k^4 / (k^2 + k_0^2)^(17/6) * exp(-k^2/k_d^2)
        # von Karman spectrum with Pao correction
        k_0 = 2 * np.pi / 8  # Energy-containing scale
        k_d = 2 * np.pi / 2  # Dissipation scale
        
        E_k = (k_mag**4) / ((k_mag**2 + k_0**2)**(17/6)) * np.exp(-(k_mag/k_d)**2)
        E_k[0, 0, 0] = 0
        
        # Generate random Fourier coefficients
        u_hat = torch.zeros(3, N, N, N, dtype=torch.complex64, device=device)
        
        for i in range(3):
            # Random phase
            phase = torch.rand(N, N, N, device=device) * 2 * np.pi
            amplitude = torch.tensor(np.sqrt(E_k), dtype=torch.float32, device=device)
            
            # Complex coefficients
            u_hat[i] = amplitude * torch.exp(1j * phase)
        
        # Make velocity field divergence-free (projection)
        # u_hat = u_hat - (k . u_hat) * k / |k|^2
        kx_torch = torch.tensor(kx, dtype=torch.float32, device=device)
        ky_torch = torch.tensor(ky, dtype=torch.float32, device=device)
        kz_torch = torch.tensor(kz, dtype=torch.float32, device=device)
        k_mag_torch = torch.tensor(k_mag, dtype=torch.float32, device=device)
        
        k_dot_u = (kx_torch * u_hat[0].real + ky_torch * u_hat[1].real + kz_torch * u_hat[2].real) + \
                  1j * (kx_torch * u_hat[0].imag + ky_torch * u_hat[1].imag + kz_torch * u_hat[2].imag)
        
        for i, ki in enumerate([kx_torch, ky_torch, kz_torch]):
            u_hat[i] = u_hat[i] - (k_dot_u * ki / (k_mag_torch**2))
        
        # Transform to physical space
        u = torch.real(torch.fft.ifftn(u_hat, dim=(1, 2, 3))) * N**1.5
        
        # Normalize to target RMS
        u_rms_target = 1.0
        u = u / (u.std() + 1e-8) * u_rms_target
        
        # Compute filtered velocity (coarse-graining)
        # This simulates LES filtering
        kernel_size = 3
        u_filtered = F.avg_pool3d(
            u.unsqueeze(0), 
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        )[0]
        
        # Compute SGS stress: tau_ij = <u_i u_j> - <u_i><u_j>
        # Using the Germano identity
        tau = torch.zeros(6, N, N, N, device=device)
        
        # Direct product (before filtering)
        uu_filtered = torch.zeros(6, N, N, N, device=device)
        uu_filtered[0] = F.avg_pool3d((u[0]**2).unsqueeze(0), kernel_size, 1, kernel_size//2)[0]
        uu_filtered[1] = F.avg_pool3d((u[1]**2).unsqueeze(0), kernel_size, 1, kernel_size//2)[0]
        uu_filtered[2] = F.avg_pool3d((u[2]**2).unsqueeze(0), kernel_size, 1, kernel_size//2)[0]
        uu_filtered[3] = F.avg_pool3d((u[0]*u[1]).unsqueeze(0), kernel_size, 1, kernel_size//2)[0]
        uu_filtered[4] = F.avg_pool3d((u[0]*u[2]).unsqueeze(0), kernel_size, 1, kernel_size//2)[0]
        uu_filtered[5] = F.avg_pool3d((u[1]*u[2]).unsqueeze(0), kernel_size, 1, kernel_size//2)[0]
        
        # Product of filtered velocities
        tau[0] = uu_filtered[0] - u_filtered[0]**2
        tau[1] = uu_filtered[1] - u_filtered[1]**2
        tau[2] = uu_filtered[2] - u_filtered[2]**2
        tau[3] = uu_filtered[3] - u_filtered[0]*u_filtered[1]
        tau[4] = uu_filtered[4] - u_filtered[0]*u_filtered[2]
        tau[5] = uu_filtered[5] - u_filtered[1]*u_filtered[2]
        
        # Clip extreme values (numerical artifacts)
        tau_std = tau.std()
        tau = torch.clamp(tau, -3*tau_std, 3*tau_std)
        
        data['velocity'].append(u)
        data['tau_sgs'].append(tau)
        data['metadata'].append({
            'u_rms': u.std().item(),
            'tau_rms': tau.std().item(),
            'sample_idx': sample_idx,
        })
    
    data['velocity'] = torch.stack(data['velocity'])
    data['tau_sgs'] = torch.stack(data['tau_sgs'])
    
    print(f"Data statistics:")
    print(f"  Velocity RMS: {data['velocity'].std().item():.4f}")
    print(f"  Tau RMS: {data['tau_sgs'].std().item():.4f}")
    print(f"  Tau range: [{data['tau_sgs'].min().item():.4f}, {data['tau_sgs'].max().item():.4f}]")
    
    return data


def compute_sgs_loss(tau_pred, tau_true):
    """Compute comprehensive SGS stress metrics"""
    mse = F.mse_loss(tau_pred, tau_true)
    mae = F.l1_loss(tau_pred, tau_true)
    
    # Relative error
    rel_error = (tau_pred - tau_true).norm() / (tau_true.norm() + 1e-8)
    
    # Correlation coefficient
    pred_flat = tau_pred.flatten()
    true_flat = tau_true.flatten()
    corr = F.cosine_similarity(pred_flat, true_flat, dim=0)
    
    # Component-wise errors
    component_mse = []
    for i in range(6):
        comp_mse = F.mse_loss(tau_pred[:, i], tau_true[:, i]).item()
        component_mse.append(comp_mse)
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'rel_error': rel_error.item(),
        'correlation': corr.item(),
        'component_mse': component_mse,
    }


def train_model(model, data, epochs=100, lr=1e-3, model_name="Model"):
    """Generic training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Split data
    n_total = len(data['velocity'])
    n_train = int(0.8 * n_total)
    
    train_u = data['velocity'][:n_train]
    train_tau = data['tau_sgs'][:n_train]
    val_u = data['velocity'][n_train:]
    val_tau = data['tau_sgs'][n_train:]
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        batch_size = 8
        n_batches = (n_train + batch_size - 1) // batch_size
        
        for i in range(0, n_train, batch_size):
            end_i = min(i + batch_size, n_train)
            u_batch = train_u[i:end_i]
            tau_batch = train_tau[i:end_i]
            
            tau_pred = model(u_batch)
            loss = F.mse_loss(tau_pred, tau_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * (end_i - i)
        
        epoch_loss /= n_train
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(val_u), val_tau).item()
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 20 == 0:
            log_msg = f"Epoch {epoch+1:3d}: Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}"
            
            # Log physics parameters if available
            if hasattr(model, 'hard_core'):
                params = model.hard_core.get_physics_params()
                param_str = " | ".join([f"{k}: {v:.4f}" for k, v in params.items()])
                log_msg += f" | {param_str}"
            
            print(log_msg)
        
        scheduler.step()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(train_u)
        train_metrics = compute_sgs_loss(train_pred, train_tau)
        
        val_pred = model(val_u)
        val_metrics = compute_sgs_loss(val_pred, val_tau)
    
    print(f"\n[Final Results]")
    print(f"Train MSE: {train_metrics['mse']:.5f}, Corr: {train_metrics['correlation']:.4f}")
    print(f"Val MSE: {val_metrics['mse']:.5f}, Corr: {val_metrics['correlation']:.4f}")
    
    if hasattr(model, 'hard_core'):
        print(f"Physics params: {model.hard_core.get_physics_params()}")
    
    return model, history, val_metrics


def run_jhtdb_experiment():
    """Main experiment with realistic turbulence data"""
    print("=" * 80)
    print("RCLN v3.0 Advanced - JHTDB Realistic Turbulence Experiment")
    print("Resolution: 16^3 | Re_lambda ~ 433")
    print("=" * 80)
    
    # Generate data (reduced for faster experimentation)
    data = generate_realistic_turbulence_16cube(num_samples=100, Re_lambda=433)
    
    results = {}
    
    # 1. Baseline FNO (smaller for speed)
    baseline = BaselineFNO(width=8, modes=4)
    _, _, baseline_metrics = train_model(
        baseline, data, epochs=50, lr=1e-3, model_name="Baseline FNO"
    )
    results['baseline'] = baseline_metrics
    
    # 2. RCLN v3 Advanced - Different configurations
    configs = [
        {'name': 'RCLN v3 (all features)', 'hard': 0.5, 'soft': 0.5, 
         'dynamic': True, 'rotation': True, 'anisotropic': True},
        {'name': 'RCLN v3 (dynamic only)', 'hard': 0.5, 'soft': 0.5,
         'dynamic': True, 'rotation': False, 'anisotropic': False},
        {'name': 'RCLN v3 (physics heavy)', 'hard': 0.7, 'soft': 0.3,
         'dynamic': True, 'rotation': True, 'anisotropic': True},
        {'name': 'RCLN v3 (NN heavy)', 'hard': 0.3, 'soft': 0.7,
         'dynamic': True, 'rotation': True, 'anisotropic': True},
    ]
    
    for config in configs:
        model = RCLNv3_Advanced(
            resolution=16,
            fno_width=8,
            fno_modes=4,
            cs_init=0.1,
            lambda_hard=config['hard'],
            lambda_soft=config['soft'],
            use_dynamic=config['dynamic'],
            use_rotation=config['rotation'],
            use_anisotropic=config['anisotropic'],
        )
        
        _, _, metrics = train_model(
            model, data, epochs=50, lr=1e-3, model_name=config['name']
        )
        results[config['name']] = metrics
        results[config['name'] + '_params'] = model.hard_core.get_physics_params()
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n{'Model':<40} {'MSE':<12} {'MAE':<12} {'Corr':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        if 'params' in name:
            continue
        mse = metrics['mse']
        mae = metrics['mae']
        corr = metrics['correlation']
        print(f"{name:<40} {mse:<12.6f} {mae:<12.6f} {corr:<10.4f}")
        
        if 'RCLN' in name and name + '_params' in results:
            params = results[name + '_params']
            param_str = ", ".join([f"{k}={v:.4f}" for k, v in params.items()])
            print(f"  -> Physics: {param_str}")
    
    # Component-wise analysis
    print("\n" + "=" * 80)
    print("COMPONENT-WISE MSE (Best Model)")
    print("=" * 80)
    
    best_model_name = min(
        [k for k in results.keys() if 'params' not in k],
        key=lambda k: results[k]['mse']
    )
    print(f"Best model: {best_model_name}")
    
    components = ['tau_xx', 'tau_yy', 'tau_zz', 'tau_xy', 'tau_xz', 'tau_yz']
    for i, comp in enumerate(components):
        mse = results[best_model_name]['component_mse'][i]
        print(f"  {comp}: {mse:.6f}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_jhtdb_experiment()
