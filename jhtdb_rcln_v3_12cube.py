# -*- coding: utf-8 -*-
"""
JHTDB Experiment - RCLN v3.0
Resolution: 12³ (ultra-lightweight for fast iteration)

Features:
- RCLN v3 with Hard Core 3.0 + GENERIC coupling
- Learnable viscosity (nu)
- Comparison with baseline FNO (without Hard Core)

Architecture:
    RCLN v3: tau = lambda_rev * tau_hard + lambda_irr * tau_soft
    where tau_hard comes from learnable NS energy gradient
    and tau_soft comes from FNO neural network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from axiom_os.layers.rcln_v3 import RCLNv3_Turbulence
from axiom_os.layers.fno3d import FNO3d


class BaselineFNO(nn.Module):
    """Baseline: Pure FNO without physics"""
    def __init__(self, width=8, modes=4):
        super().__init__()
        self.fno = FNO3d(
            in_channels=3, out_channels=6,
            width=width, modes1=modes, modes2=modes, modes3=modes,
            n_layers=2,
        )
    
    def forward(self, u):
        return self.fno(u)


def generate_jhtdb_sample_12cube(num_samples=100):
    """
    Generate synthetic JHTDB-like turbulence at 12³ resolution
    
    Creates:
    - Velocity fields with Kolmogorov -5/3 spectrum
    - SGS stresses from filtered DNS
    """
    print("Generating 12^3 JHTDB-like data...")
    
    # Physical parameters
    Re_lambda = 433  # Taylor Reynolds number
    nu_dns = 1.0 / Re_lambda
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = {'velocity': [], 'tau_sgs': [], 'metadata': []}
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Random energy spectrum
        k_peak = np.random.uniform(2, 4)
        
        # Generate velocity field
        u = torch.zeros(3, 12, 12, 12, device=device)
        
        # Add Fourier modes
        for kx in range(-3, 4):
            for ky in range(-3, 4):
                for kz in range(-3, 4):
                    if kx == ky == kz == 0:
                        continue
                    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
                    
                    # Kolmogorov spectrum with peak
                    amplitude = (k_mag / k_peak) * np.exp(-k_mag / (2*k_peak))
                    amplitude *= (1 + 0.3 * torch.randn(1, device=device))
                    
                    # Random phase
                    phase_x = torch.rand(1, device=device) * 2 * np.pi
                    phase_y = torch.rand(1, device=device) * 2 * np.pi
                    phase_z = torch.rand(1, device=device) * 2 * np.pi
                    
                    # Add to field
                    for ix in range(12):
                        x = 2 * np.pi * ix / 12
                        u[0, ix] += (amplitude * torch.cos(kx*x + phase_x)).item()
                        u[1, ix] += (amplitude * torch.cos(ky*x + phase_y)).item() if ky != 0 else 0
                        u[2, ix] += (amplitude * torch.cos(kz*x + phase_z)).item() if kz != 0 else 0
        
        # Normalize
        u = u / (u.std() + 1e-8)
        
        # Compute SGS stress (ground truth)
        # tau_ij = <u_i u_j> - <u_i><u_j> (simplified)
        u_filtered = F.avg_pool3d(u.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
        
        tau = torch.zeros(6, 12, 12, 12, device=device)
        tau[0] = u[0]**2 - u_filtered[0]**2  # xx
        tau[1] = u[1]**2 - u_filtered[1]**2  # yy
        tau[2] = u[2]**2 - u_filtered[2]**2  # zz
        tau[3] = u[0]*u[1] - u_filtered[0]*u_filtered[1]  # xy
        tau[4] = u[0]*u[2] - u_filtered[0]*u_filtered[2]  # xz
        tau[5] = u[1]*u[2] - u_filtered[1]*u_filtered[2]  # yz
        
        data['velocity'].append(u)
        data['tau_sgs'].append(tau)
        data['metadata'].append({
            'k_peak': k_peak,
            'u_rms': u.std().item(),
            'tau_rms': tau.std().item(),
        })
    
    data['velocity'] = torch.stack(data['velocity'])
    data['tau_sgs'] = torch.stack(data['tau_sgs'])
    
    return data


def compute_sgs_loss(tau_pred, tau_true):
    """Compute SGS stress prediction loss with multiple metrics"""
    mse = F.mse_loss(tau_pred, tau_true)
    mae = F.l1_loss(tau_pred, tau_true)
    
    # Relative error
    rel_error = (tau_pred - tau_true).norm() / (tau_true.norm() + 1e-8)
    
    # Correlation coefficient
    pred_flat = tau_pred.flatten()
    true_flat = tau_true.flatten()
    corr = F.cosine_similarity(pred_flat, true_flat, dim=0)
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'rel_error': rel_error.item(),
        'correlation': corr.item(),
    }


def train_rcln_v3(
    data,
    epochs=50,
    lr=1e-3,
    lambda_hard=0.5,
    lambda_soft=0.5,
    nu_init=0.001,
):
    """Train RCLN v3.0"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("Training RCLN v3.0")
    print(f"Device: {device}")
    print(f"Lambda hard: {lambda_hard}, Lambda soft: {lambda_soft}")
    print(f"Initial nu: {nu_init}")
    print(f"{'='*70}\n")
    
    # Model
    model = RCLNv3_Turbulence(
        resolution=12,
        fno_width=8,
        fno_modes=4,
        nu_init=nu_init,
        lambda_hard=lambda_hard,
        lambda_soft=lambda_soft,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Split data
    n_total = len(data['velocity'])
    n_train = int(0.8 * n_total)
    
    train_u = data['velocity'][:n_train]
    train_tau = data['tau_sgs'][:n_train]
    val_u = data['velocity'][n_train:]
    val_tau = data['tau_sgs'][n_train:]
    
    history = {'train_loss': [], 'val_loss': [], 'nu_history': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        # Mini-batch training
        batch_size = 4
        n_batches = (n_train + batch_size - 1) // batch_size
        
        for i in range(0, n_train, batch_size):
            end_i = min(i + batch_size, n_train)
            u_batch = train_u[i:end_i].clone()
            tau_batch = train_tau[i:end_i].clone()
            
            # Forward
            tau_pred, phys = model(u_batch, return_physics=True)
            
            # Loss
            loss = F.mse_loss(tau_pred, tau_batch)
            
            # Optional: add physics regularization
            # Can add constraints here if needed
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end_i - i)
        
        epoch_loss /= n_train
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_u, return_physics=True)
            val_loss = F.mse_loss(val_pred, val_tau).item()
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['nu_history'].append(model.hard_core.nu.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss: {epoch_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"nu: {model.hard_core.nu.item():.6f}")
        
        scheduler.step()
    
    # Final metrics
    model.eval()
    with torch.no_grad():
        train_pred, _ = model(train_u, return_physics=True)
        train_metrics = compute_sgs_loss(train_pred, train_tau)
        
        val_pred, _ = model(val_u, return_physics=True)
        val_metrics = compute_sgs_loss(val_pred, val_tau)
    
    print(f"\n[Final Results]")
    print(f"Train MSE: {train_metrics['mse']:.4f}, Correlation: {train_metrics['correlation']:.4f}")
    print(f"Val MSE: {val_metrics['mse']:.4f}, Correlation: {val_metrics['correlation']:.4f}")
    print(f"Final nu: {model.hard_core.nu.item():.6f}")
    
    return model, history, val_metrics


def train_baseline(data, epochs=50, lr=1e-3):
    """Train baseline FNO"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("Training Baseline FNO")
    print(f"{'='*70}\n")
    
    model = BaselineFNO(width=8, modes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_total = len(data['velocity'])
    n_train = int(0.8 * n_total)
    
    train_u = data['velocity'][:n_train]
    train_tau = data['tau_sgs'][:n_train]
    val_u = data['velocity'][n_train:]
    val_tau = data['tau_sgs'][n_train:]
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        batch_size = 4
        for i in range(0, n_train, batch_size):
            end_i = min(i + batch_size, n_train)
            u_batch = train_u[i:end_i]
            tau_batch = train_tau[i:end_i]
            
            tau_pred = model(u_batch)
            loss = F.mse_loss(tau_pred, tau_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end_i - i)
        
        epoch_loss /= n_train
        
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(val_u), val_tau).item()
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss: {epoch_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
        
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        val_pred = model(val_u)
        val_metrics = compute_sgs_loss(val_pred, val_tau)
    
    print(f"\n[Final Results]")
    print(f"Val MSE: {val_metrics['mse']:.4f}, Correlation: {val_metrics['correlation']:.4f}")
    
    return model, history, val_metrics


def run_comparison():
    """Run RCLN v3 vs Baseline comparison"""
    print("=" * 70)
    print("RCLN v3.0 vs Baseline - JHTDB 12^3 Experiment")
    print("=" * 70)
    
    # Generate data
    data = generate_jhtdb_sample_12cube(num_samples=100)
    
    # Train models
    results = {}
    
    # Baseline
    _, _, baseline_metrics = train_baseline(data, epochs=50, lr=1e-3)
    results['baseline'] = baseline_metrics
    
    # RCLN v3 with different lambda settings
    for lh, ls in [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]:
        key = f"rcln_v3_hard{lh}_soft{ls}"
        model, history, metrics = train_rcln_v3(
            data, epochs=50, lr=1e-3,
            lambda_hard=lh, lambda_soft=ls,
            nu_init=0.001,
        )
        results[key] = metrics
        results[key + '_nu_final'] = model.hard_core.nu.item()
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<40} {'MSE':<10} {'MAE':<10} {'Corr':<10}")
    print("-" * 70)
    
    for name, metrics in results.items():
        if 'nu_final' in name:
            continue
        mse = metrics['mse']
        mae = metrics['mae']
        corr = metrics['correlation']
        print(f"{name:<40} {mse:<10.4f} {mae:<10.4f} {corr:<10.4f}")
        
        if 'rcln_v3' in name and name + '_nu_final' in results:
            print(f"    -> Final nu: {results[name + '_nu_final']:.6f}")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_comparison()
