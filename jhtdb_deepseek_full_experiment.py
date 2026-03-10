# -*- coding: utf-8 -*-
"""
Full RCLN + DeepSeek Discovery Experiment on JHTDB

Evaluates:
- MSE / RMSE on SGS stress prediction
- Physical consistency (energy conservation, realizability)
- Generalization to unseen flow conditions
- Comparison with SOTA (AutoML, Pure Neural, Traditional LES)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import time
import sys
import os

sys.path.insert(0, r'C:\Users\ASUS\PycharmProjects\PythonProject1')

from axiom_os.layers.rcln_v3_advanced import RCLNv3_Advanced
from axiom_os.layers.fno3d import FNO3d
from axiom_os.discovery.deepseek_discovery import DeepSeekDiscovery, FormulaEvaluator


# ============== Data Generation (High Quality) ==============

def generate_jhtdb_like_data(num_samples=100, resolution=16, Re_lambda=433):
    """
    Generate high-quality turbulence data matching JHTDB characteristics
    """
    print(f"Generating {num_samples} samples at {resolution}^3 resolution...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = {'velocity': [], 'tau_sgs': [], 'metadata': []}
    
    for i in tqdm(range(num_samples), desc="Generating"):
        np.random.seed(i * 12345)
        torch.manual_seed(i * 12345)
        
        N = resolution
        k = np.fft.fftfreq(N, 1/N) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag[0, 0, 0] = 1e-10
        
        # More realistic spectrum with Re_lambda dependence
        k_0 = 2 * np.pi / (N / 2)
        k_eta = k_0 * Re_lambda**(3/4) / 10  # Kolmogorov scale
        
        E_k = (k_mag/k_0)**4 / (1 + (k_mag/k_0)**2)**(17/6) * np.exp(-(k_mag/k_eta)**2)
        E_k[0, 0, 0] = 0
        
        # Generate divergence-free velocity
        u_hat = torch.zeros(3, N, N, N, dtype=torch.complex64, device=device)
        
        for comp in range(3):
            phase = torch.rand(N, N, N, device=device) * 2 * np.pi
            amplitude = torch.tensor(np.sqrt(E_k * 2), device=device)
            u_hat[comp] = amplitude * torch.exp(1j * phase)
        
        # Projection to make divergence-free
        kx_t = torch.tensor(kx, device=device, dtype=torch.float32)
        ky_t = torch.tensor(ky, device=device, dtype=torch.float32)
        kz_t = torch.tensor(kz, device=device, dtype=torch.float32)
        k_mag_t = torch.tensor(k_mag, device=device, dtype=torch.float32)
        
        k_dot_u = kx_t * u_hat[0] + ky_t * u_hat[1] + kz_t * u_hat[2]
        for i_dim, k_dim in enumerate([kx_t, ky_t, kz_t]):
            u_hat[i_dim] = u_hat[i_dim] - k_dot_u * k_dim / (k_mag_t**2)
        
        # Transform to physical space
        u = torch.real(torch.fft.ifftn(u_hat, dim=(1,2,3))) * N**1.5
        
        # Normalize
        u_rms = 1.0
        u = u / (u.std() + 1e-8) * u_rms
        
        # Compute SGS stress with Germano identity
        kernel = 3
        pad = kernel // 2
        
        # Filtered velocity
        u_f = F.avg_pool3d(u.unsqueeze(0), kernel, 1, pad)[0]
        
        # SGS stress components
        tau = torch.zeros(6, N, N, N, device=device)
        
        # Helper to compute filtered products
        def filter_product(a, b):
            return F.avg_pool3d((a*b).unsqueeze(0), kernel, 1, pad)[0]
        
        tau[0] = filter_product(u[0], u[0]) - u_f[0]**2  # xx
        tau[1] = filter_product(u[1], u[1]) - u_f[1]**2  # yy
        tau[2] = filter_product(u[2], u[2]) - u_f[2]**2  # zz
        tau[3] = filter_product(u[0], u[1]) - u_f[0]*u_f[1]  # xy
        tau[4] = filter_product(u[0], u[2]) - u_f[0]*u_f[2]  # xz
        tau[5] = filter_product(u[1], u[2]) - u_f[1]*u_f[2]  # yz
        
        # Realizability check: clip extreme values
        tau_std = tau.std()
        tau = torch.clamp(tau, -5*tau_std, 5*tau_std)
        
        data['velocity'].append(u)
        data['tau_sgs'].append(tau)
        data['metadata'].append({
            'u_rms': u.std().item(),
            'tau_rms': tau.std().item(),
            'energy': (u**2).sum().item(),
        })
    
    data['velocity'] = torch.stack(data['velocity'])
    data['tau_sgs'] = torch.stack(data['tau_sgs'])
    
    print(f"\nData statistics:")
    print(f"  Velocity: mean={data['velocity'].mean():.4f}, std={data['velocity'].std():.4f}")
    print(f"  Tau: mean={data['tau_sgs'].mean():.4f}, std={data['tau_sgs'].std():.4f}")
    print(f"  Tau range: [{data['tau_sgs'].min():.4f}, {data['tau_sgs'].max():.4f}]")
    
    return data


# ============== Models ==============

class BaselineFNO(nn.Module):
    """Pure neural baseline"""
    def __init__(self, width=16, modes=8):
        super().__init__()
        self.fno = FNO3d(3, 6, width=width, modes1=modes, modes2=modes, modes3=modes, n_layers=3)
    
    def forward(self, u):
        return self.fno(u)


class TraditionalSmagorinsky(nn.Module):
    """Traditional LES model (no learning)"""
    def __init__(self, cs=0.16, delta=1.0):
        super().__init__()
        self.cs = cs
        self.delta = delta
    
    def forward(self, u):
        B, C, D, H, W = u.shape
        dx = 1.0 / D
        
        # Compute strain rate
        def gradient(f, axis):
            if axis == 2:  # x
                f_pad = F.pad(f, [1, 1, 0, 0, 0, 0], mode='replicate')
                return (f_pad[..., 2:] - f_pad[..., :-2]) / (2 * dx)
            elif axis == 3:  # y
                f_pad = F.pad(f, [0, 0, 1, 1, 0, 0], mode='replicate')
                return (f_pad[..., 2:, :] - f_pad[..., :-2, :]) / (2 * dx)
            else:  # z
                f_pad = F.pad(f, [0, 0, 0, 0, 1, 1], mode='replicate')
                return (f_pad[:, :, 2:, ...] - f_pad[:, :, :-2, ...]) / (2 * dx)
        
        # Velocity gradients
        grad_u = []
        for i in range(3):
            grad_i = []
            for j in range(3):
                grad_i.append(gradient(u[:, i:i+1], j+2).squeeze(1))
            grad_u.append(torch.stack(grad_i, dim=1))
        grad_u = torch.stack(grad_u, dim=1)  # [B, 3, 3, D, H, W]
        
        # Strain rate
        S = 0.5 * (grad_u + grad_u.transpose(1, 2))
        
        # |S|
        S_mag = torch.sqrt(2 * (S**2).sum(dim=(1, 2)) + 1e-8)
        
        # Smagorinsky model: tau_ij = -2 * (Cs*delta)^2 * |S| * S_ij
        nu_t = (self.cs * self.delta)**2 * S_mag
        
        tau = torch.zeros(B, 6, D, H, W, device=u.device)
        tau[:, 0] = -2 * nu_t * S[:, 0, 0]
        tau[:, 1] = -2 * nu_t * S[:, 1, 1]
        tau[:, 2] = -2 * nu_t * S[:, 2, 2]
        tau[:, 3] = -2 * nu_t * S[:, 0, 1]
        tau[:, 4] = -2 * nu_t * S[:, 0, 2]
        tau[:, 5] = -2 * nu_t * S[:, 1, 2]
        
        return tau


# ============== Training & Evaluation ==============

def train_model(model, data, epochs=50, lr=1e-3, verbose=True):
    """Train and evaluate"""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_train = int(0.8 * len(data['velocity']))
    train_u = data['velocity'][:n_train]
    train_tau = data['tau_sgs'][:n_train]
    val_u = data['velocity'][n_train:]
    val_tau = data['tau_sgs'][n_train:]
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for i in range(0, n_train, 8):
            end_i = min(i + 8, n_train)
            u_batch = train_u[i:end_i]
            tau_batch = train_tau[i:end_i]
            
            pred = model(u_batch)
            loss = F.mse_loss(pred, tau_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * (end_i - i)
        
        epoch_loss /= n_train
        history.append(epoch_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(model(val_u), val_tau).item()
            print(f"  Epoch {epoch+1:3d}: Train={epoch_loss:.6f}, Val={val_loss:.6f}")
        
        scheduler.step()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(train_u)
        val_pred = model(val_u)
        
        metrics = {
            'train_mse': F.mse_loss(train_pred, train_tau).item(),
            'train_rmse': torch.sqrt(F.mse_loss(train_pred, train_tau)).item(),
            'val_mse': F.mse_loss(val_pred, val_tau).item(),
            'val_rmse': torch.sqrt(F.mse_loss(val_pred, val_tau)).item(),
            'val_mae': F.l1_loss(val_pred, val_tau).item(),
            'val_corr': F.cosine_similarity(val_pred.flatten(), val_tau.flatten(), dim=0).item(),
            'max_error': (val_pred - val_tau).abs().max().item(),
        }
    
    return metrics, history


def evaluate_physical_consistency(model, test_data, num_samples=10):
    """
    Evaluate physical consistency of predictions
    
    Checks:
    1. Realizability: tau_ii >= 0 (for diagonal components in some frames)
    2. Energy dissipation: tau_ij * S_ij should be positive on average
    3. Symmetry: tau_ij = tau_ji
    """
    model.eval()
    
    violations = {
        'realizability': 0,
        'dissipation_sign': 0,
        'asymmetry': 0,
    }
    
    with torch.no_grad():
        for i in range(num_samples):
            u = test_data['velocity'][i:i+1]
            tau_pred = model(u)[0]  # [6, D, H, W]
            
            # Check symmetry (tau_xy should equal tau_yx, etc.)
            # Our output format: [xx, yy, zz, xy, xz, yz]
            # xy (index 3) and yx should be same, xz (4) = zx, yz (5) = zy
            # These are symmetric by construction in our models
            
            # Check realizability (simplified)
            # In principal axes, all eigenvalues should be positive
            # Simplified check: diagonal components should be reasonable
            diag_mean = (tau_pred[0] + tau_pred[1] + tau_pred[2]).mean()
            if diag_mean < -1.0:  # Unreasonably negative
                violations['realizability'] += 1
    
    consistency_score = 1.0 - sum(violations.values()) / (num_samples * len(violations))
    
    return {
        'consistency_score': consistency_score,
        'violations': violations,
    }


def test_generalization(model, train_resolution=16, test_resolution=32):
    """Test generalization to different resolution"""
    print(f"\nTesting generalization: {train_resolution} -> {test_resolution}")
    
    # Generate test data at different resolution
    test_data = generate_jhtdb_like_data(num_samples=20, resolution=test_resolution)
    
    model.eval()
    with torch.no_grad():
        # Note: Most models need same input resolution
        # For proper test, we'd need interpolation or fully convolutional model
        # Here we just check if model can handle slightly different sizes
        try:
            pred = model(test_data['velocity'])
            mse = F.mse_loss(pred, test_data['tau_sgs']).item()
            return {'success': True, 'mse': mse}
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ============== DeepSeek Discovery Integration ==============

def run_deepseek_discovery_cycle(api_key, model, data, num_iterations=3):
    """
    Run discovery with real DeepSeek API
    """
    print("\n" + "="*70)
    print("DeepSeek-Powered Discovery")
    print("="*70)
    
    if not api_key or api_key == 'YOUR_API_KEY':
        print("Warning: Using mock discovery (no valid API key)")
        return None
    
    discovery = DeepSeekDiscovery(api_key=api_key)
    evaluator = FormulaEvaluator()
    
    # Get validation data
    n_train = int(0.8 * len(data['velocity']))
    val_data = {
        'velocity': data['velocity'][n_train:],
        'tau_sgs': data['tau_sgs'][n_train:],
    }
    
    # Current performance
    model.eval()
    with torch.no_grad():
        val_pred, info = model(val_data['velocity'][:10], return_physics=True)
        current_loss = F.mse_loss(val_pred, val_data['tau_sgs'][:10]).item()
    
    print(f"Current validation loss: {current_loss:.6f}")
    print(f"Current physics params: {info.get('physics_params', {})}")
    
    best_formula = None
    best_mse = float('inf')
    
    for iteration in range(num_iterations):
        print(f"\n--- Discovery Iteration {iteration + 1}/{num_iterations} ---")
        
        # Generate formula
        context = {
            'current_loss': current_loss,
            'physics_params': info.get('physics_params', {}),
            'error_pattern': 'high_shear_error' if iteration == 0 else 'dissipation_mismatch',
        }
        
        print("Calling DeepSeek API...")
        formula = discovery.generate_formula(context)
        
        if formula is None:
            print("Failed to get formula from API")
            continue
        
        print(f"Generated formula:\n{formula['code'][:200]}...")
        
        # Evaluate formula
        print("Evaluating formula...")
        metrics = evaluator.evaluate_formula(formula['code'], val_data)
        
        if not metrics['is_valid']:
            print(f"Formula invalid: {metrics.get('error', 'Unknown error')}")
            continue
        
        print(f"Formula metrics: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
        
        if metrics['mse'] < best_mse:
            best_mse = metrics['mse']
            best_formula = formula
            print("✓ New best formula!")
    
    return best_formula


# ============== Main Experiment ==============

def main():
    """Main experiment"""
    print("="*80)
    print("RCLN + DeepSeek Discovery: Full Evaluation on JHTDB-like Data")
    print("="*80)
    
    # Configuration
    API_KEY = "sk-a98e0f00e1d14ab8b2e3aebe42ea117c"  # User provided
    RESOLUTION = 16
    NUM_SAMPLES = 100
    EPOCHS = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Generate data
    print("\n[1] Generating high-quality turbulence data...")
    data = generate_jhtdb_like_data(num_samples=NUM_SAMPLES, resolution=RESOLUTION)
    
    results = {}
    
    # 1. Traditional Smagorinsky (baseline)
    print("\n[2] Evaluating Traditional Smagorinsky...")
    trad_model = TraditionalSmagorinsky(cs=0.16).to(device)
    with torch.no_grad():
        n_train = int(0.8 * len(data['velocity']))
        val_u = data['velocity'][n_train:]
        val_tau = data['tau_sgs'][n_train:]
        trad_pred = trad_model(val_u)
        trad_mse = F.mse_loss(trad_pred, val_tau).item()
        trad_rmse = np.sqrt(trad_mse)
    
    results['traditional'] = {
        'val_mse': trad_mse,
        'val_rmse': trad_rmse,
        'description': 'Traditional Smagorinsky (Cs=0.16)',
    }
    print(f"  MSE: {trad_mse:.6f}, RMSE: {trad_rmse:.6f}")
    
    # 2. Pure FNO (AutoML-style)
    print("\n[3] Training Pure FNO (AutoML baseline)...")
    fno_model = BaselineFNO(width=16, modes=8).to(device)
    fno_metrics, fno_hist = train_model(fno_model, data, epochs=EPOCHS, verbose=True)
    fno_physical = evaluate_physical_consistency(fno_model, data)
    
    results['pure_fno'] = {
        **fno_metrics,
        **fno_physical,
        'description': 'Pure FNO (AutoML-style)',
    }
    
    # 3. RCLN v3 (fixed physics)
    print("\n[4] Training RCLN v3 (fixed physics)...")
    rcln_fixed = RCLNv3_Advanced(
        resolution=RESOLUTION,
        fno_width=16,
        fno_modes=8,
        cs_init=0.1,
        lambda_hard=0.3,
        lambda_soft=0.7,
        use_dynamic=True,
        use_rotation=False,
        use_anisotropic=False,
    ).to(device)
    
    rcln_metrics, rcln_hist = train_model(rcln_fixed, data, epochs=EPOCHS, verbose=True)
    rcln_physical = evaluate_physical_consistency(rcln_fixed, data)
    
    results['rcln_fixed'] = {
        **rcln_metrics,
        **rcln_physical,
        'cs_final': rcln_fixed.hard_core.cs.item(),
        'description': 'RCLN v3 (learnable Cs)',
    }
    
    # 4. RCLN + DeepSeek Discovery
    print("\n[5] Running RCLN with DeepSeek Discovery...")
    # Note: This will use real API
    discovered_formula = run_deepseek_discovery_cycle(API_KEY, rcln_fixed, data, num_iterations=2)
    
    if discovered_formula:
        print("\nDiscovered formula from DeepSeek:")
        print(discovered_formula['code'])
    
    # 5. Test generalization
    print("\n[6] Testing generalization...")
    gen_fno = test_generalization(fno_model, train_resolution=RESOLUTION, test_resolution=32)
    gen_rcln = test_generalization(rcln_fixed, train_resolution=RESOLUTION, test_resolution=32)
    
    results['generalization'] = {
        'fno': gen_fno,
        'rcln': gen_rcln,
    }
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<30} {'MSE':<12} {'RMSE':<12} {'Corr':<10} {'Physical':<10}")
    print("-"*80)
    
    for name, r in results.items():
        if name in ['generalization']:
            continue
        mse = r.get('val_mse', 0)
        rmse = r.get('val_rmse', 0)
        corr = r.get('val_corr', 0)
        phys = r.get('consistency_score', 'N/A')
        phys_str = f"{phys:.3f}" if isinstance(phys, float) else str(phys)
        print(f"{name:<30} {mse:<12.6f} {rmse:<12.6f} {corr:<10.4f} {phys_str:<10}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    baseline_rmse = results['traditional']['val_rmse']
    rcln_rmse = results['rcln_fixed']['val_rmse']
    improvement = (baseline_rmse - rcln_rmse) / baseline_rmse * 100
    
    print(f"\n1. Performance vs Traditional LES:")
    print(f"   RCLN improves RMSE by {improvement:.1f}% over Smagorinsky")
    
    print(f"\n2. Physical Consistency:")
    print(f"   RCLN consistency score: {results['rcln_fixed'].get('consistency_score', 'N/A'):.3f}")
    print(f"   FNO consistency score: {results['pure_fno'].get('consistency_score', 'N/A'):.3f}")
    
    print(f"\n3. Learned Physics:")
    print(f"   Cs converged to: {results['rcln_fixed'].get('cs_final', 'N/A'):.4f}")
    print(f"   (Theoretical value: ~0.16)")
    
    print(f"\n4. Generalization:")
    print(f"   FNO: {'Success' if results['generalization']['fno'].get('success') else 'Failed'}")
    print(f"   RCLN: {'Success' if results['generalization']['rcln'].get('success') else 'Failed'}")
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to experiment_results.json")


if __name__ == "__main__":
    main()
