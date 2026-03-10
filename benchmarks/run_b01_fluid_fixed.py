"""
B-01 Benchmark: Fluid SGS Modeling (FIXED VERSION with Normalization)
Compares Smagorinsky (Physics), Pure FNO (AI), and Axiom Hybrid (Physics+AI)

Fixes:
1. Data normalization (Z-score) - CRITICAL for neural networks
2. Higher Reynolds number (5000 vs 1000) for stronger turbulence
3. Longer warmup (2000 steps) for fully developed turbulence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from tqdm import tqdm

from axiom_os.layers.fluid_core import (
    SmagorinskyCore, FNO2d, HybridFluidRCLN,
    compute_correlation_coefficient, compute_mse
)


# ==============================================================================
# Data Normalizer - CRITICAL FIX
# ==============================================================================

class DataNormalizer:
    """
    Z-Score Normalizer for SGS modeling.
    
    Why: SGS stress values are typically O(1e-5), causing vanishing gradients.
    Solution: Normalize to N(0,1), train, then denormalize for evaluation.
    """
    
    def __init__(self):
        self.u_mean = None
        self.u_std = None
        self.tau_mean = None
        self.tau_std = None
        
    def fit(self, u: torch.Tensor, tau: torch.Tensor):
        """Compute normalization statistics from training data."""
        self.u_mean = u.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
        self.u_std = u.std(dim=(0, 2, 3), keepdim=True) + 1e-8
        
        self.tau_mean = tau.mean(dim=(0, 2, 3), keepdim=True)
        self.tau_std = tau.std(dim=(0, 2, 3), keepdim=True) + 1e-8
        
        print(f"  Normalization stats:")
        print(f"    u: mean={self.u_mean.squeeze()}, std={self.u_std.squeeze()}")
        print(f"    tau: mean={self.tau_mean.squeeze()}, std={self.tau_std.squeeze()}")
        
    def normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        """Normalize velocity input."""
        return (u - self.u_mean) / self.u_std
    
    def normalize_tau(self, tau: torch.Tensor) -> torch.Tensor:
        """Normalize stress target."""
        return (tau - self.tau_mean) / self.tau_std
    
    def denormalize_tau(self, tau_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize stress prediction back to physical space."""
        return tau_norm * self.tau_std + self.tau_mean


# ==============================================================================
# Kolmogorov Flow Simulation (Enhanced)
# ==============================================================================

class KolmogorovFlow:
    """2D Kolmogorov Flow with higher Re for stronger turbulence."""
    
    def __init__(self, N: int = 256, Re: float = 5000.0, dt: float = 0.001, 
                 force_amp: float = 0.5, force_wavenum: int = 4):
        self.N = N
        self.Re = Re
        self.nu = 1.0 / Re
        self.dt = dt
        self.force_amp = force_amp
        self.force_wavenum = force_wavenum
        
        self.x = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.y = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.k = np.fft.fftfreq(N, 1.0/N) * 2 * np.pi
        self.kx, self.ky = np.meshgrid(self.k, self.k)
        self.k_squared = self.kx**2 + self.ky**2
        self.k_squared[0, 0] = 1.0
        
        self.dealias_mask = (np.abs(self.kx) < (2.0/3.0) * N//2) & \
                           (np.abs(self.ky) < (2.0/3.0) * N//2)
    
    def forcing(self) -> Tuple[np.ndarray, np.ndarray]:
        fx = self.force_amp * np.sin(self.force_wavenum * self.Y)
        fy = np.zeros_like(fx)
        return fx, fy
    
    def vorticity_to_velocity(self, omega_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_hat = 1j * self.ky * omega_hat / self.k_squared
        v_hat = -1j * self.kx * omega_hat / self.k_squared
        return u_hat, v_hat
    
    def velocity_to_vorticity(self, u_hat: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
        return 1j * self.kx * v_hat - 1j * self.ky * u_hat
    
    def nonlinear_term(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u_phys = np.real(np.fft.ifft2(u))
        v_phys = np.real(np.fft.ifft2(v))
        omega = self.velocity_to_vorticity(u, v)
        omega_phys = np.real(np.fft.ifft2(omega))
        
        domega_dx = np.real(np.fft.ifft2(1j * self.kx * omega))
        domega_dy = np.real(np.fft.ifft2(1j * self.ky * omega))
        
        nonlinear = -(u_phys * domega_dx + v_phys * domega_dy)
        nonlinear_hat = np.fft.fft2(nonlinear)
        
        return nonlinear_hat * self.dealias_mask
    
    def step(self, omega_hat: np.ndarray) -> np.ndarray:
        def rhs(oh):
            u_hat, v_hat = self.vorticity_to_velocity(oh)
            nl = self.nonlinear_term(u_hat, v_hat)
            visc = -self.nu * self.k_squared * oh
            fx, fy = self.forcing()
            fx_hat = np.fft.fft2(fx)
            fy_hat = np.fft.fft2(fy)
            curl_f = 1j * self.kx * fy_hat - 1j * self.ky * fx_hat
            return nl + visc + curl_f
        
        k1 = rhs(omega_hat)
        k2 = rhs(omega_hat + 0.5 * self.dt * k1)
        k3 = rhs(omega_hat + 0.5 * self.dt * k2)
        k4 = rhs(omega_hat + self.dt * k3)
        
        return omega_hat + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def simulate(self, n_steps: int = 200, warmup_steps: int = 2000) -> np.ndarray:
        """Run simulation with longer warmup for fully developed turbulence."""
        print(f"Simulating 2D Kolmogorov Flow ({self.N}x{self.N}, Re={self.Re})...")
        
        np.random.seed(42)
        omega_hat = np.fft.fft2(np.random.randn(self.N, self.N) * 0.5)
        
        print(f"  Warmup: {warmup_steps} steps (developing turbulence)...")
        for _ in tqdm(range(warmup_steps), desc="Warmup"):
            omega_hat = self.step(omega_hat)
        
        print(f"  Collecting: {n_steps} snapshots...")
        snapshots = []
        for step in tqdm(range(n_steps), desc="Simulation"):
            omega_hat = self.step(omega_hat)
            u_hat, v_hat = self.vorticity_to_velocity(omega_hat)
            u = np.real(np.fft.ifft2(u_hat))
            v = np.real(np.fft.ifft2(v_hat))
            snapshots.append(np.stack([u, v], axis=0))
        
        return np.array(snapshots)


def filter_to_les(u_dns: np.ndarray, filter_ratio: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Filter DNS field to LES resolution and compute true SGS stress."""
    N_dns = u_dns.shape[-1]
    N_les = N_dns // filter_ratio
    
    u_hat = np.fft.fft2(u_dns[0])
    v_hat = np.fft.fft2(u_dns[1])
    
    k_cut = N_les // 2
    
    # Filter velocity
    u_hat_f = np.zeros((N_les, N_les), dtype=complex)
    v_hat_f = np.zeros((N_les, N_les), dtype=complex)
    
    u_hat_f[:k_cut, :k_cut] = u_hat[:k_cut, :k_cut] / (filter_ratio**2)
    u_hat_f[-k_cut:, :k_cut] = u_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    v_hat_f[:k_cut, :k_cut] = v_hat[:k_cut, :k_cut] / (filter_ratio**2)
    v_hat_f[-k_cut:, :k_cut] = v_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    
    u_les = np.real(np.fft.ifft2(u_hat_f))
    v_les = np.real(np.fft.ifft2(v_hat_f))
    u_les = np.stack([u_les, v_les], axis=0)
    
    # Compute SGS stress using Leonard decomposition
    uu_dns = u_dns[0] * u_dns[0]
    uv_dns = u_dns[0] * u_dns[1]
    vv_dns = u_dns[1] * u_dns[1]
    
    uu_hat = np.fft.fft2(uu_dns)
    uv_hat = np.fft.fft2(uv_dns)
    vv_hat = np.fft.fft2(vv_dns)
    
    uu_f = np.zeros((N_les, N_les), dtype=complex)
    uv_f = np.zeros((N_les, N_les), dtype=complex)
    vv_f = np.zeros((N_les, N_les), dtype=complex)
    
    uu_f[:k_cut, :k_cut] = uu_hat[:k_cut, :k_cut] / (filter_ratio**2)
    uu_f[-k_cut:, :k_cut] = uu_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    uv_f[:k_cut, :k_cut] = uv_hat[:k_cut, :k_cut] / (filter_ratio**2)
    uv_f[-k_cut:, :k_cut] = uv_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    vv_f[:k_cut, :k_cut] = vv_hat[:k_cut, :k_cut] / (filter_ratio**2)
    vv_f[-k_cut:, :k_cut] = vv_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    
    uu_les = np.real(np.fft.ifft2(uu_f))
    uv_les = np.real(np.fft.ifft2(uv_f))
    vv_les = np.real(np.fft.ifft2(vv_f))
    
    tau_xx = uu_les - u_les[0] * u_les[0]
    tau_xy = uv_les - u_les[0] * u_les[1]
    tau_yy = vv_les - u_les[1] * u_les[1]
    
    tau_true = np.stack([tau_xx, tau_xy, tau_yy], axis=0)
    
    return u_les, tau_true


def prepare_dataset(n_samples: int = 200, dns_resolution: int = 256, 
                   les_resolution: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate training/test dataset."""
    print("\n" + "="*70)
    print("STEP 1: Data Generation (Enhanced Turbulence)")
    print("="*70)
    
    flow = KolmogorovFlow(N=dns_resolution, Re=5000.0, dt=0.001, force_amp=0.5)
    snapshots = flow.simulate(n_steps=n_samples, warmup_steps=2000)
    
    print(f"\nFiltering DNS ({dns_resolution}) -> LES ({les_resolution})...")
    u_les_list = []
    tau_true_list = []
    
    for i in tqdm(range(n_samples), desc="Filtering"):
        u_les, tau_true = filter_to_les(snapshots[i], 
                                        filter_ratio=dns_resolution//les_resolution)
        u_les_list.append(u_les)
        tau_true_list.append(tau_true)
    
    u_les = torch.tensor(np.array(u_les_list), dtype=torch.float32)
    tau_true = torch.tensor(np.array(tau_true_list), dtype=torch.float32)
    
    print(f"\nDataset shape: u_les={u_les.shape}, tau_true={tau_true.shape}")
    print(f"  u_les range: [{u_les.min():.4f}, {u_les.max():.4f}], std={u_les.std():.6f}")
    print(f"  tau_true range: [{tau_true.min():.6f}, {tau_true.max():.6f}], std={tau_true.std():.8f}")
    
    return u_les, tau_true


# ==============================================================================
# Training with Normalization
# ==============================================================================

def train_model_normalized(model: nn.Module, normalizer: DataNormalizer,
                            u_train: torch.Tensor, tau_train: torch.Tensor,
                            u_val: torch.Tensor, tau_val: torch.Tensor, 
                            n_epochs: int = 100, lr: float = 1e-3, 
                            model_name: str = "Model") -> Dict:
    """Train a model with normalized data."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2_norm': [], 'val_r2_phys': [], 'best_r2': -1.0}
    
    print(f"\nTraining {model_name} (Normalized)...")
    best_model_state = None
    
    # Normalize training data
    u_train_norm = normalizer.normalize_u(u_train)
    tau_train_norm = normalizer.normalize_tau(tau_train)
    u_val_norm = normalizer.normalize_u(u_val)
    tau_val_norm = normalizer.normalize_tau(tau_val)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        
        batch_size = 16
        n_batches = len(u_train_norm) // batch_size
        
        for i in range(n_batches):
            u_batch = u_train_norm[i*batch_size:(i+1)*batch_size].to(device)
            tau_batch = tau_train_norm[i*batch_size:(i+1)*batch_size].to(device)
            
            optimizer.zero_grad()
            tau_pred_norm = model(u_batch)
            loss = F.mse_loss(tau_pred_norm, tau_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= n_batches
        
        # Validation - compute R2 in both normalized and physical space
        model.eval()
        with torch.no_grad():
            tau_val_pred_norm = model(u_val_norm.to(device))
            val_loss = F.mse_loss(tau_val_pred_norm, tau_val_norm.to(device)).item()
            val_r2_norm = compute_correlation_coefficient(tau_val_pred_norm, tau_val_norm.to(device))
            
            # Denormalize to physical space for true R2
            tau_val_pred_phys = normalizer.denormalize_tau(tau_val_pred_norm.cpu())
            val_r2_phys = compute_correlation_coefficient(tau_val_pred_phys, tau_val)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2_norm'].append(val_r2_norm)
        history['val_r2_phys'].append(val_r2_phys)
        
        if val_r2_phys > history['best_r2']:
            history['best_r2'] = val_r2_phys
            best_model_state = model.state_dict().copy()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Train Loss={train_loss:.6f}, "
                  f"Val R2(norm)={val_r2_norm:.4f}, R2(phys)={val_r2_phys:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_model_normalized(model: nn.Module, normalizer: DataNormalizer,
                               u_test: torch.Tensor, tau_test: torch.Tensor,
                               model_name: str) -> Dict:
    """Evaluate model - predictions are denormalized to physical space."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    u_test_norm = normalizer.normalize_u(u_test)
    
    with torch.no_grad():
        tau_pred_norm = model(u_test_norm.to(device))
        tau_pred = normalizer.denormalize_tau(tau_pred_norm.cpu())
        
        mse = compute_mse(tau_pred, tau_test)
        r2 = compute_correlation_coefficient(tau_pred, tau_test)
    
    print(f"\n{model_name} Test Results (Physical Space):")
    print(f"  MSE: {mse:.8f}")
    print(f"  R^2: {r2:.4f}")
    
    return {'mse': mse, 'r2': r2, 'predictions': tau_pred}


# ==============================================================================
# Visualization
# ==============================================================================

def plot_results(results: Dict, u_test: torch.Tensor, tau_test: torch.Tensor,
                 save_dir: str = "."):
    """Generate benchmark plots."""
    
    # Figure 1: R2 Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    r2_scores = [results[m]['r2'] for m in models]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    bars = ax.bar(models, r2_scores, color=colors, edgecolor='black', linewidth=2)
    
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Correlation Coefficient R^2', fontsize=14)
    ax.set_title('B-01 Benchmark: SGS Stress Prediction (FIXED)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'b01_benchmark_results.png'), dpi=150)
    print(f"\nSaved: {os.path.join(save_dir, 'b01_benchmark_results.png')}")
    plt.close()
    
    # Figure 2: Flow Comparison
    sample_idx = 0
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    titles = ['True SGS', 'Smagorinsky', 'Pure FNO', 'Axiom Hybrid']
    row_labels = ['tau_xx', 'tau_xy', 'tau_yy']
    
    tau_true = tau_test[sample_idx].cpu().numpy()
    tau_smag = results['Smagorinsky']['predictions'][sample_idx].numpy()
    tau_fno = results['Pure FNO']['predictions'][sample_idx].numpy()
    tau_axiom = results['Axiom Hybrid']['predictions'][sample_idx].numpy()
    
    data_matrix = [
        [tau_true[0], tau_smag[0], tau_fno[0], tau_axiom[0]],
        [tau_true[1], tau_smag[1], tau_fno[1], tau_axiom[1]],
        [tau_true[2], tau_smag[2], tau_fno[2], tau_axiom[2]],
    ]
    
    # Per-row color scaling for better visualization
    for i, row in enumerate(data_matrix):
        vmin = min([d.min() for d in row])
        vmax = max([d.max() for d in row])
        for j, data in enumerate(row):
            im = axes[i, j].imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[i, j].set_title(titles[j] if i == 0 else '', fontsize=12)
            axes[i, j].set_ylabel(row_labels[i] if j == 0 else '', fontsize=12)
            axes[i, j].axis('off')
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='SGS Stress')
    
    fig.suptitle('B-01: SGS Stress Field Comparison (FIXED with Normalization)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(save_dir, 'b01_flow_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(save_dir, 'b01_flow_comparison.png')}")
    plt.close()


# ==============================================================================
# Main Benchmark
# ==============================================================================

def run_benchmark():
    """Run complete B-01 benchmark with normalization fix."""
    
    print("="*70)
    print("B-01 BENCHMARK: FLUID SGS MODELING (FIXED VERSION)")
    print("="*70)
    print("\nFixes Applied:")
    print("  1. Data Normalization (Z-score) - CRITICAL FIX")
    print("  2. Higher Reynolds Number (5000 vs 1000)")
    print("  3. Longer Warmup (2000 steps)")
    print("\nModels:")
    print("  1. Smagorinsky (Physics Baseline)")
    print("  2. Pure FNO (AI Baseline)")
    print("  3. Axiom Hybrid (Physics + AI)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Generate data
    u_les, tau_true = prepare_dataset(n_samples=200, dns_resolution=256, les_resolution=64)
    
    # Split data
    n_train = 150
    n_val = 30
    
    u_train = u_les[:n_train]
    tau_train = tau_true[:n_train]
    u_val = u_les[n_train:n_train+n_val]
    tau_val = tau_true[n_train:n_train+n_val]
    u_test = u_les[n_train+n_val:]
    tau_test = tau_true[n_train+n_val:]
    
    print(f"\nData split: Train={len(u_train)}, Val={len(u_val)}, Test={len(u_test)}")
    
    # Initialize normalizer and fit on training data
    print("\n" + "="*70)
    print("STEP 2: Data Normalization")
    print("="*70)
    normalizer = DataNormalizer()
    normalizer.fit(u_train, tau_train)
    
    results = {}
    
    # ==============================================================================
    # Smagorinsky (No training - direct evaluation)
    # ==============================================================================
    print("\n" + "="*70)
    print("STEP 3a: Smagorinsky (Physics Baseline)")
    print("="*70)
    
    smag = SmagorinskyCore(cs=0.1)
    with torch.no_grad():
        tau_pred_smag = smag(u_test)
    
    mse_smag = compute_mse(tau_pred_smag, tau_test)
    r2_smag = compute_correlation_coefficient(tau_pred_smag, tau_test)
    
    print(f"\nSmagorinsky Results:")
    print(f"  MSE: {mse_smag:.8f}")
    print(f"  R^2: {r2_smag:.4f}")
    
    results['Smagorinsky'] = {'mse': mse_smag, 'r2': r2_smag, 'predictions': tau_pred_smag}
    
    # ==============================================================================
    # Pure FNO
    # ==============================================================================
    print("\n" + "="*70)
    print("STEP 3b: Pure FNO (AI Baseline)")
    print("="*70)
    
    fno = FNO2d(in_channels=2, out_channels=3, width=32, modes=(12, 12), n_layers=4)
    fno_history = train_model_normalized(fno, normalizer, u_train, tau_train, u_val, tau_val,
                                         n_epochs=100, lr=1e-3, model_name="Pure FNO")
    results['Pure FNO'] = evaluate_model_normalized(fno, normalizer, u_test, tau_test, "Pure FNO")
    
    # ==============================================================================
    # Axiom Hybrid
    # ==============================================================================
    print("\n" + "="*70)
    print("STEP 3c: Axiom Hybrid (Physics + AI)")
    print("="*70)
    
    hybrid = HybridFluidRCLN(cs=0.1, fno_width=32, fno_modes=(12, 12), fno_layers=4)
    hybrid_history = train_model_normalized(hybrid, normalizer, u_train, tau_train, u_val, tau_val,
                                            n_epochs=100, lr=1e-3, model_name="Axiom Hybrid")
    results['Axiom Hybrid'] = evaluate_model_normalized(hybrid, normalizer, u_test, tau_test, 
                                                        "Axiom Hybrid")
    
    # ==============================================================================
    # Results Summary
    # ==============================================================================
    print("\n" + "="*70)
    print("B-01 BENCHMARK FINAL RESULTS")
    print("="*70)
    
    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"  R^2 = {res['r2']:.4f}")
        print(f"  MSE = {res['mse']:.8f}")
    
    axiom_r2 = results['Axiom Hybrid']['r2']
    fno_r2 = results['Pure FNO']['r2']
    smag_r2 = results['Smagorinsky']['r2']
    
    print("\n" + "-"*70)
    if axiom_r2 > fno_r2 and axiom_r2 > smag_r2:
        improvement = ((axiom_r2 - max(fno_r2, smag_r2)) / max(fno_r2, smag_r2)) * 100
        print(f"*** VERDICT: Axiom-OS WINS! (R^2 = {axiom_r2:.4f}) ***")
        print(f"  Improvement over best baseline: +{improvement:.1f}%")
        print("  Physics + AI > Pure Physics or Pure AI")
    elif fno_r2 > axiom_r2 and fno_r2 > smag_r2:
        print(f"Result: Pure FNO best (R^2 = {fno_r2:.4f})")
        print(f"  Axiom={axiom_r2:.4f}, FNO={fno_r2:.4f}, Smag={smag_r2:.4f}")
    else:
        print(f"Result: Smagorinsky best (unexpected)")
        print(f"  Axiom={axiom_r2:.4f}, FNO={fno_r2:.4f}, Smag={smag_r2:.4f}")
    print("-"*70)
    
    # Generate plots
    print("\n" + "="*70)
    print("STEP 4: Generating Visualizations")
    print("="*70)
    
    save_dir = os.path.dirname(__file__)
    plot_results(results, u_test, tau_test, save_dir)
    
    print("\n" + "="*70)
    print("B-01 BENCHMARK COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_benchmark()
