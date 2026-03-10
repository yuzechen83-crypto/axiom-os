"""
B-01 Benchmark: Fluid SGS Modeling
Compares Smagorinsky (Physics), Pure FNO (AI), and Axiom Hybrid (Physics+AI)
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
import time

from axiom_os.layers.fluid_core import (
    SmagorinskyCore, FNO2d, HybridFluidRCLN,
    compute_correlation_coefficient, compute_mse
)


# ==============================================================================
# Step 1: Data Generation - 2D Kolmogorov Turbulence
# ==============================================================================

class KolmogorovFlow:
    """
    2D Kolmogorov Flow Simulation.
    
    Forced Navier-Stokes with sinusoidal forcing:
    du/dt + (u·grad)u = -gradp + nugrad^2u + f
    grad·u = 0
    
    Where forcing f = F sin(ky) x̂ (drives large-scale rolls)
    """
    
    def __init__(self, N: int = 256, Re: float = 5000.0, dt: float = 0.001, 
                 force_amp: float = 0.5, force_wavenum: int = 4):
        self.N = N
        self.Re = Re
        self.nu = 1.0 / Re
        self.dt = dt
        self.force_amp = force_amp
        self.force_wavenum = force_wavenum
        
        # Grid
        self.x = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.y = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Wavenumbers for spectral method
        self.k = np.fft.fftfreq(N, 1.0/N) * 2 * np.pi
        self.kx, self.ky = np.meshgrid(self.k, self.k)
        self.k_squared = self.kx**2 + self.ky**2
        self.k_squared[0, 0] = 1.0  # Avoid division by zero
        
        # Dealiasing mask (2/3 rule)
        self.dealias_mask = (np.abs(self.kx) < (2.0/3.0) * N//2) & \
                           (np.abs(self.ky) < (2.0/3.0) * N//2)
    
    def forcing(self) -> Tuple[np.ndarray, np.ndarray]:
        """Body forcing to drive turbulence."""
        fx = self.force_amp * np.sin(self.force_wavenum * self.Y)
        fy = np.zeros_like(fx)
        return fx, fy
    
    def vorticity_to_velocity(self, omega_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert vorticity to velocity in Fourier space."""
        u_hat = 1j * self.ky * omega_hat / self.k_squared
        v_hat = -1j * self.kx * omega_hat / self.k_squared
        return u_hat, v_hat
    
    def velocity_to_vorticity(self, u_hat: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
        """Convert velocity to vorticity in Fourier space."""
        return 1j * self.kx * v_hat - 1j * self.ky * u_hat
    
    def nonlinear_term(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute nonlinear term (vorticity form)."""
        # Transform to physical space
        u_phys = np.real(np.fft.ifft2(u))
        v_phys = np.real(np.fft.ifft2(v))
        
        # Compute vorticity
        omega = self.velocity_to_vorticity(u, v)
        omega_phys = np.real(np.fft.ifft2(omega))
        
        # Advection term
        domega_dx = np.real(np.fft.ifft2(1j * self.kx * omega))
        domega_dy = np.real(np.fft.ifft2(1j * self.ky * omega))
        
        nonlinear = -(u_phys * domega_dx + v_phys * domega_dy)
        nonlinear_hat = np.fft.fft2(nonlinear)
        
        return nonlinear_hat * self.dealias_mask
    
    def step(self, omega_hat: np.ndarray) -> np.ndarray:
        """Take one timestep using RK4."""
        def rhs(oh):
            u_hat, v_hat = self.vorticity_to_velocity(oh)
            
            # Nonlinear term
            nl = self.nonlinear_term(u_hat, v_hat)
            
            # Viscous term
            visc = -self.nu * self.k_squared * oh
            
            # Forcing
            fx, fy = self.forcing()
            fx_hat = np.fft.fft2(fx)
            fy_hat = np.fft.fft2(fy)
            curl_f = 1j * self.kx * fy_hat - 1j * self.ky * fx_hat
            
            return nl + visc + curl_f
        
        # RK4
        k1 = rhs(omega_hat)
        k2 = rhs(omega_hat + 0.5 * self.dt * k1)
        k3 = rhs(omega_hat + 0.5 * self.dt * k2)
        k4 = rhs(omega_hat + self.dt * k3)
        
        omega_hat_new = omega_hat + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return omega_hat_new
    
    def simulate(self, n_steps: int = 1000, warmup_steps: int = 500) -> np.ndarray:
        """
        Run simulation and collect snapshots.
        
        Returns:
            snapshots: Array of velocity fields (n_snapshots, 2, N, N)
        """
        print(f"Simulating 2D Kolmogorov Flow ({self.N}x{self.N})...")
        
        # Initial condition (random)
        np.random.seed(42)
        omega_hat = np.fft.fft2(np.random.randn(self.N, self.N) * 0.1)
        
        # Warmup to reach statistical steady state
        print(f"  Warmup: {warmup_steps} steps...")
        for _ in tqdm(range(warmup_steps), desc="Warmup"):
            omega_hat = self.step(omega_hat)
        
        # Collect snapshots
        print(f"  Collecting: {n_steps} snapshots...")
        snapshots = []
        for step in tqdm(range(n_steps), desc="Simulation"):
            omega_hat = self.step(omega_hat)
            
            if step % 1 == 0:  # Store every step
                u_hat, v_hat = self.vorticity_to_velocity(omega_hat)
                u = np.real(np.fft.ifft2(u_hat))
                v = np.real(np.fft.ifft2(v_hat))
                snapshots.append(np.stack([u, v], axis=0))
        
        return np.array(snapshots)


def filter_to_les(u_dns: np.ndarray, filter_ratio: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter DNS field to LES resolution and compute SGS stress.
    
    Args:
        u_dns: DNS velocity field (2, N_dns, N_dns)
        filter_ratio: Ratio of DNS to LES resolution (e.g., 4 for 256->64)
    
    Returns:
        u_les: Filtered velocity (2, N_les, N_les)
        tau_true: True SGS stress (3, N_les, N_les) [tau_xx, tau_xy, tau_yy]
    """
    N_dns = u_dns.shape[-1]
    N_les = N_dns // filter_ratio
    
    # Transform to Fourier space
    u_hat = np.fft.fft2(u_dns[0])
    v_hat = np.fft.fft2(u_dns[1])
    
    # Low-pass filter (sharp spectral cutoff)
    u_hat_filtered = np.zeros((N_les, N_les), dtype=complex)
    v_hat_filtered = np.zeros((N_les, N_les), dtype=complex)
    
    # Keep low wavenumbers
    k_cut = N_les // 2
    
    u_hat_filtered[:k_cut, :k_cut] = u_hat[:k_cut, :k_cut] / (filter_ratio**2)
    u_hat_filtered[-k_cut:, :k_cut] = u_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    v_hat_filtered[:k_cut, :k_cut] = v_hat[:k_cut, :k_cut] / (filter_ratio**2)
    v_hat_filtered[-k_cut:, :k_cut] = v_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    
    # Transform back
    u_les = np.real(np.fft.ifft2(u_hat_filtered))
    v_les = np.real(np.fft.ifft2(v_hat_filtered))
    u_les = np.stack([u_les, v_les], axis=0)
    
    # Compute true SGS stress using Leonard decomposition
    # tau_ij = u_i*u_j_bar - u_i_bar * u_j_bar
    uu_dns = u_dns[0] * u_dns[0]
    uv_dns = u_dns[0] * u_dns[1]
    vv_dns = u_dns[1] * u_dns[1]
    
    uu_hat = np.fft.fft2(uu_dns)
    uv_hat = np.fft.fft2(uv_dns)
    vv_hat = np.fft.fft2(vv_dns)
    
    uu_filtered = np.zeros((N_les, N_les), dtype=complex)
    uv_filtered = np.zeros((N_les, N_les), dtype=complex)
    vv_filtered = np.zeros((N_les, N_les), dtype=complex)
    
    uu_filtered[:k_cut, :k_cut] = uu_hat[:k_cut, :k_cut] / (filter_ratio**2)
    uu_filtered[-k_cut:, :k_cut] = uu_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    uv_filtered[:k_cut, :k_cut] = uv_hat[:k_cut, :k_cut] / (filter_ratio**2)
    uv_filtered[-k_cut:, :k_cut] = uv_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    vv_filtered[:k_cut, :k_cut] = vv_hat[:k_cut, :k_cut] / (filter_ratio**2)
    vv_filtered[-k_cut:, :k_cut] = vv_hat[-k_cut:, :k_cut] / (filter_ratio**2)
    
    uu_les = np.real(np.fft.ifft2(uu_filtered))
    uv_les = np.real(np.fft.ifft2(uv_filtered))
    vv_les = np.real(np.fft.ifft2(vv_filtered))
    
    tau_xx = uu_les - u_les[0] * u_les[0]
    tau_xy = uv_les - u_les[0] * u_les[1]
    tau_yy = vv_les - u_les[1] * u_les[1]
    
    tau_true = np.stack([tau_xx, tau_xy, tau_yy], axis=0)
    
    return u_les, tau_true


def prepare_dataset(n_samples: int = 200, dns_resolution: int = 256, 
                   les_resolution: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training/test dataset.
    
    Returns:
        u_les: LES velocity fields (n_samples, 2, H, W)
        tau_true: True SGS stresses (n_samples, 3, H, W)
    """
    print("\n" + "="*70)
    print("STEP 1: Data Generation")
    print("="*70)
    
    # Run DNS
    flow = KolmogorovFlow(N=dns_resolution, Re=1000.0, dt=0.001)
    snapshots = flow.simulate(n_steps=n_samples, warmup_steps=500)
    
    # Filter to LES
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
    
    print(f"\nDataset shape:")
    print(f"  u_les: {u_les.shape}")
    print(f"  tau_true: {tau_true.shape}")
    
    return u_les, tau_true


# ==============================================================================
# Step 2: Training
# ==============================================================================

def train_model(model: nn.Module, u_train: torch.Tensor, tau_train: torch.Tensor,
                u_val: torch.Tensor, tau_val: torch.Tensor, 
                n_epochs: int = 100, lr: float = 1e-3, 
                model_name: str = "Model") -> Dict:
    """Train a model and track metrics."""
    
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'best_r2': -1.0
    }
    
    print(f"\nTraining {model_name}...")
    best_model_state = None
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Simple batch processing
        batch_size = 16
        n_batches = len(u_train) // batch_size
        
        for i in range(n_batches):
            u_batch = u_train[i*batch_size:(i+1)*batch_size].to(device)
            tau_batch = tau_train[i*batch_size:(i+1)*batch_size].to(device)
            
            optimizer.zero_grad()
            tau_pred = model(u_batch)
            loss = F.mse_loss(tau_pred, tau_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            tau_val_pred = model(u_val.to(device))
            val_loss = F.mse_loss(tau_val_pred, tau_val.to(device)).item()
            val_r2 = compute_correlation_coefficient(tau_val_pred, tau_val.to(device))
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        
        if val_r2 > history['best_r2']:
            history['best_r2'] = val_r2
            best_model_state = model.state_dict().copy()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Train Loss={train_loss:.6f}, "
                  f"Val Loss={val_loss:.6f}, R^2={val_r2:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


# ==============================================================================
# Step 3: Evaluation & Visualization
# ==============================================================================

def evaluate_model(model: nn.Module, u_test: torch.Tensor, tau_test: torch.Tensor,
                  model_name: str, device: torch.device = None) -> Dict:
    """Evaluate model on test set."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        tau_pred = model(u_test.to(device))
        
        mse = compute_mse(tau_pred, tau_test.to(device))
        r2 = compute_correlation_coefficient(tau_pred, tau_test.to(device))
    
    print(f"\n{model_name} Test Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R^2:  {r2:.4f}")
    
    return {
        'mse': mse,
        'r2': r2,
        'predictions': tau_pred.cpu()
    }


def plot_results(results: Dict, u_test: torch.Tensor, tau_test: torch.Tensor,
                 save_dir: str = "."):
    """Generate benchmark plots."""
    
    # Figure 1: R^2 Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    r2_scores = [results[m]['r2'] for m in models]
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    bars = ax.bar(models, r2_scores, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Correlation Coefficient R^2', fontsize=14)
    ax.set_title('B-01 Benchmark: SGS Stress Prediction (Higher is Better)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Guess')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'b01_benchmark_results.png'), dpi=150)
    print(f"\n✓ Saved: {os.path.join(save_dir, 'b01_benchmark_results.png')}")
    plt.close()
    
    # Figure 2: Flow Field Comparison
    sample_idx = 0
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    titles = ['True SGS', 'Smagorinsky', 'Pure FNO', 'Axiom Hybrid']
    row_labels = ['tau_xx', 'tau_xy', 'tau_yy']
    
    # Get predictions
    tau_true = tau_test[sample_idx].cpu().numpy()
    tau_smag = results['Smagorinsky']['predictions'][sample_idx].numpy()
    tau_fno = results['Pure FNO']['predictions'][sample_idx].numpy()
    tau_axiom = results['Axiom Hybrid']['predictions'][sample_idx].numpy()
    
    data_matrix = [
        [tau_true[0], tau_smag[0], tau_fno[0], tau_axiom[0]],  # tau_xx
        [tau_true[1], tau_smag[1], tau_fno[1], tau_axiom[1]],  # tau_xy
        [tau_true[2], tau_smag[2], tau_fno[2], tau_axiom[2]],  # tau_yy
    ]
    
    # Find common color scale
    vmin = min([d.min() for row in data_matrix for d in row])
    vmax = max([d.max() for row in data_matrix for d in row])
    
    for i, row in enumerate(data_matrix):
        for j, data in enumerate(row):
            im = axes[i, j].imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[i, j].set_title(titles[j] if i == 0 else '', fontsize=12)
            axes[i, j].set_ylabel(row_labels[i] if j == 0 else '', fontsize=12)
            axes[i, j].axis('off')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='SGS Stress')
    
    fig.suptitle('B-01: SGS Stress Field Comparison (Sample Test Case)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(save_dir, 'b01_flow_comparison.png'), dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'b01_flow_comparison.png')}")
    plt.close()


def run_benchmark():
    """Run complete B-01 benchmark."""
    
    print("="*70)
    print("B-01 BENCHMARK: FLUID SGS MODELING")
    print("="*70)
    print("\nModels:")
    print("  1. Smagorinsky (Physics Baseline)")
    print("  2. Pure FNO (AI Baseline)")
    print("  3. Axiom Hybrid (Physics + AI)")
    print("\nTask: Predict subgrid-scale stress from resolved velocity")
    print("Data: 2D Kolmogorov Turbulence (256x256 DNS -> 64x64 LES)")
    print("="*70)
    
    # Device
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
    
    print(f"\nData split:")
    print(f"  Train: {len(u_train)} samples")
    print(f"  Val:   {len(u_val)} samples")
    print(f"  Test:  {len(u_test)} samples")
    
    results = {}
    
    # ==============================================================================
    # Baseline A: Smagorinsky (No training needed)
    # ==============================================================================
    print("\n" + "="*70)
    print("STEP 2a: Smagorinsky (Physics Baseline)")
    print("="*70)
    
    smag = SmagorinskyCore(cs=0.1).to(device)
    results['Smagorinsky'] = evaluate_model(smag, u_test, tau_test, "Smagorinsky")
    
    # ==============================================================================
    # Baseline B: Pure FNO
    # ==============================================================================
    print("\n" + "="*70)
    print("STEP 2b: Pure FNO (AI Baseline)")
    print("="*70)
    
    fno = FNO2d(in_channels=2, out_channels=3, width=32, modes=(12, 12), n_layers=4).to(device)
    fno_history = train_model(fno, u_train, tau_train, u_val, tau_val,
                              n_epochs=100, lr=1e-3, model_name="Pure FNO")
    results['Pure FNO'] = evaluate_model(fno, u_test, tau_test, "Pure FNO")
    
    # ==============================================================================
    # Axiom-OS: Hybrid
    # ==============================================================================
    print("\n" + "="*70)
    print("STEP 2c: Axiom Hybrid (Physics + AI)")
    print("="*70)
    
    hybrid = HybridFluidRCLN(cs=0.1, fno_width=32, fno_modes=(12, 12), fno_layers=4).to(device)
    hybrid_history = train_model(hybrid, u_train, tau_train, u_val, tau_val,
                                 n_epochs=100, lr=1e-3, model_name="Axiom Hybrid")
    results['Axiom Hybrid'] = evaluate_model(hybrid, u_test, tau_test, "Axiom Hybrid")
    
    # ==============================================================================
    # Results Summary
    # ==============================================================================
    print("\n" + "="*70)
    print("B-01 BENCHMARK RESULTS")
    print("="*70)
    
    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"  R^2 = {res['r2']:.4f}")
        print(f"  MSE = {res['mse']:.6f}")
    
    # Check if Axiom wins
    axiom_r2 = results['Axiom Hybrid']['r2']
    fno_r2 = results['Pure FNO']['r2']
    smag_r2 = results['Smagorinsky']['r2']
    
    print("\n" + "-"*70)
    if axiom_r2 > fno_r2 and axiom_r2 > smag_r2:
        improvement = ((axiom_r2 - max(fno_r2, smag_r2)) / max(fno_r2, smag_r2)) * 100
        print(f"✓ VERDICT: Axiom-OS WINS! (R^2 = {axiom_r2:.4f})")
        print(f"  Improvement over best baseline: +{improvement:.1f}%")
        print("  Physics + AI > Pure Physics or Pure AI")
    else:
        print(f"✗ Axiom-OS did not achieve best performance")
        print(f"  Expected: Hybrid > Pure AI > Physics")
        print(f"  Got: Axiom={axiom_r2:.4f}, FNO={fno_r2:.4f}, Smag={smag_r2:.4f}")
    print("-"*70)
    
    # Generate plots
    print("\n" + "="*70)
    print("STEP 3: Generating Visualizations")
    print("="*70)
    
    save_dir = os.path.dirname(__file__)
    plot_results(results, u_test, tau_test, save_dir)
    
    print("\n" + "="*70)
    print("B-01 BENCHMARK COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_benchmark()
