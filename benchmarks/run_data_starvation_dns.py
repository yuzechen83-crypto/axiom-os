"""
Data Starvation Benchmark - Using Real DNS Data
"Hunger Games" - Axiom-OS vs Pure FNO on Spectral DNS Data

This version uses actual spectral DNS to generate data, creating
a more realistic scenario where physics prior provides advantage.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'C:\\Users\\ASUS\\PycharmProjects\\PythonProject1')

from axiom_os.layers.fluid_core import HybridFluidRCLN, FNO2d

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# Spectral DNS Simulator
# =============================================================================
class SpectralDNS:
    """2D Spectral DNS for Kolmogorov Flow."""
    
    def __init__(self, N=128, Re=5000, dt=0.001):
        self.N = N
        self.Re = Re
        self.dt = dt
        self.nu = 1.0 / Re
        
        # Spectral grid
        k = np.fft.fftfreq(N, 1.0/N) * 2 * np.pi
        self.kx, self.ky = np.meshgrid(k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1.0  # Avoid division by zero
        
        # Dealias mask (2/3 rule)
        self.dealias = (np.abs(self.kx) < 2*N/3) & (np.abs(self.ky) < 2*N/3)
        
    def rhs(self, w_hat):
        """Compute vorticity RHS."""
        # Streamfunction
        psi_hat = -w_hat / self.k2
        
        # Velocity in Fourier space
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        
        # Velocity in physical space
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        
        # Vorticity gradient
        w_x = np.fft.ifft2(1j * self.kx * w_hat).real
        w_y = np.fft.ifft2(1j * self.ky * w_hat).real
        
        # Kolmogorov forcing
        kf = 4
        forcing = self.Re * np.sin(kf * np.linspace(0, 2*np.pi, self.N, endpoint=False))
        
        # Nonlinear term
        nonlinear = -(u * w_x + v * w_y) + forcing[:, None]
        nonlinear_hat = np.fft.fft2(nonlinear) * self.dealias
        
        # RHS: -u·∇ω + ν∇²ω + forcing
        rhs = nonlinear_hat - self.nu * self.k2 * w_hat
        
        return rhs
    
    def step_rk4(self, w_hat):
        """One RK4 time step."""
        k1 = self.rhs(w_hat)
        k2 = self.rhs(w_hat + 0.5*self.dt*k1)
        k3 = self.rhs(w_hat + 0.5*self.dt*k2)
        k4 = self.rhs(w_hat + self.dt*k3)
        return w_hat + (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def compute_stress(self, w_hat):
        """Compute SGS stress from DNS."""
        psi_hat = -w_hat / self.k2
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        
        # Filter to LES scale
        filter_width = 4
        filter_mask = (np.abs(self.kx) < self.N/filter_width) & (np.abs(self.ky) < self.N/filter_width)
        
        # DNS velocity
        u_dns = np.fft.ifft2(u_hat).real
        v_dns = np.fft.ifft2(v_hat).real
        
        # LES velocity (filtered)
        u_les_hat = u_hat * filter_mask
        v_les_hat = v_hat * filter_mask
        u_les = np.fft.ifft2(u_les_hat).real
        v_les = np.fft.ifft2(v_les_hat).real
        
        # SGS stress: τ_ij = u_i*u_j - ũ_i*ũ_j (at LES scale)
        # Compute full product, then filter
        uu_hat = np.fft.fft2(u_dns * u_dns)
        uv_hat = np.fft.fft2(u_dns * v_dns)
        vv_hat = np.fft.fft2(v_dns * v_dns)
        
        uu_les = np.fft.ifft2(uu_hat * filter_mask).real
        uv_les = np.fft.ifft2(uv_hat * filter_mask).real
        vv_les = np.fft.ifft2(vv_hat * filter_mask).real
        
        # Stress components
        tau_xx = uu_les - u_les * u_les
        tau_xy = uv_les - u_les * v_les
        tau_yy = vv_les - v_les * v_les
        
        return (u_les, v_les, tau_xx, tau_xy, tau_yy)
    
    def generate_data(self, n_samples=100, warmup=500, sample_interval=50):
        """Generate training data."""
        print(f"Generating {n_samples} DNS samples (N={self.N}, Re={self.Re})...")
        
        # Initial vorticity
        w_hat = np.random.randn(self.N, self.N) + 1j*np.random.randn(self.N, self.N)
        w_hat = w_hat * np.exp(-self.k2 / 100)  # Smooth initial
        
        # Warmup
        print(f"  Warmup: {warmup} steps...")
        for _ in range(warmup):
            w_hat = self.step_rk4(w_hat)
        
        # Collect samples
        u_data, tau_data = [], []
        
        print(f"  Collecting {n_samples} samples...")
        for i in range(n_samples * sample_interval):
            w_hat = self.step_rk4(w_hat)
            
            if i % sample_interval == 0:
                u_les, v_les, tau_xx, tau_xy, tau_yy = self.compute_stress(w_hat)
                
                u_tensor = torch.FloatTensor(np.stack([u_les, v_les], axis=0))
                tau_tensor = torch.FloatTensor(np.stack([tau_xx, tau_xy, tau_yy], axis=0))
                
                u_data.append(u_tensor)
                tau_data.append(tau_tensor)
        
        u_tensor = torch.stack(u_data)
        tau_tensor = torch.stack(tau_data)
        
        print(f"  Generated: u {u_tensor.shape}, tau {tau_tensor.shape}")
        return u_tensor, tau_tensor

# =============================================================================
# Training Functions
# =============================================================================
def train_on_subset(model, u_train, tau_train, n_epochs=100, batch_size=4):
    """Train model on a subset."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_samples = u_train.shape[0]
    
    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            u_batch = u_train[batch_idx].to(device)
            tau_batch = tau_train[batch_idx].to(device)
            
            optimizer.zero_grad()
            tau_pred = model(u_batch)
            loss = torch.nn.functional.mse_loss(tau_pred, tau_batch)
            loss.backward()
            optimizer.step()
    
    return model

def evaluate_r2(model, u_test, tau_test):
    """Compute R2 score."""
    model.eval()
    with torch.no_grad():
        u_test = u_test.to(device)
        tau_test = tau_test.to(device)
        tau_pred = model(u_test)
        
        mse = torch.nn.functional.mse_loss(tau_pred, tau_test)
        var = torch.var(tau_test)
        r2 = 1 - mse / var
        
    return max(0, r2.item())

# =============================================================================
# Main Benchmark
# =============================================================================
def run_dns_benchmark():
    print("=" * 70)
    print("DATA STARVATION BENCHMARK - DNS VERSION")
    print("=" * 70)
    
    # Generate DNS dataset
    print("\n[1] Generating DNS dataset...")
    dns = SpectralDNS(N=64, Re=3000)  # Smaller for speed
    u_full, tau_full = dns.generate_data(n_samples=250, warmup=1000, sample_interval=20)
    
    # Split
    u_train_full = u_full[:200]
    tau_train_full = tau_full[:200]
    u_test = u_full[200:]
    tau_test = tau_full[200:]
    
    # Normalize
    u_mean, u_std = u_train_full.mean(), u_train_full.std()
    tau_mean, tau_std = tau_train_full.mean(), tau_train_full.std()
    
    u_train_full = (u_train_full - u_mean) / (u_std + 1e-8)
    tau_train_full = (tau_train_full - tau_mean) / (tau_std + 1e-8)
    u_test_norm = (u_test - u_mean) / (u_std + 1e-8)
    tau_test_norm = (tau_test - tau_mean) / (tau_std + 1e-8)
    
    # Test sizes
    subset_sizes = [10, 20, 50, 100, 200]
    n_epochs_list = [200, 200, 150, 100, 60]
    
    results = {'Pure FNO': [], 'Axiom Hybrid': []}
    
    print("\n[2] Running starvation tests...")
    print("-" * 70)
    
    for size, n_epochs in zip(subset_sizes, n_epochs_list):
        print(f"\n>>> Training with {size} frames (epochs={n_epochs})")
        
        u_subset = u_train_full[:size]
        tau_subset = tau_train_full[:size]
        
        for model_name in ['Pure FNO', 'Axiom Hybrid']:
            print(f"  {model_name}...", end=' ')
            
            if model_name == 'Pure FNO':
                model = FNO2d(modes=(12, 12), width=32, in_channels=2, out_channels=3)
            else:
                model = HybridFluidRCLN(fno_modes=(12, 12), fno_width=32)
            
            model = train_on_subset(model, u_subset, tau_subset, n_epochs, min(4, size))
            r2 = evaluate_r2(model, u_test_norm, tau_test_norm)
            results[model_name].append(r2)
            
            print(f"R2 = {r2:.4f}")
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Frames':<10} {'Pure FNO':<15} {'Axiom Hybrid':<15} {'Delta':<15}")
    print("-" * 70)
    
    for i, size in enumerate(subset_sizes):
        fno = results['Pure FNO'][i]
        axiom = results['Axiom Hybrid'][i]
        print(f"{size:<10} {fno:<15.4f} {axiom:<15.4f} {axiom-fno:<15.4f}")
    
    # Plot
    print("\n[3] Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(subset_sizes, results['Pure FNO'], 'b-o', linewidth=2, markersize=8, label='Pure FNO')
    ax.plot(subset_sizes, results['Axiom Hybrid'], 'r-s', linewidth=2, markersize=8, label='Axiom Hybrid')
    ax.fill_between(subset_sizes, results['Pure FNO'], results['Axiom Hybrid'], 
                    alpha=0.3, color='green', label='Physics Advantage')
    
    ax.set_xlabel('Training Frames', fontsize=12)
    ax.set_ylabel('R2 Score', fontsize=12)
    ax.set_title('Data Starvation: DNS Data (Spectral Simulation)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlim([8, 250])
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('benchmarks/data_starvation_dns.png', dpi=150, bbox_inches='tight')
    print("  Saved: benchmarks/data_starvation_dns.png")
    
    print("=" * 70)
    return results

if __name__ == "__main__":
    results = run_dns_benchmark()
