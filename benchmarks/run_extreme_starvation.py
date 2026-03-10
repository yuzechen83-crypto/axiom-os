"""
Extreme Data Starvation Test
Tests models with only 1-10 frames to highlight physics prior advantage
"""
import sys, torch, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, 'C:\\Users\\ASUS\\PycharmProjects\\PythonProject1')
from axiom_os.layers.fluid_core import HybridFluidRCLN, FNO2d, SmagorinskyCore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_complex_data(n_samples=100, N=64):
    """Generate data with complex physics (Smagorinsky + noise + anisotropy)."""
    print(f"Generating {n_samples} complex samples...")
    
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u_data, tau_data = [], []
    
    for _ in range(n_samples):
        # Multiple forcing scales (not just Smagorinsky)
        u = np.zeros((N, N))
        v = np.zeros((N, N))
        
        # Large scale forcing
        for k in [2, 4, 8]:
            amp = 1.0 / k
            phase = np.random.rand(4) * 2 * np.pi
            u += amp * np.sin(k*X + phase[0]) * np.cos(k*Y + phase[1])
            v += amp * np.cos(k*X + phase[2]) * np.sin(k*Y + phase[3])
        
        # Add small-scale anisotropic structures (harder for FNO to learn)
        for _ in range(5):
            kx, ky = np.random.randint(1, 16, 2)
            amp = 0.2 / np.sqrt(kx**2 + ky**2)
            phase = np.random.rand(2) * 2 * np.pi
            u += amp * np.sin(kx*X + phase[0]) * np.cos(ky*Y)
            v += amp * np.cos(kx*X) * np.sin(ky*Y + phase[1])
        
        # Velocity gradients
        du_dx = np.gradient(u, x, axis=0)
        du_dy = np.gradient(u, y, axis=1)
        dv_dx = np.gradient(v, x, axis=0)
        dv_dy = np.gradient(v, y, axis=1)
        
        # Strain rate
        S11, S12, S22 = du_dx, 0.5*(du_dy+dv_dx), dv_dy
        S_mag = np.sqrt(2*(S11**2 + 2*S12**2 + S22**2))
        
        # SGS stress with anisotropic correction
        Cs, Delta = 0.15, 2*np.pi/N
        
        # Base Smagorinsky
        tau11_base = -2*(Cs*Delta)**2 * S_mag * S11
        tau12_base = -2*(Cs*Delta)**2 * S_mag * S12
        tau22_base = -2*(Cs*Delta)**2 * S_mag * S22
        
        # Add anisotropic correction (the "gap" Axiom should learn)
        aniso_strength = 0.3
        tau11 = tau11_base * (1 + aniso_strength * np.random.randn(N, N))
        tau12 = tau12_base * (1 + aniso_strength * np.random.randn(N, N))
        tau22 = tau22_base * (1 + aniso_strength * np.random.randn(N, N))
        
        u_data.append(torch.FloatTensor(np.stack([u, v], axis=0)))
        tau_data.append(torch.FloatTensor(np.stack([tau11, tau12, tau22], axis=0)))
    
    return torch.stack(u_data), torch.stack(tau_data)

def train_and_eval(model, u_train, tau_train, u_test, tau_test, epochs=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = u_train.shape[0]
    
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, 2):
            batch = idx[i:i+2]
            u_batch = u_train[batch].to(device)
            tau_batch = tau_train[batch].to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(u_batch), tau_batch)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pred = model(u_test.to(device))
        mse = torch.nn.functional.mse_loss(pred, tau_test.to(device))
        var = torch.var(tau_test)
        return max(0, 1 - mse/var).item()

def run_extreme_test():
    print("="*70)
    print("EXTREME DATA STARVATION TEST (1-10 frames)")
    print("="*70)
    
    # Generate data
    u_full, tau_full = generate_complex_data(120, N=64)
    
    # Split: 100 train, 20 test
    u_train_full, tau_train_full = u_full[:100], tau_full[:100]
    u_test, tau_test = u_full[100:], tau_full[100:]
    
    # Normalize
    u_mean, u_std = u_train_full.mean(), u_train_full.std()
    tau_mean, tau_std = tau_train_full.mean(), tau_train_full.std()
    u_train_full = (u_train_full - u_mean) / (u_std + 1e-8)
    tau_train_full = (tau_train_full - tau_mean) / (tau_std + 1e-8)
    u_test = (u_test - u_mean) / (u_std + 1e-8)
    tau_test = (tau_test - tau_mean) / (tau_std + 1e-8)
    
    # Extreme starvation: 1, 2, 3, 5, 10 frames
    subset_sizes = [1, 2, 3, 5, 10]
    epochs_list = [500, 400, 300, 200, 150]
    
    results = {'Pure FNO': [], 'Axiom Hybrid': [], 'Smagorinsky (baseline)': []}
    
    # Get Smagorinsky baseline (no training needed)
    smag = SmagorinskyCore(cs=0.15).to(device)
    with torch.no_grad():
        tau_smag = smag(u_test.to(device))
        mse_smag = torch.nn.functional.mse_loss(tau_smag, tau_test.to(device))
        r2_smag = max(0, 1 - mse_smag / torch.var(tau_test)).item()
    
    print(f"\nSmagorinsky baseline R2: {r2_smag:.4f}")
    print("-"*70)
    
    for size, epochs in zip(subset_sizes, epochs_list):
        print(f"\n>>> Training with {size} frame(s) (epochs={epochs})")
        u_sub = u_train_full[:size]
        tau_sub = tau_train_full[:size]
        
        for name in ['Pure FNO', 'Axiom Hybrid']:
            print(f"  {name}...", end=' ')
            if name == 'Pure FNO':
                model = FNO2d(modes=(12, 12), width=32, in_channels=2, out_channels=3)
            else:
                model = HybridFluidRCLN(fno_modes=(12, 12), fno_width=32)
            
            r2 = train_and_eval(model, u_sub, tau_sub, u_test, tau_test, epochs)
            results[name].append(r2)
            results['Smagorinsky (baseline)'].append(r2_smag)
            print(f"R2 = {r2:.4f}")
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Frames':<10} {'Pure FNO':<15} {'Axiom Hybrid':<15} {'Smagorinsky':<15}")
    print("-"*70)
    
    for i, size in enumerate(subset_sizes):
        print(f"{size:<10} {results['Pure FNO'][i]:<15.4f} "
              f"{results['Axiom Hybrid'][i]:<15.4f} "
              f"{results['Smagorinsky (baseline)'][i]:<15.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(subset_sizes, results['Pure FNO'], 'b-o', linewidth=2, markersize=8, label='Pure FNO')
    ax.plot(subset_sizes, results['Axiom Hybrid'], 'r-s', linewidth=2, markersize=8, label='Axiom Hybrid')
    ax.axhline(y=r2_smag, color='g', linestyle='--', linewidth=2, label='Smagorinsky (no training)')
    
    ax.fill_between(subset_sizes, results['Pure FNO'], results['Axiom Hybrid'], 
                    alpha=0.3, color='red', where=[a > b for a, b in zip(results['Axiom Hybrid'], results['Pure FNO'])],
                    label='Physics Prior Advantage')
    
    ax.set_xlabel('Training Frames', fontsize=12)
    ax.set_ylabel('R2 Score', fontsize=12)
    ax.set_title('Extreme Data Starvation: Complex Physics Data', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('benchmarks/extreme_starvation.png', dpi=150, bbox_inches='tight')
    print("\nSaved: benchmarks/extreme_starvation.png")
    
    # Key insight
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    
    if results['Axiom Hybrid'][0] > results['Pure FNO'][0]:
        print(f"[OK] At 1 frame: Axiom R2 = {results['Axiom Hybrid'][0]:.3f}, Pure FNO R2 = {results['Pure FNO'][0]:.3f}")
        print(f"    Physical prior provides {results['Axiom Hybrid'][0] - results['Pure FNO'][0]:.3f} advantage!")
    else:
        print(f"[INFO] At 1 frame: Axiom R2 = {results['Axiom Hybrid'][0]:.3f}, Pure FNO R2 = {results['Pure FNO'][0]:.3f}")
        print(f"    Both models struggle with 1 frame. Need more data for complex physics.")
    
    print("="*70)
    return results

if __name__ == "__main__":
    run_extreme_test()
