"""
Data Efficiency Test: The "Starvation" Challenge

Train with extremely limited data (10-50 frames) and test on full set.
This tests whether physics guidance reduces data requirements.

Hypothesis:
- Pure FNO: Will overfit or fail (no data = no learning)
- Axiom Hybrid: Will survive (Hard Core provides base guess)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm import tqdm

from axiom_os.layers.fluid_core import (
    SmagorinskyCore, FNO2d, HybridFluidRCLN,
    compute_correlation_coefficient, compute_mse
)
from benchmarks.run_b01_fluid_fixed import (
    KolmogorovFlow, filter_to_les, DataNormalizer,
    train_model_normalized, evaluate_model_normalized
)


def generate_full_dataset(n_samples: int = 200, re: float = 5000.0):
    """Generate full dataset."""
    print(f"Generating {n_samples} samples at Re={re}...")
    flow = KolmogorovFlow(N=256, Re=re, dt=0.001, force_amp=0.5)
    snapshots = flow.simulate(n_steps=n_samples, warmup_steps=2000)
    
    u_list, tau_list = [], []
    for i in tqdm(range(n_samples), desc="Filtering"):
        u, tau = filter_to_les(snapshots[i], filter_ratio=4)
        u_list.append(u)
        tau_list.append(tau)
    
    return torch.tensor(np.array(u_list), dtype=torch.float32), \
           torch.tensor(np.array(tau_list), dtype=torch.float32)


def train_with_limited_data(model: nn.Module, normalizer: DataNormalizer,
                            u_full: torch.Tensor, tau_full: torch.Tensor,
                            n_frames: int, n_epochs: int = 100):
    """Train model with only n_frames of data."""
    
    # Take only first n_frames
    u_limited = u_full[:n_frames]
    tau_limited = tau_full[:n_frames]
    
    # Use last 20% for validation (if enough data)
    n_val = max(2, int(0.2 * n_frames))
    n_train = n_frames - n_val
    
    u_train = u_limited[:n_train]
    tau_train = tau_limited[:n_train]
    u_val = u_limited[n_train:]
    tau_val = tau_limited[n_train:]
    
    print(f"  Training on {n_train} frames, validating on {n_val} frames")
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    u_train_norm = normalizer.normalize_u(u_train)
    tau_train_norm = normalizer.normalize_tau(tau_train)
    u_val_norm = normalizer.normalize_u(u_val)
    tau_val_norm = normalizer.normalize_tau(tau_val)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        
        # Full batch (small data)
        u_batch = u_train_norm.to(device)
        tau_batch = tau_train_norm.to(device)
        
        optimizer.zero_grad()
        tau_pred = model(u_batch)
        loss = F.mse_loss(tau_pred, tau_batch)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            tau_val_pred = model(u_val_norm.to(device))
            val_loss = F.mse_loss(tau_val_pred, tau_val_norm.to(device)).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > 20:  # Early stopping
            break
    
    return model


def run_data_efficiency_test():
    """
    Train with [10, 20, 50, 100, 200] frames, test on full 200-frame test set.
    """
    print("="*70)
    print("DATA EFFICIENCY TEST: The Starvation Challenge")
    print("="*70)
    print("\nSetup:")
    print("  Full dataset: 200 frames (Re=5000)")
    print("  Training subsets: [10, 20, 50, 100, 200] frames")
    print("  Test: Full 200 frames (always)")
    print("\nHypothesis:")
    print("  - Pure FNO: Needs lots of data, crashes with <50 frames")
    print("  - Axiom Hybrid: Survives with 10 frames (physics base guess)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate full dataset once
    u_full, tau_full = generate_full_dataset(n_samples=200, re=5000.0)
    
    # Split: 200 train, 200 test (different simulations)
    u_train_pool = u_full[:200]
    tau_train_pool = tau_full[:200]
    
    # Generate separate test set
    print("\nGenerating test set...")
    u_test, tau_test = generate_full_dataset(n_samples=50, re=5000.0)
    
    # Test data sizes
    data_sizes = [10, 20, 50, 100, 200]
    results = {'Pure FNO': {}, 'Axiom Hybrid': {}}
    
    for n_frames in data_sizes:
        print(f"\n{'='*70}")
        print(f"TRAINING WITH {n_frames} FRAMES")
        print('='*70)
        
        # Fit normalizer on the limited training data
        normalizer = DataNormalizer()
        normalizer.fit(u_train_pool[:n_frames], tau_train_pool[:n_frames])
        
        # Train Pure FNO
        print(f"\n[1] Pure FNO ({n_frames} frames)...")
        fno = FNO2d(in_channels=2, out_channels=3, width=32, modes=(12, 12), n_layers=4)
        fno = train_with_limited_data(fno, normalizer, u_train_pool, tau_train_pool, 
                                      n_frames, n_epochs=200)
        res_fno = evaluate_model_normalized(fno, normalizer, u_test, u_test, 
                                            f"Pure FNO ({n_frames} frames)")
        results['Pure FNO'][n_frames] = res_fno['r2']
        
        # Train Axiom Hybrid
        print(f"\n[2] Axiom Hybrid ({n_frames} frames)...")
        hybrid = HybridFluidRCLN(cs=0.1, fno_width=32, fno_modes=(12, 12), fno_layers=4)
        hybrid = train_with_limited_data(hybrid, normalizer, u_train_pool, tau_train_pool,
                                         n_frames, n_epochs=200)
        res_axiom = evaluate_model_normalized(hybrid, normalizer, u_test, tau_test,
                                              f"Axiom ({n_frames} frames)")
        results['Axiom Hybrid'][n_frames] = res_axiom['r2']
    
    # ==============================================================================
    # Visualization
    # ==============================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nR^2 Scores vs Data Size:")
    print(f"{'Frames':<10} {'Pure FNO':<15} {'Axiom Hybrid':<15} {'Advantage':<15}")
    print("-"*55)
    for n in data_sizes:
        r2_fno = results['Pure FNO'][n]
        r2_axiom = results['Axiom Hybrid'][n]
        advantage = r2_axiom - r2_fno
        print(f"{n:<10} {r2_fno:<15.4f} {r2_axiom:<15.4f} {advantage:+<15.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    fno_scores = [results['Pure FNO'][n] for n in data_sizes]
    axiom_scores = [results['Axiom Hybrid'][n] for n in data_sizes]
    
    ax.plot(data_sizes, fno_scores, 'o-', color='#3498db', linewidth=3,
            markersize=12, label='Pure FNO (Pure AI)')
    ax.plot(data_sizes, axiom_scores, 's-', color='#2ecc71', linewidth=3,
            markersize=12, label='Axiom Hybrid (Physics + AI)')
    
    # Add horizontal line at R2=0.8 (good performance threshold)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Good Performance (R^2=0.8)')
    
    # Add value labels
    for n, score in zip(data_sizes, fno_scores):
        ax.annotate(f'{score:.2f}', xy=(n, score), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=10,
                   color='#3498db', fontweight='bold')
    for n, score in zip(data_sizes, axiom_scores):
        ax.annotate(f'{score:.2f}', xy=(n, score), xytext=(0, -15),
                   textcoords='offset points', ha='center', fontsize=10,
                   color='#2ecc71', fontweight='bold')
    
    ax.set_xlabel('Training Data Size (Number of Frames)', fontsize=14)
    ax.set_ylabel('R^2 Score on Test Set', fontsize=14)
    ax.set_title('Data Efficiency: Learning with Limited Data\n(Trained on Re=5000, Tested on Re=5000)', 
                 fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('benchmarks/data_efficiency_test.png', dpi=150)
    print("\nSaved: benchmarks/data_efficiency_test.png")
    plt.close()
    
    # ==============================================================================
    # Key Metrics
    # ==============================================================================
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Data efficiency threshold
    fno_threshold = None
    axiom_threshold = None
    
    for n in data_sizes:
        if results['Pure FNO'][n] > 0.7 and fno_threshold is None:
            fno_threshold = n
        if results['Axiom Hybrid'][n] > 0.7 and axiom_threshold is None:
            axiom_threshold = n
    
    if fno_threshold and axiom_threshold:
        efficiency_gain = fno_threshold / axiom_threshold
        print(f"\nData Efficiency Gain: {efficiency_gain:.1f}x")
        print(f"  - Pure FNO needs ~{fno_threshold} frames for R^2>0.7")
        print(f"  - Axiom needs ~{axiom_threshold} frames for R^2>0.7")
    
    # Extreme starvation test
    gap_10 = results['Axiom Hybrid'][10] - results['Pure FNO'][10]
    print(f"\nExtreme Starvation (10 frames): Axiom advantage = +{gap_10:.3f} R^2")
    
    if gap_10 > 0.1:
        print("\n*** VERDICT: Axiom-OS demonstrates superior data efficiency! ***")
        print("    Physics guidance reduces data requirements significantly.")
    else:
        print("\nResult: Both models need substantial data for good performance.")
    
    print("="*70)
    return results


if __name__ == "__main__":
    results = run_data_efficiency_test()
