"""
OOD Test: Reynolds Shift - The "Wilderness" Challenge

Train on Re=1000 (smooth flow), test on Re=10000 (chaotic turbulence).
This tests whether physics guidance helps generalization.

Hypothesis:
- Pure FNO: Will crash (memorized Re=1000 patterns don't apply to Re=10000)
- Axiom Hybrid: Will survive (Smagorinsky provides physics-based safety net)
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


def generate_dataset_re(n_samples: int, re: float, warmup: int = 2000):
    """Generate dataset at specific Reynolds number."""
    print(f"\nGenerating data at Re={re:.0f}...")
    flow = KolmogorovFlow(N=256, Re=re, dt=0.001, force_amp=0.5)
    snapshots = flow.simulate(n_steps=n_samples, warmup_steps=warmup)
    
    u_list, tau_list = [], []
    for i in tqdm(range(n_samples), desc="Filtering"):
        u, tau = filter_to_les(snapshots[i], filter_ratio=4)
        u_list.append(u)
        tau_list.append(tau)
    
    return torch.tensor(np.array(u_list), dtype=torch.float32), \
           torch.tensor(np.array(tau_list), dtype=torch.float32)


def run_reynolds_shift_test():
    """
    Train on Re=1000, test on Re=[500, 1000, 5000, 10000]
    """
    print("="*70)
    print("OOD TEST: REYNOLDS SHIFT (The Wilderness Challenge)")
    print("="*70)
    print("\nSetup:")
    print("  Train: Re=1000 (Laminar/Transition)")
    print("  Test:  Re=[500, 1000, 5000, 10000] (Interpolation & Extrapolation)")
    print("\nHypothesis:")
    print("  - Pure FNO: Will crash on Re=5000/10000 (distribution shift)")
    print("  - Axiom Hybrid: Will survive (physics safety net)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate training data (Re=1000)
    u_train, tau_train = generate_dataset_re(150, re=1000.0, warmup=2000)
    u_val, tau_val = generate_dataset_re(30, re=1000.0, warmup=500)
    
    # Fit normalizer on training data
    normalizer = DataNormalizer()
    normalizer.fit(u_train, tau_train)
    
    # ==============================================================================
    # Train Models
    # ==============================================================================
    print("\n" + "="*70)
    print("TRAINING PHASE (Re=1000)")
    print("="*70)
    
    # Pure FNO
    print("\n[1] Training Pure FNO...")
    fno = FNO2d(in_channels=2, out_channels=3, width=32, modes=(12, 12), n_layers=4)
    train_model_normalized(fno, normalizer, u_train, tau_train, u_val, tau_val,
                          n_epochs=100, lr=1e-3, model_name="Pure FNO")
    
    # Axiom Hybrid
    print("\n[2] Training Axiom Hybrid...")
    hybrid = HybridFluidRCLN(cs=0.1, fno_width=32, fno_modes=(12, 12), fno_layers=4)
    train_model_normalized(hybrid, normalizer, u_train, tau_train, u_val, tau_val,
                          n_epochs=100, lr=1e-3, model_name="Axiom Hybrid")
    
    # ==============================================================================
    # OOD Test on multiple Reynolds numbers
    # ==============================================================================
    print("\n" + "="*70)
    print("OOD TESTING PHASE")
    print("="*70)
    
    test_reynolds = [500, 1000, 5000, 10000]
    results = {'Pure FNO': {}, 'Axiom Hybrid': {}}
    
    for re_test in test_reynolds:
        print(f"\n--- Testing on Re={re_test} ---")
        u_test, tau_test = generate_dataset_re(20, re=float(re_test), warmup=2000)
        
        # Evaluate Pure FNO
        res_fno = evaluate_model_normalized(fno, normalizer, u_test, tau_test, 
                                            f"Pure FNO @ Re={re_test}")
        results['Pure FNO'][re_test] = res_fno['r2']
        
        # Evaluate Axiom
        res_axiom = evaluate_model_normalized(hybrid, normalizer, u_test, tau_test,
                                              f"Axiom @ Re={re_test}")
        results['Axiom Hybrid'][re_test] = res_axiom['r2']
    
    # ==============================================================================
    # Visualization: Generalization Gap
    # ==============================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nR^2 Scores across Reynolds numbers:")
    print(f"{'Re':<10} {'Pure FNO':<15} {'Axiom Hybrid':<15} {'Gap':<15}")
    print("-"*55)
    for re in test_reynolds:
        r2_fno = results['Pure FNO'][re]
        r2_axiom = results['Axiom Hybrid'][re]
        gap = r2_axiom - r2_fno
        print(f"{re:<10} {r2_fno:<15.4f} {r2_axiom:<15.4f} {gap:+<15.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    re_values = np.array(test_reynolds)
    fno_scores = [results['Pure FNO'][re] for re in test_reynolds]
    axiom_scores = [results['Axiom Hybrid'][re] for re in test_reynolds]
    
    # Mark training point
    train_re = 1000
    ax.axvline(x=train_re, color='gray', linestyle='--', alpha=0.5, label='Training Regime')
    ax.axvspan(500, 1500, alpha=0.1, color='green', label='Interpolation Zone')
    ax.axvspan(2000, 11000, alpha=0.1, color='red', label='Extrapolation Zone (Wilderness)')
    
    # Plot lines
    ax.plot(re_values, fno_scores, 'o-', color='#3498db', linewidth=3, 
            markersize=10, label='Pure FNO (Pure AI)')
    ax.plot(re_values, axiom_scores, 's-', color='#2ecc71', linewidth=3,
            markersize=10, label='Axiom Hybrid (Physics + AI)')
    
    # Add value labels
    for re, score in zip(re_values, fno_scores):
        ax.annotate(f'{score:.3f}', xy=(re, score), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=10,
                   color='#3498db', fontweight='bold')
    for re, score in zip(re_values, axiom_scores):
        ax.annotate(f'{score:.3f}', xy=(re, score), xytext=(0, -15),
                   textcoords='offset points', ha='center', fontsize=10,
                   color='#2ecc71', fontweight='bold')
    
    ax.set_xlabel('Test Reynolds Number (Re)', fontsize=14)
    ax.set_ylabel('R^2 Score (Higher is Better)', fontsize=14)
    ax.set_title('OOD Generalization: Reynolds Shift Test\n(Trained on Re=1000)', 
                 fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylim(-0.2, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('benchmarks/ood_reynolds_test.png', dpi=150)
    print("\nSaved: benchmarks/ood_reynolds_test.png")
    plt.close()
    
    # ==============================================================================
    # Final Verdict
    # ==============================================================================
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    # Check extrapolation performance
    re_5000_gap = results['Axiom Hybrid'][5000] - results['Pure FNO'][5000]
    re_10000_gap = results['Axiom Hybrid'][10000] - results['Pure FNO'][10000]
    
    print(f"\nExtrapolation Gain (Re=5000):  +{re_5000_gap:.3f} R^2")
    print(f"Extrapolation Gain (Re=10000): +{re_10000_gap:.3f} R^2")
    
    if re_5000_gap > 0.1 or re_10000_gap > 0.1:
        print("\n*** VERDICT: Axiom-OS demonstrates superior OOD generalization! ***")
        print("    Physics guidance provides robustness in the wilderness.")
    else:
        print("\nResult: Gap smaller than expected. Pure AI also generalizes well.")
        print("    (This is also a valid scientific finding!)")
    
    print("="*70)
    return results


if __name__ == "__main__":
    results = run_reynolds_shift_test()
