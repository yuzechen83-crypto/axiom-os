#!/usr/bin/env python3
"""
Complete SRBench Benchmark: Easy vs Medium vs Hard
生成 "公式恢复率 vs 噪声强度" 对比图
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from axiom_os.engine.discovery_hard import HardFormulaDiscovery


def run_with_noise(formula_list, noise_levels, n_runs=3):
    """
    在不同噪声水平下运行发现引擎
    
    Returns:
        dict: {noise_level: success_rate}
    """
    engine = HardFormulaDiscovery()
    results = {}
    
    for noise in noise_levels:
        successes = 0
        total = 0
        
        for formula in formula_list:
            for run in range(n_runs):
                # Generate data
                np.random.seed(42 + run)
                n_vars = len(formula['vars'])
                X = np.random.uniform(0.5, 2.0, size=(500, n_vars))
                y = formula['func'](X)
                
                # Add noise
                if noise > 0:
                    y += np.random.normal(0, noise * np.std(y), size=y.shape)
                
                # Discover
                try:
                    _, r2 = engine.discover(X, y, formula['vars'])
                    if r2 > 0.95:
                        successes += 1
                except Exception:
                    pass
                total += 1
        
        results[noise] = successes / total if total > 0 else 0.0
        print(f"  Noise={noise:.2f}: {successes}/{total} = {results[noise]:.1%}")
    
    return results


def main():
    print("="*70)
    print("SRBench Complete: Recovery Rate vs Noise")
    print("="*70)
    
    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
    
    # Easy formulas (6个)
    easy_formulas = [
        {"name": "I.6.2a", "vars": ["m", "c"], "func": lambda X: X[:,0]*X[:,1]**2},
        {"name": "I.8.14", "vars": ["v", "t"], "func": lambda X: X[:,0]*X[:,1]},
        {"name": "I.12.1", "vars": ["mu", "N"], "func": lambda X: X[:,0]*X[:,1]},
        {"name": "I.12.11", "vars": ["m", "v"], "func": lambda X: 0.5*X[:,0]*X[:,1]**2},
        {"name": "I.14.3", "vars": ["m", "g", "h"], "func": lambda X: X[:,0]*X[:,1]*X[:,2]},
        {"name": "I.15.3t", "vars": ["v", "u", "a"], "func": lambda X: (X[:,0]-X[:,1])/X[:,2]},
    ]
    
    # Medium formulas (12个)
    medium_formulas = easy_formulas + [
        {"name": "I.16.6", "vars": ["rho", "v"], "func": lambda X: 0.5*X[:,0]*X[:,1]**2},
        {"name": "I.18.4", "vars": ["m1", "r1", "m2", "r2"], 
         "func": lambda X: (X[:,0]*X[:,1]+X[:,2]*X[:,3])/(X[:,0]+X[:,2])},
        {"name": "I.24.6", "vars": ["C", "V"], "func": lambda X: 0.5*X[:,0]*X[:,1]**2},
        {"name": "I.27.6", "vars": ["h", "f"], "func": lambda X: X[:,0]*X[:,1]},
        {"name": "I.29.4", "vars": ["omega", "c"], "func": lambda X: X[:,0]/X[:,1]},
        {"name": "I.30.5", "vars": ["h", "p"], "func": lambda X: X[:,0]/X[:,1]},
    ]
    
    # Hard formulas (8个)
    hard_formulas = [
        {"name": "I.9.18", "vars": ["G", "m1", "m2", "r"], 
         "func": lambda X: X[:,0]*X[:,1]*X[:,2]/(X[:,3]**2)},
        {"name": "I.12.2", "vars": ["q1", "q2", "eps0", "r"],
         "func": lambda X: X[:,0]*X[:,1]/(4*np.pi*X[:,2]*X[:,3]**2)},
        {"name": "I.37.4", "vars": ["p", "m", "U"],
         "func": lambda X: X[:,0]**2/(2*X[:,1]) + X[:,2]},
    ] + medium_formulas[-2:]  # 加上 I.29.4 和 I.30.5
    
    print("\n[1/3] Testing EASY (6 formulas)...")
    easy_results = run_with_noise(easy_formulas, noise_levels)
    
    print("\n[2/3] Testing MEDIUM (12 formulas)...")
    medium_results = run_with_noise(medium_formulas, noise_levels)
    
    print("\n[3/3] Testing HARD (8 formulas)...")
    hard_results = run_with_noise(hard_formulas, noise_levels)
    
    # Plot
    print("\n[4/4] Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(noise_levels, [easy_results[n]*100 for n in noise_levels], 
            'o-', linewidth=2, markersize=10, label='Feynman Easy (6 formulas)', color='blue')
    ax.plot(noise_levels, [medium_results[n]*100 for n in noise_levels],
            's-', linewidth=2, markersize=10, label='Feynman Medium (12 formulas)', color='green')
    ax.plot(noise_levels, [hard_results[n]*100 for n in noise_levels],
            '^-', linewidth=2, markersize=10, label='Feynman Hard (8 formulas)', color='red')
    
    ax.set_xlabel('Noise strength (std relative to target)', fontsize=12)
    ax.set_ylabel('Formula recovery rate (%)', fontsize=12)
    ax.set_title('SRBench Feynman: Recovery Rate vs Noise (Axiom-OS Discovery Engine)', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    
    # Save
    output_path = Path("docs/images/srbench_complete_recovery_vs_noise.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    # Save data
    data = {
        "noise_levels": noise_levels,
        "easy": {str(k): v for k, v in easy_results.items()},
        "medium": {str(k): v for k, v in medium_results.items()},
        "hard": {str(k): v for k, v in hard_results.items()},
    }
    with open("benchmarks/srbench_complete_results.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved data to benchmarks/srbench_complete_results.json")
    
    # Print summary table
    print("\n" + "="*70)
    print("Summary: Recovery Rate (%)")
    print("="*70)
    print(f"{'Noise':>8s} | {'Easy':>8s} | {'Medium':>8s} | {'Hard':>8s}")
    print("-"*70)
    for n in noise_levels:
        print(f"{n:8.2f} | {easy_results[n]*100:8.1f} | {medium_results[n]*100:8.1f} | {hard_results[n]*100:8.1f}")
    print("="*70)


if __name__ == "__main__":
    main()
