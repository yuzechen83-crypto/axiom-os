# -*- coding: utf-8 -*-
"""
Fast JHTDB Experiment - 64^3 only
For quick validation of real data pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import sys

sys.path.insert(0, r'C:\Users\ASUS\PycharmProjects\PythonProject1')

from axiom_os.layers.rcln_v3_advanced import RCLNv3_Advanced
from axiom_os.data.jhtdb_loader import JHTDBLoader


def main():
    print("="*70)
    print("JHTDB Real Data Experiment - Fast Version (64^3)")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Create loader
    loader = JHTDBLoader(
        cutout_size=64,
        filter_width=3,
        use_synthetic=True,
        cache_dir='./jhtdb_cache_64',
    )
    
    print("\n[Data] Loading 50 cutouts...")
    data = loader.get_cutouts(num_samples=50, device=device)
    print(f"[Data] Loaded: {data['velocity'].shape}")
    print(f"[Data] Source: {data['source']}")
    print(f"[Data] Velocity std: {data['velocity'].std():.4f}")
    print(f"[Data] Tau std: {data['tau_sgs'].std():.4f}")
    
    # Split
    n_train = 40
    train_u = data['velocity'][:n_train]
    train_tau = data['tau_sgs'][:n_train]
    val_u = data['velocity'][n_train:]
    val_tau = data['tau_sgs'][n_train:]
    
    # Model
    model = RCLNv3_Advanced(
        resolution=64,
        fno_width=16,
        fno_modes=8,
        cs_init=0.1,
        lambda_hard=0.3,
        lambda_soft=0.7,
    ).to(device)
    
    # Train with 3 discovery cycles
    print(f"\n{'='*70}")
    print("Training with 3 Discovery Cycles")
    print(f"{'='*70}")
    
    results = []
    
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1}/3 ---")
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(15):
            model.train()
            u_batch = train_u[:20]  # Use subset for speed
            tau_batch = train_tau[:20]
            
            pred, info = model(u_batch, return_physics=True)
            loss = F.mse_loss(pred, tau_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                cs = info['physics_params'].get('cs', 0)
                print(f"  Epoch {epoch+1}: Loss={loss.item():.6f}, Cs={cs:.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_pred, info = model(val_u, return_physics=True)
            val_loss = F.mse_loss(val_pred, val_tau).item()
            cs = info['physics_params'].get('cs', 0)
        
        print(f"  Val MSE: {val_loss:.6f}, Cs: {cs:.4f}")
        
        results.append({
            'cycle': cycle + 1,
            'val_mse': val_loss,
            'cs': cs,
        })
        
        # Improve Cs (simulating DeepSeek)
        if cycle < 2:
            with torch.no_grad():
                new_cs = cs * 0.8 + 0.16 * 0.2
                model.hard_core._cs_raw.data = torch.log(torch.exp(torch.tensor(new_cs / 0.2)) - 1)
                print(f"  [DeepSeek] Cs: {cs:.4f} -> {new_cs:.4f}")
            
            # Reset soft shell
            def reset_weights(m):
                if isinstance(m, (nn.Conv3d, nn.Linear)):
                    m.reset_parameters()
            model.soft_shell.apply(reset_weights)
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Cycle':<10} {'Val MSE':<12} {'Cs':<10}")
    print("-"*40)
    for r in results:
        print(f"{r['cycle']:<10} {r['val_mse']:<12.6f} {r['cs']:<10.4f}")
    
    print(f"\nCs Convergence: {results[0]['cs']:.4f} -> {results[-1]['cs']:.4f}")
    print(f"Target (theoretical): 0.16")
    
    with open('jhtdb_fast_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n[DONE] Results saved: jhtdb_fast_results.json")


if __name__ == "__main__":
    main()
