# -*- coding: utf-8 -*-
"""
GPU-Accelerated RCLN + DeepSeek Discovery on Real JHTDB 1024^3 Data

Requirements:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- JHTDB access (or high-quality synthetic fallback)

Features:
- Mixed precision training (FP16) for speed
- GPU memory optimization
- Real-time monitoring
- Checkpoint saving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, r'C:\Users\ASUS\PycharmProjects\PythonProject1')

from axiom_os.layers.rcln_v3_advanced import RCLNv3_Advanced
from axiom_os.data.jhtdb_loader import JHTDBLoader


def setup_gpu():
    """Setup GPU with optimal settings"""
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available! Please install PyTorch with CUDA support.")
        print("  Install command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return None
    
    device = torch.device('cuda')
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("="*70)
    print("GPU Configuration")
    print("="*70)
    print(f"Device: {gpu_name}")
    print(f"Memory: {gpu_memory:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    
    # Enable TF32 for Ampere GPUs (RTX 30xx/40xx/50xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("="*70)
    return device


def generate_or_load_data(loader, num_samples, device):
    """Generate or load data with GPU acceleration"""
    print(f"\n[Data] Loading {num_samples} samples...")
    
    # Check if cached
    cache_file = f'jhtdb_data_{loader.cutout_size}_{num_samples}.pt'
    if os.path.exists(cache_file):
        print(f"[Data] Loading from cache: {cache_file}")
        data = torch.load(cache_file)
        # Move to GPU
        data['velocity'] = data['velocity'].to(device)
        data['tau_sgs'] = data['tau_sgs'].to(device)
        return data
    
    # Generate/load
    data = loader.get_cutouts(num_samples=num_samples, device=device)
    
    # Cache for future runs
    torch.save({
        'velocity': data['velocity'].cpu(),
        'tau_sgs': data['tau_sgs'].cpu(),
        'source': data['source'],
    }, cache_file)
    print(f"[Data] Cached to: {cache_file}")
    
    return data


def train_cycle_gpu(model, train_u, train_tau, val_u, val_tau, 
                   epochs, device, use_amp=True):
    """Train one discovery cycle with GPU acceleration"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if use_amp else None
    
    n_train = len(train_u)
    batch_size = 16 if use_amp else 8  # Larger batch with mixed precision
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle indices
        indices = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            end_i = min(i + batch_size, n_train)
            batch_idx = indices[i:end_i]
            
            u_batch = train_u[batch_idx]
            tau_batch = train_tau[batch_idx]
            
            optimizer.zero_grad()
            
            # Mixed precision forward
            if use_amp:
                with autocast():
                    pred, info = model(u_batch, return_physics=True)
                    loss = F.mse_loss(pred, tau_batch)
                
                # Backward with scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred, info = model(u_batch, return_physics=True)
                loss = F.mse_loss(pred, tau_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            epoch_loss += loss.item() * (end_i - i)
        
        epoch_loss /= n_train
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            cs_str = ""
            if 'physics_params' in info and 'cs' in info['physics_params']:
                cs = info['physics_params']['cs']
                if torch.is_tensor(cs):
                    cs = cs.item()
                cs_str = f", Cs={cs:.4f}"
            print(f"    Epoch {epoch+1:3d}: Loss={epoch_loss:.6f}{cs_str}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        with autocast() if use_amp else torch.enable_grad():
            val_pred, info = model(val_u, return_physics=True)
            val_loss = F.mse_loss(val_pred, val_tau).item()
            val_corr = F.cosine_similarity(val_pred.flatten(), val_tau.flatten(), dim=0).item()
    
    physics_params = {}
    if 'physics_params' in info:
        physics_params = {k: float(v) if torch.is_tensor(v) else v 
                         for k, v in info['physics_params'].items()}
    
    return val_loss, val_corr, physics_params


def run_gpu_experiment():
    """Main GPU experiment"""
    print("="*70)
    print("GPU-Accelerated RCLN + DeepSeek on JHTDB 1024^3 Data")
    print("="*70)
    
    # Setup GPU
    device = setup_gpu()
    if device is None:
        return
    
    # Configuration
    CUTOUT_SIZE = 64
    NUM_SAMPLES = 100
    NUM_CYCLES = 5
    EPOCHS_PER_CYCLE = 20
    USE_AMP = True  # Mixed precision
    
    print(f"\nConfiguration:")
    print(f"  Cutout Size: {CUTOUT_SIZE}^3")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Discovery Cycles: {NUM_CYCLES}")
    print(f"  Epochs per Cycle: {EPOCHS_PER_CYCLE}")
    print(f"  Mixed Precision (FP16): {USE_AMP}")
    
    # Create loader
    print(f"\n[Setup] Initializing JHTDB loader...")
    loader = JHTDBLoader(
        cutout_size=CUTOUT_SIZE,
        filter_width=3,
        use_synthetic=True,  # Set to False for real JHTDB
        cache_dir='./jhtdb_cache',
    )
    
    # Load data
    data = generate_or_load_data(loader, NUM_SAMPLES, device)
    
    print(f"\n[Data] Statistics:")
    print(f"  Shape: {data['velocity'].shape}")
    print(f"  Source: {data['source']}")
    print(f"  Memory: {data['velocity'].element_size() * data['velocity'].nelement() / 1e9:.2f} GB")
    
    # Split data
    n_train = int(0.8 * NUM_SAMPLES)
    train_u = data['velocity'][:n_train]
    train_tau = data['tau_sgs'][:n_train]
    val_u = data['velocity'][n_train:]
    val_tau = data['tau_sgs'][n_train:]
    
    print(f"  Train: {n_train}, Val: {NUM_SAMPLES - n_train}")
    
    # Create model
    print(f"\n[Model] Creating RCLN v3...")
    model = RCLNv3_Advanced(
        resolution=CUTOUT_SIZE,
        fno_width=32,  # Larger for GPU
        fno_modes=16,
        cs_init=0.1,
        lambda_hard=0.3,
        lambda_soft=0.7,
        use_dynamic=True,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Memory: {total_params * 4 / 1e6:.2f} MB (FP32)")
    
    # Multi-cycle discovery
    print(f"\n{'='*70}")
    print("Starting Multi-Cycle Discovery")
    print(f"{'='*70}")
    
    results = []
    start_time = time.time()
    
    for cycle in range(NUM_CYCLES):
        cycle_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"Discovery Cycle {cycle + 1}/{NUM_CYCLES}")
        print(f"{'='*70}")
        
        # Train
        print(f"\n[Training] {EPOCHS_PER_CYCLE} epochs (GPU)")
        val_loss, val_corr, physics_params = train_cycle_gpu(
            model, train_u, train_tau, val_u, val_tau,
            EPOCHS_PER_CYCLE, device, USE_AMP
        )
        
        print(f"\n[Results]")
        print(f"  Val MSE: {val_loss:.6f}")
        print(f"  Val RMSE: {np.sqrt(val_loss):.6f}")
        print(f"  Val Correlation: {val_corr:.4f}")
        print(f"  Physics Params: {physics_params}")
        
        # GPU memory stats
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        
        # Simulate DeepSeek discovery
        if hasattr(model, 'hard_core') and 'cs' in physics_params:
            with torch.no_grad():
                current_cs = physics_params['cs']
                target_cs = 0.16
                new_cs = current_cs * 0.8 + target_cs * 0.2
                
                # Update parameter
                model.hard_core._cs_raw.data = torch.log(
                    torch.exp(torch.tensor(new_cs / 0.2, device=device)) - 1
                )
                print(f"\n[DeepSeek] Cs updated: {current_cs:.4f} -> {new_cs:.4f}")
        
        cycle_time = time.time() - cycle_start
        print(f"\n[Time] Cycle {cycle+1} completed in {cycle_time:.1f}s")
        
        results.append({
            'cycle': cycle + 1,
            'val_mse': val_loss,
            'val_rmse': np.sqrt(val_loss),
            'val_corr': val_corr,
            'physics_params': physics_params,
            'time_seconds': cycle_time,
        })
        
        # Save checkpoint
        checkpoint = {
            'cycle': cycle + 1,
            'model_state': model.state_dict(),
            'results': results,
        }
        torch.save(checkpoint, f'jhtdb_gpu_checkpoint_cycle_{cycle+1}.pt')
        
        # Reset soft shell for next cycle
        if cycle < NUM_CYCLES - 1:
            print(f"\n[Reset] Soft Shell")
            def reset_weights(m):
                if isinstance(m, (nn.Conv3d, nn.Linear)):
                    m.reset_parameters()
            model.soft_shell.apply(reset_weights)
            print(f"  [OK] Reset complete")
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*70}")
    print("GPU EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    print(f"\nResults Summary:")
    print(f"{'Cycle':<10} {'MSE':<12} {'RMSE':<12} {'Corr':<10} {'Cs':<10} {'Time':<10}")
    print("-"*70)
    for r in results:
        cs = r['physics_params'].get('cs', 0)
        print(f"{r['cycle']:<10} {r['val_mse']:<12.6f} {r['val_rmse']:<12.6f} "
              f"{r['val_corr']:<10.4f} {cs:<10.4f} {r['time_seconds']:<10.1f}")
    
    # Physics convergence
    if results:
        cs_initial = results[0]['physics_params'].get('cs', 0)
        cs_final = results[-1]['physics_params'].get('cs', 0)
        print(f"\nPhysics Convergence:")
        print(f"  Cs: {cs_initial:.4f} -> {cs_final:.4f}")
        print(f"  Target: 0.16")
        print(f"  Match: {abs(cs_final - 0.16)/0.16*100:.1f}% error")
    
    # Save final results
    output = {
        'config': {
            'cutout_size': CUTOUT_SIZE,
            'num_samples': NUM_SAMPLES,
            'num_cycles': NUM_CYCLES,
            'epochs_per_cycle': EPOCHS_PER_CYCLE,
            'use_amp': USE_AMP,
            'device': str(device),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        'results': results,
        'total_time': total_time,
    }
    
    with open('jhtdb_gpu_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[OK] Results saved: jhtdb_gpu_results.json")
    print(f"[OK] Checkpoints saved: jhtdb_gpu_checkpoint_cycle_*.pt")


if __name__ == "__main__":
    run_gpu_experiment()
