#!/usr/bin/env python3
"""
LES SGS Benchmark: Axiom-OS TBNN vs Smagorinsky on JHTDB
=========================================================

Flow:
1. Load JHTDB DNS velocity (fine grid)
2. Coarsen to simulate LES grid
3. Compute SGS stress tensor (target)
4. Train TBNN (Axiom-OS) vs Smagorinsky (baseline)
5. Compare: MSE, R2, correlation
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.datasets.jhtdb_turbulence import load_jhtdb_for_les_sgs
from axiom_os.core.turbulence_invariants import extract_invariants_and_basis_normalized
from axiom_os.layers.tbnn import TBNN, stack_tensor_basis


def smagorinsky_model(u_coarse: np.ndarray, Cs: float = 0.17) -> np.ndarray:
    """
    Smagorinsky SGS model: tau_ij = -2 * (Cs * Delta)^2 * |S| * S_ij
    
    Args:
        u_coarse: (nx, ny, nz, 3) coarse velocity
        Cs: Smagorinsky constant (default 0.17)
    
    Returns:
        tau_smag: (nx, ny, nz, 3, 3) SGS stress tensor
    """
    from axiom_os.core.turbulence_invariants import grad_u_from_velocity
    
    # Compute gradient
    grad_u = grad_u_from_velocity(u_coarse)  # (nx, ny, nz, 3, 3)
    
    # Strain rate S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    S = 0.5 * (grad_u + np.swapaxes(grad_u, -2, -1))
    
    # |S| = sqrt(2 * S_ij * S_ij)
    S_mag = np.sqrt(2 * np.sum(S * S, axis=(-2, -1), keepdims=True))
    
    # Filter width (assume uniform grid)
    Delta = 1.0  # normalized
    
    # Eddy viscosity
    nu_sgs = (Cs * Delta) ** 2 * S_mag
    
    # tau_ij = -2 * nu_sgs * S_ij
    tau_smag = -2 * nu_sgs * S
    
    return tau_smag


def compute_error_metrics(tau_pred: np.ndarray, tau_true: np.ndarray) -> dict:
    """Compute MSE, RMSE, R2, correlation"""
    diff = tau_pred - tau_true
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    
    # R2
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((tau_true - np.mean(tau_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Pearson correlation
    pred_flat = tau_pred.ravel()
    true_flat = tau_true.ravel()
    corr = np.corrcoef(pred_flat, true_flat)[0, 1]
    
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "correlation": corr,
    }


def train_tbnn(u_coarse, tau_target, n_epochs=500):
    """Train TBNN model"""
    import torch
    
    # Extract features
    inv, basis, sigma = extract_invariants_and_basis_normalized(u_coarse)
    inv_flat = inv.reshape(-1, 5).astype(np.float32)
    tb_stacked = stack_tensor_basis(basis).numpy()
    tb_flat = tb_stacked.reshape(-1, 10, 3, 3)
    sigma_flat = sigma.reshape(-1).astype(np.float32)
    tau_flat = tau_target.reshape(-1, 9).astype(np.float32)
    
    # Split train/test (spatial)
    n_total = len(inv_flat)
    n_train = int(0.7 * n_total)
    
    # Shuffle indices
    np.random.seed(42)
    idx = np.random.permutation(n_total)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    
    X_train = torch.from_numpy(inv_flat[train_idx])
    tb_train = torch.from_numpy(tb_flat[train_idx].astype(np.float32))
    sigma_train = torch.from_numpy(sigma_flat[train_idx])
    y_train = torch.from_numpy(tau_flat[train_idx])
    
    X_test = torch.from_numpy(inv_flat[test_idx])
    tb_test = torch.from_numpy(tb_flat[test_idx].astype(np.float32))
    sigma_test = torch.from_numpy(sigma_flat[test_idx])
    y_test = torch.from_numpy(tau_flat[test_idx])
    
    # Model
    model = TBNN(n_invariants=5, n_tensors=10, hidden=64, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50, factor=0.5)
    
    print(f"  Training TBNN: {n_epochs} epochs...")
    losses = []
    start = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        tau_pred = model(X_train, tb_train, sigma_train)
        loss = ((tau_pred.reshape(-1, 9) - y_train) ** 2).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step(loss.item())
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}: loss={loss.item():.6f}")
    
    train_time = time.time() - start
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        tau_train_pred = model(X_train, tb_train, sigma_train).reshape(-1, 9).numpy()
        tau_test_pred = model(X_test, tb_test, sigma_test).reshape(-1, 9).numpy()
    
    train_metrics = compute_error_metrics(tau_train_pred, y_train.numpy())
    test_metrics = compute_error_metrics(tau_test_pred, y_test.numpy())
    
    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_time": train_time,
        "losses": losses,
    }


def run_benchmark():
    """Main benchmark: TBNN vs Smagorinsky"""
    print("="*70)
    print("LES SGS Benchmark: Axiom-OS TBNN vs Smagorinsky")
    print("Data: JHTDB Isotropic Turbulence (DNS)")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading JHTDB data...")
    try:
        u_fine, u_coarse, tau_target, meta = load_jhtdb_for_les_sgs(
            fine_size=16, coarse_ratio=2, dataset="isotropic8192", timepoint=1
        )
        print(f"  Fine: {u_fine.shape}, Coarse: {u_coarse.shape}")
        print(f"  Tau target: {tau_target.shape}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Note: JHTDB requires internet connection and test token (4096 points max)")
        return
    
    # Smagorinsky baseline
    print("\n[2/5] Running Smagorinsky model...")
    tau_smag = smagorinsky_model(u_coarse, Cs=0.17)
    smag_metrics = compute_error_metrics(tau_smag, tau_target)
    print(f"  Smagorinsky: MSE={smag_metrics['mse']:.6f}, R2={smag_metrics['r2']:.4f}, "
          f"Corr={smag_metrics['correlation']:.4f}")
    
    # Train TBNN
    print("\n[3/5] Training TBNN (Axiom-OS)...")
    tbnn_results = train_tbnn(u_coarse, tau_target, n_epochs=500)
    tbnn_metrics = tbnn_results["test_metrics"]
    print(f"  TBNN: MSE={tbnn_metrics['mse']:.6f}, R2={tbnn_metrics['r2']:.4f}, "
          f"Corr={tbnn_metrics['correlation']:.4f}")
    print(f"  Training time: {tbnn_results['train_time']:.2f}s")
    
    # Compare
    print("\n[4/5] Comparison:")
    print(f"  MSE improvement: {smag_metrics['mse']/tbnn_metrics['mse']:.2f}x better")
    print(f"  R2 improvement: {tbnn_metrics['r2'] - smag_metrics['r2']:+.4f}")
    print(f"  Correlation improvement: {tbnn_metrics['correlation'] - smag_metrics['correlation']:+.4f}")
    
    # Plot
    print("\n[5/5] Generating plots...")
    import torch
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training curve
    ax = axes[0, 0]
    ax.plot(tbnn_results["losses"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title("TBNN Training Curve")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    # Metrics comparison
    ax = axes[0, 1]
    models = ["Smagorinsky", "TBNN"]
    mse_vals = [smag_metrics["mse"], tbnn_metrics["mse"]]
    r2_vals = [smag_metrics["r2"], tbnn_metrics["r2"]]
    
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, mse_vals, width, label="MSE", color="red", alpha=0.7)
    ax.set_ylabel("MSE", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    
    ax2 = ax.twinx()
    ax2.bar(x + width/2, r2_vals, width, label="R2", color="blue", alpha=0.7)
    ax2.set_ylabel("R²", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    
    # Scatter plot (TBNN)
    ax = axes[1, 0]
    # Sample points for visualization
    tau_true_flat = tau_target.ravel()
    with torch.no_grad():
        import torch
        inv, basis, sigma = extract_invariants_and_basis_normalized(u_coarse)
        inv_flat = inv.reshape(-1, 5).astype(np.float32)
        tb_stacked = stack_tensor_basis(basis).numpy()
        tb_flat = tb_stacked.reshape(-1, 10, 3, 3)
        sigma_flat = sigma.reshape(-1).astype(np.float32)
        
        X_all = torch.from_numpy(inv_flat)
        tb_all = torch.from_numpy(tb_flat.astype(np.float32))
        sigma_all = torch.from_numpy(sigma_flat)
        
        model = tbnn_results["model"]
        model.eval()
        tau_pred_all = model(X_all, tb_all, sigma_all).reshape(-1, 9).numpy()
    
    sample_idx = np.random.choice(len(tau_true_flat), 1000, replace=False)
    ax.scatter(tau_true_flat[sample_idx], tau_pred_all.ravel()[sample_idx], alpha=0.5, s=10)
    ax.plot([tau_true_flat.min(), tau_true_flat.max()], 
            [tau_true_flat.min(), tau_true_flat.max()], 'r--', label="Perfect")
    ax.set_xlabel("True SGS Stress")
    ax.set_ylabel("Predicted SGS Stress")
    ax.set_title(f"TBNN Prediction (R²={tbnn_metrics['r2']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scatter plot (Smagorinsky)
    ax = axes[1, 1]
    ax.scatter(tau_true_flat[sample_idx], tau_smag.ravel()[sample_idx], alpha=0.5, s=10, color="orange")
    ax.plot([tau_true_flat.min(), tau_true_flat.max()], 
            [tau_true_flat.min(), tau_true_flat.max()], 'r--', label="Perfect")
    ax.set_xlabel("True SGS Stress")
    ax.set_ylabel("Predicted SGS Stress")
    ax.set_title(f"Smagorinsky Prediction (R²={smag_metrics['r2']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = ROOT / "docs/images/les_sgs_benchmark.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to {output_path}")
    
    # Save results
    results = {
        "smagorinsky": smag_metrics,
        "tbnn": tbnn_metrics,
        "improvement": {
            "mse_ratio": smag_metrics["mse"] / tbnn_metrics["mse"],
            "r2_delta": tbnn_metrics["r2"] - smag_metrics["r2"],
            "corr_delta": tbnn_metrics["correlation"] - smag_metrics["correlation"],
        },
        "metadata": {
            "data_shape": list(u_coarse.shape),
            "n_train": int(0.7 * len(inv_flat)),
            "n_test": int(0.3 * len(inv_flat)),
        }
    }
    
    import json
    with open(ROOT / "experiments/les_sgs_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to experiments/les_sgs_results.json")
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Smagorinsky: R²={smag_metrics['r2']:.4f}, Corr={smag_metrics['correlation']:.4f}")
    print(f"TBNN:        R²={tbnn_metrics['r2']:.4f}, Corr={tbnn_metrics['correlation']:.4f}")
    print(f"Improvement: {results['improvement']['mse_ratio']:.2f}x MSE reduction")
    print("="*70)


if __name__ == "__main__":
    run_benchmark()
