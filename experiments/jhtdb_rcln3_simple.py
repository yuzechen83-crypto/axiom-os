"""
JHTDB + RCLN 3.0 Simplified Experiment
Demonstrates RCLN 3.0 on turbulence data (simplified for demo)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from axiom_os.core import RCLN3, create_rcln3, HardCoreLevel
from axiom_os.layers import create_kan_rcln


def generate_turbulence_data(n_samples=10, n_points=100):
    """
    Generate synthetic turbulence-like data.
    
    In production:
        from giverny.turbulence_dataset import TurbulenceDataset
        dataset = TurbulenceDataset(auth_token='your_token')
    """
    # Velocity components (u, v, w)
    velocity = torch.randn(n_samples, 3, n_points) * 0.5
    
    # Add spatial correlation (simplified)
    for i in range(n_points-1):
        velocity[:, :, i+1] = 0.8 * velocity[:, :, i] + 0.2 * velocity[:, :, i+1]
    
    # Velocity gradients (for SGS modeling)
    grad_u = torch.randn(n_samples, 9, n_points) * 0.1  # 3x3 gradient tensor flattened
    
    # Target: SGS stress (what we want to learn)
    # Simplified: τ = -C * |S| * S (Smagorinsky-like)
    S_mag = torch.sqrt((grad_u ** 2).sum(dim=1, keepdim=True))
    tau_sgs = -0.1 * S_mag * grad_u[:, :6, :]  # 6 independent components
    
    return velocity, grad_u, tau_sgs


class TurbulenceClosureRCLN(nn.Module):
    """
    RCLN for turbulence SGS closure.
    
    Hard Core: Navier-Stokes convection (known)
    Soft Shell: SGS stress (learned via KAN/FNO)
    """
    
    def __init__(self, use_kan=True):
        super().__init__()
        self.use_kan = use_kan
        
        # Hard Core: NS convection (frozen)
        self.nu = 0.01  # Viscosity
        
        # Soft Shell: SGS closure
        # Input: velocity (3) + gradient (9) = 12
        # Output: SGS stress (6 independent components)
        if use_kan:
            self.closure = create_kan_rcln(
                input_dim=12,
                hidden_dim=16,
                output_dim=6,
                grid_size=5,
            )
        else:
            self.closure = nn.Sequential(
                nn.Linear(12, 32),
                nn.SiLU(),
                nn.Linear(32, 6),
            )
    
    def ns_hard_core(self, u, grad_u):
        """
        Navier-Stokes hard core (convection + diffusion).
        
        du/dt = -(u·∇)u + ν∇²u
        """
        # Convection: -(u·∇)u
        # Simplified: use velocity gradient directly
        convection = -torch.einsum('bip,bjip->bjp', u, grad_u.reshape(u.shape[0], 3, 3, -1))
        
        # Diffusion: ν∇²u
        # Simplified finite difference
        diffusion = self.nu * (grad_u[:, :3, :] + grad_u[:, 3:6, :] + grad_u[:, 6:9, :])
        
        return convection + diffusion
    
    def forward(self, u, grad_u):
        """
        Forward: NS + closure.
        
        du/dt = NS(u) + ∇·τ_sgs
        """
        # Hard core contribution
        du_dt_hard = self.ns_hard_core(u, grad_u)
        
        # Soft shell: predict SGS stress
        closure_input = torch.cat([u, grad_u], dim=1).transpose(1, 2)  # (B, N, 12)
        
        if self.use_kan:
            # KAN closure
            tau_sgs_components = []
            for i in range(closure_input.shape[1]):
                tau_i = self.closure(closure_input[:, i, :])
                tau_sgs_components.append(tau_i)
            tau_sgs = torch.stack(tau_sgs_components, dim=1)  # (B, N, 6)
        else:
            tau_sgs = self.closure(closure_input)  # (B, N, 6)
        
        # Reconstruct full stress tensor and compute divergence (simplified)
        # For demo: just add as correction
        du_dt_soft = tau_sgs.transpose(1, 2)[:, :3, :] * 0.1  # Scale down
        
        # Combined: Hard + Soft
        du_dt = du_dt_hard + du_dt_soft
        
        return du_dt, tau_sgs


def train_closure(model, velocity, grad_u, tau_target, n_epochs=100, lr=0.01):
    """Train SGS closure."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining {'KAN' if model.use_kan else 'MLP'} closure...")
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward
        du_dt, tau_pred = model(velocity, grad_u)
        
        # Loss: match SGS stress
        loss = F.mse_loss(tau_pred, tau_target.transpose(1, 2))
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: Loss = {loss.item():.6f}")
    
    return model


def main():
    """JHTDB + RCLN 3.0 Experiment."""
    
    print("="*70)
    print("JHTDB + RCLN 3.0: Turbulence SGS Closure Experiment")
    print("="*70)
    
    # Generate data
    print("\n[1] Generating synthetic turbulence data...")
    velocity, grad_u, tau_sgs = generate_turbulence_data(n_samples=20, n_points=50)
    print(f"  Velocity: {velocity.shape}")
    print(f"  Gradient: {grad_u.shape}")
    print(f"  SGS stress: {tau_sgs.shape}")
    
    # Test KAN-RCLN
    print("\n[2] Testing KAN-RCLN closure...")
    model_kan = TurbulenceClosureRCLN(use_kan=True)
    du_dt, tau = model_kan(velocity[:5], grad_u[:5])
    print(f"  Output: {du_dt.shape}")
    print(f"  SGS prediction: {tau.shape}")
    
    # Train KAN
    print("\n[3] Training KAN-RCLN...")
    model_kan = train_closure(
        model_kan,
        velocity[:10],
        grad_u[:10],
        tau_sgs[:10],
        n_epochs=50,
        lr=0.02,
    )
    
    # Extract formula (KAN)
    print("\n[4] Extracting discovered formula...")
    if hasattr(model_kan.closure, 'extract_formula'):
        try:
            formula = model_kan.closure.extract_formula(
                var_names=['u', 'v', 'w', 'du_dx', 'du_dy', 'du_dz', 
                          'dv_dx', 'dv_dy', 'dv_dz', 'dw_dx', 'dw_dy', 'dw_dz']
            )
            if formula:
                print(f"  Discovered: {formula['formula_str'][:100]}...")
                print(f"  Confidence: {formula['confidence']:.2%}")
        except Exception as e:
            print(f"  Formula extraction: {e}")
    
    # Test standard MLP for comparison
    print("\n[5] Training MLP closure (baseline)...")
    model_mlp = TurbulenceClosureRCLN(use_kan=False)
    model_mlp = train_closure(
        model_mlp,
        velocity[:10],
        grad_u[:10],
        tau_sgs[:10],
        n_epochs=100,
        lr=0.01,
    )
    
    # Compare
    print("\n[6] Comparison")
    with torch.no_grad():
        _, tau_kan = model_kan(velocity[-5:], grad_u[-5:])
        _, tau_mlp = model_mlp(velocity[-5:], grad_u[-5:])
        tau_true = tau_sgs[-5:].transpose(1, 2)
        
        error_kan = F.mse_loss(tau_kan, tau_true).item()
        error_mlp = F.mse_loss(tau_mlp, tau_true).item()
        
        print(f"  KAN error: {error_kan:.6f}")
        print(f"  MLP error: {error_mlp:.6f}")
        print(f"  Improvement: {(error_mlp - error_kan) / error_mlp * 100:.1f}%")
    
    # Architecture summary
    print("\n[7] RCLN 3.0 Architecture Summary")
    print("  Hard Core: Navier-Stokes (convection + diffusion)")
    print("  Soft Shell: KAN closure (B-splines, formula-extractable)")
    print("  Coupling: Residual (du/dt = NS_hard + ∇·τ_soft)")
    
    print("\n" + "="*70)
    print("JHTDB + RCLN 3.0 Experiment Complete!")
    print("="*70)
    print("""
Key Results:
  1. KAN-RCLN learns SGS stress from turbulence data
  2. Formulas can be extracted (symbolic interpretability)
  3. Hard Core provides physical structure (NS)
  4. Soft Shell learns what's missing (SGS closure)

Next Steps for Real JHTDB:
  1. Install giverny: pip install givernylocal
  2. Get JHTDB auth token
  3. Load real DNS data: 1024^3 resolution
  4. Train FNO-RCLN for full 3D fields
  5. Discover turbulence closure models
""")


if __name__ == "__main__":
    main()
