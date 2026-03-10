# -*- coding: utf-8 -*-
"""
RCLN v1.0 vs v2.0 Coupling Comparison (Simplified)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("RCLN v1.0 vs v2.0 Coupling Comparison")
print("=" * 70)

from axiom_os.layers.rcln_v2_generic import RCLNv2_GENERIC


class SimpleAdditionRCLN(nn.Module):
    """RCLN v1.0 - Simple addition"""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.spring_k = nn.Parameter(torch.tensor(1.0))
        self.soft_shell = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.lambda_res = 0.5
    
    def hard_core(self, z):
        n = z.shape[-1] // 2
        q, p = z[..., :n], z[..., n:]
        return torch.cat([p, -self.spring_k * q], dim=-1)
    
    def forward(self, z):
        return self.hard_core(z) + self.lambda_res * self.soft_shell(z)


def simulate(model, z0, dt=0.01, n_steps=500):
    """Simulate dynamics"""
    trajectory = [z0.clone()]
    z = z0.clone()
    
    for _ in range(n_steps):
        if hasattr(model, 'step'):
            z = model.step(z, dt, method='rk4')
        else:
            z_dot = model(z)
            z = z + dt * z_dot
        trajectory.append(z.clone())
    
    return torch.stack(trajectory, dim=0)


def compute_energy(trajectory):
    """Compute total energy E = 0.5 * (q^2 + p^2)"""
    n = trajectory.shape[-1] // 2
    q = trajectory[..., :n]
    p = trajectory[..., n:]
    return 0.5 * (q**2 + p**2).sum(dim=-1)


# Setup
state_dim = 4
hidden_dim = 32
dt = 0.01
n_steps = 500
z0 = torch.tensor([[1.0, 0.5, 0.0, 0.0]])

print("\n[Setup]")
print(f"  State dim: {state_dim}, Hidden dim: {hidden_dim}")
print(f"  dt: {dt}, Steps: {n_steps}")

print("\n" + "=" * 70)
print("Experiment 1: RCLN v1.0 (Addition: y = y_hard + lambda * y_soft)")
print("=" * 70)

model_v1 = SimpleAdditionRCLN(state_dim, hidden_dim)
traj_v1 = simulate(model_v1, z0, dt, n_steps)
energy_v1 = compute_energy(traj_v1)

print(f"Initial energy: {energy_v1[0].item():.4f}")
print(f"Final energy:   {energy_v1[-1].item():.4f}")
print(f"Energy drift:   {energy_v1.max().item() - energy_v1.min().item():.6f}")
print(f"Parameters:     {sum(p.numel() for p in model_v1.parameters()):,}")

print("\n" + "=" * 70)
print("Experiment 2: RCLN v2.0 (GENERIC: z_dot = L * grad_E + M * grad_S)")
print("=" * 70)

model_v2 = RCLNv2_GENERIC(state_dim, hidden_dim)
traj_v2 = simulate(model_v2, z0, dt, n_steps)
energy_v2 = compute_energy(traj_v2)

print(f"Initial energy: {energy_v2[0].item():.4f}")
print(f"Final energy:   {energy_v2[-1].item():.4f}")
print(f"Energy drift:   {energy_v2.max().item() - energy_v2.min().item():.6f}")
print(f"Parameters:     {sum(p.numel() for p in model_v2.parameters()):,}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"\n{'Metric':<25} {'v1.0 (Add)':<15} {'v2.0 (GENERIC)':<15}")
print("-" * 55)
print(f"{'Energy Drift':<25} {energy_v1.max().item() - energy_v1.min().item():<15.6f} {energy_v2.max().item() - energy_v2.min().item():<15.6f}")
print(f"{'Final Energy':<25} {energy_v1[-1].item():<15.4f} {energy_v2[-1].item():<15.4f}")
print(f"{'Parameters':<25} {sum(p.numel() for p in model_v1.parameters()):<15,} {sum(p.numel() for p in model_v2.parameters()):<15,}")

print("\n" + "=" * 70)
print("Key Differences")
print("=" * 70)

print("\nv1.0 Addition Coupling:")
print("  Equation:     y = F_hard + lambda * F_soft")
print("  Soft Output:  Vector (same as hard)")
print("  Physics:      Loose coupling")
print("  Guarantees:   None")

print("\nv2.0 GENERIC Coupling:")
print("  Equation:     z_dot = L(z) * grad_E(z) + M(z) * grad_S(z)")
print("  Soft Output:  S(z) [entropy] + M(z) [friction matrix]")
print("  Physics:      Structured coupling")
print("  Guarantees:   Energy conservation, Entropy production >= 0")

print("\nSignificance:")
print("  This is NOT just changing network architecture.")
print("  This is changing the physical WORLDVIEW.")
print("  The Soft Shell is FORCED to respect thermodynamic laws.")
