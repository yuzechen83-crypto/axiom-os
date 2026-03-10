# -*- coding: utf-8 -*-
"""
Compare RCLN v1.0 vs v2.0 Coupling Methods

Comparison:
- v1.0 (Addition): y = F_hard + F_soft
- v2.0 (GENERIC): z_dot = L * grad_E + M * grad_S

Test: Harmonic Oscillator with friction
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("RCLN v1.0 vs v2.0 Coupling Comparison")
print("=" * 70)

from axiom_os.layers.rcln_v2_generic import RCLNv2_GENERIC


class SimpleAdditionRCLN(nn.Module):
    """RCLN v1.0 - Simple addition coupling"""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        # Hard Core: linear spring force
        self.spring_k = nn.Parameter(torch.tensor(1.0))
        
        # Soft Shell: friction correction
        self.soft_shell = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        self.lambda_res = 0.5
    
    def hard_core(self, z):
        """F_hard = -k * z (spring force)"""
        n = z.shape[-1] // 2
        q, p = z[..., :n], z[..., n:]
        # Hamiltonian: dq/dt = p, dp/dt = -k*q
        return torch.cat([p, -self.spring_k * q], dim=-1)
    
    def forward(self, z):
        y_hard = self.hard_core(z)
        y_soft = self.soft_shell(z)
        return y_hard + self.lambda_res * y_soft


class GENERICRCLN(nn.Module):
    """RCLN v2.0 - GENERIC coupling"""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        # Use RCLNv2_GENERIC
        self.model = RCLNv2_GENERIC(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, z):
        return self.model(z)


def simulate(model, z0, dt=0.01, n_steps=1000):
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
    """Compute total energy"""
    # E = 0.5 * (q^2 + p^2)
    n = trajectory.shape[-1] // 2
    q = trajectory[..., :n]
    p = trajectory[..., n:]
    return 0.5 * (q**2 + p**2).sum(dim=-1)


def compute_thermodynamic_consistency(trajectory, model):
    """Check thermodynamic consistency for GENERIC model"""
    if not hasattr(model, 'model'):
        return None
    
    energies = []
    entropies = []
    dE_dts = []
    dS_dts = []
    
    for i in range(len(trajectory) - 1):
        z = trajectory[i].unsqueeze(0)
        _, info = model.model(z, return_thermodynamics=True)
        energies.append(info['energy'].item())
        entropies.append(info['entropy'].item())
        dE_dts.append(info['dE_dt'].item())
        dS_dts.append(info['dS_dt'].item())
    
    return {
        'energy_drift': max(energies) - min(energies),
        'avg_dE_dt': np.mean(dE_dts),
        'avg_dS_dt': np.mean(dS_dts),
        'entropy_production': sum([x for x in dS_dts if x > 0]),
    }


print("\n[Setup]")
state_dim = 4  # 2D position + 2D momentum
hidden_dim = 32
dt = 0.01
n_steps = 500

z0 = torch.tensor([[1.0, 0.5, 0.0, 0.0]])  # Initial state

print(f"  State dim: {state_dim}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Time step: {dt}")
print(f"  Steps: {n_steps}")

print("\n" + "=" * 70)
print("Experiment 1: RCLN v1.0 (Addition Coupling)")
print("=" * 70)

model_v1 = SimpleAdditionRCLN(state_dim, hidden_dim)
traj_v1 = simulate(model_v1, z0, dt, n_steps)
energy_v1 = compute_energy(traj_v1)

print(f"Initial energy: {energy_v1[0].item():.4f}")
print(f"Final energy: {energy_v1[-1].item():.4f}")
print(f"Energy drift: {energy_v1.max().item() - energy_v1.min().item():.6f}")
print(f"Parameters: {sum(p.numel() for p in model_v1.parameters()):,}")

print("\n" + "=" * 70)
print("Experiment 2: RCLN v2.0 (GENERIC Coupling)")
print("=" * 70)

model_v2 = GENERICRCLN(state_dim, hidden_dim)
traj_v2 = simulate(model_v2, z0, dt, n_steps)
energy_v2 = compute_energy(traj_v2)
thermo_v2 = compute_thermodynamic_consistency(traj_v2, model_v2)

print(f"Initial energy: {energy_v2[0].item():.4f}")
print(f"Final energy: {energy_v2[-1].item():.4f}")
print(f"Energy drift: {energy_v2.max().item() - energy_v2.min().item():.6f}")
print(f"Parameters: {sum(p.numel() for p in model_v2.parameters()):,}")

if thermo_v2:
    print(f"\nThermodynamic Consistency:")
    print(f"  Average dE/dt: {thermo_v2['avg_dE_dt']:.6f} (should be ~0)")
    print(f"  Average dS/dt: {thermo_v2['avg_dS_dt']:.6f} (should be >= 0)")

print("\n" + "=" * 70)
print("Comparison Summary")
print("=" * 70)

print(f"\n{'Metric':<30} {'v1.0 (Add)':<15} {'v2.0 (GENERIC)':<15}")
print("-" * 60)
print(f"{'Energy Drift':<30} {energy_v1.max().item() - energy_v1.min().item():<15.6f} {energy_v2.max().item() - energy_v2.min().item():<15.6f}")
print(f"{'Final Energy':<30} {energy_v1[-1].item():<15.4f} {energy_v2[-1].item():<15.4f}")
print(f"{'Parameters':<30} {sum(p.numel() for p in model_v1.parameters()):<15,} {sum(p.numel() for p in model_v2.parameters()):<15,}")

# Plot comparison
try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory in phase space
    ax = axes[0, 0]
    ax.plot(traj_v1[:, 0, 0].numpy(), traj_v1[:, 0, 2].numpy(), 'b-', label='v1.0 Position', alpha=0.7)
    ax.plot(traj_v2[:, 0, 0].numpy(), traj_v2[:, 0, 2].numpy(), 'r-', label='v2.0 Position', alpha=0.7)
    ax.set_xlabel('q_x')
    ax.set_ylabel('p_x')
    ax.set_title('Phase Space Trajectory')
    ax.legend()
    ax.grid(True)
    
    # Energy evolution
    ax = axes[0, 1]
    t = np.arange(len(energy_v1)) * dt
    ax.plot(t, energy_v1.numpy(), 'b-', label='v1.0 Energy', alpha=0.7)
    ax.plot(t, energy_v2.numpy(), 'r-', label='v2.0 Energy', alpha=0.7)
    ax.axhline(y=energy_v1[0].item(), color='b', linestyle='--', alpha=0.3)
    ax.axhline(y=energy_v2[0].item(), color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(True)
    
    # Position over time
    ax = axes[1, 0]
    ax.plot(t, traj_v1[:, 0, 0].numpy(), 'b-', label='v1.0 q_x', alpha=0.7)
    ax.plot(t, traj_v2[:, 0, 0].numpy(), 'r-', label='v2.0 q_x', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title('Position Evolution')
    ax.legend()
    ax.grid(True)
    
    # Momentum over time
    ax = axes[1, 1]
    ax.plot(t, traj_v1[:, 0, 2].numpy(), 'b-', label='v1.0 p_x', alpha=0.7)
    ax.plot(t, traj_v2[:, 0, 2].numpy(), 'r-', label='v2.0 p_x', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Momentum')
    ax.set_title('Momentum Evolution')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('rcln_v1_vs_v2_comparison.png', dpi=150)
    print(f"\n[Plot saved to rcln_v1_vs_v2_comparison.png]")
except Exception as e:
    print(f"\n[Plotting skipped: {e}]")

print("\n" + "=" * 70)
print("[SUCCESS] Comparison Complete!")
print("=" * 70)
print("\nKey Insights:")
print("  v1.0 (Addition):")
print("    - Simple: y = y_hard + lambda * y_soft")
print("    - Loose physical meaning")
print("    - No thermodynamic guarantees")
print("\n  v2.0 (GENERIC):")
print("    - Structured: z_dot = L * grad_E + M * grad_S")
print("    - Soft Shell must output S(z) and M(z)")
print("    - Enforces energy conservation + entropy production")
print("    - This is changing the physical worldview, not just network layers")
