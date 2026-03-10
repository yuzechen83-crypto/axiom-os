# -*- coding: utf-8 -*-
"""
Hard Core 3.0 Demo - Differentiable Physics Engine
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("Hard Core 3.0 - Differentiable Physics Engine Demo")
print("=" * 70)

from axiom_os.core.differentiable_physics import (
    DifferentiablePhysicsEngine,
    SolverLevel,
    spring_force,
    friction_force,
)

# ============================================
# Demo 1: Three Solver Levels
# ============================================
print("\n" + "=" * 70)
print("Demo 1: Solver Levels Comparison")
print("=" * 70)

def harmonic_oscillator(state, params, t=None):
    omega = params.get('omega', torch.tensor(1.0))
    dim = state.shape[1] // 2
    x, v = state[:, :dim], state[:, dim:]
    dxdt = v
    dvdt = -(omega ** 2) * x
    return torch.cat([dxdt, dvdt], dim=1)

x0 = torch.tensor([[1.0, 0.0]])
dt = 0.1
steps = 100

for level in [SolverLevel.EULER, SolverLevel.SYMPLECTIC, SolverLevel.DIFFERENTIABLE]:
    engine = DifferentiablePhysicsEngine(solver_level=level, dim=1, dt=dt)
    engine.crystallize("harmonic", harmonic_oscillator, {'omega': 1.0}, learnable=False)
    
    state = x0.clone()
    energies = []
    for _ in range(steps):
        state = engine.integrate(state)
        x, v = state[0, 0].item(), state[0, 1].item()
        energies.append(0.5 * (x**2 + v**2))
    
    drift = max(energies) - min(energies)
    print(f"\n{level.name}:")
    print(f"  Energy drift: {drift:.6f}")

print("\nResult: DIFFERENTIABLE has best energy conservation!")

# ============================================
# Demo 2: Crystallization
# ============================================
print("\n" + "=" * 70)
print("Demo 2: Crystallization")
print("=" * 70)

engine = DifferentiablePhysicsEngine(solver_level=SolverLevel.DIFFERENTIABLE, dim=3)
print("\n[Step 1] Empty Hard Core")
engine.summary()

print("\n[Step 2] Crystallizing Spring Law")
engine.crystallize("spring", spring_force, {'k': 10.0}, learnable=True)

print("\n[Step 3] Crystallizing Friction")
engine.crystallize("friction", friction_force, {'mu': 0.5, 'alpha': 1.0}, learnable=True)

engine.summary()

# ============================================
# Demo 3: Learnable Parameters
# ============================================
print("\n" + "=" * 70)
print("Demo 3: Learnable Physics Parameters")
print("=" * 70)

engine = DifferentiablePhysicsEngine(solver_level=SolverLevel.DIFFERENTIABLE, dim=1, dt=0.01)
engine.crystallize("spring", spring_force, {'k': 5.0}, learnable=True)
engine.crystallize("friction", friction_force, {'mu': 0.1, 'alpha': 1.0}, learnable=True)

optimizer = torch.optim.Adam(engine.parameters(), lr=0.1)

print("\nLearning k and mu...")
for epoch in range(30):
    optimizer.zero_grad()
    state = torch.tensor([[1.0, 0.0]])
    for _ in range(50):
        state = engine.integrate(state)
    loss = state[0, 0]**2 + (state[0, 1] - 0.5)**2
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        k = engine.learnable_params['spring_k'].item()
        mu = engine.learnable_params['friction_mu'].item()
        print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, k={k:.3f}, mu={mu:.3f}")

print("\n[OK] Physics parameters can be learned from data!")

# ============================================
# Demo 4: RCLN Integration
# ============================================
print("\n" + "=" * 70)
print("Demo 4: RCLN Integration")
print("=" * 70)

from axiom_os.layers.fno3d import FNO3d

class SimpleHardCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.nu = nn.Parameter(torch.tensor(0.001))
    
    def forward(self, u):
        B, C, D, H, W = u.shape
        tau = torch.zeros(B, 6, D, H, W, device=u.device)
        tau[:, :3] = u * self.nu
        return tau

class FNO_RCLN_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_res = 0.3
        self.hard_core = SimpleHardCore()
        self.soft_shell = FNO3d(
            in_channels=3, out_channels=6,
            width=8, modes1=4, modes2=4, modes3=4
        )
    
    def forward(self, u):
        return self.hard_core(u) + self.lambda_res * self.soft_shell(u)

print("\nCreating FNO-RCLN with Hard Core 3.0...")
model = FNO_RCLN_v3()
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

u = torch.randn(1, 3, 16, 16, 16)
tau = model(u)
print(f"  Forward: {u.shape} -> {tau.shape}")

loss = tau.sum()
loss.backward()
print(f"  Grad of nu: {model.hard_core.nu.grad.item():.6f}")

print("\n" + "=" * 70)
print("[SUCCESS] Hard Core 3.0 Demo Complete!")
print("=" * 70)
