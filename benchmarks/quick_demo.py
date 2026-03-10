"""Quick demo using Axiom-OS native environment (faster)"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from axiom_os.core.upi import UPIState
from axiom_os.orchestrator.mpc import (
    ImaginationMPC,
    double_pendulum_H,
    angle_normalize,
)

# Use the native AcrobotEnv from demo_acrobot
from axiom_os.demo_acrobot import AcrobotEnv

PI = np.pi

print('=' * 70)
print('Axiom-OS Physics-Informed Control Demo')
print('Environment: Custom Acrobot (Native Physics)')
print('Controller: ImaginationMPC + Hamiltonian Dynamics')
print('=' * 70)

# Setup
H = double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
env = AcrobotEnv(H=H, dt=0.02, friction=0.1, noise_std=0.005, seed=42)

# MPC with moderate samples for speed
mpc = ImaginationMPC(
    H=H,
    horizon_steps=40,
    n_samples=100,  # Reduced for speed
    dt=0.02,
    friction=0.1,
    action_std=2.0,
    target_state=np.array([PI, PI]),
    distance_threshold=0.5,
)

# Reset to near upright
obs, _ = env.reset(q=np.array([PI - 0.3, PI - 0.2]), p=np.array([0.0, 0.0]))
q, p = obs[:2], obs[2:4]

print(f'\nInitial state: q1={q[0]:.3f}, q2={q[1]:.3f} (target: {PI:.3f})')
print(f'Initial error: {np.sqrt(np.sum(angle_normalize(q - PI) ** 2)):.3f} rad')
print('\nRunning control loop...')

# Run simulation
t_max = 150
history = {"t": [], "q1": [], "q2": [], "action": [], "err": [], "energy": []}

for t in range(t_max):
    # 1. Observe -> UPIState
    upi_state = UPIState(
        values=torch.tensor(obs, dtype=torch.float64),
        units=[0, 0, 0, 0, 0],
        semantics="AcrobotState",
    )
    
    # 2. Think (MPC planning)
    action = mpc.plan(q, p)
    
    # 3. Act
    obs, reward, done, truncated, info = env.step(action)
    q, p = obs[:2], obs[2:4]
    
    # Compute energy
    H_val = H(np.concatenate([q, p]))
    err = np.sqrt(np.sum(angle_normalize(q - PI) ** 2))
    
    # Record
    history["t"].append(t * env.dt)
    history["q1"].append(q[0])
    history["q2"].append(q[1])
    history["action"].append(action)
    history["err"].append(err)
    history["energy"].append(H_val)
    
    if (t + 1) % 30 == 0:
        print(f"  t={t+1:3d}: q1={q[0]:.3f}, q2={q[1]:.3f}, err={err:.3f}, E={H_val:.2f}")

# Results
print('\n' + '=' * 70)
print('Results')
print('=' * 70)
err_final = np.mean(history["err"][-30:])
print(f'Final avg error (last 30 steps): {err_final:.3f} rad')
print(f'Initial error: {history["err"][0]:.3f} rad')
print(f'Error reduction: {(1 - err_final/history["err"][0])*100:.1f}%')

stabilized = err_final < 0.5
print(f'Stabilized: {stabilized}')

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Joint angles
ax = axes[0]
ax.plot(history["t"], history["q1"], 'b-', label=r'$\theta_1$')
ax.plot(history["t"], history["q2"], 'g-', label=r'$\theta_2$')
ax.axhline(PI, color='k', linestyle='--', alpha=0.5, label='Target')
ax.set_ylabel('Angle (rad)')
ax.set_title('Axiom-OS Acrobot: Physics-Informed MPC Control')
ax.legend()
ax.grid(True, alpha=0.3)

# Error
ax = axes[1]
ax.plot(history["t"], history["err"], 'r-', label='Error')
ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Stabilization threshold')
ax.set_ylabel('Error (rad)')
ax.legend()
ax.grid(True, alpha=0.3)

# Control
ax = axes[2]
ax.plot(history["t"], history["action"], 'purple', label='Torque')
ax.set_ylabel('Torque (N·m)')
ax.legend()
ax.grid(True, alpha=0.3)

# Energy
ax = axes[3]
ax.plot(history["t"], history["energy"], 'orange', label='Total Energy')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy (J)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = Path(__file__).parent / 'quick_demo_result.png'
plt.savefig(plot_path, dpi=150)
print(f'\nPlot saved to: {plot_path}')

# Also save data
import json
data_path = Path(__file__).parent / 'quick_demo_data.json'
with open(data_path, 'w') as f:
    json.dump({
        'stabilized': bool(stabilized),
        'final_error': float(err_final),
        'initial_error': float(history["err"][0]),
        'total_steps': int(t_max),
        'dt': float(env.dt),
    }, f, indent=2)
print(f'Data saved to: {data_path}')

print('=' * 70)
