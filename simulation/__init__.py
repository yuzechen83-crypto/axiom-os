"""
Axiom-OS Simulation Module

This module provides integration between Axiom-OS and NVIDIA Isaac Sim
for high-fidelity robot simulation and Sim-to-Real verification.

Components:
    - IsaacGo1Env: Environment wrapper for Unitree Go1 quadruped
    - ChaosInjector: Domain randomization for robustness testing
    - DiscoveryLogger: Adaptive learning hook

Usage:
    from simulation.isaac_env import IsaacGo1Env, Go1Config
    
    env = IsaacGo1Env(headless=False)
    obs = env.reset()
    
    for _ in range(1000):
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)
"""

from .isaac_env import (
    IsaacGo1Env,
    Go1Config,
    ChaosInjector,
    IS_ISAAC_AVAILABLE,
)

__all__ = [
    'IsaacGo1Env',
    'Go1Config',
    'ChaosInjector',
    'IS_ISAAC_AVAILABLE',
]
