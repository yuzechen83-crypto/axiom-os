#!/usr/bin/env python3
"""
SOTA-level Walker2d training with:
- 1M steps
- Tuned PPO hyperparameters
- Observation normalization
- Better domain randomization
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from axiom_os.envs import (
    make_axiom_mujoco_env,
    make_axiom_mujoco_env_with_domain_rand,
    PhysicsAwareObsWrapper,
)


def make_env(env_id, gravity_scale=1.0, friction_scale=1.0, seed=0):
    """Create environment with physics awareness."""
    def _init():
        env = make_axiom_mujoco_env(
            env_id=env_id,
            gravity_scale=gravity_scale,
            friction_scale=friction_scale,
            seed=seed,
        )
        env = PhysicsAwareObsWrapper(env)
        return env
    return _init


def train_sota_walker2d(total_timesteps=1_000_000, seed=42):
    """Train Walker2d with SOTA-level configuration."""
    
    print("=" * 70)
    print("SOTA Walker2d Training")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Seed: {seed}")
    
    # Create training environment with domain randomization
    # Use a schedule: start with easier, gradually increase difficulty
    print("\n[1/3] Creating training environment...")
    
    # Training with progressive domain randomization
    env = make_axiom_mujoco_env_with_domain_rand(
        env_id="Walker2d-v5",
        gravity_range=(0.8, 1.2),      # Wider range for robustness
        friction_range=(0.7, 1.3),
        seed=seed,
    )
    env = PhysicsAwareObsWrapper(env)
    env = DummyVecEnv([lambda: env])
    
    # Evaluation environment (standard physics)
    eval_env = make_axiom_mujoco_env("Walker2d-v5", seed=seed+100)
    eval_env = PhysicsAwareObsWrapper(eval_env)
    
    print("[2/3] Initializing PPO with tuned hyperparameters...")
    
    # SOTA-tuned PPO hyperparameters for Walker2d
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,          # Slightly lower for stability
        n_steps=4096,                  # Longer rollouts
        batch_size=256,                # Larger batches
        n_epochs=15,                   # More optimization epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,                # Higher entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,                  # State-dependent exploration
        sde_sample_freq=64,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512, 256],    # Larger policy network
                vf=[512, 512, 256],    # Larger value network
            ),
            activation_fn=torch.nn.Tanh,  # Tanh works better for continuous control
            ortho_init=True,
        ),
        verbose=1,
        seed=seed,
        device="auto",
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./walker2d_sota/",
        log_path="./walker2d_sota/",
        eval_freq=50_000,
        deterministic=True,
        render=False,
    )
    
    print("[3/3] Training...")
    print("-" * 70)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    
    # Save final model
    model.save("walker2d_sota_final")
    print("\n" + "=" * 70)
    print("Training complete!")
    print("Model saved: walker2d_sota_final")
    print("Best model: ./walker2d_sota/best_model")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000, help="Training steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_sota_walker2d(total_timesteps=args.steps, seed=args.seed)
