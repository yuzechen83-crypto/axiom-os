#!/usr/bin/env python3
"""
main_gym.py: Gym/MuJoCo benchmark with Online Identification (Discovery Engine)

- Trains a PhysicsAwarePolicy on Walker2d (and optionally Humanoid) under standard physics.
- Runs sensitivity tests: for each (gravity_scale, friction_scale), the Discovery Engine
  buffers (s, a, s'), compares HardCore.predict(s,a) vs Real(s'), and updates g and μ
  via least-squares/heuristic or gradient descent. The updated (g_scale, friction_scale)
  are passed to the MPC/Policy so the controller plans for the actual physics.
- Saves sensitivity results to JSON for robustness heatmap (utils/visualize.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _train_policy(env_id: str, episodes: int, steps_per_episode: int, seed: int):
    """Train PhysicsAwarePolicy under standard (g=1, f=1)."""
    from axiom_os.envs import make_axiom_mujoco_env, get_physics_params
    from axiom_os.envs.axiom_policy import PhysicsAwarePolicy, train_policy_reinforce

    env = make_axiom_mujoco_env(env_id=env_id, seed=seed)
    obs, _ = env.reset(seed=seed)
    policy = PhysicsAwarePolicy(obs.shape[0], env.action_space.shape[0])
    train_policy_reinforce(
        policy, env, episodes=episodes, steps_per_episode=steps_per_episode, seed=seed
    )
    env.close()
    return policy


def run_one_env_with_discovery(
    env_id: str,
    policy,
    steps: int,
    gravity_scale: float,
    friction_scale: float,
    seed: int,
    render: bool,
    use_discovery: bool,
    discovery_buffer_size: int = 100,
    discovery_update_interval: int = 20,
) -> float:
    """
    Run env; if use_discovery, Discovery Engine updates (g_est, f_est) from (s,a,s')
    and policy uses (g_est, f_est) instead of nominal (1, 1).
    """
    from axiom_os.envs import make_axiom_mujoco_env, get_physics_params
    from axiom_os.core.online_identification import DiscoveryEngine

    env = make_axiom_mujoco_env(
        env_id=env_id,
        gravity_scale=gravity_scale,
        friction_scale=friction_scale,
        seed=seed,
    )
    params = get_physics_params(env)
    obs, info = env.reset(seed=seed)

    discovery = None
    if use_discovery:
        # MuJoCo Walker2d/Humanoid obs: positions then velocities
        obs_dim = obs.shape[0]
        # Auto-detect velocity start index based on environment
        if env_id == "Walker2d-v5":
            vel_hint = 9  # 9 pos + 8 vel = 17
        elif env_id == "Humanoid-v5":
            vel_hint = None  # Let auto-detect handle it
        elif env_id == "Hopper-v5":
            vel_hint = 6  # 6 pos + 5 vel = 11
        else:
            vel_hint = obs_dim // 2
        discovery = DiscoveryEngine(
            buffer_size=discovery_buffer_size,
            predict_fn=None,  # use improved analytical g/f from (s,s') velocity/accel
            dt=0.002,
            g_nominal=1.0,
            friction_nominal=1.0,
            g_bounds=(0.2, 2.0),
            friction_bounds=(0.1, 2.0),
            update_interval=discovery_update_interval,
            vel_dim_hint=vel_hint,
        )
        print(f"[DiscoveryEngine] Initialized for {env_id}, obs_dim={obs_dim}, vel_hint={vel_hint}")

    total_reward = 0.0
    param_history = []  # Track parameter updates for diagnostics
    
    for step in range(steps):
        g_scale, f_scale = gravity_scale, friction_scale
        if discovery is not None:
            g_scale, f_scale = discovery.get_physics_scale()
        action = policy.act(
            obs,
            deterministic=True,
            gravity_scale=g_scale,
            friction_scale=f_scale,
            body_masses=params.body_masses,
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if discovery is not None:
            discovery.observe(obs, action, next_obs)
            new_g, new_f = discovery.maybe_update()
            # Log parameter changes
            if abs(new_g - g_scale) > 0.01 or abs(new_f - f_scale) > 0.01:
                param_history.append({
                    'step': step,
                    'g_scale': float(new_g),
                    'f_scale': float(new_f),
                })
        obs = next_obs
        if render:
            env.render()
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    
    # Print diagnostics
    if discovery is not None and param_history:
        print(f"    [Discovery] Parameter updates: {len(param_history)}")
        print(f"    [Discovery] Final g_scale={discovery.get_physics_scale()[0]:.3f}, "
              f"f_scale={discovery.get_physics_scale()[1]:.3f}")
    
    return total_reward


def run_walker2d_sensitivity(
    policy,
    seed: int,
    steps_per_condition: int = 200,
    use_discovery: bool = True,
    output_path: str | None = None,
) -> list:
    """Run Walker2d over (gravity_scale, friction_scale) grid; return list of dicts."""
    grid = [
        (1.0, 1.0, "Standard"),
        (1.05, 0.95, "g+5% f-5%"),
        (0.95, 1.05, "g-5% f+5%"),
        (1.1, 0.9, "g+10% f-10%"),
        (0.9, 1.1, "g-10% f+10%"),
        (1.2, 0.8, "High Gravity / Low Friction"),
        (0.8, 1.2, "Low Gravity / High Friction"),
    ]
    results = []
    for gs, fs, label in grid:
        print(f"\n  Testing: {label} (g={gs}, f={fs})")
        r = run_one_env_with_discovery(
            "Walker2d-v5",
            policy=policy,
            steps=steps_per_condition,
            gravity_scale=gs,
            friction_scale=fs,
            seed=seed,
            render=False,
            use_discovery=use_discovery,
        )
        results.append({
            "label": label,
            "g": gs,
            "f": fs,
            "reward": float(r),
            "discovery_enabled": use_discovery,
        })
        print(f"  Result: reward={r:.1f}")
    
    # Compute robustness metrics
    rewards = [r["reward"] for r in results]
    baseline_reward = results[0]["reward"]  # Standard condition
    min_reward = min(rewards)
    mean_reward = sum(rewards) / len(rewards)
    
    summary = {
        "walker2d_sensitivity": results,
        "robustness_metrics": {
            "baseline_reward": float(baseline_reward),
            "min_reward": float(min_reward),
            "mean_reward": float(mean_reward),
            "reward_drop_worst": float(baseline_reward - min_reward),
            "robustness_score": float(mean_reward / (baseline_reward + 1e-6)),
        },
        "metadata": {
            "env": "Walker2d-v5",
            "steps_per_condition": steps_per_condition,
            "discovery_enabled": use_discovery,
            "seed": seed,
        }
    }
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved detailed results to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("Robustness Summary:")
        print(f"  Baseline (Standard): {baseline_reward:.1f}")
        print(f"  Worst case: {min_reward:.1f} (drop: {baseline_reward - min_reward:.1f})")
        print(f"  Mean across conditions: {mean_reward:.1f}")
        print(f"  Robustness Score: {summary['robustness_metrics']['robustness_score']:.2%}")
        print("="*60)
    
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Gym/MuJoCo with Online Identification")
    parser.add_argument("--env", default="Walker2d-v5", choices=["Walker2d-v5", "Humanoid-v5"])
    parser.add_argument("--train-episodes", type=int, default=40, help="Training episodes (standard physics)")
    parser.add_argument("--train-steps", type=int, default=200, help="Steps per episode during training")
    parser.add_argument("--sensitivity-steps", type=int, default=200, help="Steps per condition in sensitivity test")
    parser.add_argument("--no-discovery", action="store_true", help="Disable Discovery Engine (fixed Hard Core)")
    parser.add_argument("--output", "-o", default="sensitivity_results.json", help="Sensitivity results JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from axiom_os.envs import make_axiom_mujoco_env
        from axiom_os.envs.axiom_policy import PhysicsAwarePolicy
    except ImportError as e:
        print(f"Install deps: pip install gymnasium mujoco; error: {e}")
        return 1

    print("=" * 60)
    print("main_gym: Online Identification + Sensitivity Test")
    print("=" * 60)
    print(f"Env: {args.env}")
    print(f"Train: {args.train_episodes} episodes x {args.train_steps} steps (standard g=1, f=1)")
    print(f"Discovery Engine: {'OFF' if args.no_discovery else 'ON'}")
    print("-" * 60)

    # Train under nominal physics
    print("[1/2] Training policy...")
    policy = _train_policy(
        args.env,
        episodes=args.train_episodes,
        steps_per_episode=args.train_steps,
        seed=args.seed,
    )

    # Sensitivity test (Walker2d grid)
    print("[2/2] Walker2d sensitivity test...")
    if args.env == "Walker2d-v5":
        run_walker2d_sensitivity(
            policy,
            seed=args.seed,
            steps_per_condition=args.sensitivity_steps,
            use_discovery=not args.no_discovery,
            output_path=args.output,
        )
    else:
        # Humanoid: single run with discovery
        r = run_one_env_with_discovery(
            args.env,
            policy=policy,
            steps=args.sensitivity_steps,
            gravity_scale=1.0,
            friction_scale=1.0,
            seed=args.seed,
            render=False,
            use_discovery=not args.no_discovery,
        )
        print(f"Humanoid total reward: {r:.1f}")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"humanoid_reward": float(r)}, f, indent=2)

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
