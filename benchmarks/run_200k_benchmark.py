#!/usr/bin/env python3
"""
20万步全面性能测试与多算法对比

- 统一训练量：200,000 步（PPO/SAC 为 total_timesteps，Axiom 为 1000 episodes × 200 steps）
- 多种子：默认 3 个 seed，汇报 mean ± std
- 算法：PPO 标准、PPO+域随机、PPO+物理感知、Axiom 固定 Hard Core、Axiom+Discovery
- 评估：标称 (g=1,f=1) + 敏感性 7 工况
- 输出：JSON 详细结果 + 终端汇总表
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOTAL_STEPS = 200_000
SENSITIVITY_GRID = [
    (1.0, 1.0, "Standard"),
    (1.05, 0.95, "g+5% f-5%"),
    (0.95, 1.05, "g-5% f+5%"),
    (1.1, 0.9, "g+10% f-10%"),
    (0.9, 1.1, "g-10% f+10%"),
    (1.2, 0.8, "High Gravity / Low Friction"),
    (0.8, 1.2, "Low Gravity / High Friction"),
]


def _eval_ppo(
    env_id: str,
    sb3_model,
    gravity_scale: float,
    friction_scale: float,
    steps: int,
    seed: int,
    physics_aware: bool,
) -> float:
    """Run PPO/SAC model on env with given (g, f); return total reward."""
    from axiom_os.envs import make_axiom_mujoco_env, get_physics_params, PhysicsAwareObsWrapper

    env = make_axiom_mujoco_env(
        env_id=env_id,
        gravity_scale=gravity_scale,
        friction_scale=friction_scale,
        seed=seed,
    )
    if physics_aware:
        env = PhysicsAwareObsWrapper(
            env, eval_gravity_scale=gravity_scale, eval_friction_scale=friction_scale
        )
    obs, _ = env.reset(seed=seed)
    params = get_physics_params(env) if physics_aware else None
    total = 0.0
    for _ in range(steps):
        action, _ = sb3_model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return total


def _train_ppo(
    env_id: str,
    seed: int,
    domain_rand: bool,
    physics_aware: bool,
    total_timesteps: int = TOTAL_STEPS,
):
    from axiom_os.envs.axiom_policy import train_ppo_sac

    return train_ppo_sac(
        env_id=env_id,
        algo="PPO",
        total_timesteps=total_timesteps,
        domain_rand=domain_rand,
        physics_aware=physics_aware,
        gravity_range=(0.9, 1.1),
        friction_range=(0.8, 1.2),
        seed=seed,
    )


def _train_axiom(env_id: str, seed: int, total_steps: int = TOTAL_STEPS):
    """Train PhysicsAwarePolicy with REINFORCE for total_steps (episodes × steps_per_episode)."""
    from axiom_os.envs import make_axiom_mujoco_env
    from axiom_os.envs.axiom_policy import PhysicsAwarePolicy, train_policy_reinforce

    episodes = max(1, total_steps // 200)
    steps_per_episode = max(200, total_steps // episodes)
    env = make_axiom_mujoco_env(env_id=env_id, seed=seed)
    obs, _ = env.reset(seed=seed)
    policy = PhysicsAwarePolicy(obs.shape[0], env.action_space.shape[0])
    train_policy_reinforce(
        policy, env, episodes=episodes, steps_per_episode=steps_per_episode, seed=seed
    )
    env.close()
    return policy


def _eval_axiom(
    env_id: str,
    policy,
    gravity_scale: float,
    friction_scale: float,
    steps: int,
    seed: int,
    use_discovery: bool,
) -> float:
    from main_gym import run_one_env_with_discovery

    return run_one_env_with_discovery(
        env_id=env_id,
        policy=policy,
        steps=steps,
        gravity_scale=gravity_scale,
        friction_scale=friction_scale,
        seed=seed,
        render=False,
        use_discovery=use_discovery,
    )


def run_full_benchmark(
    env_id: str = "Walker2d-v5",
    seeds: list[int] | None = None,
    eval_steps_per_condition: int = 200,
    total_steps: int = TOTAL_STEPS,
    algorithms: list[str] | None = None,
    output_dir: str | Path = "benchmarks/200k_results",
) -> dict:
    """
    Train each algorithm for total_steps, evaluate on sensitivity grid for each seed.
    Returns aggregated results and saves JSON.
    """
    seeds = seeds or [42, 43, 44]
    algorithms = algorithms or [
        "ppo",
        "ppo_domain_rand",
        "ppo_physics",
        "axiom_fixed",
        "axiom_discovery",
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = {algo: {cond: [] for cond in [c[2] for c in SENSITIVITY_GRID]} for algo in algorithms}
    raw["_meta"] = {
        "env_id": env_id,
        "total_steps": total_steps,
        "seeds": seeds,
        "eval_steps_per_condition": eval_steps_per_condition,
    }

    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

        # --- PPO (no domain rand, no physics) ---
        if "ppo" in algorithms:
            print("[Train] PPO (standard)...")
            ppo_model = _train_ppo(env_id, seed, domain_rand=False, physics_aware=False, total_timesteps=total_steps)
            for gs, fs, label in SENSITIVITY_GRID:
                r = _eval_ppo(env_id, ppo_model, gs, fs, eval_steps_per_condition, seed, physics_aware=False)
                raw["ppo"][label].append(r)
            del ppo_model

        # --- PPO + domain rand ---
        if "ppo_domain_rand" in algorithms:
            print("[Train] PPO + domain_rand...")
            ppo_dr = _train_ppo(env_id, seed, domain_rand=True, physics_aware=False, total_timesteps=total_steps)
            for gs, fs, label in SENSITIVITY_GRID:
                r = _eval_ppo(env_id, ppo_dr, gs, fs, eval_steps_per_condition, seed, physics_aware=False)
                raw["ppo_domain_rand"][label].append(r)
            del ppo_dr

        # --- PPO + physics aware ---
        if "ppo_physics" in algorithms:
            print("[Train] PPO + physics_aware + domain_rand...")
            ppo_phys = _train_ppo(env_id, seed, domain_rand=True, physics_aware=True, total_timesteps=total_steps)
            for gs, fs, label in SENSITIVITY_GRID:
                r = _eval_ppo(env_id, ppo_phys, gs, fs, eval_steps_per_condition, seed, physics_aware=True)
                raw["ppo_physics"][label].append(r)
            del ppo_phys

        # --- Axiom (one policy, eval fixed vs discovery) ---
        if "axiom_fixed" in algorithms or "axiom_discovery" in algorithms:
            print("[Train] Axiom (PhysicsAwarePolicy REINFORCE)...")
            axiom_policy = _train_axiom(env_id, seed, total_steps=total_steps)
            for gs, fs, label in SENSITIVITY_GRID:
                if "axiom_fixed" in algorithms:
                    r = _eval_axiom(env_id, axiom_policy, gs, fs, eval_steps_per_condition, seed, use_discovery=False)
                    raw["axiom_fixed"][label].append(r)
                if "axiom_discovery" in algorithms:
                    r = _eval_axiom(env_id, axiom_policy, gs, fs, eval_steps_per_condition, seed, use_discovery=True)
                    raw["axiom_discovery"][label].append(r)

    # Aggregate mean ± std
    summary = {}
    for algo in algorithms:
        if algo.startswith("_"):
            continue
        summary[algo] = {}
        for label, rewards in raw[algo].items():
            arr = np.array(rewards)
            summary[algo][label] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": len(arr)}

    # Save
    out_json = output_dir / "200k_benchmark_results.json"
    to_save = {
        "raw": {k: v for k, v in raw.items() if k != "_meta" and isinstance(v, dict)},
        "summary": summary,
        "meta": raw["_meta"],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_json}")

    # Print table
    _print_table(summary)
    return to_save


def _print_table(summary: dict) -> None:
    """Print comparison table: algorithms × conditions, mean ± std."""
    labels = [c[2] for c in SENSITIVITY_GRID]
    algos = [a for a in summary.keys() if isinstance(summary[a], dict)]
    col_w = 14
    print("\n" + "=" * (2 + (1 + col_w) * (1 + len(labels))))
    print("20万步 全面性能对比 (mean ± std)")
    print("=" * (2 + (1 + col_w) * (1 + len(labels))))
    header = "Algorithm".ljust(22) + "".join(l[:10].ljust(col_w) for l in labels)
    print(header)
    print("-" * len(header))
    for algo in algos:
        row = algo.ljust(22)
        for label in labels:
            m = summary[algo][label]["mean"]
            s = summary[algo][label]["std"]
            row += f"{m:.1f}±{s:.1f}".ljust(col_w)
        print(row)
    print("=" * len(header))
    # Robustness: mean over conditions / (std over conditions + eps)
    print("\n鲁棒性 (各工况 reward 均值 / 标准差，越高越稳):")
    for algo in algos:
        means = [summary[algo][l]["mean"] for l in labels]
        stds = [summary[algo][l]["std"] for l in labels]
        robust = np.mean(means) / (np.std(means) + 1e-6)
        worst = min(means)
        print(f"  {algo}: robust_ratio={robust:.2f}, worst_condition_reward={worst:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="20万步全面性能测试与对比")
    parser.add_argument("--env", default="Walker2d-v5", help="MuJoCo env")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds")
    parser.add_argument("--eval-steps", type=int, default=200, help="Eval steps per condition")
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS, help="Training steps per run")
    parser.add_argument("--algorithms", type=str, default=None, help="Comma-separated: ppo,ppo_domain_rand,ppo_physics,axiom_fixed,axiom_discovery")
    parser.add_argument("--output-dir", default="benchmarks/200k_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 seed, 2000 steps, 50 eval steps")
    args = parser.parse_args()

    if args.quick:
        args.seeds = "42"
        args.total_steps = 2000
        args.eval_steps = 50
    seeds = [int(s) for s in args.seeds.split(",")]
    algorithms = args.algorithms.split(",") if args.algorithms else None

    try:
        from axiom_os.envs import make_axiom_mujoco_env
        from axiom_os.envs.axiom_policy import train_ppo_sac
    except ImportError as e:
        print(f"Install: pip install gymnasium mujoco stable-baselines3; error: {e}")
        return 1

    run_full_benchmark(
        env_id=args.env,
        seeds=seeds,
        eval_steps_per_condition=args.eval_steps,
        total_steps=args.total_steps,
        algorithms=algorithms,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
