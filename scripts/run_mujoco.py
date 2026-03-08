#!/usr/bin/env python3
"""
运行 MuJoCo 环境：Axiom-OS 控制测试

测试科目：Swimmer / Hopper / Walker2d / Humanoid
Sim-to-Real：--gravity-scale / --friction-scale 测试自适应能力

用法: axiom mujoco --env Hopper-v5 [--gravity-scale 1.1] [--steps 1000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _analyze_walker2d(args) -> int:
    """Walker2d Sim-to-Real 敏感性分析：不同 gravity/friction 下的表现"""
    from axiom_os.envs import make_axiom_mujoco_env
    from axiom_os.envs.axiom_policy import make_mujoco_policy, train_policy_reinforce

    print("=" * 60)
    print("Walker2d Sim-to-Real 敏感性分析")
    print("=" * 60)
    env = make_axiom_mujoco_env("Walker2d-v5", seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    policy = make_mujoco_policy(obs.shape[0], env.action_space.shape[0])
    train_policy_reinforce(policy, env, episodes=30, steps_per_episode=200, seed=args.seed)
    env.close()

    grid = [
        (1.0, 1.0, "标准"),
        (1.05, 0.95, "g+5% f-5%"),
        (0.95, 1.05, "g-5% f+5%"),
        (1.1, 0.9, "g+10% f-10%"),
        (0.9, 1.1, "g-10% f+10%"),
    ]
    results = []
    for gs, fs, label in grid:
        r = run_one_env(
            "Walker2d-v5", steps=200,
            gravity_scale=gs, friction_scale=fs,
            seed=args.seed, render=False, policy=policy,
        )
        results.append((label, gs, fs, r))
        print(f"  {label}: reward={r:.1f}")
    print("-" * 60)
    if args.output:
        data = {"walker2d_sensitivity": [{"label": l, "g": g, "f": f, "reward": r} for l, g, f, r in results]}
        Path(args.output).write_text(__import__("json").dumps(data, indent=2), encoding="utf-8")
    print("=" * 60)
    return 0


def _write_results(path: str, results: dict, args) -> None:
    """输出结果到 JSON 或 MD 文件"""
    import json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "results": results,
        "config": {
            "steps": args.steps,
            "gravity_scale": args.gravity_scale,
            "friction_scale": args.friction_scale,
            "seed": args.seed,
            "train_episodes": getattr(args, "train", 0),
            "timesteps": getattr(args, "timesteps", 0),
            "algo": getattr(args, "algo", "PPO"),
            "domain_rand": getattr(args, "domain_rand", False),
            "physics_aware": getattr(args, "physics_aware", False),
        },
    }
    if p.suffix.lower() in (".json",):
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        lines = [
            "# MuJoCo 测试科目结果",
            "",
            "| 科目 | 总奖励 |",
            "|------|--------|",
        ]
        for env_id, r in results.items():
            lines.append(f"| {env_id} | {r:.1f} |" if r is not None else f"| {env_id} | FAIL |")
        extra = []
        if getattr(args, "domain_rand", False):
            extra.append("域随机化")
        if getattr(args, "physics_aware", False):
            extra.append("物理感知")
        ts = getattr(args, "timesteps", 0)
        train_info = f"{getattr(args, 'train', 0)} episodes" if ts == 0 else f"{ts} steps ({getattr(args, 'algo', 'PPO')})"
        lines.extend(["", f"步数: {args.steps}, 训练: {train_info}" + (" (" + ", ".join(extra) + ")" if extra else "")])
        p.write_text("\n".join(lines), encoding="utf-8")


def run_one_env(
    env_id: str,
    steps: int,
    gravity_scale: float,
    friction_scale: float,
    seed: int,
    render: bool,
    policy=None,
    sb3_model=None,
    physics_aware: bool = False,
) -> float:
    """运行单个环境，返回总奖励。policy 为自定义 Policy，sb3_model 为 SB3 模型。"""
    from axiom_os.envs import make_axiom_mujoco_env, get_physics_params, PhysicsAwareObsWrapper
    from axiom_os.envs.axiom_policy import make_mujoco_policy, PhysicsAwarePolicy

    env = make_axiom_mujoco_env(
        env_id=env_id,
        gravity_scale=gravity_scale,
        friction_scale=friction_scale,
        seed=seed,
    )
    if sb3_model is not None and physics_aware:
        env = PhysicsAwareObsWrapper(env, eval_gravity_scale=gravity_scale, eval_friction_scale=friction_scale)
    obs, info = env.reset(seed=seed)
    obs_dim, action_dim = obs.shape[0], env.action_space.shape[0]
    if sb3_model is not None:
        pass  # use sb3_model.predict
    elif policy is None:
        if physics_aware:
            policy = PhysicsAwarePolicy(obs_dim=obs_dim, action_dim=action_dim)
        else:
            policy = make_mujoco_policy(obs_dim=obs_dim, action_dim=action_dim)

    params = get_physics_params(env) if physics_aware and policy is not None else None
    total_reward = 0.0
    for _ in range(steps):
        if sb3_model is not None:
            action, _ = sb3_model.predict(obs, deterministic=True)
        elif physics_aware and hasattr(policy, "act"):
            action = policy.act(obs, deterministic=True, gravity_scale=gravity_scale, friction_scale=friction_scale, body_masses=params.body_masses if params else None)
        else:
            action = policy.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    return total_reward


def main() -> int:
    parser = argparse.ArgumentParser(description="Axiom MuJoCo 控制测试")
    parser.add_argument(
        "--env",
        default="Hopper-v5",
        choices=["Swimmer-v5", "Hopper-v5", "Walker2d-v5", "Humanoid-v5", "all"],
        help="MuJoCo 环境（all=全部测试科目）",
    )
    parser.add_argument("--steps", type=int, default=100_000, help="每环境运行步数（默认 10 万步）")
    parser.add_argument("--gravity-scale", type=float, default=1.0, help="重力缩放（Sim-to-Real）")
    parser.add_argument("--friction-scale", type=float, default=1.0, help="摩擦缩放（Sim-to-Real）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--render", action="store_true", help="渲染画面")
    parser.add_argument("--train", type=int, default=0, metavar="N", help="训练 N episodes 或 --timesteps 步后再评估")
    parser.add_argument("--timesteps", type=int, default=0, help="PPO/SAC 总步数（>0 时用 PPO/SAC 替代 REINFORCE）")
    parser.add_argument("--algo", default="PPO", choices=["PPO", "SAC"], help="PPO 或 SAC（仅当 --timesteps>0 有效）")
    parser.add_argument("--domain-rand", action="store_true", help="训练时域随机化 (gravity/friction)")
    parser.add_argument("--physics-aware", action="store_true", help="使用 Hard Core 物理感知 Policy")
    parser.add_argument("--analyze-walker2d", action="store_true", help="Walker2d Sim-to-Real 敏感性分析")
    parser.add_argument("--output", "-o", help="输出结果到 JSON 或 MD 文件")
    args = parser.parse_args()

    try:
        from axiom_os.envs import make_axiom_mujoco_env, get_physics_params, MUJOCO_ENV_IDS
    except ImportError as e:
        print(f"❌ 请安装: pip install gymnasium mujoco")
        print(f"   或: pip install 'axiom-os[mujoco]'")
        print(f"   错误: {e}")
        return 1

    if args.analyze_walker2d:
        return _analyze_walker2d(args)

    print("=" * 60)
    print("Axiom-OS MuJoCo 控制测试")
    print("=" * 60)
    print(f"环境: {args.env}")
    print(f"步数: {args.steps}")
    if args.gravity_scale != 1.0 or args.friction_scale != 1.0:
        print(f"Sim-to-Real: gravity_scale={args.gravity_scale}, friction_scale={args.friction_scale}")
    print("-" * 60)

    use_ppo_sac = args.train > 0 and (args.timesteps > 0 or True)  # 默认用 PPO/SAC
    total_ts = args.timesteps if args.timesteps > 0 else args.train * 256

    if args.env == "all":
        results = {}
        for env_id in MUJOCO_ENV_IDS:
            print(f"\n>>> {env_id}")
            try:
                policy, sb3_model = None, None
                if args.train > 0:
                    if env_id == "Walker2d-v5":
                        gravity_range, friction_range = (0.85, 1.15), (0.75, 1.25)
                    elif env_id == "Swimmer-v5":
                        gravity_range, friction_range = (0.95, 1.05), (0.95, 1.05)
                    else:
                        gravity_range, friction_range = (0.9, 1.1), (0.8, 1.2)
                    if use_ppo_sac:
                        from axiom_os.envs.axiom_policy import train_ppo_sac
                        sb3_model = train_ppo_sac(
                            env_id=env_id,
                            algo=args.algo,
                            total_timesteps=total_ts,
                            domain_rand=args.domain_rand or True,
                            physics_aware=args.physics_aware or True,
                            gravity_range=gravity_range,
                            friction_range=friction_range,
                            learning_rate=3e-4,
                            n_steps=2048,
                            batch_size=64,
                            n_epochs=10,
                            seed=args.seed,
                        )
                        print(f"    训练 {total_ts} steps ({args.algo}, 域随机化, 物理感知)")
                    else:
                        from axiom_os.envs import make_axiom_mujoco_env_with_domain_rand
                        from axiom_os.envs.axiom_policy import make_mujoco_policy, PhysicsAwarePolicy, train_policy_reinforce, train_with_domain_rand
                        env_train = make_axiom_mujoco_env_with_domain_rand(env_id=env_id, gravity_range=gravity_range, friction_range=friction_range, seed=args.seed)
                        obs, _ = env_train.reset(seed=args.seed)
                        policy = (PhysicsAwarePolicy if args.physics_aware else make_mujoco_policy)(obs.shape[0], env_train.action_space.shape[0])
                        hist = train_with_domain_rand(policy, env_id, episodes=args.train, steps_per_episode=min(300, args.steps), gravity_range=gravity_range, friction_range=friction_range, seed=args.seed)
                        env_train.close()
                        print(f"    训练 {args.train} episodes (REINFORCE, 域随机化), 末轮: {hist[-1]:.1f}")
                r = run_one_env(
                    env_id=env_id,
                    steps=args.steps,
                    gravity_scale=args.gravity_scale,
                    friction_scale=args.friction_scale,
                    seed=args.seed,
                    render=args.render,
                    policy=policy,
                    sb3_model=sb3_model,
                    physics_aware=args.physics_aware or (sb3_model is not None),
                )
                results[env_id] = r
                print(f"    完成。总奖励: {r:.1f}")
            except Exception as e:
                results[env_id] = None
                print(f"    失败: {e}")
        print("\n" + "=" * 60)
        print("测试科目汇总")
        print("-" * 60)
        for env_id, r in results.items():
            status = f"{r:.1f}" if r is not None else "FAIL"
            print(f"  {env_id}: {status}")
        print("=" * 60)
        if args.output:
            _write_results(args.output, results, args)
        return 0 if all(r is not None for r in results.values()) else 1

    # 单环境
    env = make_axiom_mujoco_env(
        env_id=args.env,
        gravity_scale=args.gravity_scale,
        friction_scale=args.friction_scale,
        seed=args.seed,
    )
    params = get_physics_params(env)
    print(f"物理参数: gravity={params.gravity}, friction={params.friction}")
    print(f"  body_masses 数量: {len(params.body_masses)}")
    print("-" * 60)

    policy, sb3_model = None, None
    use_ppo_sac = args.train > 0 and (args.timesteps > 0 or True)
    total_ts = args.timesteps if args.timesteps > 0 else args.train * 256
    if args.env == "Walker2d-v5":
        gravity_range, friction_range = (0.85, 1.15), (0.75, 1.25)
    elif args.env == "Swimmer-v5":
        gravity_range, friction_range = (0.95, 1.05), (0.95, 1.05)
    else:
        gravity_range, friction_range = (0.9, 1.1), (0.8, 1.2)
    if args.train > 0:
        if use_ppo_sac:
            from axiom_os.envs.axiom_policy import train_ppo_sac
            sb3_model = train_ppo_sac(
                env_id=args.env,
                algo=args.algo,
                total_timesteps=total_ts,
                domain_rand=args.domain_rand or True,
                physics_aware=args.physics_aware or True,
                gravity_range=gravity_range,
                friction_range=friction_range,
                seed=args.seed,
            )
            env.close()
            print(f"训练 {total_ts} steps ({args.algo}, 域随机化, 物理感知) 完成")
        else:
            from axiom_os.envs import make_axiom_mujoco_env_with_domain_rand
            from axiom_os.envs.axiom_policy import make_mujoco_policy, PhysicsAwarePolicy, train_with_domain_rand
            obs, _ = env.reset(seed=args.seed)
            policy = (PhysicsAwarePolicy if args.physics_aware else make_mujoco_policy)(obs.shape[0], env.action_space.shape[0])
            env.close()
            train_with_domain_rand(policy, args.env, episodes=args.train, steps_per_episode=min(200, args.steps), seed=args.seed, gravity_range=gravity_range, friction_range=friction_range)
            print(f"训练 {args.train} episodes (REINFORCE, 域随机化) 完成")

    total_reward = run_one_env(
        env_id=args.env,
        steps=args.steps,
        gravity_scale=args.gravity_scale,
        friction_scale=args.friction_scale,
        seed=args.seed,
        render=args.render,
        policy=policy,
        sb3_model=sb3_model,
        physics_aware=args.physics_aware or (sb3_model is not None),
    )

    print("-" * 60)
    print(f"完成。总奖励: {total_reward:.1f}")
    print("=" * 60)
    if args.output:
        _write_results(args.output, {args.env: total_reward}, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
