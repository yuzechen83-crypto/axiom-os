#!/usr/bin/env python3
"""
严格测试：依次运行 Acrobot-v1 与 Pendulum-v1 的 Axiom-OS Gym 基准，
保存结果与 reward 曲线，并输出汇总。
"""
import sys
import os

# 确保可导入 gym_adapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.gym_adapter import run_benchmark, plot_reward_curve, save_results


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "strict_test_out")
    os.makedirs(out_dir, exist_ok=True)

    strict = {
        "episodes": 5,
        "steps": 200,
        "fast": True,
        "use_adaptation": True,
        "render": False,
    }

    all_results = {}
    for env_name in ["Acrobot-v1", "Pendulum-v1"]:
        print("\n" + "=" * 70)
        print(f"严格测试: {env_name}")
        print("=" * 70)
        results = run_benchmark(
            env_name=env_name,
            n_episodes=strict["episodes"],
            max_steps=strict["steps"],
            render=strict["render"],
            fast=strict["fast"],
            save_path=os.path.join(out_dir, f"strict_{env_name.replace('-', '_')}.json"),
            use_adaptation=strict["use_adaptation"],
        )
        all_results[env_name] = results
        plot_reward_curve(
            results,
            save_path=os.path.join(out_dir, f"strict_{env_name.replace('-', '_')}_reward_curve.png"),
        )

    # 汇总
    print("\n" + "=" * 70)
    print("严格测试汇总")
    print("=" * 70)
    for env_name, r in all_results.items():
        print(f"  {env_name}: success_rate={r['success_rate']*100:.1f}%, "
              f"mean_reward={r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
    print(f"结果与曲线已保存到: {out_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
