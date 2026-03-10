"""
Axiom-OS MuJoCo 高考 (Gaokao) - 完整控制测试
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

测试科目：
1. Swimmer-v5: 多刚体动力学基础
2. Hopper-v5: 单腿跳跃平衡
3. Walker2d-v5: 双足行走 + Sim-to-Real 敏感性分析
4. Humanoid-v5: 终极 BOSS - 与 PPO 对比

Sim-to-Real 挑战：
- 修改 gravity_scale (0.8-1.2)
- 修改 friction_scale (0.7-1.3)
- 测试 Axiom 自适应能力
"""

import sys
from pathlib import Path
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def run_basic_tests():
    """基础测试科目：Swimmer, Hopper, Walker2d"""
    print("="*70)
    print("Axiom-OS MuJoCo 高考 - 基础科目")
    print("="*70)
    
    from axiom_os.scripts.run_mujoco import main as mujoco_main
    
    # 运行所有基础环境
    sys.argv = [
        "run_mujoco.py",
        "--env", "all",
        "--steps", "500",
        "--train", "10",  # 快速训练
        "--timesteps", "10000",
        "--algo", "PPO",
        "--domain-rand",
        "--physics-aware",
        "--seed", "42",
        "--output", "./outputs/mujoco_gaokao_basic.md"
    ]
    
    return mujoco_main()


def run_sim2real_challenge():
    """Sim-to-Real 挑战：Walker2d 敏感性分析"""
    print("\n" + "="*70)
    print("Axiom-OS MuJoCo 高考 - Sim-to-Real 挑战")
    print("="*70)
    
    from axiom_os.scripts.run_mujoco import main as mujoco_main
    
    # Walker2d 敏感性分析
    sys.argv = [
        "run_mujoco.py",
        "--analyze-walker2d",
        "--seed", "42",
        "--output", "./outputs/mujoco_gaokao_sim2real.json"
    ]
    
    return mujoco_main()


def run_humanoid_boss():
    """终极 BOSS：Humanoid vs PPO"""
    print("\n" + "="*70)
    print("Axiom-OS MuJoCo 高考 - 终极 BOSS: Humanoid")
    print("="*70)
    
    from axiom_os.scripts.run_mujoco import main as mujoco_main
    
    # Humanoid 长训练
    sys.argv = [
        "run_mujoco.py",
        "--env", "Humanoid-v5",
        "--steps", "1000",
        "--train", "50",
        "--timesteps", "50000",
        "--algo", "PPO",
        "--domain-rand",
        "--physics-aware",
        "--seed", "42",
        "--output", "./outputs/mujoco_gaokao_humanoid.md"
    ]
    
    return mujoco_main()


def run_gravity_friction_tests():
    """重力/摩擦扰动测试"""
    print("\n" + "="*70)
    print("Axiom-OS MuJoCo 高考 - 重力/摩擦扰动测试")
    print("="*70)
    
    from axiom_os.envs import make_axiom_mujoco_env
    from axiom_os.envs.axiom_policy import make_mujoco_policy, PhysicsAwarePolicy, train_policy_reinforce
    import numpy as np
    
    env_id = "Hopper-v5"
    results = []
    
    # 测试不同重力/摩擦组合
    test_conditions = [
        (1.0, 1.0, "标准条件"),
        (1.2, 0.8, "高重力/低摩擦"),
        (0.8, 1.2, "低重力/高摩擦"),
        (1.1, 0.9, "g+10%/f-10%"),
        (0.9, 1.1, "g-10%/f+10%"),
    ]
    
    for g_scale, f_scale, label in test_conditions:
        print(f"\n测试: {label} (g={g_scale}, f={f_scale})")
        
        env = make_axiom_mujoco_env(
            env_id=env_id,
            gravity_scale=g_scale,
            friction_scale=f_scale,
            seed=42,
        )
        
        obs, _ = env.reset(seed=42)
        policy = make_mujoco_policy(obs.shape[0], env.action_space.shape[0])
        
        # 快速训练
        print("  训练中...")
        train_policy_reinforce(policy, env, episodes=5, steps_per_episode=200, seed=42)
        
        # 评估
        obs, _ = env.reset(seed=42)
        total_reward = 0
        for _ in range(500):
            action = policy.act(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            if term or trunc:
                obs, _ = env.reset()
        
        env.close()
        
        results.append({
            "condition": label,
            "gravity_scale": g_scale,
            "friction_scale": f_scale,
            "reward": float(total_reward)
        })
        
        print(f"  奖励: {total_reward:.1f}")
    
    # 保存结果
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "mujoco_gaokao_perturbation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("扰动测试完成!")
    print("="*70)
    
    # 计算鲁棒性分数
    rewards = [r["reward"] for r in results]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    robustness_score = mean_reward / (std_reward + 1e-6)
    
    print(f"\n鲁棒性分析:")
    print(f"  平均奖励: {mean_reward:.1f}")
    print(f"  奖励标准差: {std_reward:.1f}")
    print(f"  鲁棒性分数: {robustness_score:.2f}")
    
    return results


def print_summary():
    """打印高考成绩单"""
    print("\n" + "="*70)
    print("Axiom-OS MuJoCo 高考成绩单")
    print("="*70)
    
    output_dir = Path("./outputs")
    
    # 读取各科目成绩
    files = {
        "基础科目": "mujoco_gaokao_basic.md",
        "Walker2d分析": "mujoco_gaokao_sim2real.json",
        "Humanoid终极": "mujoco_gaokao_humanoid.md",
        "扰动测试": "mujoco_gaokao_perturbation.json",
    }
    
    for subject, filename in files.items():
        filepath = output_dir / filename
        if filepath.exists():
            print(f"\n{subject}: {filename}")
            if filename.endswith('.md'):
                content = filepath.read_text()
                print(content[:500] + "..." if len(content) > 500 else content)
            elif filename.endswith('.json'):
                data = json.loads(filepath.read_text())
                print(json.dumps(data, indent=2)[:500])
        else:
            print(f"\n{subject}: 未完成")
    
    print("\n" + "="*70)


def main():
    """主函数：运行完整高考"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Axiom-OS MuJoCo Gaokao (College Entrance Exam)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行完整高考
  python run_mujoco_gaokao.py --full
  
  # 只运行基础科目
  python run_mujoco_gaokao.py --basic
  
  # 只运行 Sim-to-Real 挑战
  python run_mujoco_gaokao.py --sim2real
  
  # 只运行 Humanoid 终极 BOSS
  python run_mujoco_gaokao.py --humanoid
  
  # 只运行扰动测试
  python run_mujoco_gaokao.py --perturbation
        """
    )
    
    parser.add_argument("--full", action="store_true", help="运行完整高考 (所有科目)")
    parser.add_argument("--basic", action="store_true", help="基础科目 (Swimmer/Hopper/Walker2d)")
    parser.add_argument("--sim2real", action="store_true", help="Sim-to-Real 挑战")
    parser.add_argument("--humanoid", action="store_true", help="Humanoid 终极 BOSS")
    parser.add_argument("--perturbation", action="store_true", help="重力/摩擦扰动测试")
    parser.add_argument("--summary", action="store_true", help="打印成绩单")
    
    args = parser.parse_args()
    
    # 如果没有指定，默认运行完整测试
    if not any([args.full, args.basic, args.sim2real, args.humanoid, args.perturbation, args.summary]):
        args.full = True
    
    if args.summary:
        print_summary()
        return
    
    results = {}
    
    if args.full or args.basic:
        print("\n[1/4] 基础科目测试...")
        results["basic"] = run_basic_tests()
    
    if args.full or args.sim2real:
        print("\n[2/4] Sim-to-Real 挑战...")
        results["sim2real"] = run_sim2real_challenge()
    
    if args.full or args.humanoid:
        print("\n[3/4] Humanoid 终极 BOSS...")
        results["humanoid"] = run_humanoid_boss()
    
    if args.full or args.perturbation:
        print("\n[4/4] 重力/摩擦扰动测试...")
        results["perturbation"] = run_gravity_friction_tests()
    
    # 打印总结
    print_summary()
    
    print("\n" + "="*70)
    print("Axiom-OS MuJoCo 高考完成!")
    print("="*70)
    print("\n成绩单保存在: ./outputs/")
    print("  - mujoco_gaokao_basic.md")
    print("  - mujoco_gaokao_sim2real.json")
    print("  - mujoco_gaokao_humanoid.md")
    print("  - mujoco_gaokao_perturbation.json")


if __name__ == "__main__":
    main()
