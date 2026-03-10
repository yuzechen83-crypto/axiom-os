"""
Axiom-OS Real-World Control Challenge
Copyright (c) 2026 yuzechen83-crypto. All Rights Reserved.

真实世界控制挑战主入口

用法：
  # 模拟模式测试
  python run_real_world_challenge.py --mock --duration 10
  
  # 连接宇树 Go1
  python run_real_world_challenge.py --platform unitree_go1 --duration 30
  
  # 加载训练好的 Policy
  python run_real_world_challenge.py --mock --policy outputs/mujoco_gaokao_model.pt
  
  # 基准测试
  python run_real_world_challenge.py --benchmark
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Axiom-OS Real-World Control Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mock mode (simulation)
  python run_real_world_challenge.py --mock
  
  # Benchmark control frequency
  python run_real_world_challenge.py --benchmark
  
  # Deploy trained policy
  python run_real_world_challenge.py --policy model.pt --duration 60
        """
    )
    
    parser.add_argument(
        "--platform",
        choices=["mock", "unitree_go1", "unitree_h1", "ros2"],
        default="mock",
        help="Robot platform"
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="Path to trained policy checkpoint"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Episode duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run frequency benchmark only"
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=50.0,
        help="Control frequency in Hz (default: 50)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Axiom-OS Real-World Control Challenge")
    print("="*70)
    print(f"Platform: {args.platform}")
    print(f"Control freq: {args.freq}Hz")
    if args.policy:
        print(f"Policy: {args.policy}")
    print("-"*70)
    
    # Import here to handle missing dependencies gracefully
    try:
        from axiom_os.real_world import (
            DeployController,
            RobotPlatform,
            RobotConfig,
            DeploymentConfig,
            RealTimeController,
            SafetyConstraint,
        )
    except ImportError as e:
        print(f"[Error] Failed to import real_world module: {e}")
        print("Please check dependencies: pip install numpy torch")
        return 1
    
    # Map platform string to enum
    platform_map = {
        "mock": RobotPlatform.MOCK,
        "unitree_go1": RobotPlatform.UNITREE_GO1,
        "unitree_h1": RobotPlatform.UNITREE_H1,
        "ros2": RobotPlatform.ROS2_GENERIC,
    }
    
    robot_platform = platform_map.get(args.platform, RobotPlatform.MOCK)
    
    if args.benchmark:
        print("\n[Mode] Benchmarking control frequency...")
        deploy = DeployController(robot_platform=robot_platform)
        deploy.benchmark(duration=args.duration)
    else:
        print(f"\n[Mode] Running control episode ({args.duration}s)...")
        deploy = DeployController(
            robot_platform=robot_platform,
            policy_path=args.policy
        )
        deploy.run_episode(duration=args.duration)
    
    print("\n" + "="*70)
    print("Challenge completed!")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
