#!/usr/bin/env python
"""
Axiom-OS UPI 入口 - 通过 UPI 接口调用完整 Axiom 系统

用法:
  py -m axiom_os.run_axiom_upi meta_validation [--galaxies 50] [--epochs 500]
  py -m axiom_os.run_axiom_upi acrobot [--steps 200]
  py -m axiom_os.run_axiom_upi turbulence_3d
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Axiom-OS UPI 接口")
    parser.add_argument("mode", choices=["acrobot", "meta_validation", "turbulence_3d"], help="运行模式")
    parser.add_argument("--galaxies", type=int, default=50, help="Meta-Validation 星系数量")
    parser.add_argument("--epochs", type=int, default=500, help="Meta-Validation 训练轮数")
    parser.add_argument("--steps", type=int, default=200, help="Acrobot 步数")
    args = parser.parse_args()

    from axiom_os.api import run_axiom

    print("=" * 70)
    print("Axiom-OS UPI 接口 - 完整系统调用")
    print("=" * 70)

    if args.mode == "meta_validation":
        res = run_axiom(mode="meta_validation", n_galaxies=args.galaxies, meta_epochs=args.epochs)
        print(f"\n数据源: {res['data_source']}, 星系数: {res['n_galaxies']}")
        print(f"NFW 胜: {res['nfw_wins']}, Meta-Axis 胜: {res['meta_wins']}")
        print(f"L 均值: {res['L_mean']:.2f} ± {res['L_std']:.2f}")
        print(f"知识库条目: {len(res['knowledge'])}")
    elif args.mode == "acrobot":
        res = run_axiom(mode="acrobot", n_steps=args.steps)
        print(f"\n知识库: {list(res['knowledge'].keys())}")
    else:
        res = run_axiom(mode="turbulence_3d")
        print(res.get("message", "完成"))

    print("=" * 70)


if __name__ == "__main__":
    main()
