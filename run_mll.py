"""
MLL 训练入口 - 多领域学习、协议耦合

用法:
  python run_mll.py                    # 默认：rar + battery + turbulence
  python run_mll.py --domains rar battery  # 仅指定领域
  python run_mll.py --generate my_domain  # 生成新领域协议模板
"""

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="MLL Multi-Domain Learning")
    parser.add_argument("--domains", nargs="+", default=["rar", "battery", "turbulence"],
                        help="Domains to run: rar, battery, turbulence, acrobot")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs per domain")
    parser.add_argument("--crystallize", action="store_true", help="Crystallize discovered formulas")
    parser.add_argument("--generate", type=str, default=None, help="Generate protocol template for new domain")
    args = parser.parse_args()

    if args.generate:
        from axiom_os.mll.domain_protocols import generate_protocol_template
        code = generate_protocol_template(args.generate)
        out_path = ROOT / "axiom_os" / "mll" / f"protocol_{args.generate}.py"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Generated: {out_path}")
        return

    print("=" * 60)
    print("MLL - Multi-Layer Learning")
    print("多领域学习 + 协议耦合")
    print("=" * 60)

    from axiom_os.core import Hippocampus
    from axiom_os.mll import MLLOrchestrator
    from axiom_os.mll.domain_protocols import PROTOCOL_REGISTRY
    from axiom_os.mll.orchestrator import CouplingConfig

    hippocampus = Hippocampus(dim=64, capacity=5000)
    protocols = {k: v for k, v in PROTOCOL_REGISTRY.items() if k in args.domains}
    config = CouplingConfig(order=args.domains, hippocampus_shared=True)
    orchestrator = MLLOrchestrator(protocols=protocols, hippocampus=hippocampus, config=config)

    print(f"\nDomains: {args.domains}, Epochs: {args.epochs}, Crystallize: {args.crystallize}\n")

    t0 = time.perf_counter()
    results = orchestrator.run_all(
        epochs_per_domain={d: args.epochs for d in protocols},
        do_discover=True,
        do_crystallize=args.crystallize,
    )
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print("MLL 完成")
    print("=" * 60)
    ok_count = sum(1 for r in results.values() if r.ok)
    print(f"  {ok_count}/{len(results)} OK, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
