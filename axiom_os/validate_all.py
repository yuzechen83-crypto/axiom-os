"""
Axiom-OS 全流程验证脚本

按建议的下一步操作：跑通 RAR + 电池 + 主循环 Discovery 全流程。
用于验证各模块集成、基准测试、CI 回归。

用法:
  python -m axiom_os.validate_all           # 运行全部（精简参数）
  python -m axiom_os.validate_all --main   # 仅主循环
  python -m axiom_os.validate_all --rar    # 仅 RAR
  python -m axiom_os.validate_all --battery  # 仅 Battery
  python -m axiom_os.validate_all --turbulence  # 仅 Turbulence
  python -m axiom_os.validate_all --diagnose    # RAR g0 诊断
  python -m axiom_os.validate_all --partition   # RAR 智能分区发现
  python -m axiom_os.validate_all --turbulence-partition  # 湍流智能分区发现
  python -m axiom_os.validate_all --turbulence-perturbation  # 湍流+海马体扰动（联想/直觉）
"""

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # axiom_os/ 的父目录 = 项目根
sys.path.insert(0, str(ROOT))


def run_main(quick: bool = True) -> dict:
    """主循环：双摆 + Discovery + 结晶"""
    from axiom_os.main import main, AxiomConfig
    cfg = AxiomConfig(
        T_max=200 if quick else 500,  # 快速验证用 200 步
        discover_interval=50,
        soft_activity_threshold=0.02,
    )
    t0 = time.perf_counter()
    main(config=cfg, use_ai=True)
    elapsed = time.perf_counter() - t0
    return {"ok": True, "elapsed": elapsed, "name": "main"}


def run_rar(n_galaxies: int = 30, epochs: int = 200) -> dict:
    """RAR Discovery"""
    from axiom_os.experiments.discovery_rar import run_rar_discovery, _plot_rar
    t0 = time.perf_counter()
    res = run_rar_discovery(n_galaxies=n_galaxies, epochs=epochs)
    _plot_rar(res, ROOT / "axiom_os" / "discovery_rar_plot.png")
    elapsed = time.perf_counter() - t0
    ok = "error" not in res
    return {"ok": ok, "elapsed": elapsed, "name": "rar", "r2": res.get("r2"), "n_samples": res.get("n_samples")}


def run_battery(quick: bool = False) -> dict:
    """Battery RUL。quick=True 时减少 seeds/epochs/bootstrap，用于 E2E 基准"""
    from axiom_os.main_battery import main
    t0 = time.perf_counter()
    main(quick=quick)
    elapsed = time.perf_counter() - t0
    return {"ok": True, "elapsed": elapsed, "name": "battery"}


def run_turbulence() -> dict:
    """Turbulence 3D 完整测试（含 Coach、Baseline 对比）"""
    from axiom_os.run_turbulence_full import main
    t0 = time.perf_counter()
    main()
    elapsed = time.perf_counter() - t0
    return {"ok": True, "elapsed": elapsed, "name": "turbulence"}


def run_partition() -> dict:
    """RAR 智能分区发现"""
    from axiom_os.experiments.discovery_rar_partitioned import run_rar_partitioned_discovery
    t0 = time.perf_counter()
    res = run_rar_partitioned_discovery(n_galaxies=50, min_samples=10, integrate_mode="soft")
    elapsed = time.perf_counter() - t0
    ok = "error" not in res
    return {"ok": ok, "elapsed": elapsed, "name": "partition", "r2": res.get("r2_integrated"), "order": res.get("order")}


def run_turbulence_partition() -> dict:
    """3D Turbulence 智能分区发现"""
    from axiom_os.experiments.discovery_turbulence_partitioned import run_turbulence_partitioned_discovery
    t0 = time.perf_counter()
    res = run_turbulence_partitioned_discovery(min_samples=50, partition_epochs=300)
    elapsed = time.perf_counter() - t0
    ok = "error" not in res
    return {
        "ok": ok,
        "elapsed": elapsed,
        "name": "turbulence_partition",
        "mae_u": res.get("mae_u"),
        "mae_v": res.get("mae_v"),
        "mae_u_baseline": res.get("mae_u_baseline"),
        "mae_v_baseline": res.get("mae_v_baseline"),
    }


def run_diagnose() -> dict:
    """RAR g0 诊断：分析 g0=321 vs g0=3700 残差"""
    from axiom_os.experiments.diagnose_rar_g0 import main as diagnose_main
    t0 = time.perf_counter()
    diagnose_main()
    elapsed = time.perf_counter() - t0
    return {"ok": True, "elapsed": elapsed, "name": "diagnose"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Axiom-OS 全流程验证")
    parser.add_argument("--main", action="store_true", help="运行主循环")
    parser.add_argument("--rar", action="store_true", help="运行 RAR Discovery")
    parser.add_argument("--battery", action="store_true", help="运行 Battery RUL")
    parser.add_argument("--turbulence", action="store_true", help="运行 Turbulence")
    parser.add_argument("--diagnose", action="store_true", help="运行 RAR g0 诊断 (g0=321 vs 3700)")
    parser.add_argument("--partition", action="store_true", help="运行 RAR 智能分区发现")
    parser.add_argument("--turbulence-partition", action="store_true", help="运行湍流智能分区发现")
    parser.add_argument("--turbulence-perturbation", action="store_true", help="湍流+海马体扰动（联想/直觉）")
    parser.add_argument("--quick", action="store_true", default=True, help="精简参数（默认）")
    args = parser.parse_args()

    run_all = not (args.main or args.rar or args.battery or args.turbulence or args.diagnose or
                   args.partition or args.turbulence_partition or args.turbulence_perturbation)

    results = []
    print("=" * 60)
    print("Axiom-OS 全流程验证")
    print("=" * 60)

    if run_all or args.main:
        print("\n[1/8] 主循环 (Acrobot + Discovery)...")
        try:
            r = run_main(quick=args.quick)
            results.append(r)
            print(f"  OK  {r['elapsed']:.1f}s")
        except Exception as e:
            results.append({"ok": False, "name": "main", "error": str(e)})
            print(f"  FAIL: {e}")

    if run_all or args.rar:
        print("\n[2/8] RAR Discovery...")
        try:
            r = run_rar(n_galaxies=30, epochs=200)
            results.append(r)
            print(f"  OK  {r['elapsed']:.1f}s  R2={r.get('r2', 'N/A')}  n={r.get('n_samples', 'N/A')}")
        except Exception as e:
            results.append({"ok": False, "name": "rar", "error": str(e)})
            print(f"  FAIL: {e}")

    if run_all or args.battery:
        print("\n[3/8] Battery RUL...")
        try:
            r = run_battery()
            results.append(r)
            print(f"  OK  {r['elapsed']:.1f}s")
        except Exception as e:
            results.append({"ok": False, "name": "battery", "error": str(e)})
            print(f"  FAIL: {e}")

    if run_all or args.turbulence:
        print("\n[4/8] Turbulence 3D (Coach + Baseline)...")
        try:
            r = run_turbulence()
            results.append(r)
            print(f"  OK  {r['elapsed']:.1f}s")
        except Exception as e:
            results.append({"ok": False, "name": "turbulence", "error": str(e)})
            print(f"  FAIL: {e}")

    if run_all or args.partition:
        print("\n[5/8] RAR 智能分区发现...")
        try:
            r = run_partition()
            results.append(r)
            print(f"  OK  {r['elapsed']:.1f}s  R2={r.get('r2','N/A')}  order={r.get('order','')}")
        except Exception as e:
            results.append({"ok": False, "name": "partition", "error": str(e)})
            print(f"  FAIL: {e}")

    if run_all or args.turbulence_partition:
        print("\n[6/8] 湍流智能分区发现...")
        try:
            r = run_turbulence_partition()
            results.append(r)
            if r.get("ok"):
                mu, mv = r.get("mae_u"), r.get("mae_v")
                bu, bv = r.get("mae_u_baseline"), r.get("mae_v_baseline")
                print(f"  OK  {r['elapsed']:.1f}s  MAE u={mu:.4f} v={mv:.4f} (baseline u={bu:.4f} v={bv:.4f})")
            else:
                print(f"  FAIL: {r.get('error', 'unknown')}")
        except Exception as e:
            results.append({"ok": False, "name": "turbulence_partition", "error": str(e)})
            print(f"  FAIL: {e}")

    if run_all or args.turbulence_perturbation:
        print("\n[7/8] 湍流+海马体扰动...")
        try:
            from axiom_os.experiments.discovery_turbulence_partitioned import (
                run_turbulence_partitioned_discovery,
                run_turbulence_with_perturbation,
            )
            t0 = time.perf_counter()
            res_part = run_turbulence_partitioned_discovery(min_samples=50, partition_epochs=300)
            if "error" in res_part:
                raise RuntimeError(res_part["error"])
            res = run_turbulence_with_perturbation(
                res_part["hippocampus"],
                res_part["u_mean"],
                res_part["v_mean"],
                alpha_pert=0.1,
                epochs=20000,
                use_learned_perturbation_gate=True,
            )
            elapsed = time.perf_counter() - t0
            results.append({"ok": "error" not in res, "elapsed": elapsed, "name": "turbulence_perturbation", **res})
            if res.get("mae_u") is not None:
                print(f"  OK  {elapsed:.1f}s  MAE u={res['mae_u']:.4f} v={res['mae_v']:.4f} (可学习直觉门控)")
            else:
                print(f"  FAIL: {res.get('error', 'unknown')}")
        except Exception as e:
            results.append({"ok": False, "name": "turbulence_perturbation", "error": str(e)})
            print(f"  FAIL: {e}")

    if run_all or args.diagnose:
        print("\n[8/8] RAR g0 诊断...")
        try:
            r = run_diagnose()
            results.append(r)
            print(f"  OK  {r['elapsed']:.1f}s")
        except Exception as e:
            results.append({"ok": False, "name": "diagnose", "error": str(e)})
            print(f"  FAIL: {e}")

    print("\n" + "-" * 60)
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"完成: {n_ok}/{len(results)} 通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
