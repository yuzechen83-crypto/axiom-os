"""
RAR 渐近约束 Ablation：测试高/低 g_bar 端约束权重对 Log R² 的影响。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from axiom_os.experiments.discovery_rar import run_rar_discovery


def ablation_test_rar_constraints(
    n_galaxies: int = 20,
    epochs: int = 100,
) -> dict:
    """
    测试 4 种约束配置对 R² 的影响：
    baseline (无), soft (0.05), medium (0.1, 当前), hard (0.2)。
    返回 {config_name: {"r2_linear": ..., "r2_log": ...}}
    """
    configs = [
        ("baseline", 0.0, 0.0),
        ("soft", 0.05, 0.05),
        ("medium", 0.1, 0.1),
        ("hard", 0.2, 0.2),
    ]
    results = {}
    baseline_r2_log = None

    print("\n" + "=" * 60)
    print("RAR constraint ablation")
    print("=" * 60)

    for name, w_high, w_low in configs:
        res = run_rar_discovery(
            n_galaxies=n_galaxies,
            epochs=epochs,
            apply_mass_calibration=True,
            loss_weight_high=w_high,
            loss_weight_low=w_low,
        )
        if res.get("error"):
            print(f"    {name:20s}: 错误 - {res['error']}")
            continue
        r2 = res["r2"]
        r2_log = res["r2_log"]
        results[name] = {"r2_linear": r2, "r2_log": r2_log}
        if baseline_r2_log is None:
            baseline_r2_log = r2_log
        delta = r2_log - baseline_r2_log if baseline_r2_log is not None else 0.0
        print(f"    {name:20s}: Log R2 = {r2_log:.4f}, delta = {delta:+.4f}")

    # 对比表格
    print("\n" + "-" * 50)
    print("Constraint  |  Log R2  |  Impact  |  Note")
    print("-" * 50)
    for name, w_high, w_low in configs:
        if name not in results:
            continue
        r2_log = results[name]["r2_log"]
        if baseline_r2_log is None:
            baseline_r2_log = r2_log
        delta = r2_log - baseline_r2_log
        pct = 100 * (delta / baseline_r2_log) if baseline_r2_log else 0
        if name == "baseline":
            conclusion = ""
        elif name == "medium":
            conclusion = "  (current)"
        else:
            conclusion = ""
        label = "None" if name == "baseline" else f"{w_high:.2f}"
        print(f"  {label:8s}   |  {r2_log:.3f}   |  {pct:+.1f}%   | {conclusion}")
    print("-" * 50)

    # 结论
    if results:
        medium_r2 = results.get("medium", {}).get("r2_log")
        baseline_r2 = results.get("baseline", {}).get("r2_log")
        if medium_r2 is not None and baseline_r2 is not None:
            impact_pct = 100 * (baseline_r2 - medium_r2) / baseline_r2
            if abs(impact_pct) < 1.0:
                print("Conclusion: constraint impact on R2 < 1%, negligible.")
            else:
                print(f"Conclusion: constraint impact ~{impact_pct:.1f}%.")

    return results


if __name__ == "__main__":
    ablation_test_rar_constraints(n_galaxies=20, epochs=100)
