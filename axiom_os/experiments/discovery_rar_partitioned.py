"""
RAR 智能分区发现 - 先学单一场景，再整合

Curriculum: rar_low (深 MOND) → rar_mid (过渡) → rar_high (牛顿)
整合: 硬切换 或 软门控
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from axiom_os.datasets.sparc import load_sparc_rar, get_available_sparc_galaxies, SPARC_GALAXIES
from axiom_os.core.partition import RAR_PARTITIONS, get_partitions_curriculum_order
from axiom_os.core.partition_learner import learn_curriculum
from axiom_os.core.partition_integrator import integrate_hard_switch, integrate_soft_gate
from axiom_os.core.hippocampus import Hippocampus


def run_rar_partitioned_discovery(
    n_galaxies: int = 50,
    min_samples: int = 10,
    integrate_mode: str = "soft",
) -> dict:
    """
    分区 RAR 发现：先学各分区，再整合。

    Args:
        n_galaxies: 星系数量
        min_samples: 每分区最少样本
        integrate_mode: "hard" | "soft"

    Returns:
        {partition_results, pred_integrated, r2_integrated, ...}
    """
    avail = get_available_sparc_galaxies()
    n = min(n_galaxies, len(avail) if avail else len(SPARC_GALAXIES))
    g_bar, g_obs, names = load_sparc_rar(n_galaxies=n, use_mock_if_fail=True, use_real=True)
    if g_bar is None or len(g_bar) < 10:
        return {"error": "Insufficient data"}

    X = g_bar.reshape(-1, 1)
    y = g_obs

    # Curriculum 学习
    curriculum = learn_curriculum(X, y, min_samples=min_samples)

    # 整合
    if integrate_mode == "hard":
        pred = integrate_hard_switch(g_bar, curriculum["partition_results"])
    else:
        pred = integrate_soft_gate(g_bar, curriculum["partition_results"])

    # R2
    ss_res = np.sum((pred - g_obs) ** 2)
    ss_tot = np.sum((g_obs - g_obs.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot

    # 存储到 Hippocampus：作为扰动项（联想/直觉），不替代主预测
    hippo = Hippocampus()
    for pid, pr in curriculum["partition_results"].items():
        if "formula" in pr and "error" not in pr:
            formula = pr["formula"]
            # RAR 公式用 x 表示 g_bar，输出 g_obs
            hippo.register_perturbation(
                formula,
                partition_id=pid,
                domain="mechanics",
                output_dim=1,
                g0=pr.get("g0"),
                r2=pr.get("r2"),
                n_samples=pr.get("n_samples"),
            )

    return {
        "partition_results": curriculum["partition_results"],
        "order": curriculum["order"],
        "pred_integrated": pred,
        "r2_integrated": float(r2),
        "g_bar": g_bar,
        "g_obs": g_obs,
        "n_samples": len(g_bar),
        "hippocampus": hippo,
    }


def main():
    print("=" * 60)
    print("RAR 智能分区发现 - 先学单一场景，再整合")
    print("=" * 60)

    res = run_rar_partitioned_discovery(n_galaxies=50, min_samples=10, integrate_mode="soft")

    if "error" in res:
        print(f"Error: {res['error']}")
        return

    print("\n[1] Curriculum 学习顺序:", res["order"])
    print("\n[2] 各分区结果:")
    for pid, pr in res["partition_results"].items():
        print(f"  {pid}: formula={pr.get('formula','?')} g0={pr.get('g0')} R2={pr.get('r2',0):.4f} n={pr.get('n_samples',0)}")

    print(f"\n[3] 整合 R2: {res['r2_integrated']:.4f}")
    print(f"    样本数: {res['n_samples']}")

    print("\n[4] 知识库 (按 partition):")
    for pid in res["order"]:
        laws = res["hippocampus"].list_by_partition(pid)
        for law in laws:
            print(f"  {law['id']}: {law.get('formula','?')}")

    print("=" * 60)


if __name__ == "__main__":
    main()
