"""
Feynman 风格公式发现基准
对标 SRBench / Feynman-AI 等标准 benchmark 数据集。
运行: python -m axiom_os.experiments.benchmark_feynman
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# 标准 Feynman 风格公式（子集，对标 SRBench/Feynman-AI）
# 多项式可恢复: F1.1-F1.2, F1.4, F1.7；多项式近似: F1.3, F1.5-F1.6, F1.8
# 格式: (name, func, n_vars, n_samples, noise_pct)
FEYNMAN_equations = [
    ("F1.1", lambda X: X[:, 0] * X[:, 1], 2, 500, 0.02),  # y = x0*x1
    ("F1.2", lambda X: X[:, 0] ** 2 + 0.5 * X[:, 1], 2, 500, 0.02),  # y = x0^2 + 0.5*x1
    ("F1.3", lambda X: 1.0 / (1 + X[:, 0] ** 2) + 0.5 * X[:, 1], 2, 500, 0.04),  # 有理式（多项式近似）
    ("F1.4", lambda X: X[:, 0] * X[:, 1] + X[:, 2] ** 2, 3, 500, 0.03),
    ("F1.5", lambda X: np.exp(0.3 * X[:, 0]) + np.log1p(np.clip(X[:, 1] + 1.5, 0.1, None)), 2, 500, 0.04),  # 超越（多项式近似）
    ("F1.6", lambda X: np.sin(X[:, 0]) + 0.5 * X[:, 1], 2, 500, 0.03),
    ("F1.7", lambda X: X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3], 4, 600, 0.03),
    ("F1.8", lambda X: X[:, 0] * X[:, 1] + 0.3 * X[:, 2], 3, 500, 0.02),  # 多项式可恢复
]


def run_single(
    name: str,
    func,
    n_vars: int,
    n_samples: int,
    noise_pct: float,
    seed: int = 42,
    use_pysr: bool = True,
) -> Tuple[float, Optional[str]]:
    """单公式运行，返回 (r2, formula_str)。use_pysr=True 时启用 PySR（exp/log/sin/cos）"""
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(seed)
    X = np.random.randn(n_samples, max(n_vars, 4)).astype(np.float64)
    y_true = func(X[:, :n_vars])
    y = y_true + noise_pct * np.std(y_true) * np.random.randn(n_samples)

    var_names = [f"x{i}" for i in range(max(n_vars, 4))]
    engine = DiscoveryEngine(use_pysr=use_pysr)
    formula, pred, _ = engine.discover_multivariate(X, y, var_names=var_names, selector="bic")
    r2 = 1 - np.mean((pred - y_true) ** 2) / (np.var(y_true) + 1e-12) if formula else 0.0
    r2 = float(r2) if np.isfinite(r2) else 0.0
    return r2, formula


def main(seed: int = 42, n_seeds: int = 1, use_pysr: bool = True) -> None:
    from axiom_os.benchmarks.seed_utils import set_global_seed

    set_global_seed(seed)
    print("=" * 60)
    print("Feynman 风格公式发现基准")
    print("=" * 60)

    results = []
    for name, func, n_vars, n_samples, noise_pct in FEYNMAN_equations:
        r2_list = []
        for s in range(n_seeds):
            r2, formula = run_single(name, func, n_vars, n_samples, noise_pct, seed=seed + s, use_pysr=use_pysr)
            r2_list.append(r2)
        r2_mean = np.mean(r2_list)
        r2_std = np.std(r2_list) if n_seeds > 1 else 0.0
        results.append((name, r2_mean, r2_std, formula))
        fmt = f"{r2_mean:.4f} +/- {r2_std:.4f}" if n_seeds > 1 else f"{r2_mean:.4f}"
        fshort = (formula[:60] + "...") if formula and len(formula) > 60 else (formula or "-")
        print(f"  {name}: R2 = {fmt}  |  formula: {fshort}")

    print("\n" + "-" * 60)
    r2_vals = [r[1] for r in results if np.isfinite(r[1])]
    r2_mean_all = np.mean(r2_vals) if r2_vals else 0.0
    print(f"Avg R2: {r2_mean_all:.4f}")
    print("=" * 60)

    # 保存到 benchmark results
    out_dir = ROOT / "axiom_os" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feynman_benchmark.json"
    import json
    data = {
        "seed": seed,
        "n_seeds": n_seeds,
        "results": [{"name": r[0], "r2_mean": r[1], "r2_std": r[2], "formula": r[3]} for r in results],
        "avg_r2": float(r2_mean_all),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1, help="多 seeds 统计")
    parser.add_argument("--no-pysr", action="store_true", help="禁用 PySR，仅用多项式 Lasso")
    args = parser.parse_args()
    main(seed=args.seed, n_seeds=args.n_seeds, use_pysr=not args.no_pysr)
