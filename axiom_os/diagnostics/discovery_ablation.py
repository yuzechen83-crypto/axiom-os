"""
Discovery 消融实验：噪声、样本量、selector 对 R² 的影响。
运行: python -m axiom_os.diagnostics.discovery_ablation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from axiom_os.engine.discovery import DiscoveryEngine
from axiom_os.benchmarks.seed_utils import set_global_seed


def run_discovery(X, y, var_names, selector="bic", seed=42):
    np.random.seed(seed)
    engine = DiscoveryEngine(use_pysr=False)
    formula, pred, _ = engine.discover_multivariate(
        X, y, var_names=var_names, selector=selector
    )
    y_true = X[:, 0] ** 2 + 0.5 * X[:, 1]
    r2 = 1 - np.mean((pred - y_true) ** 2) / (np.var(y_true) + 1e-12) if formula else 0.0
    return float(r2), formula


def ablation_noise(n_samples=300, seed=42):
    """噪声消融：2%, 5%, 8%, 15%"""
    set_global_seed(seed)
    np.random.seed(seed)
    X = np.random.randn(n_samples, 4).astype(np.float64)
    y_true = X[:, 0] ** 2 + 0.5 * X[:, 1]
    var_names = ["x0", "x1", "x2", "x3"]

    configs = [(0.02, "2%"), (0.05, "5%"), (0.08, "8%"), (0.15, "15%")]
    results = {}
    print("\n[Discovery] 噪声消融 (y = x0^2 + 0.5*x1)")
    print("-" * 50)
    for noise, label in configs:
        y = y_true + noise * np.std(y_true) * np.random.randn(n_samples)
        r2, formula = run_discovery(X, y, var_names, seed=seed)
        results[label] = r2
        print(f"  噪声 {label:4s}: R² = {r2:.4f}")
    return results


def ablation_n_samples(noise=0.05, seed=42):
    """样本量消融：50, 100, 200, 300, 500"""
    set_global_seed(seed)
    configs = [50, 100, 200, 300, 500]
    results = {}
    print("\n[Discovery] 样本量消融 (噪声 5%)")
    print("-" * 50)
    for n in configs:
        np.random.seed(seed)
        X = np.random.randn(n, 4).astype(np.float64)
        y_true = X[:, 0] ** 2 + 0.5 * X[:, 1]
        y = y_true + noise * np.std(y_true) * np.random.randn(n)
        r2, _ = run_discovery(X, y, ["x0", "x1", "x2", "x3"], seed=seed)
        results[n] = r2
        print(f"  n={n:4d}: R² = {r2:.4f}")
    return results


def ablation_selector(n_samples=300, noise=0.05, seed=42):
    """Selector 消融：bic vs aic"""
    set_global_seed(seed)
    np.random.seed(seed)
    X = np.random.randn(n_samples, 4).astype(np.float64)
    y_true = X[:, 0] ** 2 + 0.5 * X[:, 1]
    y = y_true + noise * np.std(y_true) * np.random.randn(n_samples)
    var_names = ["x0", "x1", "x2", "x3"]

    results = {}
    for sel in ["bic", "aic"]:
        r2, formula = run_discovery(X, y, var_names, selector=sel, seed=seed)
        results[sel] = r2
        print(f"  selector={sel}: R² = {r2:.4f}")
    return results


def main():
    print("=" * 60)
    print("Discovery 消融实验")
    print("=" * 60)
    ablation_noise()
    ablation_n_samples()
    print("\n[Discovery] Selector 消融")
    print("-" * 50)
    ablation_selector()
    print("=" * 60)


if __name__ == "__main__":
    main()
