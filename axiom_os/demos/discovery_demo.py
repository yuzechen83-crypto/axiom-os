#!/usr/bin/env python3
"""
端到端 Demo：从数据自动发现物理定律

任务：给定 y = x0² + 0.5*x1 的带噪数据，恢复符号公式
对比：Axiom Discovery (多项式) vs 线性回归
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    n = 400
    X = np.random.randn(n, 4).astype(np.float64)
    y_true = X[:, 0] ** 2 + 0.5 * X[:, 1]
    y = y_true + 0.08 * np.random.randn(n)

    print("=" * 60)
    print("Axiom-OS 公式发现 Demo")
    print("真实公式: y = x0² + 0.5*x1")
    print("=" * 60)

    # Axiom Discovery
    engine = DiscoveryEngine(use_pysr=False)
    formula, pred, coefs = engine.discover_multivariate(
        X, y, var_names=["x0", "x1", "x2", "x3"], selector="bic"
    )
    r2_axiom = 1 - np.mean((pred - y_true) ** 2) / (np.var(y_true) + 1e-12)

    # Baseline: 线性回归
    X_lin = np.column_stack([X[:, 0], X[:, 1], np.ones(n)])
    coef, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
    pred_lin = X_lin @ coef
    r2_lin = 1 - np.mean((pred_lin - y_true) ** 2) / (np.var(y_true) + 1e-12)

    print("\n📊 结果对比")
    print("-" * 40)
    print(f"  线性回归 R²:  {r2_lin:.4f}")
    print(f"  Axiom Discovery R²: {r2_axiom:.4f}")
    print(f"  R² 提升: {(r2_axiom - r2_lin)*100:+.1f}%")
    print("-" * 40)
    print(f"\n  发现公式: {formula or '(未发现)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
