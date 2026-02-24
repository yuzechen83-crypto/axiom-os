"""
Axiom-OS 性能基准测试入口

用法:
  python -m axiom_os.benchmarks.run_benchmarks                    # 默认 standard
  python -m axiom_os.benchmarks.run_benchmarks --config quick      # 快速
  python -m axiom_os.benchmarks.run_benchmarks --config full        # 完整
  python -m axiom_os.benchmarks.run_benchmarks --unit              # 仅单元
  python -m axiom_os.benchmarks.run_benchmarks --hard              # 高难度 Discovery 套件
  python -m axiom_os.benchmarks.run_benchmarks -o report.json      # 输出 JSON
  python -m axiom_os.benchmarks.run_benchmarks --report             # 生成 MD/HTML 报告
  python -m axiom_os.benchmarks.run_benchmarks --trend             # 生成趋势图
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "axiom_os" / "benchmarks" / "results"
sys.path.insert(0, str(ROOT))

RESULTS: List[Dict[str, Any]] = []


def _record(name: str, metric: str, value: float, unit: str = "s", extra: Dict = None):
    entry = {"name": name, "metric": metric, "value": value, "unit": unit}
    if extra:
        entry["extra"] = extra
    RESULTS.append(entry)
    return value


def bench_rcln_forward(n_warmup: int = 10, n_runs: int = 100, batch_size: int = 1024) -> float:
    """RCLN forward 延迟 (ms)"""
    import torch
    from axiom_os.layers.rcln import RCLNLayer

    rcln = RCLNLayer(input_dim=4, hidden_dim=64, output_dim=2, hard_core_func=None, lambda_res=1.0)
    x = torch.randn(batch_size, 4)
    for _ in range(n_warmup):
        _ = rcln(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = rcln(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) / n_runs * 1000
    throughput = batch_size * n_runs / (t1 - t0) if (t1 - t0) > 0 else 0
    _record("rcln_forward", "latency_ms", elapsed_ms, "ms", {"batch_size": batch_size})
    _record("rcln_forward", "throughput", throughput, "samples/s", {"batch_size": batch_size})
    return elapsed_ms


def bench_discovery(n_samples: int = 200, n_runs: int = 3) -> float:
    """Discovery discover_multivariate 单次耗时 (s)"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    X = np.random.randn(n_samples, 4).astype(np.float64)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    engine = DiscoveryEngine(use_pysr=False)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _, _, _ = engine.discover_multivariate(X, y, var_names=["x0", "x1", "x2", "x3"], selector="bic")
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    _record("discovery_multivariate", "latency_s", avg, "s", {"n_samples": n_samples})
    return avg


def bench_discovery_vs_baseline(n_samples: int = 300) -> None:
    """Discovery vs Baseline: y = x0**2 + 0.5*x1 (多项式优于线性)"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    X = np.random.randn(n_samples, 4).astype(np.float64)
    y_true = X[:, 0] ** 2 + 0.5 * X[:, 1]
    y = y_true + 0.05 * np.random.randn(n_samples)

    # Axiom Discovery (多项式形式可拟合 x0^2)
    engine = DiscoveryEngine(use_pysr=False)
    formula_axiom, pred_axiom, _ = engine.discover_multivariate(
        X, y, var_names=["x0", "x1", "x2", "x3"], selector="bic"
    )
    r2_axiom = 1 - np.mean((pred_axiom - y_true) ** 2) / (np.var(y_true) + 1e-12) if formula_axiom else 0

    # Baseline: 线性回归 (无法拟合 x0^2)
    X_lin = np.column_stack([X[:, 0], X[:, 1], np.ones(n_samples)])
    coef, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
    pred_lin = X_lin @ coef
    r2_lin = 1 - np.mean((pred_lin - y_true) ** 2) / (np.var(y_true) + 1e-12)

    formula_short = (formula_axiom[:60] + "..") if formula_axiom and len(formula_axiom) > 60 else (formula_axiom or "N/A")
    _record("discovery_vs_baseline", "r2_axiom", float(r2_axiom), "", {"formula": formula_short})
    _record("discovery_vs_baseline", "r2_linear", float(r2_lin), "", {})
    _record("discovery_vs_baseline", "r2_gain", float(r2_axiom - r2_lin), "", {"n_samples": n_samples})


def bench_discovery_robustness() -> None:
    """Discovery 鲁棒性：不同噪声水平下的 R²"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    r2_list = []
    for noise in [0.02, 0.08, 0.15]:
        X = np.random.randn(300, 4).astype(np.float64)
        y_true = X[:, 0] ** 2 + 0.5 * X[:, 1]
        y = y_true + noise * np.random.randn(300)
        engine = DiscoveryEngine(use_pysr=False)
        formula, pred, _ = engine.discover_multivariate(
            X, y, var_names=["x0", "x1", "x2", "x3"], selector="bic"
        )
        r2 = 1 - np.mean((pred - y_true) ** 2) / (np.var(y_true) + 1e-12) if formula else 0
        r2_list.append(float(r2))
    r2_mean = sum(r2_list) / len(r2_list)
    r2_min = min(r2_list)
    _record("discovery_robustness", "r2_mean", r2_mean, "", {"noise_levels": "0.02,0.08,0.15"})
    _record("discovery_robustness", "r2_min", r2_min, "", {})


def bench_discovery_vs_sklearn_poly(n: int = 300) -> None:
    """Discovery vs sklearn 多项式回归（始终可用，标准配置运行）"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    X = np.random.randn(n, 4).astype(np.float64)
    y_true = X[:, 0] ** 2 + 0.5 * X[:, 1]
    y = y_true + 0.05 * np.random.randn(n)

    engine = DiscoveryEngine(use_pysr=False)
    _, pred_axiom, _ = engine.discover_multivariate(X, y, var_names=["x0", "x1", "x2", "x3"], selector="bic")
    r2_axiom = 1 - np.mean((pred_axiom - y_true) ** 2) / (np.var(y_true) + 1e-12)

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    model = Ridge(alpha=1e-3)
    model.fit(X_poly, y)
    pred_poly = model.predict(X_poly)
    r2_poly = 1 - np.mean((pred_poly - y_true) ** 2) / (np.var(y_true) + 1e-12)
    _record("discovery_vs_sklearn_poly", "r2_poly", float(r2_poly), "", {})
    _record("discovery_vs_sklearn_poly", "r2_axiom", float(r2_axiom), "", {})


def bench_discovery_vs_pysr_sindy() -> None:
    """Discovery vs 可选 PySR/SINDy（需安装，--compare-pysr 时运行）"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 4).astype(np.float64)
    y_true = X[:, 0] ** 2 + 0.5 * X[:, 1]
    y = y_true + 0.05 * np.random.randn(n)

    engine = DiscoveryEngine(use_pysr=False)
    _, pred_axiom, _ = engine.discover_multivariate(X, y, var_names=["x0", "x1", "x2", "x3"], selector="bic")
    r2_axiom = 1 - np.mean((pred_axiom - y_true) ** 2) / (np.var(y_true) + 1e-12)

    # PySR（需 Julia，首次较慢）
    try:
        from pysr import PySRRegressor
        model = PySRRegressor(niterations=5, binary_operators=["+", "*"], unary_operators=["square"])
        model.fit(X, y)
        pred_pysr = np.asarray(model.predict(X)).ravel()
        if pred_pysr.shape[0] == n and np.all(np.isfinite(pred_pysr)):
            r2_pysr = 1 - np.mean((pred_pysr - y_true) ** 2) / (np.var(y_true) + 1e-12)
            _record("discovery_vs_pysr", "r2_pysr", float(r2_pysr), "", {})
            _record("discovery_vs_pysr", "r2_axiom", float(r2_axiom), "", {})
    except (ImportError, Exception):
        pass

    # SINDy
    try:
        from pysindy import SINDy
        from pysindy.feature_library import PolynomialLibrary
        lib = PolynomialLibrary(degree=2)
        model = SINDy(feature_library=lib)
        model.fit(X, y.reshape(-1, 1))
        pred_sindy = model.predict(X)
        if pred_sindy is not None:
            pred_sindy = np.asarray(pred_sindy).ravel()
            if pred_sindy.shape[0] == n and np.all(np.isfinite(pred_sindy)):
                r2_sindy = 1 - np.mean((pred_sindy - y_true) ** 2) / (np.var(y_true) + 1e-12)
                _record("discovery_vs_sindy", "r2_sindy", float(r2_sindy), "", {})
                _record("discovery_vs_sindy", "r2_axiom", float(r2_axiom), "", {})
    except (ImportError, Exception):
        pass


# -----------------------------------------------------------------------------
# 高难度 Discovery 基准（专业人士级）
# -----------------------------------------------------------------------------


def bench_discovery_hard_sparse(n_samples: int = 200, n_vars: int = 20, noise: float = 0.18) -> None:
    """高维稀疏：20 维中仅 x3,x7 有效，y = sin(x3) + 0.5*x7，18% 噪声"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    X = np.random.randn(n_samples, n_vars).astype(np.float64)
    y_true = np.sin(X[:, 3]) + 0.5 * X[:, 7]
    y = y_true + noise * np.std(y_true) * np.random.randn(n_samples)

    engine = DiscoveryEngine(use_pysr=False)
    var_names = [f"x{i}" for i in range(n_vars)]
    formula, pred, _ = engine.discover_multivariate(X, y, var_names=var_names, selector="bic")
    r2 = 1 - np.mean((pred - y_true) ** 2) / (np.var(y_true) + 1e-12) if formula else 0
    _record("hard_sparse", "r2", float(r2), "", {"n_vars": n_vars, "noise_pct": int(noise * 100)})
    _record("hard_sparse", "formula_ok", 1.0 if formula and len(formula) > 5 else 0, "", {})


def bench_discovery_hard_extrapolation(n_train: int = 150, n_test: int = 150, noise: float = 0.08) -> None:
    """外推：训练 x∈[0,1]，测试 x∈[1,2]；真实 y = exp(0.3*x0) + log(1+x1)。仅训练集拟合，基线在测试集评估。"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    X_train = np.random.uniform(0, 1, (n_train, 4)).astype(np.float64)
    y_true_train = np.exp(0.3 * X_train[:, 0]) + np.log1p(X_train[:, 1])
    y_train = y_true_train + noise * np.std(y_true_train) * np.random.randn(n_train)
    X_test = np.random.uniform(1, 2, (n_test, 4)).astype(np.float64)
    y_true_test = np.exp(0.3 * X_test[:, 0]) + np.log1p(X_test[:, 1])

    engine = DiscoveryEngine(use_pysr=False)
    formula, pred_train, _ = engine.discover_multivariate(
        X_train, y_train, var_names=["x0", "x1", "x2", "x3"], selector="bic"
    )
    r2_train = 1 - np.mean((pred_train - y_true_train) ** 2) / (np.var(y_true_train) + 1e-12) if formula else 0
    # 线性基线：训练集拟合，测试集外推（通常崩）
    X_lin = np.column_stack([X_train[:, 0], X_train[:, 1], np.ones(n_train)])
    coef, _, _, _ = np.linalg.lstsq(X_lin, y_train, rcond=None)
    X_test_lin = np.column_stack([X_test[:, 0], X_test[:, 1], np.ones(n_test)])
    pred_lin_test = X_test_lin @ coef
    r2_extrap_linear = 1 - np.mean((pred_lin_test - y_true_test) ** 2) / (np.var(y_true_test) + 1e-12)
    _record("hard_extrapolation", "r2_train", float(r2_train), "", {})
    _record("hard_extrapolation", "r2_extrap_linear", float(r2_extrap_linear), "", {"n_train": n_train, "n_test": n_test})


def bench_discovery_hard_small_n(n_samples: int = 50, noise: float = 0.10) -> None:
    """小样本：n=50，y = x0*x1 + x2**2 + 0.3*x3"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    X = np.random.randn(n_samples, 4).astype(np.float64)
    y_true = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + 0.3 * X[:, 3]
    y = y_true + noise * np.std(y_true) * np.random.randn(n_samples)

    engine = DiscoveryEngine(use_pysr=False)
    formula, pred, _ = engine.discover_multivariate(X, y, var_names=["x0", "x1", "x2", "x3"], selector="bic")
    r2 = 1 - np.mean((pred - y_true) ** 2) / (np.var(y_true) + 1e-12) if formula else 0
    _record("hard_small_n", "r2", float(r2), "", {"n_samples": n_samples})


def bench_discovery_hard_feynman(n_samples: int = 300, noise: float = 0.06) -> None:
    """Feynman 风格：y = 1/(1+x0**2) + 0.5*x1（有理式 + 线性）"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    X = np.random.randn(n_samples, 4).astype(np.float64)
    y_true = 1.0 / (1 + X[:, 0] ** 2) + 0.5 * X[:, 1]
    y = y_true + noise * np.std(y_true) * np.random.randn(n_samples)

    engine = DiscoveryEngine(use_pysr=False)
    formula, pred, _ = engine.discover_multivariate(X, y, var_names=["x0", "x1", "x2", "x3"], selector="bic")
    r2 = 1 - np.mean((pred - y_true) ** 2) / (np.var(y_true) + 1e-12) if formula else 0
    _record("hard_feynman", "r2", float(r2), "", {"formula_type": "1/(1+x0^2)+0.5*x1"})


def bench_lorenz_recovery(n_steps: int = 2000, dt: float = 0.01, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> None:
    """Lorenz 混沌系统：从轨迹数值导数恢复 dx/dt 与 (y-x) 的线性关系系数（σ）"""
    import numpy as np
    from axiom_os.engine.discovery import DiscoveryEngine

    np.random.seed(42)
    # 生成 Lorenz 轨迹
    x, y, z = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    x[0], y[0], z[0] = 1.0, 1.0, 1.0
    for i in range(n_steps - 1):
        x[i + 1] = x[i] + dt * (sigma * (y[i] - x[i]))
        y[i + 1] = y[i] + dt * (x[i] * (rho - z[i]) - y[i])
        z[i + 1] = z[i] + dt * (x[i] * y[i] - beta * z[i])
    # 数值导数（中心差分）
    dx_dt = np.gradient(x, dt)
    dy_dt = np.gradient(y, dt)
    dz_dt = np.gradient(z, dt)
    # 特征: (x, y, z), 目标: dx_dt = sigma*(y-x)
    X = np.column_stack([x, y, z]).astype(np.float64)
    y_target = dx_dt
    # 去掉首尾若干点减小边界效应
    trim = 20
    X_t, y_t = X[trim:-trim], y_target[trim:-trim]
    engine = DiscoveryEngine(use_pysr=False)
    formula, pred, _ = engine.discover_multivariate(
        X_t, y_t, var_names=["x", "y", "z"], selector="bic"
    )
    r2 = 1 - np.mean((pred - y_t) ** 2) / (np.var(y_t) + 1e-12) if formula else 0
    _record("hard_lorenz", "r2_dx_dt", float(r2), "", {"sigma_true": sigma})
    _record("hard_lorenz", "n_steps", float(n_steps), "", {})


def bench_causal_discovery(n_samples: int = 500, state_dim: int = 4) -> None:
    """约束因果发现：从线性 SCM 数据恢复 DAG，用辛约束过滤，计算边 F1。"""
    import numpy as np
    from axiom_os.core.causal_constraints import get_symplectic_causal_edges, allowed_edges
    from axiom_os.causal.discovery import discover_causal_graph, adjacency_to_edges

    np.random.seed(42)
    n = state_dim
    # 真实 DAG 为辛允许边的子集：0->2, 1->3（q 驱动 dp）
    true_edges = [(0, 2), (1, 3)]
    X = np.zeros((n_samples, n))
    X[:, 0] = 0.5 * np.random.randn(n_samples)
    X[:, 1] = 0.5 * np.random.randn(n_samples)
    X[:, 2] = 0.6 * X[:, 0] + 0.3 * np.random.randn(n_samples)
    X[:, 3] = 0.6 * X[:, 1] + 0.3 * np.random.randn(n_samples)
    X = X.astype(np.float64)

    allowed = allowed_edges(n, use_symplectic=True, use_light_cone=False)
    adj = discover_causal_graph(X, allowed_edges=allowed, alpha=0.05, use_cond_indep=True, max_cond_size=1)
    pred_edges = set(adjacency_to_edges(adj))
    true_set = set(true_edges)
    tp = len(pred_edges & true_set)
    prec = tp / len(pred_edges) if pred_edges else 0
    rec = tp / len(true_set) if true_set else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    _record("causal_discovery", "f1_edges", float(f1), "", {"state_dim": n, "n_samples": n_samples})
    _record("causal_discovery", "precision", float(prec), "", {})
    _record("causal_discovery", "recall", float(rec), "", {})


def bench_causal_do_effect(n_samples: int = 400) -> None:
    """SCM do 干预效应：从数据拟合线性 SCM，估计 do(X_0=1) 的效应，与真实 SCM 对比。"""
    import numpy as np
    from axiom_os.causal.scm import LinearSCM, fit_linear_scm_from_data

    np.random.seed(42)
    B_true = np.array([
        [0, 0, 0.5, 0],
        [0, 0, 0, 0.5],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float64)
    scm_true = LinearSCM(B_true)
    X = scm_true.sample(n_samples, u_std=np.array([0.5, 0.5, 0.3, 0.3]), seed=42)
    adj = (B_true != 0).astype(np.float64)
    scm_fit = fit_linear_scm_from_data(X, adj)
    x_one = X[0:1]
    do_val = 1.0
    out_true = scm_true.do({0: do_val}, x_one)
    out_fit = scm_fit.do({0: do_val}, x_one)
    mae = float(np.abs(out_true - out_fit).mean())
    _record("causal_do_effect", "mae", mae, "", {"n_samples": n_samples})
    _record("causal_do_effect", "ok", 1.0 if mae < 0.5 else 0.0, "", {})


def run_hard_benchmarks():
    """运行高难度 Discovery 套件"""
    print("\n[Hard] 高维稀疏 (20 维, sin(x3)+0.5*x7, 18% 噪声)...")
    bench_discovery_hard_sparse()
    print("[Hard] 外推 (train [0,1], test [1,2], exp+log)...")
    bench_discovery_hard_extrapolation()
    print("[Hard] 小样本 (n=50, x0*x1+x2^2+0.3*x3)...")
    bench_discovery_hard_small_n()
    print("[Hard] Feynman 风格 (1/(1+x0^2)+0.5*x1)...")
    bench_discovery_hard_feynman()
    print("[Hard] Lorenz 混沌 (从轨迹恢复 dx/dt)...")
    bench_lorenz_recovery()
    print("[Hard] 约束因果发现 (辛结构 DAG 恢复)...")
    bench_causal_discovery()
    print("[Hard] SCM do 干预效应...")
    bench_causal_do_effect()


def bench_hippocampus(n_queries: int = 100) -> float:
    """Hippocampus retrieve_by_query 延迟 (ms)"""
    from axiom_os.core.hippocampus import Hippocampus, _formula_to_callable

    h = Hippocampus(use_semantic_rag=False)
    h.register_perturbation("x[0]**2 + x[1]", "rar_low", "mechanics", output_dim=1)
    h.register_perturbation("x[0] * 0.5", "rar_high", "mechanics", output_dim=1)
    t0 = time.perf_counter()
    for _ in range(n_queries):
        _ = h.retrieve_by_query("RAR rotation", top_k=5, domain="mechanics")
    elapsed_ms = (time.perf_counter() - t0) / n_queries * 1000
    _record("hippocampus_retrieve", "latency_ms", elapsed_ms, "ms", {"n_queries": n_queries})
    return elapsed_ms


def bench_coach(n_samples: int = 10000, n_runs: int = 20) -> float:
    """Coach coach_score / coach_loss_torch 延迟 (ms)"""
    import numpy as np
    import torch
    from axiom_os.coach import coach_score, coach_loss_torch

    np.random.seed(42)
    y_pred = np.random.randn(n_samples, 2).astype(np.float64) * 5
    x = np.random.rand(n_samples, 4)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = coach_score(x, y_pred, "fluids")
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
    _record("coach_score", "latency_ms", elapsed_ms, "ms", {"n_samples": n_samples})

    y_t = torch.from_numpy(y_pred).float()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = coach_loss_torch(y_t, "fluids")
    elapsed_ms2 = (time.perf_counter() - t0) / n_runs * 1000
    _record("coach_loss_torch", "latency_ms", elapsed_ms2, "ms", {"n_samples": n_samples})
    return elapsed_ms


def bench_bundle_field(n_queries: int = 500) -> float:
    """Bundle Field Section.select 延迟 (ms)"""
    import numpy as np
    from axiom_os.core.bundle_field import MetaAxisBundleField, BasePoint, Regime, ResidualRole
    from axiom_os.core.hippocampus import _formula_to_callable

    bf = MetaAxisBundleField()
    fn = _formula_to_callable("x[0]**2", 1)
    bf.crystallize("x[0]**2", fn, 1, "mechanics", "rar_low", formula_id="law_bf")
    x = np.random.rand(10, 4)
    t0 = time.perf_counter()
    for _ in range(n_queries):
        _ = bf.select(x, "mechanics")
    elapsed_ms = (time.perf_counter() - t0) / n_queries * 1000
    _record("bundle_field_select", "latency_ms", elapsed_ms, "ms", {"n_queries": n_queries})
    return elapsed_ms


def bench_turbulence(epochs: int = 200, batch_size: int = None) -> float:
    """湍流训练 200 epochs 耗时 (s)"""
    import numpy as np
    import torch
    from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
    from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
    from axiom_os.layers.rcln import RCLNLayer
    from axiom_os.coach import coach_loss_torch

    np.random.seed(42)
    torch.manual_seed(42)
    coords, targets, _ = load_atmospheric_turbulence_3d(
        n_lat=3, n_lon=3, delta_deg=0.15, forecast_days=3, use_synthetic_if_fail=True
    )
    n = len(coords)
    split = int(0.8 * n)
    X_train = torch.from_numpy(coords[:split]).float()
    Y_train = torch.from_numpy(targets[:split]).float()
    u_mean = float(Y_train[:, 0].mean())
    v_mean = float(Y_train[:, 1].mean())
    hard_core = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=5.0, use_enhanced=False)
    rcln = RCLNLayer(input_dim=4, hidden_dim=64, output_dim=2, hard_core_func=hard_core, lambda_res=1.0)
    opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for ep in range(epochs):
        opt.zero_grad()
        pred = rcln(X_train)
        loss = torch.nn.functional.huber_loss(pred, Y_train, delta=1.0) + 0.15 * coach_loss_torch(pred)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - t0
    _record("turbulence_training", "elapsed_s", elapsed, "s", {"epochs": epochs})
    _record("turbulence_training", "epoch_time_ms", elapsed / epochs * 1000, "ms", {})
    return elapsed


def bench_rar(n_galaxies: int = 20, epochs: int = 100, batch_size: int = None) -> float:
    """RAR Discovery 耗时 (s)"""
    from axiom_os.experiments.discovery_rar import run_rar_discovery

    t0 = time.perf_counter()
    res = run_rar_discovery(n_galaxies=n_galaxies, epochs=epochs)
    elapsed = time.perf_counter() - t0
    ok = "error" not in res
    _record("rar_discovery", "elapsed_s", elapsed, "s", {"n_galaxies": n_galaxies, "epochs": epochs, "ok": ok})
    _record("rar_discovery", "r2", res.get("r2", 0), "", {"n_samples": res.get("n_samples", 0)})
    if res.get("r2_log") is not None:
        _record("rar_discovery", "r2_log", res["r2_log"], "", {})
    return elapsed


def bench_validate_all() -> float:
    """validate_all 全流程耗时 (s)"""
    from axiom_os.validate_all import (
        run_main, run_rar, run_battery, run_turbulence,
        run_partition, run_turbulence_partition, run_diagnose,
    )

    total = 0.0
    for name, fn in [
        ("main", lambda: run_main(quick=True)),
        ("rar", lambda: run_rar(n_galaxies=20, epochs=100)),
        ("battery", lambda: run_battery()),
        ("turbulence", lambda: run_turbulence()),
        ("partition", lambda: run_partition()),
        ("turbulence_partition", lambda: run_turbulence_partition()),
        ("diagnose", lambda: run_diagnose()),
    ]:
        try:
            t0 = time.perf_counter()
            r = fn()
            elapsed = time.perf_counter() - t0
            total += elapsed
            _record(f"e2e_{name}", "elapsed_s", elapsed, "s", {"ok": r.get("ok", False)})
        except Exception as e:
            _record(f"e2e_{name}", "elapsed_s", -1, "s", {"ok": False, "error": str(e)})
    _record("e2e_total", "elapsed_s", total, "s", {})
    return total


def run_unit_benchmarks(robustness: bool = False, compare_pysr: bool = False):
    """单元级基准"""
    print("\n[Unit] RCLN forward...")
    bench_rcln_forward()
    print("[Unit] Discovery...")
    bench_discovery()
    print("[Unit] Discovery vs Baseline (公式恢复)...")
    bench_discovery_vs_baseline()
    print("[Unit] Discovery vs sklearn 多项式...")
    bench_discovery_vs_sklearn_poly()
    if robustness:
        print("[Unit] Discovery 鲁棒性 (噪声 0.02/0.08/0.15)...")
        bench_discovery_robustness()
    if compare_pysr:
        print("[Unit] Discovery vs PySR/SINDy...")
        bench_discovery_vs_pysr_sindy()
    print("[Unit] Hippocampus...")
    bench_hippocampus()
    print("[Unit] Coach...")
    bench_coach()
    print("[Unit] Bundle Field...")
    bench_bundle_field()


def run_integration_benchmarks(cfg=None):
    """集成级基准"""
    from .configs import STANDARD
    cfg = cfg or STANDARD
    print(f"\n[Integration] Turbulence ({cfg.turbulence_epochs} epochs)...")
    bench_turbulence(epochs=cfg.turbulence_epochs)
    print(f"[Integration] RAR Discovery (n={cfg.rar_galaxies}, ep={cfg.rar_epochs})...")
    bench_rar(n_galaxies=cfg.rar_galaxies, epochs=cfg.rar_epochs)


def run_e2e_benchmarks():
    """端到端基准"""
    print("\n[E2E] validate_all 全流程...")
    bench_validate_all()


def bench_memory():
    """峰值内存 (MB) - 湍流训练期间"""
    try:
        import tracemalloc
        import numpy as np
        import torch
        from axiom_os.core.wind_hard_core import make_wind_hard_core_adaptive
        from axiom_os.datasets.atmospheric_turbulence import load_atmospheric_turbulence_3d
        from axiom_os.layers.rcln import RCLNLayer

        tracemalloc.start()
        np.random.seed(42)
        torch.manual_seed(42)
        coords, targets, _ = load_atmospheric_turbulence_3d(
            n_lat=2, n_lon=2, delta_deg=0.2, forecast_days=2, use_synthetic_if_fail=True
        )
        split = int(0.8 * len(coords))
        X = torch.from_numpy(coords[:split]).float()
        Y = torch.from_numpy(targets[:split]).float()
        u_mean, v_mean = float(Y[:, 0].mean()), float(Y[:, 1].mean())
        hc = make_wind_hard_core_adaptive(u_mean, v_mean, threshold=5.0, use_enhanced=False)
        rcln = RCLNLayer(input_dim=4, hidden_dim=32, output_dim=2, hard_core_func=hc, lambda_res=1.0)
        opt = torch.optim.Adam(rcln.parameters(), lr=1e-3)
        for _ in range(50):
            opt.zero_grad()
            pred = rcln(X)
            loss = torch.nn.functional.mse_loss(pred, Y)
            loss.backward()
            opt.step()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024
        _record("memory_peak", "mb", peak_mb, "MB", {})
        return peak_mb
    except Exception as e:
        _record("memory_peak", "mb", -1, "MB", {"error": str(e)})
        return -1


def main():
    from .configs import CONFIG_MAP, STANDARD
    from .report import (
        check_thresholds,
        save_dated_json,
        load_historical_jsons,
        generate_trend_chart,
        generate_comparison_chart,
        generate_markdown_report,
        generate_html_report,
    )

    parser = argparse.ArgumentParser(description="Axiom-OS 性能基准测试")
    parser.add_argument("--config", choices=["quick", "standard", "full"], default="standard",
                        help="配置: quick(快) | standard(标准) | full(完整)")
    parser.add_argument("--unit", action="store_true", help="仅单元基准")
    parser.add_argument("--integration", action="store_true", help="仅集成基准")
    parser.add_argument("--e2e", action="store_true", help="仅端到端基准")
    parser.add_argument("--memory", action="store_true", help="内存基准")
    parser.add_argument("--quick", action="store_true", help="E2E 快速模式(跳过最慢步骤)")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出 JSON 文件")
    parser.add_argument("--report", action="store_true", help="生成 Markdown/HTML 报告")
    parser.add_argument("--trend", action="store_true", help="生成趋势图")
    parser.add_argument("--robustness", action="store_true", help="Discovery 鲁棒性测试（噪声变化）")
    parser.add_argument("--compare-pysr", action="store_true", help="与 PySR/SINDy 对比（需安装）")
    parser.add_argument("--hard", action="store_true", help="高难度 Discovery 套件（稀疏/外推/小样本/Feynman/Lorenz）")
    args = parser.parse_args()

    cfg = CONFIG_MAP[args.config]
    use_config = not (args.unit or args.integration or args.e2e or args.memory)
    run_robustness = args.robustness or (use_config and cfg.name == "standard")
    run_compare_pysr = args.compare_pysr
    run_hard = args.hard

    print("=" * 60)
    print(f"Axiom-OS 性能基准测试 [config={cfg.name}]")
    print("=" * 60)

    if use_config or args.unit:
        run_unit_benchmarks(robustness=run_robustness, compare_pysr=run_compare_pysr)
    if run_hard:
        run_hard_benchmarks()
    if use_config or args.integration:
        run_integration_benchmarks(cfg)
    run_e2e_quick = (use_config and cfg.e2e_quick) or args.quick
    run_e2e_full = (use_config and cfg.e2e) or args.e2e
    if run_e2e_quick or run_e2e_full:
        if run_e2e_quick:
            print("\n[E2E Quick] main + rar + battery...")
            from axiom_os.validate_all import run_main, run_rar, run_battery
            total = 0
            for name, fn in [
                ("main", lambda: run_main(quick=True)),
                ("rar", lambda: run_rar(cfg.rar_galaxies, cfg.rar_epochs // 2)),
                ("battery", lambda: run_battery(quick=True)),
            ]:
                t0 = time.perf_counter()
                r = fn()
                elapsed = time.perf_counter() - t0
                total += elapsed
                _record(f"e2e_quick_{name}", "elapsed_s", elapsed, "s", {"ok": r.get("ok", False)})
            _record("e2e_quick_total", "elapsed_s", total, "s", {})
        elif run_e2e_full:
            run_e2e_benchmarks()
    if use_config or args.memory:
        if cfg.memory:
            print("\n[Memory] 峰值内存...")
            bench_memory()

    # Phase 4.2: 阈值告警
    alerts = check_thresholds(RESULTS)
    if alerts:
        print("\n" + "!" * 60)
        print("⚠️ 阈值告警:")
        for a in alerts:
            print(f"  - {a['message']}")
        print("!" * 60)

    print("\n" + "-" * 60)
    print("结果汇总:")
    for r in RESULTS:
        v = r["value"]
        u = r.get("unit", "")
        extra = r.get("extra", {})
        ex_str = " " + str(extra) if extra else ""
        print(f"  {r['name']} [{r['metric']}]: {v:.4f} {u}{ex_str}")

    # 输出
    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, indent=2, ensure_ascii=False)
        print(f"\n已保存 JSON: {out_path}")

    # Phase 4.1: 按日期保存 + 趋势图
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    dated_path = save_dated_json(RESULTS, BENCH_DIR)
    print(f"已保存 dated: {dated_path}")

    if args.trend or args.report:
        historical = load_historical_jsons(BENCH_DIR)
        if args.trend:
            trend_path = BENCH_DIR / "benchmark_trend.png"
            generate_trend_chart(RESULTS, historical, trend_path)
            print(f"已保存趋势图: {trend_path}")
        if args.report:
            cmp_path = BENCH_DIR / "benchmark_comparison.png"
            if generate_comparison_chart(RESULTS, cmp_path):
                print(f"已保存对比图: {cmp_path}")

    # Phase 4.4: 报告生成
    if args.report:
        md = generate_markdown_report(RESULTS, alerts, cfg.name)
        md_path = BENCH_DIR / "benchmark_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"已保存 Markdown: {md_path}")

        cmp_path = BENCH_DIR / "benchmark_comparison.png"
        html = generate_html_report(RESULTS, alerts, cfg.name, comparison_chart_path=cmp_path)
        html_path = BENCH_DIR / "benchmark_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"已保存 HTML: {html_path}")

    print("=" * 60)
    if alerts:
        sys.exit(1)


if __name__ == "__main__":
    main()
