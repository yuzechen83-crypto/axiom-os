"""
Axiom-OS 性能基准测试入口

用法:
  python -m axiom_os.benchmarks.run_benchmarks                    # 默认 standard
  python -m axiom_os.benchmarks.run_benchmarks --config quick      # 快速
  python -m axiom_os.benchmarks.run_benchmarks --config full       # 完整
  python -m axiom_os.benchmarks.run_benchmarks --unit               # 仅单元
  python -m axiom_os.benchmarks.run_benchmarks -o report.json       # 输出 JSON
  python -m axiom_os.benchmarks.run_benchmarks --report            # 生成 MD/HTML 报告
  python -m axiom_os.benchmarks.run_benchmarks --trend              # 生成趋势图
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


def run_unit_benchmarks():
    """单元级基准"""
    print("\n[Unit] RCLN forward...")
    bench_rcln_forward()
    print("[Unit] Discovery...")
    bench_discovery()
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
    args = parser.parse_args()

    cfg = CONFIG_MAP[args.config]
    use_config = not (args.unit or args.integration or args.e2e or args.memory)

    print("=" * 60)
    print(f"Axiom-OS 性能基准测试 [config={cfg.name}]")
    print("=" * 60)

    if use_config or args.unit:
        run_unit_benchmarks()
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

    # Phase 4.4: 报告生成
    if args.report:
        md = generate_markdown_report(RESULTS, alerts, cfg.name)
        md_path = BENCH_DIR / "benchmark_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"已保存 Markdown: {md_path}")

        html = generate_html_report(RESULTS, alerts, cfg.name)
        html_path = BENCH_DIR / "benchmark_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"已保存 HTML: {html_path}")

    print("=" * 60)
    if alerts:
        sys.exit(1)


if __name__ == "__main__":
    main()
