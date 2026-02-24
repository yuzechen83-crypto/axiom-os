"""
Phase 4.1 趋势图 + Phase 4.2 阈值告警 + Phase 4.4 报告生成
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# 阈值配置 (Phase 4.2)
THRESHOLDS = {
    "rcln_forward": {"metric": "latency_ms", "max": 10.0, "unit": "ms"},
    "e2e_quick_total": {"metric": "elapsed_s", "max": 120.0, "unit": "s"},
    "e2e_total": {"metric": "elapsed_s", "max": 900.0, "unit": "s"},  # 15 min
    "turbulence_training": {"metric": "elapsed_s", "max": 120.0, "unit": "s"},
    "rar_discovery": {"metric": "elapsed_s", "max": 90.0, "unit": "s"},
}


def check_thresholds(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Phase 4.2: 检查阈值，返回告警列表"""
    alerts = []
    by_key = {}
    for r in results:
        by_key[(r["name"], r["metric"])] = r

    for name, cfg in THRESHOLDS.items():
        key = (name, cfg["metric"])
        if key not in by_key:
            continue
        r = by_key[key]
        v = r.get("value", 0)
        if v < 0:
            continue
        if v > cfg["max"]:
            alerts.append({
                "name": name,
                "value": v,
                "threshold": cfg["max"],
                "unit": cfg["unit"],
                "message": f"{name} {v:.2f}{cfg['unit']} > 阈值 {cfg['max']}{cfg['unit']}",
            })
    return alerts


def save_dated_json(results: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Phase 4.1: 按日期保存 JSON 用于趋势分析"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = output_dir / f"benchmark_{ts}.json"
    data = {"timestamp": datetime.now().isoformat(), "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_historical_jsons(bench_dir: Path) -> List[Dict[str, Any]]:
    """加载历史 benchmark JSON"""
    if not bench_dir.exists():
        return []
    entries = []
    for p in sorted(bench_dir.glob("benchmark_*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("timestamp", p.stem)
            entries.append({"path": str(p), "timestamp": ts, "results": data.get("results", [])})
        except Exception:
            continue
    return entries


def _get_value(results: List[Dict], name: str, metric: str) -> Optional[float]:
    for r in results:
        if r.get("name") == name and r.get("metric") == metric:
            return r.get("value")
    return None


def generate_trend_chart(results: List[Dict], historical: List[Dict], out_path: Path) -> None:
    """Phase 4.1: 生成趋势图 (matplotlib)"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return

    # 合并当前结果到历史（用于首次运行或确保最新点）
    all_entries = list(historical)
    all_entries.append({"timestamp": datetime.now().isoformat(), "results": results})

    metrics_to_plot = [
        ("rcln_forward", "latency_ms", "RCLN latency (ms)"),
        ("turbulence_training", "elapsed_s", "Turbulence (s)"),
        ("rar_discovery", "elapsed_s", "RAR (s)"),
    ]
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 4 * len(metrics_to_plot)))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, (name, metric, title) in zip(axes, metrics_to_plot):
        dates, values = [], []
        for h in all_entries:
            v = _get_value(h["results"], name, metric)
            if v is not None:
                try:
                    ts = h["timestamp"][:19] if isinstance(h["timestamp"], str) else h["timestamp"]
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
                    dates.append(dt)
                    values.append(v)
                except Exception:
                    continue
        if dates and values:
            ax.plot(dates, values, "o-", markersize=6)
            ax.set_title(title)
            ax.set_ylabel(metric)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def generate_markdown_report(
    results: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    config_name: str = "standard",
) -> str:
    """Phase 4.4: 生成 Markdown 报告"""
    lines = [
        "# Axiom-OS 性能基准测试报告",
        "",
        f"**配置**: {config_name}",
        f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 结果汇总",
        "",
        "| 指标 | 值 | 单位 | 备注 |",
        "|------|-----|------|------|",
    ]
    for r in results:
        v = r.get("value", 0)
        u = r.get("unit", "")
        extra = r.get("extra", {})
        ex = ", ".join(f"{k}={v}" for k, v in list(extra.items())[:3]) if extra else ""
        lines.append(f"| {r['name']} [{r['metric']}] | {v:.4f} | {u} | {ex} |")

    if alerts:
        lines.extend([
            "",
            "## ⚠️ 阈值告警",
            "",
        ])
        for a in alerts:
            lines.append(f"- **{a['name']}**: {a['message']}")
    else:
        lines.extend(["", "## ✅ 无阈值告警", ""])

    return "\n".join(lines)


def generate_html_report(
    results: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    config_name: str = "standard",
) -> str:
    """Phase 4.4: 生成 HTML 报告"""
    html = """<!DOCTYPE html><html><head><meta charset='utf-8'><title>Axiom Benchmark</title>
<style>body{font-family:sans-serif;margin:2em;max-width:900px}
table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:8px 12px}
th{background:#f0f0f0}.alert{color:#c00}.ok{color:#080}</style></head><body>"""
    html += f"<h1>Axiom-OS 性能基准测试报告</h1>"
    html += f"<p><b>配置</b>: {config_name} | <b>时间</b>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
    html += "<h2>结果汇总</h2><table><tr><th>指标</th><th>值</th><th>单位</th><th>备注</th></tr>"
    for r in results:
        v = r.get("value", 0)
        u = r.get("unit", "")
        extra = r.get("extra", {})
        ex = ", ".join(f"{k}={v}" for k, v in list(extra.items())[:3]) if extra else ""
        html += f"<tr><td>{r['name']} [{r['metric']}]</td><td>{v:.4f}</td><td>{u}</td><td>{ex}</td></tr>"
    html += "</table>"
    if alerts:
        html += "<h2 class='alert'>⚠️ 阈值告警</h2><ul>"
        for a in alerts:
            html += f"<li>{a['message']}</li>"
        html += "</ul>"
    else:
        html += "<h2 class='ok'>✅ 无阈值告警</h2>"
    html += "</body></html>"
    return html
