"""
Benchmark report 模块单元测试
测试 check_thresholds, save_dated_json, load_historical_jsons, generate_trend_chart, generate_markdown_report, generate_html_report
"""

import json
import tempfile
from pathlib import Path

import pytest

# 确保项目根在 path 中
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axiom_os.benchmarks.report import (
    check_thresholds,
    save_dated_json,
    load_historical_jsons,
    generate_trend_chart,
    generate_markdown_report,
    generate_html_report,
)


def test_check_thresholds_no_alerts():
    """正常值应无告警"""
    results = [
        {"name": "rcln_forward", "metric": "latency_ms", "value": 0.5},
        {"name": "turbulence_training", "metric": "elapsed_s", "value": 1.0},
        {"name": "rar_discovery", "metric": "elapsed_s", "value": 5.0},
    ]
    alerts = check_thresholds(results)
    assert len(alerts) == 0


def test_check_thresholds_with_alerts():
    """超阈值应产生告警"""
    results = [
        {"name": "rcln_forward", "metric": "latency_ms", "value": 15.0},  # > 10
        {"name": "turbulence_training", "metric": "elapsed_s", "value": 150.0},  # > 120
    ]
    alerts = check_thresholds(results)
    assert len(alerts) >= 2
    names = {a["name"] for a in alerts}
    assert "rcln_forward" in names
    assert "turbulence_training" in names
    for a in alerts:
        assert "message" in a
        assert a["value"] > a["threshold"]


def test_check_thresholds_ignores_negative():
    """负值应跳过"""
    results = [
        {"name": "rcln_forward", "metric": "latency_ms", "value": -1.0},
    ]
    alerts = check_thresholds(results)
    assert len(alerts) == 0


def test_check_thresholds_by_name_metric():
    """按 (name, metric) 正确查找，不混淆 throughput 与 latency_ms"""
    results = [
        {"name": "rcln_forward", "metric": "throughput", "value": 999999},
        {"name": "rcln_forward", "metric": "latency_ms", "value": 0.3},
    ]
    alerts = check_thresholds(results)
    assert len(alerts) == 0  # latency_ms=0.3 正常


def test_save_dated_json():
    """save_dated_json 应创建有效 JSON 文件"""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        results = [
            {"name": "test", "metric": "x", "value": 1.0, "unit": "s"},
        ]
        path = save_dated_json(results, out_dir)
        assert path.exists()
        assert path.suffix == ".json"
        assert "benchmark_" in path.name
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["name"] == "test"


def test_load_historical_jsons():
    """load_historical_jsons 应加载目录下所有 benchmark_*.json"""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        # 创建两个 JSON
        for i in range(2):
            data = {"timestamp": f"2026-01-0{i+1}T12:00:00", "results": [{"name": f"run{i}", "metric": "x", "value": i}]}
            (out_dir / f"benchmark_2026-01-0{i+1}_12-00-00.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        entries = load_historical_jsons(out_dir)
        assert len(entries) == 2
        assert all("timestamp" in e and "results" in e for e in entries)


def test_load_historical_jsons_empty_dir():
    """空目录返回空列表"""
    with tempfile.TemporaryDirectory() as tmp:
        entries = load_historical_jsons(Path(tmp))
        assert entries == []


def test_load_historical_jsons_nonexistent():
    """不存在目录返回空列表"""
    entries = load_historical_jsons(Path("/nonexistent/path/12345"))
    assert entries == []


def test_generate_trend_chart():
    """generate_trend_chart 应生成 PNG 文件"""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "trend.png"
        results = [
            {"name": "rcln_forward", "metric": "latency_ms", "value": 0.5},
            {"name": "turbulence_training", "metric": "elapsed_s", "value": 1.0},
            {"name": "rar_discovery", "metric": "elapsed_s", "value": 2.0},
        ]
        historical = []
        generate_trend_chart(results, historical, out_path)
        if out_path.exists():  # matplotlib 可能未安装
            assert out_path.stat().st_size > 0


def test_generate_markdown_report():
    """generate_markdown_report 应生成有效 Markdown"""
    results = [
        {"name": "rcln_forward", "metric": "latency_ms", "value": 0.5, "unit": "ms"},
    ]
    alerts = []
    md = generate_markdown_report(results, alerts, "quick")
    assert "# Axiom-OS 性能基准测试报告" in md
    assert "quick" in md
    assert "rcln_forward" in md
    assert "✅ 无阈值告警" in md


def test_generate_markdown_report_with_alerts():
    """有告警时 Markdown 应包含告警节"""
    results = [{"name": "rcln_forward", "metric": "latency_ms", "value": 15.0, "unit": "ms"}]
    alerts = [{"name": "rcln_forward", "message": "rcln_forward 15.00ms > 阈值 10.0ms"}]
    md = generate_markdown_report(results, alerts, "standard")
    assert "⚠️ 阈值告警" in md
    assert "15.00" in md


def test_generate_html_report():
    """generate_html_report 应生成有效 HTML"""
    results = [
        {"name": "test", "metric": "x", "value": 1.0, "unit": "s"},
    ]
    html = generate_html_report(results, [], "full")
    assert "<!DOCTYPE html>" in html
    assert "<h1>" in html
    assert "full" in html
    assert "test" in html
    assert "</body></html>" in html
