"""
Axiom Agent 工具：供大模型调用的可执行能力。
包括：跑基准、读报告、跑 RAR/Discovery、列出领域、建议优化、提交扩展代码等。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_benchmark_quick() -> Dict[str, Any]:
    """运行快速基准，返回摘要（R²、耗时等）。"""
    try:
        from axiom_os.benchmarks.run_benchmarks import main as bench_main
        old = sys.argv
        sys.argv = ["run_benchmarks", "--config", "quick", "--report", "--no-fail-on-alerts"]
        try:
            bench_main()
        finally:
            sys.argv = old
        report_path = ROOT / "axiom_os" / "benchmarks" / "results" / "benchmark_report.md"
        summary = report_path.read_text(encoding="utf-8")[:2000] if report_path.exists() else "基准已跑完，见 results/"
        return {"ok": True, "summary": summary, "report_path": str(report_path)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_benchmark_report() -> Dict[str, Any]:
    """读取最新基准报告内容（用于大模型分析）。"""
    report_path = ROOT / "axiom_os" / "benchmarks" / "results" / "benchmark_report.md"
    if not report_path.exists():
        return {"ok": False, "error": "报告不存在，请先运行基准"}
    text = report_path.read_text(encoding="utf-8")
    return {"ok": True, "content": text[:8000], "path": str(report_path)}


def run_rar(n_galaxies: int = 20, epochs: int = 200) -> Dict[str, Any]:
    """运行 RAR 发现，返回 R² 与样本数。"""
    try:
        from axiom_os.experiments.discovery_rar import run_rar_discovery
        res = run_rar_discovery(n_galaxies=n_galaxies, epochs=epochs)
        if res.get("error"):
            return {"ok": False, "error": res["error"]}
        return {
            "ok": True,
            "r2_log": res.get("r2_log"),
            "r2_linear": res.get("r2"),
            "n_samples": res.get("n_samples"),
            "n_galaxies": res.get("n_galaxies"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_discovery_demo() -> Dict[str, Any]:
    """运行公式发现 Demo。"""
    try:
        from axiom_os.demos.discovery_demo import main
        main()
        return {"ok": True, "message": "公式发现 Demo 已执行"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_domains() -> Dict[str, Any]:
    """列出 Axiom 当前支持的领域与协议。"""
    try:
        from axiom_os.mll.domain_protocols import PROTOCOL_REGISTRY
        domains = list(PROTOCOL_REGISTRY.keys())
        return {"ok": True, "domains": domains, "description": "RAR=星系旋转, battery=电池RUL, turbulence=湍流"}
    except ImportError:
        return {"ok": True, "domains": ["rar", "battery", "turbulence"], "description": "MLL 未加载，默认领域列表"}


def suggest_optimization(context: str) -> Dict[str, Any]:
    """
    根据当前报告/指标上下文，让大模型生成优化建议（不执行）。
    调用方传入 context（如报告摘要），大模型返回建议文本。
    此处仅返回占位，实际建议由 Gemini 在 agent 中生成。
    """
    return {"ok": True, "context_received": context[:500], "hint": "由 LLM 根据 get_benchmark_report / run_rar 结果生成建议"}


def apply_domain_extension(domain_name: str, code_snippet: str) -> Dict[str, Any]:
    """
    提交新领域或扩展的代码片段（由大模型生成）。
    当前为占位：仅保存到 agent_output，不自动注册到 MLL。
    后续可解析并注册到 PROTOCOL_REGISTRY 或执行安全沙箱验证。
    """
    out_dir = ROOT / "agent_output" / "extensions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{domain_name}.py"
    path.write_text(code_snippet, encoding="utf-8")
    return {"ok": True, "saved_path": str(path), "message": "扩展已保存，需人工审核后集成"}


# 供 Gemini 等调用的工具声明（名称与参数描述）
TOOL_DEFS = [
    {
        "name": "run_benchmark_quick",
        "description": "运行 Axiom 快速基准测试（约 30 秒），得到 R²、耗时等指标",
        "params": {},
    },
    {
        "name": "get_benchmark_report",
        "description": "读取最新基准报告全文，用于分析当前性能",
        "params": {},
    },
    {
        "name": "run_rar",
        "description": "运行 RAR 星系旋转曲线发现，返回对数 R² 与样本数",
        "params": {"n_galaxies": "int, 星系数，默认 20", "epochs": "int, 训练轮数，默认 200"},
    },
    {
        "name": "run_discovery_demo",
        "description": "运行公式发现 Demo（从带噪数据恢复符号公式）",
        "params": {},
    },
    {
        "name": "list_domains",
        "description": "列出 Axiom 当前支持的领域（如 rar, battery, turbulence）",
        "params": {},
    },
    {
        "name": "apply_domain_extension",
        "description": "提交大模型生成的新领域/扩展代码，保存到 agent_output/extensions",
        "params": {"domain_name": "str", "code_snippet": "str, Python 代码"},
    },
]


def run_tool(name: str, **kwargs: Any) -> Dict[str, Any]:
    """根据工具名执行并返回结果。"""
    if name == "run_benchmark_quick":
        return run_benchmark_quick()
    if name == "get_benchmark_report":
        return get_benchmark_report()
    if name == "run_rar":
        return run_rar(n_galaxies=kwargs.get("n_galaxies", 20), epochs=kwargs.get("epochs", 200))
    if name == "run_discovery_demo":
        return run_discovery_demo()
    if name == "list_domains":
        return list_domains()
    if name == "apply_domain_extension":
        return apply_domain_extension(
            domain_name=kwargs.get("domain_name", "custom"),
            code_snippet=kwargs.get("code_snippet", ""),
        )
    return {"ok": False, "error": f"未知工具: {name}"}
