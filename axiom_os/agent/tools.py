"""
Axiom Agent 工具：供大模型调用的可执行能力。
包括：跑基准、读报告、跑 RAR/Discovery、列出领域、建议优化、提交扩展代码等。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    此处仅返回占位，实际建议由 LLM 在 agent 中生成。
    """
    return {"ok": True, "context_received": context[:500], "hint": "由 LLM 根据 get_benchmark_report / run_rar 结果生成建议"}


def run_grid_mpc(disturbance_MW: float = 800.0, nadir_safe_Hz: float = 49.0) -> Dict[str, Any]:
    """运行 Grid Pulse MPC：预测-决策-执行闭环"""
    try:
        from axiom_os.experiments.run_grid_mpc import main as grid_main
        result = grid_main(disturbance_MW=disturbance_MW, nadir_safe_Hz=nadir_safe_Hz)
        return {
            "ok": True,
            "nadir_no_ctrl": result.get("nadir_no_ctrl"),
            "nadir_after": result.get("nadir_after"),
            "load_shed_MW": result.get("load_shed"),
            "storage_MW": result.get("storage"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_cad_model(
    shape: str = "l_bracket",
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """运行 CAD 建模：构建 3D 形状并导出 STL。shape: box|cylinder|sphere|l_bracket|simple_gear"""
    try:
        from axiom_os.tools.cad_model import run_cad_model as _run
        return _run(shape=shape, output_path=output_path, **kwargs)
    except ImportError:
        return {"ok": False, "error": "axiom_os.tools.cad_model 未找到"}


def list_cad_shapes() -> Dict[str, Any]:
    """列出支持的 CAD 形状"""
    try:
        from axiom_os.tools.cad_model import list_cad_shapes as _list
        return _list()
    except ImportError:
        return {"ok": False, "error": "axiom_os.tools.cad_model 未找到"}


def read_workspace_doc(doc_name: str) -> Dict[str, Any]:
    """
    读取 workspace 文档内容，用于 Agent 学习领域知识。
    doc_name: IDENTITY.md | SOUL.md | PROJECTS.md | ENGINE_DESIGN.md | CHIP_DEV.md 等
    """
    try:
        from axiom_os.config.loader import get_workspace_path, list_workspace_docs
        ws = get_workspace_path()
        if not ws.exists():
            return {"ok": False, "error": "Workspace 不存在，请运行 axiom init"}
        path = ws / doc_name
        if not path.exists():
            available = list_workspace_docs()
            return {"ok": False, "error": f"文档不存在: {doc_name}", "available": available}
        text = path.read_text(encoding="utf-8")
        return {"ok": True, "content": text[:12000], "path": str(path)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_workspace_docs() -> Dict[str, Any]:
    """列出 workspace 下所有可读文档"""
    try:
        from axiom_os.config.loader import list_workspace_docs as _list
        docs = _list()
        return {"ok": True, "docs": docs}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def fetch_url(
    url: str,
    save_to_workspace: Optional[str] = None,
    max_chars: int = 30000,
) -> Dict[str, Any]:
    """抓取 URL 正文，供 Agent 学习。可选 save_to_workspace 保存到 workspace 文档"""
    try:
        from axiom_os.tools.web_fetch import fetch_url as _fetch
        return _fetch(url=url, save_to_workspace=save_to_workspace, max_chars=max_chars)
    except ImportError:
        return {"ok": False, "error": "axiom_os.tools.web_fetch 未找到"}


def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """网页搜索，返回摘要列表，用于发现学习资源"""
    try:
        from axiom_os.tools.web_fetch import web_search as _search
        return _search(query=query, max_results=max_results)
    except ImportError:
        return {"ok": False, "error": "axiom_os.tools.web_fetch 未找到"}


def retrieve_hippocampus(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    从 Hippocampus 知识库检索与问题相关的物理定律。
    用于 Agent 在回答前获取已结晶的公式。
    """
    try:
        from axiom_os.core.hippocampus import Hippocampus
        hippo = Hippocampus(dim=32, capacity=5000)
        result = hippo.retrieve_by_query(question, top_k=top_k)
        return {"ok": True, "content": result or "(无匹配定律)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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


# 供 DeepSeek 等 LLM 调用的工具声明（名称与参数描述）
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
        "name": "run_grid_mpc",
        "description": "运行 Grid Pulse MPC：电网频率扰动预测-决策-执行，缺发电扰动时 MPC 选切负荷/储能",
        "params": {"disturbance_MW": "float, 缺发电MW，默认800", "nadir_safe_Hz": "float, 安全线Hz，默认49"},
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
    {
        "name": "run_cad_model",
        "description": "CAD 建模：构建 3D 形状并导出 STL。shape: box/cylinder/sphere/l_bracket/simple_gear。box 可用 width/height/depth，cylinder 可用 radius/height",
        "params": {"shape": "str, 形状名", "width": "float, box宽度", "height": "float, box高度或cylinder高度", "depth": "float, box深度", "radius": "float, 圆柱/球半径", "output_path": "str, 可选"},
    },
    {
        "name": "list_cad_shapes",
        "description": "列出支持的 CAD 形状及参数",
        "params": {},
    },
    {
        "name": "read_workspace_doc",
        "description": "读取 workspace 文档（IDENTITY/SOUL/PROJECTS/ENGINE_DESIGN/CHIP_DEV 等），用于学习领域知识",
        "params": {"doc_name": "str, 文档名如 IDENTITY.md"},
    },
    {
        "name": "list_workspace_docs",
        "description": "列出 workspace 下所有可读文档",
        "params": {},
    },
    {
        "name": "retrieve_hippocampus",
        "description": "从 Hippocampus 知识库检索与问题相关的物理定律",
        "params": {"question": "str, 查询问题", "top_k": "int, 返回条数默认5"},
    },
    {
        "name": "fetch_url",
        "description": "从网上抓取 URL 正文，供自主学习。可选 save_to_workspace 保存到 workspace（如 ENGINE_DESIGN.md）",
        "params": {"url": "str, 网页URL", "save_to_workspace": "str, 可选，保存到的文档名", "max_chars": "int, 最大字符数默认30000"},
    },
    {
        "name": "web_search",
        "description": "网页搜索，返回摘要列表，用于发现学习资源。如「发动机设计」「芯片功耗」",
        "params": {"query": "str, 搜索关键词", "max_results": "int, 返回条数默认5"},
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
    if name == "run_grid_mpc":
        return run_grid_mpc(
            disturbance_MW=kwargs.get("disturbance_MW", 800),
            nadir_safe_Hz=kwargs.get("nadir_safe_Hz", 49),
        )
    if name == "list_domains":
        return list_domains()
    if name == "apply_domain_extension":
        return apply_domain_extension(
            domain_name=kwargs.get("domain_name", "custom"),
            code_snippet=kwargs.get("code_snippet", ""),
        )
    if name == "run_cad_model":
        return run_cad_model(
            shape=kwargs.get("shape", "l_bracket"),
            output_path=kwargs.get("output_path"),
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            depth=kwargs.get("depth"),
            radius=kwargs.get("radius"),
        )
    if name == "list_cad_shapes":
        return list_cad_shapes()
    if name == "read_workspace_doc":
        return read_workspace_doc(doc_name=kwargs.get("doc_name", "IDENTITY.md"))
    if name == "list_workspace_docs":
        return list_workspace_docs()
    if name == "retrieve_hippocampus":
        return retrieve_hippocampus(
            question=kwargs.get("question", ""),
            top_k=kwargs.get("top_k", 5),
        )
    if name == "fetch_url":
        return fetch_url(
            url=kwargs.get("url", ""),
            save_to_workspace=kwargs.get("save_to_workspace"),
            max_chars=kwargs.get("max_chars", 30000),
        )
    if name == "web_search":
        return web_search(
            query=kwargs.get("query", ""),
            max_results=kwargs.get("max_results", 5),
        )
    return {"ok": False, "error": f"未知工具: {name}"}
