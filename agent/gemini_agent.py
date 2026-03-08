"""
Axiom Gemini Agent：用 Gemini 大模型收集多领域数据、分析报告、生成扩展代码并建议优化。
通过工具调用（run_benchmark, get_report, run_rar, apply_domain_extension 等）与 Axiom 交互。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .tools import TOOL_DEFS, run_tool

SYSTEM_PROMPT = """你是 Axiom-OS 的扩展与优化助手。你可以通过工具完成以下能力：

1. **收集与评估**：运行基准 (run_benchmark_quick)、读取报告 (get_benchmark_report)、运行 RAR/公式发现 (run_rar, run_discovery_demo)、查看当前领域 (list_domains)。
2. **扩展 Axiom**：根据需求生成新领域或新协议的 Python 代码，通过 apply_domain_extension 提交保存。
3. **优化建议**：根据基准报告与 R² 等指标，给出超参数建议（如 rar_epochs、n_galaxies）、数据增强或协议顺序建议。

规则：
- 若用户要求「跑基准」「看报告」「分析一下」：先调用 run_benchmark_quick 或 get_benchmark_report，再根据结果用自然语言总结或建议。
- 若用户要求「扩展」「新领域」「加一个 XXX 协议」：可先 list_domains 了解现有领域，再生成符合 Axiom 接口的代码，用 apply_domain_extension 提交。
- 每次只能调用一个工具。调用格式必须严格为一行：
  CALL: 工具名 参数1=值1 参数2=值2
  例如：CALL: run_rar n_galaxies=30 epochs=400
  例如：CALL: get_benchmark_report
  无参数时：CALL: 工具名
- 收到工具返回后，用中文总结结果并给出下一步建议或直接回答用户。若无需再调用工具，直接给出最终回复。
"""

# 工具名 -> 参数列表
TOOL_PARAMS = {
    "run_benchmark_quick": [],
    "get_benchmark_report": [],
    "run_rar": ["n_galaxies", "epochs"],
    "run_discovery_demo": [],
    "list_domains": [],
    "apply_domain_extension": ["domain_name", "code_snippet"],
}


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中解析 CALL: tool_name k=v ..."""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("CALL:"):
            rest = line[5:].strip()
            parts = rest.split()
            if not parts:
                return None
            name = parts[0].strip()
            params = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    try:
                        v = int(v)
                    except ValueError:
                        pass
                    params[k.strip()] = v
            return {"name": name, "params": params}
    return None


def _invoke_gemini(api_key: str, model: str, prompt: str) -> str:
    """调用 Gemini API，单次生成。"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel(model)
        resp = gemini.generate_content(prompt)
        if resp.text:
            return resp.text.strip()
        return ""
    except ImportError:
        raise ImportError("请安装: pip install google-generativeai")
    except Exception as e:
        return f"[Gemini 调用异常: {e}]"


def run_agent_loop(
    user_message: str,
    api_key: str,
    model: str = "gemini-1.5-flash",
    max_tool_rounds: int = 5,
) -> str:
    """
    与 Gemini 多轮对话：用户输入 -> Gemini 可能返回工具调用 -> 执行工具 -> 结果回传 -> 直至最终回复。
    返回最后一轮模型生成的自然语言回复。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    tool_results: List[str] = []
    final_reply = ""

    for _ in range(max_tool_rounds):
        # 构造当前发给模型的消息：系统 + 历史 + 若有工具结果则追加
        prompt_parts = [SYSTEM_PROMPT, "\n\n用户: " + user_message]
        for tr in tool_results:
            prompt_parts.append("\n\n[工具返回]\n" + tr)
        if tool_results:
            prompt_parts.append("\n\n请根据上述工具返回给出总结或下一步（若需再调用工具，请用 CALL: 格式）。")
        prompt = "\n".join(prompt_parts)

        reply = _invoke_gemini(api_key, model, prompt)
        if not reply or reply.startswith("["):
            final_reply = reply or "无回复"
            break

        call = _parse_tool_call(reply)
        if not call:
            final_reply = reply
            break

        name = call["name"]
        params = call.get("params") or {}
        if name not in [t["name"] for t in TOOL_DEFS]:
            tool_results.append(json.dumps({"ok": False, "error": f"未知工具: {name}"}, ensure_ascii=False))
            continue
        result = run_tool(name, **params)
        tool_results.append(json.dumps(result, ensure_ascii=False, indent=0))

    if tool_results and not final_reply:
        # 最后一轮把工具结果喂回去让模型总结
        prompt = "\n\n".join([SYSTEM_PROMPT, "用户: " + user_message] + ["[工具返回]\n" + tr for tr in tool_results])
        prompt += "\n\n请根据以上工具返回，用中文给出最终总结与建议。"
        final_reply = _invoke_gemini(api_key, model, prompt)

    return (final_reply or "未得到有效回复").strip()


def run_agent_single(
    user_message: str,
    api_key: str,
    model: str = "gemini-1.5-flash",
) -> str:
    """单轮：仅让 Gemini 根据当前上下文回复（不执行工具）。用于简单问答。"""
    prompt = SYSTEM_PROMPT + "\n\n用户: " + user_message
    return _invoke_gemini(api_key, model, prompt)
