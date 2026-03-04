#!/usr/bin/env python3
"""
Axiom 自主学习脚本：让 Agent 自主搜索、学习 AI 与物理领域内容，并进化改进。

用法: python -m axiom_os.scripts.self_learn
      或: axiom self-learn
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 加载 LLM 配置
_env_path = ROOT / "axiom_os" / "config" / "axiom_os_llm.env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

LEARN_PROMPT = """你正在执行自主学习任务。请按以下步骤进行，尽量多轮调用工具：

1. **搜索 AI 领域**：用 web_search 搜索「physics-informed neural networks」「PINN」「scientific machine learning」「AI for physics」等，获取最新论文、综述、开源项目链接。

2. **搜索物理领域**：用 web_search 搜索「symbolic regression」「formula discovery」「physics discovery」「neural ODE」等，发现可借鉴的方法。

3. **抓取学习**：对搜索到的有价值 URL，用 fetch_url 抓取正文。若内容对 Axiom 有价值，用 save_to_workspace 保存到 ENGINE_DESIGN.md 或新建 LEARNING_AI_PHYSICS.md。

4. **检索已有知识**：用 retrieve_hippocampus 检索与「物理定律」「公式发现」相关的内容；用 read_workspace_doc 读取 IDENTITY.md、PROJECTS.md 了解当前能力。

5. **提出改进**：基于检索与学习结果，用自然语言总结学到的要点，并提出对 Axiom 系统的 3–5 条具体改进建议（如新协议、新工具、算法优化等）。可将建议写入 workspace 的 IMPROVEMENTS.md。

请开始执行，每步调用相应工具，不要跳过。"""


def main() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 请配置 DEEPSEEK_API_KEY（axiom_os/config/axiom_os_llm.env）")
        return 1

    # 确保 workspace 存在
    from axiom_os.config.loader import load_config, ensure_workspace
    load_config()
    ensure_workspace()
    print("Axiom 自主学习模式启动...")
    print("=" * 60)

    from axiom_os.agent.deepseek_agent import run_agent_loop

    reply = run_agent_loop(
        user_message=LEARN_PROMPT,
        api_key=api_key,
        max_tool_rounds=25,  # 允许更多轮工具调用以深入探索
    )

    print("\n" + "=" * 60)
    print("自主学习完成。")
    print(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
