"""
Axiom-OS CLI - OpenClaw 风格统一入口

用法:
  axiom agent --message "..."
  axiom api
  axiom ui
  axiom init
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_env():
    """加载 LLM 配置"""
    env_path = ROOT / "axiom_os" / "config" / "axiom_os_llm.env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass


def cmd_agent(args):
    """运行 agent 单轮：axiom agent --message "..." """
    _load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 请配置 DEEPSEEK_API_KEY（axiom_os/config/axiom_os_llm.env）")
        return 1
    from axiom_os.agent.deepseek_agent import run_agent_loop
    reply = run_agent_loop(
        user_message=args.message,
        api_key=api_key,
        max_tool_rounds=args.max_rounds or 5,
    )
    print(reply)
    return 0


def cmd_api(args):
    """启动 REST API"""
    port = args.port or int(os.environ.get("AXIOM_API_PORT", "8000"))
    try:
        import uvicorn
        from axiom_os.api.server import app
        uvicorn.run(app, host=args.host or "0.0.0.0", port=port)
    except ImportError:
        print("❌ 请安装: pip install axiom-os[full]")
        return 1
    return 0


def cmd_ui(args):
    """启动 Streamlit UI"""
    try:
        import subprocess
        ui_path = ROOT / "axiom_os" / "agent" / "chat_ui.py"
        return subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(ui_path), "--server.headless", "true"],
            cwd=str(ROOT),
            env=os.environ.copy(),
        ).returncode
    except Exception as e:
        print(f"❌ {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        prog="axiom",
        description="Axiom-OS: Physics-Aware AI. OpenClaw 风格。",
    )
    sub = parser.add_subparsers(dest="cmd", help="子命令")

    # axiom agent --message "..."
    p_agent = sub.add_parser("agent", help="运行 agent 单轮（DeepSeek + 工具）")
    p_agent.add_argument("--message", "-m", required=True, help="用户消息")
    p_agent.add_argument("--session-id", help="会话 ID（预留）")
    p_agent.add_argument("--max-rounds", type=int, default=5, help="最大工具轮数")
    p_agent.set_defaults(func=cmd_agent)

    # axiom api
    p_api = sub.add_parser("api", help="启动 REST API 服务")
    p_api.add_argument("--port", "-p", type=int, help="端口")
    p_api.add_argument("--host", help="主机")
    p_api.set_defaults(func=cmd_api)

    # axiom ui
    p_ui = sub.add_parser("ui", help="启动 Streamlit 交互界面")
    p_ui.set_defaults(func=cmd_ui)

    # axiom arena（多 AI 讨论）
    p_arena = sub.add_parser("arena", help="多 AI 讨论与竞争（支持任务驱动、Kimi CLI 自编程）")
    p_arena.add_argument("-m", "--message", help="讨论主题（纯讨论模式）")
    p_arena.add_argument("--task", help="任务驱动: high_freq | lorenz_pinn_lstm | real_data_long_term")
    p_arena.add_argument("--rounds", type=int, default=2, help="每 agent 发言轮数")
    p_arena.add_argument("--no-kimi", action="store_true", help="不使用 Kimi API，仅 DeepSeek 双 agent")
    p_arena.add_argument("--self-program", action="store_true", help="讨论后调用 Kimi CLI 根据建议修改代码")
    p_arena.set_defaults(func=lambda a: _cmd_arena(a))

    # axiom init（初始化 workspace）
    p_init = sub.add_parser("init", help="初始化 ~/.axiom_os 配置与 workspace")
    p_init.set_defaults(func=lambda a: _init_workspace())

    # axiom self-learn（自主学习：搜索 AI/物理，进化改进）
    p_learn = sub.add_parser("self-learn", help="自主搜索学习 AI 与物理领域，进化改进")
    p_learn.set_defaults(func=lambda a: _cmd_self_learn())

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return 0
    return args.func(args)


def _cmd_arena(args):
    """多 AI 讨论（纯讨论、任务驱动、Kimi CLI 自编程）"""
    _load_env()
    from axiom_os.arena.multi_agent import main_cli
    return main_cli(
        message=getattr(args, "message", None),
        rounds=args.rounds,
        task=getattr(args, "task", None),
        use_kimi=not getattr(args, "no_kimi", False),
        self_program=getattr(args, "self_program", False),
    )


def _init_workspace():
    from axiom_os.config.loader import load_config, ensure_workspace
    load_config()
    ensure_workspace()
    print("[OK] Initialized ~/.axiom_os")
    return 0


def _cmd_self_learn():
    """运行自主学习：Agent 搜索 AI/物理，学习并进化"""
    _load_env()
    import subprocess
    return subprocess.run(
        [sys.executable, "-m", "axiom_os.scripts.self_learn"],
        cwd=str(ROOT),
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
