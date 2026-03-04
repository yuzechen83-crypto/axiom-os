"""
Axiom-OS 配置加载（OpenClaw 风格）
配置路径：~/.axiom_os/axiom_os.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

AXIOM_HOME = Path(os.environ.get("AXIOM_HOME", Path.home() / ".axiom_os"))
CONFIG_PATH = AXIOM_HOME / "axiom_os.json"
WORKSPACE_PATH = AXIOM_HOME / "workspace"


def _default_config() -> Dict[str, Any]:
    return {
        "identity": {
            "name": "Axiom",
            "theme": "Physics-Aware AI Assistant",
            "emoji": "🔬",
        },
        "agent": {
            "workspace": str(WORKSPACE_PATH),
            "model": {"primary": "deepseek-chat"},
        },
        "gateway": {
            "port": 8000,
            "host": "0.0.0.0",
        },
        "tools": {
            "run_benchmark": True,
            "run_rar": True,
            "run_grid_mpc": True,
            "run_discovery": True,
        },
    }


def load_config() -> Dict[str, Any]:
    """加载配置，不存在则创建默认并返回"""
    AXIOM_HOME.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
                # 合并默认值
                default = _default_config()
                for k, v in default.items():
                    if k not in cfg:
                        cfg[k] = v
                return cfg
        except Exception:
            pass
    cfg = _default_config()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def ensure_workspace():
    """确保 workspace 目录及默认文件存在"""
    ws_str = load_config()["agent"]["workspace"]
    ws = Path(ws_str).expanduser()
    ws.mkdir(parents=True, exist_ok=True)
    for name, content in [
        ("IDENTITY.md", "# Axiom Agent\n\n物理 AI 助手，支持基准、RAR、Grid Pulse、CAD 建模等。\n"),
        ("SOUL.md", "# 行为准则\n\n- 优先执行用户请求的物理仿真\n- 工具调用后给出简洁回复\n- 支持 CAD 建模、公式发现、领域扩展\n"),
        ("PROJECTS.md", "# 项目列表\n\n- grid_pulse: 电网 MPC\n- rar: 星系旋转\n- battery: 电池 RUL\n- turbulence: 湍流\n- benchmark: 基准测试\n- cad: 3D 建模\n"),
        ("ENGINE_DESIGN.md", "# 发动机设计领域\n\n待扩展：接入 CFD 仿真数据或代理模型。\n- 热力学：效率 vs 温度/压力\n- 流体：流量、压降\n- 结构：应力、振动\n"),
        ("CHIP_DEV.md", "# 芯片开发领域\n\n待扩展：接入 EDA 仿真或时序/功耗数据。\n- 功耗：P vs 频率/电压\n- 时序：延迟 vs 负载\n- 面积：逻辑单元数\n"),
    ]:
        p = ws / name
        if not p.exists():
            p.write_text(content, encoding="utf-8")


def get_workspace_path() -> Path:
    """返回 workspace 目录路径"""
    ws_str = load_config()["agent"]["workspace"]
    return Path(ws_str).expanduser()


def list_workspace_docs() -> list:
    """列出 workspace 下所有 .md 文件"""
    ws = get_workspace_path()
    if not ws.exists():
        return []
    return [f.name for f in ws.glob("*.md")]
