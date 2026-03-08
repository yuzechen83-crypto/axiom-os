"""
Multi-Agent Arena - 多 AI 讨论与竞争
Phase 1: 双 agent 轮流讨论，输出汇总
Phase 2: 任务驱动模式 - 先执行高难度任务，再基于真实结果讨论、分析、汇总
Phase 3: 自编程模式 - DeepSeek 讨论后，调用 Kimi CLI 根据建议修改代码

用法:
  axiom arena -m "如何优化 Grid Pulse MPC 的 Nadir 预测？"
  axiom arena --task high_freq --rounds 2   # 执行任务后讨论结果
  axiom arena --task lorenz_pinn_lstm --self-program  # 讨论后 Kimi CLI 自编程
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# 默认双 agent 角色（可扩展）
DEFAULT_AGENTS = [
    {"id": "conservative", "name": "保守派", "system": "你是 Axiom-OS 的保守派分析师。优先考虑稳定性、物理约束、可验证性。给出谨慎、可落地的建议。"},
    {"id": "explorer", "name": "探索派", "system": "你是 Axiom-OS 的探索派分析师。敢于提出创新方案、新协议、激进优化。与保守派形成对比与补充。"},
]

# 三 agent 讨论组（含 Kimi 公式发现者）
# Kimi 支持两种密钥来源，通过 KIMI_BASE_URL 选择：
# - Kimi Code 密钥 (sk-kimi-xxx): base_url=https://api.kimi.com/coding/v1, model=kimi-for-coding
#   注意：Kimi Code 仅限 Kimi CLI/Claude Code/Roo Code 等编码工具，通用调用可能返回 403
# - Moonshot 密钥: base_url=https://api.moonshot.ai/v1, model=moonshot-v1-8k 或 kimi-k2-turbo-preview
# - Kimi K2 密钥: base_url=https://kimi-k2.ai/api/v1, model=kimi-k2 或 kimi-k2-0905
def _kimi_formula_seeker_config() -> dict:
    base = (os.environ.get("KIMI_BASE_URL") or "https://api.moonshot.ai/v1").rstrip("/")
    if not base.startswith("http"):
        base = f"https://{base}"
    if "kimi.com" in base:
        model = "kimi-for-coding"
    elif "kimi-k2.ai" in base:
        model = "kimi-k2-0905"
    else:
        model = "moonshot-v1-8k"
    return {
        "id": "formula_seeker",
        "name": "公式发现者",
        "system": "你是 Axiom-OS 的公式发现者。擅长从数据中归纳物理规律、提出符号公式假设、建议可验证的改进。与保守派、探索派协作，推动从黑箱到可解释的公式结晶。",
        "model": model,
        "base_url": base,
        "api_key_env": "KIMI_API_KEY",
    }


def _get_agents_with_kimi() -> List[Dict[str, Any]]:
    """返回含 Kimi 公式发现者的 agent 列表（Kimi 配置由 KIMI_BASE_URL 决定）"""
    return [
        {"id": "conservative", "name": "保守派", "system": "你是 Axiom-OS 的保守派分析师。优先考虑稳定性、物理约束、可验证性。给出谨慎、可落地的建议。",
         "model": "deepseek-chat", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
        {"id": "explorer", "name": "探索派", "system": "你是 Axiom-OS 的探索派分析师。敢于提出创新方案、新协议、激进优化。与保守派形成对比与补充。",
         "model": "deepseek-chat", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
        _kimi_formula_seeker_config(),
    ]


AGENTS_WITH_KIMI = None  # 占位，实际用 _get_agents_with_kimi()

# 任务注册：{name: (runner_func, description)}
# runner_func() -> dict，必须返回 {"ok": bool, "results": Any, ...}
TASK_REGISTRY: Dict[str, tuple] = {}


def _register_tasks() -> None:
    """延迟注册可执行任务"""
    if TASK_REGISTRY:
        return

    def _run_high_freq() -> dict:
        from axiom_os.experiments.high_freq_benchmark import run_benchmark, DIFFICULTY_LEVELS
        t0 = time.perf_counter()
        results = run_benchmark(
            difficulty_keys=["L1_easy", "L2_medium", "L3_hard"],
            epochs=150,
            seq_len=8,
            seed=42,
        )
        elapsed = time.perf_counter() - t0
        return {"ok": True, "results": results, "elapsed": elapsed, "task": "high_freq"}

    def _run_lorenz_pinn_lstm() -> dict:
        from axiom_os.experiments.run_lorenz_pinn_lstm import main as lorenz_main
        t0 = time.perf_counter()
        results = lorenz_main()
        elapsed = time.perf_counter() - t0
        return {"ok": True, "results": results or {}, "elapsed": elapsed, "task": "lorenz_pinn_lstm"}

    def _run_real_data_long_term() -> dict:
        from axiom_os.experiments.real_data_long_term_benchmark import run_benchmark
        t0 = time.perf_counter()
        results = run_benchmark(
            forecast_days=5,
            seq_len=12,
            rollout_steps=[1, 10, 20, 50],
            epochs=300,
            use_synthetic_if_fail=True,
            seed=42,
        )
        elapsed = time.perf_counter() - t0
        if "error" in results:
            return {"ok": False, "error": results.get("error", "运行失败"), "results": results, "elapsed": elapsed}
        return {"ok": True, "results": results, "elapsed": elapsed, "task": "real_data_long_term"}

    TASK_REGISTRY["high_freq"] = (_run_high_freq, "高频失效基准：元轴+FNO 方案 A/B/C (L1-L3, 150 epochs)")
    TASK_REGISTRY["lorenz_pinn_lstm"] = (_run_lorenz_pinn_lstm, "Lorenz 混沌系统 PINN-LSTM 预测")
    TASK_REGISTRY["real_data_long_term"] = (_run_real_data_long_term, "真实数据长时间预测：Open-Meteo 大气湍流，1/10/20/50 步")


def _run_task(task_name: str) -> dict:
    """执行指定任务，返回 {ok, results, elapsed, error?}"""
    _register_tasks()
    if task_name not in TASK_REGISTRY:
        return {"ok": False, "error": f"未知任务: {task_name}", "available": list(TASK_REGISTRY.keys())}
    runner, _ = TASK_REGISTRY[task_name]
    try:
        return runner()
    except Exception as e:
        return {"ok": False, "error": str(e), "task": task_name}


def _invoke_kimi_cli(
    prompt: str,
    work_dir: Optional[Path] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    调用 Kimi CLI 执行自编程。使用 axiom_os_llm.env 中的 KIMI_API_KEY（Kimi Code 密钥）。
    返回 {ok, stdout, stderr, returncode, error?}
    """
    kimi_cmd = shutil.which("kimi") or shutil.which("kimi.exe")
    if not kimi_cmd:
        # 尝试 uv 安装的 kimi-cli
        uv_cmd = shutil.which("uv") or shutil.which("uv.exe")
        kimi_cmd = f"{uv_cmd} tool run kimi" if uv_cmd else None
    if not kimi_cmd:
        return {"ok": False, "error": "未找到 kimi 命令，请安装 Kimi Code CLI: https://code.kimi.com 或 uv tool install kimi-cli"}
    api_key = os.environ.get("KIMI_API_KEY") or ""
    if not api_key or not str(api_key).strip():
        return {"ok": False, "error": "请配置 KIMI_API_KEY（axiom_os/config/axiom_os_llm.env）"}
    work_dir = work_dir or ROOT
    config = {
        "default_model": "kimi-for-coding",
        "providers": {
            "kimi-for-coding": {
                "type": "kimi",
                "base_url": "https://api.kimi.com/coding/v1",
                "api_key": api_key.strip(),
            }
        },
        "models": {
            "kimi-for-coding": {
                "provider": "kimi-for-coding",
                "model": "kimi-for-coding",
                "max_context_size": 262144,
            }
        },
    }
    if " " in (kimi_cmd or ""):
        cmd = kimi_cmd.split() + ["--config", json.dumps(config), "--print", "--yolo", "--prompt", prompt]
    else:
        cmd = [kimi_cmd or "kimi", "--config", json.dumps(config), "--print", "--yolo", "--prompt", prompt]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Kimi CLI 超时 ({timeout}s)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _format_task_results(task_name: str, task_out: dict) -> str:
    """将任务输出格式化为可读文本，供 agent 分析"""
    lines = [
        f"【任务】{task_name}",
        f"【状态】{'成功' if task_out.get('ok') else '失败'}",
        f"【耗时】{task_out.get('elapsed', 0):.1f} 秒",
    ]
    if not task_out.get("ok"):
        lines.append(f"【错误】{task_out.get('error', '')}")
        return "\n".join(lines)

    res = task_out.get("results") or {}
    if task_name == "high_freq":
        lines.append("\n【各难度 R²】")
        for dk, row in res.get("by_difficulty", {}).items():
            for scheme, m in row.items():
                if isinstance(m, dict):
                    r2 = m.get("r2", m.get("r2_mean", 0))
                    lines.append(f"  {dk} / {scheme}: R2={r2:.4f}")
        lines.append("\n【方案汇总】")
        for scheme, agg in res.get("axiom_scores", {}).items():
            r2 = agg.get("r2_mean", agg.get("r2", 0))
            axi = agg.get("axiom_score_mean", agg.get("axiom_score", 0))
            lines.append(f"  {scheme}: R2={r2:.4f} Axiom_score={axi:.3f}")
    elif task_name == "lorenz_pinn_lstm":
        lines.append(f"\n【指标】R2={res.get('r2', 0):.4f} MAE={res.get('mae', 0):.4f} Euler_R2={res.get('r2_euler', 0):.4f}")
    elif task_name == "real_data_long_term":
        lines.append("\n【数值精度】")
        np_ = res.get("numerical_precision", {})
        lines.append(f"  1步: MAE={np_.get('mae_1step', 0):.4f} R2={np_.get('r2_1step', 0):.4f}")
        lines.append(f"  长时: MAE={np_.get('mae_longterm', 0):.4f} R2={np_.get('r2_longterm', 0):.4f}")
        lines.append("\n【物理一致性】")
        pc = res.get("physical_consistency", {})
        lines.append(f"  PhysOK={pc.get('phys_bounds_ok', 0):.2%} Coach={pc.get('coach_score', 0):.4f}")
        for k, v in res.get("by_rollout", {}).items():
            lines.append(f"  {k}: MAE={v.get('mae', 0):.4f} R2={v.get('r2', 0):.4f}")
    else:
        lines.append("\n【原始结果】")
        lines.append(json.dumps(res, indent=2, ensure_ascii=False)[:2000])
    return "\n".join(lines)


def _invoke_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    base_url: str = "https://api.deepseek.com",
    temperature: float = 0.7,
    max_tokens: int = 1500,
) -> str:
    """单轮 LLM 调用（无工具）"""
    kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if base_url and "deepseek" in base_url:
        pass  # client 已用 base_url 初始化
    r = client.chat.completions.create(**kwargs)
    if not r.choices:
        return ""
    return (r.choices[0].message.content or "").strip()


def _get_client_for_agent(agent: dict, default_key: str, default_base_url: str) -> tuple:
    """返回 (client, model) 用于该 agent。支持 per-agent model/base_url/api_key_env。"""
    model = agent.get("model") or "deepseek-chat"
    base_url = agent.get("base_url") or default_base_url
    key_env = agent.get("api_key_env") or "DEEPSEEK_API_KEY"
    key = os.environ.get(key_env) or default_key
    if not key or not str(key).strip():
        return None, model
    return OpenAI(api_key=key, base_url=base_url), model


def run_arena(
    topic: str,
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    agents: Optional[List[Dict[str, Any]]] = None,
    rounds: int = 2,
    base_url: str = "https://api.deepseek.com",
    use_kimi: bool = False,
) -> Dict[str, Any]:
    """
    运行多 agent 讨论。
    topic: 讨论主题
    agents: 可选，自定义 agent 列表。若 use_kimi=True 则用 AGENTS_WITH_KIMI（含 Kimi 公式发现者）
    rounds: 每 agent 发言轮数
    use_kimi: 是否加入 Kimi 作为第三 agent（公式发现者）
    Returns: {topic, turns: [{agent, name, content}], summary}
    """
    if not HAS_OPENAI:
        return {"ok": False, "error": "请安装 openai: pip install openai"}

    default_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if use_kimi:
        agents = _get_agents_with_kimi()
        kimi_key = os.environ.get("KIMI_API_KEY")
        if not kimi_key or not str(kimi_key).strip():
            return {"ok": False, "error": "请配置 KIMI_API_KEY（axiom_os/config/axiom_os_llm.env）"}
    else:
        if not default_key or not str(default_key).strip():
            return {"ok": False, "error": "请配置 DEEPSEEK_API_KEY"}

    agents = agents or DEFAULT_AGENTS

    turns: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []

    for r in range(rounds):
        for i, agent in enumerate(agents):
            aid = agent.get("id", f"agent_{i}")
            name = agent.get("name", aid)
            system = agent.get("system", "你是 Axiom-OS 的分析师。")

            client, agent_model = _get_client_for_agent(agent, default_key or "", base_url)
            if client is None:
                content = f"[{name} 未配置 API Key]"
            else:
                ctx = f"【讨论主题】\n{topic}\n\n"
                if history:
                    ctx += "【此前发言】\n"
                    for h in history[-6:]:
                        c = h.get("content", "")
                        ctx += f"- {h.get('name', '?')}: {c[:300]}...\n" if len(c) > 300 else f"- {h.get('name', '?')}: {c}\n"
                    ctx += "\n请基于上述讨论，给出你的观点或补充（简洁，200 字内）。\n"
                else:
                    ctx += "请给出你的初步分析与建议（简洁，200 字内）。\n"

                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": ctx},
                ]
                try:
                    content = _invoke_llm(client, agent_model, messages)
                except Exception as e:
                    err = str(e)
                    if "403" in err and "kimi.com" in (agent.get("base_url") or ""):
                        err += " (Kimi Code 密钥仅限 Kimi CLI/Claude Code/Roo Code 等编码工具，Arena 需用 Moonshot 密钥: platform.moonshot.ai)"
                    content = f"[调用失败: {err}]"

            turns.append({"agent": aid, "name": name, "content": content})
            history.append({"name": name, "content": content})

    # 汇总：优先用 DeepSeek
    summary_client = OpenAI(api_key=default_key or os.environ.get("DEEPSEEK_API_KEY"), base_url=base_url)
    summary_prompt = f"【讨论主题】{topic}\n\n【发言记录】\n"
    for t in turns:
        summary_prompt += f"{t['name']}: {t['content']}\n\n"
    summary_prompt += "\n请用 3-5 句话汇总各方观点与共识/分歧，并给出可执行的改进建议（含公式发现方向）。"

    try:
        summary_msg = [
            {"role": "system", "content": "你是讨论记录员，负责客观汇总多 agent 讨论结果。"},
            {"role": "user", "content": summary_prompt},
        ]
        summary = _invoke_llm(summary_client, model, summary_msg)
    except Exception as e:
        summary = f"汇总生成失败: {e}"

    result = {
        "ok": True,
        "topic": topic,
        "turns": turns,
        "summary": summary,
    }
    return result


def run_arena_with_self_program(
    topic: str,
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    agents: Optional[List[Dict[str, Any]]] = None,
    rounds: int = 2,
    base_url: str = "https://api.deepseek.com",
) -> Dict[str, Any]:
    """纯讨论 + Kimi CLI 自编程。讨论仅用 DeepSeek，汇总后调用 Kimi CLI。"""
    arena_result = run_arena(
        topic=topic,
        api_key=api_key,
        model=model,
        agents=agents or DEFAULT_AGENTS,
        rounds=rounds,
        base_url=base_url,
        use_kimi=False,
    )
    if not arena_result.get("ok"):
        return arena_result

    kimi_prompt = (
        f"""根据以下 DeepSeek 讨论的改进建议，分析并修改当前项目代码。

【讨论主题】
{topic[:1500]}...

【讨论汇总与可执行建议】
{arena_result["summary"]}

请：
1. 理解建议内容，定位相关文件
2. 实施具体代码修改（优先高价值、低风险）
3. 简要说明做了哪些改动
"""
    )
    kimi_out = _invoke_kimi_cli(kimi_prompt, work_dir=ROOT)
    arena_result["kimi_cli"] = kimi_out
    return arena_result


def run_arena_task(
    task_name: str,
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    agents: Optional[List[Dict[str, str]]] = None,
    rounds: int = 2,
    base_url: str = "https://api.deepseek.com",
    use_kimi: bool = True,
    self_program: bool = False,
) -> Dict[str, Any]:
    """
    任务驱动 Arena：先执行高难度任务，再基于真实结果讨论、分析、汇总。
    流程：运行任务 -> 分析结果（多 agent）-> 汇总 -> [可选] Kimi CLI 自编程
    use_kimi: 是否加入 Kimi API 参与讨论（若 self_program=True 则讨论阶段仅用 DeepSeek）
    self_program: 讨论后调用 Kimi CLI 根据建议修改代码
    """
    task_out = _run_task(task_name)
    if not task_out.get("ok"):
        return {"ok": False, "error": task_out.get("error", "任务执行失败"), "task": task_name}

    results_text = _format_task_results(task_name, task_out)
    formula_prompt = (
        " 4) 尝试从数据/结果中发现物理规律、提出可验证的符号公式假设，推动公式结晶与改进。"
        if use_kimi and not self_program else ""
    )
    topic = (
        f"【真实运行结果】\n{results_text}\n\n"
        "请基于上述真实运行结果进行分析：1) 各方案表现如何、优劣在哪；2) 可能的问题与瓶颈；"
        "3) 可执行的改进建议；" + formula_prompt
    )

    # 自编程时讨论阶段仅用 DeepSeek，Kimi 通过 CLI 执行
    arena_use_kimi = use_kimi and not self_program
    arena_result = run_arena(
        topic=topic,
        api_key=api_key,
        model=model,
        agents=agents,
        rounds=rounds,
        base_url=base_url,
        use_kimi=arena_use_kimi,
    )
    if not arena_result.get("ok"):
        return arena_result

    arena_result["task"] = task_name
    arena_result["task_elapsed"] = task_out.get("elapsed", 0)
    arena_result["task_results"] = task_out.get("results")

    if self_program:
        kimi_prompt = (
            f"""根据以下 DeepSeek 讨论的改进建议，分析并修改当前项目代码。

【讨论主题】
{topic[:1500]}...

【讨论汇总与可执行建议】
{arena_result["summary"]}

请：
1. 理解建议内容，定位相关文件
2. 实施具体代码修改（优先高价值、低风险）
3. 简要说明做了哪些改动
"""
        )
        kimi_out = _invoke_kimi_cli(kimi_prompt, work_dir=ROOT)
        arena_result["kimi_cli"] = kimi_out

    return arena_result


def main_cli(
    message: Optional[str] = None,
    rounds: int = 2,
    task: Optional[str] = None,
    use_kimi: bool = True,
    self_program: bool = False,
) -> int:
    """CLI 入口。支持 --task 任务驱动、-m 纯讨论、--self-program Kimi CLI 自编程"""
    _env = ROOT / "axiom_os" / "config" / "axiom_os_llm.env"
    if _env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env)
        except ImportError:
            pass

    if task:
        result = run_arena_task(
            task_name=task,
            rounds=rounds,
            use_kimi=use_kimi,
            self_program=self_program,
        )
    else:
        if not message:
            print("请提供 -m/--message 或 --task")
            return 1
        if self_program:
            result = run_arena_with_self_program(topic=message, rounds=rounds)
        else:
            result = run_arena(topic=message, rounds=rounds, use_kimi=use_kimi)

    if not result.get("ok"):
        print(result.get("error", "Unknown error"))
        return 1

    print("=" * 60)
    print("Multi-Agent Arena" + (f" [Task: {result.get('task', '')}]" if result.get("task") else "") + (" [Self-Program]" if self_program else ""))
    print("=" * 60)
    if result.get("task_elapsed") is not None:
        print(f"Task elapsed: {result['task_elapsed']:.1f}s")
    print(f"Topic: {result['topic'][:200]}...\n" if len(result.get("topic", "")) > 200 else f"Topic: {result.get('topic', '')}\n")
    for t in result["turns"]:
        print(f"[{t['name']}]")
        print(t["content"])
        print()
    print("-" * 60)
    print("Summary:")
    print(result["summary"])
    print("=" * 60)

    if self_program and result.get("kimi_cli"):
        kc = result["kimi_cli"]
        print("\n" + "=" * 60)
        print("Kimi CLI 自编程")
        print("=" * 60)
        if not kc.get("ok"):
            print(f"状态: 失败 - {kc.get('error', kc.get('stderr', ''))}")
        else:
            print("状态: 完成")
            if kc.get("stdout"):
                print("\n输出:")
                print(kc["stdout"][:3000] + ("..." if len(kc.get("stdout", "")) > 3000 else ""))
        print("=" * 60)

    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--message", help="Discussion topic (纯讨论模式)")
    p.add_argument("--task", help="任务驱动: high_freq | lorenz_pinn_lstm | real_data_long_term")
    p.add_argument("--rounds", type=int, default=2, help="Rounds per agent")
    p.add_argument("--no-kimi", action="store_true", help="不使用 Kimi API，仅 DeepSeek 双 agent")
    p.add_argument("--self-program", action="store_true", help="讨论后调用 Kimi CLI 根据建议修改代码")
    args = p.parse_args()
    sys.exit(main_cli(
        message=args.message,
        rounds=args.rounds,
        task=args.task,
        use_kimi=not args.no_kimi,
        self_program=args.self_program,
    ))
