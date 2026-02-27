"""
Axiom 调度器：根据 LLM 层意图调用对应 Axiom 模块，返回统一 result 结构。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

# 项目根
ROOT = Path(__file__).resolve().parents[2]

from axiom_os.agent.llm_layer import (
    INTENT_BATTERY,
    INTENT_BENCHMARK_QUICK,
    INTENT_DISCOVERY_DEMO,
    INTENT_MAIN_SIM,
    INTENT_RAR,
    INTENT_TURBULENCE_PINN_LSTM,
    INTENT_VALIDATE_ALL,
    INTENT_GENERAL,
)


def run_intent(intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行意图对应的 Axiom 任务。
    返回: {"ok": bool, "result": dict, "error": str | None, "elapsed": float}
    """
    t0 = time.perf_counter()
    try:
        if intent == INTENT_RAR:
            from axiom_os.experiments.discovery_rar import run_rar_discovery
            n_galaxies = params.get("n_galaxies", 20)
            epochs = params.get("epochs", 200)
            res = run_rar_discovery(n_galaxies=n_galaxies, epochs=epochs)
            if res.get("error"):
                return {"ok": False, "result": res, "error": res["error"], "elapsed": time.perf_counter() - t0}
            return {"ok": True, "result": res, "error": None, "elapsed": time.perf_counter() - t0}

        if intent == INTENT_DISCOVERY_DEMO:
            from axiom_os.demos.discovery_demo import main
            main()
            return {"ok": True, "result": {"done": True}, "error": None, "elapsed": time.perf_counter() - t0}

        if intent == INTENT_BENCHMARK_QUICK:
            import sys
            from axiom_os.benchmarks.run_benchmarks import main as bench_main
            # 避免污染 sys.argv
            old_argv = sys.argv
            sys.argv = ["run_benchmarks", "--config", "quick", "--report", "--no-fail-on-alerts"]
            try:
                bench_main()
            finally:
                sys.argv = old_argv
            return {"ok": True, "result": {"report_dir": "axiom_os/benchmarks/results"}, "error": None, "elapsed": time.perf_counter() - t0}

        if intent == INTENT_VALIDATE_ALL:
            from axiom_os.validate_all import main as validate_main
            validate_main()
            return {"ok": True, "result": {"done": True}, "error": None, "elapsed": time.perf_counter() - t0}

        if intent == INTENT_BATTERY:
            from axiom_os.main_battery import main as battery_main
            battery_main(quick=True)
            return {"ok": True, "result": {"done": True}, "error": None, "elapsed": time.perf_counter() - t0}

        if intent == INTENT_TURBULENCE_PINN_LSTM:
            from axiom_os.experiments.run_turbulence_pinn_lstm import main as pinn_main
            pinn_main()
            return {"ok": True, "result": {"done": True}, "error": None, "elapsed": time.perf_counter() - t0}

        if intent == INTENT_MAIN_SIM:
            # 走原有 Coder + Runner 流程，由调用方传入生成的 artifacts 并调用 runner
            return {"ok": True, "result": {"dispatched": "main_sim", "need_artifacts": True}, "error": None, "elapsed": 0}

        if intent == INTENT_GENERAL:
            return {"ok": True, "result": {"reply": "你可以说：跑一下 RAR、跑基准、公式发现、全流程验证、电池 RUL 等。"}, "error": None, "elapsed": 0}

        return {"ok": False, "result": {}, "error": f"未知意图: {intent}", "elapsed": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "result": {}, "error": str(e), "elapsed": time.perf_counter() - t0}
