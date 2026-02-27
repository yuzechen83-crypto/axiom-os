"""
Axiom LLM 层：自然语言 -> 意图识别与参数抽取；执行结果 -> 自然语言回复。
支持 OpenAI/Anthropic 或基于关键词的回退。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

# 支持的意图
INTENT_RAR = "run_rar"
INTENT_DISCOVERY_DEMO = "run_discovery_demo"
INTENT_BENCHMARK_QUICK = "run_benchmark_quick"
INTENT_VALIDATE_ALL = "run_validate_all"
INTENT_BATTERY = "run_battery"
INTENT_TURBULENCE_PINN_LSTM = "run_turbulence_pinn_lstm"
INTENT_MAIN_SIM = "run_main_sim"
INTENT_GENERAL = "general_chat"

INTENTS = [
    INTENT_RAR,
    INTENT_DISCOVERY_DEMO,
    INTENT_BENCHMARK_QUICK,
    INTENT_VALIDATE_ALL,
    INTENT_BATTERY,
    INTENT_TURBULENCE_PINN_LSTM,
    INTENT_MAIN_SIM,
    INTENT_GENERAL,
]

# 关键词 -> 意图（回退用）
KEYWORD_INTENT = [
    (["rar", "星系", "旋转曲线", "径向加速度", "sparc"], INTENT_RAR),
    (["公式发现", "discovery", "符号回归", "多项式恢复"], INTENT_DISCOVERY_DEMO),
    (["基准", "benchmark", "quick", "跑分"], INTENT_BENCHMARK_QUICK),
    (["全流程", "validate", "验证", "全部"], INTENT_VALIDATE_ALL),
    (["电池", "battery", "rul", "寿命"], INTENT_BATTERY),
    (["湍流 pinn", "pinn lstm", "湍流lstm", "turbulence pinn", "时空预测"], INTENT_TURBULENCE_PINN_LSTM),
    (["双摆", "仿真", "模拟", "pendulum", "mpc", "控制"], INTENT_MAIN_SIM),
]

SYSTEM_PROMPT_INTENT = """你是一个物理 AI 助手，负责理解用户意图并选择 Axiom-OS 要执行的任务。

可选意图（只返回其中一个）：
- run_rar: 运行 RAR 星系旋转曲线发现（径向加速度关系）
- run_discovery_demo: 运行公式发现 Demo（从带噪数据恢复 y=x0²+0.5*x1）
- run_benchmark_quick: 运行快速基准测试（约 30 秒）
- run_validate_all: 运行全流程验证（main + rar + battery）
- run_battery: 仅运行电池 RUL 预测
- run_turbulence_pinn_lstm: 湍流 PINN-LSTM 时空预测（Hard Core + LSTM）
- run_main_sim: 双摆/控制仿真（需要生成代码）
- general_chat: 仅对话，不执行 Axiom 任务

请根据用户输入判断意图，并返回 JSON（仅一行，无 markdown）：
{"intent": "上述之一", "params": {}, "reasoning": "简短理由"}

params 可包含：n_galaxies（RAR 星系数，默认 20）, epochs（RAR 训练轮数，默认 200）。
"""


def _keyword_plan(user_message: str) -> Dict[str, Any]:
    """基于关键词的意图识别（无 LLM 时使用）"""
    msg_lower = user_message.lower().strip()
    for keywords, intent in KEYWORD_INTENT:
        if any(kw in msg_lower or kw in user_message for kw in keywords):
            params = {}
            if intent == INTENT_RAR:
                m = re.search(r"(\d+)\s*个?星系", user_message)
                if m:
                    params["n_galaxies"] = min(int(m.group(1)), 100)
            return {"intent": intent, "params": params, "reasoning": "关键词匹配"}
    # 短句或模糊 -> 默认快速基准或对话
    if len(msg_lower) <= 4 or msg_lower in ("跑一下", "测试", "试一下"):
        return {"intent": INTENT_BENCHMARK_QUICK, "params": {}, "reasoning": "默认快速验证"}
    return {"intent": INTENT_GENERAL, "params": {}, "reasoning": "未匹配到任务"}


class AxiomLLMLayer:
    """
    LLM 层：意图识别 + 自然语言回复。
    有 API Key 时用 LLM，否则用关键词回退。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url
        self._client = None
        if api_key:
            self._init_client()

    def _init_client(self) -> None:
        if self.provider == "openai":
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key or ""}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                self._client = None
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key or "")
            except ImportError:
                self._client = None
        else:
            self._client = None

    def plan(self, user_message: str) -> Dict[str, Any]:
        """
        从用户输入解析意图与参数。
        返回: {"intent": str, "params": dict, "reasoning": str}
        """
        if not self._client or not self.api_key:
            return _keyword_plan(user_message)

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_INTENT},
                {"role": "user", "content": user_message},
            ]
            if self.provider == "openai":
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=256,
                )
                content = (resp.choices[0].message.content or "").strip()
            elif self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=SYSTEM_PROMPT_INTENT,
                    messages=[{"role": "user", "content": user_message}],
                )
                content = (resp.content[0].text if resp.content else "").strip()
            else:
                return _keyword_plan(user_message)

            # 抽取 JSON
            for line in content.splitlines():
                line = line.strip().strip("`")
                if line.startswith("{"):
                    data = json.loads(line)
                    intent = data.get("intent", INTENT_GENERAL)
                    if intent not in INTENTS:
                        intent = INTENT_GENERAL
                    return {
                        "intent": intent,
                        "params": data.get("params") or {},
                        "reasoning": data.get("reasoning") or "",
                    }
        except Exception:
            pass
        return _keyword_plan(user_message)

    def format_reply(self, intent: str, result: Dict[str, Any], error: Optional[str] = None) -> str:
        """将执行结果格式化为自然语言回复。"""
        if error:
            return f"执行出错：{error}"

        if intent == INTENT_RAR:
            r2 = result.get("r2_log") or result.get("r2")
            n = result.get("n_samples", 0)
            return f"RAR 发现已完成。对数空间 R² = {r2:.4f}，样本数 = {n}。"
        if intent == INTENT_DISCOVERY_DEMO:
            return "公式发现 Demo 已跑完，详见控制台或报告。"
        if intent == INTENT_BENCHMARK_QUICK:
            return "快速基准已跑完，报告在 axiom_os/benchmarks/results/（含 benchmark_report.html）。"
        if intent == INTENT_VALIDATE_ALL:
            return "全流程验证已跑完（main + RAR + battery）。"
        if intent == INTENT_BATTERY:
            return "电池 RUL 任务已跑完。"
        if intent == INTENT_TURBULENCE_PINN_LSTM:
            return "湍流 PINN-LSTM 时空预测已跑完，详见控制台与 axiom_os/experiments/ 下的图表。"
        if intent == INTENT_MAIN_SIM:
            discovered = result.get("discovered", [])
            if discovered:
                return f"仿真完成，发现公式：{discovered[0]}"
            return "仿真完成，未发现新公式。"
        if intent == INTENT_GENERAL:
            return result.get("reply", "请直接描述你想执行的 Axiom 任务，例如：跑一下 RAR、跑基准、公式发现、湍流 PINN-LSTM。")
        return str(result)
