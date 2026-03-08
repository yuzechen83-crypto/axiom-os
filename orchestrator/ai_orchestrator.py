"""
AI Orchestrator - 智能化编排层

整合 ChiefScientist、活动监控、自动发现建议。
- recommend_discovery(): 当 soft_activity > 阈值时建议运行 Discovery
- get_chief_scientist(): 返回配置好的 ChiefScientist 供 Mission Control 使用
- auto_discovery_hint(): 返回 AI 提示（高活动时建议 discovery）
- auto_discovery(): 高活动时自动建议并可选执行 Discovery
- get_proactive_suggestion(): 主动建议：基于当前状态返回下一步建议
"""

from typing import Optional, Any, Tuple

from .llm_brain import ChiefScientist


class AIIntelligence:
    """
    AI 智能化编排：活动感知 + ChiefScientist + 自动发现建议。
    """

    def __init__(
        self,
        rcln: Optional[Any] = None,
        hippocampus: Optional[Any] = None,
        discovery: Optional[Any] = None,
        activity_threshold: float = 0.05,
        consecutive_high_steps: int = 3,
        use_mock_llm: bool = True,
        backend: str = "ollama",
        model: str = "llama3.2",
    ):
        self.rcln = rcln
        self.hippocampus = hippocampus
        self.discovery = discovery
        self.activity_threshold = activity_threshold
        self.consecutive_high_steps = consecutive_high_steps
        self._high_activity_count = 0

        self.chief = ChiefScientist(
            rcln=rcln,
            hippocampus=hippocampus,
            discovery=discovery,
            backend=backend,
            model=model,
            use_mock=use_mock_llm,
            max_tool_calls=5,
            max_steps=5,
        )

    def set_data_buffer(self, data_buffer: Optional[list]) -> None:
        """设置 Discovery 用的 (X, y_soft) 缓冲。"""
        self.chief.set_data_buffer(data_buffer)

    def get_activity(self) -> float:
        """获取当前 soft activity。"""
        return self.chief._tool_get_activity()

    def recommend_discovery(self) -> bool:
        """
        当 soft_activity 连续超过阈值时返回 True，建议运行 Discovery。
        """
        act = self.get_activity()
        if act > self.activity_threshold:
            self._high_activity_count += 1
            return self._high_activity_count >= self.consecutive_high_steps
        self._high_activity_count = 0
        return False

    def reset_activity_counter(self) -> None:
        """重置高活动计数（例如 Discovery 完成后）。"""
        self._high_activity_count = 0

    def get_proactive_suggestion(self) -> str:
        """
        主动建议：基于当前状态返回下一步建议（无需调用 LLM）。
        """
        return self.chief._tool_suggest_next_action()

    def auto_discovery_hint(self) -> str:
        """
        返回 AI 提示：高活动时建议 run_discovery。
        """
        act = self.get_activity()
        if act > self.activity_threshold:
            return f"[AI 提示] Soft activity = {act:.4f} (高). 建议 run_discovery() 提取新物理."
        return ""

    def auto_discovery(
        self,
        data_buffer: Optional[list] = None,
        execute: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        高活动时自动建议并可选执行 Discovery。
        Returns: (recommended, formula_or_message)
        """
        if not self.recommend_discovery():
            return False, None
        if not execute or self.discovery is None or self.rcln is None:
            return True, "建议执行 run_discovery()"
        buf = data_buffer or getattr(self.chief, "_data_buffer", None)
        if buf is None or (isinstance(buf, list) and len(buf) < 10):
            return True, "数据不足，需更多 (X, y_soft) 样本"
        try:
            import numpy as np
            n_in = len(np.asarray(buf[0][0]).ravel())
            formula = self.discovery.distill(
                self.rcln, buf, input_units=[[0, 0, 0, 0, 0]] * n_in
            )
            self.reset_activity_counter()
            return True, str(formula) if formula else "未发现公式"
        except Exception as e:
            return True, f"Discovery 错误: {e}"

    def ask(self, question: str, **kwargs) -> str:
        """
        向 ChiefScientist 提问。自动注入 activity_hint。
        """
        act = self.get_activity()
        return self.chief.ask(question, activity_hint=act, **kwargs)

    def get_chief_scientist(self) -> ChiefScientist:
        """返回 ChiefScientist 实例，供 Mission Control 等使用。"""
        return self.chief
