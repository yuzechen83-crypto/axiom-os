"""
Task Decomposer - 超级智能化 L3 编排层
将自然语言目标分解为可执行的子任务 DAG。
"""

from typing import Optional, List, Any
import re

from axiom_os.core.task import Task, Goal, TaskType, create_goal


# 规则模板：关键词 -> 子任务序列
GOAL_TEMPLATES = {
    "摩擦": ["simulate", "discover", "validate", "crystallize"],
    "耗散": ["simulate", "discover", "validate", "crystallize"],
    "能量": ["simulate", "read_kb", "discover"],
    "知识库": ["read_kb"],
    "定律": ["read_kb", "discover"],
    "发现": ["discover", "validate", "crystallize"],
    "结晶": ["discover", "crystallize"],
    "双摆": ["simulate", "discover"],
    "acrobot": ["simulate", "discover"],
    "RAR": ["run_rar_discovery"],
    "宇宙": ["run_rar_discovery"],
}


def _rule_based_decompose(goal_text: str) -> List[Task]:
    """规则分解：根据关键词生成子任务"""
    goal_lower = goal_text.lower()
    tasks = []
    used_types = set()

    for keyword, task_types in GOAL_TEMPLATES.items():
        if keyword in goal_lower:
            for i, tt in enumerate(task_types):
                if tt in used_types:
                    continue
                used_types.add(tt)
                tid = f"task_{len(tasks)}"
                deps = [tasks[j].id for j in range(max(0, len(tasks) - 1))] if tasks else []
                task_type = TaskType.CUSTOM
                if tt == "simulate":
                    task_type = TaskType.SIMULATE
                elif tt == "discover":
                    task_type = TaskType.DISCOVER
                elif tt == "crystallize":
                    task_type = TaskType.CRYSTALLIZE
                elif tt == "validate":
                    task_type = TaskType.VALIDATE
                elif tt == "read_kb":
                    task_type = TaskType.READ_KB
                tasks.append(Task(
                    id=tid,
                    description=_task_desc(tt, goal_text),
                    task_type=task_type,
                    dependencies=deps[-1:] if deps else [],
                    metadata={"action": tt},
                ))
            break

    if not tasks:
        tasks = [
            Task(id="task_0", description="运行仿真收集数据", task_type=TaskType.SIMULATE, metadata={"action": "simulate"}),
            Task(id="task_1", description="运行 Discovery 提取公式", task_type=TaskType.DISCOVER, dependencies=["task_0"], metadata={"action": "discover"}),
            Task(id="task_2", description="结晶到知识库", task_type=TaskType.CRYSTALLIZE, dependencies=["task_1"], metadata={"action": "crystallize"}),
        ]
    return tasks


def _task_desc(action: str, goal: str) -> str:
    d = {
        "simulate": "运行仿真收集数据",
        "discover": "运行 Discovery 提取物理公式",
        "validate": "验证发现的公式",
        "crystallize": "结晶公式到知识库",
        "read_kb": "读取知识库中的定律",
        "run_rar_discovery": "运行 RAR/Meta-Axis 宇宙学发现",
    }
    return d.get(action, action)


class TaskDecomposer:
    """
    目标分解器：自然语言 -> 子任务 DAG。
    支持 LLM 或规则回退。
    """

    def __init__(self, use_llm: bool = False, chief_scientist: Optional[Any] = None):
        self.use_llm = use_llm and chief_scientist is not None
        self.chief = chief_scientist

    def decompose(self, goal_text: str) -> Goal:
        """
        将目标文本分解为 Goal（含子任务 DAG）。
        """
        goal_text = goal_text.strip()
        if not goal_text:
            return create_goal("(空目标)", [])

        if self.use_llm and self.chief:
            tasks = self._llm_decompose(goal_text)
        else:
            tasks = _rule_based_decompose(goal_text)

        root = Task(
            id="goal_0",
            description=goal_text,
            task_type=TaskType.GOAL,
            sub_tasks=[],
        )
        for t in tasks:
            root.add_sub_task(t)
        return Goal(description=goal_text, root_task=root)

    def _llm_decompose(self, goal_text: str) -> List[Task]:
        """调用 LLM 分解（简化：解析 LLM 返回的 action 序列）"""
        prompt = f"""将以下目标分解为步骤，每行一个动作。可用动作：simulate, discover, validate, crystallize, read_kb, run_rar_discovery。
目标：{goal_text}
输出格式（每行一个）：action: 描述
"""
        try:
            resp = self.chief.ask(prompt, max_steps=1, max_tool_calls=0)
            return self._parse_llm_response(resp, goal_text)
        except Exception:
            return _rule_based_decompose(goal_text)

    def _parse_llm_response(self, resp: str, goal: str) -> List[Task]:
        """解析 LLM 返回的 action 列表"""
        tasks = []
        for line in resp.split("\n"):
            line = line.strip()
            if ":" in line:
                part = line.split(":", 1)
                action = part[0].strip().lower()
                desc = part[1].strip() if len(part) > 1 else _task_desc(action, goal)
            else:
                m = re.search(r"(simulate|discover|validate|crystallize|read_kb|run_rar_discovery)", line, re.I)
                action = m.group(1).lower() if m else "discover"
                desc = line
            tid = f"task_{len(tasks)}"
            deps = [tasks[-1].id] if tasks else []
            task_type = TaskType.CUSTOM
            if action == "simulate":
                task_type = TaskType.SIMULATE
            elif action == "discover":
                task_type = TaskType.DISCOVER
            elif action == "crystallize":
                task_type = TaskType.CRYSTALLIZE
            elif action == "validate":
                task_type = TaskType.VALIDATE
            elif action == "read_kb":
                task_type = TaskType.READ_KB
            tasks.append(Task(id=tid, description=desc, task_type=task_type, dependencies=deps, metadata={"action": action}))
        return tasks if tasks else _rule_based_decompose(goal)
