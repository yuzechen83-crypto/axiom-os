"""
Task & Goal Representation - 超级智能化 L5 认知层
目标树、子任务、依赖、状态。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskType(Enum):
    """任务类型"""
    GOAL = "goal"           # 顶层目标
    DISCOVER = "discover"   # 运行 Discovery
    CRYSTALLIZE = "crystallize"  # 结晶公式
    SIMULATE = "simulate"   # 运行仿真
    VALIDATE = "validate"   # 验证公式
    READ_KB = "read_kb"     # 读取知识库
    CUSTOM = "custom"       # 自定义


@dataclass
class Task:
    """
    任务表示：goal, sub_tasks, status, dependencies.
    支持目标树与 DAG 依赖。
    """
    id: str
    description: str
    task_type: TaskType = TaskType.CUSTOM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # 依赖的 task id
    sub_tasks: List["Task"] = field(default_factory=list)
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_sub_task(self, task: "Task") -> None:
        self.sub_tasks.append(task)

    def is_ready(self, completed_ids: set) -> bool:
        """依赖是否已满足"""
        return all(dep in completed_ids for dep in self.dependencies)

    def all_ids(self) -> set:
        """递归收集所有 task id"""
        ids = {self.id}
        for st in self.sub_tasks:
            ids.update(st.all_ids())
        return ids


@dataclass
class Goal:
    """
    顶层目标：自然语言描述 + 根任务。
    """
    description: str
    root_task: Task
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_goal(description: str, sub_tasks: Optional[List[Task]] = None) -> Goal:
    """创建目标，根任务为 GOAL 类型"""
    root = Task(
        id="goal_0",
        description=description,
        task_type=TaskType.GOAL,
        sub_tasks=sub_tasks or [],
    )
    return Goal(description=description, root_task=root)
