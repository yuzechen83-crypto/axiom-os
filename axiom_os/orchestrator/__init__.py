"""
Axiom-OS Orchestrator: Imagination-Augmented Controller, Chief Scientist, Policy Distillation, AI Intelligence
"""

from .mpc import ImaginationMPC
from .llm_brain import ChiefScientist
from .distillation import PolicyTrainer, StudentPolicy, ReplayBuffer, DoublePendulumEnv
from .ai_orchestrator import AIIntelligence
from .task_decomposer import TaskDecomposer
from .task_executor import TaskExecutor

__all__ = [
    "ImaginationMPC",
    "ChiefScientist",
    "AIIntelligence",
    "PolicyTrainer",
    "StudentPolicy",
    "ReplayBuffer",
    "DoublePendulumEnv",
    "TaskDecomposer",
    "TaskExecutor",
]
