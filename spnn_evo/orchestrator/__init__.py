"""SPNN-Evo Orchestrator: MPC, MPPI, Imagination-Augmented Control, Policy Distillation"""

from .mpc import ParallelImaginationController
from .distillation import (
    PolicyTrainer,
    StudentPolicy,
    DoublePendulumEnv,
    ReplayBuffer,
)
