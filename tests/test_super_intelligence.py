"""
Unit tests for Super Intelligentization (Task, TaskDecomposer, TaskExecutor, SymplecticCausal)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_task_representation():
    """Test Task and Goal dataclasses."""
    from axiom_os.core.task import Task, Goal, TaskType, TaskStatus, create_goal

    t = Task(id="t1", description="discover", task_type=TaskType.DISCOVER)
    assert t.status == TaskStatus.PENDING
    assert t.is_ready(set())
    assert not t.is_ready(set()) or "t0" in t.dependencies or t.dependencies == []

    goal = create_goal("test", [t])
    assert goal.root_task.sub_tasks


def test_task_decomposer():
    """Test TaskDecomposer rule-based decomposition."""
    from axiom_os.orchestrator.task_decomposer import TaskDecomposer

    d = TaskDecomposer()
    goal = d.decompose("研究双摆摩擦")
    assert goal.description == "研究双摆摩擦"
    assert len(goal.root_task.sub_tasks) >= 1


def test_symplectic_causal():
    """Test symplectic causal constraints."""
    from axiom_os.core.symplectic_causal import (
        get_symplectic_causal_edges,
        get_symplectic_allowed_inputs,
        filter_formula_by_symplectic,
    )

    edges = get_symplectic_causal_edges(4)
    assert len(edges) > 0

    allowed = get_symplectic_allowed_inputs(0, 4)  # dq/dt
    assert 0 not in allowed
    assert 1 not in allowed
    assert 2 in allowed
    assert 3 in allowed

    assert filter_formula_by_symplectic("0.5*x2 + 0.3*x3", 0, 4)
    assert not filter_formula_by_symplectic("0.5*x0", 0, 4)


def test_light_cone_filter():
    """Test light cone check."""
    from axiom_os.core.light_cone_filter import check_light_cone

    assert check_light_cone(0, [0, 0, 0], 1, [0.5, 0, 0], c=1)
    assert not check_light_cone(0, [0, 0, 0], 0.5, [2, 0, 0], c=1)


def test_task_executor():
    """Test TaskExecutor with mock components."""
    import torch
    from axiom_os.core import Hippocampus
    from axiom_os.engine import DiscoveryEngine
    from axiom_os.layers import RCLNLayer
    from axiom_os.orchestrator import TaskDecomposer, TaskExecutor

    def hc(x):
        return torch.zeros_like(torch.as_tensor(x.values if hasattr(x, "values") else x, dtype=torch.float32))
    rcln = RCLNLayer(1, 32, 1, hard_core_func=hc, lambda_res=1.0)
    hippo = Hippocampus()
    eng = DiscoveryEngine(use_pysr=False)
    decomposer = TaskDecomposer()
    goal = decomposer.decompose("读取知识库")
    executor = TaskExecutor(rcln=rcln, hippocampus=hippo, discovery=eng)
    result = executor.execute_goal(goal)
    assert "results" in result
    assert len(result["results"]) >= 1 or "completed" in result


if __name__ == "__main__":
    test_task_representation()
    test_task_decomposer()
    test_symplectic_causal()
    test_light_cone_filter()
    test_task_executor()
    print("All super intelligence tests passed.")
