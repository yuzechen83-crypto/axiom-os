"""
Task Executor - 超级智能化 L3 编排层
按依赖顺序执行子任务，调用 Discovery/Hippocampus/ChiefScientist。
"""

from typing import Optional, Any, List
import numpy as np

from axiom_os.core.task import Task, Goal, TaskStatus, TaskType


class TaskExecutor:
    """
    任务执行器：按 DAG 依赖执行子任务。
    """

    def __init__(
        self,
        rcln: Optional[Any] = None,
        hippocampus: Optional[Any] = None,
        discovery: Optional[Any] = None,
        chief_scientist: Optional[Any] = None,
        data_buffer: Optional[List[tuple]] = None,
    ):
        self.rcln = rcln
        self.hippocampus = hippocampus
        self.discovery = discovery
        self.chief = chief_scientist
        self._data_buffer = data_buffer
        self._last_formula: Optional[str] = None

    def set_data_buffer(self, data_buffer: Optional[List[tuple]]) -> None:
        self._data_buffer = data_buffer
        if self.chief:
            self.chief.set_data_buffer(data_buffer)

    def execute_goal(self, goal: Goal) -> dict:
        """
        执行目标，返回 {task_id: result, ...} 及整体状态。
        """
        root = goal.root_task
        results = {}
        completed = set()

        # 扁平化子任务（按依赖拓扑排序）
        def collect_ready(t: Task, out: List[Task]) -> None:
            for st in t.sub_tasks:
                if st.id not in [x.id for x in out]:
                    out.append(st)

        flat = []
        collect_ready(root, flat)

        for task in flat:
            if not task.is_ready(completed):
                continue
            task.status = TaskStatus.RUNNING
            try:
                res = self._execute_action(task)
                task.result = res
                task.status = TaskStatus.SUCCESS
                results[task.id] = res
                completed.add(task.id)
                if task.metadata.get("action") == "discover" and res:
                    s = str(res)
                    self._last_formula = s.replace("Found formula: ", "").strip() if "Found formula:" in s else s
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.result = str(e)
                results[task.id] = f"Error: {e}"
                completed.add(task.id)

        return {"results": results, "completed": list(completed), "last_formula": self._last_formula}

    def _execute_action(self, task: Task) -> Any:
        """执行单个任务"""
        action = task.metadata.get("action", "discover")

        if action == "read_kb":
            return self._do_read_kb()
        if action == "simulate":
            return self._do_simulate()
        if action == "discover":
            return self._do_discover()
        if action == "validate":
            return self._do_validate()
        if action == "crystallize":
            return self._do_crystallize()
        if action == "run_rar_discovery":
            return self._do_rar_discovery()

        if self.chief:
            return self.chief.ask(f"执行: {task.description}", max_steps=3)
        return f"Action {action} (no executor)"

    def _do_read_kb(self) -> str:
        if self.hippocampus is None:
            return "No hippocampus"
        kb = dict(self.hippocampus.knowledge_base)
        return f"Knowledge base: {len(kb)} laws. " + (str(list(kb.keys())[:5]) if kb else "")

    def _do_simulate(self) -> str:
        return "Simulation: run main loop to collect data. (Use run_phase or main())"

    def _do_discover(self) -> str:
        if self.discovery is None or self.rcln is None:
            return "Error: No discovery or RCLN"
        buf = self._data_buffer or (getattr(self.chief, "_data_buffer", None) if self.chief else None)
        if buf is None or (isinstance(buf, list) and len(buf) < 10):
            return "No data buffer. Run simulation first."
        n_in = len(np.asarray(buf[0][0]).ravel())
        formula = self.discovery.distill(
            self.rcln, buf, input_units=[[0, 0, 0, 0, 0]] * n_in
        )
        self._last_formula = str(formula) if formula else None
        return f"Found formula: {formula}" if formula else "No formula found"

    def _do_validate(self) -> str:
        if self._last_formula and self.discovery and self._data_buffer:
            X = np.stack([np.asarray(p[0]).ravel() for p in self._data_buffer[-100:]])
            y = np.stack([np.asarray(p[1]).ravel() for p in self._data_buffer[-100:]])
            valid, mse = self.discovery.validate_formula(self._last_formula, X, y, output_dim=y.shape[1])
            return f"Valid: {valid}, MSE: {mse:.6f}"
        return "No formula to validate. Run discover first."

    def _do_crystallize(self) -> str:
        if not self._last_formula or self.hippocampus is None or self.rcln is None:
            return "No formula to crystallize. Run discover first."
        try:
            fid = self.hippocampus.crystallize(self._last_formula, self.rcln)
            return f"Crystallized as {fid}"
        except Exception as e:
            return f"Error: {e}"

    def _do_rar_discovery(self) -> str:
        if self.chief:
            return self.chief._tool_run_rar_discovery()
        try:
            from axiom_os.experiments.discovery_rar import run_rar_discovery
            out = run_rar_discovery(n_galaxies=50, epochs=100)
            return str(out)[:300]
        except Exception as e:
            return f"Error: {e}"
