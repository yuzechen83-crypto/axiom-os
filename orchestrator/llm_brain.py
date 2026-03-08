"""
Chief Scientist - LLM Agent for Axiom-OS (ReAct style)
Tools: get_activity, read_hippocampus, write_hippocampus, run_discovery, run_rar_discovery, plot_results

Production-grade: max_tool_calls, RAG (Hippocampus retrieval), retry with circuit breaker.
Supports: Ollama (open-source), OpenAI.
AI 智能化: activity-aware prompts, auto-suggest discovery, RAR/Meta 实验工具.
"""

from typing import Optional, List, Dict, Any
import json
import re

import numpy as np

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Tool signatures (structured for LLM)
TOOL_DESCRIPTIONS = """
Available tools:
- get_activity(): Get RCLN soft shell activity (high = unmodeled dynamics).
- get_uncertainty(): Get uncertainty status if RCLN has return_uncertainty.
- read_hippocampus(): Read crystallized laws from knowledge base.
- read_hippocampus_partition(partition_id): Read laws in a specific partition (e.g. rar_low, battery_early).
- write_hippocampus(formula, formula_id): Crystallize a formula into physics core.
- run_discovery(): Run Discovery Engine on RCLN residuals.
- run_rar_discovery(): Run RAR/Meta-Axis discovery (SPARC galaxies, McGaugh fit).
- plot_results(path): Save current results to image.
- diagnose_system(): Get system health: activity, uncertainty, knowledge count.
- suggest_next_action(): AI suggests next step based on current state.
- run_goal(goal_text): Decompose goal into tasks and execute (super-intelligentization).
"""


def _rag_retrieve(
    hippocampus: Any,
    question: str,
    top_k: int = 5,
    partition_id: Optional[str] = None,
    domain: Optional[str] = None,
) -> str:
    """
    RAG: Retrieve relevant laws from Hippocampus for context.
    支持 partition_id/domain 过滤（分区检索增强）。
    """
    if hippocampus is None or not hasattr(hippocampus, "knowledge_base"):
        return ""
    if hasattr(hippocampus, "retrieve_by_query"):
        return (
            hippocampus.retrieve_by_query(
                question, top_k=top_k, partition_id=partition_id, domain=domain
            )
            or ""
        )
    kb = hippocampus.knowledge_base
    if not kb:
        return ""
    items = []
    for fid, meta in list(kb.items())[:top_k]:
        formula = meta.get("formula", str(meta)) if isinstance(meta, dict) else str(meta)
        items.append(f"- {fid}: {formula}")
    return "Known laws:\n" + "\n".join(items) if items else ""


class ChiefScientist:
    """
    LLM Agent that reasons about physics models.
    ReAct loop: Thought -> Action -> Observation -> Reply.
    Production: max_tool_calls, RAG, retry, circuit breaker.
    """

    def __init__(
        self,
        rcln: Optional[Any] = None,
        hippocampus: Optional[Any] = None,
        discovery: Optional[Any] = None,
        api_key: Optional[str] = None,
        model: str = "llama3.2",
        backend: str = "ollama",
        base_url: Optional[str] = None,
        use_mock: bool = False,
        max_tool_calls: int = 5,
        max_steps: int = 5,
        retry_times: int = 2,
    ):
        """
        backend: "ollama" | "openai"
        base_url: For Ollama use "http://localhost:11434/v1"
        max_tool_calls: Hard limit per ask() to prevent runaway.
        retry_times: Retries on tool/LLM failure before giving up.
        """
        self.rcln = rcln
        self.hippocampus = hippocampus
        self.discovery = discovery
        self.api_key = api_key or ("ollama" if backend == "ollama" else "")
        self.model = model
        self.backend = backend
        self.base_url = base_url or ("http://localhost:11434/v1" if backend == "ollama" else None)
        self.use_mock = use_mock
        self.max_tool_calls = max_tool_calls
        self.max_steps = max_steps
        self.retry_times = retry_times
        self._history: List[Dict[str, str]] = []
        self._client: Optional[Any] = None
        self._data_buffer: Optional[List[tuple]] = None  # Set by main loop for run_discovery

        if not use_mock and HAS_OPENAI:
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)

    def set_data_buffer(self, data_buffer: Optional[List[tuple]]) -> None:
        """Set (x, y_soft) buffer for run_discovery tool."""
        self._data_buffer = data_buffer

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM (Ollama/OpenAI or mock)."""
        if self.use_mock or self._client is None:
            return self._mock_response(messages)
        for attempt in range(self.retry_times + 1):
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=512,
                )
                return r.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.retry_times:
                    return f"[LLM Error: {e}]"
        return "[LLM Error: max retries]"

    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        """Mock LLM for testing without API."""
        last = messages[-1].get("content", "") if messages else ""
        if "diagnose" in last.lower() or "system" in last.lower():
            diag = self._tool_diagnose_system()
            return f"Thought: Diagnosing system.\nAction: diagnose_system()\nObservation: {diag}\nReply: {diag}"
        if "run_goal" in last or "目标" in last or "goal" in last.lower():
            goal = last.split("run_goal")[-1].split(")")[0].strip("(\"' ")
            if not goal:
                goal = "研究双摆摩擦"
            res = self._tool_run_goal(goal)
            return f"Thought: Executing goal.\nAction: run_goal(\"{goal}\")\nObservation: {res}\nReply: {res}"
        if "suggest" in last.lower() or "next" in last.lower():
            sug = self._tool_suggest_next_action()
            return f"Thought: Suggesting next action.\nAction: suggest_next_action()\nObservation: {sug}\nReply: {sug}"
        if "get_activity" in last.lower() or "activity" in last.lower():
            act = self._tool_get_activity()
            return f"Thought: The user asked about activity. I will check.\nAction: get_activity()\nObservation: {act}\nReply: Soft shell activity is {act:.4f}. High values indicate unmodeled dynamics."
        if "discovery" in last.lower() or "run_discovery" in last.lower():
            res = self._tool_run_discovery()
            return f"Thought: Running discovery on residuals.\nAction: run_discovery()\nObservation: {res}\nReply: {res}"
        if "hippocampus" in last.lower() or "read" in last.lower():
            kb = self._tool_read_hippocampus()
            return f"Thought: Reading knowledge base.\nAction: read_hippocampus()\nObservation: {kb}\nReply: Knowledge base contains {len(kb)} laws."
        return "Thought: I understand. I will analyze the system. Reply: Based on the tools available, I recommend checking soft shell activity and running discovery if it is high."

    def _tool_get_activity(self) -> float:
        if self.rcln is None:
            return 0.0
        return getattr(self.rcln, "get_soft_activity", lambda: 0.0)()

    def _tool_get_uncertainty(self) -> str:
        """Get uncertainty status if RCLN supports return_uncertainty."""
        if self.rcln is None:
            return "No RCLN"
        if hasattr(self.rcln, "get_uncertainty_status"):
            return str(self.rcln.get_uncertainty_status())
        return "Uncertainty not available (RCLN has no get_uncertainty_status)"

    def _tool_diagnose_system(self) -> str:
        """System health: activity, uncertainty, knowledge count."""
        act = self._tool_get_activity()
        unc = self._tool_get_uncertainty()
        kb = self._tool_read_hippocampus()
        n_laws = len(kb)
        status = "healthy" if act < 0.05 else "high_activity"
        return f"Status: {status} | Activity: {act:.4f} | Laws: {n_laws} | {unc}"

    def _tool_run_goal(self, goal_text: str) -> str:
        """目标驱动执行：分解并执行子任务"""
        try:
            from axiom_os.orchestrator.task_decomposer import TaskDecomposer
            from axiom_os.orchestrator.task_executor import TaskExecutor
            decomposer = TaskDecomposer(use_llm=False, chief_scientist=self)
            goal = decomposer.decompose(goal_text)
            executor = TaskExecutor(
                rcln=self.rcln,
                hippocampus=self.hippocampus,
                discovery=self.discovery,
                chief_scientist=self,
                data_buffer=self._data_buffer,
            )
            result = executor.execute_goal(goal)
            summary = "; ".join(f"{k}: {str(v)[:80]}" for k, v in result.get("results", {}).items())
            return f"Goal executed. {summary}"
        except Exception as e:
            return f"Error: {e}"

    def _tool_suggest_next_action(self) -> str:
        """AI-suggested next step based on current state."""
        act = self._tool_get_activity()
        kb = self._tool_read_hippocampus()
        if act > 0.05:
            return "High soft activity detected. Recommend: run_discovery() to extract new physics."
        if len(kb) == 0:
            return "Knowledge base empty. Recommend: run simulation, then run_discovery()."
        return "System stable. Consider: read_hippocampus() to review laws, or run_rar_discovery() for cosmology."

    def _tool_read_hippocampus(self, partition_id: Optional[str] = None) -> Dict[str, Any]:
        if self.hippocampus is None:
            return {}
        if partition_id and hasattr(self.hippocampus, "list_by_partition"):
            laws = self.hippocampus.list_by_partition(partition_id)
            return {l["id"]: {k: v for k, v in l.items() if k != "id"} for l in laws}
        return dict(self.hippocampus.knowledge_base)

    def _tool_write_hippocampus(self, formula: str, formula_id: Optional[str] = None) -> str:
        if self.hippocampus is None or self.rcln is None:
            return "Error: No hippocampus or RCLN"
        for attempt in range(self.retry_times + 1):
            try:
                fid = self.hippocampus.crystallize(formula, self.rcln, formula_id=formula_id)
                return f"Crystallized as {fid}"
            except Exception as e:
                if attempt == self.retry_times:
                    return f"Error: {e}"
        return "Error: max retries"

    def _tool_run_discovery(self) -> str:
        if self.discovery is None or self.rcln is None:
            return "Error: No discovery engine or RCLN"
        for attempt in range(self.retry_times + 1):
            try:
                y_soft = getattr(self.rcln, "_last_y_soft", None)
                if y_soft is None:
                    return "No soft output available. Run forward pass first."
                data_buffer = self._data_buffer or getattr(self.rcln, "_last_data_buffer", None)
                if data_buffer is None or (isinstance(data_buffer, list) and len(data_buffer) < 10):
                    return "No input buffer or too few samples. Need (X, y_soft) for discovery."
                n_in = 4
                if isinstance(data_buffer, list) and data_buffer:
                    n_in = len(np.asarray(data_buffer[0][0]).ravel())
                formula = self.discovery.distill(
                    self.rcln, data_buffer, input_units=[[0, 0, 0, 0, 0]] * n_in
                )
                return f"Found formula: {formula}" if formula else "No formula found"
            except Exception as e:
                if attempt == self.retry_times:
                    return f"Error: {e}"
        return "Error: max retries"

    def _tool_plot_results(self, path: str = "results.png") -> str:
        return f"Plot would be saved to {path} (not implemented in stub)"

    def _tool_run_rar_discovery(self) -> str:
        """Run RAR/Meta-Axis discovery (SPARC galaxies, McGaugh ν(g) fit)."""
        try:
            from axiom_os.experiments.discovery_rar import run_rar_discovery
            out = run_rar_discovery(n_galaxies=100, epochs=200)
            if isinstance(out, dict) and "error" in out:
                return f"RAR discovery error: {out['error']}"
            cry = out.get("crystallized", {}) if isinstance(out, dict) else {}
            g0 = cry.get("g_dagger") or cry.get("a0_or_gdagger")
            a0_si = cry.get("a0_si")
            if g0 is not None:
                from axiom_os.layers.meta_kernel import compute_meta_length
                meta = compute_meta_length(a0_si or 1.2e-10)
                L_gly = meta.get("L_Gly", "?")
                return f"RAR fit: g0={g0:.1f} (km/s)²/kpc, L≈{L_gly:.2f} Gly. Formula: g_obs = g_bar/(1-exp(-sqrt(g_bar/g0)))"
            return str(out)[:500]
        except Exception as e:
            return f"RAR discovery error: {e}"

    def ask(
        self,
        question: str,
        max_steps: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        use_rag: bool = True,
        activity_hint: Optional[float] = None,
        partition_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> str:
        """
        ReAct loop with hard limits.
        use_rag: Inject Hippocampus laws into system prompt.
        activity_hint: If > 0.05, inject "High soft activity - consider run_discovery".
        """
        steps = max_steps if max_steps is not None else self.max_steps
        tool_limit = max_tool_calls if max_tool_calls is not None else self.max_tool_calls

        rag_ctx = ""
        if use_rag and self.hippocampus:
            rag_ctx = _rag_retrieve(
                self.hippocampus, question, top_k=5,
                partition_id=partition_id, domain=domain,
            )
            if rag_ctx:
                rag_ctx = "\n\n" + rag_ctx

        activity_ctx = ""
        act_val = activity_hint if activity_hint is not None else self._tool_get_activity()
        if act_val > 0.05:
            activity_ctx = f"\n\n[AI Hint] Soft activity = {act_val:.4f} (high). Consider run_discovery() to find new physics."

        system = f"""You are the Chief Scientist of Axiom-OS, a physics-AI system.
You reason about RCLN (Residual Coupler Linking Neuron), Hippocampus (knowledge base), and Discovery Engine.
{TOOL_DESCRIPTIONS}
Format: Thought: ... Action: tool_name() or Action: tool_name(arg1, arg2) Observation: ... Reply: ...
If you need to use a tool, say Action: tool_name(). The system will provide Observation.
Then give a final Reply to the user. Use at most {tool_limit} tool calls.{rag_ctx}{activity_ctx}"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        tool_call_count = 0
        last_response = ""

        for step in range(steps):
            response = self._call_llm(messages)
            last_response = response
            messages.append({"role": "assistant", "content": response})

            action, action_args = self._parse_action(response)
            if action:
                if tool_call_count >= tool_limit:
                    messages.append({
                        "role": "user",
                        "content": f"Observation: [Circuit breaker: max tool calls {tool_limit} reached. Stop.]",
                    })
                    break
                obs = self._execute_action(action, action_args)
                tool_call_count += 1
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                if "Reply:" in response:
                    return response.split("Reply:")[-1].strip()
                return response

        if "Reply:" in last_response:
            return last_response.split("Reply:")[-1].strip()
        return last_response

    def _parse_action(self, response: str) -> tuple:
        """Return (action_name, args_dict). args_dict empty if no args."""
        if "Action:" not in response:
            return None, {}
        lines = [l for l in response.split("\n") if "Action:" in l]
        if not lines:
            return None, {}
        raw = lines[0].split("Action:")[-1].strip()
        name = raw.split("(")[0].strip()
        args = {}
        if "(" in raw and ")" in raw:
            inner = raw[raw.index("(") + 1 : raw.rindex(")")].strip()
            if inner:
                parts = re.split(r",\s*(?![^\[\]]*\])", inner)
                if name == "write_hippocampus" and len(parts) >= 1:
                    formula = parts[0].strip().strip('"\'')
                    args["formula"] = formula
                    if len(parts) >= 2:
                        fid = parts[1].strip().strip('"\'')
                        args["formula_id"] = fid
                elif name == "plot_results" and len(parts) >= 1:
                    args["path"] = parts[0].strip().strip('"\'') or "results.png"
                elif name == "run_goal" and len(parts) >= 1:
                    args["goal_text"] = parts[0].strip().strip('"\'')
                elif name == "read_hippocampus_partition" and len(parts) >= 1:
                    args["partition_id"] = parts[0].strip().strip('"\'')
        return name, args

    def _execute_action(self, action: str, args: Optional[Dict[str, Any]] = None) -> str:
        args = args or {}
        if action == "get_activity":
            return str(self._tool_get_activity())
        if action == "get_uncertainty":
            return self._tool_get_uncertainty()
        if action == "diagnose_system":
            return self._tool_diagnose_system()
        if action == "suggest_next_action":
            return self._tool_suggest_next_action()
        if action == "run_goal":
            goal_text = args.get("goal_text", args.get("goal", ""))
            if goal_text:
                return self._tool_run_goal(goal_text)
            return "Use run_goal(goal_text) with a goal string."
        if action == "read_hippocampus":
            return json.dumps(self._tool_read_hippocampus(), default=str)[:500]
        if action == "read_hippocampus_partition":
            pid = args.get("partition_id", "")
            return json.dumps(
                self._tool_read_hippocampus(partition_id=pid or None),
                default=str,
            )[:500]
        if action == "run_discovery":
            return self._tool_run_discovery()
        if action == "run_rar_discovery":
            return self._tool_run_rar_discovery()
        if action == "plot_results":
            return self._tool_plot_results(path=args.get("path", "results.png"))
        if action == "write_hippocampus":
            formula = args.get("formula", "")
            if formula and len(formula) > 2:
                return self._tool_write_hippocampus(formula, args.get("formula_id"))
            return "Use write_hippocampus(formula, formula_id) with explicit formula string."
        return f"Unknown action: {action}"
