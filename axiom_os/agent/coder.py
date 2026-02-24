"""
Axiom-Agent: The Code Generator (agent/coder.py)
LLM wrapper that translates user problems into Axiom-OS modules.
Generates: physics.py, config.yaml, objective.py
"""

from typing import Optional, Dict, Any
import json
import re

SYSTEM_PROMPT = """You are a Physics Engineer using the Axiom-OS framework.
Your job is to translate a user problem into 3 artifacts:

1. **physics.py**: Define the HardCore function and Hamiltonian H using NumPy/PyTorch.
   - HardCore must accept x (torch.Tensor or UPIState with .values) and return next state as torch.Tensor.
   - Input shape: (batch, state_dim*2) where first half is q (positions), second half is p (momenta).
   - Use axiom_os.orchestrator.mpc._step_controlled for controlled dynamics.
   - Export: hard_core_func, H_func, state_dim

2. **config.yaml**: UPI units and MPC hyperparameters.
   - state_dim: int (number of position/momentum pairs, e.g. 2 for double pendulum)
   - input_units: list of [M,L,T,Q,Theta] per state dimension (use [0,0,0,0,0] for unitless angles)
   - horizon_steps: 30-100
   - n_samples: 500-3000
   - dt: 0.01-0.05
   - friction: 0.01-0.3
   - target_state: list of target angles (radians, e.g. [3.14159, 3.14159] for upright)

3. **objective.py**: Cost function for the MPC controller.
   - Define cost(q_traj, p_traj, action_sequence, target_state) -> float
   - Penalize distance from target_state and action magnitude.

STRICT CONSTRAINTS:
- All code must follow UPIState protocol from axiom_os.core.upi (values, units).
- HardCore input: x can be Tensor or object with .values attribute.
- Use only: numpy, torch, math. No exec, import, or file operations.
- Output valid Python/YAML only. No markdown code fences in the final code blocks.
"""

USER_PROMPT_TEMPLATE = """User problem: {problem}

Generate the 3 files. Format your response as:

---physics.py---
<valid Python code>
---config.yaml---
<valid YAML>
---objective.py---
<valid Python code>
"""


def _extract_block(text: str, marker: str) -> Optional[str]:
    """Extract content between ---marker--- and next --- or end."""
    pattern = rf"---{re.escape(marker)}---\s*\n(.*?)(?=\n---|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else None


def _clean_code(block: str) -> str:
    """Remove markdown fences if present."""
    block = block.strip()
    for start in ["```python", "```yaml", "```"]:
        if block.startswith(start):
            block = block[len(start):].strip()
        if block.endswith("```"):
            block = block[:-3].strip()
    return block


class AxiomCoder:
    """
    LLM wrapper for generating Axiom-OS modules.
    Supports OpenAI and Anthropic APIs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        """
        provider: "openai" | "anthropic"
        model: e.g. "gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
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
                raise ImportError("pip install openai")
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key or "")
            except ImportError:
                raise ImportError("pip install anthropic")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(self, problem: str) -> Dict[str, str]:
        """
        Generate physics.py, config.yaml, objective.py from natural language problem.
        Returns: {"physics.py": str, "config.yaml": str, "objective.py": str}
        """
        user_msg = USER_PROMPT_TEMPLATE.format(problem=problem)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content or ""
        elif self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            content = resp.content[0].text if resp.content else ""
        else:
            content = ""

        physics = _extract_block(content, "physics.py")
        config = _extract_block(content, "config.yaml")
        objective = _extract_block(content, "objective.py")

        if physics:
            physics = _clean_code(physics)
        if config:
            config = _clean_code(config)
        if objective:
            objective = _clean_code(objective)

        return {
            "physics.py": physics or "# Failed to generate",
            "config.yaml": config or "state_dim: 2",
            "objective.py": objective or "# Failed to generate",
        }
