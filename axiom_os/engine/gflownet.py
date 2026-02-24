"""
GFlowNet for Equation Discovery - Bengio et al.
State: Partial equation tree (e.g., "F = m × ?")
Action: Add operator (+, -, *, sin, ∂x) or variable (x0, x1, ...)
Reward: R = exp(-MSE). Policy constructs equations proportional to reward.
"""

from typing import List, Optional, Tuple, Set
import numpy as np
import torch
import torch.nn as nn

# Operators and variables (simple infix)
OPERATORS = ["+", "-", "*", "/"]
VARIABLES = ["x0", "x1", "x2", "x3", "x4"]
TERMINAL = ["END"]


class EquationState:
    """Partial equation as a sequence of tokens."""

    def __init__(self, tokens: Optional[List[str]] = None):
        self.tokens = tokens or ["?"]

    def copy(self) -> "EquationState":
        return EquationState(self.tokens.copy())

    def add(self, token: str) -> "EquationState":
        s = self.copy()
        idx = s.tokens.index("?") if "?" in s.tokens else len(s.tokens)
        s.tokens[idx] = token
        if token in OPERATORS:
            s.tokens.append("?")
        return s

    def is_terminal(self) -> bool:
        return "?" not in self.tokens

    def to_formula(self) -> str:
        return " ".join(t for t in self.tokens if t != "?")

    def __hash__(self) -> int:
        return hash(tuple(self.tokens))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EquationState) and self.tokens == other.tokens


def eval_formula(formula: str, X: np.ndarray) -> np.ndarray:
    """Safely evaluate formula on data. Returns predictions or zeros on error."""
    if not formula or formula.strip() == "":
        return np.zeros(X.shape[0])
    safe = {"np": np, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt, "exp": np.exp, "log": np.log}
    for i in range(min(10, X.shape[1])):
        safe[f"x{i}"] = X[:, i]
    try:
        return np.asarray(eval(formula, {"__builtins__": {}}, safe), dtype=np.float64).ravel()
    except Exception:
        return np.zeros(X.shape[0])


def formula_from_tokens(tokens: List[str]) -> str:
    """Convert token list to evaluable formula. Simple: x0 + x1 * x2."""
    s = " ".join(t for t in tokens if t not in ("?", "END"))
    return s.strip() or "0"


class GFlowNetPolicy(nn.Module):
    """Policy network: state -> logits over actions."""

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 20,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.embed = nn.Embedding(100, 16)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.mlp(state_vec)


class GFlowNetDiscovery:
    """
    GFlowNet for symbolic equation discovery.
    Reward R = exp(-MSE). Samples equations proportionally to reward.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_actions: int = 20,
        temperature: float = 1.0,
    ):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.n_samples = len(self.y)
        self.n_vars = self.X.shape[1]
        self.temperature = temperature

        self.actions = VARIABLES[: self.n_vars] + OPERATORS + ["END"]
        self.n_actions = len(self.actions)
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}

        self.policy = GFlowNetPolicy(
            state_dim=64,
            n_actions=self.n_actions,
        )

    def state_to_vec(self, state: EquationState) -> np.ndarray:
        """Encode state as fixed-size vector."""
        vec = np.zeros(64)
        for i, t in enumerate(state.tokens[:32]):
            idx = self.action_to_idx.get(t, 0)
            vec[i * 2] = idx / max(1, self.n_actions)
            vec[i * 2 + 1] = 1.0 if t == "?" else 0.0
        return vec.astype(np.float32)

    def reward(self, state: EquationState) -> float:
        """R = exp(-MSE)."""
        if not state.is_terminal():
            return 0.0
        formula = formula_from_tokens(state.tokens)
        pred = eval_formula(formula, self.X)
        mse = np.mean((pred - self.y) ** 2) + 1e-10
        return float(np.exp(-mse))

    def get_actions(self, state: EquationState) -> List[str]:
        """Valid actions from current state."""
        if state.is_terminal():
            return []
        actions = []
        if "?" in state.tokens:
            actions = self.actions.copy()
        return actions

    def sample_equation(self, max_steps: int = 20) -> Tuple[EquationState, float]:
        """Sample one equation from the policy (or uniform for now)."""
        state = EquationState()
        for _ in range(max_steps):
            if state.is_terminal():
                break
            actions = self.get_actions(state)
            if not actions:
                break
            # Uniform sampling (policy would use learned logits)
            idx = np.random.randint(len(actions))
            state = state.add(actions[idx])
        r = self.reward(state)
        return state, r

    def discover(
        self,
        n_samples: int = 50,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Sample equations, return top-k by reward.
        """
        results: List[Tuple[str, float]] = []
        for _ in range(n_samples):
            state, r = self.sample_equation()
            if state.is_terminal() and r > 0:
                formula = formula_from_tokens(state.tokens)
                results.append((formula, r))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
