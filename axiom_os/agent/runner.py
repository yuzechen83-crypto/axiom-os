"""
Axiom-Agent: The Execution Sandbox (agent/runner.py)
Loads LLM-generated code, runs Axiom loop, captures stdout and feedback.
"""

from typing import Optional, Dict, Any, Tuple
import os
import sys
import tempfile
import io
import contextlib
from pathlib import Path

import numpy as np

# Project root for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_in_sandbox(
    physics_code: str,
    config_dict: Dict[str, Any],
    objective_code: str,
    output_dir: Optional[Path] = None,
    max_steps: int = 100,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Execute generated code in isolated context.
    Returns: (stdout_capture, result_dict, error_message)
    """
    stdout_capture = io.StringIO()
    result = {"discovered": [], "trajectory": None, "plots": [], "log": []}

    with tempfile.TemporaryDirectory(prefix="axiom_agent_") as tmpdir:
        tmp = Path(tmpdir)
        physics_path = tmp / "physics.py"
        config_path = tmp / "config.py"  # Load as Python for safety
        objective_path = tmp / "objective.py"

        physics_path.write_text(physics_code, encoding="utf-8")
        objective_path.write_text(objective_code, encoding="utf-8")

        # config.yaml -> config dict for Python
        state_dim = config_dict.get("state_dim", 2)
        horizon_steps = min(config_dict.get("horizon_steps", 50), 30)
        n_samples = min(config_dict.get("n_samples", 500), 100)
        dt = config_dict.get("dt", 0.02)
        friction = config_dict.get("friction", 0.1)
        target_state = np.array(config_dict.get("target_state", [np.pi, np.pi]))

        try:
            with contextlib.redirect_stdout(stdout_capture):
                # Import generated physics
                import importlib.util
                spec = importlib.util.spec_from_file_location("physics", physics_path)
                physics_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(physics_mod)

                hard_core_func = getattr(physics_mod, "hard_core_func", None)
                H_func = getattr(physics_mod, "H_func", None)
                gen_state_dim = getattr(physics_mod, "state_dim", state_dim)
                state_dim = gen_state_dim

                if hard_core_func is None or H_func is None:
                    return stdout_capture.getvalue(), result, "physics.py must export hard_core_func and H_func"

                import torch
                from axiom_os.core import UPIState, Hippocampus
                from axiom_os.layers import RCLNLayer
                from axiom_os.engine import DiscoveryEngine
                from axiom_os.orchestrator import ImaginationMPC
                from axiom_os.orchestrator.mpc import step_env

                PI = np.pi
                rcln = RCLNLayer(
                    input_dim=state_dim * 2,
                    hidden_dim=64,
                    output_dim=state_dim * 2,
                    hard_core_func=hard_core_func,
                    lambda_res=0.5,
                )
                mpc = ImaginationMPC(
                    H=H_func,
                    horizon_steps=horizon_steps,
                    n_samples=n_samples,
                    dt=dt,
                    friction=friction,
                    state_dim=state_dim,
                    target_state=target_state,
                )
                hippocampus = Hippocampus()
                discovery = DiscoveryEngine(use_pysr=False)
                optimizer = torch.optim.Adam(rcln.soft_shell.parameters(), lr=1e-3)
                data_buffer = []
                buffer_cap = 200

                q = np.array(target_state) - 0.2
                p = np.array([0.1] * state_dim)
                log_q, log_p = [q.copy()], [p.copy()]

                for t in range(max_steps):
                    action = mpc.plan(q, p)
                    q_next, p_next = step_env(q, p, action, H_func, dt=dt, friction=friction, state_dim=state_dim)
                    obs = np.concatenate([q, p]).astype(np.float32)
                    target = np.concatenate([q_next, p_next]).astype(np.float32)
                    x_t = torch.from_numpy(obs).float().unsqueeze(0)
                    y_t = torch.from_numpy(target).float().unsqueeze(0)
                    y_pred = rcln(x_t)
                    loss = torch.nn.functional.mse_loss(y_pred, y_t)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    y_soft = rcln._last_y_soft.cpu().numpy() if rcln._last_y_soft is not None else np.zeros(state_dim * 2)
                    data_buffer.append((obs.copy(), y_soft.copy()))
                    if len(data_buffer) > buffer_cap:
                        data_buffer.pop(0)

                    if (t + 1) % 50 == 0 and rcln.get_soft_activity() > 0.05 and len(data_buffer) >= 30:
                        buf = data_buffer[-50:]
                        formula = discovery.distill(rcln, buf, input_units=[[0, 0, 0, 0, 0]] * (state_dim * 2))
                        if formula and len(str(formula)) > 2:
                            X = np.stack([np.asarray(p[0]).ravel() for p in buf])
                            y = np.stack([np.asarray(p[1]).ravel() for p in buf])
                            valid, _ = discovery.validate_formula(formula, X, y, output_dim=state_dim * 2)
                            if valid:
                                result["discovered"].append(str(formula))
                                hippocampus.crystallize(formula, rcln, formula_id=f"law_{t}")

                    q, p = q_next, p_next
                    log_q.append(q.copy())
                    log_p.append(p.copy())

                result["trajectory"] = {"q": log_q, "p": log_p}
                result["log"] = result["discovered"]

                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    traj_path = output_dir / "trajectory.npz"
                    np.savez(traj_path, q=np.array(log_q), p=np.array(log_p))

        except Exception as e:
            return stdout_capture.getvalue(), result, str(e)

    return stdout_capture.getvalue(), result, None


class AxiomRunner:
    """
    Loads and runs LLM-generated Axiom modules.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else None

    def run(
        self,
        physics_code: str,
        config_yaml: str,
        objective_code: str,
        max_steps: int = 100,
    ) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """
        Execute generated code.
        config_yaml: YAML string, will be parsed for state_dim, horizon_steps, etc.
        Returns: (stdout, result_dict, error)
        """
        config_dict = self._parse_config(config_yaml)
        out_dir = self.output_dir
        return _run_in_sandbox(
            physics_code,
            config_dict,
            objective_code,
            output_dir=out_dir,
            max_steps=max_steps,
        )

    def _parse_config(self, yaml_str: str) -> Dict[str, Any]:
        """Parse config YAML into dict."""
        try:
            import yaml
            return yaml.safe_load(yaml_str) or {}
        except ImportError:
            pass
        config = {}
        for line in yaml_str.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k == "state_dim":
                    config[k] = int(v)
                elif k in ("horizon_steps", "n_samples"):
                    config[k] = int(v)
                elif k in ("dt", "friction"):
                    config[k] = float(v)
                elif k == "target_state":
                    try:
                        config[k] = [float(x.strip()) for x in v.strip("[]").split(",")]
                    except ValueError:
                        config[k] = [3.14159, 3.14159]
        return config
