#!/usr/bin/env python
"""
Axiom-Agent: Text-to-Physics Pipeline
User inputs natural language -> LLM generates Axiom Config/Code -> Axiom Solves -> Result.
Usage:
  python run_agent.py "Simulate a double pendulum with friction"
  python run_agent.py "Triple pendulum with air resistance"
  OPENAI_API_KEY=sk-... python run_agent.py "Your problem"
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure axiom_os is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Axiom-Agent: Text-to-Physics")
    parser.add_argument("problem", nargs="?", default="", help="Natural language physics problem")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max-steps", type=int, default=80, help="Simulation steps")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory for artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Use mock generator (no API)")
    args = parser.parse_args()

    problem = args.problem or "Simulate a double pendulum (acrobot) with friction balancing upright."
    output_dir = args.output or ROOT / "agent_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Axiom-Agent: Text-to-Physics Pipeline")
    print("=" * 60)
    print(f"Problem: {problem}")
    print()

    # Step 1: Generate code
    if args.dry_run:
        print("[Dry-run] Using built-in double pendulum template")
        artifacts = _mock_generate(problem)
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: No API key. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
            print("Falling back to mock generator.")
            artifacts = _mock_generate(problem)
        else:
            from axiom_os.agent.coder import AxiomCoder
            coder = AxiomCoder(api_key=api_key, provider=args.provider, model=args.model)
            artifacts = coder.generate(problem)

    # Write artifacts
    for name, content in artifacts.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        print(f"  Wrote {path}")

    # Step 2: Run
    print("\nRunning Axiom simulation...")
    from axiom_os.agent.runner import AxiomRunner
    runner = AxiomRunner(output_dir=output_dir)
    stdout, result, err = runner.run(
        artifacts["physics.py"],
        artifacts["config.yaml"],
        artifacts["objective.py"],
        max_steps=args.max_steps,
    )

    if err:
        print(f"Error: {err}")
        if stdout:
            print("Stdout:", stdout[:500])
        return 1

    # Step 3: Report
    print("\n" + "-" * 40)
    print("Results:")
    if result.get("discovered"):
        print("  Discovered formulas:", result["discovered"])
    if result.get("trajectory"):
        q = result["trajectory"]["q"]
        print(f"  Trajectory: {len(q)} steps")
    if stdout:
        print("  Log:", stdout[:300] + ("..." if len(stdout) > 300 else ""))
    print("=" * 60)
    return 0


def _mock_generate(problem: str) -> dict:
    """Fallback when no API: use double pendulum template."""
    physics = '''"""Generated physics - double pendulum with friction."""
import numpy as np
import torch

def double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0):
    def H(qp):
        qp = np.atleast_1d(np.asarray(qp, dtype=np.float64))
        if len(qp) < 4:
            return 0.0
        q1, q2, p1, p2 = float(qp[0]), float(qp[1]), float(qp[2]), float(qp[3])
        dq = q1 - q2
        denom = 2.0 * (1.0 + np.sin(dq) ** 2)
        if abs(denom) < 1e-12:
            denom = 1e-12
        T = (p1**2 + 2*p2**2 - 2*p1*p2*np.cos(dq)) / denom
        V = g_over_L * (L1 * (1 - np.cos(q1)) + L2 * (1 - np.cos(q2)))
        return T + V
    return H

H_func = double_pendulum_H(g_over_L=10.0, L1=1.0, L2=1.0)
state_dim = 2

def hard_core_func(x):
    from axiom_os.orchestrator.mpc import _step_controlled
    if isinstance(x, torch.Tensor):
        vals = x
    elif hasattr(x, "values"):
        vals = x.values
    else:
        vals = x
    vals = torch.as_tensor(vals, dtype=torch.float32)
    if vals.dim() == 1:
        vals = vals.unsqueeze(0)
    v = vals.cpu().numpy()
    q, p = v[:, :2], v[:, 2:4]
    out_list = []
    for i in range(v.shape[0]):
        qn, pn = _step_controlled(q[i], p[i], 0.0, H_func, 2, 0.02, 0.1)
        out_list.append(np.concatenate([qn, pn]))
    return torch.from_numpy(np.stack(out_list).astype(np.float32))
'''
    config = '''state_dim: 2
horizon_steps: 50
n_samples: 500
dt: 0.02
friction: 0.1
target_state: [3.14159, 3.14159]
'''
    objective = '''"""Cost function for MPC."""
import numpy as np

def cost(q_traj, p_traj, action_sequence, target_state):
    err = np.array(q_traj) - np.array(target_state)
    err = (err + np.pi) % (2 * np.pi) - np.pi
    pos_cost = np.sum(err ** 2)
    vel_cost = np.sum(np.array(p_traj) ** 2)
    act_cost = 0.0001 * np.sum(np.array(action_sequence) ** 2)
    return pos_cost + 0.1 * vel_cost + act_cost
'''
    return {"physics.py": physics, "config.yaml": config, "objective.py": objective}


if __name__ == "__main__":
    sys.exit(main())
