"""
Axiom-Agent: Conversation Loop (agent/chat.py)
CLI chat interface: User -> LLM generates -> Axiom runs -> Agent interprets result.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _mock_generate_fallback(problem: str) -> dict:
    """Fallback when run_agent not importable: minimal double pendulum template."""
    return {
        "physics.py": "# Mock template - set OPENAI_API_KEY for LLM",
        "config.yaml": "state_dim: 2\nhorizon_steps: 50",
        "objective.py": "# Mock",
    }


def chat_loop(use_streamlit: bool = False):
    """Run chat loop. use_streamlit=True for Streamlit UI."""
    if use_streamlit:
        _run_streamlit()
        return

    print("Axiom-Agent Chat (CLI)")
    print("Enter a physics problem. 'quit' to exit.\n")

    from axiom_os.agent.coder import AxiomCoder
    from axiom_os.agent.runner import AxiomRunner

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY for LLM generation. Using mock for now.")
        coder = None
    else:
        coder = AxiomCoder(api_key=api_key, provider="openai")

    runner = AxiomRunner()
    output_dir = ROOT / "agent_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            problem = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not problem or problem.lower() in ("quit", "exit", "q"):
            break

        # Generate
        if coder:
            artifacts = coder.generate(problem)
        else:
            try:
                from run_agent import _mock_generate
            except ImportError:
                from axiom_os.agent.chat import _mock_generate_fallback
                _mock_generate = _mock_generate_fallback
            artifacts = _mock_generate(problem)

        # Run
        stdout, result, err = runner.run(
            artifacts["physics.py"],
            artifacts["config.yaml"],
            artifacts["objective.py"],
            max_steps=50,
        )

        # Reply
        if err:
            print(f"Agent: Error - {err}\n")
        else:
            discovered = result.get("discovered", [])
            if discovered:
                print(f"Agent: Simulation complete. I detected: {discovered[0]}\n")
            else:
                print("Agent: Simulation complete. No new physics discovered.\n")

    print("Goodbye.")


def _run_streamlit():
    """Launch Streamlit chat UI."""
    import subprocess
    script = Path(__file__).parent / "chat_ui.py"
    if script.exists():
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(script)], cwd=ROOT)
    else:
        print("chat_ui.py not found. Use CLI: python -m axiom_os.agent.chat")


if __name__ == "__main__":
    chat_loop(use_streamlit="--ui" in sys.argv)
