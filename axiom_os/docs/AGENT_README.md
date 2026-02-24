# Axiom-Agent: LLM Integration

**Text-to-Physics**: User inputs natural language → LLM generates Axiom Config/Code → Axiom Solves → LLM interprets result.

## Quick Start

```bash
# With OpenAI API key
export OPENAI_API_KEY=sk-...
python run_agent.py "Simulate a double pendulum with air resistance"

# Dry-run (no API, uses built-in template)
python run_agent.py "Triple pendulum" --dry-run --max-steps 50

# CLI chat
python -m axiom_os.agent.chat
```

## Architecture

| Component | Path | Role |
|-----------|------|------|
| **Coder** | `agent/coder.py` | LLM wrapper → generates physics.py, config.yaml, objective.py |
| **Runner** | `agent/runner.py` | Loads generated code, runs Axiom loop, captures stdout |
| **Pipeline** | `run_agent.py` | End-to-end: problem → generate → run → report |
| **Chat** | `agent/chat.py` | CLI conversation loop |

## Generated Artifacts

1. **physics.py**: `hard_core_func`, `H_func`, `state_dim` — must follow UPIState protocol
2. **config.yaml**: `state_dim`, `horizon_steps`, `n_samples`, `dt`, `friction`, `target_state`
3. **objective.py**: `cost(q_traj, p_traj, action_sequence, target_state)` for MPC

## API Support

- **OpenAI**: `OPENAI_API_KEY`, `--provider openai`, `--model gpt-4o-mini`
- **Anthropic**: `ANTHROPIC_API_KEY`, `--provider anthropic`, `--model claude-3-5-sonnet-20241022`

## Safety

- Generated code is written to `agent_output/` and loaded via `importlib`
- No `exec()` of arbitrary strings; modules are loaded from disk
- Config uses YAML parsing (or simple line parser fallback)
