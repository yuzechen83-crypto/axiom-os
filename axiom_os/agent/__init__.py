"""
Axiom-Agent: LLM Integration for Text-to-Physics
User inputs natural language -> LLM layer (intent) -> Dispatcher runs Axiom -> LLM formats reply.
"""

try:
    from .coder import AxiomCoder
    from .runner import AxiomRunner
except ImportError:
    AxiomCoder = None
    AxiomRunner = None

try:
    from .llm_layer import AxiomLLMLayer
    from .dispatcher import run_intent
except ImportError:
    AxiomLLMLayer = None
    run_intent = None

__all__ = ["AxiomCoder", "AxiomRunner", "AxiomLLMLayer", "run_intent"]
