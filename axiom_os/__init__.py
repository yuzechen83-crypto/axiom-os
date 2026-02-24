"""
Axiom-OS - Physics-AI Hybrid Operating System
"""

__version__ = "0.1.0"

from .neurons import (
    BaseNeuron,
    NeuronRegistry,
    get_registry,
    AcrobotResidualNeuron,
    TurbulenceResidualNeuron,
    ControlResidualNeuron,
)

__all__ = [
    "__version__",
    "BaseNeuron",
    "NeuronRegistry",
    "get_registry",
    "AcrobotResidualNeuron",
    "TurbulenceResidualNeuron",
    "ControlResidualNeuron",
]
