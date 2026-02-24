"""
Axiom-OS Neurons - Physical Domain Neurons
BaseNeuron + NeuronRegistry for pluggable physics-AI modules.
"""

from .base import BaseNeuron
from .registry import NeuronRegistry, get_registry
from .mechanics import AcrobotResidualNeuron
from .fluids import TurbulenceResidualNeuron
from .control import ControlResidualNeuron

__all__ = [
    "get_registry",
    "BaseNeuron",
    "NeuronRegistry",
    "AcrobotResidualNeuron",
    "TurbulenceResidualNeuron",
    "ControlResidualNeuron",
]
