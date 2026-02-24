"""
SPNN-Opt-Rev5 (Axiom-OS): Physics-Informed Neural Network Framework
========================================================================
SPNN = ⟨A, N, C, I, M, O, H, B⟩
- A: Physical Anchoring System
- N: Neuron Network (RCLN)
- C: Coupling Coordinator
- I: UPI Interface System
- M: Memory System (Hippocampus)
- O: Main Brain Orchestrator
- H: Hardware Abstraction Layer
- B: Physical Scale System
"""

__version__ = "5.0.0-rev5"

from .core.physical_scale import PhysicalScaleSystem, UniversalConstants
from .core.upi_interface import UPIInterface
from .neurons.rcln import RCLN
from .memory.hippocampus import Hippocampus
from .orchestrator.axiom_os import AxiomOS
from .model import SPNN
from .physical_ai import PhysicalAI, PhysicalAIResult
from .complete_system import SPNNCompleteSystem

__all__ = [
    "SPNN",
    "PhysicalScaleSystem",
    "UniversalConstants",
    "UPIInterface",
    "RCLN",
    "Hippocampus",
    "AxiomOS",
    "PhysicalAI",
    "PhysicalAIResult",
    "SPNNCompleteSystem",
]
