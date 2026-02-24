"""
BaseNeuron - Abstract base class for physical domain neurons.
Designers inherit, implement forward(), declare input/output units.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# UPIState can be from core.upi
def _get_values(x) -> "torch.Tensor":
    if hasattr(x, "values"):
        return x.values.float() if HAS_TORCH else np.asarray(x.values)
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32) if HAS_TORCH else np.asarray(x)


def _make_upi_state(values, units: List[int], semantics: str = ""):
    """Lazy import to avoid circular deps."""
    from axiom_os.core.upi import UPIState
    return UPIState(values=values, units=units, semantics=semantics)


class BaseNeuron(ABC):
    """
    Ecological neuron base class.
    Organize by physical domain: mechanics, fluids, electromagnetism, thermodynamics, control.
    """

    # Subclasses must define
    DOMAIN: str = "generic"
    INPUT_UNITS: List[List[int]] = [[0, 0, 0, 0, 0]]  # [M,L,T,Q,Θ] per input dim
    OUTPUT_UNITS: List[List[int]] = [[0, 0, 0, 0, 0]]
    INPUT_SEMANTICS: List[str] = []
    OUTPUT_SEMANTICS: str = ""

    def __init__(self, neuron_id: str = ""):
        self.neuron_id = neuron_id or self.__class__.__name__

    @abstractmethod
    def forward(self, x: Union["UPIState", "torch.Tensor"]) -> Union["UPIState", "torch.Tensor"]:
        """Input -> Output. Must preserve or declare units."""
        pass

    def get_soft_activity(self) -> float:
        """Default 0; override if neuron has soft/residual part."""
        return 0.0

    def get_contribution_buffer(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Optional: contribute (X, y_soft) to Discovery.
        Return None to skip discovery.
        """
        return None

    def __call__(self, x):
        return self.forward(x)
