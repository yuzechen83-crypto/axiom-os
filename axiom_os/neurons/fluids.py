"""
Fluids Domain Neurons
Turbulence, vorticity, Navier-Stokes residuals.
"""

from typing import List, Optional, Union, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseNeuron


class TurbulenceResidualNeuron(BaseNeuron, nn.Module):
    """
    Fluids: Turbulence residual (u, v, T) -> acceleration residual.
    Input units: u,v [L/T], T [Θ]
    Output: acceleration residual [L/T²]
    """

    DOMAIN = "fluids"
    INPUT_UNITS = [
        [0, 1, -1, 0, 0],  # u velocity
        [0, 1, -1, 0, 0],  # v velocity
        [0, 0, 0, 0, 1],  # T temperature
    ]
    OUTPUT_UNITS = [[0, 1, -2, 0, 0]]  # acceleration
    INPUT_SEMANTICS = ["u", "v", "T"]
    OUTPUT_SEMANTICS = "a_res"

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, output_dim: int = 1, neuron_id: str = ""):
        BaseNeuron.__init__(self, neuron_id)
        nn.Module.__init__(self)
        if not HAS_TORCH:
            raise ImportError("PyTorch required for TurbulenceResidualNeuron")
        self.soft = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._last_soft: Optional[torch.Tensor] = None
        self._last_x: Optional[torch.Tensor] = None

    def forward(self, x: Union["torch.Tensor", Any]) -> "torch.Tensor":
        from axiom_os.core.upi import UPIState
        if isinstance(x, UPIState):
            v = x.values.float()
        else:
            v = torch.as_tensor(x, dtype=torch.float32)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        y = self.soft(v)
        self._last_soft = y.detach()
        self._last_x = v.detach()
        return y

    def get_soft_activity(self) -> float:
        if self._last_soft is None:
            return 0.0
        return float(self._last_soft.abs().mean().item())

    def get_contribution_buffer(self) -> Optional[tuple]:
        if self._last_x is None or self._last_soft is None:
            return None
        return (self._last_x.cpu().numpy(), self._last_soft.cpu().numpy())
