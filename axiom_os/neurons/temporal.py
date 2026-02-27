"""
Temporal Physics Neuron - PINN-LSTM for spatiotemporal prediction.
Wraps PhysicsInformedLSTM for use in NeuronRegistry / MLL.
"""

from typing import Optional, Callable, Union, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseNeuron


class TemporalPhysicsNeuron(BaseNeuron, nn.Module):
    """
    时空物理神经元：Hard Core + LSTM 残差，用于湍流、气象等时序预测。
    输入: (B, seq_len, 4) = (t, x, y, z) 序列
    输出: (B, 2) = (u, v) 风速
    """

    DOMAIN = "fluids"
    INPUT_UNITS = [
        [0, 0, 0, 0, 0],  # t
        [0, 1, 0, 0, 0],  # x
        [0, 1, 0, 0, 0],  # y
        [0, 1, 0, 0, 0],  # z
    ]
    OUTPUT_UNITS = [[0, 1, -1, 0, 0], [0, 1, -1, 0, 0]]  # u, v
    INPUT_SEMANTICS = ["t", "x", "y", "z"]
    OUTPUT_SEMANTICS = "u,v"

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 2,
        seq_len: int = 8,
        hard_core_func: Optional[Callable] = None,
        lambda_res: float = 1.0,
        num_layers: int = 2,
        neuron_id: str = "",
    ):
        BaseNeuron.__init__(self, neuron_id)
        nn.Module.__init__(self)
        if not HAS_TORCH:
            raise ImportError("PyTorch required for TemporalPhysicsNeuron")
        from axiom_os.layers.pinn_lstm import PhysicsInformedLSTM
        self.core = PhysicsInformedLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            hard_core_func=hard_core_func,
            lambda_res=lambda_res,
            num_layers=num_layers,
        )
        self._last_soft: Optional[torch.Tensor] = None
        self._last_x: Optional[torch.Tensor] = None

    def forward(self, x: Union["torch.Tensor", Any]) -> "torch.Tensor":
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        y = self.core(x)
        self._last_soft = y.detach()
        self._last_x = x.detach()
        return y

    def set_lambda_decay(self, epoch: int, total_epochs: int, decay_min: float = 0.5) -> None:
        self.core.set_lambda_decay(epoch, total_epochs, decay_min=decay_min)

    def get_soft_activity(self) -> float:
        if self._last_soft is None:
            return 0.0
        return float(self._last_soft.abs().mean().item())

    def get_contribution_buffer(self) -> Optional[tuple]:
        if self._last_x is None or self._last_soft is None:
            return None
        return (self._last_x.cpu().numpy(), self._last_soft.cpu().numpy())
