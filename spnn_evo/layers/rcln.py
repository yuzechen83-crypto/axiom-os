"""
RCLN - The Reactor (跨学科熔炉)
y = F_hard(x, Hippocampus) + F_soft(x, NeuralNet)

The Spark: Activity Monitor on F_soft
If |F_soft| > threshold consistently → "Discovery Hotspot"
(known physics failed, Neural Net found something new)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, NamedTuple
from collections import deque

from ..core.upi import UPIState, Units
from ..core.hippocampus import HippocampusLibrary


class DiscoveryHotspot(NamedTuple):
    """Flagged when F_soft consistently exceeds threshold"""
    instance_id: str
    avg_soft_magnitude: float
    sample_count: int
    last_input: Optional[torch.Tensor] = None
    last_soft_output: Optional[torch.Tensor] = None


class ActivityMonitor:
    """
    Monitors |F_soft|. If consistently > threshold, flag as Discovery Hotspot.
    """

    def __init__(self, threshold: float = 0.5, window_size: int = 32):
        self.threshold = threshold
        self.window_size = window_size
        self._magnitudes: deque = deque(maxlen=window_size)
        self._last_input: Optional[torch.Tensor] = None
        self._last_soft: Optional[torch.Tensor] = None
        self.instance_id: str = ""

    def update(self, f_soft: torch.Tensor, x: Optional[torch.Tensor] = None) -> Optional[DiscoveryHotspot]:
        mag = f_soft.detach().abs().mean().item()
        self._magnitudes.append(mag)
        self._last_input = x.detach() if x is not None else self._last_input
        self._last_soft = f_soft.detach()

        if len(self._magnitudes) >= self.window_size:
            avg = sum(self._magnitudes) / len(self._magnitudes)
            if avg > self.threshold:
                return DiscoveryHotspot(
                    instance_id=self.instance_id,
                    avg_soft_magnitude=avg,
                    sample_count=len(self._magnitudes),
                    last_input=self._last_input,
                    last_soft_output=self._last_soft,
                )
        return None

    def reset(self) -> None:
        self._magnitudes.clear()
        self._last_input = None
        self._last_soft = None


class FHard(nn.Module):
    """
    F_hard(x, Hippocampus): Known physics from library
    Uses symbolic lookup - simplified as linear projection from library embedding
    """

    def __init__(self, dim: int, library: Optional[HippocampusLibrary] = None):
        super().__init__()
        self.dim = dim
        self.library = library or HippocampusLibrary()
        self.W_hard = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.register_buffer("_lib_size", torch.tensor(len(self.library)))

    def forward(self, x: torch.Tensor, semantics: Optional[str] = None) -> torch.Tensor:
        """Apply hard constraint from known physics"""
        return x @ self.W_hard


class FSoft(nn.Module):
    """
    F_soft(x, NeuralNet): Neural correction / exploration
    """

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RCLNLayer(nn.Module):
    """
    RCLN: y = F_hard(x) + λ·F_soft(x)
    Hard Core: explicit physical equations (e.g., Navier-Stokes, Hamiltonian).
    Soft Shell: SiLU-activated MLP for residuals/unknowns.
    Activity Monitor: High |F_soft| triggers Discovery Engine.
    """

    def __init__(
        self,
        dim: int,
        library: Optional[HippocampusLibrary] = None,
        lambda_soft: float = 1.0,
        soft_threshold: float = 0.5,
        monitor_window: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.lambda_soft = lambda_soft
        self.f_hard = FHard(dim, library)
        self.f_soft = FSoft(dim)
        self.monitor = ActivityMonitor(threshold=soft_threshold, window_size=monitor_window)
        self.monitor.instance_id = id(self)

    def forward(
        self,
        x: torch.Tensor,
        semantics: Optional[str] = None,
        return_hotspot: bool = False,
    ) -> Tuple[torch.Tensor, Optional[DiscoveryHotspot]]:
        y_hard = self.f_hard(x, semantics)
        y_soft = self.f_soft(x)

        # y = F_hard(x) + λ·F_soft(x)
        lam = getattr(self, "lambda_soft", 1.0)
        y = y_hard + lam * y_soft

        hotspot = None
        if return_hotspot:
            hotspot = self.monitor.update(y_soft, x)

        return y, hotspot

    def reset_soft_weights(self, scale: float = 0.01) -> None:
        """
        The "Hunter" has finished the job; crystallize and reset.
        Set F_soft weights to near-zero (knowledge now in Hard).
        """
        for p in self.f_soft.parameters():
            p.data.mul_(scale)

    def reset_monitor(self) -> None:
        self.monitor.reset()


# Backward compatibility
RCLNModule = RCLNLayer
