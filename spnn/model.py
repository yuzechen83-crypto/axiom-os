"""
SPNN-Opt-Rev5 Full Model
六阶段认知学习全链路 + 主脑调度
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from .core.physical_scale import PhysicalScaleSystem
from .core.upi_interface import UPIInterface
from .neurons.rcln import RCLN
from .memory.hippocampus import Hippocampus
from .orchestrator.axiom_os import AxiomOS


class StructEncoder(nn.Module):
    """
    阶段2: 结构蒸馏编码
    e = Encoder_struct(x̃; θ_e)
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MainNeuron(nn.Module):
    """
    阶段3: 主神经元计算
    h_i^{(l+1)} = σ(W h_i^{(l)} + b + Σ_{j∈N(i)} Φ(h_j^{(l)}))
    """

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.W = nn.Linear(dim, hidden_dim)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.phi = nn.Linear(dim, hidden_dim)

    def forward(self, h: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.W(h) + self.b
        if adj is not None and adj.numel() > 0:
            msg = self.phi(h)
            if adj.dim() == 2:
                agg = torch.matmul(adj, msg)
            else:
                agg = msg
            out = out + agg
        return torch.nn.functional.gelu(out)


class AdaptiveSelector(nn.Module):
    """
    阶段5: 自适应筛选
    N_active = {i | w_i > τ_active}
    """

    def __init__(self, dim: int, tau_active: float = 0.1, alpha: float = 0.9):
        super().__init__()
        self.dim = dim
        self.tau_active = tau_active
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(dim) / dim)

    def forward(
        self,
        h: torch.Tensor,
        h_target: Optional[torch.Tensor] = None,
        consistency: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.weights, dim=0)
        if h_target is not None and consistency is not None:
            sim = torch.sum(h * h_target, dim=-1) / (torch.norm(h, dim=-1) * torch.norm(h_target, dim=-1) + 1e-8)
            new_w = self.alpha * w + (1 - self.alpha) * sim.unsqueeze(-1) * consistency.unsqueeze(-1)
            self.weights.data = new_w.mean(dim=0).clamp(1e-6, 1e6)
            w = torch.softmax(self.weights, dim=0)
        mask = (w > self.tau_active).float()
        return h * mask.unsqueeze(0), mask


class SPNN(nn.Module):
    """
    SPNN = ⟨A, N, C, I, M, O, H, B⟩
    完整六阶段认知学习全链路
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_rcln_layers: int = 2,
        memory_capacity: int = 5000,
        tau_active: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Components
        self.scale_system = PhysicalScaleSystem()
        self.upi = UPIInterface(scale_system=self.scale_system)
        self.hippocampus = Hippocampus(dim=hidden_dim, capacity=memory_capacity)
        self.orchestrator = AxiomOS(
            scale_system=self.scale_system,
            upi=self.upi,
            hippocampus=self.hippocampus,
        )

        # Stage 1: 物理锚定 (handled in forward)
        # Stage 2: 结构蒸馏编码
        self.encoder = StructEncoder(in_dim, hidden_dim, hidden_dim)
        # Stage 3: 主神经元
        self.main_neuron = MainNeuron(hidden_dim, hidden_dim)
        # Stage 4: RCLN
        self.rcln_layers = nn.ModuleList([
            RCLN(hidden_dim, lambda_res_init=0.5) for _ in range(num_rcln_layers)
        ])
        # Stage 5: 自适应筛选
        self.selector = AdaptiveSelector(hidden_dim, tau_active=tau_active)
        # Stage 6: 输出 (尺度反推 handled in forward)
        self.head = nn.Linear(hidden_dim, out_dim)

        self.to(self.device)

    def _stage1_normalize(self, x: np.ndarray) -> np.ndarray:
        """阶段1: 物理锚定"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.scale_system.auto_detect_characteristic(x)
        return self.scale_system.normalize(x)

    def _stage2_encode(self, x_tilde: torch.Tensor, l_e: Optional[Any] = None) -> torch.Tensor:
        """阶段2: 结构蒸馏 + 海马体增强"""
        e = self.encoder(x_tilde)
        if l_e is not None and self.hippocampus._memory:
            retrieved, _ = self.hippocampus.retrieve(l_e, top_k=3)
            alpha = 0.3
            retrieved_t = torch.as_tensor(retrieved, dtype=e.dtype, device=e.device)
            if retrieved_t.shape == e.shape:
                e = e + alpha * retrieved_t
            elif retrieved_t.numel() == e.numel():
                e = e + alpha * retrieved_t.view_as(e)
        return e

    def _stage3_main(self, h: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """阶段3: 主神经元"""
        return self.main_neuron(h, adj)

    def _stage4_rcln(self, h_i: torch.Tensor, h_j: torch.Tensor, t: float = 0.0, **kwargs) -> torch.Tensor:
        """阶段4: RCLN 残差耦合"""
        for layer in self.rcln_layers:
            layer.set_lambda_res(t)
            h_coupled = layer(h_i, h_j, **kwargs)
            h_i = h_coupled
            h_j = h_coupled
        return h_i

    def _stage5_select(self, h: torch.Tensor, h_target: Optional[torch.Tensor] = None, l: Any = None) -> torch.Tensor:
        """阶段5: 自适应筛选"""
        consistency = None
        if l is not None:
            consistency = torch.tensor([self.hippocampus.get_consistency(l)], device=h.device).expand(h.shape[0])
        h_selected, _ = self.selector(h, h_target, consistency)
        return h_selected

    def _stage6_denormalize(self, y_tilde: np.ndarray) -> np.ndarray:
        """阶段6: 尺度感知反推"""
        return self.scale_system.denormalize(y_tilde)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        l_e: Optional[Any] = None,
        l_i: Optional[torch.Tensor] = None,
        l_j: Optional[torch.Tensor] = None,
        training_step: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        六阶段前向传播
        """
        # Stage 1: 物理锚定
        x_np = x.detach().cpu().numpy()
        x_tilde_np = self._stage1_normalize(x_np)
        x_tilde = torch.as_tensor(x_tilde_np, dtype=x.dtype, device=x.device)

        # UPI 验证
        if not self.upi.validate_input(x_np):
            pass  # 可记录违规

        # Stage 2: 编码
        e = self._stage2_encode(x_tilde, l_e)
        e = nn.functional.layer_norm(e, e.shape[-1:])

        # Stage 3: 主神经元
        h = self._stage3_main(e, adj)

        # Stage 4: RCLN (自耦合)
        retrieve_fn = None
        if l_i is not None and self.hippocampus._memory:
            retrieve_fn = lambda li: self.hippocampus.retrieve_tensor(li, self.device)
        h = self._stage4_rcln(h, h, t=training_step, retrieve_fn=retrieve_fn, l_i=l_i, l_j=l_j)

        # Stage 5: 筛选
        h = self._stage5_select(h, h_target=h, l=l_e)

        # 输出
        y_tilde = self.head(h)

        aux = {"hidden": h, "encoded": e}
        return y_tilde, aux

    def predict(
        self,
        x: torch.Tensor,
        return_physical: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            y_tilde, _ = self.forward(x, **kwargs)
            if return_physical:
                y_np = self._stage6_denormalize(y_tilde.cpu().numpy())
                return torch.as_tensor(y_np, dtype=y_tilde.dtype, device=y_tilde.device)
            return y_tilde
