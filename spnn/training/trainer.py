"""
SPNN Intelligent Trainer
三阶段自适应调度 + 震荡检测 + 主脑控制
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import math

from .losses import SPNNLoss
from ..model import SPNN
from ..orchestrator.axiom_os import AxiomOS
from ..core.calibration import detect_oscillation, OscillationPattern


@dataclass
class TrainingConfig:
    phases: int = 3
    epochs_per_phase: int = 100
    batch_size: int = 32
    tau_osc: float = 10.0
    osc_window: int = 10


class SPNNTrainer:
    """
    智能训练协议
    - 三阶段: 物理定型 → 柔性演化 → 精细调优
    - 震荡检测与主脑响应
    - 梯度安全 (G_max 由主脑调度)
    """

    def __init__(
        self,
        model: SPNN,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[SPNNLoss] = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.loss_fn = loss_fn or SPNNLoss(alpha=0.3, beta=0.4, gamma=0.1, delta=0.1, epsilon=0.1)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self._loss_history: list = []
        self._phase = 0
        self._global_step = 0
        self._osc_buffer: list = []
        self.G_max = 1.0

    def _get_phase_params(self, phase: int) -> Dict[str, float]:
        """三阶段自适应调度"""
        schedules = [
            {"alpha": 0.7, "lambda_res": 0.8, "eta": 1e-3},
            {"alpha": 0.5, "lambda_res": 0.3, "eta": 5e-4},
            {"alpha": 0.5, "lambda_res": 0.1, "eta": 1e-4},
        ]
        return schedules[min(phase, 2)]

    def _detect_oscillation(self) -> bool:
        """震荡检测 (自相关 + 符号切换增强)"""
        res = detect_oscillation(
            self._loss_history,
            tau=self.config.osc_window,
            theta_osc=self.config.tau_osc,
        )
        if res.pattern == OscillationPattern.DIVERGING:
            return True
        return res.is_oscillation

    def _clip_gradients(self, params, max_norm: Optional[float] = None):
        """梯度安全机制: 主脑监控 G_max"""
        max_norm = max_norm or self.G_max
        torch.nn.utils.clip_grad_norm_(params, max_norm)

    def setup_optimizer(self):
        params = self._get_phase_params(self._phase)
        lr = params["eta"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs_per_phase * 10,
        )

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        l_e: Optional[Any] = None,
        Q_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        pred, aux = self.model.forward(
            x, adj=adj, l_e=l_e,
            training_step=float(self._global_step) / 1000.0,
        )

        # 主脑调度: 更新 λ_res
        params = self._get_phase_params(self._phase)
        for layer in self.model.rcln_layers:
            layer.set_lambda_res(self._global_step / 1000.0)

        losses = self.loss_fn(
            pred=pred,
            target=y,
            Q_scale=Q_scale,
            hidden=aux.get("hidden"),
        )

        losses["total"].backward()
        self._clip_gradients(self.model.parameters())
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self._loss_history.append(losses["total"].item())
        self._osc_buffer.append(losses["total"].item())
        if len(self._osc_buffer) > self.config.osc_window * 2:
            self._osc_buffer = self._osc_buffer[-self.config.osc_window:]

        self._global_step += 1

        # 震荡响应
        if self._detect_oscillation() and isinstance(self.model.orchestrator, AxiomOS):
            self.G_max *= 0.9
            self.G_max = max(0.1, self.G_max)

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def train_epoch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        bs = batch_size or self.config.batch_size
        n = x.shape[0]
        agg = {}
        for start in range(0, n, bs):
            end = min(start + bs, n)
            bx, by = x[start:end], y[start:end]
            badj = adj[start:end] if adj is not None else None
            step_loss = self.train_step(bx, by, adj=badj)
            for k, v in step_loss.items():
                agg[k] = agg.get(k, 0) + v
        for k in agg:
            agg[k] /= max(1, (n + bs - 1) // bs)
        return agg

    def advance_phase(self) -> None:
        self._phase = min(self._phase + 1, self.config.phases - 1)
        self.setup_optimizer()
