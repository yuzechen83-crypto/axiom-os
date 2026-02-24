"""
SPNN-CompleteSystem 完整系统
整合原始SPNN报告所有组件的完整架构
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from .semantics import PhysicalLabelingSystem, StructureDistillationEncoder, AdaptiveScaleEncoder
from .memory.structured_hippocampus import StructuredHippocampus
from .ionization import IonizationDataModule
from .wrappers import DASWithDiagnostic
from .model import SPNN


class SPNNCompleteSystem(nn.Module):
    """
    整合所有原始SPNN组件的完整系统
    - 物理标签系统
    - 结构蒸馏编码器
    - 感知海马体
    - 自适应尺度编码
    - 离子化模块
    - DAS + RCLN
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 64,
        out_dim: int = 1,
        memory_capacity: int = 5000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.physical_labels = PhysicalLabelingSystem(embed_dim=32)
        self.structure_encoder = StructureDistillationEncoder(in_dim, hidden_dim)
        self.hippocampus = StructuredHippocampus(capacity=memory_capacity)
        self.scale_encoder = AdaptiveScaleEncoder(num_scales=8)
        self.ionization = IonizationDataModule()
        self.das = DASWithDiagnostic(diagnostic_threshold=0.8)

        self.base_model = SPNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            memory_capacity=memory_capacity,
            device=self.device,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[Dict] = None,
    ) -> torch.Tensor:
        context = context or {}
        labels = context.get("physical_labels", ["unknown"])

        # 1. 结构编码
        structure_info = self.structure_encoder(x, context)
        context["structure"] = structure_info

        # 2. 尺度编码
        scale_weights = self.scale_encoder.encode(x, context)
        context["scale"] = scale_weights

        # 3. 记忆检索
        memory_pattern = self.hippocampus.retrieve_by_physical_label(
            labels, device=x.device
        )
        if memory_pattern is not None:
            context["memory_pattern"] = memory_pattern

        # 4. 基础模型前向
        pred, aux = self.base_model(x, l_e=labels[0] if labels else None)

        # 5. 若为离子化数据，预处理
        if "ionization_type" in context:
            ion_features = self.ionization.preprocess_ionization_data(
                x, context["ionization_type"]
            )
            context["ionization_features"] = ion_features

        # 6. DAS 防护 (soft=pred, hard=物理约束投影)
        h_soft = pred
        h_hard = self.base_model.head(aux.get("hidden", self.base_model.encoder(x)))
        if h_hard.shape != h_soft.shape:
            h_hard = h_hard[:, :h_soft.shape[-1]]
        shielded = self.das(h_soft, h_hard, context)

        # 7. 新物理模式则存储
        if self._is_new_pattern(shielded, context):
            self.hippocampus.store_physical_pattern(
                shielded.detach(), labels, confidence=0.8
            )

        return shielded

    def _is_new_pattern(self, output: torch.Tensor, context: Dict) -> bool:
        return self.training and torch.rand(1).item() < 0.01

    def cognitive_maintenance(self) -> None:
        self.hippocampus.cognitive_maintenance_step()
