"""
结构蒸馏编码器 (Structure Distillation Encoder)
提取对称性、守恒律、主导模态、耦合特征 S_couple
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List


class PhysicalGuidedAttention(nn.Module):
    """物理引导的注意力"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim // 4)
        self.k = nn.Linear(dim, dim // 4)
        self.scale = (dim // 4) ** -0.5

    def forward(
        self,
        latent: torch.Tensor,
        physical_labels: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        B = latent.shape[0]
        q = self.q(latent)
        k = self.k(latent)
        attn = torch.softmax((q @ k.T) * self.scale, dim=-1)
        w = attn.mean(dim=-1)
        return {
            "symmetry": w.unsqueeze(-1).expand(-1, 16),
            "conservation": w.unsqueeze(-1).expand(-1, 16),
            "dominant_mode": w.unsqueeze(-1).expand(-1, 16),
            "coupling": w.unsqueeze(-1).expand(-1, 16),
        }


class StructureDistillationEncoder(nn.Module):
    """
    原始SPNN的结构蒸馏编码器
    输出: symmetry, conservation, dominant_mode, coupling (S_couple)
    """

    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, latent_dim),
            nn.Tanh(),
        )
        self.structure_heads = nn.ModuleDict({
            "symmetry": nn.Linear(latent_dim, 16),
            "conservation": nn.Linear(latent_dim, 16),
            "dominant_mode": nn.Linear(latent_dim, 16),
            "coupling": nn.Linear(latent_dim, 16),
        })
        self.physical_attention = PhysicalGuidedAttention(latent_dim)

    def forward(
        self,
        physical_state: torch.Tensor,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        latent = self.encoder(physical_state)
        structure_features = {k: h(latent) for k, h in self.structure_heads.items()}
        if context and "physical_labels" in context:
            attn_weights = self.physical_attention(latent, context["physical_labels"])
            weighted = [
                structure_features[k] * attn_weights.get(k, torch.ones_like(structure_features[k]))
                for k in structure_features
            ]
            aggregated = torch.cat(weighted, dim=-1)
        else:
            aggregated = torch.cat(list(structure_features.values()), dim=-1)
        return {
            "vector": aggregated,
            "coupling_feature": structure_features["coupling"],
            "symmetry_feature": structure_features["symmetry"],
            "conservation_feature": structure_features["conservation"],
        }
