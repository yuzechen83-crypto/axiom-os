"""
物理标签系统 (Physical Labeling System)
l_i: 物理标签 (重力/动量/离子数密度等)
W_T: 标签映射矩阵, A(S;ξ) = T(l_i)ᵀ W_T T(l_j) 语义锚点
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from enum import Enum

# 物理标签词表
PHYSICAL_LABEL_VOCAB = {
    "gravity": 0,
    "momentum": 1,
    "ion_density": 2,
    "temperature": 3,
    "pressure": 4,
    "magnetic_field": 5,
    "electric_field": 6,
    "velocity": 7,
    "stress": 8,
    "flux": 9,
    "enthalpy": 10,
    "entropy": 11,
    "charge": 12,
    "plasma": 13,
    "electrolyte": 14,
    "gas_ionized": 15,
    "unknown": 16,
}


class PhysicalLabelingSystem:
    """物理标签分类与嵌入"""

    def __init__(self, embed_dim: int = 32):
        self.embed_dim = embed_dim
        self.vocab_size = len(PHYSICAL_LABEL_VOCAB)
        self.label_embedder = nn.Embedding(self.vocab_size, embed_dim)
        self.label_mapping_matrix = nn.Parameter(torch.eye(embed_dim, embed_dim) * 0.1)
        self._label_cache: Dict[str, int] = {}

    def label_to_index(self, label: str) -> int:
        return PHYSICAL_LABEL_VOCAB.get(label.lower(), PHYSICAL_LABEL_VOCAB["unknown"])

    def get_semantic_embedding(self, labels: List[str]) -> torch.Tensor:
        """A(S;ξ) = T(l_i)ᵀ · W_T · T(l_j) 的语义嵌入"""
        indices = [self.label_to_index(l) for l in labels]
        idx_t = torch.tensor(indices, dtype=torch.long)
        base = self.label_embedder(idx_t)
        semantic = base @ self.label_mapping_matrix
        return semantic.mean(dim=0)

    def label(self, data: torch.Tensor, context: Optional[Dict] = None) -> Dict[str, Any]:
        labels = context.get("physical_labels", ["unknown"]) if context else ["unknown"]
        embedded = self.get_semantic_embedding(labels)
        return {"data": data, "labels": labels, "embedding": embedded}


class UPIContractWithSemantics(nn.Module):
    """
    增强版UPI契约，包含物理语义标签
    语义验证：标签兼容性 + 本体一致性
    """

    def __init__(
        self,
        physical_labels: List[str],
        embed_dim: int = 32,
        ontology_uri: Optional[str] = None,
    ):
        super().__init__()
        self.physical_labels = physical_labels
        self.ontology_uri = ontology_uri
        self.label_system = PhysicalLabelingSystem(embed_dim)

    def semantic_verification(self, other_contract: "UPIContractWithSemantics") -> bool:
        """标签兼容性检查"""
        if not other_contract:
            return True
        overlap = set(self.physical_labels) & set(other_contract.physical_labels)
        return len(overlap) > 0 or "unknown" in self.physical_labels

    def get_semantic_embedding(self) -> torch.Tensor:
        return self.label_system.get_semantic_embedding(self.physical_labels)
