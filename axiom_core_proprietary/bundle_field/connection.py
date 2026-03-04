"""
元轴丛场 - 联络
公式间的修正链、组合规则，支持连锁反应
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
from enum import Enum

from .axes import BasePoint


class ConnectionType(Enum):
    PERTURBATION = "perturbation"  # A 在 B 的扰动下修正
    COMPOSITE = "composite"        # A 与 B 组合
    CASCADE = "cascade"            # A 触发 B


@dataclass
class ConnectionEdge:
    """联络边：公式 A → 公式 B 的关系"""
    source_id: str
    target_id: str
    conn_type: ConnectionType
    weight: float = 1.0
    condition: Optional[Callable] = None


class Connection:
    """
    联络：公式间的修正链、组合规则
    存储为有向图：formula_id -> [ConnectionEdge]
    """
    def __init__(self):
        self._edges: Dict[str, List[ConnectionEdge]] = {}
        self._reverse: Dict[str, List[str]] = {}  # target -> [source]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        conn_type: ConnectionType = ConnectionType.PERTURBATION,
        weight: float = 1.0,
    ) -> None:
        edge = ConnectionEdge(
            source_id=source_id,
            target_id=target_id,
            conn_type=conn_type,
            weight=weight,
        )
        if source_id not in self._edges:
            self._edges[source_id] = []
        self._edges[source_id].append(edge)
        if target_id not in self._reverse:
            self._reverse[target_id] = []
        self._reverse[target_id].append(source_id)

    def get_perturbation_chain(self, principal_id: str) -> List[str]:
        """返回 principal 的扰动修正链 [pert_id1, pert_id2, ...]"""
        out = []
        for edge in self._edges.get(principal_id, []):
            if edge.conn_type == ConnectionType.PERTURBATION:
                out.append(edge.target_id)
        return out

    def get_composite_partners(self, formula_id: str) -> List[str]:
        """返回与 formula_id 组合的公式 id 列表"""
        out = []
        for edge in self._edges.get(formula_id, []):
            if edge.conn_type == ConnectionType.COMPOSITE:
                out.append(edge.target_id)
        return out

    def link_perturbation(self, principal_id: str, perturbation_id: str) -> None:
        """建立 principal -> perturbation 的联络边"""
        self.add_edge(principal_id, perturbation_id, ConnectionType.PERTURBATION)
